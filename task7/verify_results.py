"""Reload saved models and independently recompute predictions from uncached test images."""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import metrics
from .audit import main as audit_data
from .data import OUTPUT_ROOT, as_batch, load_images, preprocess, read_split
from .recognize import load_baseline, load_cnn


def main():
    audit = audit_data()
    assert audit['checks']['saved_split_matches_grouped_seed'] is True
    images, labels = load_images()
    _, _, test = read_split()
    model, config = load_cnn()
    bundle, baseline_metrics = load_baseline()
    cnn_metrics = json.loads((OUTPUT_ROOT / 'cnn/metrics.json').read_text())
    assert np.array_equal(test, np.load(OUTPUT_ROOT / 'cnn/test_indices.npy'))
    cnn_pre = config['preprocessing']
    x = np.stack([preprocess(images[i], mode=cnn_pre['mode'], size=cnn_pre['size']) for i in test])
    cnn_pred = model.predict(as_batch(x), batch_size=256)
    best = baseline_metrics['best']
    x = np.stack([preprocess(images[i], mode=best['preprocessing'], size=best['size']) for i in test])
    svm_pred = bundle['svm'].predict(bundle['pca'].transform(bundle['scaler'].transform(x.reshape(len(x), -1))))
    checks = {}
    for name, pred, saved in (('cnn', cnn_pred, cnn_metrics), ('baseline', svm_pred, baseline_metrics)):
        assert np.array_equal(pred, np.load(OUTPUT_ROOT / name / 'test_predictions.npy'))
        cm = metrics.confusion_matrix(labels[test], pred, 7)
        assert np.array_equal(cm, saved['confusion_matrix'])
        # JSON object keys are strings; the in-memory report uses integer class ids.
        recomputed = json.loads(json.dumps(metrics.classification_report(cm)))
        assert recomputed == saved['report']
        checks[name] = {'prediction_arrays_identical': True, 'confusion_matrix_identical': True,
                        'report_identical': True, 'errors': int((pred != labels[test]).sum())}
    cnn_ok, svm_ok = cnn_pred == labels[test], svm_pred == labels[test]
    paired = {'both_correct': int((cnn_ok & svm_ok).sum()),
              'cnn_only_correct': int((cnn_ok & ~svm_ok).sum()),
              'svm_only_correct': int((~cnn_ok & svm_ok).sum()),
              'both_wrong': int((~cnn_ok & ~svm_ok).sum())}
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / 'task7').glob('*.py')) + [root/'run_task7.py', root/'segment_characters.py']
    payload = {'checks': checks, 'test_samples': len(test), 'paired_outcomes': paired,
               'scope': 'Reloaded models; each test image was preprocessed without cached arrays. Repeated test evaluation verifies reproducibility, not hyperparameter selection.',
               'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUTPUT_ROOT / 'verification/reloaded_models.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('Reloaded models verified:', checks, flush=True)
    return payload


if __name__ == '__main__':
    main()
