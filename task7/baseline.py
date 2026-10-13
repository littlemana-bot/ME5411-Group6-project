"""Task 7-2: a non-CNN classifier - PCA feature reduction followed by an RBF SVM.

The pipeline is deliberately classical: flatten the preprocessed character,
standardise it, project it onto the leading principal components and classify
the projection with a support vector machine.  Task 7-3 compares this against
the hand-written CNN of Task 7-1.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import metrics
from .data import (CLASSES, OUTPUT_ROOT, SPLIT_SEED, build_matrix, load_images,
                   read_split, split_summary, stratified_split, subset_indices, write_split)

BASELINE_DIR = OUTPUT_ROOT / 'baseline'

PREPROCESSING_CANDIDATES = [('crop_pad', 32), ('crop', 32), ('none', 32), ('crop_pad', 16)]
PCA_CANDIDATES = [30, 50, 100, 200]
C_CANDIDATES = [1.0, 10.0, 100.0]
GAMMA_CANDIDATES = ['scale', 0.01, 0.001]


def prepare(per_class: int | None = None, seed: int = SPLIT_SEED):
    """Reuse the CNN split when it exists so both methods see the same data."""
    images, labels = load_images()
    saved = read_split() if per_class is None else None
    if saved is None:
        train, val, test = stratified_split(labels, seed=seed)
        if per_class is None:
            write_split((train, val, test), summary=split_summary(labels, train, val, test))
    else:
        train, val, test = saved
    train = subset_indices(train, labels, per_class, seed)
    val = subset_indices(val, labels, per_class, seed)
    test = subset_indices(test, labels, per_class, seed)
    return images, labels, train, val, test


def build_pipeline(x_train: np.ndarray, n_components: int, C: float, gamma):
    scaler = StandardScaler().fit(x_train)
    pca = PCA(n_components=n_components, random_state=SPLIT_SEED).fit(scaler.transform(x_train))
    svm = SVC(kernel='rbf', C=C, gamma=gamma)
    return scaler, pca, svm


def transform(scaler, pca, x: np.ndarray) -> np.ndarray:
    return pca.transform(scaler.transform(x))


def evaluate(scaler, pca, svm, x, y) -> dict:
    features = transform(scaler, pca, x)
    pred = svm.predict(features)
    cm = metrics.confusion_matrix(y, pred, len(CLASSES))
    return {'confusion_matrix': cm, 'report': metrics.classification_report(cm), 'pred': pred}


def run_study(images, labels, train, val) -> dict:
    """Two-stage study on the validation split: preprocessing/PCA, then SVM grid."""
    x_train_full = {key: build_matrix(images, mode=key[0], size=key[1]) for key in PREPROCESSING_CANDIDATES}
    flat = {key: x.reshape(x.shape[0], -1) for key, x in x_train_full.items()}

    preprocessing = []
    for mode, size in PREPROCESSING_CANDIDATES:
        key = (mode, size)
        x_tr, x_va = flat[key][train], flat[key][val]
        scaler, pca, svm = build_pipeline(x_tr, n_components=100, C=10.0, gamma='scale')
        start = time.perf_counter()
        svm.fit(transform(scaler, pca, x_tr), labels[train])
        seconds = time.perf_counter() - start
        result = evaluate(scaler, pca, svm, x_va, labels[val])
        entry = {'preprocessing': mode, 'size': size, 'pca_components': 100, 'C': 10.0,
                 'gamma': 'scale', 'val_accuracy': result['report']['accuracy'],
                 'train_seconds': round(seconds, 2),
                 'explained_variance': round(float(pca.explained_variance_ratio_.sum()), 4)}
        preprocessing.append(entry)
        print(f'  [svm-pre] {mode}/{size:<3d} val_acc={entry["val_accuracy"]:.4f} '
              f'pca_var={entry["explained_variance"]:.3f} {seconds:5.1f}s', flush=True)
    preprocessing.sort(key=lambda e: -e['val_accuracy'])
    best_pre = (preprocessing[0]['preprocessing'], preprocessing[0]['size'])

    pca_study = []
    for n_components in PCA_CANDIDATES:
        x_tr, x_va = flat[best_pre][train], flat[best_pre][val]
        scaler, pca, svm = build_pipeline(x_tr, n_components=n_components, C=10.0, gamma='scale')
        start = time.perf_counter()
        svm.fit(transform(scaler, pca, x_tr), labels[train])
        seconds = time.perf_counter() - start
        result = evaluate(scaler, pca, svm, x_va, labels[val])
        entry = {'pca_components': n_components, 'C': 10.0, 'gamma': 'scale',
                 'val_accuracy': result['report']['accuracy'], 'train_seconds': round(seconds, 2),
                 'explained_variance': round(float(pca.explained_variance_ratio_.sum()), 4)}
        pca_study.append(entry)
        print(f'  [svm-pca] n={n_components:<4d} val_acc={entry["val_accuracy"]:.4f} '
              f'pca_var={entry["explained_variance"]:.3f} {seconds:5.1f}s', flush=True)
    pca_study.sort(key=lambda e: -e['val_accuracy'])
    best_n = pca_study[0]['pca_components']

    svm_study = []
    for C in C_CANDIDATES:
        for gamma in GAMMA_CANDIDATES:
            x_tr, x_va = flat[best_pre][train], flat[best_pre][val]
            scaler, pca, svm = build_pipeline(x_tr, n_components=best_n, C=C, gamma=gamma)
            start = time.perf_counter()
            svm.fit(transform(scaler, pca, x_tr), labels[train])
            seconds = time.perf_counter() - start
            result = evaluate(scaler, pca, svm, x_va, labels[val])
            entry = {'C': C, 'gamma': gamma, 'val_accuracy': result['report']['accuracy'],
                     'train_seconds': round(seconds, 2),
                     'n_support_vectors': int(svm.support_vectors_.shape[0])}
            svm_study.append(entry)
            print(f'  [svm-grid] C={C:<6g} gamma={str(gamma):<6} val_acc={entry["val_accuracy"]:.4f} '
                  f'sv={entry["n_support_vectors"]:<5d} {seconds:5.1f}s', flush=True)
    svm_study.sort(key=lambda e: -e['val_accuracy'])
    return {'preprocessing': preprocessing, 'pca': pca_study, 'svm_grid': svm_study,
            'best': {'preprocessing': best_pre[0], 'size': best_pre[1], 'pca_components': best_n,
                     'C': svm_study[0]['C'], 'gamma': svm_study[0]['gamma']}}


def main(args) -> dict:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    print('Task 7-2: PCA + RBF SVM baseline')
    images, labels, train, val, test = prepare(per_class=args.per_class)
    print(f'  split: train={len(train)} val={len(val)} test={len(test)}')
    study = run_study(images, labels, train, val)
    best = study['best']
    print(f'  best configuration: {best}')

    x = build_matrix(images, mode=best['preprocessing'], size=best['size'])
    flat = x.reshape(x.shape[0], -1)
    scaler, pca, svm = build_pipeline(flat[train], n_components=best['pca_components'],
                                      C=best['C'], gamma=best['gamma'])
    start = time.perf_counter()
    svm.fit(transform(scaler, pca, flat[train]), labels[train])
    train_seconds = time.perf_counter() - start
    result = evaluate(scaler, pca, svm, flat[test], labels[test])
    print(f'  test accuracy {result["report"]["accuracy"]:.4f} '
          f'macro-F1 {result["report"]["macro"]["f1"]:.4f} ({train_seconds:.1f}s training)')

    # Inference timing: batched call plus single-image latency.
    features_test = transform(scaler, pca, flat[test])
    start = time.perf_counter()
    svm.predict(features_test)
    batched = time.perf_counter() - start
    sample = 200
    start = time.perf_counter()
    for i in range(sample):
        svm.predict(features_test[i:i + 1])
    single = (time.perf_counter() - start) / sample

    with open(BASELINE_DIR / 'model.pkl', 'wb') as handle:
        pickle.dump({'scaler': scaler, 'pca': pca, 'svm': svm, 'classes': CLASSES}, handle)
    np.save(BASELINE_DIR / 'test_predictions.npy', result['pred'])
    metrics.draw_confusion_matrix(result['confusion_matrix'], CLASSES,
                                  BASELINE_DIR / 'confusion_matrix.png',
                                  title='PCA + SVM test confusion matrix, accuracy '
                                        + metrics.as_percent(result['report']['accuracy']))
    metrics.draw_series_lines(
        {'cumulative explained variance': np.cumsum(pca.explained_variance_ratio_)},
        BASELINE_DIR / 'pca_variance.png',
        title=f'PCA cumulative explained variance ({best["preprocessing"]}/{best["size"]})',
        xlabel='principal components', ylabel='cumulative variance ratio')
    wrong = np.flatnonzero(result['pred'] != labels[test])[:16]
    metrics.draw_image_grid([x[test[i]] for i in wrong],
                            [f'{CLASSES[labels[test][i]]}->{CLASSES[result["pred"][i]]}' for i in wrong],
                            BASELINE_DIR / 'errors.png', cols=8, cell=72,
                            title='PCA + SVM misclassified test samples (true -> predicted)')

    payload = {'study': study, 'best': best, 'epochs_note': 'SVM has no epochs (closed-form fit)',
               'train_seconds': round(train_seconds, 2),
               'n_support_vectors': int(svm.support_vectors_.shape[0]),
               'n_features_after_pca': int(pca.n_components_),
               'explained_variance': round(float(pca.explained_variance_ratio_.sum()), 4),
               'confusion_matrix': result['confusion_matrix'].tolist(),
               'report': result['report'],
               'timing': {'batched_seconds': round(batched, 3),
                          'ms_per_image_batched': round(1000 * batched / len(test), 3),
                          'ms_per_image_single': round(1000 * single, 3)},
               'model_bytes': int((BASELINE_DIR / 'model.pkl').stat().st_size)}
    payload['top_confusions'] = metrics.top_confusions(result['confusion_matrix'], CLASSES)
    (BASELINE_DIR / 'metrics.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train the PCA + SVM baseline (Task 7-2).')
    parser.add_argument('--per-class', type=int, default=None,
                        help='limit samples per class (quick smoke test)')
    return parser


if __name__ == '__main__':
    main(build_parser().parse_args())
