"""Task 7-3: compare the CNN (7-1) with the PCA + SVM baseline (7-2).

Reads the metrics written by the two training runs, produces report-ready tables
(both JSON and Markdown) and the comparison figures.  Efficiency is compared
with three practical numbers: training time, single-image latency and the size
of the stored model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import metrics
from .data import CLASSES, OUTPUT_ROOT

CNN_DIR = OUTPUT_ROOT / 'cnn'
BASELINE_DIR = OUTPUT_ROOT / 'baseline'
COMPARE_DIR = OUTPUT_ROOT / 'comparison'


def load_metrics(cnn_path: Path, baseline_path: Path) -> tuple:
    if not cnn_path.exists() or not baseline_path.exists():
        raise FileNotFoundError('Run task7.train_cnn and task7.baseline first.')
    cnn = json.loads(cnn_path.read_text(encoding='utf-8'))
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    return cnn, baseline


def build_table(cnn: dict, baseline: dict) -> dict:
    """Key numbers of the two methods, side by side."""
    return {
        'accuracy': {'cnn': cnn['report']['accuracy'], 'pca_svm': baseline['report']['accuracy']},
        'macro_f1': {'cnn': cnn['report']['macro']['f1'], 'pca_svm': baseline['report']['macro']['f1']},
        'macro_precision': {'cnn': cnn['report']['macro']['precision'],
                            'pca_svm': baseline['report']['macro']['precision']},
        'macro_recall': {'cnn': cnn['report']['macro']['recall'],
                         'pca_svm': baseline['report']['macro']['recall']},
        'train_seconds': {'cnn': cnn['train_seconds'], 'pca_svm': baseline['train_seconds']},
        'ms_per_image_single': {'cnn': cnn['timing']['ms_per_image_single'],
                                'pca_svm': baseline['timing']['ms_per_image_single']},
        'ms_per_image_batched': {'cnn': cnn['timing']['ms_per_image_batched'],
                                 'pca_svm': baseline['timing']['ms_per_image_batched']},
        'model_bytes': {'cnn': cnn['model_bytes'], 'pca_svm': baseline['model_bytes']},
        'extra_notes': {
            'cnn': f"{cnn['spec']['preprocessing']}/{cnn['spec']['size']} input, "
                   f"channels {cnn['spec']['channels']}, lr {cnn['spec']['lr']}, "
                   f"augmentation {cnn['spec']['augmentation']}, seed {cnn['spec']['seed']}",
            'pca_svm': f"{baseline['best']['preprocessing']}/{baseline['best']['size']} input, "
                       f"{baseline['n_features_after_pca']} PCA components "
                       f"({baseline['explained_variance']:.3f} variance), "
                       f"C={baseline['best']['C']}, gamma={baseline['best']['gamma']}, "
                       f"{baseline['n_support_vectors']} support vectors",
        },
    }


def per_class_table(cnn: dict, baseline: dict) -> list:
    rows = []
    for index, label in enumerate(CLASSES):
        cnn_row = cnn['report']['per_class'][str(index)]
        base_row = baseline['report']['per_class'][str(index)]
        rows.append({'class': label, 'support': cnn_row['support'],
                     'cnn_f1': cnn_row['f1'], 'pca_svm_f1': base_row['f1'],
                     'cnn_recall': cnn_row['recall'], 'pca_svm_recall': base_row['recall']})
    return rows


def markdown_report(table: dict, per_class: list, cnn: dict, baseline: dict) -> str:
    acc_cnn, acc_svm = table['accuracy']['cnn'], table['accuracy']['pca_svm']
    t_cnn, t_svm = table['train_seconds']['cnn'], table['train_seconds']['pca_svm']
    i_cnn = table['ms_per_image_single']['cnn']
    i_svm = table['ms_per_image_single']['pca_svm']
    lines = []
    lines.append('# Task 7-3 comparison (numbers generated from the saved metrics)\n')
    lines.append('## Overall accuracy on the held-out 25% test set\n')
    lines.append('| metric | CNN (7-1) | PCA + SVM (7-2) | difference |')
    lines.append('| --- | --- | --- | --- |')
    for key, label, fmt in (('accuracy', 'accuracy', '{:.4f}'),
                            ('macro_f1', 'macro F1', '{:.4f}'),
                            ('macro_precision', 'macro precision', '{:.4f}'),
                            ('macro_recall', 'macro recall', '{:.4f}')):
        a, b = table[key]['cnn'], table[key]['pca_svm']
        lines.append(f'| {label} | {fmt.format(a)} | {fmt.format(b)} | {a - b:+.4f} |')
    lines.append('')
    lines.append('## Cost\n')
    lines.append('| metric | CNN (7-1) | PCA + SVM (7-2) | ratio |')
    lines.append('| --- | --- | --- | --- |')
    lines.append(f"| training time (s) | {t_cnn:.1f} | {t_svm:.1f} | "
                 f"{t_cnn / max(t_svm, 1e-6):.0f}x |")
    lines.append(f"| inference, 1 image (ms) | {i_cnn:.3f} | {i_svm:.3f} | "
                 f"{i_cnn / max(i_svm, 1e-6):.1f}x |")
    lines.append(f"| inference, batched (ms/image) | {table['ms_per_image_batched']['cnn']:.3f} | "
                 f"{table['ms_per_image_batched']['pca_svm']:.3f} | "
                 f"{table['ms_per_image_batched']['cnn'] / max(table['ms_per_image_batched']['pca_svm'], 1e-6):.1f}x |")
    lines.append(f"| stored model (bytes) | {table['model_bytes']['cnn']} | "
                 f"{table['model_bytes']['pca_svm']} | "
                 f"{table['model_bytes']['cnn'] / max(table['model_bytes']['pca_svm'], 1):.1f}x |")
    lines.append('')
    lines.append('## Per-class F1\n')
    lines.append('| class | test samples | CNN F1 | PCA + SVM F1 |')
    lines.append('| --- | --- | --- | --- |')
    for row in per_class:
        lines.append(f"| {row['class']} | {row['support']} | {row['cnn_f1']:.4f} | {row['pca_svm_f1']:.4f} |")
    lines.append('')
    lines.append('## Configurations compared\n')
    lines.append(f"- CNN: {table['extra_notes']['cnn']}")
    lines.append(f"- PCA + SVM: {table['extra_notes']['pca_svm']}")
    lines.append('')
    lines.append('## Observations\n')
    verdict = ('the CNN is more accurate' if acc_cnn > acc_svm else
               'the PCA + SVM baseline is more accurate' if acc_svm > acc_cnn else
               'both methods reach the same accuracy')
    lines.append(f'- On the same stratified split, {verdict} '
                 f'({acc_cnn:.4f} vs {acc_svm:.4f}, gap {abs(acc_cnn - acc_svm):.4f}).')
    lines.append(f'- Training cost differs by roughly {t_cnn / max(t_svm, 1e-6):.0f}x '
                 f'({t_cnn:.1f}s versus {t_svm:.1f}s), while single-image inference is '
                 f'{"faster" if i_cnn < i_svm else "slower"} for the CNN '
                 f'({i_cnn:.3f} ms versus {i_svm:.3f} ms).')
    cnn_top = cnn.get('top_confusions') or []
    svm_top = baseline.get('top_confusions') or []
    if cnn_top:
        lines.append('- CNN residual errors: ' + ', '.join(
            f"{c['true']} misread as {c['predicted']} ({c['count']}x)" for c in cnn_top[:3]) + '.')
    if svm_top:
        lines.append('- PCA + SVM residual errors: ' + ', '.join(
            f"{c['true']} misread as {c['predicted']} ({c['count']}x)" for c in svm_top[:3]) + '.')
    lines.append('- Both models use the same training/validation/test indices. Small accuracy differences on this one split do not establish general superiority; no data-efficiency experiment was performed.')
    lines.append('- Training timing excludes image loading/resizing and hyperparameter search for both methods. The SVM includes scaler/PCA fitting and feature transforms; the CNN includes per-epoch validation and checkpoint selection.')
    lines.append('- Inference uses the same warmed-up protocol: median of three repeats, batches of 256 and 200 randomly selected single images. SVM inference includes scaler/PCA transforms; both exclude image preprocessing and file I/O.')
    lines.append('- Model bytes are compared directly. CNN trainable parameters and SVM/PCA stored state are different quantities and are not presented as an equivalent parameter-count ratio.')
    lines.append('')
    return '\n'.join(lines)


def main(args) -> dict:
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    cnn, baseline = load_metrics(CNN_DIR / 'metrics.json', BASELINE_DIR / 'metrics.json')
    table = build_table(cnn, baseline)
    per_class = per_class_table(cnn, baseline)

    metrics.draw_bars(['CNN (7-1)', 'PCA + SVM (7-2)'],
                      [{'name': 'training time (s)', 'colour': (40, 90, 200),
                        'values': [table['train_seconds']['cnn'], table['train_seconds']['pca_svm']]}],
                      COMPARE_DIR / 'training_time.png',
                      title='Training cost on the same training split',
                      ylabel='seconds', value_format='{:.1f}')
    metrics.draw_bars(['CNN (7-1)', 'PCA + SVM (7-2)'],
                      [{'name': 'inference, 1 image', 'colour': (200, 90, 40),
                        'values': [table['ms_per_image_single']['cnn'],
                                   table['ms_per_image_single']['pca_svm']]},
                       {'name': 'inference, batched', 'colour': (230, 170, 90),
                        'values': [table['ms_per_image_batched']['cnn'],
                                   table['ms_per_image_batched']['pca_svm']]}],
                      COMPARE_DIR / 'latency.png',
                      title='Inference latency (milliseconds per image)',
                      ylabel='ms / image', value_format='{:.3f}')
    metrics.draw_bars([row['class'] for row in per_class],
                      [{'name': 'CNN F1', 'colour': (40, 90, 200),
                        'values': [row['cnn_f1'] for row in per_class]},
                       {'name': 'PCA + SVM F1', 'colour': (200, 90, 40),
                        'values': [row['pca_svm_f1'] for row in per_class]}],
                      COMPARE_DIR / 'per_class_f1.png', title='Per-class F1 score on the test set',
                      ylabel='F1', value_format='{:.3f}')
    metrics.draw_bars(['CNN', 'PCA + SVM'],
                      [{'name': 'accuracy', 'colour': (60, 140, 80),
                        'values': [table['accuracy']['cnn'], table['accuracy']['pca_svm']]},
                       {'name': 'macro F1', 'colour': (120, 180, 120),
                        'values': [table['macro_f1']['cnn'], table['macro_f1']['pca_svm']]}],
                      COMPARE_DIR / 'accuracy_bars.png',
                      title='Test accuracy and macro F1 (25% held-out split)',
                      ylabel='score', value_format='{:.4f}')

    payload = {'table': table, 'per_class': per_class,
               'test_set_size': cnn['report']['n_samples'],
               'classes': CLASSES}
    (COMPARE_DIR / 'comparison.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    report = markdown_report(table, per_class, cnn, baseline)
    (COMPARE_DIR / 'comparison.md').write_text(report, encoding='utf-8')
    print(report)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Compare CNN and PCA + SVM (Task 7-3).')
    return parser


if __name__ == '__main__':
    main(build_parser().parse_args())
