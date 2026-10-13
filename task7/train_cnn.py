"""Task 7-1 driver: preprocessing / hyper-parameter study and final CNN training."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import metrics
from .cnn import Adam, CharCNN, fit, numerical_gradient_check
from .data import (CLASSES, OUTPUT_ROOT, SPLIT_SEED, as_batch, build_matrix, load_images,
                   pad_for_augmentation, read_split, split_summary, stratified_split,
                   subset_indices, write_split)

CNN_DIR = OUTPUT_ROOT / 'cnn'

# (name, preprocessing mode, image size, channels, learning rate, augmentation)
EXPERIMENTS = [
    ('raw32',        'none',     32, (16, 32), 3e-3, True),
    ('crop32',       'crop',     32, (16, 32), 3e-3, True),
    ('croppad32',    'crop_pad', 32, (16, 32), 3e-3, True),
    ('croppad32_lr', 'crop_pad', 32, (16, 32), 1e-3, True),
    ('croppad32_sm', 'crop_pad', 32, (8, 16),  3e-3, True),
    ('croppad32_na', 'crop_pad', 32, (16, 32), 3e-3, False),
    ('croppad24',    'crop_pad', 24, (16, 32), 3e-3, True),
]


def prepare(per_class: int | None = None, seed: int = SPLIT_SEED):
    """Load the dataset, make the stratified split and record it on disk.

    The split file is only written for a full run; a --per-class smoke test must
    not overwrite the split that the real experiments share.
    """
    images, labels = load_images()
    train, val, test = stratified_split(labels, seed=seed)
    train = subset_indices(train, labels, per_class, seed)
    val = subset_indices(val, labels, per_class, seed)
    test = subset_indices(test, labels, per_class, seed)
    summary = split_summary(labels, train, val, test)
    if per_class is None:
        write_split((train, val, test), summary=summary)
    return images, labels, train, val, test, summary


def measure_inference(model: CharCNN, x: np.ndarray, batch_size: int) -> dict:
    """Batched throughput plus single-image latency."""
    start = time.perf_counter()
    model.predict(x, batch_size=batch_size)
    batched = time.perf_counter() - start
    sample = x[:min(200, x.shape[0])]
    start = time.perf_counter()
    for i in range(sample.shape[0]):
        model.predict(sample[i:i + 1], batch_size=1)
    single = (time.perf_counter() - start) / sample.shape[0]
    return {'batched_seconds': round(batched, 3),
            'ms_per_image_batched': round(1000 * batched / x.shape[0], 3),
            'ms_per_image_single': round(1000 * single, 3)}


def run_experiments(images, labels, train, val, epochs: int, seed: int = 0) -> list:
    """Compare preprocessing variants and hyper-parameters on the validation set."""
    results = []
    matrices: dict = {}
    for name, mode, size, channels, lr, augment in EXPERIMENTS:
        key = (mode, size)
        if key not in matrices:
            matrices[key] = build_matrix(images, mode=mode, size=size)
        x = matrices[key]
        x_train = as_batch(pad_for_augmentation(x[train], 4)) if augment else as_batch(x[train])
        x_val = as_batch(x[val])
        model = CharCNN(in_size=size, channels=channels, hidden=128, n_classes=len(CLASSES),
                        dropout=0.3, seed=seed)
        start = time.perf_counter()
        history = fit(model, x_train, labels[train], x_val, labels[val], epochs=epochs,
                      batch_size=64, lr=lr, weight_decay=1e-4, augment=augment,
                      extra=4 if augment else 0, seed=seed, verbose=False)
        seconds = time.perf_counter() - start
        entry = {'name': name, 'preprocessing': mode, 'size': size, 'channels': list(channels),
                 'lr': lr, 'augmentation': augment, 'epochs': epochs,
                 'n_parameters': model.n_parameters(),
                 'val_accuracy': round(history['best_val_accuracy'], 4),
                 'best_epoch': history['best_epoch'], 'train_seconds': round(seconds, 2)}
        results.append(entry)
        print(f'  [cnn-exp] {name:<14} val_acc={entry["val_accuracy"]:.4f} '
              f'params={entry["n_parameters"]:>7d} {seconds:6.1f}s', flush=True)
    results.sort(key=lambda r: -r['val_accuracy'])
    return results


def train_final(images, labels, train, val, test, spec: dict, epochs: int, seed: int = 0,
                per_class: int | None = None) -> dict:
    """Train the selected configuration and evaluate it on the untouched test set."""
    mode, size = spec['preprocessing'], spec['size']
    channels = tuple(spec['channels'])
    augment = spec['augmentation']
    x = build_matrix(images, mode=mode, size=size)
    x_train = as_batch(pad_for_augmentation(x[train], 4)) if augment else as_batch(x[train])
    x_val, x_test = as_batch(x[val]), as_batch(x[test])
    model = CharCNN(in_size=size, channels=channels, hidden=128, n_classes=len(CLASSES),
                    dropout=0.3, seed=seed)
    start = time.perf_counter()
    history = fit(model, x_train, labels[train], x_val, labels[val], epochs=epochs,
                  batch_size=64, lr=spec['lr'], weight_decay=1e-4, augment=augment,
                  extra=4 if augment else 0, seed=seed, verbose=True, log_every=5)
    train_seconds = time.perf_counter() - start
    probs = model.predict_proba(x_test)
    pred = probs.argmax(axis=1)
    cm = metrics.confusion_matrix(labels[test], pred, len(CLASSES))
    report = metrics.classification_report(cm)
    timing = measure_inference(model, x_test, batch_size=256)
    CNN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CNN_DIR / 'model.npz', **model.state_dict())
    (CNN_DIR / 'config.json').write_text(json.dumps({
        'architecture': 'conv3x3-%d -> pool -> conv3x3-%d -> pool -> dense-128 -> dense-%d'
                        % (channels[0], channels[1], len(CLASSES)),
        'config': model.config, 'preprocessing': {'mode': mode, 'size': size,
                                                  'ink': 'white on black, values in [0,1]'},
        'augmentation': augment, 'lr': spec['lr'], 'weight_decay': 1e-4,
        'batch_size': 64, 'epochs': epochs, 'seed': seed, 'split_seed': SPLIT_SEED,
        'per_class_limit': per_class}, indent=2), encoding='utf-8')
    np.save(CNN_DIR / 'test_predictions.npy', pred)
    np.save(CNN_DIR / 'test_indices.npy', test)
    np.save(CNN_DIR / 'test_probabilities.npy', probs.astype(np.float32))
    result = {'spec': {**spec, 'seed': seed}, 'epochs': epochs, 'history': history,
              'train_seconds': round(train_seconds, 2),
              'n_parameters': model.n_parameters(),
              'confusion_matrix': cm.tolist(),
              'report': report, 'timing': timing,
              'model_bytes': int((CNN_DIR / 'model.npz').stat().st_size)}
    return result


def error_sheet(x: np.ndarray, labels, test, pred, path: Path, limit: int = 16) -> Path:
    wrong = np.flatnonzero(pred != labels[test])
    titles = [f'{CLASSES[labels[test][i]]}->{CLASSES[pred[i]]}' for i in wrong[:limit]]
    images = [x[test[i]] for i in wrong[:limit]]
    return metrics.draw_image_grid(images, titles, path, cols=8, cell=72,
                                   title='Misclassified test samples (true -> predicted)')


def preprocessing_sheet(images, labels, train, path: Path, size: int = 32, limit: int = 8) -> Path:
    """Show what each preprocessing mode does to the same eight samples."""
    picks = train[:limit]
    tiles, titles = [], []
    for mode in ('none', 'crop', 'crop_pad'):
        x = build_matrix(images, mode=mode, size=size)
        for idx in picks:
            tiles.append(x[idx])
            titles.append(f'{mode} {CLASSES[labels[idx]]}')
    return metrics.draw_image_grid(tiles, titles, path, cols=limit, cell=64,
                                   title=f'Preprocessing comparison at {size}x{size} (none / crop / crop_pad)')


def dataset_sheet(images, labels, path: Path, per_class: int = 12, seed: int = 0) -> Path:
    """One row per class: how much the typeface varies inside a class."""
    rng = np.random.default_rng(seed)
    tiles, titles = [], []
    for class_id in range(len(CLASSES)):
        idx = np.flatnonzero(labels == class_id)
        picks = rng.permutation(idx)[:per_class]
        for i in picks:
            tiles.append(1.0 - images[i].astype(np.float32) / 255.0)   # ink becomes white
            titles.append(CLASSES[class_id])
    return metrics.draw_image_grid(tiles, titles, path, cols=per_class, cell=52,
                                   title=f'Dataset samples: one row per class, {per_class} random '
                                         f'examples each (ink shown white)')


def render_figures(payload: dict, images, labels, train, test, pred, title: str) -> None:
    """Draw (or redraw) every Task 7-1 figure from a metrics payload."""
    spec = payload['spec']
    x = build_matrix(images, mode=spec['preprocessing'], size=spec['size'])
    error_sheet(x, labels, test, pred, CNN_DIR / 'errors.png')
    preprocessing_sheet(images, labels, train, CNN_DIR / 'preprocessing.png', size=spec['size'])
    dataset_sheet(images, labels, CNN_DIR / 'dataset_samples.png')
    metrics.draw_curves(payload['history'], CNN_DIR / 'training_curves.png', title=title)
    metrics.draw_confusion_matrix(np.asarray(payload['confusion_matrix']), CLASSES,
                                  CNN_DIR / 'confusion_matrix.png',
                                  title=f'CNN test confusion matrix, accuracy '
                                        f'{metrics.as_percent(payload["report"]["accuracy"])}')


def figures_only(args) -> dict:
    """Redraw the figures of an existing run without retraining."""
    payload = json.loads((CNN_DIR / 'metrics.json').read_text(encoding='utf-8'))
    images, labels = load_images()
    saved = read_split()
    if saved is None:
        raise FileNotFoundError('output/task7/split.json is missing; run the training first.')
    train, _, _ = saved
    test = np.load(CNN_DIR / 'test_indices.npy')
    pred = np.load(CNN_DIR / 'test_predictions.npy')
    render_figures(payload, images, labels, train, test, pred,
                   title=f'CNN training ({payload["spec"]["name"]}, {payload["epochs"]} epochs)')
    print('Task 7-1 figures regenerated from output/task7/cnn/metrics.json')
    return payload


def main(args) -> dict:
    CNN_DIR.mkdir(parents=True, exist_ok=True)
    if getattr(args, 'figures_only', False):
        return figures_only(args)
    print('Task 7-1: CNN (NumPy, from scratch)')
    grad = numerical_gradient_check()
    worst = max(grad.values())
    print(f'  gradient check: worst relative error {worst:.2e} over {len(grad)} tensors')
    if worst > 1e-4:
        raise RuntimeError('Analytic gradients disagree with the numerical check.')
    images, labels, train, val, test, summary = prepare(per_class=args.per_class)
    print(f'  split: train={summary["sizes"]["train"]} val={summary["sizes"]["validation"]} '
          f'test={summary["sizes"]["test"]} (classes {CLASSES})')
    experiments = run_experiments(images, labels, train, val, epochs=args.experiment_epochs)
    experiments_path = CNN_DIR / 'experiments.json'
    experiments_path.write_text(json.dumps({'gradient_check': grad, 'experiments': experiments},
                                           indent=2), encoding='utf-8')
    best = experiments[0]
    print(f'  best configuration: {best["name"]} (val_acc={best["val_accuracy"]:.4f})')
    final = train_final(images, labels, train, val, test, best, epochs=args.epochs,
                        per_class=args.per_class)
    print(f'  test accuracy {final["report"]["accuracy"]:.4f} '
          f'macro-F1 {final["report"]["macro"]["f1"]:.4f} '
          f'({final["train_seconds"]:.1f}s training, {final["timing"]["ms_per_image_single"]:.2f} ms/image)')
    final['top_confusions'] = metrics.top_confusions(np.asarray(final['confusion_matrix']), CLASSES)
    final['experiments'] = experiments
    (CNN_DIR / 'metrics.json').write_text(json.dumps(final, indent=2), encoding='utf-8')
    pred = np.asarray(np.load(CNN_DIR / 'test_predictions.npy'))
    render_figures(final, images, labels, train, test, pred,
                   title=f'CNN training ({best["name"]}, {args.epochs} epochs)')
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train the from-scratch NumPy CNN (Task 7-1).')
    parser.add_argument('--epochs', type=int, default=25, help='epochs for the final model')
    parser.add_argument('--experiment-epochs', type=int, default=8, help='epochs per experiment')
    parser.add_argument('--per-class', type=int, default=None,
                        help='limit samples per class (quick smoke test)')
    parser.add_argument('--figures-only', action='store_true',
                        help='redraw the figures of the last run without training')
    return parser


if __name__ == '__main__':
    main(build_parser().parse_args())
