"""Task 7 data utilities: loading, stratified split and preprocessing variants.

The image set provided with the assignment (dataset_2026.zip) holds seven
character classes.  Every file is a 128x128 8-bit grayscale PNG with dark ink
on a light background.  This module turns those files into the arrays used by
both the from-scratch CNN (Task 7-1) and the PCA + SVM baseline (Task 7-2).
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / 'dataset_2026_updated' / 'dataset_2026'
OUTPUT_ROOT = PROJECT_ROOT / 'output' / 'task7'

# Folder names are the class labels; the index of a label is its class id.
CLASSES = ['0', '4', '7', '8', 'A', 'D', 'H']
LABEL_TO_INDEX = {label: i for i, label in enumerate(CLASSES)}
INDEX_TO_LABEL = {i: label for label, i in LABEL_TO_INDEX.items()}

SPLIT_SEED = 42
TEST_FRACTION = 0.25          # (b) test portion required by the assignment
VAL_FRACTION_OF_ALL = 0.10    # carved out of the remaining 75% for tuning
INK_THRESHOLD = 128           # background is ~255, ink is ~0; the gap is wide


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_images(root: Path = DATASET_ROOT):
    """Return (images uint8 [N,128,128], labels int64 [N]).

    Files are read class by class in name order, so the array order is stable
    and the experiment is reproducible.
    """
    images, labels = [], []
    for label in CLASSES:
        files = sorted((root / label).glob('*.png'))
        if not files:
            raise FileNotFoundError(f'No PNG files found in {root / label}')
        for path in files:
            with Image.open(path) as im:
                images.append(np.asarray(im.convert('L'), dtype=np.uint8))
            labels.append(LABEL_TO_INDEX[label])
    shapes = {a.shape for a in images}
    if shapes != {(128, 128)}:
        raise ValueError(f'Unexpected image shapes: {sorted(shapes)}')
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def stratified_split(labels: np.ndarray, seed: int = SPLIT_SEED, images=None):
    """Split into train / validation / test, stratified by class.

    Test is TEST_FRACTION of every class (the assignment asks for a 25% test
    set).  The remaining 75% is split again so that the validation set is
    VAL_FRACTION_OF_ALL of the whole class and the rest is training data.  The
    test set is never used for tuning, only for the final comparison.
    """
    if images is not None:
        return grouped_stratified_split(images, labels, seed)
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for class_id in np.unique(labels):
        idx = np.flatnonzero(labels == class_id)
        idx = rng.permutation(idx)
        n_total = len(idx)
        n_test = int(round(n_total * TEST_FRACTION))
        n_val = int(round(n_total * VAL_FRACTION_OF_ALL))
        test_idx.append(idx[:n_test])
        val_idx.append(idx[n_test:n_test + n_val])
        train_idx.append(idx[n_test + n_val:])
    train = np.sort(np.concatenate(train_idx))
    val = np.sort(np.concatenate(val_idx))
    test = np.sort(np.concatenate(test_idx))
    return train, val, test


def grouped_stratified_split(images, labels, seed=SPLIT_SEED):
    """Keep identical decoded images together while meeting exact class counts.

    Subset-sum selects whole duplicate groups for test and validation. It fails
    explicitly if exact counts are impossible rather than splitting a group.
    """
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    digest_labels = {}
    for image, label in zip(images, labels):
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        if digest in digest_labels and digest_labels[digest] != int(label):
            raise ValueError('Identical images have conflicting labels.')
        digest_labels[digest] = int(label)

    def select(groups, target):
        reachable = {0: None}
        for j, group in enumerate(groups):
            for total in list(reachable):
                new = total + len(group)
                if new <= target and new not in reachable:
                    reachable[new] = (total, j)
            if target in reachable:
                break
        if target not in reachable:
            raise ValueError(f'Cannot allocate {target} images without splitting duplicates.')
        selected, total = set(), target
        while total:
            total, j = reachable[total]
            selected.add(j)
        return ([i for j in sorted(selected) for i in groups[j]],
                [g for j, g in enumerate(groups) if j not in selected])

    for class_id in np.unique(labels):
        groups_by_hash = {}
        idx = np.flatnonzero(labels == class_id)
        for i in idx:
            digest = hashlib.sha256(images[i].tobytes()).hexdigest()
            groups_by_hash.setdefault(digest, []).append(int(i))
        groups = list(groups_by_hash.values())
        groups = [groups[j] for j in rng.permutation(len(groups))]
        test, remaining = select(groups, int(round(len(idx) * TEST_FRACTION)))
        val, remaining = select(remaining, int(round(len(idx) * VAL_FRACTION_OF_ALL)))
        parts[0].extend(i for g in remaining for i in g)
        parts[1].extend(val)
        parts[2].extend(test)
    return tuple(np.asarray(sorted(part), dtype=np.int64) for part in parts)


def subset_indices(indices: np.ndarray, labels: np.ndarray, per_class: int | None,
                   seed: int = SPLIT_SEED) -> np.ndarray:
    """Keep at most `per_class` samples of every class (for quick smoke runs)."""
    if per_class is None:
        return indices
    rng = np.random.default_rng(seed)
    kept = []
    for class_id in np.unique(labels[indices]):
        idx = indices[labels[indices] == class_id]
        kept.append(rng.permutation(idx)[:per_class])
    return np.sort(np.concatenate(kept))


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def ink_bbox(image: np.ndarray, threshold: int | None = INK_THRESHOLD):
    """Bounding box (x0, y0, x1, y1) of pixels darker than the threshold."""
    threshold = INK_THRESHOLD if threshold is None else threshold
    ys, xs = np.where(image < threshold)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def preprocess(image: np.ndarray, size: int = 32, mode: str = 'crop_pad',
               threshold: int = INK_THRESHOLD, pad: int = 2) -> np.ndarray:
    """Return a float32 [size, size] image with ink = 1 and background = 0.

    Modes:
      none      whole 128x128 frame resized (keeps absolute position and scale)
      crop      ink bounding box cropped, then resized (aspect ratio changes)
      crop_pad  ink bounding box cropped, padded to a square, then resized
                (aspect ratio preserved, position and scale removed)
    """
    image = np.asarray(image, dtype=np.uint8)
    if mode == 'none':
        canvas = Image.fromarray(image)
    else:
        box = ink_bbox(image, threshold)
        if box is None:
            canvas = Image.fromarray(image)
        else:
            x0, y0, x1, y1 = box
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1 = min(image.shape[1], x1 + pad)
            y1 = min(image.shape[0], y1 + pad)
            crop = image[y0:y1, x0:x1]
            if mode == 'crop':
                canvas = Image.fromarray(crop)
            elif mode == 'crop_pad':
                h, w = crop.shape
                side = max(h, w)
                square = np.full((side, side), 255, dtype=np.uint8)
                oy, ox = (side - h) // 2, (side - w) // 2
                square[oy:oy + h, ox:ox + w] = crop
                canvas = Image.fromarray(square)
            else:
                raise ValueError(f'Unknown preprocessing mode: {mode}')
    resized = canvas.resize((size, size), Image.Resampling.LANCZOS)
    ink = np.asarray(resized, dtype=np.float32) / 255.0
    return 1.0 - ink          # ink white on black, like the segmentation output


def build_matrix(images: np.ndarray, mode: str = 'crop_pad', size: int = 32,
                 cache_dir: Path | None = None) -> np.ndarray:
    """Preprocess every image once, caching the result under output/task7/cache."""
    cache_dir = OUTPUT_ROOT / 'cache' if cache_dir is None else cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'x_{mode}_{size}.npy'
    if cache_file.exists():
        cached = np.load(cache_file)
        if cached.shape[0] == images.shape[0]:
            return cached
    matrix = np.stack([preprocess(im, size=size, mode=mode) for im in images])
    np.save(cache_file, matrix)
    return matrix


def pad_for_augmentation(matrix: np.ndarray, extra: int = 4) -> np.ndarray:
    """Pad preprocessed images with background so random crops can shift them."""
    if extra <= 0:
        return matrix
    return np.pad(matrix, ((0, 0), (extra, extra), (extra, extra)), constant_values=0.0)


def as_batch(matrix: np.ndarray) -> np.ndarray:
    """Add the single-channel axis the convolutional network expects."""
    return matrix[:, None]


def split_summary(labels: np.ndarray, train, val, test) -> dict:
    """Class counts of the three splits, for the report and the JSON record."""
    def counts(idx):
        return {CLASSES[c]: int((labels[idx] == c).sum()) for c in range(len(CLASSES))}
    return {
        'classes': CLASSES,
        'seed': SPLIT_SEED,
        'sizes': {'train': int(len(train)), 'validation': int(len(val)), 'test': int(len(test))},
        'fractions': {
            'train': round(len(train) / len(labels), 4),
            'validation': round(len(val) / len(labels), 4),
            'test': round(len(test) / len(labels), 4),
        },
        'per_class': {'train': counts(train), 'validation': counts(val), 'test': counts(test)},
    }


def write_split(indices, path: Path | None = None, labels: np.ndarray | None = None,
                summary: dict | None = None) -> Path:
    """Persist the split indices so every experiment uses exactly the same data."""
    path = OUTPUT_ROOT / 'split.json' if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: [int(i) for i in idx] for name, idx in zip(('train', 'validation', 'test'), indices)}
    if summary is not None:
        payload['summary'] = summary
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return path


def read_split(path: Path | None = None):
    """Read back the recorded split, or None when it has not been written yet."""
    path = OUTPUT_ROOT / 'split.json' if path is None else path
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    return (np.asarray(payload['train'], dtype=np.int64),
            np.asarray(payload['validation'], dtype=np.int64),
            np.asarray(payload['test'], dtype=np.int64))
