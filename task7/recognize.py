"""End-to-end check: classify the characters segmented from the chip photo.

Task 6 produced ten character crops in output/segmentation/characters.  This
module pushes those crops through both classifiers trained in Task 7 and prints
the decoded string, which links the classical image-processing part of the
project to the learning part.

Two input variants are tested on purpose:
  * ``normalized`` - the cleaned, binarised, contrast-normalised crops, i.e. the
    best case the classical pipeline can offer;
  * ``gray``       - the raw grayscale crops taken from the photograph, with
    automatic polarity correction.  They contain sensor noise, uneven
    illumination and blur that the clean dataset images do not have, so the
    accuracy drop quantifies the domain gap.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data import CLASSES, OUTPUT_ROOT, PROJECT_ROOT, as_batch, preprocess
from .metrics import load_font

SEGMENTATION_DIR = PROJECT_ROOT / 'output' / 'segmentation'
CNN_DIR = OUTPUT_ROOT / 'cnn'
BASELINE_DIR = OUTPUT_ROOT / 'baseline'
RECOGNIZE_DIR = OUTPUT_ROOT / 'recognition'
GROUND_TRUTH = 'HD44780A00'


def otsu_threshold(values: np.ndarray) -> float:
    """Classic Otsu threshold of a 1-D array of intensities."""
    histogram, edges = np.histogram(values, bins=256)
    centres = (edges[:-1] + edges[1:]) / 2.0
    total = histogram.sum()
    weight_back = np.cumsum(histogram)
    weight_fore = total - weight_back
    sum_back = np.cumsum(histogram * centres)
    sum_total = sum_back[-1]
    valid = (weight_back > 0) & (weight_fore > 0)
    mean_back = np.divide(sum_back, weight_back, out=np.zeros_like(sum_back), where=weight_back > 0)
    mean_fore = np.divide(sum_total - sum_back, weight_fore,
                          out=np.zeros_like(sum_back), where=weight_fore > 0)
    variance = weight_back * weight_fore * (mean_back - mean_fore) ** 2
    variance[~valid] = -1.0
    return float(centres[int(np.argmax(variance))])


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest four-connected foreground component."""
    seen = np.zeros(mask.shape, dtype=bool)
    best = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    for y, x in zip(*np.where(mask)):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        pixels = []
        while stack:
            yy, xx = stack.pop()
            pixels.append((yy, xx))
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                ny, nx = yy + dy, xx + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(pixels) > int(best.sum()):
            best = np.zeros(mask.shape, dtype=bool)
            for yy, xx in pixels:
                best[yy, xx] = True
    return best


def enhance_photo_crop(image: np.ndarray) -> np.ndarray:
    """Contrast enhancement plus light denoising for a low-contrast photo crop.

    The chip photograph has little contrast between glyph and substrate, so the
    crop is stretched to the full range and smoothed slightly.  This is the
    cheapest fix for the domain gap and needs no training data.
    """
    from PIL import ImageFilter, ImageOps
    stretched = ImageOps.autocontrast(Image.fromarray(image.astype(np.uint8)))
    return np.asarray(stretched.filter(ImageFilter.GaussianBlur(1.5)))


def load_cnn():
    from .cnn import CharCNN
    config = json.loads((CNN_DIR / 'config.json').read_text(encoding='utf-8'))
    model = CharCNN(**config['config'])
    model.load_state_dict(dict(np.load(CNN_DIR / 'model.npz')))
    return model, config


def load_baseline():
    with open(BASELINE_DIR / 'model.pkl', 'rb') as handle:
        bundle = pickle.load(handle)
    metrics = json.loads((BASELINE_DIR / 'metrics.json').read_text(encoding='utf-8'))
    return bundle, metrics


def fix_polarity(image: np.ndarray) -> np.ndarray:
    """Make ink dark on light for the raw photo crops (bright glyphs on dark).

    The background is the dominant gray level of a character crop, so the mode
    of the histogram tells the polarity without any manual tuning.
    """
    values, counts = np.unique(image, return_counts=True)
    background_level = values[int(counts.argmax())]
    return 255 - image if background_level < 128 else image


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def read_character_files(variant: str) -> list:
    files = sorted(SEGMENTATION_DIR.joinpath('characters').glob(f'char_*_{variant}.png'))
    if not files:
        raise FileNotFoundError(f'No char_*_{variant}.png files in {SEGMENTATION_DIR / "characters"}')
    return files


def classify(model, config, bundle, baseline_metrics, image: np.ndarray,
             threshold: float | None = None) -> dict:
    """Return both predictions with a confidence score for one character crop."""
    from .data import INK_THRESHOLD
    threshold = INK_THRESHOLD if threshold is None else threshold
    cnn_mode = config['preprocessing']['mode']
    cnn_size = config['preprocessing']['size']
    x_cnn = as_batch(preprocess(image, size=cnn_size, mode=cnn_mode, threshold=threshold)[None])
    probs = model.predict_proba(x_cnn)[0]
    cnn_index = int(probs.argmax())

    base_mode = baseline_metrics['best']['preprocessing']
    base_size = baseline_metrics['best']['size']
    flat = preprocess(image, size=base_size, mode=base_mode, threshold=threshold).reshape(1, -1)
    features = bundle['pca'].transform(bundle['scaler'].transform(flat))
    svm_index = int(bundle['svm'].predict(features)[0])
    decision = bundle['svm'].decision_function(features)[0]
    svm_score = float(softmax(np.atleast_1d(decision))[svm_index])

    return {'cnn': {'class': CLASSES[cnn_index], 'confidence': round(float(probs[cnn_index]), 4)},
            'pca_svm': {'class': CLASSES[svm_index], 'confidence': round(svm_score, 4)},
            'cnn_probabilities': {CLASSES[i]: round(float(p), 4) for i, p in enumerate(probs)}}


def make_figure(entries: list, path: Path, variant: str) -> Path:
    """One row per character: crop, CNN decision, SVM decision."""
    cell, row_h, left = 96, 96, 210
    width = left + cell * 6
    height = 50 + row_h * len(entries)
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    title_font, font = load_font(17), load_font(14)
    draw.text((12, 14), f'Segmented chip characters classified ({variant} input)', fill='black', font=title_font)
    header = ('position', 'crop', 'CNN', 'conf.', 'PCA + SVM', 'conf.')
    for i, text in enumerate(header):
        draw.text((12 + i * (cell + 40) if i == 0 else left + (i - 1) * cell + 10, 40), text,
                  fill='black', font=font)
    for r, entry in enumerate(entries):
        y = 50 + r * row_h
        draw.text((12, y + 20), f"#{entry['index']:02d}", fill='black', font=font)
        tile = Image.fromarray(np.uint8(np.clip(entry['preprocessed'], 0, 1) * 255)).convert('RGB')
        tile = tile.resize((cell - 20, cell - 20), Image.Resampling.NEAREST)
        image.paste(tile, (left, y + 4))
        cnn_ok = entry['cnn']['class'] == entry['expected']
        svm_ok = entry['pca_svm']['class'] == entry['expected']
        draw.text((left + cell + 10, y + 20), entry['cnn']['class'],
                  fill=(20, 120, 20) if cnn_ok else (190, 30, 30), font=font)
        draw.text((left + 2 * cell + 10, y + 20), f"{entry['cnn']['confidence']:.3f}", fill='black', font=font)
        draw.text((left + 3 * cell + 10, y + 20), entry['pca_svm']['class'],
                  fill=(20, 120, 20) if svm_ok else (190, 30, 30), font=font)
        draw.text((left + 4 * cell + 10, y + 20), f"{entry['pca_svm']['confidence']:.3f}", fill='black', font=font)
    draw.text((12, height - 24), f'expected string: {GROUND_TRUTH}', fill='black', font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def run_variant(variant: str, source: str, model, config, bundle, baseline_metrics,
                process=None, otsu_bbox: bool = False, save_figure: bool = True) -> dict:
    entries, cnn_string, svm_string = [], '', ''
    for position, path in enumerate(read_character_files(source), start=1):
        raw = np.asarray(Image.open(path).convert('L'))
        corrected = fix_polarity(raw) if source == 'gray' else raw
        if process is not None:
            corrected = process(corrected)
        threshold = otsu_threshold(corrected) if otsu_bbox else None
        prediction = classify(model, config, bundle, baseline_metrics, corrected, threshold=threshold)
        expected = GROUND_TRUTH[position - 1] if position <= len(GROUND_TRUTH) else '?'
        preprocessed = preprocess(corrected, size=32, mode='crop_pad', threshold=threshold)
        entries.append({'index': position, 'file': path.name, 'expected': expected,
                        'preprocessed': preprocessed, **prediction})
        cnn_string += prediction['cnn']['class']
        svm_string += prediction['pca_svm']['class']
    figure_entries = [{k: v for k, v in e.items() if k != 'cnn_probabilities'} for e in entries]
    if save_figure:
        make_figure(figure_entries, RECOGNIZE_DIR / f'label_{variant}.png', variant)
    payload = {'variant': variant,
               'cnn_string': cnn_string, 'pca_svm_string': svm_string,
               'ground_truth': GROUND_TRUTH,
               'cnn_correct': int(sum(e['cnn']['class'] == e['expected'] for e in entries)),
               'pca_svm_correct': int(sum(e['pca_svm']['class'] == e['expected'] for e in entries)),
               'n_characters': len(entries),
               'characters': [{k: v for k, v in e.items() if k != 'preprocessed'} for e in entries]}
    return payload


def main(args) -> dict:
    RECOGNIZE_DIR.mkdir(parents=True, exist_ok=True)
    model, config = load_cnn()
    bundle, baseline_metrics = load_baseline()
    results = {}
    # The three input variants isolate the effect of image quality on accuracy:
    # clean segmentation output, the raw photo crops, and the same crops after a
    # contrast-enhancement/denoising step.
    for variant, source, process, otsu_bbox in (('normalized', 'normalized', None, False),
                                                ('gray', 'gray', None, False),
                                                ('gray_enhanced', 'gray', enhance_photo_crop, True)):
        payload = run_variant(variant, source, model, config, bundle, baseline_metrics,
                              process=process, otsu_bbox=otsu_bbox)
        results[variant] = payload
        print(f'[{variant:>10}] CNN: {payload["cnn_string"]} '
              f'({payload["cnn_correct"]}/{payload["n_characters"]} correct) | '
              f'PCA+SVM: {payload["pca_svm_string"]} '
              f'({payload["pca_svm_correct"]}/{payload["n_characters"]} correct) | '
              f'expected: {GROUND_TRUTH}')
    (RECOGNIZE_DIR / 'recognition.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Recognise the segmented chip characters end to end.')
    return parser


if __name__ == '__main__':
    main(build_parser().parse_args())
