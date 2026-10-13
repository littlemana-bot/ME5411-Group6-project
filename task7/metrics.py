"""Evaluation metrics and report figures (Pillow only, no plotting package)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r'C:\Windows\Fonts\arial.ttf',
    r'C:\Windows\Fonts\segoeui.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]


def load_font(size: int = 14):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def classification_report(cm: np.ndarray) -> dict:
    """Per-class precision/recall/F1 plus accuracy and macro averages."""
    total = cm.sum()
    accuracy = float(np.trace(cm) / total) if total else 0.0
    per_class = {}
    for c in range(cm.shape[0]):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {'precision': round(precision, 4), 'recall': round(recall, 4),
                        'f1': round(f1, 4), 'support': int(cm[c, :].sum())}
    macro = {
        'precision': round(float(np.mean([v['precision'] for v in per_class.values()])), 4),
        'recall': round(float(np.mean([v['recall'] for v in per_class.values()])), 4),
        'f1': round(float(np.mean([v['f1'] for v in per_class.values()])), 4),
    }
    return {'accuracy': round(accuracy, 4), 'macro': macro, 'per_class': per_class,
            'n_samples': int(total)}


def top_confusions(cm: np.ndarray, classes, limit: int = 5) -> list:
    """Off-diagonal cells sorted by count, for the error discussion."""
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append({'true': str(classes[i]), 'predicted': str(classes[j]),
                              'count': int(cm[i, j])})
    pairs.sort(key=lambda x: -x['count'])
    return pairs[:limit]


def as_percent(value: float) -> str:
    return f'{100 * value:.2f}%'


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _canvas(width: int, height: int, background='white'):
    image = Image.new('RGB', (width, height), background)
    return image, ImageDraw.Draw(image)


def draw_confusion_matrix(cm: np.ndarray, classes, path: Path, title: str = '') -> Path:
    cell, left, top, bottom = 64, 90, 78, 60
    width = left + cell * len(classes) + 140
    height = top + cell * len(classes) + bottom
    image, draw = _canvas(width, height)
    font = load_font(15)
    small = load_font(13)
    normalised = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    if title:
        draw.text((left, 10), title, fill='black', font=load_font(18))
    for i in range(len(classes)):          # rows = true label
        for j in range(len(classes)):      # columns = predicted label
            x0, y0 = left + j * cell, top + i * cell
            shade = int(255 - 200 * normalised[i, j])
            draw.rectangle((x0, y0, x0 + cell - 1, y0 + cell - 1), fill=(shade, shade, shade),
                           outline=(120, 120, 120))
            text = str(cm[i, j])
            box = draw.textbbox((0, 0), text, font=font)
            colour = 'white' if shade < 120 else 'black'
            draw.text((x0 + (cell - box[2] + box[0]) / 2, y0 + (cell - box[3] + box[1]) / 2),
                      text, fill=colour, font=font)
    for j, label in enumerate(classes):
        box = draw.textbbox((0, 0), str(label), font=font)
        draw.text((left + j * cell + (cell - box[2] + box[0]) / 2, top - 30), str(label), fill='black', font=font)
    for i, label in enumerate(classes):
        box = draw.textbbox((0, 0), str(label), font=font)
        draw.text((left - 30 - (box[2] - box[0]), top + i * cell + (cell - box[3] + box[1]) / 2),
                  str(label), fill='black', font=font)
    draw.text((max(6, left - 84), top - 24), 'true \\ predicted', fill='black', font=small)
    legend_x = left + cell * len(classes) + 20
    draw.text((legend_x, top), 'row = true', fill='black', font=small)
    draw.text((legend_x, top + 20), 'col = predicted', fill='black', font=small)
    draw.text((legend_x, top + 50), 'shading = recall', fill='black', font=small)
    draw.text((legend_x, top + 80), 'counts shown', fill='black', font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def draw_curves(history: dict, path: Path, title: str = 'Training history') -> Path:
    """Two stacked panels: accuracy (top) and loss (bottom), each with its own scale."""
    width, height = 700, 560
    left, right, top = 70, 180, 50
    panel_h, gap = 190, 70
    image, draw = _canvas(width, height)
    font, small = load_font(14), load_font(13)
    draw.text((left - 40, 14), title, fill='black', font=load_font(17))
    epochs = len(history['loss'])
    plot_w = width - left - right
    if not epochs:
        image.save(path)
        return path

    def panel(y0: int, series: list, vmax: float, label: str) -> None:
        draw.rectangle((left, y0, left + plot_w, y0 + panel_h), outline=(120, 120, 120))
        draw.text((left - 56, y0 - 8), f'{vmax:.2f}', fill='black', font=small)
        draw.text((left + 8, y0 + 6), label, fill=(90, 90, 90), font=small)
        for s, (name, values, colour) in enumerate(series):
            points = [(left + i * plot_w / max(1, epochs - 1),
                       y0 + panel_h * (1 - max(0.0, min(1.0, v / vmax)))) for i, v in enumerate(values)]
            draw.line(points, fill=colour, width=2)
            legend_y = y0 + 6 + 22 * s
            draw.line((left + plot_w + 16, legend_y + 7, left + plot_w + 44, legend_y + 7), fill=colour, width=2)
            draw.text((left + plot_w + 50, legend_y), name, fill=colour, font=small)
        draw.text((left - 16, y0 + panel_h - 9), '0', fill='black', font=small)

    accuracy = history.get('train_accuracy') or []
    panel(top, [('train accuracy', accuracy, (200, 40, 40)),
                ('validation accuracy', history['val_accuracy'], (40, 90, 200))], 1.0, 'accuracy')
    loss_max = max(history['loss'] + history['val_loss']) * 1.1
    panel(top + panel_h + gap, [('train loss', history['loss'], (60, 160, 60)),
                                ('validation loss', history['val_loss'], (230, 150, 30))],
          loss_max, 'cross-entropy loss')
    draw.text((left, height - 40), f'epochs: 1 .. {epochs}', fill='black', font=font)
    image.save(path)
    return path


def draw_bars(groups: list, series: list, path: Path, title: str = '',
              ylabel: str = '', value_format: str = '{:.3f}') -> Path:
    """groups: x labels; series: list of dicts {name, values, colour}."""
    width, height = max(620, 130 * len(groups) + 260), 380
    pad_l, pad_b, pad_t = 80, 80, 60
    image, draw = _canvas(width, height)
    font, small = load_font(14), load_font(13)
    draw.text((pad_l, 18), title, fill='black', font=load_font(17))
    plot_h = height - pad_b - pad_t
    plot_w = width - pad_l - 220
    maximum = max([max(s['values']) for s in series] + [1e-9]) * 1.15
    draw.line((pad_l, pad_t, pad_l, pad_t + plot_h), fill='black')
    draw.line((pad_l, pad_t + plot_h, pad_l + plot_w, pad_t + plot_h), fill='black')
    draw.text((pad_l - 60, pad_t - 20), ylabel, fill='black', font=small)
    slot = plot_w / max(1, len(groups))
    bar_w = slot / (len(series) + 1)
    for g, label in enumerate(groups):
        for s, spec in enumerate(series):
            value = spec['values'][g]
            x0 = pad_l + g * slot + bar_w * (s + 0.5)
            y0 = pad_t + plot_h * (1 - value / maximum)
            draw.rectangle((x0, y0, x0 + bar_w * 0.9, pad_t + plot_h), fill=spec['colour'])
            draw.text((x0, y0 - 18), value_format.format(value), fill='black', font=small)
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((pad_l + g * slot + (slot - box[2] + box[0]) / 2, pad_t + plot_h + 12),
                  label, fill='black', font=font)
    for s, spec in enumerate(series):
        y = pad_t + 8 + 22 * s
        draw.rectangle((pad_l + plot_w + 20, y, pad_l + plot_w + 40, y + 14), fill=spec['colour'])
        draw.text((pad_l + plot_w + 48, y - 2), spec['name'], fill='black', font=small)
    image.save(path)
    return path


def draw_series_lines(series: dict, path: Path, title: str = '', xlabel: str = '',
                      ylabel: str = '') -> Path:
    """Line chart for one or more 1-D series (e.g. cumulative PCA variance)."""
    width, height, pad_l, pad_b, pad_t = 760, 380, 90, 70, 60
    image, draw = _canvas(width, height)
    font, small = load_font(14), load_font(13)
    draw.text((pad_l, 18), title, fill='black', font=load_font(17))
    plot_w, plot_h = width - pad_l - 220, height - pad_b - pad_t
    draw.rectangle((pad_l, pad_t, pad_l + plot_w, pad_t + plot_h), outline=(120, 120, 120))
    colours = [(40, 90, 200), (200, 40, 40), (40, 150, 60), (160, 60, 160)]
    for s, (name, values) in enumerate(series.items()):
        values = np.asarray(values, dtype=float)
        n = len(values)
        points = [(pad_l + i * plot_w / max(1, n - 1),
                   pad_t + plot_h * (1 - values[i])) for i in range(n)]
        draw.line(points, fill=colours[s % len(colours)], width=2)
        draw.text((pad_l + plot_w + 16, pad_t + 10 + 22 * s), name, fill=colours[s % len(colours)], font=small)
    draw.text((pad_l + plot_w / 2 - 40, pad_t + plot_h + 20), xlabel, fill='black', font=font)
    draw.text((12, pad_t + plot_h / 2), ylabel, fill='black', font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def draw_image_grid(images: list, titles: list, path: Path, cols: int = 8, cell: int = 64,
                    title: str = '') -> Path:
    """Contact sheet of prepared samples; images are float arrays with ink = 1."""
    rows = (len(images) + cols - 1) // cols
    header = 34 if title else 16
    width, height = cols * cell, header + rows * (cell + 18)
    image, draw = _canvas(width, height)
    font = load_font(13)
    if title:
        draw.text((6, 8), title, fill='black', font=load_font(16))
    for i, (array, label) in enumerate(zip(images, titles)):
        r, c = divmod(i, cols)
        tile = Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255)).resize((cell, cell), Image.Resampling.NEAREST)
        x, y = c * cell, header + r * (cell + 18)
        image.paste(tile.convert('RGB'), (x, y))
        draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=(150, 150, 150))
        draw.text((x + 4, y + cell + 2), str(label), fill='black', font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
