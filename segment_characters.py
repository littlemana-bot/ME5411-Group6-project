"""Segment the microchip label using Pillow and NumPy; no class labels are used."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


def components(mask):
    """Find four-connected foreground components, including their pixels."""
    seen = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    result = []
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
        pts = np.asarray(pixels)
        y0, x0 = pts.min(axis=0)
        y1, x1 = pts.max(axis=0) + 1
        result.append((tuple(map(int, (x0, y0, x1, y1))), pts))
    return result


def split_boxes(mask, boxes):
    # Estimate single-character width from the narrower half of components.
    widths = sorted(b[2] - b[0] for b in boxes)
    typical = float(np.median(widths[:max(1, (len(widths) + 1) // 2)]))
    output, cuts = [], []
    for x0, y0, x1, y1 in boxes:
        width = x1 - x0
        count = max(1, int(np.floor(width / typical + 0.5))) if width > 1.45 * typical else 1
        projection = mask[y0:y1, x0:x1].sum(axis=0).astype(float)
        smooth = np.convolve(projection, np.ones(5) / 5, mode='same')
        edges = [x0]
        for j in range(1, count):
            target = j * width / count
            lo = max(1, int(target - 0.25 * typical))
            hi = min(width - 1, int(target + 0.25 * typical) + 1)
            # Cut through the weakest vertical foreground near the expected gap.
            local = smooth[lo:hi]
            candidates = np.flatnonzero(local == local.min()) + lo
            cut = int(candidates[np.argmin(abs(candidates - target))]) + x0
            edges.append(cut)
            cuts.append(cut)
        edges.append(x1)
        for left, right in zip(edges[:-1], edges[1:]):
            yy, xx = np.where(mask[y0:y1, left:right])
            if len(xx):
                output.append((left + int(xx.min()), y0 + int(yy.min()),
                               left + int(xx.max()) + 1, y0 + int(yy.max()) + 1))
    return sorted(output), cuts, typical


def normalize(binary, size=64, margin=6):
    """Black character on white; preserve aspect ratio and center with padding."""
    character = ImageOps.invert(binary)
    character.thumbnail((size - 2 * margin, size - 2 * margin), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (size, size), 255)
    canvas.paste(character, ((size - character.width) // 2, (size - character.height) // 2))
    return canvas


def run(args):
    source = Image.open(args.input).convert('RGB')
    roi = tuple(args.roi)
    if not (0 <= roi[0] < roi[2] <= source.width and 0 <= roi[1] < roi[3] <= source.height):
        raise ValueError('ROI must lie inside the image, with positive width and height.')
    if args.blur <= 0 or args.background <= args.blur or args.min_area < 1:
        raise ValueError('Require 0 < blur < background and min-area >= 1.')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    chars = out / 'characters'
    chars.mkdir(exist_ok=True)
    # This folder contains only generated numeric character files.
    for old in chars.glob('char_*_*.png'):
        old.unlink()
    source.save(out / '01_original.png')
    gray = source.convert('L')
    ImageOps.autocontrast(gray).save(out / '02_contrast.png')
    for size in (3, 5, 7):
        mean_filter(gray, size).save(out / f'03_mean_{size}x{size}.png')
    crop = gray.crop(roi)
    crop.save(out / '04_roi.png')
    # Work from the raw grayscale crop, not the contrast demonstration image.
    smoothed = crop.filter(ImageFilter.GaussianBlur(args.blur))
    background = crop.filter(ImageFilter.GaussianBlur(args.background))
    response = np.asarray(smoothed, dtype=float) - np.asarray(background, dtype=float)
    smoothed.save(out / '05_smoothed.png')
    background.save(out / '06_background.png')
    Image.fromarray(np.uint8(np.clip(response + 128, 0, 255))).save(out / '07_local_contrast.png')
    binary = response > args.threshold
    Image.fromarray(binary.astype('uint8') * 255).save(out / '08_binary.png')
    cleaned = np.zeros_like(binary)
    boxes = []
    for box, pts in components(binary):
        if len(pts) >= args.min_area and box[3] - box[1] >= 0.4 * crop.height:
            cleaned[pts[:, 0], pts[:, 1]] = True
            boxes.append(box)
    if not boxes:
        raise RuntimeError('No character candidates; inspect ROI and threshold.')
    clean_image = Image.fromarray(cleaned.astype('uint8') * 255)
    clean_image.save(out / '09_cleaned.png')
    boxes, cuts, typical = split_boxes(cleaned, boxes)
    # Foreground boundary includes external contours and hole boundaries.
    padded = np.pad(cleaned, 1)
    interior = cleaned & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    Image.fromarray(np.uint8(cleaned & ~interior) * 255).save(out / '10_contours.png')
    annotated = source.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle((roi[0], roi[1], roi[2]-1, roi[3]-1), outline='yellow', width=2)
    sheet = Image.new('RGB', (110 * len(boxes), 170), 'white')
    sd = ImageDraw.Draw(sheet)
    records = []
    for i, box in enumerate(boxes, 1):
        full = (box[0] + roi[0], box[1] + roi[1], box[2] + roi[0], box[3] + roi[1])
        draw.rectangle((full[0], full[1], full[2]-1, full[3]-1), outline='lime', width=2)
        draw.text((full[0], max(0, full[1]-15)), str(i), fill='yellow')
        gray.crop(full).save(chars / f'char_{i:02d}_gray.png')
        raw = clean_image.crop(box)
        raw.save(chars / f'char_{i:02d}_binary.png')
        norm = normalize(raw)
        norm.save(chars / f'char_{i:02d}_normalized.png')
        thumb = raw.convert('RGB')
        thumb.thumbnail((100, 135))
        sheet.paste(thumb, ((i-1)*110 + (110-thumb.width)//2, 25))
        sd.text(((i-1)*110+8, 5), f'{i:02d}', fill='black')
        records.append({'index': i, 'bbox_roi_xyxy': box, 'bbox_original_xyxy': full,
                        'foreground_pixels': int(cleaned[box[1]:box[3], box[0]:box[2]].sum())})
    annotated.save(out / '11_annotated.png')
    sheet.save(out / '12_character_sheet.png')
    projection = cleaned.sum(axis=0)
    plot = Image.new('RGB', (crop.width, 220), 'white')
    pd = ImageDraw.Draw(plot)
    points = [(x, 195-int(v / max(1, projection.max()) * 175)) for x, v in enumerate(projection)]
    pd.line(points, fill='black', width=2)
    for cut in cuts:
        pd.line((cut, 10, cut, 200), fill='red')
    pd.text((5, 202), 'Vertical foreground projection; red = split of touching components', fill='black')
    plot.save(out / '13_projection.png')
    meta = {'input': str(args.input), 'roi_xyxy': roi, 'parameters': {'blur': args.blur,
            'background': args.background, 'threshold': args.threshold, 'min_area': args.min_area},
            'count': len(boxes), 'estimated_character_width': typical, 'split_columns_roi': cuts,
            'coordinate_convention': 'xyxy, right/bottom exclusive', 'characters': records}
    (out / 'segmentation.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Segmented {len(boxes)} characters. Results: {out.resolve()}')


def mean_filter(image, size):
    """Explicit normalized box average, with replicated image borders."""
    a = np.asarray(image, dtype=float)
    p = np.pad(a, size//2, mode='edge')
    total = np.zeros_like(a)
    for y in range(size):
        for x in range(size):
            total += p[y:y+a.shape[0], x:x+a.shape[1]]
    return Image.fromarray(np.uint8(np.round(total / (size * size))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='dataset_2026_updated/dataset_2026/Test_Image_2026.bmp')
    parser.add_argument('--output', default='output/segmentation')
    parser.add_argument('--roi', nargs=4, type=int, default=[10, 185, 980, 350], metavar=('X0','Y0','X1','Y1'))
    parser.add_argument('--blur', type=float, default=3)
    parser.add_argument('--background', type=float, default=25)
    parser.add_argument('--threshold', type=float, default=10)
    parser.add_argument('--min-area', type=int, default=100)
    run(parser.parse_args())
