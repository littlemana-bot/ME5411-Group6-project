"""Fixed parameter sensitivity grid on one chip image; never used to select classifiers."""
import json

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

from segment_characters import components, normalize, split_boxes
from .data import OUTPUT_ROOT, PROJECT_ROOT
from .metrics import load_font
from .recognize import GROUND_TRUTH, classify, load_baseline, load_cnn


def main():
    roi = (10, 185, 980, 350)
    source = PROJECT_ROOT / 'dataset_2026_updated/dataset_2026/Test_Image_2026.bmp'
    image = Image.open(source).convert('L').crop(roi)
    background = np.asarray(image.filter(ImageFilter.GaussianBlur(25)), dtype=float)
    model, config = load_cnn()
    bundle, baseline = load_baseline()
    records, tiles, titles = [], [], []
    reference = json.loads((PROJECT_ROOT / 'output/segmentation/segmentation.json').read_text())
    ref_boxes = [c['bbox_roi_xyxy'] for c in reference['characters']]
    for blur in (2, 3, 4):
        response = np.asarray(image.filter(ImageFilter.GaussianBlur(blur)), dtype=float) - background
        for threshold in (8, 10, 12):
            mask = response > threshold
            cleaned = np.zeros_like(mask)
            candidates = []
            for box, pts in components(mask):
                if len(pts) >= 100 and box[3] - box[1] >= 0.4 * image.height:
                    cleaned[pts[:, 0], pts[:, 1]] = True
                    candidates.append(box)
            boxes, _, _ = split_boxes(cleaned, candidates) if candidates else ([], [], 0)
            binary = Image.fromarray(cleaned.astype('uint8') * 255)
            predictions = [classify(model, config, bundle, baseline,
                           np.asarray(normalize(binary.crop(box)))) for box in boxes]
            cnn_string = ''.join(p['cnn']['class'] for p in predictions)
            svm_string = ''.join(p['pca_svm']['class'] for p in predictions)
            entry = {'blur': blur, 'threshold': threshold, 'count': len(boxes),
                     'boxes_roi_xyxy': boxes, 'cnn_string': cnn_string, 'pca_svm_string': svm_string,
                     'cnn_exact_string': cnn_string == GROUND_TRUTH,
                     'svm_exact_string': svm_string == GROUND_TRUTH}
            # IoU describes stability relative to the default algorithm, not ground-truth detection accuracy.
            if len(boxes) == len(ref_boxes):
                ious = []
                for a, b in zip(boxes, ref_boxes):
                    intersection = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
                    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
                    ious.append(intersection / union)
                entry['mean_box_iou_vs_default'] = float(np.mean(ious))
            records.append(entry)
            tiles.append(np.asarray(binary, dtype=float)/255)
            titles.append(f'blur={blur}, threshold={threshold}, count={len(boxes)}')
            print('Segmentation study:', entry['blur'], entry['threshold'], len(boxes), cnn_string, svm_string, flush=True)
    target = OUTPUT_ROOT / 'verification'
    target.mkdir(parents=True, exist_ok=True)
    # Preserve the wide line's aspect ratio rather than stretching into square tiles.
    canvas = Image.new('RGB', (image.width, 40 + len(tiles) * (image.height + 30)), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), 'Nine fixed settings on the same chip image', fill='black', font=load_font(18))
    for i, (tile, title) in enumerate(zip(tiles, titles)):
        y = 40 + i * (image.height + 30)
        draw.text((8, y), title, fill='black', font=load_font(15))
        canvas.paste(Image.fromarray(np.uint8(tile * 255)).convert('RGB'), (0, y + 25))
    canvas.save(target/'segmentation_grid.png')
    payload = {'roi': roi, 'background_sigma': 25, 'min_area': 100, 'settings': records,
               'limitations': 'One image only. Counts and box IoU assess sensitivity, not ground-truth segmentation accuracy. No model was trained or selected using the chip image.'}
    (target/'segmentation_study.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload


if __name__ == '__main__':
    main()
