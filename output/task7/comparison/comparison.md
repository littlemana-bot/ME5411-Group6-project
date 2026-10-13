# Task 7-3 comparison (numbers generated from the saved metrics)

## Overall accuracy on the held-out 25% test set

| metric | CNN (7-1) | PCA + SVM (7-2) | difference |
| --- | --- | --- | --- |
| accuracy | 0.9910 | 0.9921 | -0.0011 |
| macro F1 | 0.9910 | 0.9921 | -0.0011 |
| macro precision | 0.9910 | 0.9921 | -0.0011 |
| macro recall | 0.9910 | 0.9921 | -0.0011 |

## Cost

| metric | CNN (7-1) | PCA + SVM (7-2) | ratio |
| --- | --- | --- | --- |
| training time (s) | 255.2 | 0.2 | 1418x |
| inference, 1 image (ms) | 0.850 | 0.300 | 2.8x |
| inference, batched (ms/image) | 0.606 | 0.089 | 6.8x |
| stored model (bytes) | 1073874 | 1050368 | 1.0x |
| free parameters | 267975 | 72100 | 3.7x |

## Per-class F1

| class | test samples | CNN F1 | PCA + SVM F1 |
| --- | --- | --- | --- |
| 0 | 254 | 0.9843 | 0.9861 |
| 4 | 254 | 0.9961 | 0.9961 |
| 7 | 254 | 1.0000 | 0.9980 |
| 8 | 254 | 0.9902 | 0.9882 |
| A | 254 | 0.9941 | 0.9902 |
| D | 254 | 0.9862 | 0.9921 |
| H | 254 | 0.9861 | 0.9941 |

## Configurations compared

- CNN: crop_pad/32 input, channels [16, 32], lr 0.003, augmentation False, seed 0
- PCA + SVM: crop/32 input, 100 PCA components (0.935 variance), C=10.0, gamma=scale, 714 support vectors

## Observations

- On the same stratified split, the PCA + SVM baseline is more accurate (0.9910 vs 0.9921, gap 0.0011).
- Training cost differs by roughly 1418x (255.2s versus 0.2s), while single-image inference is slower for the CNN (0.850 ms versus 0.300 ms).
- CNN residual errors: 0 misread as D (2x), 8 misread as 0 (2x), A misread as H (2x).
- PCA + SVM residual errors: 0 misread as 8 (4x), 8 misread as 0 (2x), D misread as A (2x).
- Interpretation: with only seven visually distinct glyphs, both methods separate the classes well; the classical pipeline reaches its result in seconds and with far less data, while the CNN needs many epochs to reach the same level. The CNN advantage would be expected to grow with more classes, more style variation or noisier inputs.
