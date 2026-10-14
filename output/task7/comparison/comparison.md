# Task 7-3 comparison (numbers generated from the saved metrics)

## Overall accuracy on the held-out 25% test set

| metric | CNN (7-1) | PCA + SVM (7-2) | difference |
| --- | --- | --- | --- |
| accuracy | 0.9921 | 0.9910 | +0.0011 |
| macro F1 | 0.9921 | 0.9910 | +0.0011 |
| macro precision | 0.9922 | 0.9911 | +0.0011 |
| macro recall | 0.9921 | 0.9910 | +0.0011 |

## Cost

| metric | CNN (7-1) | PCA + SVM (7-2) | ratio |
| --- | --- | --- | --- |
| training time (s) | 299.5 | 1.0 | 312x |
| inference, 1 image (ms) | 0.970 | 0.780 | 1.2x |
| inference, batched (ms/image) | 0.779 | 0.165 | 4.7x |
| stored model (bytes) | 1073874 | 1196912 | 0.9x |

## Per-class F1

| class | test samples | CNN F1 | PCA + SVM F1 |
| --- | --- | --- | --- |
| 0 | 254 | 1.0000 | 0.9902 |
| 4 | 254 | 0.9961 | 0.9941 |
| 7 | 254 | 0.9980 | 1.0000 |
| 8 | 254 | 0.9961 | 0.9960 |
| A | 254 | 0.9785 | 0.9820 |
| D | 254 | 0.9980 | 0.9902 |
| H | 254 | 0.9781 | 0.9844 |

## Configurations compared

- CNN: crop_pad/32 input, channels [16, 32], lr 0.003, augmentation False, seed 0
- PCA + SVM: crop_pad/32 input, 100 PCA components (0.916 variance), C=10.0, gamma=scale, 886 support vectors

## Observations

- On the same stratified split, the CNN is more accurate (0.9921 vs 0.9910, gap 0.0011).
- Training cost differs by roughly 312x (299.5s versus 1.0s), while single-image inference is slower for the CNN (0.970 ms versus 0.780 ms).
- CNN residual errors: H misread as A (6x), A misread as H (3x), 4 misread as A (1x).
- PCA + SVM residual errors: A misread as H (7x), 0 misread as D (2x), A misread as 4 (2x).
- Both models use the same training/validation/test indices. Small accuracy differences on this one split do not establish general superiority; no data-efficiency experiment was performed.
- Training timing excludes image loading/resizing and hyperparameter search for both methods. The SVM includes scaler/PCA fitting and feature transforms; the CNN includes per-epoch validation and checkpoint selection.
- Inference uses the same warmed-up protocol: median of three repeats, batches of 256 and 200 randomly selected single images. SVM inference includes scaler/PCA transforms; both exclude image preprocessing and file I/O.
- Model bytes are compared directly. CNN trainable parameters and SVM/PCA stored state are different quantities and are not presented as an equivalent parameter-count ratio.
