"""Shared, warmed-up timing on already resized image tensors (no file I/O)."""
import time

import numpy as np


def benchmark_predict(predict, x, repeats=3, batch_size=256, sample_size=200):
    predict(x[:min(batch_size, len(x))])
    ids = np.random.default_rng(42).permutation(len(x))[:min(sample_size, len(x))]
    batch_seconds, single_ms = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        for i in range(0, len(x), batch_size):
            predict(x[i:i + batch_size])
        batch_seconds.append(time.perf_counter() - start)
        start = time.perf_counter()
        for i in ids:
            predict(x[i:i + 1])
        single_ms.append(1000 * (time.perf_counter() - start) / len(ids))
    seconds = float(np.median(batch_seconds))
    return {'batched_seconds': seconds,
            'ms_per_image_batched': 1000 * seconds / len(x),
            'ms_per_image_single': float(np.median(single_ms)),
            'batch_seconds_runs': batch_seconds, 'single_ms_runs': single_ms,
            'repeats': repeats, 'batch_size': batch_size, 'single_sample_size': len(ids),
            'scope': 'From preprocessed image tensor to class; SVM includes scaler and PCA transforms; excludes image resizing and file I/O'}
