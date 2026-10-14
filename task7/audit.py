"""Audit split integrity and exact pixel duplicates before model selection."""
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from importlib.metadata import version

import numpy as np

from .data import CLASSES, OUTPUT_ROOT, load_images, read_split, split_summary, stratified_split


def main():
    images, labels = load_images()
    splits = stratified_split(labels, images=images)
    saved = read_split()
    if saved is not None:
        saved_matches = all(np.array_equal(a, b) for a, b in zip(saved, splits))
    else:
        saved_matches = None
    joined = np.concatenate(splits)
    assert len(np.unique(joined)) == len(images) == len(joined)
    assert np.array_equal(np.sort(joined), np.arange(len(images)))
    hashes = [hashlib.sha256(im.tobytes()).hexdigest() for im in images]
    groups = defaultdict(list)
    for i, digest in enumerate(hashes):
        groups[digest].append(i)
    owner = {int(i): part for part, ids in zip(('train', 'validation', 'test'), splits) for i in ids}
    duplicates = [ids for ids in groups.values() if len(ids) > 1]
    cross = [ids for ids in duplicates if len({owner[i] for i in ids}) > 1]
    conflict = [ids for ids in duplicates if len(set(labels[ids])) > 1]
    legacy_splits = stratified_split(labels)
    legacy_owner = {int(i): part for part, ids in zip(('train', 'validation', 'test'), legacy_splits) for i in ids}
    legacy_cross = [ids for ids in duplicates if len({legacy_owner[i] for i in ids}) > 1]
    legacy_train_test = [ids for ids in duplicates if {'train', 'test'} <= {legacy_owner[i] for i in ids}]
    payload = {'environment': {'python': sys.version, 'executable': sys.executable,
               'platform': platform.platform(), 'processor': platform.processor(),
               'versions': {n: version(n) for n in ('numpy', 'Pillow', 'scikit-learn', 'scipy')},
               'thread_environment': {n: os.environ.get(n) for n in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}},
               'summary': split_summary(labels, *splits),
               'checks': {'complete_disjoint_partition': True, 'saved_split_matches_grouped_seed': saved_matches, 'split_method': 'sha256 decoded pixels grouped, class-stratified, exact 65/10/25 counts'},
               'decoded_pixel_dataset_sha256': hashlib.sha256(''.join(hashes).encode()).hexdigest(),
               'exact_duplicate_groups': len(duplicates), 'cross_split_duplicate_groups': len(cross),
               'conflicting_label_groups': len(conflict),
               'legacy_random_split': {'cross_split_duplicate_groups': len(legacy_cross),
                                      'train_test_duplicate_groups': len(legacy_train_test)},
               'cross_split_examples': [{'indices': ids, 'splits': [owner[i] for i in ids],
                                        'labels': [CLASSES[labels[i]] for i in ids]} for ids in cross[:20]],
               'limitations': 'Pixel hashes cannot detect near-duplicates or shared typefaces. No writer/font group metadata is provided.'}
    target = OUTPUT_ROOT / 'verification'
    target.mkdir(parents=True, exist_ok=True)
    (target / 'data_audit.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('Data audit:', payload['summary']['sizes'], 'cross-split exact duplicate groups:', len(cross), flush=True)
    assert not cross, 'Grouped split contains cross-split exact duplicates'
    if conflict:
        raise RuntimeError('Identical pixels have conflicting class labels; investigate before training.')
    return payload


if __name__ == '__main__':
    main()
