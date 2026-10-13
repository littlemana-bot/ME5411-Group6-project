"""Run the complete Task 7 pipeline in order.

    python run_task7.py                 # full run (~75% train / 25% test)
    python run_task7.py --quick         # 60 images per class, mostly a smoke test

Steps: 1) CNN with preprocessing and hyper-parameter study, 2) PCA + SVM
baseline on the identical split, 3) the 7-3 comparison tables and figures,
4) recognition of the characters segmented from the chip photo.
"""
from __future__ import annotations

import argparse
import time

from task7 import baseline, compare, recognize, train_cnn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='small subset, few epochs')
    parser.add_argument('--per-class', type=int, default=None, help='samples per class (smoke test)')
    parser.add_argument('--epochs', type=int, default=25, help='epochs for the final CNN')
    parser.add_argument('--experiment-epochs', type=int, default=8, help='epochs per CNN experiment')
    args = parser.parse_args()
    if args.quick:
        args.per_class = args.per_class or 60
        args.epochs = min(args.epochs, 3)
        args.experiment_epochs = 1

    start = time.perf_counter()
    train_cnn.main(argparse.Namespace(epochs=args.epochs,
                                      experiment_epochs=args.experiment_epochs,
                                      per_class=args.per_class))
    baseline.main(argparse.Namespace(per_class=args.per_class))
    compare.main(argparse.Namespace())
    recognize.main(argparse.Namespace())
    print(f'Task 7 pipeline finished in {time.perf_counter() - start:.1f}s')


if __name__ == '__main__':
    main()
