#!/usr/bin/env python
"""
Train Script for IndoorLoc

Usage:
    python tools/train.py configs/wifi/knn_ujindoorloc.yaml

    # With overrides
    python tools/train.py configs/wifi/knn_ujindoorloc.yaml --model.k 7

    # Specify work directory
    python tools/train.py configs/wifi/knn_ujindoorloc.yaml --work-dir work_dirs/exp1
"""
import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Train an indoor localization model')

    parser.add_argument('config', help='Path to config file')
    parser.add_argument('--work-dir', type=str, help='Working directory for outputs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-validate', action='store_true', help='Skip validation during training')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')

    # Config overrides (format: --key value or --key.subkey value)
    parser.add_argument('--opts', nargs='+', default=[], help='Config overrides')

    args, unknown = parser.parse_known_args()

    # Parse unknown args as config overrides
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--'):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                args.opts.extend([key, unknown[i + 1]])
                i += 2
            else:
                i += 1
        else:
            i += 1

    return args


def apply_overrides(cfg: dict, overrides: list) -> dict:
    """Apply command-line overrides to config.

    Args:
        cfg: Config dictionary.
        overrides: List of key-value pairs ['key1', 'val1', 'key2', 'val2', ...]

    Returns:
        Modified config dictionary.
    """
    import copy
    cfg = copy.deepcopy(cfg)

    for i in range(0, len(overrides), 2):
        if i + 1 >= len(overrides):
            break

        key = overrides[i]
        value = overrides[i + 1]

        # Parse value
        try:
            value = eval(value)
        except:
            pass  # Keep as string

        # Navigate to nested key
        keys = key.split('.')
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    return cfg


def main():
    args = parse_args()

    # Import after adding to path
    import indoorloc as iloc
    from indoorloc.utils.config import Config
    from indoorloc.evaluation import Evaluator

    print("=" * 60)
    print("IndoorLoc Training")
    print("=" * 60)

    # Set random seed
    np.random.seed(args.seed)

    # Load config
    print(f"\nLoading config: {args.config}")
    cfg = Config.fromfile(args.config)
    cfg_dict = cfg.to_dict()

    # Apply overrides
    if args.opts:
        cfg_dict = apply_overrides(cfg_dict, args.opts)
        print(f"Applied {len(args.opts) // 2} config overrides")

    # Setup work directory
    work_dir = args.work_dir or cfg_dict.get('work_dir', 'work_dirs/default')
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Work directory: {work_dir}")

    # Build dataset
    print("\nBuilding dataset...")
    dataset_cfg = cfg_dict.get('dataset', {})
    data_root = dataset_cfg.get('data_root', 'data/ujindoorloc')

    try:
        train_dataset = iloc.datasets.UJIndoorLocDataset(
            data_root=data_root,
            split='train',
            normalize=dataset_cfg.get('normalize', True),
            normalize_method=dataset_cfg.get('normalize_method', 'minmax')
        )
        print(f"  Training samples: {len(train_dataset)}")

        if not args.no_validate:
            val_dataset = iloc.datasets.UJIndoorLocDataset(
                data_root=data_root,
                split='test',
                normalize=dataset_cfg.get('normalize', True),
                normalize_method=dataset_cfg.get('normalize_method', 'minmax')
            )
            print(f"  Validation samples: {len(val_dataset)}")
    except FileNotFoundError as e:
        print(f"\nError: Dataset not found!")
        print(f"Please download the UJIndoorLoc dataset:")
        print(f"  python -c \"from indoorloc.datasets import UJIndoorLocDataset; UJIndoorLocDataset.download('{data_root}')\"")
        print(f"\nOr manually download from:")
        print(f"  https://archive.ics.uci.edu/dataset/310/ujiindoorloc")
        return 1

    # Build model
    print("\nBuilding model...")
    model_cfg = cfg_dict.get('model', {})
    model = iloc.build_model(model_cfg)
    print(f"  Model: {model}")

    # Training
    print("\nTraining...")
    start_time = time.time()

    model.fit(train_dataset.signals, train_dataset.locations)

    train_time = time.time() - start_time
    print(f"  Training completed in {train_time:.2f}s")

    # Save model
    model_path = work_dir / 'model.pkl'
    model.save(str(model_path))
    print(f"  Model saved to: {model_path}")

    # Validation
    if not args.no_validate:
        print("\nValidating...")
        start_time = time.time()

        predictions = model.predict_batch(val_dataset.signals)

        val_time = time.time() - start_time
        print(f"  Inference completed in {val_time:.2f}s")
        print(f"  Throughput: {len(predictions) / val_time:.1f} samples/s")

        # Evaluation
        evaluator = Evaluator()
        evaluator.print_results(predictions, val_dataset.locations)

        # Save results
        results_path = work_dir / 'results.txt'
        with open(results_path, 'w') as f:
            results = evaluator.evaluate(predictions, val_dataset.locations)
            f.write("Evaluation Results\n")
            f.write("=" * 50 + "\n")
            for name, value in results.items():
                if isinstance(value, dict):
                    f.write(f"\n{name}:\n")
                    for k, v in value.items():
                        f.write(f"  {k}: {v:.2f}%\n")
                else:
                    f.write(f"{name}: {value:.4f}\n")

        print(f"Results saved to: {results_path}")

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
