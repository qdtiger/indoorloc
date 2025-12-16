#!/usr/bin/env python
"""
Train Script for IndoorLoc

Usage:
    indoorloc-train configs/wifi/knn_ujindoorloc.yaml

    # With overrides
    indoorloc-train configs/wifi/knn_ujindoorloc.yaml --model.k 7

    # Specify work directory
    indoorloc-train configs/wifi/knn_ujindoorloc.yaml --work-dir work_dirs/exp1
"""
import argparse
import ast
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Train an indoor localization model')

    parser.add_argument('config', help='Path to config file')
    parser.add_argument('--work-dir', type=str, help='Working directory for outputs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-validate', action='store_true', help='Skip validation during training')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--opts', nargs='+', default=[], help='Config overrides')

    args, unknown = parser.parse_known_args()

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
    """Apply command-line overrides to config."""
    import copy
    cfg = copy.deepcopy(cfg)

    for i in range(0, len(overrides), 2):
        if i + 1 >= len(overrides):
            break

        key = overrides[i]
        value = overrides[i + 1]

        # Safe evaluation of literals only
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass  # Keep as string

        keys = key.split('.')
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    return cfg


def build_dataset_from_config(cfg: dict, split: str):
    """Build dataset from config dictionary.

    Args:
        cfg: Dataset configuration with 'type' and other parameters.
        split: Dataset split ('train', 'test', 'val').

    Returns:
        Dataset instance.
    """
    from ..registry import DATASETS

    cfg = cfg.copy()
    dataset_type = cfg.pop('type', 'UJIndoorLocDataset')

    dataset_cls = DATASETS.get(dataset_type)
    if dataset_cls is None:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. "
            f"Available: {DATASETS.list_modules()}"
        )

    return dataset_cls(split=split, **cfg)


def main() -> int:
    args = parse_args()

    import indoorloc as iloc
    from ..utils.config import Config
    from ..evaluation import Evaluator

    print("=" * 60)
    print("IndoorLoc Training")
    print("=" * 60)

    np.random.seed(args.seed)

    print(f"\nLoading config: {args.config}")
    cfg = Config.fromfile(args.config)
    cfg_dict = cfg.to_dict()

    if args.opts:
        cfg_dict = apply_overrides(cfg_dict, args.opts)
        print(f"Applied {len(args.opts) // 2} config overrides")

    work_dir = args.work_dir or cfg_dict.get('work_dir', 'work_dirs/default')
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Work directory: {work_dir}")

    print("\nBuilding dataset...")
    dataset_cfg = cfg_dict.get('dataset', {})

    try:
        train_dataset = build_dataset_from_config(dataset_cfg, split='train')
        print(f"  Training samples: {len(train_dataset)}")

        val_dataset = None
        if not args.no_validate:
            val_dataset = build_dataset_from_config(dataset_cfg, split='test')
            print(f"  Validation samples: {len(val_dataset)}")
    except (FileNotFoundError, RuntimeError) as e:
        dataset_type = dataset_cfg.get('type', 'UJIndoorLocDataset')
        data_root = dataset_cfg.get('data_root', 'data/ujindoorloc')
        print(f"\nError: Dataset not found!")
        print(f"Please download the dataset using:")
        print(f"  python -c \"import indoorloc as iloc; iloc.datasets.{dataset_type}.download('{data_root}')\"")
        print(f"\nOr with download=True:")
        print(f"  iloc.load_dataset('{dataset_type}', download=True)")
        return 1

    print("\nBuilding model...")
    model_cfg = cfg_dict.get('model', {})
    model = iloc.build_model(model_cfg)
    print(f"  Model: {model}")

    # Extract training config
    train_cfg = cfg_dict.get('train', {})
    fit_kwargs = {
        'epochs': train_cfg.get('epochs', 100),
        'batch_size': train_cfg.get('batch_size', 32),
        'lr': train_cfg.get('lr', 1e-3),
        'weight_decay': train_cfg.get('weight_decay', 1e-4),
        'early_stopping': train_cfg.get('early_stopping', 10),
        'verbose': train_cfg.get('verbose', True),
        'device': train_cfg.get('device'),
        'fp16': train_cfg.get('fp16', False),
    }

    if val_dataset is not None:
        fit_kwargs['val_data'] = val_dataset

    print("\nTraining...")
    start_time = time.time()

    # Pass dataset directly if model supports it, otherwise use signals/locations
    if hasattr(model, 'fit'):
        try:
            model.fit(train_dataset, **fit_kwargs)
        except TypeError:
            # Fallback for traditional models that don't accept kwargs
            model.fit(train_dataset.signals, train_dataset.locations)

    train_time = time.time() - start_time
    print(f"  Training completed in {train_time:.2f}s")

    model_path = work_dir / 'model.pkl'
    model.save(str(model_path))
    print(f"  Model saved to: {model_path}")

    if not args.no_validate and val_dataset is not None:
        print("\nValidating...")
        start_time = time.time()

        predictions = model.predict_batch(val_dataset.signals)

        val_time = time.time() - start_time
        print(f"  Inference completed in {val_time:.2f}s")
        print(f"  Throughput: {len(predictions) / val_time:.1f} samples/s")

        evaluator = Evaluator()
        evaluator.print_results(predictions, val_dataset.locations)

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
