#!/usr/bin/env python
"""
Test Script for IndoorLoc

Evaluate a trained model on a test dataset.

Usage:
    python tools/test.py configs/wifi/knn_ujindoorloc.yaml checkpoint.pkl

    # With custom data root
    python tools/test.py configs/wifi/knn_ujindoorloc.yaml checkpoint.pkl --data-root data/ujindoorloc

    # Save predictions
    python tools/test.py configs/wifi/knn_ujindoorloc.yaml checkpoint.pkl --out predictions.csv
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
    parser = argparse.ArgumentParser(description='Test an indoor localization model')

    parser.add_argument('config', help='Path to config file')
    parser.add_argument('checkpoint', help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, help='Override dataset root path')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'val'],
                        help='Dataset split to evaluate')
    parser.add_argument('--out', type=str, help='Output file for predictions')
    parser.add_argument('--show', action='store_true', help='Show sample predictions')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')

    return parser.parse_args()


def save_predictions(predictions, ground_truths, output_path):
    """Save predictions to CSV file."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'pred_x', 'pred_y', 'pred_floor', 'pred_building',
            'gt_x', 'gt_y', 'gt_floor', 'gt_building',
            'error_2d', 'floor_correct', 'building_correct'
        ])

        for pred, gt in zip(predictions, ground_truths):
            pred_loc = pred.location if hasattr(pred, 'location') else pred
            error_2d = pred_loc.distance_to(gt)
            floor_correct = 1 if pred_loc.floor == gt.floor else 0
            building_correct = 1 if pred_loc.building_id == gt.building_id else 0

            writer.writerow([
                pred_loc.coordinate.x, pred_loc.coordinate.y,
                pred_loc.floor, pred_loc.building_id,
                gt.coordinate.x, gt.coordinate.y,
                gt.floor, gt.building_id,
                error_2d, floor_correct, building_correct
            ])


def main():
    args = parse_args()

    # Import after adding to path
    import indoorloc as iloc
    from indoorloc.utils.config import Config
    from indoorloc.evaluation import Evaluator

    print("=" * 60)
    print("IndoorLoc Testing")
    print("=" * 60)

    # Load config
    print(f"\nLoading config: {args.config}")
    cfg = Config.fromfile(args.config)
    cfg_dict = cfg.to_dict()

    # Build dataset
    print("\nLoading dataset...")
    dataset_cfg = cfg_dict.get('dataset', {})
    data_root = args.data_root or dataset_cfg.get('data_root', 'data/ujindoorloc')

    try:
        test_dataset = iloc.datasets.UJIndoorLocDataset(
            data_root=data_root,
            split=args.split,
            normalize=dataset_cfg.get('normalize', True),
            normalize_method=dataset_cfg.get('normalize_method', 'minmax')
        )
        print(f"  Test samples: {len(test_dataset)}")
        print(f"  Split: {args.split}")
    except FileNotFoundError as e:
        print(f"\nError: Dataset not found at {data_root}")
        print(f"Please download the UJIndoorLoc dataset first.")
        return 1

    # Print dataset statistics
    stats = test_dataset.get_statistics()
    print(f"  Buildings: {stats['num_buildings']}")
    print(f"  Floors: {stats['num_floors']}")
    print(f"  Avg detected APs: {stats.get('avg_detected_aps', 'N/A'):.1f}")

    # Load model
    print(f"\nLoading model: {args.checkpoint}")
    model_cfg = cfg_dict.get('model', {})
    model = iloc.build_model(model_cfg)

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1

    model.load(args.checkpoint)
    print(f"  Model: {model}")
    print(f"  Trained: {model.is_trained}")

    # Inference
    print("\nRunning inference...")
    start_time = time.time()

    predictions = model.predict_batch(test_dataset.signals)

    inference_time = time.time() - start_time
    print(f"  Inference completed in {inference_time:.2f}s")
    print(f"  Throughput: {len(predictions) / inference_time:.1f} samples/s")
    print(f"  Latency: {inference_time / len(predictions) * 1000:.2f} ms/sample")

    # Evaluation
    print("\nEvaluation Results:")
    evaluator = Evaluator()
    evaluator.print_results(predictions, test_dataset.locations)

    # Show sample predictions
    if args.show:
        print("\nSample Predictions (first 5):")
        print("-" * 80)
        for i in range(min(5, len(predictions))):
            pred = predictions[i]
            gt = test_dataset.locations[i]
            pred_loc = pred.location if hasattr(pred, 'location') else pred
            error = pred_loc.distance_to(gt)

            print(f"  Sample {i+1}:")
            print(f"    Predicted: ({pred_loc.coordinate.x:.1f}, {pred_loc.coordinate.y:.1f}), "
                  f"Floor {pred_loc.floor}, Building {pred_loc.building_id}")
            print(f"    Ground Truth: ({gt.coordinate.x:.1f}, {gt.coordinate.y:.1f}), "
                  f"Floor {gt.floor}, Building {gt.building_id}")
            print(f"    Error: {error:.2f}m")
        print("-" * 80)

    # Save predictions
    if args.out:
        print(f"\nSaving predictions to: {args.out}")
        save_predictions(predictions, test_dataset.locations, args.out)
        print("  Done!")

    print("\n" + "=" * 60)
    print("Testing completed!")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
