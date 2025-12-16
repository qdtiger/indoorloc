#!/usr/bin/env python
"""
Benchmark Comparison Tool for IndoorLoc

Compare your results against published paper benchmarks.

Usage:
    indoorloc-benchmark --dataset ujindoorloc --mean-error 10.5 --floor-accuracy 0.92

    # List available datasets with benchmarks
    indoorloc-benchmark --list

    # Show detailed benchmarks for a dataset
    indoorloc-benchmark --dataset ujindoorloc --show
"""
import argparse
import sys
from typing import Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare results against published benchmarks'
    )

    parser.add_argument('--dataset', type=str, help='Dataset name (e.g., ujindoorloc)')
    parser.add_argument('--mean-error', type=float, help='Your mean position error in meters')
    parser.add_argument('--median-error', type=float, help='Your median position error in meters')
    parser.add_argument('--floor-accuracy', type=float,
                        help='Your floor accuracy as fraction 0-1')
    parser.add_argument('--building-accuracy', type=float,
                        help='Your building accuracy as fraction 0-1')
    parser.add_argument('--list', action='store_true', help='List datasets with benchmarks')
    parser.add_argument('--show', action='store_true', help='Show benchmark details')

    return parser.parse_args()


def print_benchmarks(dataset_name: str):
    """Print benchmark details for a dataset."""
    from ..evaluation import get_benchmarks_for_dataset

    benchmarks = get_benchmarks_for_dataset(dataset_name)
    if benchmarks is None:
        print(f"No benchmarks found for dataset: {dataset_name}")
        return

    print(f"\n{'=' * 70}")
    print(f"Benchmarks for: {benchmarks.display_name}")
    print(f"{'=' * 70}")

    entries = benchmarks.sorted_by_error()
    sota = benchmarks.get_sota()

    print(f"\n{'Method':<25} {'Mean Error':<12} {'Floor Acc':<12} {'Year':<6} {'Source'}")
    print("-" * 70)

    for entry in entries:
        sota_marker = " *" if entry is sota else ""
        floor_acc = f"{entry.floor_accuracy*100:.1f}%" if entry.floor_accuracy else "N/A"
        year = str(entry.year) if entry.year else "N/A"

        print(f"{entry.method:<25} {entry.mean_error:<12.2f} {floor_acc:<12} {year:<6} {entry.source[:30]}{sota_marker}")

    print("-" * 70)
    print("* = Current best (SOTA)")


def compare_results(
    dataset_name: str,
    mean_error: Optional[float] = None,
    median_error: Optional[float] = None,
    floor_accuracy: Optional[float] = None,
    building_accuracy: Optional[float] = None,
):
    """Compare user results against benchmarks."""
    from ..evaluation import get_benchmarks_for_dataset

    benchmarks = get_benchmarks_for_dataset(dataset_name)
    if benchmarks is None:
        print(f"No benchmarks found for dataset: {dataset_name}")
        return

    print(f"\n{'=' * 70}")
    print(f"Comparison against {benchmarks.display_name} Benchmarks")
    print(f"{'=' * 70}")

    entries = benchmarks.sorted_by_error()
    sota = benchmarks.get_sota()

    if mean_error is not None:
        print(f"\nYour Mean Error: {mean_error:.2f}m")

        # Find ranking
        better_than = sum(1 for e in entries if e.mean_error > mean_error)
        worse_than = sum(1 for e in entries if e.mean_error < mean_error)
        total = len(entries)

        print(f"Ranking: {worse_than + 1}/{total + 1} (better than {better_than}/{total} benchmarks)")

        if sota:
            diff = mean_error - sota.mean_error
            if diff > 0:
                print(f"  vs SOTA ({sota.method}): +{diff:.2f}m worse")
            else:
                print(f"  vs SOTA ({sota.method}): {diff:.2f}m better (new SOTA!)")

    if floor_accuracy is not None:
        print(f"\nYour Floor Accuracy: {floor_accuracy*100:.1f}%")

        entries_with_floor = [e for e in entries if e.floor_accuracy is not None]
        if entries_with_floor:
            better_than = sum(1 for e in entries_with_floor if e.floor_accuracy < floor_accuracy)
            print(f"  Better than {better_than}/{len(entries_with_floor)} benchmarks")

    if building_accuracy is not None:
        print(f"\nYour Building Accuracy: {building_accuracy*100:.1f}%")

        entries_with_building = [e for e in entries if e.building_accuracy is not None]
        if entries_with_building:
            better_than = sum(1 for e in entries_with_building if e.building_accuracy < building_accuracy)
            print(f"  Better than {better_than}/{len(entries_with_building)} benchmarks")

    print(f"\n{'=' * 70}")


def main() -> int:
    args = parse_args()

    from ..evaluation import list_datasets_with_benchmarks

    if args.list:
        datasets = list_datasets_with_benchmarks()
        print("\nDatasets with available benchmarks:")
        print("-" * 40)
        for ds in datasets:
            print(f"  - {ds}")
        print("\nUse --dataset <name> --show to see benchmark details")
        return 0

    if args.dataset is None:
        print("Error: --dataset is required")
        print("Use --list to see available datasets")
        return 1

    if args.show:
        print_benchmarks(args.dataset)
        return 0

    if args.mean_error is None and args.median_error is None:
        print("Error: At least --mean-error or --median-error is required for comparison")
        print("Use --show to just view benchmarks")
        return 1

    compare_results(
        args.dataset,
        mean_error=args.mean_error,
        median_error=args.median_error,
        floor_accuracy=args.floor_accuracy,
        building_accuracy=args.building_accuracy,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
