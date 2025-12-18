#!/usr/bin/env python3
"""
Dataset Health Check Script

Runs smoke tests on all registered datasets to verify:
1. Download works (if applicable)
2. Data loads correctly
3. Basic training works (KNN)
4. Prediction works
5. Metrics can be computed

Output: Status matrix showing which datasets work/fail at each stage.
"""
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class Status(Enum):
    PASS = "✅"
    FAIL = "❌"
    SKIP = "⏭️"
    WARN = "⚠️"
    PENDING = "⏳"


@dataclass
class DatasetResult:
    name: str
    download: Status = Status.PENDING
    load: Status = Status.PENDING
    train: Status = Status.PENDING
    predict: Status = Status.PENDING
    metric: Status = Status.PENDING
    error_msg: str = ""
    load_time: float = 0.0
    train_time: float = 0.0
    num_samples: int = 0
    notes: str = ""


def test_dataset(dataset_name: str, timeout: int = 300) -> DatasetResult:
    """Run smoke test on a single dataset."""
    result = DatasetResult(name=dataset_name)

    try:
        from indoorloc.registry import DATASETS
        dataset_cls = DATASETS.get(dataset_name)

        if dataset_cls is None:
            result.error_msg = f"Dataset {dataset_name} not found in registry"
            result.download = Status.FAIL
            return result

        # Step 1: Download / Load
        start_time = time.time()
        try:
            # Try with download=True first
            train_ds = dataset_cls(split='train', download=True)
            result.download = Status.PASS
            result.load = Status.PASS
            result.load_time = time.time() - start_time
            result.num_samples = len(train_ds)
        except FileNotFoundError as e:
            # Dataset doesn't support auto-download
            result.download = Status.SKIP
            result.notes = "Manual download required"
            try:
                train_ds = dataset_cls(split='train', download=False)
                result.load = Status.PASS
                result.load_time = time.time() - start_time
                result.num_samples = len(train_ds)
            except Exception as e2:
                result.load = Status.FAIL
                result.error_msg = str(e2)[:200]
                return result
        except Exception as e:
            result.download = Status.FAIL
            result.load = Status.FAIL
            result.error_msg = str(e)[:200]
            return result

        # Step 2: Check data format
        try:
            sample = train_ds[0]
            if not isinstance(sample, tuple) or len(sample) < 2:
                result.notes += "; Unusual sample format"
            signal, location = sample[0], sample[1]
        except Exception as e:
            result.load = Status.WARN
            result.notes += f"; Sample access issue: {str(e)[:50]}"

        # Step 3: Train simple KNN
        start_time = time.time()
        try:
            import indoorloc as iloc
            model = iloc.create_model('knn', k=3)

            # Get signals and locations
            if hasattr(train_ds, 'signals') and hasattr(train_ds, 'locations'):
                signals = train_ds.signals
                locations = train_ds.locations
            else:
                # Fallback: iterate through dataset
                signals = []
                locations = []
                max_samples = min(1000, len(train_ds))  # Limit for speed
                for i in range(max_samples):
                    s, l = train_ds[i]
                    signals.append(s)
                    locations.append(l)
                import numpy as np
                signals = np.array(signals)
                locations = np.array(locations) if not hasattr(locations[0], 'x') else locations

            model.fit(signals, locations)
            result.train = Status.PASS
            result.train_time = time.time() - start_time
        except Exception as e:
            result.train = Status.FAIL
            result.error_msg = f"Train: {str(e)[:150]}"
            return result

        # Step 4: Predict
        try:
            test_signal = signals[0] if len(signals) > 0 else train_ds[0][0]
            pred = model.predict(test_signal)
            result.predict = Status.PASS
        except Exception as e:
            result.predict = Status.FAIL
            result.error_msg = f"Predict: {str(e)[:150]}"
            return result

        # Step 5: Compute metric
        try:
            if hasattr(pred, 'x') and hasattr(locations[0], 'x'):
                true_loc = locations[0]
                error = ((pred.x - true_loc.x)**2 + (pred.y - true_loc.y)**2)**0.5
            else:
                # Array-based locations
                import numpy as np
                true_loc = locations[0]
                if hasattr(pred, 'x'):
                    error = ((pred.x - true_loc[0])**2 + (pred.y - true_loc[1])**2)**0.5
                else:
                    error = np.sqrt(np.sum((np.array(pred[:2]) - np.array(true_loc[:2]))**2))
            result.metric = Status.PASS
            result.notes += f"; Test error: {error:.2f}m" if error < 10000 else ""
        except Exception as e:
            result.metric = Status.WARN
            result.notes += f"; Metric issue: {str(e)[:50]}"

    except Exception as e:
        result.error_msg = f"Unexpected: {str(e)[:200]}"
        traceback.print_exc()

    return result


def print_status_matrix(results: List[DatasetResult]):
    """Print formatted status matrix."""
    print("\n" + "=" * 90)
    print("                         DATASET HEALTH CHECK STATUS MATRIX")
    print("=" * 90)
    print(f"{'Dataset':<28} {'Download':^10} {'Load':^10} {'Train':^10} {'Predict':^10} {'Metric':^10}")
    print("-" * 90)

    for r in results:
        print(f"{r.name:<28} {r.download.value:^10} {r.load.value:^10} {r.train.value:^10} {r.predict.value:^10} {r.metric.value:^10}")

    print("=" * 90)

    # Summary
    total = len(results)
    fully_working = sum(1 for r in results if r.download in (Status.PASS, Status.SKIP)
                        and r.load == Status.PASS
                        and r.train == Status.PASS
                        and r.predict == Status.PASS)
    download_ok = sum(1 for r in results if r.download == Status.PASS)
    load_ok = sum(1 for r in results if r.load == Status.PASS)

    print(f"\nSummary:")
    print(f"  Total datasets: {total}")
    print(f"  Fully working (download→predict): {fully_working}/{total}")
    print(f"  Auto-download available: {download_ok}/{total}")
    print(f"  Load OK: {load_ok}/{total}")

    # Print errors
    errors = [r for r in results if r.error_msg]
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for r in errors:
            print(f"  - {r.name}: {r.error_msg[:70]}...")

    # Print notes
    with_notes = [r for r in results if r.notes.strip()]
    if with_notes:
        print(f"\nNotes:")
        for r in with_notes:
            print(f"  - {r.name}: {r.notes.strip()}")


def main():
    """Run health check on all datasets."""
    from indoorloc.registry import DATASETS

    dataset_names = sorted(DATASETS.module_dict.keys())

    print(f"Running health check on {len(dataset_names)} datasets...")
    print("This may take a while (downloading data, training models)...\n")

    results = []

    for i, name in enumerate(dataset_names, 1):
        print(f"[{i}/{len(dataset_names)}] Testing {name}...", end=" ", flush=True)
        result = test_dataset(name)
        status = "✅" if result.predict == Status.PASS else "❌"
        print(f"{status} ({result.load_time:.1f}s)")
        results.append(result)

    print_status_matrix(results)

    # Save results to JSON
    import json
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(results),
        "fully_working": sum(1 for r in results if r.predict == Status.PASS),
        "datasets": [
            {
                "name": r.name,
                "download": r.download.name,
                "load": r.load.name,
                "train": r.train.name,
                "predict": r.predict.name,
                "metric": r.metric.name,
                "num_samples": r.num_samples,
                "error": r.error_msg,
                "notes": r.notes,
            }
            for r in results
        ]
    }

    output_path = Path(__file__).parent / "dataset_status.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
