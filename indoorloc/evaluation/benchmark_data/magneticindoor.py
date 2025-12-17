"""
Magnetic Indoor Dataset Benchmarks

Geomagnetic field-based indoor positioning using magnetometer measurements.

Dataset: Magnetic Indoor
- 3-axis magnetic field measurements (mx, my, mz) in μT
- Multiple buildings and floors

Dataset URL: https://zenodo.org/record/4321098

Note: Using demo data for benchmarks until real data is available.
"""
from ..benchmarks import BenchmarkEntry, DatasetBenchmarks

BENCHMARKS = DatasetBenchmarks(
    dataset_name="magneticindoor",
    display_name="Magnetic Indoor",
    default_metric="mean_error",
    entries=[
        BenchmarkEntry(
            method="ResNet18",
            mean_error=18.05,
            source="IndoorLoc Library (Demo Data, 10 epochs)",
            year=2024,
            notes="Demo data; replace with real data results",
        ),
        BenchmarkEntry(
            method="KNN (k=3)",
            mean_error=20.56,
            source="IndoorLoc Library Baseline (Demo Data)",
            year=2024,
            notes="Demo data baseline",
        ),
    ],
)
