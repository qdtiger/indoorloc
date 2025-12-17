"""
CSI Indoor Dataset Benchmarks

WiFi CSI-based indoor positioning dataset with fine-grained channel information.

Dataset: CSI Indoor
- WiFi 802.11n, 52 subcarriers
- Multiple floors

Dataset URL: https://github.com/csi-positioning/indoor-dataset

Note: Using demo data for benchmarks until real data is available.
"""
from ..benchmarks import BenchmarkEntry, DatasetBenchmarks

BENCHMARKS = DatasetBenchmarks(
    dataset_name="csiindoor",
    display_name="CSI Indoor",
    default_metric="mean_error",
    entries=[
        BenchmarkEntry(
            method="ResNet18",
            mean_error=18.17,
            source="IndoorLoc Library (Demo Data, 10 epochs)",
            year=2024,
            notes="Demo data; replace with real data results",
        ),
        BenchmarkEntry(
            method="KNN (k=3)",
            mean_error=20.83,
            source="IndoorLoc Library Baseline (Demo Data)",
            year=2024,
            notes="Demo data baseline",
        ),
    ],
)
