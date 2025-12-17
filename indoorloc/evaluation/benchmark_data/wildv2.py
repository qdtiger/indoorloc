"""
WILD-v2 Dataset Benchmarks

WiFi Indoor Localization Dataset v2 - Robot-collected WiFi CSI for dense indoor mapping.

Dataset: WILD-v2
- 8 indoor environments (simple/complex, LOS/NLOS mixed)
- Dense scanning along multiple trajectories
- Robot SLAM/external system provides ground truth 2D positions

Dataset URL: https://www.kaggle.com/c/wild-v2

Note: Using demo data for benchmarks until real data is available.
"""
from ..benchmarks import BenchmarkEntry, DatasetBenchmarks

BENCHMARKS = DatasetBenchmarks(
    dataset_name="wildv2",
    display_name="WILD-v2",
    default_metric="mean_error",
    entries=[
        BenchmarkEntry(
            method="ResNet18",
            mean_error=8.40,
            source="IndoorLoc Library (Demo Data, 10 epochs)",
            year=2024,
            notes="Demo data; replace with real data results",
        ),
        BenchmarkEntry(
            method="KNN (k=3)",
            mean_error=8.74,
            source="IndoorLoc Library Baseline (Demo Data)",
            year=2024,
            notes="Demo data baseline",
        ),
    ],
)
