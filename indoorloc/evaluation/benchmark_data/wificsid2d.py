"""
WiFi CSI D2D Dataset Benchmarks

WiFi CSI dataset for device-to-device localization with distance and angle measurements.

Dataset: WiFi CSI D2D
- 5 GHz band, Intel 5300 NIC
- 3 Rx antennas x 30 subcarriers = 90 features
- 5 distances (1-5m) x 2 angles (30deg, -60deg)

Dataset URL: https://doi.org/10.6084/m9.figshare.20943706.v1

Note: Using demo data for benchmarks until real data is available.
"""
from ..benchmarks import BenchmarkEntry, DatasetBenchmarks

BENCHMARKS = DatasetBenchmarks(
    dataset_name="wificsid2d",
    display_name="WiFi CSI D2D",
    default_metric="mean_error",
    entries=[
        BenchmarkEntry(
            method="ResNet18",
            mean_error=2.54,
            source="IndoorLoc Library (Demo Data, 10 epochs)",
            year=2024,
            notes="Demo data; replace with real data results",
        ),
        BenchmarkEntry(
            method="KNN (k=3)",
            mean_error=2.65,
            source="IndoorLoc Library Baseline (Demo Data)",
            year=2024,
            notes="Demo data baseline",
        ),
    ],
)
