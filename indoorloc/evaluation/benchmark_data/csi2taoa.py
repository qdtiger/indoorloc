"""
CSI2TAoA Dataset Benchmarks

Arena indoor RF testbed CSI dataset for Time-Angle of Arrival estimation.

Dataset: CSI2TAoA
- Arena indoor RF testbed
- CSI features for TAoA estimation
- 2D grid coordinate labels

Dataset URL: https://service.tib.eu/ldmservice/dataset/csi2taoa

Note: Using demo data for benchmarks until real data is available.
"""
from ..benchmarks import BenchmarkEntry, DatasetBenchmarks

BENCHMARKS = DatasetBenchmarks(
    dataset_name="csi2taoa",
    display_name="CSI2TAoA",
    default_metric="mean_error",
    entries=[
        BenchmarkEntry(
            method="ResNet18",
            mean_error=4.25,
            source="IndoorLoc Library (Demo Data, 10 epochs)",
            year=2024,
            notes="Demo data; replace with real data results",
        ),
        BenchmarkEntry(
            method="KNN (k=3)",
            mean_error=4.22,
            source="IndoorLoc Library Baseline (Demo Data)",
            year=2024,
            notes="Demo data baseline",
        ),
    ],
)
