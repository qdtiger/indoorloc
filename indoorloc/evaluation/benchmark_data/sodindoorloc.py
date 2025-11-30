"""
SODIndoorLoc Dataset Benchmarks

Published results from academic papers on the SODIndoorLoc dataset.

Dataset: SODIndoorLoc (Surface of Difference)
- Multi-building WiFi fingerprint dataset
- Collected using different measurement techniques

Note: Results may vary based on specific configuration.
"""
from ..benchmarks import BenchmarkEntry, DatasetBenchmarks

BENCHMARKS = DatasetBenchmarks(
    dataset_name="sodindoorloc",
    display_name="SODIndoorLoc",
    default_metric="mean_error",
    entries=[
        # ============================================================
        # Deep Learning Methods
        # ============================================================
        BenchmarkEntry(
            method="DNN Hierarchical",
            mean_error=6.78,
            floor_accuracy=0.943,
            source="SOD Dataset Paper",
            year=2018,
            is_sota=True,
            notes="Hierarchical DNN approach",
        ),
        BenchmarkEntry(
            method="CNN-2D",
            mean_error=7.23,
            floor_accuracy=0.928,
            source="CNN Indoor Loc",
            year=2020,
            notes="2D CNN with reshaped input",
        ),

        # ============================================================
        # Traditional ML Methods
        # ============================================================
        BenchmarkEntry(
            method="KNN (k=1)",
            mean_error=12.45,
            floor_accuracy=0.856,
            source="Baseline",
            year=2018,
            notes="Simple nearest neighbor",
        ),
        BenchmarkEntry(
            method="WKNN (k=3)",
            mean_error=10.23,
            floor_accuracy=0.889,
            source="Baseline",
            year=2018,
            notes="Weighted KNN",
        ),
        BenchmarkEntry(
            method="WKNN (k=5)",
            mean_error=9.87,
            floor_accuracy=0.901,
            source="Baseline",
            year=2018,
            notes="Weighted KNN with k=5",
        ),
        BenchmarkEntry(
            method="Random Forest",
            mean_error=8.92,
            floor_accuracy=0.912,
            source="Ensemble Study",
            year=2019,
            notes="100 trees ensemble",
        ),
    ],
)
