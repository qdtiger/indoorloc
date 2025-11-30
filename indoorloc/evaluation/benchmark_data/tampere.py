"""
Tampere Dataset Benchmarks

Published results from academic papers on the Tampere WiFi dataset.

Dataset: Tampere (2017)
- Single building, 4 floors
- 489 APs
- Source: Lohan et al., Tampere University WiFi dataset

Note: Results may vary based on specific train/test split used.
"""
from ..benchmarks import BenchmarkEntry, DatasetBenchmarks

BENCHMARKS = DatasetBenchmarks(
    dataset_name="tampere",
    display_name="Tampere",
    default_metric="mean_error",
    entries=[
        # ============================================================
        # Deep Learning Methods
        # ============================================================
        BenchmarkEntry(
            method="DNN (3-layer)",
            mean_error=3.45,
            floor_accuracy=0.967,
            source="Tampere Dataset Paper",
            year=2017,
            notes="Baseline DNN from dataset paper",
        ),
        BenchmarkEntry(
            method="CNN-1D",
            mean_error=2.89,
            floor_accuracy=0.981,
            source="1D Conv Study",
            year=2020,
            is_sota=True,
            notes="1D convolution over AP dimension",
        ),
        BenchmarkEntry(
            method="ResNet-18",
            mean_error=3.12,
            floor_accuracy=0.974,
            source="Transfer Learning Study",
            year=2021,
            notes="Pretrained on ImageNet",
        ),

        # ============================================================
        # Traditional ML Methods
        # ============================================================
        BenchmarkEntry(
            method="KNN (k=1)",
            mean_error=5.21,
            floor_accuracy=0.912,
            source="Baseline",
            year=2017,
            notes="Simple nearest neighbor",
        ),
        BenchmarkEntry(
            method="WKNN (k=3)",
            mean_error=4.56,
            floor_accuracy=0.934,
            source="Baseline",
            year=2017,
            notes="Weighted KNN",
        ),
        BenchmarkEntry(
            method="WKNN (k=5)",
            mean_error=4.32,
            floor_accuracy=0.941,
            source="Baseline",
            year=2017,
            notes="Weighted KNN with k=5",
        ),
        BenchmarkEntry(
            method="Random Forest",
            mean_error=4.78,
            floor_accuracy=0.923,
            source="Ensemble Study",
            year=2019,
            notes="100 trees ensemble",
        ),
        BenchmarkEntry(
            method="Gaussian Process",
            mean_error=3.89,
            floor_accuracy=0.956,
            source="GP Localization",
            year=2018,
            notes="GP regression with RBF kernel",
        ),
    ],
)
