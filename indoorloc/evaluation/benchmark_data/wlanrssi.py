"""
Wireless Indoor Localization Dataset (UCI) Benchmarks

Room classification benchmark results from the UCI Wireless Indoor
Localization dataset.

Dataset: UCI Wireless Indoor Localization (ID 422)
- 7 WiFi access points
- 4 rooms (500 samples each)
- 2,000 total samples

Original Dataset:
    Bhatt, R. (2017). Wireless Indoor Localization. UCI Machine Learning
    Repository. DOI: 10.24432/C51880

Note: This is a room classification task (4 classes), not coordinate regression.
Metrics are classification accuracy (%).
"""
from ..benchmarks import BenchmarkEntry, DatasetBenchmarks

BENCHMARKS = DatasetBenchmarks(
    dataset_name="wlanrssi",
    display_name="Wireless Indoor Localization (UCI)",
    default_metric="accuracy",
    entries=[
        # ============================================================
        # Published Results
        # ============================================================
        BenchmarkEntry(
            method="Logistic Regression",
            accuracy=97.2,
            source="Subhash Karthik, WiFi Indoor Localization Tutorial",
            year=2020,
            notes="Multi-class logistic regression",
        ),
        # ============================================================
        # IndoorLoc Library Baselines
        # ============================================================
        BenchmarkEntry(
            method="KNN (k=1)",
            accuracy=98.3,
            source="IndoorLoc Library Baseline",
            year=2024,
            notes="1400 train / 600 test (70/30 split)",
        ),
        BenchmarkEntry(
            method="KNN (k=3)",
            accuracy=98.2,
            source="IndoorLoc Library Baseline",
            year=2024,
        ),
        BenchmarkEntry(
            method="WKNN (k=5)",
            accuracy=98.3,
            source="IndoorLoc Library Baseline",
            year=2024,
            is_sota=True,
        ),
    ],
)
