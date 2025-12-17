"""
BLE RSSI UCI Dataset Benchmarks

Benchmark results from the BLE RSSI Indoor Localization dataset (UCI).

Dataset: BLE RSSI UCI (2017)
- 13 iBeacons (b3001-b3013)
- 1,420 labeled samples, 105 locations
- Waldo Library, Western Michigan University

Original Dataset:
    Mohammadi, M. & Al-Fuqaha, A. (2017). BLE RSSI Dataset for Indoor
    localization and Navigation. UCI Machine Learning Repository.
    DOI: 10.24432/C54G80

Note: This dataset uses room-level grid labels (e.g., "K04", "J06").
Results include both classification accuracy (%) and positioning error (m).
"""
from ..benchmarks import BenchmarkEntry, DatasetBenchmarks

BENCHMARKS = DatasetBenchmarks(
    dataset_name="ble_rssi_uci",
    display_name="BLE RSSI UCI",
    default_metric="mean_error",
    entries=[
        # ============================================================
        # Published Paper Results (Classification Accuracy)
        # ============================================================
        BenchmarkEntry(
            method="SVM",
            accuracy=88.67,
            source="PMC8000105, Optimized CNNs to Indoor Localization through BLE Sensors, 2021",
            year=2021,
            notes="11.33% error rate",
        ),
        BenchmarkEntry(
            method="Logistic Regression",
            accuracy=86.32,
            source="PMC8000105, Optimized CNNs to Indoor Localization through BLE Sensors, 2021",
            year=2021,
            notes="13.68% error rate",
        ),
        BenchmarkEntry(
            method="KNN",
            accuracy=85.0,
            source="MDPI Telecom 2023, Improved RSSI Indoor Localization",
            year=2023,
        ),
        BenchmarkEntry(
            method="Decision Tree",
            accuracy=83.95,
            source="PMC8000105, Optimized CNNs to Indoor Localization through BLE Sensors, 2021",
            year=2021,
            notes="16.05% error rate",
        ),
        # ============================================================
        # IndoorLoc Library Baselines (Positioning Error)
        # ============================================================
        BenchmarkEntry(
            method="WKNN (k=5)",
            mean_error=1.80,
            source="IndoorLoc Library Baseline",
            year=2024,
            notes="Grid coordinates (1m spacing); 993 train / 427 test",
        ),
        BenchmarkEntry(
            method="KNN (k=3)",
            mean_error=1.82,
            source="IndoorLoc Library Baseline",
            year=2024,
        ),
    ],
)
