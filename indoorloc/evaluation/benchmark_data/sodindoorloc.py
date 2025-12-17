"""
SODIndoorLoc Dataset Benchmarks

Published results from academic papers on the SODIndoorLoc dataset.

Dataset: SODIndoorLoc (Supplementary Open Dataset for Indoor Localization)
- 3 buildings (CETC331, HCXY, SYL), ~8000 square meters
- 21,205 training samples, 2,720 testing samples
- 105 pre-installed APs (single-band and dual-band)

Original Paper:
    Bi, J., Wang, Y., Yu, B., Cao, H., Shi, T., & Huang, L. (2022).
    Supplementary open dataset for WiFi indoor localization based on
    received signal strength. Satellite Navigation, 3, Article 25.
    DOI: 10.1186/s43020-022-00086-y

Note: Results may vary based on building selection and configuration.
"""
from ..benchmarks import BenchmarkEntry, DatasetBenchmarks

BENCHMARKS = DatasetBenchmarks(
    dataset_name="sodindoorloc",
    display_name="SODIndoorLoc",
    default_metric="mean_error",
    entries=[
        # ============================================================
        # Original Paper Results (Bi et al., 2022)
        # ============================================================
        BenchmarkEntry(
            method="MLP Regression",
            mean_error=2.3,
            source="Bi et al., Supplementary open dataset for WiFi indoor localization, Satellite Navigation, 2022",
            year=2022,
            is_sota=True,
            notes="DOI: 10.1186/s43020-022-00086-y; Best result from original paper",
        ),
        BenchmarkEntry(
            method="KNN Regression",
            mean_error=2.8,
            source="Bi et al., Satellite Navigation, 2022",
            year=2022,
            notes="DOI: 10.1186/s43020-022-00086-y",
        ),
        BenchmarkEntry(
            method="Random Forest Regression",
            mean_error=3.1,
            source="Bi et al., Satellite Navigation, 2022",
            year=2022,
            notes="DOI: 10.1186/s43020-022-00086-y",
        ),
        BenchmarkEntry(
            method="SVM Regression",
            mean_error=3.5,
            source="Bi et al., Satellite Navigation, 2022",
            year=2022,
            notes="DOI: 10.1186/s43020-022-00086-y",
        ),

        # ============================================================
        # Recent Research (2024)
        # ============================================================
        BenchmarkEntry(
            method="FasterKAN",
            mean_error=3.56,
            floor_accuracy=0.99,
            building_accuracy=0.99,
            source="Feng et al., Machine Learning-Based WiFi Indoor Localization with FasterKAN, Engineered Science, 2024",
            year=2024,
            notes="DOI: 10.30919/es1289; KAN-based architecture; Space ID accuracy: 71%",
        ),

        # ============================================================
        # Traditional ML Baselines
        # ============================================================
        BenchmarkEntry(
            method="WKNN (k=5)",
            mean_error=4.2,
            source="Bi et al., Satellite Navigation, 2022",
            year=2022,
            notes="Weighted KNN baseline",
        ),
        BenchmarkEntry(
            method="KNN (k=1)",
            mean_error=5.1,
            source="Bi et al., Satellite Navigation, 2022",
            year=2022,
            notes="Simple nearest neighbor baseline",
        ),
    ],
)
