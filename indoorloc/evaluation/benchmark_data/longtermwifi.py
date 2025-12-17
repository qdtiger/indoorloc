"""
Long-Term WiFi Dataset Benchmarks

Benchmark results from the UJI Library Long-Term WiFi dataset.

Dataset: Long-Term WiFi (UJI Library)
- 25 months of data collection (2016-2018)
- 2 floors (3rd and 5th), 308 m² library area
- 620 APs detected, 104,154 total samples

Original Paper:
    Mendoza-Silva, G.M., Richter, P., Torres-Sospedra, J., Lohan, E.S., Huerta, J. (2018).
    Long-Term WiFi Fingerprinting Dataset for Research on Robust Indoor Positioning.
    Data, 3(1), 3.
    DOI: 10.3390/data3010003

Note: Results vary based on month selection. The dataset is designed to study
temporal WiFi signal variations - training on one month and testing on others
can evaluate model robustness to long-term changes.
"""
from ..benchmarks import BenchmarkEntry, DatasetBenchmarks

BENCHMARKS = DatasetBenchmarks(
    dataset_name="longtermwifi",
    display_name="Long-Term WiFi (UJI Library)",
    default_metric="mean_error",
    entries=[
        # ============================================================
        # Deep Learning Methods (Published Results)
        # ============================================================
        BenchmarkEntry(
            method="KD-3D CNN (Student)",
            mean_error=0.84,
            source="Rizwan et al., Knowledge Distillation-Driven 3D CNNs for Indoor Localization, Monash University, 2025",
            year=2025,
            is_sota=True,
            notes="Knowledge distillation from teacher model; APE improved from 1.07m to 0.84m",
        ),
        BenchmarkEntry(
            method="3D-Separable CNN",
            mean_error=1.04,
            source="Rizwan et al., Enhancing Indoor Localization With Temporally-Aware Separable Group Shuffled CNNs, IEEE Access, 2025",
            year=2025,
            notes="DOI: 10.1109/ACCESS.2025.3535948",
        ),
        BenchmarkEntry(
            method="2D-CNN + KD",
            mean_error=1.33,
            source="Rizwan et al., Precise indoor localization using lightweight 2D-CNN with adaptive temperature guided KD, Scientific Reports, 2025",
            year=2025,
            notes="2D Average Positioning Error",
        ),

        # ============================================================
        # Traditional ML Baselines
        # ============================================================
        BenchmarkEntry(
            method="WKNN (k=5)",
            mean_error=2.29,
            source="IndoorLoc Library Baseline",
            year=2024,
            notes="Train: month 1-25, Test: month 1-25, floor 3&5; 23,034 train / 81,120 test",
        ),
        BenchmarkEntry(
            method="KNN (k=3)",
            mean_error=2.37,
            source="IndoorLoc Library Baseline",
            year=2024,
            notes="Train: month 1-25, Test: month 1-25, floor 3&5",
        ),
    ],
)
