"""
TUJI1 Dataset Benchmarks

Published results from the TUJI1 dataset paper.

Dataset: TUJI1 (2024)
- Single floor, fine-grained grid (~45cm spacing)
- 310 APs, 8899 samples (6752 train, 2147 test)
- 5 devices: Samsung S20, S7, POCO, Tab S7, A12

Original Paper:
    Klus, L., Klus, R., Lohan, E.S., Nurmi, J., Granell, C., Valkama, M.,
    Talvitie, J., Casteleyn, S., & Torres-Sospedra, J. (2024).
    TUJI1 Dataset: Multi-device dataset for indoor localization with
    high measurement density. Data in Brief, 110356.
    DOI: 10.1016/j.dib.2024.110356

Note: Results may vary based on device selection and parameter tuning.
"""
from ..benchmarks import BenchmarkEntry, DatasetBenchmarks

BENCHMARKS = DatasetBenchmarks(
    dataset_name="tuji1",
    display_name="TUJI1",
    default_metric="mean_error",
    entries=[
        # ============================================================
        # Original Paper Benchmarks (Klus et al., 2024)
        # ============================================================
        BenchmarkEntry(
            method="kNN (Best Tuned)",
            mean_error=1.8,
            source="Klus et al., TUJI1 Dataset: Multi-device dataset for indoor localization, Data in Brief, 2024",
            year=2024,
            is_sota=True,
            notes="DOI: 10.1016/j.dib.2024.110356; Best coefficient benchmark with tuned parameters",
        ),
        BenchmarkEntry(
            method="1NN (Baseline)",
            mean_error=2.5,
            source="Klus et al., Data in Brief, 2024",
            year=2024,
            notes="DOI: 10.1016/j.dib.2024.110356; Untuned baseline benchmark",
        ),

        # ============================================================
        # Traditional ML Baselines
        # ============================================================
        BenchmarkEntry(
            method="WKNN (k=5)",
            mean_error=2.73,
            source="IndoorLoc Library Baseline",
            year=2024,
            notes="Distance-weighted KNN baseline from IndoorLoc testing",
        ),
        BenchmarkEntry(
            method="KNN (k=3)",
            mean_error=2.9,
            source="IndoorLoc Library Baseline",
            year=2024,
            notes="Simple KNN baseline",
        ),
    ],
)
