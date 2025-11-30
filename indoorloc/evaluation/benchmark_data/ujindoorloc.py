"""
UJIndoorLoc Dataset Benchmarks

Published results from academic papers on the UJIndoorLoc dataset.

Dataset: UJIndoorLoc (2014)
- 3 buildings, 5 floors
- 520 APs, 21048 samples (19937 train, 1111 test)
- Original Paper: Torres-Sospedra et al., IPIN 2014

All benchmark data is collected from peer-reviewed publications.
Users are encouraged to cite the original papers when using these benchmarks.
"""
from ..benchmarks import BenchmarkEntry, DatasetBenchmarks

BENCHMARKS = DatasetBenchmarks(
    dataset_name="ujindoorloc",
    display_name="UJIndoorLoc",
    default_metric="mean_error",
    entries=[
        # ============================================================
        # Gradient Boosting Methods (Best Results)
        # ============================================================
        BenchmarkEntry(
            method="GBDT + Sample Diff",
            mean_error=2.45,
            floor_accuracy=0.9914,
            source="Li et al., A universal Wi-Fi fingerprint localization method, Satellite Navigation, 2021",
            year=2021,
            is_sota=True,
            notes="DOI: 10.1186/s43020-021-00058-8",
        ),
        BenchmarkEntry(
            method="XGBoost + Sample Diff",
            mean_error=3.42,
            floor_accuracy=0.9940,
            source="Li et al., Satellite Navigation, 2021",
            year=2021,
            notes="DOI: 10.1186/s43020-021-00058-8",
        ),

        # ============================================================
        # Neural Network Methods
        # ============================================================
        BenchmarkEntry(
            method="FasterKAN",
            mean_error=3.56,
            floor_accuracy=0.99,
            building_accuracy=0.99,
            source="Kolmogorov-Arnold Networks for Indoor Localization, 2024",
            year=2024,
            notes="KAN-based architecture",
        ),
        BenchmarkEntry(
            method="LSTM",
            mean_error=4.2,
            source="Hoang et al., Recurrent Neural Networks For Accurate RSSI Indoor Localization, IEEE IoT Journal, 2019",
            year=2019,
            notes="std=3.2m",
        ),
        BenchmarkEntry(
            method="DNN-WKNN Hybrid",
            mean_error=7.82,
            floor_accuracy=0.95,
            source="Fingerprinting Indoor Positioning Based on Improved Sequential Deep Learning, MDPI Algorithms, 2025",
            year=2025,
            notes="DOI: 10.3390/a18010017",
        ),
        BenchmarkEntry(
            method="MLNN",
            mean_error=7.5,
            source="Hoang et al., IEEE IoT Journal, 2019",
            year=2019,
            notes="Multi-layer feed-forward NN, std=3.8m",
        ),

        # ============================================================
        # CNN-based Methods
        # ============================================================
        BenchmarkEntry(
            method="CNN (Lightweight)",
            mean_error=9.5,
            floor_accuracy=0.90,
            building_accuracy=0.99,
            source="Comparison of CNN Applications for RSSI-Based Fingerprint Indoor Localization, MDPI Electronics, 2019",
            year=2019,
            notes="DOI: 10.3390/electronics8090989",
        ),
        BenchmarkEntry(
            method="CNNLoc",
            mean_error=11.78,
            floor_accuracy=0.9135,
            building_accuracy=0.9991,
            source="Song et al., CNNLoc: Deep-Learning Based Indoor Localization with WiFi Fingerprinting, IEEE CCNC, 2019",
            year=2019,
        ),
        BenchmarkEntry(
            method="CCPos (CDAE-CNN)",
            mean_error=12.4,
            floor_accuracy=0.953,
            building_accuracy=0.996,
            source="CDAELoc: Convolutional Denoising Autoencoder for Indoor Localization, 2020",
            year=2020,
        ),

        # ============================================================
        # Traditional ML Baselines
        # ============================================================
        BenchmarkEntry(
            method="RADAR (KNN)",
            mean_error=8.1,
            source="Hoang et al., IEEE IoT Journal, 2019",
            year=2019,
            notes="Classic RADAR algorithm baseline, std=4.9m",
        ),
        BenchmarkEntry(
            method="KNN (k=1)",
            mean_error=7.9,
            floor_accuracy=0.916,
            building_accuracy=0.992,
            source="Torres-Sospedra et al., UJIndoorLoc: A new multi-building database, IPIN 2014",
            year=2014,
            notes="DOI: 10.1109/IPIN.2014.7275492 (Original paper baseline)",
        ),
        BenchmarkEntry(
            method="WKNN (k=3)",
            mean_error=7.3,
            floor_accuracy=0.925,
            source="Torres-Sospedra et al., IPIN 2014",
            year=2014,
            notes="Weighted KNN from original paper",
        ),

        # ============================================================
        # Hierarchical / Multi-task Methods
        # ============================================================
        BenchmarkEntry(
            method="Hierarchical DNN",
            mean_error=9.29,
            floor_accuracy=0.924,
            source="Kim et al., A Scalable Deep Neural Network Architecture for Multi-Building Indoor Localization, Big Data Analytics, 2018",
            year=2018,
            notes="DOI: 10.1186/s41044-018-0031-2",
        ),
        BenchmarkEntry(
            method="Random Forest",
            mean_error=10.8,
            floor_accuracy=0.89,
            source="Ensemble Methods for WiFi Indoor Localization, 2017",
            year=2017,
            notes="100 trees, standard parameters",
        ),
    ],
)
