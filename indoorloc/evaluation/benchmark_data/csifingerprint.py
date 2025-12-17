"""
CSI Fingerprint Dataset Benchmarks

Benchmark results from the CSI-dataset-for-indoor-localization.

Dataset: CSI Fingerprint (2021)
- Lab: 13.5x11m, NLOS environment, 317 sampling points
- Meeting Room: 7x10m, LOS environment, 176 sampling points
- Intel 5300 NIC, 3 antennas x 30 subcarriers = 90 features

Associated Papers:
    [1] Zhu, X., Qu, W., Zhou, X., Zhao, L., Ning, Z., & Qiu, T. (2022).
        Intelligent Fingerprint-Based Localization Scheme Using CSI Images
        for Internet of Things.
        IEEE Transactions on Network Science and Engineering, 9(4), 2378-2391.
        DOI: 10.1109/TNSE.2022.3163358

    [2] Zhu, X., Qiu, T., Qu, W., Zhou, X., Atiquzzaman, M., & Wu, D. (2023).
        BLS-Location: A Wireless Fingerprint Localization Algorithm Based
        on Broad Learning.
        IEEE Transactions on Mobile Computing, 22(1), 115-128.
        DOI: 10.1109/TMC.2021.3073005

Dataset URL: https://github.com/qiang5love1314/CSI-dataset-for-indoor-localization

Note: Results vary based on area (Lab vs Meeting Room) and LOS/NLOS conditions.
"""
from ..benchmarks import BenchmarkEntry, DatasetBenchmarks

BENCHMARKS = DatasetBenchmarks(
    dataset_name="csifingerprint",
    display_name="CSI Fingerprint",
    default_metric="mean_error",
    entries=[
        # ============================================================
        # Published Paper Results (Zhu et al.)
        # TODO: Fill in actual numbers from papers
        # ============================================================
        BenchmarkEntry(
            method="BLS-Location",
            mean_error=None,  # TODO: from IEEE TMC 2023
            source="Zhu et al., BLS-Location: A Wireless Fingerprint Localization Algorithm Based on Broad Learning, IEEE TMC, 2023",
            year=2023,
            notes="DOI: 10.1109/TMC.2021.3073005; TODO: fill in results from paper",
        ),
        BenchmarkEntry(
            method="CSI Image + CNN",
            mean_error=None,  # TODO: from IEEE TNSE 2022
            source="Zhu et al., Intelligent Fingerprint-Based Localization Scheme Using CSI Images, IEEE TNSE, 2022",
            year=2022,
            notes="DOI: 10.1109/TNSE.2022.3163358; TODO: fill in results from paper",
        ),

        # ============================================================
        # Baseline Results (for reference, not from papers)
        # These should be replaced with published results when available
        # ============================================================
        # BenchmarkEntry(
        #     method="KNN (k=5)",
        #     mean_error=4.56,  # Lab+Meeting combined
        #     source="IndoorLoc Library Baseline",
        #     year=2024,
        #     notes="Lab: 4.71m, Meeting: 2.82m; Local validation only",
        # ),
    ],
)
