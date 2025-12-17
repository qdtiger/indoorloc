"""
BLE Indoor (BBIL) Dataset Benchmarks

Benchmark results from the BBIL (BLE Beacon Indoor Localization) dataset.

Dataset: BBIL (2018-2019)
- Experiment 1: Office environment (11x25m)
- Experiment 2: Lab environment (11x7m)
- Raspberry Pi receivers + BLE beacons
- Multiple participants with smartphones

Original Paper:
    Kennedy, B., Taylor, G., and Spachos, P.
    BLE beacon based patient tracking in smart care facilities.
    IEEE International Conference on Pervasive Computing and Communications
    Workshops (PerCom Workshops), March 2018.

Dataset URL: https://github.com/co60ca/BBIL
Data DOI: 10.5683/SP2/UTZTFT

Note: Results vary based on experiment selection (office vs lab).
"""
from ..benchmarks import BenchmarkEntry, DatasetBenchmarks

BENCHMARKS = DatasetBenchmarks(
    dataset_name="bleindoor",
    display_name="BLE Indoor (BBIL)",
    default_metric="mean_error",
    entries=[
        # ============================================================
        # Published Paper Results
        # TODO: Fill in actual numbers from papers
        # ============================================================
        BenchmarkEntry(
            method="Fingerprinting",
            mean_error=None,  # TODO: from PerCom 2018 paper
            source="Kennedy et al., BLE beacon based patient tracking in smart care facilities, IEEE PerCom Workshops, 2018",
            year=2018,
            notes="TODO: fill in results from paper",
        ),

        # ============================================================
        # Related Work on BLE Localization
        # ============================================================
        BenchmarkEntry(
            method="Smart Museum System",
            mean_error=None,  # TODO
            source="Spachos & Plataniotis, BLE Beacons for Indoor Positioning at an Interactive IoT-Based Smart Museum, arXiv:2001.07686, 2020",
            year=2020,
            notes="TODO: fill in results",
        ),
    ],
)
