"""
iBeacon RSSI (UJI BLE RSS) Dataset Benchmarks

Benchmark results from the UJI BLE RSS Database.

Dataset: UJI BLE RSS Database (2019)
- Library zone (lib): ~2700 fingerprints
- Geotec zone (geo): ~2000 fingerprints
- Accent Systems IBKS105 beacons
- 3 smartphones: BQ Aquaris X5+, Samsung Galaxy S6, Samsung Galaxy A5 2017

Original Paper:
    Mendoza-Silva, G.M.; Matey-Sanz, M.; Torres-Sospedra, J.; Huerta, J.
    BLE RSS Measurements Dataset for Research on Accurate Indoor Positioning.
    Data 2019, 4, 12.
    DOI: 10.3390/data4010012

Dataset URL: https://zenodo.org/record/1618692

Note: Results vary based on zone selection (lib vs geo) and device used.
"""
from ..benchmarks import BenchmarkEntry, DatasetBenchmarks

BENCHMARKS = DatasetBenchmarks(
    dataset_name="ibeaconrssi",
    display_name="iBeacon RSSI (UJI BLE)",
    default_metric="mean_error",
    entries=[
        # ============================================================
        # Published Paper Results
        # TODO: Fill in actual numbers from papers
        # ============================================================
        BenchmarkEntry(
            method="KNN",
            mean_error=None,  # TODO: from MDPI Data 2019 paper
            source="Mendoza-Silva et al., BLE RSS Measurements Dataset for Research on Accurate Indoor Positioning, Data 2019",
            year=2019,
            notes="DOI: 10.3390/data4010012; TODO: fill in results from paper",
        ),

        # ============================================================
        # Related Work
        # ============================================================
        BenchmarkEntry(
            method="CNN-based",
            mean_error=None,  # TODO
            source="Deep Learning-Based Indoor Localization Using Multi-View BLE Signal, MDPI Sensors, 2022",
            year=2022,
            notes="DOI: 10.3390/s22072759; TODO: fill in results",
        ),
    ],
)
