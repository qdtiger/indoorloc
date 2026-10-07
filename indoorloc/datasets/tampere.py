"""Tampere: crowdsourced WiFi RSSI fingerprints from a five-level university building (Lohan et al., 2017)."""
from __future__ import annotations

import numpy as np

from ..core import SampleTable
from ._base import Dataset

_FLOOR_HEIGHT = 3.7  # metres per floor; the dataset's own software uses floor = round(z / 3.7)


class Tampere(Dataset):
    """Tampere crowdsourced WiFi database (Lohan et al., Data 2017), official train/test.

    4648 fingerprints (697 training, 3951 test) collected with 21 Android devices by
    crowdsourcing in a university building in Tampere, Finland, between February and August
    2017; 992 MAC addresses. Training and test scans were mostly taken at different positions
    (26 of the 3843 distinct test positions also occur in training). The package README
    swaps its "training"/"test" headings; the ``Training_*`` files (697 rows, 15 %) are the
    training set, as in the benchmark software.

    ``X`` is (N, 992) float32 RSSI in dBm; the files' "not heard" value +100 becomes NaN. The
    RSS files have no header: columns are identified by position (``WAP001`` .. ``WAP992``,
    the same order in both splits). ``pos`` is the (x, y, z) triple of the coordinates file in
    metres, in the building's local frame, unchanged; z is the height of the floor. ``floor``
    = round(z / 3.7) (0-4), the convention of the dataset's benchmark software, and the loader
    checks that every z is a whole number of floors. The benchmark software reports the 3-D
    error (what ``evaluate`` computes on these positions), the 2-D error on correctly
    predicted floors, and the floor hit rate. Groups: ``device`` (the Android model string of
    the device file) and ``time`` (unix seconds of the date file's wall-clock time, read as
    UTC because the file states no time zone; the rows are not in chronological order).

    References
    ----------
    Lohan, E.S., Torres-Sospedra, J., Leppäkoski, H., Richter, P., Peng, Z., Huerta, J.,
    "Wi-Fi Crowdsourced Fingerprinting Dataset for Indoor Positioning", Data 2(4), 32, 2017.
    https://doi.org/10.3390/data2040032. Data: https://doi.org/10.5281/zenodo.889798
    (version 2, identical data files: https://doi.org/10.5281/zenodo.1001662).
    """

    name = "tampere"
    urls = ("https://zenodo.org/api/records/1001662/files/DISTRIBUTED_OPENSOURCE_version2.zip/content",
            "https://zenodo.org/records/1001662/files/DISTRIBUTED_OPENSOURCE_version2.zip?download=1",
            "https://zenodo.org/api/records/889798/files/DISTRIBUTED_OPENSOURCE.zip/content")
    files = {  # sha256 of the FINGERPRINTING_DB/*.csv members (identical in both Zenodo versions)
        "train": (
            ("Training_rss_21Aug17.csv", "3bafbedc75fa14d4e96a351ca053884eec44de814b95daf7f10297d3b4148f0c"),
            ("Training_coordinates_21Aug17.csv", "003f6f9ec46f48b49099e0adef329a43a80212eeae2a698c820b43d5eaaa4479"),
            ("Training_device_21Aug17.csv", "4c0d9e4e75ab041d2a53230d24199b360fa37360522572fa183a92562a033fb5"),
            ("Training_date_21Aug17.csv", "424c1ffe313fe45043dbc21030c34886b9ccfbe78029355876b6328ec0b3eedc")),
        "test": (
            ("Test_rss_21Aug17.csv", "1bf4993edb835e635b1a8500cefa0f7b971b48af3ff5a9a0019720016ee2bcf1"),
            ("Test_coordinates_21Aug17.csv", "4254f2cf777fc278fa16a9f3715c622bff15c0c48db84db44e6d97373fb23001"),
            ("Test_device_21Aug17.csv", "f06983b3269c35a7b3af49ae2cbf7f8d8b49193ff62ad3d0ec66a1a64e393047"),
            ("Test_date_21Aug17.csv", "38900ada43241528f8c9457ed18466b2bb5763c4321e7a1f4a127a3a7c4c21c7")),
    }
    # No 'validation' alias: the source has no validation file (carve one from train, e.g. kfold on groups).
    meta = {
        "modality": "wifi_rssi",
        "units": "dBm",
        "raw_missing_value": 100,
        "crs": "local",
        "pos_names": ("x", "y", "z"),
        "pos_units": "m",
        "floors": (0, 1, 2, 3, 4),
        "floor_height": _FLOOR_HEIGHT,
        "license": "CC BY 4.0 (data, FINGERPRINTING_DB/README.txt); MIT (software)",
        "doi": "10.5281/zenodo.889798",
        "citation": "Lohan et al., Wi-Fi Crowdsourced Fingerprinting Dataset for Indoor Positioning, "
                    "Data 2(4):32, 2017, doi:10.3390/data2040032",
        "url": "https://zenodo.org/records/889798",
    }
    n_aps = 992

    def _parse(self, paths, split):
        rss_path, crd_path, device_path, date_path = paths
        rssi = np.loadtxt(rss_path, delimiter=",", ndmin=2).astype(np.float32)
        crd = np.loadtxt(crd_path, delimiter=",", ndmin=2)
        device = np.array(device_path.read_text(encoding="utf-8").splitlines())
        date = np.array(date_path.read_text(encoding="ascii").splitlines(), dtype="datetime64[s]")
        n = len(rssi)
        if rssi.shape[1] != self.n_aps or crd.shape[1] != 3 or not len(crd) == len(device) == len(date) == n:
            raise ValueError(f"{split}: expected {self.n_aps} RSS columns, 3 coordinate columns and equal row "
                             f"counts; got RSS {rssi.shape}, coordinates {crd.shape}, {len(device)} devices, "
                             f"{len(date)} dates")
        rssi[rssi == self.meta["raw_missing_value"]] = np.nan
        if np.any(rssi > 0):
            raise ValueError(f"{rss_path.name}: positive RSSI other than the +100 'not heard' marker")
        level = crd[:, 2] / _FLOOR_HEIGHT
        floor = np.round(level)
        if np.any(np.abs(level - floor) > 1e-6):
            raise ValueError(f"{crd_path.name}: heights that are not a whole number of {_FLOOR_HEIGHT} m floors")
        ids = np.array([f"{split}-{i:05d}" for i in range(n)])
        return SampleTable(rssi, crd, floor.astype(np.int64), None,
                           {"device": device, "time": date.astype(np.int64)}, ids,
                           meta={"feature_names": tuple(f"WAP{j:03d}" for j in range(1, self.n_aps + 1))})
