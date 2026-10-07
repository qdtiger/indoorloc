"""TUJI1: multi-device WiFi RSSI fingerprints on a dense grid of one office floor (Klus et al., 2024)."""
from __future__ import annotations

import csv

import numpy as np

from ..core import SampleTable
from ._base import Dataset


class TUJI1(Dataset):
    """TUJI1 (Klus et al., Data in Brief 2024): five devices, one floor, official train/test.

    Collected on the top (fifth) floor of the Espaitec 2 building, Universitat Jaume I,
    Castellón, Spain, with five Android devices (Galaxy S20, Galaxy S7, POCO X3, Galaxy Tab
    S7, Galaxy A12) at the corners and at the centres of the floor's 60 x 60 cm tiles (two
    interleaved grids, 0.42 m between neighbours). 8899 scans of 310 MAC addresses: 6752
    training and 2147 test scans; the test positions are a random half of the positions of
    the test campaign (split by the authors' ``CreateDataset.m`` with ``rng(2)``).

    ``X`` is (N, 310) float32 RSSI in dBm; the files' "missing" value +100 becomes NaN. The RSS
    files have no header: columns are identified by position (``WAP001`` .. ``WAP310``, AP1 ..
    AP310 in the paper). ``pos`` is (x, y) in metres in the site's local frame, unchanged. The
    coordinates files have six columns: x, y, z, floor, building, device label. Every scan is on
    the same floor, and the z, floor and building columns are constant zeros (checked by the
    loader), so ``floor`` and ``building`` are None and 2-D errors equal the paper's 3-D errors.
    Group ``device``: the device name from ``Device_labels.csv`` (read by column name).

    References
    ----------
    Klus, L., Klus, R., Lohan, E.S., Nurmi, J., Granell, C., Valkama, M., Talvitie, J.,
    Casteleyn, S., Torres-Sospedra, J., "TUJI1 Dataset: Multi-device dataset for indoor
    localization with high measurement density", Data in Brief 54, 110356, 2024.
    https://doi.org/10.1016/j.dib.2024.110356. Data: https://doi.org/10.5281/zenodo.7641701
    """

    name = "tuji1"
    urls = ("https://zenodo.org/api/records/7641701/files/DATASET.zip/content",
            "https://zenodo.org/records/7641701/files/DATASET.zip?download=1")
    files = {  # sha256 of the DATASET/*.csv members of DATASET.zip (Zenodo md5 27e80f4a98387f2417d2e0760af2671f)
        "train": (("RSS_training.csv", "dbdcb88699cb767a65fe17b859042b0b3b2be1095674f977cafb6c981aabd6d5"),
                  ("Coordinates_training.csv", "6bee8b88c923eaff2e6af6744f5bc69e689629bd67bbc23b4ce6e1d67750fb8e"),
                  ("Device_labels.csv", "81978cef0e36c5a19525a652d0deab34757086035dd23c619f2f4fe897d8be30")),
        "test": (("RSS_testing.csv", "a81a514d76d6aae796bf78a12582dd18e760333773395402fe0ca0daa6986260"),
                 ("Coordinates_testing.csv", "67e63a53e57fba461f7db2af51218bf4716f942b3ffdbe0de2bdfd37697724f3"),
                 ("Device_labels.csv", "81978cef0e36c5a19525a652d0deab34757086035dd23c619f2f4fe897d8be30")),
    }
    # No 'validation' alias: the source has no validation file (carve one from train, e.g. kfold on groups).
    meta = {
        "modality": "wifi_rssi",
        "units": "dBm",
        "raw_missing_value": 100,
        "crs": "local",
        "pos_names": ("x", "y"),
        "pos_units": "m",
        "floors": (),  # single floor, unlabelled
        "license": "CC BY 4.0",
        "doi": "10.5281/zenodo.7641701",
        "citation": "Klus et al., TUJI1 Dataset: Multi-device dataset for indoor localization with high "
                    "measurement density, Data in Brief 54:110356, 2024, doi:10.1016/j.dib.2024.110356",
        "url": "https://zenodo.org/records/7641701",
    }
    n_aps = 310

    def _parse(self, paths, split):
        rss_path, crd_path, labels_path = paths
        rssi = np.loadtxt(rss_path, delimiter=",", ndmin=2).astype(np.float32)
        crd = np.loadtxt(crd_path, delimiter=",", ndmin=2)
        if rssi.shape[1] != self.n_aps or crd.shape[1] != 6 or len(crd) != len(rssi):
            raise ValueError(f"{split}: expected {self.n_aps} RSS columns, 6 coordinate columns and equal row "
                             f"counts; got RSS {rssi.shape}, coordinates {crd.shape}")
        if np.any(crd[:, 2:5] != 0):
            raise ValueError(f"{crd_path.name}: the z/floor/building columns hold values; they were declared unused")
        with open(labels_path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if not {"Label", "Device"} <= set(reader.fieldnames or ()):
                raise ValueError(f"{labels_path.name}: expected columns 'Device' and 'Label', got {reader.fieldnames}")
            names = {int(row["Label"]): row["Device"].strip() for row in reader}
        codes = crd[:, 5].astype(np.int64)
        if not set(np.unique(codes)) <= set(names):
            raise ValueError(f"{crd_path.name}: device labels {sorted(set(np.unique(codes)) - set(names))} "
                             f"are not in {labels_path.name}")
        rssi[rssi == self.meta["raw_missing_value"]] = np.nan
        if np.any(rssi >= 0):
            raise ValueError(f"{rss_path.name}: non-negative RSSI other than the +100 'missing' marker")
        device = np.array([names[c] for c in codes])
        ids = np.array([f"{split}-{i:05d}" for i in range(len(rssi))])
        return SampleTable(rssi, crd[:, :2], None, None, {"device": device}, ids,
                           meta={"feature_names": tuple(f"WAP{j:03d}" for j in range(1, self.n_aps + 1)),
                                 "device_names": tuple(names[k] for k in sorted(names))})
