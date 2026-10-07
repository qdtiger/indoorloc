"""WLANRSSI: room-level WiFi RSSI dataset of the UCI repository (Wireless Indoor Localization, id 422)."""
from __future__ import annotations

import numpy as np

from ..core import SampleTable
from ._base import Dataset


class WLANRSSI(Dataset):
    """UCI Wireless Indoor Localization (Bhatt, 2017): 2000 scans of 7 APs labelled with one of 4 rooms.

    Each row holds the signal strengths of seven WiFi access points observed on a smartphone
    and the room (1-4) where it was taken; 500 scans per room, no missing readings, no
    coordinates, no official split. The task is **room classification**
    (``meta["task"] = "room_classification"``).

    ``X`` is (2000, 7) float32 RSSI in dBm (``AP1`` .. ``AP7``, file order; the file has no
    header). The room label is ``groups["room"]`` (int64, 1-4). There are no metric
    coordinates, so ``pos`` is an empty ``(N, 0)`` float64 array (``meta["pos_names"] = ()``,
    ``meta["crs"] = None``) and ``floor``/``building`` are None: nothing is invented. Every
    layer still accepts the table; a position error over zero axes is identically 0 and means
    nothing, so score the room label instead. With the library's k-NN, whose neighbour vote
    works for any integer label, pass the room explicitly as that label::

        from indoorloc import create_model, load_dataset
        from indoorloc.evaluation import label_accuracy, random_split
        table = load_dataset("wlanrssi", split="all")
        train_idx, test_idx = random_split(len(table), 0.2, stratify=table.groups["room"], random_state=0)
        train, test = table[train_idx], table[test_idx]
        knn = create_model("wknn", k=5).fit(train.X, train.pos, floor=train.groups["room"])
        label_accuracy(test.groups["room"], knn.localize(test.X).floor)   # room accuracy in %

    or use any classifier on ``(table.X, table.groups["room"])``. Recommended protocol: a
    stratified split or stratified k-fold cross-validation over ``groups["room"]`` (the rows
    are grouped by room in the file, so never split by row order).

    References
    ----------
    Bhatt, R., "Wireless Indoor Localization", UCI Machine Learning Repository, 2017.
    https://doi.org/10.24432/C51880.
    Rohra, J.G., Perumal, B., Narayanan, S.J., Thakur, P., Bhatt, R.B., "User Localization in an
    Indoor Environment Using Fuzzy Hybrid of Particle Swarm Optimization & Gravitational Search
    Algorithm with Neural Networks", Proc. SocProS 2016, Advances in Intelligent Systems and
    Computing 546, Springer, 2017. https://doi.org/10.1007/978-981-10-3322-3_27
    """

    name = "wlanrssi"
    urls = ("https://archive.ics.uci.edu/static/public/422/wireless+indoor+localization.zip",)
    files = {"all": ("wifi_localization.txt", "2ae62faa28071e47b875d6bb64175238574a8d1366d1b64ab355bc1c87220294")}
    meta = {
        "modality": "wifi_rssi",
        "units": "dBm",
        "raw_missing_value": None,  # the file has no missing readings
        "task": "room_classification",
        "crs": None,
        "pos_names": (),
        "pos_units": None,
        "rooms": (1, 2, 3, 4),
        "license": "CC BY 4.0",
        "doi": "10.24432/C51880",
        "citation": "Bhatt, Wireless Indoor Localization, UCI Machine Learning Repository, 2017, "
                    "doi:10.24432/C51880",
        "url": "https://archive.ics.uci.edu/dataset/422/wireless+indoor+localization",
    }
    n_aps = 7

    def _parse(self, path, split):
        raw = np.loadtxt(path, ndmin=2)  # whitespace-separated (tabs, CRLF line ends)
        if raw.shape[1] != self.n_aps + 1:
            raise ValueError(f"{path.name}: expected {self.n_aps} RSSI columns and a room column, got {raw.shape[1]}")
        room = raw[:, self.n_aps]
        if not set(np.unique(room)) <= set(self.meta["rooms"]):
            raise ValueError(f"{path.name}: room labels {np.unique(room)} outside {self.meta['rooms']}")
        rssi = raw[:, :self.n_aps].astype(np.float32)
        if np.any(rssi >= 0):
            raise ValueError(f"{path.name}: non-negative RSSI values")
        ids = np.array([f"{split}-{i:05d}" for i in range(len(raw))])
        return SampleTable(rssi, np.empty((len(raw), 0)), None, None, {"room": room.astype(np.int64)}, ids,
                           meta={"feature_names": tuple(f"AP{j}" for j in range(1, self.n_aps + 1))})
