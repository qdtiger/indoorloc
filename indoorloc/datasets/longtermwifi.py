"""LongTermWiFi: 25 monthly WiFi RSSI campaigns in the UJI library (Mendoza-Silva et al., 2018)."""
from __future__ import annotations

import re

import numpy as np

from ..core import SampleTable
from ._base import Dataset

MONTHS = tuple(range(1, 26))
_ARCHIVE = "UJI_LIB_DB_v2.2.zip"
_SET = re.compile(r"(?:.*/)?db/(\d\d)/(trn|tst)(\d\d)(rss|crd|tms|ids)\.csv")
_S3, _A5 = "Samsung Galaxy S3", "Samsung Galaxy A5 (2017)"


def _device(month: int, kind: str, campaign: int) -> str:
    """Month 25 repeats training set 1 and test sets 1-5 with a second phone: "Only files corresponding
    to training 2 and tests 6-10 from month 25 were collected using a Samsung Galaxy A5 (2017)"
    (db/Readme.txt of the archive); trn02/tst06-10 have exactly the positions of trn01/tst01-05."""
    newer = month == 25 and ((kind == "trn" and campaign == 2) or (kind == "tst" and campaign >= 6))
    return _A5 if newer else _S3


def _unix_seconds(stamps) -> np.ndarray:
    """``YYYYMMDDhhmmssfff`` wall-clock stamps -> float64 unix seconds (read as UTC).

    Seconds are added as a number, not parsed as a clock field: four stamps of v2.2 write
    second 60 (e.g. ``20160727123260000`` between ...3256156 and ...3303736), a rounding slip
    that this reads as the next minute's second 0.
    """
    minutes = np.array([f"{t[:4]}-{t[4:6]}-{t[6:8]}T{t[8:10]}:{t[10:12]}" for t in stamps], dtype="datetime64[m]")
    seconds = np.array([int(t[12:14]) + int(t[14:17]) / 1000.0 for t in stamps])
    return minutes.astype(np.int64) * 60.0 + seconds


class LongTermWiFi(Dataset):
    """UJI library long-term WiFi database (Mendoza-Silva et al., Data 2018), v2.2, official train/test.

    One surveyor collected 104 160 scans of 620 MAC addresses with a Samsung Galaxy S3 at fixed
    positions on the 3rd and 5th floors of the Universitat Jaume I library, one campaign per
    month for 25 months (June 2016 - July 2018). Every month has a training set (month 1 has
    15) and five test sets, each measured at the same positions every month; month 25 repeats
    one training and five test sets with a Galaxy A5 (2017). Split ``train`` holds the training
    sets (``trn``) and ``test`` the test sets (``tst``) of the selected months; the authors'
    protocol trains and tests within each month, or trains on month m and tests later months.

    ``X`` is (N, 620) float32 RSSI in dBm; the files' "not detected" value +100 becomes NaN.
    The RSS files have no header: columns are identified by position (``WAP001`` .. ``WAP620``,
    the same in every month since v2.0). ``pos`` is (x, y) in metres in the library's local
    frame and ``floor`` the library floor (3 or 5), both unchanged. ``ids`` are the official
    10-digit sample ids (month, campaign, train/test, point, sample). Groups: ``month``
    (1-25), ``campaign`` (the set number within the month), ``device`` (phone model) and
    ``time`` (float64 unix seconds, millisecond resolution, of the wall-clock timestamp read
    as UTC: the files state no time zone).

    The archive itself is the checksummed file (one sha256 for its 680 CSV members); it is
    read in place, without unpacking.

    Parameters
    ----------
    month : None (all 25 months, the default), a month number 1-25, or a sequence of them.

    References
    ----------
    Mendoza-Silva, G.M., Richter, P., Torres-Sospedra, J., Lohan, E.S., Huerta, J., "Long-Term
    WiFi Fingerprinting Dataset for Research on Robust Indoor Positioning", Data 3(1), 3, 2018.
    https://doi.org/10.3390/data3010003. Data v2.2: https://doi.org/10.5281/zenodo.3748719
    """

    name = "longtermwifi"
    months = MONTHS
    urls = {_ARCHIVE: ("https://zenodo.org/api/records/3748719/files/UJI_LIB_DB_v2.2.zip/content",
                       "https://zenodo.org/records/3748719/files/UJI_LIB_DB_v2.2.zip?download=1")}
    _sha256 = "0a74d814be48359dc4b9f7aee26cf8905574d4999db722aa477cfd571a2ce505"  # Zenodo md5 a4577366...98ce
    files = {"train": (_ARCHIVE, _sha256), "test": (_ARCHIVE, _sha256)}
    # No 'validation' alias: the source has no validation file (carve one from train, e.g. kfold on groups).
    meta = {
        "modality": "wifi_rssi",
        "units": "dBm",
        "raw_missing_value": 100,
        "crs": "local",
        "pos_names": ("x", "y"),
        "pos_units": "m",
        "floors": (3, 5),
        "license": "CC BY 4.0 (data, Readme.txt); MIT (scripts)",
        "doi": "10.5281/zenodo.3748719",
        "citation": "Mendoza-Silva et al., Long-Term WiFi Fingerprinting Dataset for Research on Robust Indoor "
                    "Positioning, Data 3(1):3, 2018, doi:10.3390/data3010003",
        "url": "https://zenodo.org/records/3748719",
    }
    n_aps = 620

    def __init__(self, root=None, *, download: bool = False, verify: bool = True, month=None):
        super().__init__(root, download=download, verify=verify)
        error = ValueError(f"month must be None, a month number from 1 to 25, or a sequence of them; got {month!r}")
        try:
            wanted = MONTHS if month is None else (month,) if np.ndim(month) == 0 else tuple(np.ravel(month))
        except (TypeError, ValueError):
            raise error from None
        if not wanted or any(isinstance(m, (bool, np.bool_, str)) or m not in MONTHS for m in wanted):
            raise error
        self.selected = tuple(sorted({int(m) for m in wanted}))

    def _parse(self, path, split):
        import io  # archive readers load only when the archive is read
        import zipfile

        kind = "trn" if split == "train" else "tst"
        with zipfile.ZipFile(path) as archive:
            members = {}
            for member in archive.namelist():
                match = _SET.fullmatch(member)
                if match and match[2] == kind and int(match[1]) in self.selected:
                    members.setdefault((int(match[1]), int(match[3])), {})[match[4]] = member
            if not members:
                raise ValueError(f"{path.name}: no {kind} sets for months {self.selected}")
            read = lambda member: archive.read(member).decode("ascii").split()  # noqa: E731
            sets = []
            for (month, campaign), part in sorted(members.items()):
                if len(part) != 4:
                    raise ValueError(f"{path.name}: set {kind}{campaign:02d} of month {month} lacks "
                                     f"{sorted({'rss', 'crd', 'tms', 'ids'} - set(part))}")
                ids, tms = read(part["ids"]), read(part["tms"])
                rssi = np.loadtxt(io.BytesIO(archive.read(part["rss"])), delimiter=",", ndmin=2)
                crd = np.loadtxt(io.BytesIO(archive.read(part["crd"])), delimiter=",", ndmin=2)
                rows = {len(rssi), len(crd), len(ids), len(tms)}
                if rssi.shape[1] != self.n_aps or crd.shape[1] != 3 or len(rows) != 1:
                    raise ValueError(f"{path.name}: {part['rss']} and its companions disagree in shape")
                code = np.array([int(i) for i in ids])
                expect = month * 10**8 + campaign * 10**6 + (1 if kind == "trn" else 2) * 10**5
                if np.any(code // 10**5 * 10**5 != expect):
                    raise ValueError(f"{part['ids']}: ids do not encode month {month}, set {kind}{campaign:02d}")
                sets.append((month, campaign, rssi.astype(np.float32), crd, ids, tms))

        rssi = np.concatenate([s[2] for s in sets])
        rssi[rssi == self.meta["raw_missing_value"]] = np.nan
        if np.any(rssi > 0):
            raise ValueError(f"{path.name}: positive RSSI other than the +100 'not detected' marker")
        crd = np.concatenate([s[3] for s in sets])
        stamps = [t for s in sets for t in s[5]]  # YYYYMMDDhhmmssfff
        repeat = lambda value: np.concatenate([np.full(len(s[4]), value(s)) for s in sets])  # noqa: E731
        groups = {"month": repeat(lambda s: s[0]).astype(np.int64),
                  "campaign": repeat(lambda s: s[1]).astype(np.int64),
                  "device": repeat(lambda s: _device(s[0], kind, s[1])),
                  "time": _unix_seconds(stamps)}
        ids = np.array([i.zfill(10) for s in sets for i in s[4]])  # releases before v2.2 wrote 9 digits early on
        return SampleTable(rssi, crd[:, :2], crd[:, 2], None, groups, ids,
                           meta={"feature_names": tuple(f"WAP{j:03d}" for j in range(1, self.n_aps + 1)),
                                 "selected_months": self.selected})
