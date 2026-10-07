"""L1: public and simulated datasets as SampleTables of numpy arrays (numpy only).

Every entry is a ``"module:Class"`` string, so listing datasets imports nothing.
Measured datasets are downloaded on first use and sha256-verified; simulated ones
are generated deterministically from a seed. Figures of a dataset's layout (per floor, 2-D or
3-D) are in ``indoorloc.datasets.plot`` (import it explicitly; matplotlib).
"""
from __future__ import annotations

from ..core import Registry, SampleTable
from ._base import Dataset, default_root, sha256sum

_P = "indoorloc.datasets"
DATASETS = Registry("dataset", {
    # WiFi RSSI
    "ujiindoorloc": f"{_P}.ujiindoorloc:UJIIndoorLoc",
    "sodindoorloc": f"{_P}.sodindoorloc:SODIndoorLoc",
    "tampere": f"{_P}.tampere:Tampere",
    "tuji1": f"{_P}.tuji1:TUJI1",
    "longtermwifi": f"{_P}.longtermwifi:LongTermWiFi",
    "wlanrssi": f"{_P}.wlanrssi:WLANRSSI",
    # BLE RSSI
    "ble_indoor": f"{_P}.ble_indoor:BLEIndoor",
    "ble_rssi_uci": f"{_P}.ble_rssi_uci:BLERSSIUCI",
    "ibeacon_rssi": f"{_P}.ibeacon_rssi:IBeaconRSSI",
    # WiFi CSI
    "csi_fingerprint": f"{_P}.csi_fingerprint:CSIFingerprint",
    "hwild": f"{_P}.hwild:HWILD",
    "haloc": f"{_P}.haloc:HALOC",
    # Multi-sensor trajectories (IMU + WiFi + floor plans)
    "ilc2020": f"{_P}.ilc2020:ILC2020",
    # Simulated (no download; deterministic from a seed)
    "synthetic_office": f"{_P}.simulated.office:SyntheticOffice",
    "deepmimo": f"{_P}.simulated.deepmimo:DeepMIMO",
    # Aliases
    "uji": "ujiindoorloc",
    "sod": "sodindoorloc",
})
# 0.1 ids keep their 0.1 behaviour until 0.3: normalized LegacyDataset + FutureWarning.
# A string, so L1 never imports _legacy (compat hook, rule 5.7).
_LEGACY_IDS = {"ujindoorloc": "indoorloc._legacy:UJIndoorLoc"}


def list_datasets() -> list[str]:
    """Registry names of the built-in and registered datasets, sorted (aliases such as ``"uji"``
    left out). Listing imports no loader; ``dataset_info(name)`` gives a dataset's facts."""
    return DATASETS.names()


def register_dataset(name: str, target=None, *, force: bool = False):
    """Register a ``Dataset`` subclass or a ``"module:Class"`` string under ``name``; usable as a
    decorator (``@register_dataset("my_office")``), like ``methods.register_model``.

    ``load_dataset(name)``, ``dataset_info(name)`` and the CLI then accept the name (the CLI only
    in the process that registered it). An existing name raises ``KeyError`` unless ``force=True``.
    Names are case-insensitive. Unregistered classes load too: ``load_dataset("mypkg.data:MyData")``.
    """
    return DATASETS.register(name, target, force=force)


def load_dataset(name: str, split: str | tuple[str, ...] | None = None, *, root=None,
                 download: bool = True, verify: bool = True, **options) -> SampleTable | tuple[SampleTable, ...]:
    """``load_dataset("ujiindoorloc")`` -> (train, test); ``split="test"`` -> one table.

    ``split=None`` means the dataset's ``default_splits``: the official ``(train, test)`` when
    the dataset has both, otherwise its single table (``"all"``, else its first split; build a
    split with the ``indoorloc.evaluation`` protocols). A missing file is downloaded
    (``download=False`` to forbid) and sha256-checked (``verify=False`` to skip). Dataset-specific
    ``options`` (e.g. ``building=`` or ``seed=``) go to the dataset class.

    ``name`` is a registry name (``list_datasets()``, or one added with ``register_dataset``), a
    ``"package.module:Class"`` path, or a ``Dataset`` subclass itself (no registration needed).
    """
    if isinstance(name, str):
        name = _LEGACY_IDS.get(name.lower(), name)
    dataset = DATASETS.get(name)(root, download=download, verify=verify, **options)
    if split is None:
        split = tuple(getattr(dataset, "default_splits", ()))
        if not split:
            raise ValueError(f"{name} declares no splits; pass split=...")
        split = split[0] if len(split) == 1 else split
    if isinstance(split, str):
        return dataset.load(split)
    return tuple(dataset.load(s) for s in split)


def dataset_info(name: str) -> dict:
    """Class-level facts of a dataset (modality, crs, license, doi, splits) without loading it."""
    cls = DATASETS.get(name)
    return {"name": cls.name, "splits": tuple(cls.files), "doc": (cls.__doc__ or "").strip().split("\n")[0],
            **dict(cls.meta)}


__all__ = ["DATASETS", "Dataset", "dataset_info", "default_root", "list_datasets", "load_dataset", "register_dataset",
           "sha256sum"]
