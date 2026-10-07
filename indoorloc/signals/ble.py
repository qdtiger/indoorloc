"""Per-scan BLE view: one vector of beacon RSSI in dBm (NaN = not heard).

BLE datasets encode "not heard" with different sentinels (-200 in the UCI BLE RSSI
dataset, -100 or 0 elsewhere), and -100 dBm is also a real, weak BLE reading. So
``BLESignal.from_raw`` has no default sentinel: the caller states it. Ranging from
RSSI (log-distance model) lives in ``signals.ranging``; localization in L3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import SampleTable
from ._scan import ScanView


@dataclass(frozen=True, eq=False)
class BLESignal(ScanView):
    """One BLE scan: a (F,) array in dBm (NaN = not heard) plus optional beacon ids.

    Beacon ids are free-form strings, e.g. ``"<uuid>:<major>:<minor>"`` for iBeacon or a
    MAC address. Every L2 RSSI transform accepts a BLESignal and returns one.
    """

    rssi: np.ndarray
    beacon_ids: tuple[str, ...] | None = None

    _ids_field = "beacon_ids"

    @classmethod
    def from_raw(cls, values, missing: float, beacon_ids=None) -> BLESignal:
        """From a raw file row whose 'not detected' sentinel is ``missing`` (required: it differs
        between datasets, and a -100 default would turn real -100 dBm readings into NaN)."""
        rssi = np.asarray(values, dtype=np.float32)
        return cls(rssi=np.where(rssi == missing, np.float32(np.nan), rssi), beacon_ids=beacon_ids)

    @classmethod
    def from_readings(cls, readings, beacon_ids) -> BLESignal:
        """From advertisements ``{beacon_id: dBm}`` laid out in the order ``beacon_ids``.
        Beacons outside ``beacon_ids`` are ignored; unheard ones are NaN."""
        beacon_ids = tuple(beacon_ids)
        return cls(rssi=cls._dense(readings, beacon_ids), beacon_ids=beacon_ids)

    @classmethod
    def from_table(cls, table: SampleTable, i: int) -> BLESignal:
        """A view of row ``i`` (no copy)."""
        return cls(rssi=table.X[i], beacon_ids=table.meta.get("feature_names"))

    @staticmethod
    def beacon_id(uuid: str, major: int, minor: int) -> str:
        """The iBeacon id convention used by the BLE loaders: ``"<uuid>:<major>:<minor>"``."""
        return f"{uuid.lower()}:{int(major)}:{int(minor)}"
