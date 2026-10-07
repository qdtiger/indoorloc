"""Per-scan WiFi view for convenience. Storage is always the SampleTable; this wraps one row."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import SampleTable
from ._scan import ScanView


@dataclass(frozen=True, eq=False)
class WiFiSignal(ScanView):
    """One RSSI scan: a (F,) array in dBm (NaN = not heard) plus optional AP ids.

    ``np.asarray(signal)`` gives the array, so every L2 function accepts a WiFiSignal
    where it accepts a row, and every transform returns a WiFiSignal for one.
    Algorithms do not live here (they belong to L3/L5).
    """

    rssi: np.ndarray
    ap_ids: tuple[str, ...] | None = None

    _ids_field = "ap_ids"

    @classmethod
    def from_raw(cls, values, missing: float = 100, ap_ids=None) -> WiFiSignal:
        """From a raw file row whose 'not detected' sentinel is ``missing`` (UJIIndoorLoc: 100)."""
        rssi = np.asarray(values, dtype=np.float32)
        return cls(rssi=np.where(rssi == missing, np.float32(np.nan), rssi), ap_ids=ap_ids)

    @classmethod
    def from_readings(cls, readings, ap_ids) -> WiFiSignal:
        """From a sparse scan ``{ap_id: dBm}`` laid out in the order ``ap_ids`` (e.g. a table's
        ``meta["feature_names"]``). APs outside ``ap_ids`` are ignored; unheard ones are NaN."""
        ap_ids = tuple(ap_ids)
        return cls(rssi=cls._dense(readings, ap_ids), ap_ids=ap_ids)

    @classmethod
    def from_table(cls, table: SampleTable, i: int) -> WiFiSignal:
        """A view of row ``i`` (no copy)."""
        return cls(rssi=table.X[i], ap_ids=table.meta.get("feature_names"))
