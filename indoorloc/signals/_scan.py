"""Behaviour shared by the one-scan views (``WiFiSignal``, ``BLESignal``).

A view wraps a single ``(F,)`` RSSI vector in dBm (NaN = not heard) and, optionally,
the transmitter ids in column order. Storage is always a SampleTable; a view is a
convenience for one row and holds no algorithms.
"""
from __future__ import annotations

import dataclasses

import numpy as np


class ScanView:
    """Mixin for frozen dataclasses with an ``rssi`` field and one id field (``_ids_field``)."""

    _ids_field = "ids"

    @property
    def ids(self) -> tuple[str, ...] | None:
        return getattr(self, self._ids_field)

    def __post_init__(self):
        rssi = np.asarray(self.rssi)
        if rssi.ndim != 1:
            raise ValueError(f"one scan is a 1-D array, got shape {rssi.shape}")
        ids = self.ids
        if ids is not None:
            if len(ids) != len(rssi):
                raise ValueError(f"{len(ids)} transmitter ids for {len(rssi)} readings")
            object.__setattr__(self, self._ids_field, tuple(ids))
        object.__setattr__(self, "rssi", rssi)

    def __array__(self, dtype=None, copy=None):
        return np.array(self.rssi, dtype=dtype, copy=copy)

    def __len__(self) -> int:
        return len(self.rssi)

    @property
    def detected(self) -> dict[str, float]:
        """Heard transmitters -> reading (dBm); ids default to the column number."""
        ids = self.ids or [str(j) for j in range(len(self.rssi))]
        return {ids[j]: float(self.rssi[j]) for j in np.flatnonzero(~np.isnan(self.rssi))}

    def replace(self, **changes):
        return dataclasses.replace(self, **changes)

    def take(self, columns):
        """The view restricted to ``columns`` (positions), ids included."""
        columns = np.asarray(columns, dtype=np.intp)
        ids = self.ids
        return self.replace(rssi=self.rssi[columns],
                            **{self._ids_field: None if ids is None else tuple(ids[j] for j in columns)})

    @classmethod
    def _dense(cls, readings, ids) -> np.ndarray:
        """``{id: dBm}`` -> ``(len(ids),)`` float32 with NaN for ids without a reading."""
        position = {name: j for j, name in enumerate(ids)}
        if len(position) != len(ids):
            raise ValueError("transmitter ids must be unique")
        row = np.full(len(ids), np.nan, dtype=np.float32)
        for name, value in readings.items():
            j = position.get(name)
            if j is not None:
                row[j] = value
        return row
