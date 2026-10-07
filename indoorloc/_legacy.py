"""indoorloc 0.1 compatibility for the 0.2.x releases only (removed in 0.3; rule 5.7).

The top layer: it may import any layer and no layer imports it. Everything 0.1-shaped
lives here: the value types, the ``WiFiSignal(rssi_values=)`` constructor, the dataset
object, the 0.1 dataset id and ``EvaluationResults.from_predictions``. Lower layers
reach it only through the string table ``datasets._LEGACY_IDS`` and two duck-typed
hooks in ``methods/base.py``; each is listed in rule 5.7 and tested.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .core import Prediction, SampleTable
from .datasets.ujiindoorloc import UJIIndoorLoc
from .evaluation import EvaluationResults, evaluate
from .signals import Compose, FillMissing, RSSINormalize
from .signals import WiFiSignal as _WiFiSignal


def _warn(old: str, new: str) -> None:
    warnings.warn(f"{old} is deprecated and will be removed in 0.3; use {new}", FutureWarning, stacklevel=3)


@dataclass
class Coordinate:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Location:
    coordinate: Coordinate
    floor: int | None = None
    building_id: str | None = None
    x = property(lambda self: self.coordinate.x)
    y = property(lambda self: self.coordinate.y)


@dataclass
class LocalizationResult:
    location: Location
    coordinate = property(lambda self: self.location.coordinate)
    floor = property(lambda self: self.location.floor)
    x = property(lambda self: self.location.x)
    y = property(lambda self: self.location.y)


def _location(pos, floor, building, i: int) -> Location:
    return Location(Coordinate(*map(float, pos[i][:3])), None if floor is None else int(floor[i]),
                    None if building is None else str(int(building[i])))


_UNSET = object()


class WiFiSignal(_WiFiSignal):
    """0.1 constructor ``WiFiSignal(rssi_values=row)`` (100 = not detected) over the 0.2 view.

    The 0.1 behaviour (100 -> NaN, a FutureWarning, and ``model.predict(signal)`` returning a
    0.1 ``LocalizationResult``) applies to ``rssi_values=`` and to a positional row that holds
    the 0.1 value 100. ``WiFiSignal(rssi=...)`` and a positional row in dBm with NaN are the
    0.2 one-scan view: no warning, and methods treat them as 0.2 input.
    """

    def __init__(self, *args, rssi=None, ap_ids=None, rssi_values=_UNSET, ap_list=None, _legacy: bool = False):
        if len(args) > 2:
            raise TypeError(f"WiFiSignal takes at most 2 positional arguments ({len(args)} given)")
        if args:
            if rssi_values is not _UNSET:
                raise TypeError("WiFiSignal got rssi_values twice (positionally and by keyword)")
            rssi_values, ap_list = args[0], (args[1] if len(args) > 1 else ap_list)
            positional = True
        else:
            positional = False
        if rssi is None:
            if rssi_values is _UNSET or rssi_values is None:
                raise TypeError("WiFiSignal needs rssi= (dBm, NaN = not heard) or the 0.1 rssi_values=")
            values = np.asarray(rssi_values, dtype=np.float32)
            sentinel = bool(np.any(values == 100))
            _legacy = _legacy or sentinel or not positional  # rssi_values= is the 0.1 spelling
            if _legacy:
                _warn("WiFiSignal(rssi_values=...)", "WiFiSignal.from_raw(row, missing=100)")
            rssi = np.where(values == 100, np.float32(np.nan), values)
        super().__init__(rssi, ap_ids if ap_ids is not None else (tuple(ap_list) if ap_list else None))
        object.__setattr__(self, "_is_legacy", bool(_legacy))

    @classmethod
    def from_raw(cls, values, missing: float = 100, ap_ids=None) -> WiFiSignal:
        return cls(rssi=_WiFiSignal.from_raw(values, missing).rssi, ap_ids=ap_ids)

    @classmethod
    def from_table(cls, table: SampleTable, i: int) -> WiFiSignal:
        return cls(rssi=table.X[i], ap_ids=table.meta.get("feature_names"))

    @property
    def _legacy_wrap(self):  # BaseLocalizer.predict hook: only 0.1-style signals get a LocalizationResult
        if not getattr(self, "_is_legacy", False):
            return None
        return lambda pred: LocalizationResult(_location(pred.pos, pred.floor, pred.building, 0))


@dataclass(frozen=True, eq=False)
class LegacyDataset(SampleTable):
    """The 0.1 dataset object: min-max features in [0, 1], ``ds[i] -> (WiFiSignal, Location)``,
    ``to_tensors()``. It is a SampleTable, so every 0.2 function takes it unchanged."""

    @classmethod
    def from_table(cls, table: SampleTable) -> LegacyDataset:
        t = Compose([FillMissing(), RSSINormalize()])(table)  # 0.1 default: normalize=True
        return cls(t.X, t.pos, t.floor, t.building, t.groups, t.ids, t.meta)

    def __getitem__(self, rows):
        if not isinstance(rows, (int, np.integer)):
            return super().__getitem__(rows)
        return WiFiSignal(rssi=self.X[rows], _legacy=True), _location(self.pos, self.floor, self.building, rows)

    signals = property(lambda self: [self[i][0] for i in range(len(self))])
    locations = property(lambda self: [self[i][1] for i in range(len(self))])

    def to_tensors(self) -> tuple[np.ndarray, np.ndarray]:
        """0.1: X (N, F) float32 and y (N, 4) float32 = [x, y, floor, building], 0 if unlabelled."""
        zeros = np.zeros(len(self))
        y = np.column_stack([self.pos[:, :2], zeros if self.floor is None else self.floor,
                             zeros if self.building is None else self.building])
        return np.array(self.X), y.astype(np.float32)


class UJIndoorLoc(UJIIndoorLoc):
    """The 0.1 id ``"ujindoorloc"``: same files, but ``load`` returns the normalized LegacyDataset."""

    def load(self, split: str = "train") -> LegacyDataset:
        _warn('load_dataset("ujindoorloc"), which returns normalized 0.1 dataset objects,',
              'load_dataset("ujiindoorloc") (SampleTables in dBm) and create_model(..., preprocess=FillMissing(-104))')
        return LegacyDataset.from_table(super().load(split))


def _columns(items):
    locations = [getattr(item, "location", item) for item in items]  # LocalizationResult or Location
    labels = lambda values: None if None in values else np.array([int(v) for v in values])  # noqa: E731
    return (np.array([(loc.coordinate.x, loc.coordinate.y) for loc in locations], dtype=np.float64),
            labels([loc.floor for loc in locations]), labels([loc.building_id for loc in locations]))


def _from_predictions(cls, predictions, ground_truths) -> EvaluationResults:
    """0.1 ``EvaluationResults.from_predictions(list[Location], list[Location])``."""
    _warn("EvaluationResults.from_predictions", "evaluate(y_true, y_pred, floor_true=..., floor_pred=...)")
    (pos_t, floor_t, building_t), (pos_p, floor_p, building_p) = _columns(ground_truths), _columns(predictions)
    return evaluate(pos_t, pos_p, floor_true=floor_t, floor_pred=floor_p,
                    building_true=building_t, building_pred=building_p)


# The only attribute _legacy adds to a 0.2 class. Every way to obtain a 0.1 Location
# (iloc.Location, the indoorloc.locations path) imports this module first.
EvaluationResults.from_predictions = classmethod(_from_predictions)
