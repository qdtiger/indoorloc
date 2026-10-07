"""Hierarchical localization: building, then floor, then position, each with its own sub-models."""
from __future__ import annotations

import numpy as np

from ..core import Prediction, clone
from .base import BaseLocalizer
from .ensemble import accepts, resolve_localizer


def _make(model):
    """A fresh stage model: a localizer (name or instance) or a classifier with fit/predict."""
    if isinstance(model, (str, BaseLocalizer)):
        return resolve_localizer(model)
    if not (hasattr(model, "fit") and hasattr(model, "predict")):
        raise TypeError(f"a stage model needs fit and predict, got {type(model).__name__}")
    return clone(model)  # a scikit-learn classifier: an unfitted deep copy


def _take(labels, rows):
    return None if labels is None else labels[rows]


class HierarchicalLocalizer(BaseLocalizer):
    """Coarse-to-fine localization: building -> floor -> position, one sub-model per group.

    A building model is fitted on all scans; for every building a floor model is fitted on
    that building's scans; for every (building, floor) a position model is fitted on that
    floor's scans. A query goes down the same path: its predicted building selects the floor
    model, and its predicted (building, floor) selects the position model. A group with a
    single label needs no model (the label is certain). Setting ``building_model`` or
    ``floor_model`` to None removes that level; if the training data have no building or
    floor labels, that level is skipped. Floor/building labels not decided by a level are
    passed through from the position models.

    Each model is a registry name, a localizer instance (its ``localize(X).building`` or
    ``.floor`` is used) or, for the building/floor levels, any classifier with
    ``fit(X_2d, labels)``/``predict(X_2d)`` (e.g. a scikit-learn classifier). Models are cloned,
    never fitted in place. Missing readings (NaN) are accepted when every stage accepts them
    (e.g. all "horus"). Save/load works when every stage is an indoorloc localizer or
    registry name.

    Parameters
    ----------
    building_model : building classifier (default "knn": vote of the 5 nearest scans), or None.
    floor_model : per-building floor classifier (default "knn"), or None.
    position_model : per-floor position localizer (default "wknn").

    References
    ----------
    Marques, N., Meneses, F., Moreira, A., "Combining similarity functions and majority rules
    for multi-building, multi-floor, WiFi positioning", IPIN 2012. DOI 10.1109/IPIN.2012.6418937
    Torres-Sospedra, J., Montoliu, R., Martínez-Usó, A., et al., "UJIIndoorLoc: A new
    multi-building and multi-floor database for WLAN fingerprint-based indoor localization
    problems", IPIN 2014. DOI 10.1109/IPIN.2014.7275492
    """

    def __init__(self, building_model="knn", floor_model="knn", position_model="wknn"):
        self.building_model = building_model
        self.floor_model = floor_model
        self.position_model = position_model

    def _stages(self):
        return [m for m in (self.building_model, self.floor_model, self.position_model) if m is not None]

    @property
    def _allow_nan(self) -> bool:  # NaN / complex input only if every stage takes it
        return accepts(self._stages(), "_allow_nan")

    @property
    def _allow_complex(self) -> bool:
        return accepts(self._stages(), "_allow_complex")

    # ------------------------------------------------------------------ stages
    @staticmethod
    def _fit_stage(spec, target, X, pos, floor, building):
        labels = floor if target == "floor" else building
        classes = np.unique(labels)
        if len(classes) == 1:
            return None, classes
        model = _make(spec)
        if isinstance(model, BaseLocalizer):
            model.fit(X, pos, floor=floor, building=building)
        else:
            model.fit(X.reshape(len(X), -1), labels)
        return model, classes

    @staticmethod
    def _predict_stage(model, classes, target, X):
        if model is None:
            return np.full(len(X), classes[0], dtype=np.int64)
        if isinstance(model, BaseLocalizer):
            out = getattr(model.localize(X), target)
            if out is None:
                raise ValueError(f"the {target} model {type(model).__name__} does not predict {target} labels")
            return out
        return np.asarray(model.predict(X.reshape(len(X), -1)), dtype=np.int64)

    def _fit(self, X, pos, floor, building):
        if self.position_model is None:
            raise ValueError("HierarchicalLocalizer needs a position_model")
        self.levels_ = tuple(name for name, labels, spec in (("building", building, self.building_model),
                                                                ("floor", floor, self.floor_model))
                             if labels is not None and spec is not None)
        by_building = "building" in self.levels_
        self.building_stage_ = (self._fit_stage(self.building_model, "building", X, pos, floor, building)
                                if by_building else (None, None))

        groups = np.unique(building) if by_building else [None]
        self.floor_keys_ = np.array([g for g in groups if g is not None], dtype=np.int64)
        self.floor_stages_ = []
        if "floor" in self.levels_:
            for g in groups:
                rows = slice(None) if g is None else building == g
                self.floor_stages_.append(self._fit_stage(self.floor_model, "floor", X[rows], pos[rows],
                                                          _take(floor, rows), _take(building, rows)))

        keys = self._keys(building, floor, len(X))
        self.position_keys_ = np.unique(keys, axis=0)
        self.position_models_ = []
        for key in self.position_keys_:
            rows = np.all(keys == key, axis=1)
            model = _make(self.position_model)
            try:
                model.fit(X[rows], pos[rows], floor=_take(floor, rows), building=_take(building, rows))
            except ValueError as err:
                raise ValueError(f"position model of group {dict(zip(self.levels_, key.tolist()))} "
                                 f"({int(rows.sum())} scans): {err}") from err
            self.position_models_.append(model)

    def _keys(self, building, floor, n):
        cols = [labels for name, labels in (("building", building), ("floor", floor)) if name in self.levels_]
        return np.column_stack(cols).astype(np.int64) if cols else np.zeros((n, 0), dtype=np.int64)

    # --------------------------------------------------------------- inference
    def _localize(self, X):
        n = len(X)
        building = floor = None
        if "building" in self.levels_:
            building = self._predict_stage(*self.building_stage_, "building", X)
        if "floor" in self.levels_:
            if building is not None and not np.isin(building, self.floor_keys_).all():
                raise ValueError("the building model predicted a building that has no floor model")
            floor = np.empty(n, dtype=np.int64)
            for i, (model, classes) in enumerate(self.floor_stages_):
                rows = np.ones(n, bool) if building is None else building == self.floor_keys_[i]
                if rows.any():
                    floor[rows] = self._predict_stage(model, classes, "floor", X[rows])

        keys = self._keys(building, floor, n)
        pos, spread, done = None, np.full(n, np.nan), np.zeros(n, bool)
        passed = {"floor": np.zeros(n, np.int64), "building": np.zeros(n, np.int64)}
        complete = {"floor": True, "building": True}
        for key, model in zip(self.position_keys_, self.position_models_):
            rows = np.all(keys == key, axis=1)
            if not rows.any():
                continue
            p = model.localize(X[rows])
            if pos is None:
                pos = np.empty((n, p.pos.shape[1]))
            pos[rows] = p.pos
            spread[rows] = np.nan if p.spread is None else p.spread
            for name in passed:
                value = getattr(p, name)
                complete[name] &= value is not None
                if value is not None:
                    passed[name][rows] = value
            done |= rows
        if not done.all():
            raise ValueError(f"{int((~done).sum())} scan(s) were routed to a (building, floor) group "
                             "that has no position model")
        # a level decides its label; otherwise the position models' own labels pass through
        building = building if building is not None else (passed["building"] if complete["building"] else None)
        floor = floor if floor is not None else (passed["floor"] if complete["floor"] else None)
        return Prediction(pos, floor, building, spread=None if np.isnan(spread).any() else spread)


__all__ = ["HierarchicalLocalizer"]
