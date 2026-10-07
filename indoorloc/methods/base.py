"""The localizer contract: sklearn's regressor API over arrays, plus floor/building."""
from __future__ import annotations

import dataclasses

import numpy as np

from ..core import Estimator, Prediction, SampleTable
from ..core.table import as_labels


def _unpack(X, y=None, floor=None, building=None):
    """Arrays pass through; a SampleTable supplies whatever was not given explicitly."""
    if isinstance(X, SampleTable):
        return (X.X, X.pos if y is None else y, X.floor if floor is None else floor,
                X.building if building is None else building)
    if isinstance(y, (list, tuple)) and y and hasattr(y[0], "coordinate"):  # 0.1 hook (rule 5.7), gone in 0.3
        column = lambda values: None if None in values else [int(v) for v in values]  # noqa: E731
        floor = column([loc.floor for loc in y]) if floor is None else floor
        building = column([loc.building_id for loc in y]) if building is None else building
        y = [(loc.coordinate.x, loc.coordinate.y) for loc in y]  # fit(signals, list[Location])
    return X, y, floor, building


def _provenance(X) -> dict:
    keys = ("name", "split", "sha256", "crs")
    return {k: X.meta[k] for k in keys if k in X.meta} if isinstance(X, SampleTable) else {}


class BaseLocalizer(Estimator):
    """``fit(X, y, *, floor=None, building=None, **fit_params) -> self``; ``predict(X) -> positions``.

    ``X`` is (N, ...) with samples on the first axis, or a SampleTable (which then also
    supplies y, floor and building). ``localize`` returns the full Prediction; ``score``
    is minus the mean position error (so GridSearchCV maximises it); ``evaluate``
    returns an L4 EvaluationResults. Extra ``fit_params`` (``eval_set``, ``sample_domain``,
    ...) are passed to ``_fit``. Subclasses implement ``_fit(X, pos, floor, building,
    **fit_params)`` and ``_localize(X) -> Prediction`` and set ``_allow_nan`` /
    ``_allow_complex`` if they can take such input.

    ``_labels_optional`` (a class attribute or a property, False by default) is True for a
    model that learns nothing from the positions, e.g. a model-based method without
    calibration: ``fit(X)`` without ``y`` is then accepted and ``_fit`` receives ``pos=None``
    (it validates its options and records nothing learned from labels).
    """

    _estimator_type = "regressor"
    _allow_complex = False
    _labels_optional = False

    def fit(self, X, y=None, *, floor=None, building=None, **fit_params):
        info = _provenance(X)
        X, y, floor, building = _unpack(X, y, floor, building)
        X = self._validate(X, reset=True)
        labels = [as_labels(v, len(X), name) for v, name in ((floor, "floor"), (building, "building"))]
        if y is None:
            if not self._labels_optional:
                raise ValueError(self._missing_labels_message())
            self._fit(X, None, *labels, **fit_params)  # nothing to learn from positions
            ndim = 2
        else:
            y = np.asarray(y, dtype=np.float64)
            if y.ndim not in (1, 2) or len(y) != len(X):
                raise ValueError(f"y must be (N,) or (N, D) with N={len(X)}, got shape {y.shape}")
            if not np.all(np.isfinite(y)):
                raise ValueError("y contains NaN or inf")
            self._fit(X, y[:, None] if y.ndim == 1 else y, *labels, **fit_params)
            ndim = y.ndim
        # only now is the model fitted: a failed fit leaves it raising NotFittedError
        self.n_features_in_, self.input_shape_, self.target_ndim_ = X.shape[1], X.shape[1:], ndim
        self.train_info_ = info
        return self

    def _missing_labels_message(self) -> str:
        """The error of ``fit(X)`` without positions (subclasses add why they need them)."""
        return (f"{type(self).__name__} requires y to be passed, but the target y is None "
                "(pass positions, or a SampleTable)")

    def localize(self, X) -> Prediction:
        ids = X.ids if isinstance(X, SampleTable) else None
        pred = self._localize(self._validate(_unpack(X)[0]))
        return pred if ids is None else dataclasses.replace(pred, ids=ids)

    def predict(self, X) -> np.ndarray:
        wrap = getattr(X, "_legacy_wrap", None)  # 0.1 hook (rule 5.7), gone in 0.3
        if wrap is not None:
            return wrap(self.localize(np.asarray(X)[None]))  # predict(WiFiSignal) -> LocalizationResult
        pos = self.localize(X).pos
        return pos[:, 0] if getattr(self, "target_ndim_", 2) == 1 else pos

    def score(self, X, y=None) -> float:
        """Minus the mean position error (sklearn's convention: higher is better).

        NaN when any sample cannot be placed (a NaN prediction, e.g. too few anchors heard or an
        unresolvable geometry); ``evaluate(X, y)`` leaves such samples out of its errors and
        counts them in ``n_failed``.
        """
        from ..evaluation import position_errors  # lazy L3 -> L4 edge, like sklearn's score()

        X, y, _, _ = _unpack(X, y)
        self._check_truth(y, "score")
        return -float(position_errors(y, self.predict(X)).mean())

    def evaluate(self, X, y=None, *, floor=None, building=None, scale: float = 1.0):
        """Score on held-out data: ``model.evaluate(test_table)`` or ``(X, y, floor=...)``.

        Samples the model could not place are left out of the errors and counted in ``n_failed``."""
        from ..evaluation import evaluate

        X, y, floor, building = _unpack(X, y, floor, building)
        self._check_truth(y, "evaluate")
        return evaluate(y, self.localize(X), floor_true=floor, building_true=building, scale=scale)

    def _check_truth(self, y, what: str) -> None:
        if y is None:
            raise ValueError(f"{type(self).__name__}.{what} needs the true positions: pass y (positions) "
                             "or a SampleTable")

    def _validate(self, X, reset: bool = False) -> np.ndarray:
        """Checks X; ``reset=True`` (fit) skips the comparison with the fitted input shape."""
        if hasattr(X, "toarray"):
            raise TypeError("sparse input is not supported; pass a dense array (X.toarray())")
        X = np.asarray(X)
        if X.dtype.kind == "O":
            X = X.astype(np.float64)
        if X.dtype.kind == "c" and not self._allow_complex:
            raise ValueError(f"Complex data not supported by {type(self).__name__}; convert CSI with an "
                             "L2 transform (amplitude, phase) first")
        if X.dtype.kind not in "biufc":
            raise ValueError(f"{type(self).__name__} needs numeric features, got dtype {X.dtype}")
        if X.ndim < 2:
            raise ValueError(f"X must be (N, ...) with samples on the first axis, got shape {X.shape}. "
                             "Reshape your data: for one scan x pass x[None, :]")
        for axis, what in ((0, "sample"), (1, "feature")):
            if X.shape[axis] == 0:
                raise ValueError(f"Found array with 0 {what}(s) (shape={X.shape}) while a minimum of 1 is required.")
        if not reset:
            self._check_fitted()
            if X.shape[1:] != self.input_shape_:
                raise ValueError(f"X has {X.shape[1]} features, but {type(self).__name__} is expecting "
                                 f"{self.n_features_in_} features as input (input shape {self.input_shape_})")
        if X.dtype.kind in "fc" and (np.isinf(X).any() or (not self._allow_nan and np.isnan(X).any())):
            raise ValueError("X contains NaN or inf (missing readings?); fill them first, e.g. "
                             "create_model(..., preprocess=FillMissing(-104))")
        return X

    def _fit(self, X, pos, floor, building, **fit_params) -> None:
        raise NotImplementedError

    def _localize(self, X) -> Prediction:
        raise NotImplementedError
