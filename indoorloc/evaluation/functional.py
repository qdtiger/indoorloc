"""L4: metrics as functions of plain arrays (the sklearn.metrics style)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from ..core import Prediction, SampleTable


def position_errors(y_true, y_pred) -> np.ndarray:
    """Per-sample Euclidean error over every coordinate axis (1-D, 2-D or 3-D), float64.

    NaN where a prediction is NaN (a method that could not place the sample), and NaN for
    every sample when positions have no axes at all (room-level data such as WLANRSSI).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} and y_pred {y_pred.shape} differ in shape")
    if y_true.ndim > 1 and y_true.shape[-1] == 0:
        return np.full(y_true.shape[:-1], np.nan)
    diff = y_true - y_pred
    return np.sqrt(np.sum(diff * diff, axis=-1)) if diff.ndim > 1 else np.abs(diff)


def error_cdf(errors, thresholds=None):
    """Empirical CDF. No ``thresholds``: ``(sorted errors, P(E <= e_i))``; else ``P(E <= t)`` per t."""
    e = np.sort(np.asarray(errors, dtype=np.float64))
    if thresholds is None:
        return e, np.arange(1, len(e) + 1) / len(e)
    return np.searchsorted(e, np.asarray(thresholds, dtype=np.float64), side="right") / len(e)


def label_accuracy(true, pred) -> float | None:
    """Percentage of matching labels (every label counts, including negative floors); None if absent."""
    if true is None or pred is None:
        return None
    true, pred = np.asarray(true), np.asarray(pred)
    if true.shape != pred.shape:
        raise ValueError(f"label arrays differ in shape: {true.shape} vs {pred.shape}")
    return float(np.mean(true == pred) * 100.0)


@dataclass(frozen=True, eq=False)
class EvaluationResults:
    """Summary of one evaluation. Errors in coordinate units (times ``scale``); accuracies in percent."""

    errors: np.ndarray
    n: int
    mean_error: float
    median_error: float
    p75_error: float
    p90_error: float
    p95_error: float
    rmse: float
    max_error: float
    floor_accuracy: float | None = None
    building_accuracy: float | None = None
    n_failed: int = 0  # samples the method could not place (NaN prediction); excluded from the statistics

    def cdf(self, thresholds=None):
        return error_cdf(self.errors, thresholds)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "errors"}

    def summary(self) -> str:
        acc = lambda v: "n/a" if v is None else f"{v:.2f} %"  # noqa: E731
        failed = f", {self.n_failed} not placed" if self.n_failed else ""
        if np.isnan(self.mean_error):
            what = "no sample placed" if self.n_failed else "no positions"
            return (f"{what}  floor {acc(self.floor_accuracy)}  building {acc(self.building_accuracy)}"
                    f"  (n={self.n}{failed})")
        return (f"mean {self.mean_error:.4f}  median {self.median_error:.4f}  P90 {self.p90_error:.4f}"
                f"  floor {acc(self.floor_accuracy)}  building {acc(self.building_accuracy)}  (n={self.n}{failed})")

    __str__ = summary
    __repr__ = summary  # Jupyter shows repr: the one-line summary, not the (N,) error array


_ARGUMENT_ORDER = ("{fn}(y_true, y_pred): truth first (SampleTable or positions), estimate second "
                  "(Prediction or positions); PDRFusion.run returns (t, Prediction): pass the Prediction")


def _is_track(obj) -> bool:
    """An ``apps.pdr.StepTrack`` (duck typed: L4 never imports apps)."""
    return type(obj).__name__ == "StepTrack" or (hasattr(obj, "heading") and hasattr(obj, "position_at"))


def _misplaced(value, side: str) -> str | None:
    """Why ``value`` cannot be the ``side`` argument ("y_true" / "y_pred"), or None if it can."""
    if side == "y_true" and isinstance(value, Prediction):
        return "y_true is a Prediction (the arguments are swapped?)"
    if side == "y_pred" and isinstance(value, SampleTable):
        return "y_pred is a SampleTable (the arguments are swapped?)"
    if _is_track(value):
        return (f"{side} is a StepTrack; pass its positions (track.pos, or track.position_at(t) at the "
                "truth's times) or the Prediction of PDRFusion.run")
    if isinstance(value, tuple):
        held = [type(v).__name__ for v in value if isinstance(v, (Prediction, SampleTable)) or _is_track(v)]
        if held:
            return f"{side} is a tuple holding a {held[0]}"
        try:
            ragged = np.asarray(value).dtype == object
        except (ValueError, TypeError):  # e.g. (t, pos): arrays of different shapes
            ragged = True
        if ragged:
            return f"{side} is a tuple of arrays of different shapes (e.g. (t, pos) or (X, pos))"
    return None


def _check_arguments(y_true, y_pred, fn: str = "evaluate") -> None:
    """TypeError with the argument order when ``y_true``/``y_pred`` are swapped or not positions."""
    for value, side in ((y_true, "y_true"), (y_pred, "y_pred")):
        why = _misplaced(value, side)
        if why:
            raise TypeError(f"{why}. {_ARGUMENT_ORDER.format(fn=fn)}")


def evaluate(y_true, y_pred, *, floor_true=None, floor_pred=None, building_true=None,
             building_pred=None, scale: float = 1.0) -> EvaluationResults:
    """Score predicted positions and, if given, floor/building labels.

    Plain arrays are the interface. As a convenience ``y_true`` may be a SampleTable
    and ``y_pred`` a Prediction; explicit label arguments win over their columns, and
    if both carry ids they must match row for row.
    ``scale`` multiplies errors (e.g. ``table.meta["ground_scale"]`` for EPSG:3857 data).
    Samples with a NaN prediction are counted in ``n_failed`` and left out of the error
    statistics (always report ``n_failed`` next to them); ``errors`` keeps them as NaN, so
    ``cdf(t)`` counts them as never within ``t``. Positions with no axes (room-level data)
    give NaN error statistics; the label accuracies are still computed. A NaN in the ground
    truth is an error (an unlabelled row is not a failure of the method): select the labelled
    rows first. Swapped arguments (a Prediction as ``y_true``, a SampleTable as ``y_pred``), a
    tuple such as the ``(t, Prediction)`` of ``PDRFusion.run`` and a PDR ``StepTrack`` are a
    TypeError that states the argument order.
    """
    _check_arguments(y_true, y_pred)
    if isinstance(y_true, SampleTable) and isinstance(y_pred, Prediction) and y_pred.ids is not None:
        if not np.array_equal(y_true.ids, y_pred.ids):
            raise ValueError("y_true and y_pred carry different ids: their rows are not aligned "
                             "(was one side reordered or subset?)")
    if isinstance(y_true, SampleTable):
        floor_true = y_true.floor if floor_true is None else floor_true
        building_true = y_true.building if building_true is None else building_true
        y_true = y_true.pos
    if isinstance(y_pred, Prediction):
        floor_pred = y_pred.floor if floor_pred is None else floor_pred
        building_pred = y_pred.building if building_pred is None else building_pred
        y_pred = y_pred.pos
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    if not np.all(np.isfinite(y_true)):
        bad = int(np.count_nonzero(~np.all(np.isfinite(y_true.reshape(len(y_true), -1)), axis=1)))
        raise ValueError(f"y_true has NaN or infinite positions in {bad} of {len(y_true)} rows; evaluate "
                         "needs ground truth (select the labelled rows, e.g. table[np.isfinite(table.pos).all(1)])")
    if y_true.ndim == 2 and y_pred.ndim == 1 and y_true.shape[1] == 1:
        y_pred = y_pred[:, None]  # 1-D predictions of a single-axis (corridor) target
    errors = position_errors(y_true, y_pred) * float(scale)
    if len(errors) == 0:
        raise ValueError("nothing to evaluate: 0 samples")
    no_axes = y_true.ndim > 1 and y_true.shape[-1] == 0
    placed = errors[np.isfinite(errors)]
    if len(placed):
        p50, p75, p90, p95 = np.percentile(placed, [50, 75, 90, 95])
        mean, rmse, worst = placed.mean(), np.sqrt(np.mean(placed ** 2)), placed.max()
    else:
        p50 = p75 = p90 = p95 = mean = rmse = worst = np.nan
    return EvaluationResults(
        errors=errors, n=len(errors), mean_error=float(mean), median_error=float(p50),
        p75_error=float(p75), p90_error=float(p90), p95_error=float(p95),
        rmse=float(rmse), max_error=float(worst),
        floor_accuracy=label_accuracy(floor_true, floor_pred),
        building_accuracy=label_accuracy(building_true, building_pred),
        n_failed=0 if no_axes else int(len(errors) - len(placed)),
    )
