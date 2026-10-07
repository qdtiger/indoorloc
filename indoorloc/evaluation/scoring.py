"""L4 competition scores and error statistics: IPIN / EvAAL rules, CEP, percentiles, bootstrap CIs.

Functions of plain arrays in the ``sklearn.metrics`` style. As in :func:`evaluate`, the
truth may be passed as a :class:`~indoorloc.core.SampleTable` and the estimate as a
:class:`~indoorloc.core.Prediction`; their floor/building columns are then used unless
given explicitly.

Floor-aware ("penalised") error of one estimate, the quantity both competitions rank on::

    e = ||p_xy - p̂_xy|| * scale  +  floor_penalty * |f - f̂|  +  building_penalty * [b != b̂]

* IPIN competitions (EvAAL framework): the score is the **75th percentile** (third quartile)
  of ``e`` with ``floor_penalty = 15 m`` (Potortì et al. 2017, Section 3.2: "a penalty P = 15 m
  is added for each floor error ... if the x y error is 4 m and the estimated floor is 2 while
  it should be 0, the computed error for that estimate will be 4 + 2P = 34 m").
* IPIN off-site (smartphone log) tracks on multi-building data add ``building_penalty = 50 m``
  to the same third-quartile score (Torres-Sospedra et al. 2018, Section 3.2: "15 and 50 m
  penalties were added to the geometric error if the IPS had not correctly estimated the
  floor and building, respectively ... The final metric ... was the third quartile").
* EvAAL-ETRI 2015 off-site track on UJIIndoorLoc: the score is the **mean** of ``e`` with
  ``floor_penalty = 4 m`` per floor and ``building_penalty = 50 m`` (Torres-Sospedra et al.
  2017, Section 3.1: "we add 4 m. for each wrong floor (absolute difference between the real
  and estimated floors) and 50 m. if the building was not correctly estimated").

:func:`ipin_score` applies both IPIN penalties (on single-building data the building term
never applies). The floor term counts floors of difference, as in Potortì's worked example.
Every constant is a keyword argument.

References
----------
F. Potortì, S. Park, A. R. Jiménez Ruiz, P. Barsocchi, M. Girolami, A. Crivello, S. Y. Lee,
J. H. Lim, J. Torres-Sospedra, F. Seco, R. Montoliu, G. M. Mendoza-Silva, M. C. Pérez Rubio,
C. Losada-Gutiérrez, F. Espinosa, J. Macias-Guarasa, "Comparing the Performance of Indoor
Localization Systems through the EvAAL Framework", Sensors 17(10):2327, 2017.
DOI: 10.3390/s17102327
J. Torres-Sospedra, A. R. Jiménez, A. Moreira, T. Lungenstrass, W.-C. Lu, S. Knauth,
G. M. Mendoza-Silva, F. Seco, A. Pérez-Navarro, M. J. Nicolau, A. Costa, F. Meneses, J. Farina,
J. P. Morales, W.-C. Lu, H.-T. Cheng, S.-S. Yang, S.-H. Fang, Y.-R. Chien, Y. Tsao, "Off-Line
Evaluation of Mobile-Centric Indoor Positioning Systems: The Experiences from the 2017 IPIN
Competition", Sensors 18(2):487, 2018. DOI: 10.3390/s18020487
J. Torres-Sospedra, A. Moreira, S. Knauth, R. Berkvens, R. Montoliu, O. Belmonte,
S. Trilles, M. J. Nicolau, F. Meneses, A. Costa, A. Koukofikis, M. Weyn, H. Peremans,
"A realistic evaluation of indoor positioning systems based on Wi-Fi fingerprinting: The 2015
EvAAL-ETRI competition", Journal of Ambient Intelligence and Smart Environments 9(2):263-279,
2017. DOI: 10.3233/AIS-170421
ISO/IEC 18305:2016, "Information technology — Real time locating systems — Test and
evaluation of localization and tracking systems". URL: https://www.iso.org/standard/62090.html
(CEP and percentile-based accuracy metrics).
B. Efron, R. J. Tibshirani, "An Introduction to the Bootstrap", Chapman & Hall, 1993.
DOI: 10.1007/978-1-4899-4541-9 (percentile bootstrap intervals).
"""
from __future__ import annotations

import numpy as np

from ..core import Prediction, SampleTable
from ..core.table import as_labels
from .functional import _check_arguments, position_errors

IPIN_FLOOR_PENALTY = 15.0  # metres per floor of difference (Potortì et al. 2017, Section 3.2)
EVAAL_ETRI_FLOOR_PENALTY = 4.0  # metres per floor (Torres-Sospedra et al. 2017, Section 3.1)
BUILDING_PENALTY = 50.0  # metres for a wrong building (IPIN 2017 off-site track; EvAAL-ETRI 2015, Section 3.1)


def _unpack(y_true, y_pred, floor_true, floor_pred, building_true, building_pred):
    _check_arguments(y_true, y_pred, "penalized_errors")
    if isinstance(y_true, SampleTable) and isinstance(y_pred, Prediction) and y_pred.ids is not None:
        if not np.array_equal(y_true.ids, y_pred.ids):
            raise ValueError("y_true and y_pred carry different ids: their rows are not aligned")
    if isinstance(y_true, SampleTable):
        floor_true = y_true.floor if floor_true is None else floor_true
        building_true = y_true.building if building_true is None else building_true
        y_true = y_true.pos
    if isinstance(y_pred, Prediction):
        floor_pred = y_pred.floor if floor_pred is None else floor_pred
        building_pred = y_pred.building if building_pred is None else building_pred
        y_pred = y_pred.pos
    return y_true, y_pred, floor_true, floor_pred, building_true, building_pred


def _labels(true, pred, name: str, n: int):
    """Both label columns, or (None, None). One side only is an error: the penalty is undefined."""
    if true is None and pred is None:
        return None, None
    if true is None or pred is None:
        side = "predictions" if pred is None else "ground truth"
        raise ValueError(f"{name} labels are missing from the {side}; pass both {name}_true and {name}_pred, "
                         f"or set {name}_penalty=0 if the {name} term does not apply")
    return as_labels(true, n, f"{name}_true"), as_labels(pred, n, f"{name}_pred")  # int64; NaN/fractions refused


def penalized_errors(y_true, y_pred, *, floor_true=None, floor_pred=None, building_true=None,
                     building_pred=None, floor_penalty: float = IPIN_FLOOR_PENALTY,
                     building_penalty: float = BUILDING_PENALTY, horizontal_axes: int = 2,
                     scale: float = 1.0) -> np.ndarray:
    """Per-sample floor-aware error: horizontal distance + floor and building penalties.

    y_true, y_pred    (N, D) float coordinates (or a SampleTable / Prediction).
    floor_*           (N,) int floor labels; the penalty is ``floor_penalty * |f - f̂|``, so
                      negative floors count like any other floor. Omit both (or set
                      ``floor_penalty=0``) to skip it; one side alone is an error.
    building_*        (N,) int building labels; ``building_penalty`` is added when they differ.
    horizontal_axes   number of leading coordinate axes in the horizontal distance (2: x, y;
                      the competitions ignore height and score floors through the penalty).
    scale             multiplies the horizontal distance before the penalties are added, e.g.
                      ``meta["ground_scale"]`` so that EPSG:3857 errors and metre penalties agree.

    Returns (N,) float64 errors in the units of ``scale * coordinates`` (metres), NaN where the
    estimate is NaN (a sample the method could not place). A NaN in the ground truth's
    horizontal coordinates is an error, as in :func:`evaluate`: select the labelled rows first.
    """
    y_true, y_pred, floor_true, floor_pred, building_true, building_pred = _unpack(
        y_true, y_pred, floor_true, floor_pred, building_true, building_pred)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} and y_pred {y_pred.shape} differ in shape")
    if horizontal_axes < 1:
        raise ValueError(f"horizontal_axes must be >= 1, got {horizontal_axes}")
    n = len(y_true)
    truth = y_true[:, :horizontal_axes]
    if not np.all(np.isfinite(truth)):
        bad = int(np.count_nonzero(~np.all(np.isfinite(truth), axis=1)))
        raise ValueError(f"y_true has NaN or infinite positions in {bad} of {n} rows; a score needs ground truth "
                         "(select the labelled rows, e.g. table[np.isfinite(table.pos).all(1)])")
    err = position_errors(truth, y_pred[:, :horizontal_axes]) * float(scale)
    if floor_penalty:
        ft, fp = _labels(floor_true, floor_pred, "floor", n)
        if ft is not None:
            err = err + float(floor_penalty) * np.abs(ft - fp)
    if building_penalty:
        bt, bp = _labels(building_true, building_pred, "building", n)
        if bt is not None:
            err = err + float(building_penalty) * (bt != bp)
    return err


def _require_placed(err: np.ndarray, fn: str) -> np.ndarray:
    """The penalized errors of a competition score, which needs every sample placed."""
    failed = int(np.count_nonzero(~np.isfinite(err)))
    if failed:
        raise ValueError(f"{fn}: {failed} of {len(err)} samples were not placed (NaN estimate; evaluate() counts them "
                         f"in n_failed={failed}). The competition rules have no 'not placed' outcome: score the "
                         f"placed rows and report n_failed next to the score, e.g. placed = "
                         f"np.isfinite(pred.pos).all(axis=1); {fn}(test[placed], pred[placed])")
    return err


def ipin_score(y_true, y_pred, *, floor_true=None, floor_pred=None, building_true=None, building_pred=None,
               floor_penalty: float = IPIN_FLOOR_PENALTY, building_penalty: float = BUILDING_PENALTY,
               percentile: float = 75.0, horizontal_axes: int = 2, scale: float = 1.0,
               method: str = "linear") -> float:
    """IPIN / EvAAL accuracy score: the 75th percentile of the floor-aware error (metres).

    Lower is better. Arguments as in :func:`penalized_errors`; ``percentile`` and the
    ``np.percentile`` interpolation ``method`` ("linear" = numpy's default, the same as
    ``EvaluationResults.p75_error``) are exposed for other competition rules.

    Known result (Potortì et al. 2017, Section 3.2): an x-y error of 4 m with the estimate
    on floor 2 instead of 0 scores 4 + 2 * 15 = 34 m. The 50 m building penalty is the IPIN
    off-site track rule (Torres-Sospedra et al. 2018); pass ``building_penalty=0`` for the
    single-building on-site tracks of Potortì et al. 2017.

    Every sample must be placed: a NaN estimate is a ValueError that gives ``n_failed`` and
    how to score the placed rows (as :func:`evaal_etri_score`).
    """
    _check_arguments(y_true, y_pred, "ipin_score")
    err = penalized_errors(y_true, y_pred, floor_true=floor_true, floor_pred=floor_pred,
                           building_true=building_true, building_pred=building_pred,
                           floor_penalty=floor_penalty, building_penalty=building_penalty,
                           horizontal_axes=horizontal_axes, scale=scale)
    return percentile_error(_require_placed(err, "ipin_score"), percentile, method=method)


def evaal_etri_score(y_true, y_pred, *, floor_true=None, floor_pred=None, building_true=None,
                     building_pred=None, horizontal_axes: int = 2, scale: float = 1.0) -> float:
    """EvAAL-ETRI 2015 score: mean of (x-y error + 4 m per floor + 50 m for a wrong building).

    The rule of the UJIIndoorLoc-based off-site competition track (Torres-Sospedra et al.
    2017, Section 3.1: "we add 4 m. for each wrong floor (absolute difference between the real
    and estimated floors) and 50 m. if the building was not correctly estimated").
    Every sample must be placed: a NaN estimate is a ValueError that gives ``n_failed`` and
    how to score the placed rows (as :func:`ipin_score`).
    """
    _check_arguments(y_true, y_pred, "evaal_etri_score")
    err = penalized_errors(y_true, y_pred, floor_true=floor_true, floor_pred=floor_pred,
                           building_true=building_true, building_pred=building_pred,
                           floor_penalty=EVAAL_ETRI_FLOOR_PENALTY, building_penalty=BUILDING_PENALTY,
                           horizontal_axes=horizontal_axes, scale=scale)
    return float(_require_placed(err, "evaal_etri_score").mean())


# --------------------------------------------------------------------------- error statistics
def _errors(errors) -> np.ndarray:
    e = np.asarray(errors, dtype=np.float64).ravel()
    if len(e) == 0:
        raise ValueError("no errors given")
    if not np.all(np.isfinite(e)) or np.any(e < 0):
        raise ValueError("errors must be finite and non-negative")
    return e


def percentile_error(errors, q: float, *, method: str = "linear") -> float:
    """The ``q``-th percentile (0-100) of an error sample, ``np.percentile`` semantics.

    ``method="linear"`` (numpy's default, Hyndman & Fan type 7) matches ``evaluate``;
    ``method="hazen"`` (type 5) reproduces MATLAB's ``prctile``.
    """
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    return float(np.percentile(_errors(errors), q, method=method))


def cep(errors, p: float = 50.0, *, method: str = "linear") -> float:
    """Circular error probable: the radius holding ``p`` % of the horizontal errors.

    Empirical definition (ISO/IEC 18305): the ``p``-th percentile of the per-sample
    horizontal error, so ``cep(e)`` is the median error and ``cep(e, 95)`` is often called
    R95 / CEP95. Pass horizontal (2-D) errors to match the name.
    """
    return percentile_error(errors, p, method=method)


def success_rate(errors, threshold: float) -> float:
    """Percentage of estimates with error ``<= threshold`` (a point of the empirical CDF)."""
    e = _errors(errors)
    return float(np.count_nonzero(e <= threshold) * 100.0 / len(e))


_STATISTICS = {
    "mean": lambda e, axis: e.mean(axis=axis),
    "median": lambda e, axis: np.median(e, axis=axis),
    "rmse": lambda e, axis: np.sqrt(np.mean(e * e, axis=axis)),
    "p75": lambda e, axis: np.percentile(e, 75, axis=axis),
    "p90": lambda e, axis: np.percentile(e, 90, axis=axis),
    "p95": lambda e, axis: np.percentile(e, 95, axis=axis),
}


def bootstrap_ci(errors, statistic: str = "mean", *, confidence: float = 0.95, n_resamples: int = 2000,
                 random_state=0, batch_size: int = 200) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval of an error statistic.

    statistic     "mean", "median", "rmse", "p75", "p90" or "p95", or a callable
                  ``f(e, axis) -> array`` reducing along ``axis``.
    confidence    two-sided level, e.g. 0.95 for the 2.5 / 97.5 percentiles of the
                  bootstrap distribution (Efron & Tibshirani 1993, Section 13.3).
    Resamples are drawn with ``np.random.default_rng(random_state)``, one call per
    resample, and reduced in batches of ``batch_size`` (memory ``batch_size * N`` values),
    so the result is deterministic and independent of ``batch_size``.

    Use it to report ``mean 8.81 m [95 % CI 8.0, 9.7]`` instead of a bare number: two
    methods whose intervals overlap on one test set are not separated by it.
    """
    e = _errors(errors)
    fn = _STATISTICS.get(statistic) if isinstance(statistic, str) else statistic
    if fn is None:
        raise ValueError(f"unknown statistic {statistic!r}; use one of {sorted(_STATISTICS)} or a callable")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples < 1 or batch_size < 1:
        raise ValueError("n_resamples and batch_size must be >= 1")
    rng = np.random.default_rng(random_state)
    stats = []
    for start in range(0, n_resamples, batch_size):
        # one generator call per resample, so the draws do not depend on how resamples are batched
        rows = np.stack([rng.integers(0, len(e), size=len(e)) for _ in range(min(batch_size, n_resamples - start))])
        stats.append(np.asarray(fn(e[rows], 1), dtype=np.float64))
    stats = np.concatenate(stats)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.percentile(stats, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return float(lo), float(hi)


__all__ = ["BUILDING_PENALTY", "EVAAL_ETRI_FLOOR_PENALTY", "IPIN_FLOOR_PENALTY", "bootstrap_ci", "cep",
           "evaal_etri_score", "ipin_score", "penalized_errors", "percentile_error", "success_rate"]
