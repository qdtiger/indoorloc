"""fit/transform estimators over the functional ops (sklearn transformer contract).

Every RSSI transform accepts one scan ``(F,)``, a batch ``(N, F)``, a SampleTable or a
scan view (WiFiSignal, BLESignal), and returns the same kind of object. Statistics (if
any) are learned in ``fit`` on training data only and then frozen. A new transform
writes ``_transform``; one that learns from extra data declares it in ``fit``
(``fit(X, y=None, *, reference=None)``), so an unknown fit keyword is a TypeError,
never silently ignored. Transforms that drop or reorder columns keep
``meta["feature_names"]`` (and a view's ids) aligned with the columns.

This module: the base class, ``Compose``, missing values and scaling (``FillMissing``,
``RSSINormalize``, ``APFilter``), AP selection (``APSelect``), the data representations
of Torres-Sospedra et al. (2015) and the Hampel outlier filter. Augmentations are in
``signals.augment``, device calibration in ``signals.calibration``, CSI in ``signals.csi``.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

from ..core import Estimator, NotFittedError, SampleTable
from . import functional as F
from ._scan import ScanView


# 1 W: the most a WiFi / BLE transmitter may radiate (EIRP). A "received power" above it is a sentinel.
_IMPOSSIBLE_DBM = 30.0
_PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep  # .../indoorloc/


def _outside_level() -> int:
    """``stacklevel`` for a ``warnings.warn`` issued by the caller of this function that points
    at the first frame outside the indoorloc package (the user's line, however deep the pipeline)."""
    frame, level = sys._getframe(1), 1
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE):
        frame, level = frame.f_back, level + 1
    return level


def _features(obj) -> np.ndarray:
    if hasattr(obj, "toarray"):
        raise TypeError("sparse input is not supported; pass a dense array (X.toarray())")
    if isinstance(obj, SampleTable):
        return obj.X
    x = np.asarray(obj.rssi if isinstance(obj, ScanView) else obj)
    return x.astype(np.float64) if x.dtype.kind == "O" else x


def _apply(fn, obj):
    if isinstance(obj, SampleTable):
        return obj.replace(X=fn(obj.X))
    if isinstance(obj, ScanView):
        return obj.replace(rssi=fn(obj.rssi))
    return fn(_features(obj))


def _take_columns(obj, columns):
    """Keep the last-axis columns ``columns`` and the names that go with them."""
    columns = np.asarray(columns, dtype=np.intp)
    if isinstance(obj, ScanView):
        return obj.take(columns)
    if isinstance(obj, SampleTable):
        meta = dict(obj.meta)
        n = obj.X.shape[-1]
        for key in ("feature_names", "subcarriers"):  # per-column facts follow the columns
            values = meta.get(key)
            if values is not None and len(values) == n:
                meta[key] = values[columns] if isinstance(values, np.ndarray) else tuple(values[j] for j in columns)
        return obj.replace(X=obj.X[..., columns], meta=meta)
    return _features(obj)[..., columns]


class Transform(Estimator):
    """Base of every L2 transform: sklearn's transformer contract without sklearn.

    ``fit`` records ``n_features_in_`` (the size of the last axis); ``transform``
    applies ``_transform`` to the array inside whatever container it gets.
    ``t(X)`` is ``t.transform(X)`` (augmentations override it to draw a perturbation).
    Subclasses for RSSI set ``_real_only = True``: complex (CSI) input is then an error.
    """

    _estimator_type = "transformer"
    _allow_nan = True
    _requires_fit = False  # stateless unless a subclass learns something in fit
    _real_only = False

    def fit(self, X, y=None):
        x = _features(X)
        if self._real_only:
            F._real(x, type(self).__name__)
        if 0 in x.shape:
            what = "sample" if x.ndim > 1 and x.shape[0] == 0 else "feature"
            raise ValueError(f"Found array with 0 {what}(s) (shape={x.shape}) while a minimum of 1 is required.")
        self.n_features_in_ = x.shape[-1]
        return self

    def _check_features(self, X) -> None:
        n = getattr(self, "n_features_in_", None)
        if n is not None and _features(X).shape[-1] != n:
            raise ValueError(f"X has {_features(X).shape[-1]} features, but {type(self).__name__} "
                             f"is expecting {n} features as input")

    def transform(self, X):
        self._check_features(X)
        return _apply(self._transform, X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

    def __call__(self, X):
        return self.transform(X)

    def _transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class FillMissing(Transform):
    """Replace missing readings (NaN, or ``missing=`` e.g. 100) with ``value`` dBm.

    Distance-based methods need a number for "not heard"; the usual choice is a value
    just below the weakest real reading (-104 dBm in UJIIndoorLoc, the default), which
    ``PositiveRepresentation`` makes explicit. Stateless.

    ``missing`` is what marks a missing reading: NaN (the default, as every L1 loader
    delivers it) or the sentinel of a raw file, e.g. ``FillMissing(-104, missing=100)`` for
    UJIIndoorLoc's ``100``. With ``missing=NaN`` and a negative (dBm) ``value``, a real-valued
    input holding readings far above 0 dBm triggers a ``UserWarning``: such a reading is a raw
    sentinel left in place, which would otherwise pass through unfilled as a very strong signal.
    "Far above" is more than +30 dBm (1 W, the most a WiFi or BLE transmitter may radiate), so
    readings a few dB above 0 after ``DeviceCalibration`` or augmentation do not warn, while
    the usual sentinels (100, 127) do.

    References
        J. Torres-Sospedra, R. Montoliu, A. Martinez-Uso, J. P. Avariento, T. J. Arnau, M. Benedito-Bordonau,
        J. Huerta, "UJIIndoorLoc: a new multi-building and multi-floor database for WLAN fingerprint-based
        indoor localization problems", IPIN 2014, pp. 261-270 (weakest reading -104 dBm, "not detected"
        stored as 100). https://doi.org/10.1109/IPIN.2014.7275492
    """

    def __init__(self, value: float = F.RSSI_MIN_DBM, missing: float = np.nan):
        self.value = value
        self.missing = missing

    def _transform(self, x):
        self._warn_sentinel(x)
        return F.fill_missing(x, self.value, self.missing)

    def _warn_sentinel(self, x) -> None:
        """Warn when NaN is the missing marker but a dBm input still holds a raw sentinel."""
        try:
            value = float(self.value)
            if not (self.missing is None or np.isnan(float(self.missing))):
                return  # an explicit sentinel: the caller has said what "missing" is
        except (TypeError, ValueError):
            return
        x = np.asarray(x)
        if x.dtype.kind not in "iuf" or not value < 0:
            return  # complex CSI, or a non-dBm fill (e.g. 0 W for VLC): positive readings are real
        above = x > _IMPOSSIBLE_DBM  # NaN compares False
        if above.any():
            top = float(np.max(x[above]))
            warnings.warn(f"FillMissing(missing=NaN) got readings far above 0 dBm (max {top:g}; no received "
                          f"power exceeds +{_IMPOSSIBLE_DBM:g} dBm): a raw 'not detected' sentinel left in place? "
                          f"They are kept as they are; pass it as missing=, e.g. "
                          f"FillMissing({value:g}, missing=100) for UJIIndoorLoc's 100", UserWarning,
                          stacklevel=_outside_level())


class RSSINormalize(Transform):
    """Min-max scale RSSI to [0, 1]. ``lo``/``hi`` = None learns them from the training data.

    Readings above ``hi`` are an error unless ``clip=True``: RSSI above 0 dBm is not
    physical, so this is almost always an unconverted file sentinel (UJIIndoorLoc's 100).
    With ``lo = min`` and ``hi = 0`` after ``FillMissing(min)`` it is the "normalized"
    representation of Torres-Sospedra et al. (2015).

    References
        J. Torres-Sospedra, R. Montoliu, S. Trilles, O. Belmonte, J. Huerta, "Comprehensive analysis
        of distance and similarity measures for Wi-Fi fingerprinting indoor positioning systems",
        Expert Systems with Applications 42(23):9263-9278, 2015. https://doi.org/10.1016/j.eswa.2015.08.013
    """

    _real_only = True

    def __init__(self, lo: float | None = F.RSSI_MIN_DBM, hi: float | None = F.RSSI_MAX_DBM,
                 clip: bool = False):
        self.lo = lo
        self.hi = hi
        self.clip = clip

    @property
    def _requires_fit(self) -> bool:
        return self.lo is None or self.hi is None

    def fit(self, X, y=None):
        x = F._real(_features(X), "RSSINormalize")
        super().fit(X, y)  # validates shape
        self.lo_ = float(np.nanmin(x)) if self.lo is None else float(self.lo)
        self.hi_ = float(np.nanmax(x)) if self.hi is None else float(self.hi)
        self._check_range(self.lo_, self.hi_)
        return self

    def _check_range(self, lo, hi) -> None:
        if not float(hi) > float(lo):
            learned = " (learned from training data that hold a single value)" if self._requires_fit else ""
            raise ValueError(f"RSSINormalize needs hi > lo, got lo={lo}, hi={hi}{learned}")

    def _transform(self, x):
        F._real(x, "RSSINormalize")
        if self._requires_fit:
            if not hasattr(self, "lo_"):
                raise NotFittedError("RSSINormalize(lo=None or hi=None) must be fitted first")
            lo, hi = self.lo_, self.hi_
        else:
            lo, hi = self.lo, self.hi  # a fixed physical range needs no fit
        self._check_range(lo, hi)
        top = np.max(x, initial=-np.inf, where=~np.isnan(x)) if self.hi is not None else -np.inf
        if not self.clip and top > hi:  # only a fixed physical ceiling is checked
            raise ValueError(f"readings above hi={hi} (max {top:g}): a raw 'not detected' "
                             "sentinel? convert it first, e.g. FillMissing(missing=100), or pass clip=True")
        return F.minmax_scale(x, lo, hi, self.clip)


class Compose(Transform):
    """Apply transforms in order (torchvision's name for an sklearn Pipeline of transformers).

    ``fit``/``fit_transform`` fit each step on the output of the previous one. Fit keywords
    go to the steps whose ``fit`` declares them (a nested ``Compose`` declares those of its
    steps; a ``fit(**kwargs)`` step receives every keyword). Keywords that carry samples in
    the layout of ``X`` -- ``target`` (unlabelled data of another domain, for CORAL/TCA) and
    ``reference`` (the reference device's scans, for ``DeviceCalibration``) -- first run
    through the steps before the one that takes them, so
    ``Compose([FillMissing(-104), CORAL()]).fit(X_src, target=X_tgt_raw)`` fills both domains.
    An unknown keyword is a TypeError.
    ``transform`` chains the steps' ``transform`` (inference: augmentations pass data
    through); ``compose(X)`` chains the steps' ``__call__`` (augmentations draw, as in a
    torchvision training pipeline); ``fit_transform`` returns the training data as the
    fitted steps produced it (augmented, if the chain augments). Like sklearn's Pipeline,
    the steps are fitted in place (``LocalizerPipeline`` fits a clone).

    References
        L. Buitinck et al., "API design for machine learning software: experiences from the
        scikit-learn project", ECML PKDD Workshop on Languages for Data Mining and Machine Learning,
        2013, pp. 108-122. https://arxiv.org/abs/1309.0238
    """

    _ROUTED = ("target", "reference")  # fit keywords holding samples shaped like X

    def __init__(self, transforms=()):
        self.transforms = transforms

    @property
    def _requires_fit(self) -> bool:
        return any(getattr(t, "_requires_fit", False) for t in self.transforms)

    @staticmethod
    def _fit_keywords(step):
        """Names of the fit keywords ``step`` takes; None if it takes any (``**kwargs``)."""
        if isinstance(step, Compose):
            names = [Compose._fit_keywords(t) for t in step.transforms]
            return None if any(n is None for n in names) else set().union(*names)
        import inspect

        params = list(inspect.signature(step.fit).parameters.values())
        if any(p.kind is p.VAR_KEYWORD for p in params):
            return None
        named = [p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
        return set(named[1:]) - {"y"}  # the first parameter is X

    def fit(self, X, y=None, **fit_params):
        self.fit_transform(X, y, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        n_features = _features(X).shape[-1]
        accepted = [self._fit_keywords(t) for t in self.transforms]
        if all(names is not None for names in accepted):
            unused = set(fit_params) - set().union(*accepted)
            if unused:
                raise TypeError(f"no step of this Compose takes the fit parameter(s) {sorted(unused)}")
        routed = {k: fit_params[k] for k in self._ROUTED if fit_params.get(k) is not None}
        for i, (t, names) in enumerate(zip(self.transforms, accepted)):
            kwargs = {k: v for k, v in fit_params.items() if names is None or k in names}
            kwargs.update({k: v for k, v in routed.items() if k in kwargs})
            X = t.fit_transform(X, y, **kwargs)
            for k in list(routed):  # later steps see these samples as they see X
                later = accepted[i + 1:]
                if any(names is None or k in names for names in later):
                    routed[k] = t.transform(routed[k])
                else:
                    del routed[k]
        self.n_features_in_ = n_features  # fitted only once every step is
        return X

    def transform(self, X):
        for t in self.transforms:
            X = t.transform(X)
        return X

    def __call__(self, X):
        for t in self.transforms:
            X = t(X)
        return X


class APFilter(Transform):
    """Treat weak readings as not heard: RSSI below ``threshold_dbm`` becomes NaN.

    Readings equal to the threshold are kept. Weak readings are the least reliable
    (their variance is highest and they come and go between scans); dropping them is
    the thresholding studied by Torres-Sospedra et al. (2015). Stateless.
    ``threshold_dbm`` is negative (e.g. -90); a positive one, usually a dropped minus sign
    (``APFilter(95)``), would drop every reading and triggers a ``UserWarning``.

    References
        J. Torres-Sospedra, R. Montoliu, S. Trilles, O. Belmonte, J. Huerta, "Comprehensive analysis
        of distance and similarity measures for Wi-Fi fingerprinting indoor positioning systems",
        Expert Systems with Applications 42(23):9263-9278, 2015. https://doi.org/10.1016/j.eswa.2015.08.013
    """

    _real_only = True

    def __init__(self, threshold_dbm: float = -90.0):
        self.threshold_dbm = threshold_dbm

    def _transform(self, x):
        out = F._real(F._as_float(x), "APFilter")
        threshold = float(self.threshold_dbm)
        if threshold > 0:  # RSSI is negative dBm: every reading would be dropped
            warnings.warn(f"APFilter(threshold_dbm={threshold:g}) is positive, so every real reading (RSSI is "
                          f"below 0 dBm) is dropped; did you mean APFilter({-threshold:g})?", UserWarning,
                          stacklevel=_outside_level())
        out[out < threshold] = np.nan
        return out


class APSelect(Transform):
    """Keep the ``k`` most useful access points (columns), chosen on the training data.

    ``strategy``
        ``"variance"``   highest variance of the readings, a missing reading counting as
                         ``fill_value`` (so an AP heard in some places only varies most);
        ``"coverage"``   heard in the largest fraction of training scans;
        ``"strongest"``  highest mean reading, missing = ``fill_value`` ("MaxMean",
                         Youssef et al. 2003, the baseline of Chen et al. 2006).

    Ties are broken by column index; kept columns stay in their original order, and
    ``meta["feature_names"]`` / a view's ids follow them. Learned: ``scores_`` (F,),
    ``indices_`` (k,) sorted column positions; ``get_support()`` as in sklearn.

    References
        M. Youssef, A. Agrawala, A. U. Shankar, "WLAN location determination via clustering and
        probability distributions", IEEE PerCom 2003, pp. 143-150. https://doi.org/10.1109/PERCOM.2003.1192736
        Y. Chen, Q. Yang, J. Yin, X. Chai, "Power-efficient access-point selection for indoor location
        estimation", IEEE Transactions on Knowledge and Data Engineering 18(7):877-888, 2006.
        https://doi.org/10.1109/TKDE.2006.112
        A. Kushki, K. N. Plataniotis, A. N. Venetsanopoulos, "Kernel-based positioning in wireless local
        area networks", IEEE Transactions on Mobile Computing 6(6):689-705, 2007.
        https://doi.org/10.1109/TMC.2007.1017
    """

    _requires_fit = True
    _real_only = True
    _strategies = ("variance", "coverage", "strongest")

    def __init__(self, k: int = 100, strategy: str = "variance", fill_value: float = F.RSSI_MIN_DBM):
        self.k = k
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X, y=None):
        if self.strategy not in self._strategies:
            raise ValueError(f"strategy must be one of {self._strategies}, got {self.strategy!r}")
        x = F._real(_features(X), "APSelect")
        super().fit(X, y)
        n = x.shape[-1]
        if not (isinstance(self.k, (int, np.integer)) and 1 <= self.k <= n):
            raise ValueError(f"k must be an integer in [1, {n}] (the number of features), got {self.k!r}")
        x = x.reshape(-1, n).astype(np.float64)
        heard = ~np.isnan(x)
        if self.strategy == "coverage":
            scores = heard.mean(axis=0)
        else:
            filled = np.where(heard, x, float(self.fill_value))
            scores = filled.var(axis=0) if self.strategy == "variance" else filled.mean(axis=0)
        best = np.argsort(-scores, kind="stable")[: int(self.k)]  # ties: lower column index first
        self.scores_ = scores
        self.indices_ = np.sort(best)
        return self

    def transform(self, X):
        self._check_fitted("indices_")
        self._check_features(X)
        return _take_columns(X, self.indices_)

    def get_support(self, indices: bool = False) -> np.ndarray:
        self._check_fitted("indices_")
        if indices:
            return self.indices_.copy()
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.indices_] = True
        return mask

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        self._check_fitted("indices_")
        names = (np.asarray(input_features, dtype=object) if input_features is not None
                 else np.asarray([f"x{j}" for j in range(self.n_features_in_)], dtype=object))
        return names[self.indices_]


class _Representation(Transform):
    """Shared ``min_dbm`` handling: None learns the paper's ``min`` (lowest training reading - 1)."""

    _real_only = True

    def __init__(self, min_dbm: float | None = None):
        self.min_dbm = min_dbm

    @property
    def _requires_fit(self) -> bool:
        return self.min_dbm is None

    def fit(self, X, y=None):
        x = F._real(_features(X), type(self).__name__)
        super().fit(X, y)
        if self.min_dbm is None:
            if np.isnan(x).all():
                raise ValueError("cannot learn min_dbm: every training reading is missing")
            self.min_dbm_ = float(np.nanmin(x)) - 1.0
        else:
            self.min_dbm_ = float(self.min_dbm)
        return self

    def _min(self) -> float:
        if self.min_dbm is not None:
            return float(self.min_dbm)
        if not hasattr(self, "min_dbm_"):
            raise NotFittedError(f"{type(self).__name__}(min_dbm=None) learns min_dbm from training data; "
                                 "call fit first or pass min_dbm")
        return self.min_dbm_


class PositiveRepresentation(_Representation):
    """Positive RSSI representation: ``RSSI - min`` for heard readings, 0 otherwise.

    The linear baseline of Torres-Sospedra et al. (2015). ``min`` is the lowest RSSI in
    the training data minus 1 dB (``min_dbm=None``, learned in ``fit`` as ``min_dbm_``), or
    a fixed value. Missing readings and readings at or below ``min`` become 0, so the
    output has no NaN. Used with k-NN, the representation changes the distances; the
    paper reports that the exponential and powed ones beat it on UJIIndoorLoc. The
    paper's fourth, "normalized" representation ``positive / (-min)`` is
    ``Compose([FillMissing(min), RSSINormalize(lo=min, hi=0, clip=True)])``.

    References
        J. Torres-Sospedra, R. Montoliu, S. Trilles, O. Belmonte, J. Huerta, "Comprehensive analysis
        of distance and similarity measures for Wi-Fi fingerprinting indoor positioning systems",
        Expert Systems with Applications 42(23):9263-9278, 2015. https://doi.org/10.1016/j.eswa.2015.08.013
        G. G. Anagnostopoulos, A. Kalousis, "A reproducible analysis of RSSI fingerprinting for outdoor
        localization using Sigfox: preprocessing and hyperparameter tuning", IPIN 2019 (states the
        ``min`` = lowest value minus 1 convention and that it is computed on training data only).
        https://doi.org/10.1109/IPIN.2019.8911792
    """

    def _transform(self, x):
        return F.positive(x, self._min())


class ExponentialRepresentation(_Representation):
    """Exponential RSSI representation: ``exp(positive / alpha) / exp(-min / alpha)``.

    Emphasises strong readings (RSSI is logarithmic in power); 0 dBm maps to 1 and a
    missing reading to ``exp(min / alpha)``. ``alpha = 24`` as proposed by
    Torres-Sospedra et al. (2015); ``min`` as in ``PositiveRepresentation``.

    References
        J. Torres-Sospedra, R. Montoliu, S. Trilles, O. Belmonte, J. Huerta, "Comprehensive analysis
        of distance and similarity measures for Wi-Fi fingerprinting indoor positioning systems",
        Expert Systems with Applications 42(23):9263-9278, 2015. https://doi.org/10.1016/j.eswa.2015.08.013
    """

    def __init__(self, alpha: float = 24.0, min_dbm: float | None = None):
        self.alpha = alpha
        self.min_dbm = min_dbm

    def _transform(self, x):
        return F.exponential(x, self._min(), self.alpha)


class PowedRepresentation(_Representation):
    """Powed RSSI representation: ``positive ** beta / (-min) ** beta``.

    0 dBm maps to 1 and a missing reading to 0; ``beta = e`` as proposed by
    Torres-Sospedra et al. (2015); ``min`` as in ``PositiveRepresentation``.

    References
        J. Torres-Sospedra, R. Montoliu, S. Trilles, O. Belmonte, J. Huerta, "Comprehensive analysis
        of distance and similarity measures for Wi-Fi fingerprinting indoor positioning systems",
        Expert Systems with Applications 42(23):9263-9278, 2015. https://doi.org/10.1016/j.eswa.2015.08.013
    """

    def __init__(self, beta: float = np.e, min_dbm: float | None = None):
        self.beta = beta
        self.min_dbm = min_dbm

    def _transform(self, x):
        return F.powed(x, self._min(), self.beta)


class HampelFilter(Transform):
    """Moving-window Hampel filter: outliers along ``axis`` are replaced by the window median.

    Each value is compared with the median ``m`` of the ``2 * window + 1`` values around
    it (truncated at the edges); if it differs by more than ``n_sigmas`` robust standard
    deviations (``1.4826 * MAD``) it becomes ``m``. NaN stays NaN. Stateless.

    ``axis`` refers to a batch ``(N, ...)`` and must run along an *ordered* dimension:
    ``-1`` (default) filters across the last axis of each sample on its own, which suits
    CSI amplitude over subcarriers but not RSSI, whose AP columns have no order; ``0``
    filters across samples, i.e. over time when rows are time steps (an RSSI or CSI
    stream). With ``axis=0`` a SampleTable is filtered one ``groups["trajectory"]`` at a
    time, rows sorted by ``groups["time"]`` when present (the output keeps the input row
    order); a plain array must already be in time order. ``axis=0`` couples the rows of a
    batch, so it belongs before a model on a stream, not in a ``LocalizerPipeline`` that
    receives unrelated queries. One sample ``(F,)`` is a batch of one.

    References
        F. R. Hampel, "The influence curve and its role in robust estimation", Journal of the
        American Statistical Association 69(346):383-393, 1974. https://doi.org/10.1080/01621459.1974.10482962
        R. K. Pearson, "Outliers in process modeling and identification", IEEE Transactions on Control
        Systems Technology 10(1):55-63, 2002. https://doi.org/10.1109/87.974338
        R. K. Pearson, Y. Neuvo, J. Astola, M. Gabbouj, "Generalized Hampel filters", EURASIP Journal on
        Advances in Signal Processing 2016:87, 2016. https://doi.org/10.1186/s13634-016-0383-6
    """

    _real_only = True

    def __init__(self, window: int = 3, n_sigmas: float = 3.0, axis: int = -1):
        self.window = window
        self.n_sigmas = n_sigmas
        self.axis = axis

    def _transform(self, x):
        batch = x[None] if x.ndim == 1 else x  # axes refer to (N, ...)
        out = F.hampel(batch, self.window, self.n_sigmas, self.axis)
        return out[0] if x.ndim == 1 else out

    def transform(self, X):
        self._check_features(X)
        if (isinstance(X, SampleTable) and self._along_samples(X.X.ndim)
                and ("trajectory" in X.groups or "time" in X.groups)):
            order, starts = _series_order(X.groups)
            out = np.empty(X.X.shape, dtype=F._out_dtype(X.X))
            for start, stop in zip(starts[:-1], starts[1:]):
                rows = order[start:stop]
                out[rows] = self._transform(X.X[rows])
            return X.replace(X=out)
        return _apply(self._transform, X)

    def _along_samples(self, ndim: int) -> bool:
        return self.axis in (0, -ndim)


def _series_order(groups) -> tuple[np.ndarray, np.ndarray]:
    """Rows grouped by ``groups["trajectory"]`` and sorted by ``groups["time"]`` inside each
    (either may be absent). Returns ``(order, starts)``: series ``s`` is
    ``order[starts[s]:starts[s + 1]]``. Stable, so ties keep the table order."""
    traj, time = groups.get("trajectory"), groups.get("time")
    n = len(traj if traj is not None else time)
    codes = np.zeros(n, dtype=np.intp) if traj is None else np.unique(traj, return_inverse=True)[1].reshape(-1)
    keys = [codes] if time is None else [np.asarray(time), codes]  # lexsort: the last key is the primary one
    order = np.lexsort(keys)
    sorted_codes = codes[order]
    starts = np.r_[0, np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1, n]
    return order, starts
