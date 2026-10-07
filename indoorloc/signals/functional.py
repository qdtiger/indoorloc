"""Pure RSSI functions. Each takes one signal ``(F,)`` or a batch ``(N, F)``.

Float32 input stays float32 (integers become float32), so a pipeline built from
these functions is bit-identical whether it runs on one row or on a whole table.
Missing readings are NaN on the way in; each function states what it does with them.

Contents
    missing values   ``missing_mask``, ``fill_missing``
    scaling          ``minmax_scale``
    power units      ``dbm_to_mw``, ``mw_to_dbm``
    aggregation      ``aggregate_rssi`` (several scans -> one), ``aggregate_rssi_groups``
    representations  ``positive``, ``exponential``, ``powed`` (Torres-Sospedra et al., 2015)
    robust filter    ``hampel`` (Hampel identifier along any axis)

CSI, ranging and IMU helpers live in ``signals.csi``, ``signals.ranging`` and ``signals.imu``.

References
    J. Torres-Sospedra, R. Montoliu, S. Trilles, O. Belmonte, J. Huerta, "Comprehensive analysis of
    distance and similarity measures for Wi-Fi fingerprinting indoor positioning systems", Expert
    Systems with Applications 42(23):9263-9278, 2015. https://doi.org/10.1016/j.eswa.2015.08.013
    F. R. Hampel, "The influence curve and its role in robust estimation", Journal of the American
    Statistical Association 69(346):383-393, 1974. https://doi.org/10.1080/01621459.1974.10482962
    R. K. Pearson, "Outliers in process modeling and identification", IEEE Transactions on Control
    Systems Technology 10(1):55-63, 2002. https://doi.org/10.1109/87.974338
"""
from __future__ import annotations

import warnings

import numpy as np

RSSI_MIN_DBM = -104.0  # weakest reading in UJIIndoorLoc; the usual "not heard" floor
RSSI_MAX_DBM = 0.0
MAD_TO_SIGMA = 1.482602218505602  # 1 / Phi^-1(3/4): MAD -> standard deviation for Gaussian data


def _as_float(x) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype.kind == "O":  # e.g. a column of Python floats
        return x.astype(np.float64)
    return x.astype(x.dtype if x.dtype.kind in "fc" else np.float32, copy=True)


def _out_dtype(x: np.ndarray):
    return x.dtype if x.dtype.kind == "f" else np.float32


def _real(x: np.ndarray, who: str) -> np.ndarray:
    if x.dtype.kind == "c":
        raise ValueError(f"Complex data not supported by {who} (RSSI is real-valued); "
                         "convert CSI first, e.g. with CSIAmplitude")
    return x


def missing_mask(x, missing=np.nan) -> np.ndarray:
    """True where a reading is missing: NaN, or a file sentinel such as UJIIndoorLoc's 100."""
    x = np.asarray(x)
    return np.isnan(x) if missing is None or np.isnan(missing) else x == missing


def fill_missing(x, value: float = RSSI_MIN_DBM, missing=np.nan) -> np.ndarray:
    """Replace missing readings with ``value`` (default -104 dBm)."""
    out = _as_float(x)
    out[missing_mask(out, missing)] = value
    return out


def minmax_scale(x, lo: float = RSSI_MIN_DBM, hi: float = RSSI_MAX_DBM, clip: bool = False) -> np.ndarray:
    """``(x - lo) / (hi - lo)``: maps [lo, hi] dBm to [0, 1]; NaN stays NaN."""
    if not float(hi) > float(lo):
        raise ValueError(f"minmax_scale needs hi > lo, got lo={lo}, hi={hi}")
    out = (_as_float(x) - float(lo)) / (float(hi) - float(lo))
    return np.clip(out, 0.0, 1.0, out=out) if clip else out


# ----------------------------------------------------------------------------- power units
def dbm_to_mw(x) -> np.ndarray:
    """Power in milliwatts: ``10 ** (dBm / 10)``. NaN stays NaN."""
    x = _real(np.asarray(x), "dbm_to_mw")
    return np.power(10.0, x.astype(np.float64) / 10.0).astype(_out_dtype(x))


def mw_to_dbm(p) -> np.ndarray:
    """Power in dBm: ``10 log10(mW)``. Zero or negative power (nothing received) is NaN."""
    p = _real(np.asarray(p), "mw_to_dbm")
    p64 = p.astype(np.float64)
    out = np.full(p64.shape, np.nan)
    np.log10(p64, out=out, where=p64 > 0)
    return (10.0 * out).astype(_out_dtype(p))


# ----------------------------------------------------------------------------- aggregation
_AGGREGATES = ("mean", "median", "max", "power_mean")


def _check_method(method: str) -> None:
    if method not in _AGGREGATES:
        raise ValueError(f"method must be one of {_AGGREGATES}, got {method!r}")


def aggregate_rssi(x, method: str = "mean", axis: int = 0, min_count: int = 1) -> np.ndarray:
    """Combine several scans of the same transmitters into one reading per transmitter.

    NaN (not heard) is ignored. ``method``: ``"mean"`` (average in dBm, as radio maps
    usually do), ``"median"``, ``"max"`` or ``"power_mean"`` (average in mW, the mean
    received power, converted back to dBm). A transmitter heard in fewer than
    ``min_count`` scans is NaN in the result.
    """
    _check_method(method)
    x = _real(np.asarray(x), "aggregate_rssi")
    x64 = np.moveaxis(x.astype(np.float64), axis, 0)
    heard = ~np.isnan(x64)
    count = heard.sum(axis=0)
    if method in ("mean", "power_mean"):
        vals = np.power(10.0, x64 / 10.0) if method == "power_mean" else x64
        total = np.where(heard, vals, 0.0).sum(axis=0)
        out = np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)
        if method == "power_mean":
            out = 10.0 * np.log10(out)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns: NaN is the answer
            out = np.nanmedian(x64, axis=0) if method == "median" else np.nanmax(x64, axis=0)
    out = np.where(count >= max(int(min_count), 1), out, np.nan)
    return out.astype(_out_dtype(x))


def aggregate_rssi_groups(x, keys, method: str = "mean", min_count: int = 1):
    """Aggregate the rows of ``x`` (N, F) that share a key, e.g. repeated scans at one
    reference point (``keys=table.pos``) or one time window.

    Returns ``(unique_keys, aggregated (G, F), counts (G,))`` with groups in the sorted
    order of ``np.unique``; ``method`` and ``min_count`` as in ``aggregate_rssi``.
    """
    _check_method(method)
    x = _real(np.asarray(x), "aggregate_rssi_groups")
    if x.ndim != 2:
        raise ValueError(f"x must be (N, F), got shape {x.shape}")
    keys = np.asarray(keys)
    if len(keys) != len(x):
        raise ValueError(f"{len(keys)} keys for {len(x)} rows")
    uniq, inverse = np.unique(keys, axis=0 if keys.ndim > 1 else None, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(inverse, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(inverse[order]) != 0])
    counts = np.diff(np.r_[starts, len(order)])
    xs = x[order].astype(np.float64)
    heard = ~np.isnan(xs)
    n_heard = np.add.reduceat(heard.astype(np.int64), starts, axis=0)
    if method in ("mean", "power_mean"):
        vals = np.power(10.0, xs / 10.0) if method == "power_mean" else xs
        total = np.add.reduceat(np.where(heard, vals, 0.0), starts, axis=0)
        out = np.divide(total, n_heard, out=np.full(total.shape, np.nan), where=n_heard > 0)
        if method == "power_mean":
            out = 10.0 * np.log10(out)
    else:
        reduce = np.nanmedian if method == "median" else np.nanmax
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = np.stack([reduce(xs[s:s + c], axis=0) for s, c in zip(starts, counts)])
    out = np.where(n_heard >= max(int(min_count), 1), out, np.nan)
    return uniq, out.astype(_out_dtype(x)), counts


# ------------------------------------------------------------------- data representations
def positive(x, min_dbm: float) -> np.ndarray:
    """Positive representation (Torres-Sospedra et al., 2015): ``RSSI - min`` for heard
    readings, 0 for missing ones and for readings at or below ``min_dbm``.

    ``min_dbm`` is the paper's ``min``: the lowest RSSI of the training database minus 1,
    so the weakest real reading maps to 1 and "not heard" to 0 (UJIIndoorLoc: -105 dBm).
    """
    x = _real(_as_float(x), "positive")
    out = x - np.asarray(min_dbm, dtype=x.dtype)
    out[~(out > 0)] = 0  # NaN (not heard) and readings below min -> 0
    return out


def exponential(x, min_dbm: float, alpha: float = 24.0) -> np.ndarray:
    """Exponential representation: ``exp(positive / alpha) / exp(-min / alpha)``.

    Maps 0 dBm to 1 and a missing reading to ``exp(min / alpha)`` (it is ``positive``
    = 0 pushed through the same formula). A heard reading ``r`` above ``min`` maps to
    ``exp(r / alpha)``, whatever ``min`` is. Torres-Sospedra et al. (2015) use alpha = 24.
    """
    if not float(alpha) > 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    pos = positive(x, min_dbm)
    out = np.exp((pos.astype(np.float64) + float(min_dbm)) / float(alpha))  # = exp(pos/alpha) / exp(-min/alpha)
    return out.astype(pos.dtype)


def powed(x, min_dbm: float, beta: float = np.e) -> np.ndarray:
    """Powed representation: ``positive ** beta / (-min) ** beta``.

    Maps 0 dBm to 1 and a missing reading to 0. Torres-Sospedra et al. (2015) use beta = e.
    """
    if not float(min_dbm) < 0:
        raise ValueError(f"min_dbm must be negative (dBm), got {min_dbm}")
    if not float(beta) > 0:
        raise ValueError(f"beta must be positive, got {beta}")
    pos = positive(x, min_dbm)
    out = np.power(pos.astype(np.float64) / -float(min_dbm), float(beta))
    return out.astype(pos.dtype)


# ------------------------------------------------------------------------- Hampel filter
def _nanmedian_last(a: np.ndarray) -> np.ndarray:
    """``np.nanmedian(a, axis=-1)`` for a short last axis: sorting puts NaN last, so the median
    of the ``n`` valid values sits at positions ``(n - 1) // 2`` and ``n // 2``. All-NaN -> NaN."""
    s = np.sort(a, axis=-1)
    n = np.sum(~np.isnan(a), axis=-1, keepdims=True)
    lo = np.take_along_axis(s, np.maximum((n - 1) // 2, 0), axis=-1)
    hi = np.take_along_axis(s, np.minimum(n // 2, a.shape[-1] - 1), axis=-1)
    return ((lo + hi) / 2)[..., 0]


def hampel(x, window: int = 3, n_sigmas: float = 3.0, axis: int = -1, return_mask: bool = False):
    """Hampel identifier: replace outliers by the median of their moving window.

    For every sample ``x_i`` along ``axis`` the window holds the ``window`` neighbours on
    each side (``2 * window + 1`` samples, truncated at the edges). With the window median
    ``m_i`` and ``S_i = 1.4826 * MAD_i``, ``x_i`` is an outlier if
    ``|x_i - m_i| > n_sigmas * S_i`` and is replaced by ``m_i``. NaN is ignored inside the
    windows and stays NaN. ``return_mask=True`` also returns the boolean outlier mask.
    """
    out = _real(_as_float(x), "hampel")
    if isinstance(window, bool) or int(window) != window or window < 0:
        raise ValueError(f"window must be a non-negative integer (neighbours on each side), got {window!r}")
    k = int(window)
    mask = np.zeros(out.shape, dtype=bool)
    if k == 0 or out.size == 0:
        return (out, mask) if return_mask else out
    moved = np.moveaxis(out, axis, -1)  # views: writing to them writes to ``out``/``mask``
    moved_mask = np.moveaxis(mask, axis, -1)
    length = moved.shape[-1]
    rows = moved.reshape(-1, length)  # a copy if ``moved`` is not contiguous
    flags = np.zeros(rows.shape, dtype=bool)
    width = 2 * k + 1
    step = max(1, 4_000_000 // (length * width))  # bound the (rows, length, width) temporaries
    for start in range(0, len(rows), step):
        block = rows[start:start + step]
        padded = np.pad(block, ((0, 0), (k, k)), constant_values=np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(padded, width, axis=-1)
        med = _nanmedian_last(windows)
        mad = _nanmedian_last(np.abs(windows - med[..., None]))
        bad = np.abs(block - med) > float(n_sigmas) * MAD_TO_SIGMA * mad  # NaN compares False
        block[bad] = med[bad]
        flags[start:start + step] = bad
    moved[...] = rows.reshape(moved.shape)
    moved_mask[...] = flags.reshape(moved.shape)
    return (out, mask) if return_mask else out
