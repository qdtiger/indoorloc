"""Inertial (IMU) helpers for one time series: magnitude, smoothing, gravity removal.

Input is one trajectory ``(T, C)`` (rows = time steps, CONTRACTS.md ``imu`` layout) or
``(T,)``; for a table with several trajectories apply them per ``groups["trajectory"]``.
Step detection, heading and PDR are L5 (``apps``); these are the signal-level pieces.
"""
from __future__ import annotations

import numpy as np

GRAVITY = 9.80665  # m/s^2, standard gravity (CGPM 1901)


def _float(x) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype.kind == "c":
        raise ValueError("IMU data are real-valued")
    return x.astype(x.dtype if x.dtype.kind == "f" else np.float64, copy=True)


def magnitude(x, axis: int = -1) -> np.ndarray:
    """Euclidean norm along ``axis``, e.g. ``(T, 3)`` accelerometer -> ``(T,)`` in m/s^2.
    Orientation-free, hence the usual input of step detectors."""
    return np.linalg.norm(_float(x), axis=axis)


def moving_average(x, window: int, axis: int = 0, centered: bool = True) -> np.ndarray:
    """Mean over ``window`` samples along ``axis``; NaN ignored, windows truncated at the edges.

    ``centered=True`` averages ``window // 2`` samples on each side (zero phase, needs an
    odd ``window``); ``centered=False`` averages the current and ``window - 1`` previous
    samples (causal, usable on a stream).
    """
    w = int(window)
    if w < 1 or (centered and w % 2 == 0):
        raise ValueError(f"window must be a positive{' odd' if centered else ''} integer, got {window}")
    x = _float(x)
    moved = np.moveaxis(x, axis, 0).astype(np.float64)
    ok = ~np.isnan(moved)
    zeros = np.zeros((1, *moved.shape[1:]))
    csum = np.concatenate([zeros, np.cumsum(np.where(ok, moved, 0.0), axis=0)])
    ccnt = np.concatenate([zeros, np.cumsum(ok, axis=0)])
    n = len(moved)
    t = np.arange(n)
    lo = np.clip(t - (w // 2 if centered else w - 1), 0, n)
    hi = np.clip(t + (w // 2 if centered else 0) + 1, 0, n)
    total, count = csum[hi] - csum[lo], ccnt[hi] - ccnt[lo]
    out = np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)
    return np.moveaxis(out, 0, axis).astype(x.dtype)


def smoothing_factor(cutoff_hz: float, rate_hz: float) -> float:
    """The ``alpha`` of a first-order RC low-pass: ``dt / (RC + dt)``, ``RC = 1 / (2 pi f_c)``."""
    dt = 1.0 / float(rate_hz)
    rc = 1.0 / (2.0 * np.pi * float(cutoff_hz))
    return dt / (rc + dt)


def low_pass(x, alpha: float | None = None, *, cutoff_hz: float | None = None, rate_hz: float | None = None,
             axis: int = 0) -> np.ndarray:
    """First-order IIR low-pass (exponential smoothing) along ``axis``:
    ``y[0] = x[0]``, ``y[t] = y[t-1] + alpha * (x[t] - y[t-1])``.

    Give ``alpha`` in (0, 1] or ``cutoff_hz`` and ``rate_hz`` (see ``smoothing_factor``).
    Causal, so usable sample by sample. A NaN sample holds the previous output (the
    filter starts at the first non-NaN sample).
    """
    if alpha is None:
        if cutoff_hz is None or rate_hz is None:
            raise ValueError("give alpha, or cutoff_hz and rate_hz")
        alpha = smoothing_factor(cutoff_hz, rate_hz)
    a = float(alpha)
    if not 0.0 < a <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    x = _float(x)
    moved = np.moveaxis(x, axis, 0).astype(np.float64)
    if len(moved) == 0:
        return x
    flat = moved.reshape(len(moved), -1)
    out = np.empty_like(flat)
    y = flat[0].copy()
    out[0] = y
    for t in range(1, len(flat)):
        xt = flat[t]
        y = np.where(np.isnan(xt), y, np.where(np.isnan(y), xt, y + a * (xt - y)))
        out[t] = y
    out = out.reshape(moved.shape)
    return np.moveaxis(out, 0, axis).astype(x.dtype)


def remove_gravity(acc, alpha: float | None = 0.2, *, cutoff_hz: float | None = None,
                   rate_hz: float | None = None):
    """Split accelerometer data ``(T, 3)`` into linear acceleration and gravity.

    Gravity is the low-pass part of the specific force (``low_pass`` with ``alpha``, or a
    cutoff frequency); the linear acceleration is the rest. Returns ``(linear, gravity)``.
    The default ``alpha = 0.2`` is the Android example's ``0.8`` weight on the previous
    gravity estimate (``gravity = 0.8 * gravity + 0.2 * acc``).

    References
        Android Developers, "Motion sensors: use the accelerometer" (high-pass by subtracting a
        low-pass gravity estimate). https://developer.android.com/develop/sensors-and-location/sensors/sensors_motion
    """
    acc = _float(acc)
    if cutoff_hz is not None:
        alpha = None
    gravity = low_pass(acc, alpha, cutoff_hz=cutoff_hz, rate_hz=rate_hz, axis=0)
    return acc - gravity, gravity
