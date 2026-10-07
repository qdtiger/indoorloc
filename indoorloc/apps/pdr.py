"""L5 pedestrian dead reckoning (PDR): steps, step lengths, heading and the walked trajectory.

Inputs are IMU arrays with one row per sample, in the phone (device) frame, as Android and
iOS report them: accelerometer ``(T, 3)`` in m/s^2 (specific force, gravity included),
gyroscope ``(T, 3)`` in rad/s, magnetometer ``(T, 3)`` in any unit, and time ``t`` ``(T,)`` in
seconds (or ``rate_hz``). An IMU ``SampleTable`` (``meta["modality"] == "imu"``) is accepted
too; see :func:`imu_arrays`.

Heading convention: radians counter-clockwise from the +x axis of the map frame, so with an
east-north map frame 0 is east and pi/2 is north (``heading = pi/2 - azimuth``). Displacement of
a step of length L is ``L (cos h, sin h)``.

Pipeline: :class:`StepDetector` finds steps as peaks of the smoothed acceleration magnitude
(threshold, a valley between steps, minimum interval); :func:`weinberg_step_length` /
:func:`kim_step_length` estimate each step's length; heading comes from the gyroscope rotation
rate about gravity (:func:`yaw_rate`, :func:`integrate_heading`), the tilt-compensated compass
(:func:`magnetic_heading`) or both (:func:`complementary_heading`); :class:`PDR` chains them
into a :class:`StepTrack`. Ported from 0.1 ``signals/imu.py`` (``detect_steps``,
``compute_heading``) and ``signals/magnetometer.py``, adding the minimum step interval, the
valley condition, tilt compensation and the gyro/compass fusion that 0.1 lacked.

References
----------
H. Weinberg, "Using the ADXL202 in Pedometer and Personal Navigation Applications", Analog
    Devices Application Note AN-602, 2002.
J. W. Kim, H. J. Jang, D.-H. Hwang, C. Park, "A Step, Stride and Heading Determination for the
    Pedestrian Navigation System", Journal of Global Positioning Systems 3(1-2):273-279, 2004.
    DOI 10.5081/jgps.3.1.273.
A. R. Jimenez, F. Seco, C. Prieto, J. Guevara, "A comparison of Pedestrian Dead-Reckoning
    algorithms using a low-cost MEMS IMU", IEEE Int. Symp. on Intelligent Signal Processing
    (WISP), 2009. DOI 10.1109/WISP.2009.5286542.
R. Harle, "A Survey of Indoor Inertial Positioning Systems for Pedestrians", IEEE
    Communications Surveys & Tutorials 15(3):1281-1293, 2013. DOI 10.1109/SURV.2012.121912.00075.
T. Ozyagcilar, "Implementing a Tilt-Compensated eCompass using Accelerometer and Magnetometer
    Sensors", Freescale Semiconductor Application Note AN4248, 2012.
W. T. Higgins, "A Comparison of Complementary and Kalman Filtering", IEEE Transactions on
    Aerospace and Electronic Systems AES-11(3):321-325, 1975. DOI 10.1109/TAES.1975.308081.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..core import Estimator, SampleTable

GRAVITY = 9.80665  # standard gravity, m/s^2


# ---------------------------------------------------------------------------- small helpers
def wrap_angle(a):
    """Angle(s) wrapped to ``[-pi, pi)``."""
    return (np.asarray(a, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def _times(t, n: int, rate_hz) -> np.ndarray:
    if t is None:
        if rate_hz is None:
            raise ValueError("pass the sample times t=(T,) seconds or rate_hz=")
        return np.arange(n, dtype=np.float64) / float(rate_hz)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if t.shape != (n,) or (n > 1 and np.any(np.diff(t) < 0)):
        raise ValueError(f"t must be ({n},) non-decreasing seconds, got shape {t.shape}")
    return t


def _sample_rate(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) == 0:
        raise ValueError("need at least two distinct sample times")
    return 1.0 / float(np.median(dt))


def moving_average(x, n: int) -> np.ndarray:
    """Centred moving average of ``n`` samples along axis 0 (edges padded with the end values)."""
    x = np.asarray(x, dtype=np.float64)
    n = max(1, int(n))
    if n == 1 or len(x) == 0:
        return x.copy()
    lo, hi = (n - 1) // 2, n // 2
    pad = [(lo, hi)] + [(0, 0)] * (x.ndim - 1)
    c = np.cumsum(np.pad(x, pad, mode="edge"), axis=0)
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), c])
    return (c[n:] - c[:-n]) / n


def fill_gaps(x, t) -> np.ndarray:
    """Missing samples (NaN, the L1 convention) linearly interpolated in time, per column;
    the ends hold the nearest finite value. ``x`` is ``(T,)`` or ``(T, C)``, ``t`` ``(T,)``.
    A sensor sampled at a lower rate than the table's clock (NaN between its samples) is
    thereby resampled to that clock. A column without any finite value raises ValueError."""
    x = np.array(x, dtype=np.float64)
    bad = ~np.isfinite(x)
    if not bad.any():
        return x
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    cols, bad = x.reshape(len(x), -1), bad.reshape(len(x), -1)  # views into x
    for j in np.flatnonzero(bad.any(axis=0)):
        b = bad[:, j]
        if b.all():
            raise ValueError("an IMU channel has no finite sample")
        cols[b, j] = np.interp(t[b], t[~b], cols[~b, j])
    return x


def _vectors(a, name: str, n: int | None = None) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or (n is not None and len(a) != n):
        raise ValueError(f"{name} must be (T, 3){'' if n is None else f' with T={n}'}, got shape {a.shape}")
    return a


_SENSORS = {"acc": ("acc", "accel", "a"), "gyro": ("gyr", "gyro", "g", "w"), "mag": ("mag", "m")}


def imu_arrays(table: SampleTable) -> dict:
    """``{"acc", "gyro", "mag", "t"}`` arrays from an IMU ``SampleTable``.

    Channels are found by name in ``meta["channels"]`` (or ``meta["feature_names"]``): a sensor
    prefix (``acc``/``accel``, ``gyr``/``gyro``, ``mag``) followed by an axis letter, with any
    separator and case, e.g. ``acc_x``, ``GyroY``, ``mag.z``. Time is ``groups["time"]``, or
    the row index divided by ``meta["rate_hz"]``. Missing samples (NaN, e.g. a sensor slower than
    the table's clock) are interpolated in time (:func:`fill_gaps`); sensors without channels or
    samples are None. The table must hold one trajectory (one continuous recording).
    """
    if "trajectory" in table.groups and len(np.unique(table.groups["trajectory"])) > 1:
        raise ValueError("the IMU table holds several trajectories; select one first, e.g. "
                         "table[table.groups['trajectory'] == k]")
    names = list(table.meta.get("channels") or table.meta.get("feature_names") or [])
    X = np.asarray(table.X, dtype=np.float64).reshape(len(table), -1)
    if len(names) != X.shape[1]:
        raise ValueError("an IMU table needs meta['channels'] naming every column of X")
    keys = [re.sub(r"[^a-z]", "", str(c).lower()) for c in names]
    if "time" in table.groups:
        t = np.asarray(table.groups["time"], dtype=np.float64)
    elif table.meta.get("rate_hz"):
        t = np.arange(len(table)) / float(table.meta["rate_hz"])
    else:
        raise ValueError("an IMU table needs groups['time'] or meta['rate_hz']")
    out = {}
    for sensor, prefixes in _SENSORS.items():
        cols = []
        for axis in "xyz":
            hit = [j for j, k in enumerate(keys) if k.endswith(axis) and k[:-1] in prefixes]
            cols.append(hit[0] if hit else None)
        data = None if None in cols else X[:, cols]
        if data is not None and not np.isfinite(data).any(axis=1).any():
            data = None  # the channels exist but hold no sample
        out[sensor] = None if data is None else fill_gaps(data, t)
    out["t"] = t
    if out["acc"] is None:
        raise ValueError(f"no accelerometer channels (acc_x, acc_y, acc_z) among {names}")
    return out


# ---------------------------------------------------------------------------- step detection
@dataclass(frozen=True, eq=False)
class StepEvents:
    """Detected steps as parallel arrays (one entry per step).

    index     (S,) sample index of the step's acceleration peak
    t         (S,) time of the peak, seconds
    peak      (S,) smoothed, gravity-removed acceleration magnitude at the peak (m/s^2)
    valley    (S,) minimum of that signal within the step's window (m/s^2)
    mean_abs  (S,) mean of its absolute value within the window (m/s^2)

    A step's window runs from the midpoint with the previous step to the midpoint with the next.
    """

    index: np.ndarray
    t: np.ndarray
    peak: np.ndarray
    valley: np.ndarray
    mean_abs: np.ndarray

    def __len__(self) -> int:
        return len(self.index)


class StepDetector(Estimator):
    """Step detection by peak picking on the acceleration magnitude.

    The magnitude ``|a|`` minus gravity is smoothed by a centred moving average of
    ``smoothing`` seconds. A step is a local maximum at least ``peak_threshold`` high that
    comes at least ``min_interval`` seconds after the previous step, with the signal having
    fallen to ``valley_threshold`` or below in between (the heel-strike / push-off cycle). A
    higher peak before such a valley replaces the previous one (the same step's true peak).

    Parameters
    ----------
    peak_threshold : float
        Minimum peak height above gravity, m/s^2.
    valley_threshold : float
        Level the signal must reach between two steps, m/s^2 (0 = back under gravity).
    min_interval : float
        Minimum time between steps, seconds (0.3 s caps the cadence at 3.3 steps/s).
    smoothing : float
        Moving-average window, seconds (0 disables).
    gravity : float or None
        Value subtracted from the magnitude; None uses the median magnitude of the recording
        (robust to accelerometer scale error; offline only).

    References
    ----------
    H. Weinberg, "Using the ADXL202 in Pedometer and Personal Navigation Applications", Analog
        Devices AN-602, 2002.
    A. R. Jimenez, F. Seco, C. Prieto, J. Guevara, "A comparison of Pedestrian Dead-Reckoning
        algorithms using a low-cost MEMS IMU", WISP 2009. DOI 10.1109/WISP.2009.5286542.
    """

    _requires_fit = False

    def __init__(self, peak_threshold: float = 1.0, valley_threshold: float = 0.0, min_interval: float = 0.3,
                 smoothing: float = 0.1, gravity: float | None = None):
        self.peak_threshold = peak_threshold
        self.valley_threshold = valley_threshold
        self.min_interval = min_interval
        self.smoothing = smoothing
        self.gravity = gravity

    def signal(self, acc, t=None, *, rate_hz=None) -> tuple[np.ndarray, np.ndarray]:
        """The detection signal: smoothed ``|a| - g`` ``(T,)`` and its times ``(T,)``.
        ``acc`` is ``(T, 3)`` or an already computed magnitude ``(T,)``; NaN samples are
        interpolated in time (:func:`fill_gaps`)."""
        a = np.asarray(acc, dtype=np.float64)
        mag = np.linalg.norm(a, axis=1) if a.ndim == 2 else a.reshape(-1)
        t = _times(t, len(mag), rate_hz)
        mag = fill_gaps(mag, t) if len(mag) else mag  # a lost sample must not stop the detector
        g = float(np.median(mag)) if self.gravity is None else float(self.gravity)
        n = int(round(float(self.smoothing) * _sample_rate(t))) if self.smoothing and len(t) > 1 else 1
        return moving_average(mag - g, n), t

    def detect(self, acc, t=None, *, rate_hz=None) -> StepEvents:
        """Detect steps in ``acc`` ``(T, 3)`` (or magnitude ``(T,)``) sampled at ``t`` or ``rate_hz``."""
        s, t = self.signal(acc, t, rate_hz=rate_hz)
        if len(s) < 3:
            e = np.zeros(0)
            return StepEvents(np.zeros(0, np.int64), e, e, e, e)
        thr, low, gap = float(self.peak_threshold), float(self.valley_threshold), float(self.min_interval)
        cand = np.flatnonzero((s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:]) & (s[1:-1] >= thr)) + 1
        steps: list[int] = []
        for i in cand:
            if not steps:
                steps.append(int(i))
                continue
            j = steps[-1]
            if s[j:i + 1].min() <= low:  # a valley since the last step: a new step, if not too soon
                if t[i] - t[j] >= gap:
                    steps.append(int(i))
            elif s[i] > s[j]:  # no valley yet: still the same step, keep its highest peak
                steps[-1] = int(i)
        idx = np.asarray(steps, dtype=np.int64)
        peak, valley, mean_abs = (np.empty(len(idx)) for _ in range(3))
        if len(idx):
            # window of a step: midpoint with the previous step .. midpoint with the next; the
            # outer steps extend by half their neighbour gap (half min_interval if alone)
            gaps = np.diff(idx)
            half_first = gaps[0] // 2 if len(gaps) else max(1, int(round(gap * _sample_rate(t) / 2)))
            half_last = gaps[-1] // 2 if len(gaps) else half_first
            mids = (idx[:-1] + idx[1:] + 1) // 2
            starts = np.r_[max(0, idx[0] - half_first), mids]
            ends = np.r_[mids, min(len(s), idx[-1] + half_last + 1)]
            for k, (a, b) in enumerate(zip(starts, ends)):
                w = s[a:b]
                peak[k], valley[k], mean_abs[k] = s[idx[k]], w.min(), np.abs(w).mean()
        return StepEvents(idx, t[idx], peak, valley, mean_abs)


# ---------------------------------------------------------------------------- step length
WEINBERG_K = 0.47  # ~0.7 m for a +-2.5 m/s^2 walking bounce; calibrate per user and device
KIM_K = 0.60  # ~0.7 m for the same signal


def weinberg_step_length(peak, valley, k: float = WEINBERG_K) -> np.ndarray:
    """Weinberg (2002): ``L = k (a_max - a_min)^(1/4)`` per step (accelerations in m/s^2)."""
    amp = np.asarray(peak, dtype=np.float64) - np.asarray(valley, dtype=np.float64)
    return float(k) * np.maximum(amp, 0.0) ** 0.25


def kim_step_length(mean_abs, k: float = KIM_K) -> np.ndarray:
    """Kim et al. (2004): ``L = k (mean |a|)^(1/3)`` over the samples of each step."""
    return float(k) * np.cbrt(np.maximum(np.asarray(mean_abs, dtype=np.float64), 0.0))


def _step_feature(steps: StepEvents, model: str) -> np.ndarray:
    if model == "weinberg":
        return np.maximum(steps.peak - steps.valley, 0.0) ** 0.25
    if model == "kim":
        return np.cbrt(np.maximum(steps.mean_abs, 0.0))
    if model == "constant":
        return np.ones(len(steps))
    raise ValueError(f"step model must be 'weinberg', 'kim' or 'constant', got {model!r}")


def step_lengths(steps: StepEvents, model: str = "weinberg", k: float | None = None) -> np.ndarray:
    """``(S,)`` step lengths in metres. ``model`` is ``"weinberg"``, ``"kim"`` or ``"constant"``
    (then ``k`` is the length itself, default 0.7 m)."""
    default = {"weinberg": WEINBERG_K, "kim": KIM_K, "constant": 0.7}.get(model)
    return (default if k is None else float(k)) * _step_feature(steps, model)


def calibrate_step_length(steps: StepEvents, distance: float, model: str = "weinberg") -> float:
    """The ``k`` that makes the steps of a walk of known ``distance`` (metres) add up to it:
    ``k = distance / sum(feature_i)`` (closed form; the least-squares fit of a total)."""
    f = _step_feature(steps, model)
    if not f.sum() > 0:
        raise ValueError("no step with a positive feature to calibrate on")
    return float(distance) / float(f.sum())


# ---------------------------------------------------------------------------- heading
def up_vector(acc, t=None, *, rate_hz=None, smoothing: float = 1.0) -> np.ndarray:
    """``(T, 3)`` unit vector of 'up' in the device frame: the low-passed specific force
    (``smoothing`` seconds of moving average removes the walking accelerations)."""
    a = _vectors(acc, "acc")
    t = _times(t, len(a), rate_hz)
    n = int(round(float(smoothing) * _sample_rate(t))) if smoothing and len(t) > 1 else 1
    g = moving_average(a, n)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)


def yaw_rate(gyro, acc=None, t=None, *, rate_hz=None, smoothing: float = 1.0) -> np.ndarray:
    """Rotation rate about the vertical, rad/s ``(T,)`` (positive = counter-clockwise seen from
    above, i.e. heading increasing). ``gyro`` ``(T, 3)`` is projected on the up vector from
    ``acc`` (valid for any phone tilt); without ``acc`` the device z axis is taken as up.
    A ``(T,)`` gyro is taken as the yaw rate already."""
    w = np.asarray(gyro, dtype=np.float64)
    if w.ndim == 1:
        return w.copy()
    w = _vectors(w, "gyro")
    if acc is None:
        return w[:, 2].copy()
    up = up_vector(acc, t, rate_hz=rate_hz, smoothing=smoothing)
    if len(up) != len(w):
        raise ValueError("gyro and acc must have the same number of samples")
    return np.einsum("ij,ij->i", w, up)


def integrate_heading(rate, t=None, *, rate_hz=None, heading0: float = 0.0) -> np.ndarray:
    """Heading ``(T,)`` from a yaw rate by the trapezoid rule, starting at ``heading0``
    (not wrapped, so turns accumulate; wrap with :func:`wrap_angle` if needed)."""
    r = np.asarray(rate, dtype=np.float64).reshape(-1)
    t = _times(t, len(r), rate_hz)
    inc = (r[1:] + r[:-1]) / 2 * np.diff(t)
    return float(heading0) + np.concatenate([[0.0], np.cumsum(inc)])


def magnetic_heading(mag, acc, t=None, *, rate_hz=None, forward=(0.0, 1.0, 0.0), offset: float = 0.0,
                     smoothing: float = 1.0) -> np.ndarray:
    """Tilt-compensated compass heading ``(T,)`` of the device ``forward`` axis.

    The field is projected on the horizontal plane (normal = up from ``acc``) to give magnetic
    north; east = north x up; the heading of the horizontal projection of ``forward`` is
    ``atan2(f.north, f.east)`` (counter-clockwise from magnetic east) plus ``offset`` (the map
    frame's rotation from east-north, including the magnetic declination). ``forward`` is the
    device +y axis by default (top of a phone held flat in front of the walker).
    """
    m = _vectors(mag, "mag")
    up = up_vector(acc, t, rate_hz=rate_hz, smoothing=smoothing)
    if len(up) != len(m):
        raise ValueError("mag and acc must have the same number of samples")
    north = m - np.einsum("ij,ij->i", m, up)[:, None] * up
    north /= np.maximum(np.linalg.norm(north, axis=1, keepdims=True), 1e-12)
    east = np.cross(north, up)
    f = np.asarray(forward, dtype=np.float64)
    return np.arctan2(north @ f, east @ f) + float(offset)


def complementary_heading(rate, mag_heading, t=None, *, rate_hz=None, time_constant: float = 5.0,
                          heading0: float | None = None) -> np.ndarray:
    """Gyro/compass complementary filter ``(T,)``.

    ``h_k = h_pred + alpha * wrap(m_k - h_pred)`` with ``h_pred = h_{k-1} + mean yaw rate * dt``
    and ``alpha = dt / (time_constant + dt)``: the gyroscope passes above ``1/(2 pi tau)`` Hz,
    the compass below. A constant gyro bias ``b`` leaves a steady-state error ``b * tau``.
    NaN compass samples (e.g. flagged disturbances) are skipped. Starts at ``heading0``, or at
    the first finite compass heading.
    """
    r = np.asarray(rate, dtype=np.float64).reshape(-1)
    m = np.asarray(mag_heading, dtype=np.float64).reshape(-1)
    if m.shape != r.shape:
        raise ValueError("rate and mag_heading must have the same length")
    t = _times(t, len(r), rate_hz)
    tau = float(time_constant)
    h = np.empty(len(r))
    if len(r) == 0:
        return h
    if heading0 is None:
        finite = np.flatnonzero(np.isfinite(m))
        heading0 = m[finite[0]] if len(finite) else 0.0
    h[0] = heading0
    for k in range(1, len(r)):
        dt = t[k] - t[k - 1]
        pred = h[k - 1] + (r[k] + r[k - 1]) / 2 * dt
        if np.isfinite(m[k]):
            pred += dt / (tau + dt) * wrap_angle(m[k] - pred) if tau + dt > 0 else wrap_angle(m[k] - pred)
        h[k] = pred
    return h


# ---------------------------------------------------------------------------- PDR tracker
@dataclass(frozen=True, eq=False)
class StepTrack:
    """A PDR trajectory: one entry per step (parallel arrays), plus where it started.

    t (S,) step times; index (S,) sample indices; length (S,) metres; heading (S,) radians
    (counter-clockwise from +x); pos (S, 2) position after each step; start (2,) and t0 the
    position and time before the first step.
    """

    t: np.ndarray
    index: np.ndarray
    length: np.ndarray
    heading: np.ndarray
    pos: np.ndarray
    start: np.ndarray
    t0: float

    def __len__(self) -> int:
        return len(self.t)

    @property
    def distance(self) -> float:
        """Walked distance, metres."""
        return float(self.length.sum())

    def position_at(self, t) -> np.ndarray:
        """``(N, 2)`` positions at times ``t`` by linear interpolation between steps (held
        constant before the first and after the last step)."""
        tq = np.atleast_1d(np.asarray(t, dtype=np.float64))
        knots_t = np.r_[self.t0, self.t]
        knots = np.vstack([self.start[None], self.pos])
        return np.stack([np.interp(tq, knots_t, knots[:, i]) for i in (0, 1)], axis=1)


class PDR(Estimator):
    """Pedestrian dead reckoning: detect steps, size them, orient them, add them up.

    ``pos_k = start + sum_{j<=k} L_j (cos h_j, sin h_j)`` with ``L_j`` from the step-length
    model and ``h_j`` the heading at the step's peak sample. Heading sources:

    * ``"gyro"``: rotation rate about gravity integrated from ``initial_heading`` (0 if None);
      accurate over minutes but drifts, and needs the initial heading in the map frame;
    * ``"mag"``: tilt-compensated compass plus ``mag_offset`` (absolute but disturbed indoors);
    * ``"fused"``: complementary filter of the two with ``time_constant`` seconds;
    * or pass ``yaw=`` ``(T,)`` to :meth:`run` (e.g. from the phone's rotation vector).

    Parameters
    ----------
    detector : StepDetector or None
        Step detector (default ``StepDetector()``).
    step_model : {"weinberg", "kim", "constant"}
    k : float or None
        Step-length constant (see :func:`step_lengths`; calibrate with
        :func:`calibrate_step_length`).
    heading : {"gyro", "mag", "fused"}
    time_constant : float
        Complementary-filter time constant, seconds.
    forward : (3,) array-like
        Device axis pointing along the walking direction.
    mag_offset : float
        Map-frame rotation of magnetic east (radians), including declination.
    initial_heading : float or None
        Heading at the first sample for ``"gyro"`` / ``"fused"`` (None: 0, resp. the compass).
    smoothing : float
        Low-pass window (s) for the gravity direction used by the heading.

    References
    ----------
    R. Harle, "A Survey of Indoor Inertial Positioning Systems for Pedestrians", IEEE
        Communications Surveys & Tutorials 15(3):1281-1293, 2013. DOI 10.1109/SURV.2012.121912.00075.
    H. Weinberg, Analog Devices AN-602, 2002; J. W. Kim et al., JGPS 3(1-2):273-279, 2004.
        DOI 10.5081/jgps.3.1.273.
    W. T. Higgins, "A Comparison of Complementary and Kalman Filtering", IEEE TAES 11(3):321-325,
        1975. DOI 10.1109/TAES.1975.308081.
    """

    _requires_fit = False

    def __init__(self, detector=None, step_model: str = "weinberg", k: float | None = None, heading: str = "gyro",
                 time_constant: float = 5.0, forward=(0.0, 1.0, 0.0), mag_offset: float = 0.0,
                 initial_heading: float | None = None, smoothing: float = 1.0):
        self.detector = detector
        self.step_model = step_model
        self.k = k
        self.heading = heading
        self.time_constant = time_constant
        self.forward = forward
        self.mag_offset = mag_offset
        self.initial_heading = initial_heading
        self.smoothing = smoothing

    def headings(self, acc, gyro=None, mag=None, t=None, *, rate_hz=None) -> np.ndarray:
        """Heading ``(T,)`` at every sample from the configured source."""
        a = _vectors(acc, "acc")
        t = _times(t, len(a), rate_hz)
        if self.heading not in ("gyro", "mag", "fused"):
            raise ValueError(f"heading must be 'gyro', 'mag' or 'fused', got {self.heading!r}")
        if self.heading in ("mag", "fused"):
            if mag is None:
                raise ValueError(f"heading={self.heading!r} needs magnetometer data (mag=)")
            mh = magnetic_heading(mag, a, t, forward=self.forward, offset=self.mag_offset, smoothing=self.smoothing)
            if self.heading == "mag":
                return mh
        if gyro is None:
            raise ValueError(f"heading={self.heading!r} needs gyroscope data (gyro=)")
        rate = yaw_rate(gyro, a, t, smoothing=self.smoothing)
        if self.heading == "gyro":
            return integrate_heading(rate, t, heading0=0.0 if self.initial_heading is None else self.initial_heading)
        return complementary_heading(rate, mh, t, time_constant=self.time_constant, heading0=self.initial_heading)

    def run(self, acc, gyro=None, mag=None, t=None, *, rate_hz=None, yaw=None, start=(0.0, 0.0)) -> StepTrack:
        """Dead-reckon a recording. ``acc`` may be an IMU ``SampleTable`` (then gyro, mag and t
        come from it, see :func:`imu_arrays`). ``yaw`` ``(T,)`` overrides the heading source.
        Missing samples (NaN) of acc, gyro and mag are interpolated in time (:func:`fill_gaps`).

        Returns a :class:`StepTrack` (one entry per detected step, not per IMU sample), not a
        ``Prediction``: score ``track.pos`` against the truth at ``track.t``, or take positions
        at other times with ``track.position_at(t)``. ``PDRFusion.run`` accepts it as is."""
        if isinstance(acc, SampleTable):
            arrays = imu_arrays(acc)
            acc, t = arrays["acc"], arrays["t"]
            gyro = arrays["gyro"] if gyro is None else gyro
            mag = arrays["mag"] if mag is None else mag
        t = _times(t, len(np.asarray(acc)), rate_hz)
        a = fill_gaps(_vectors(acc, "acc"), t)  # NaN = missing sample (L1 convention): interpolate
        gyro = None if gyro is None else fill_gaps(gyro, t)
        mag = None if mag is None else fill_gaps(mag, t)
        detector = StepDetector() if self.detector is None else self.detector
        steps = detector.detect(a, t)
        if yaw is not None:
            h_all = np.asarray(yaw, dtype=np.float64).reshape(-1)
            if h_all.shape != (len(a),):
                raise ValueError(f"yaw must be ({len(a)},), got shape {h_all.shape}")
        else:
            h_all = self.headings(a, gyro, mag, t)
        L = step_lengths(steps, self.step_model, self.k)
        h = h_all[steps.index]
        start = np.asarray(start, dtype=np.float64).reshape(2)
        pos = start + np.cumsum(np.stack([L * np.cos(h), L * np.sin(h)], axis=1), axis=0)
        return StepTrack(steps.t, steps.index, L, wrap_angle(h), pos.reshape(-1, 2), start,
                         float(t[0]) if len(t) else 0.0)


__all__ = ["GRAVITY", "PDR", "StepDetector", "StepEvents", "StepTrack", "calibrate_step_length",
           "complementary_heading", "fill_gaps", "imu_arrays", "integrate_heading", "kim_step_length",
           "magnetic_heading", "moving_average", "step_lengths", "up_vector", "weinberg_step_length", "wrap_angle",
           "yaw_rate"]
