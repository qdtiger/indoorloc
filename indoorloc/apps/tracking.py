"""L5 tracking: Kalman filters and the Rauch-Tung-Striebel smoother on top of the other layers.

A fitted L3 localizer is used as a virtual position sensor. :class:`KalmanTracker` filters its
fixes with a constant-velocity (``"cv"``) or constant-acceleration (``"ca"``) motion model and
takes the measurement noise of every fix from ``Prediction.spread``; :meth:`KalmanTracker.smooth`
runs the RTS smoother over a recorded trajectory. :class:`ExtendedKalmanTracker` skips the
localizer and is fed the raw ranges to known anchors (UWB / RTT / BLE ranging): tight coupling,
so it keeps tracking with fewer anchors than a snapshot multilateration needs.

Conventions: positions are float64 in the dataset frame; time ``t`` in seconds (any
non-decreasing values; irregular steps are fine); a fix row with NaN is a missing fix (predict
only). Every time argument of the online API (``update(t, ...)``, ``predict_to(t)``,
``reset(t)``) is an absolute time on the stream's clock, never an increment. ``filter`` and
``smooth`` return an ``indoorloc.core.Prediction`` with one row per input row whose ``spread``
is ``sqrt(trace(P_pos))`` (the RMS radial error of the Gaussian estimate), so the output of a
tracker feeds ``indoorloc.evaluation.evaluate`` or another L5 stage directly.

``ConstantVelocityKF`` and ``track`` are the minimal 0.2 sketch API, kept unchanged.
"""
from __future__ import annotations

import numpy as np

from ..core import Estimator, Prediction

_BLOCKS = {"cv": 2, "ca": 3}  # state blocks per axis: position, velocity (, acceleration)


def motion_model(dt: float, motion: str = "cv", *, dim: int = 2, q: float = 1.0,
                 noise: str = "continuous") -> tuple[np.ndarray, np.ndarray]:
    """Transition ``F`` and process noise ``Q`` of a nearly-constant velocity / acceleration model.

    The state is ``[p (dim), v (dim)]`` (``"cv"``) or ``[p, v, a]`` (``"ca"``), each axis an
    independent copy of the 1-D model (``F = kron(F1, I)``, ``Q = kron(Q1, I)``).

    ``noise="continuous"`` (default): white acceleration (cv) or white jerk (ca) of power
    spectral density ``q`` (m^2/s^3, resp. m^2/s^5); exact for any ``dt``, so two predictions
    of ``dt/2`` equal one of ``dt`` -- the right choice for irregular scan times.
    ``noise="discrete"``: acceleration (cv) or acceleration increment (ca) constant over each
    step with variance ``q`` (Bar-Shalom et al., Sec. 6.3.2-6.3.3).

    References: Y. Bar-Shalom, X. R. Li, T. Kirubarajan, "Estimation with Applications to
    Tracking and Navigation", Wiley, 2001, Sec. 6.2-6.3. DOI 10.1002/0471221279.
    """
    if motion not in _BLOCKS:
        raise ValueError(f"motion must be one of {sorted(_BLOCKS)}, got {motion!r}")
    dt = float(dt)
    if motion == "cv":
        F1 = np.array([[1.0, dt], [0.0, 1.0]])
        if noise == "continuous":
            Q1 = np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        elif noise == "discrete":
            g = np.array([dt ** 2 / 2, dt])
            Q1 = np.outer(g, g)
        else:
            raise ValueError(f"noise must be 'continuous' or 'discrete', got {noise!r}")
    else:
        F1 = np.array([[1.0, dt, dt ** 2 / 2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        if noise == "continuous":
            Q1 = np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                           [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                           [dt ** 3 / 6, dt ** 2 / 2, dt]])
        elif noise == "discrete":
            g = np.array([dt ** 2 / 2, dt, 1.0])
            Q1 = np.outer(g, g)
        else:
            raise ValueError(f"noise must be 'continuous' or 'discrete', got {noise!r}")
    eye = np.eye(int(dim))
    return np.kron(F1, eye), float(q) * np.kron(Q1, eye)


def _kf_predict(x, P, F, Q):
    P = F @ P @ F.T + Q
    return F @ x, (P + P.T) / 2


def _kf_correct(x, P, y, H, R):
    """Update with innovation ``y`` (Joseph form, stays symmetric positive semi-definite)."""
    S = H @ P @ H.T + R
    K = np.linalg.solve(S, H @ P).T  # P H' S^-1 with S, P symmetric
    x = x + K @ y
    A = np.eye(len(x)) - K @ H
    P = A @ P @ A.T + K @ R @ K.T
    return x, (P + P.T) / 2


def _nis(y, H, P, R) -> float:
    """Normalised innovation squared ``y' S^-1 y`` (chi-square with len(y) dof if consistent)."""
    return float(y @ np.linalg.solve(H @ P @ H.T + R, y))


def rts_smooth(x_filt, P_filt, x_pred, P_pred, F) -> tuple[np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel fixed-interval smoother.

    ``x_filt`` ``(T, S)`` / ``P_filt`` ``(T, S, S)`` are the filtered moments of steps 0..T-1,
    ``x_pred`` / ``P_pred`` the one-step predictions of step k from k-1 and ``F[k]`` the
    transition from k-1 to k (row 0 of these three is not used). Returns the smoothed
    ``(x, P)``: ``C_k = P_k F_{k+1}' P_{k+1|k}^-1``,
    ``x_k^s = x_k + C_k (x_{k+1}^s - x_{k+1|k})``, ``P_k^s = P_k + C_k (P_{k+1}^s - P_{k+1|k}) C_k'``.

    References: H. E. Rauch, F. Tung, C. T. Striebel, "Maximum likelihood estimates of linear
    dynamic systems", AIAA Journal 3(8):1445-1450, 1965. DOI 10.2514/3.3166.
    """
    xs, Ps = np.array(x_filt, dtype=np.float64), np.array(P_filt, dtype=np.float64)
    for k in range(len(xs) - 2, -1, -1):
        C = np.linalg.solve(P_pred[k + 1], F[k + 1] @ P_filt[k]).T  # P_k F' P_pred^-1, both symmetric
        xs[k] = x_filt[k] + C @ (xs[k + 1] - x_pred[k + 1])
        P = P_filt[k] + C @ (Ps[k + 1] - P_pred[k + 1]) @ C.T
        Ps[k] = (P + P.T) / 2
    return xs, Ps


def _times(t, n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) if t is None else np.asarray(t, dtype=np.float64).reshape(-1)
    if t.shape != (n,):
        raise ValueError(f"t must have one time per row ({n}), got shape {t.shape}")
    if n and (not np.all(np.isfinite(t)) or np.any(np.diff(t) < 0)):
        raise ValueError("t must be finite and non-decreasing")
    return t


class _GaussTracker(Estimator):
    """Shared machinery of the Kalman trackers: time update, bookkeeping, filter/smooth loops.

    Subclasses define ``_state_dim()``, ``_initialize(z, spread, t) -> bool`` and
    ``_correct(z, spread) -> (nis, accepted)``.
    """

    _requires_fit = False

    def _blocks(self) -> int:
        if self.motion not in _BLOCKS:
            raise ValueError(f"motion must be one of {sorted(_BLOCKS)}, got {self.motion!r}")
        return _BLOCKS[self.motion]

    # ---------------------------------------------------------------- online API
    def reset(self, t: float = 0.0):
        """Forget the state; the next valid measurement initialises the track at time ``t``."""
        self.x_, self.P_, self.t_ = None, None, float(t)
        return self

    def _advance(self, t: float) -> np.ndarray:
        dt = float(t) - self.t_
        if dt < 0:
            raise ValueError(f"time went backwards: {t} < {self.t_}")
        F, Q = motion_model(dt, self.motion, dim=self.dim_, q=self.process_noise, noise=self.noise)
        self.x_, self.P_ = _kf_predict(self.x_, self.P_, F, Q)
        self.t_ = float(t)
        return F

    def predict_to(self, t: float) -> np.ndarray:
        """Advance the state to the **absolute** time ``t`` (seconds, the clock of :meth:`update`)
        without a measurement; returns the position. Before the first fix it only moves the
        clock. The same name and meaning as :meth:`ParticleFilter.predict_to
        <indoorloc.apps.particle.ParticleFilter.predict_to>`."""
        if getattr(self, "x_", None) is None:
            self.t_ = float(t)
        else:
            self._advance(t)
        return self.estimate()[0]

    def predict(self, t: float) -> np.ndarray:
        """Same as :meth:`predict_to`: ``t`` is an **absolute** time, not an increment.

        ``ParticleFilter.predict(dt)`` takes an increment instead; :meth:`predict_to` has one
        meaning on every tracker, so prefer it in code that may swap trackers.
        """
        return self.predict_to(t)

    def estimate(self) -> tuple[np.ndarray, float]:
        """Current ``(position (D,), spread)``; NaN before the first measurement."""
        if getattr(self, "x_", None) is None:
            return np.full(getattr(self, "dim_", 2), np.nan), float("nan")
        d = self.dim_
        return self.x_[:d].copy(), float(np.sqrt(np.trace(self.P_[:d, :d])))

    def _step(self, t, z, spread):
        """One online step; returns ``(F or None, nis, accepted)``."""
        if getattr(self, "x_", None) is None:
            self.t_ = float(t)
            return None, np.nan, bool(self._initialize(z, spread, t))
        F = self._advance(t)
        nis, ok = self._correct(z, spread)
        return F, nis, ok

    # ---------------------------------------------------------------- offline API
    def _sequence(self, Z, t, spreads, smooth: bool):
        T = len(Z)
        t = _times(t, T)
        self.reset(t[0] if T else 0.0)
        rows = []  # (k, x_filt, P_filt, x_pred, P_pred, F); the first row has no prediction
        nis, accepted = np.full(T, np.nan), np.zeros(T, dtype=bool)
        for k in range(T):
            if self.x_ is None:
                self.t_ = float(t[k])
                accepted[k] = self._initialize(Z[k], spreads[k], t[k])
                if accepted[k]:
                    rows.append((k, self.x_.copy(), self.P_.copy(), None, None, None))
                continue
            F = self._advance(t[k])
            xp, Pp = self.x_.copy(), self.P_.copy()
            nis[k], accepted[k] = self._correct(Z[k], spreads[k])
            rows.append((k, self.x_.copy(), self.P_.copy(), xp, Pp, F))
        d, S = self.dim_, self._state_dim()
        states, covs = np.full((T, S), np.nan), np.full((T, S, S), np.nan)
        if rows:
            ks = np.array([r[0] for r in rows])
            xf, Pf = np.stack([r[1] for r in rows]), np.stack([r[2] for r in rows])
            if smooth and len(rows) > 1:  # row 0 of the predicted moments is never read
                xp = np.stack([np.zeros(S)] + [r[3] for r in rows[1:]])
                Pp = np.stack([np.eye(S)] + [r[4] for r in rows[1:]])
                Fs = np.stack([np.eye(S)] + [r[5] for r in rows[1:]])
                xf, Pf = rts_smooth(xf, Pf, xp, Pp, Fs)
            states[ks], covs[ks] = xf, Pf
        self.states_, self.covariances_, self.nis_, self.accepted_ = states, covs, nis, accepted
        return states[:, :d], np.sqrt(np.trace(covs[:, :d, :d], axis1=1, axis2=2))


def _unpack_fixes(z, spread):
    """Positions, per-row spread and the label columns to pass through from a Prediction."""
    labels = {}
    if isinstance(z, Prediction):
        labels = {"floor": z.floor, "building": z.building, "ids": z.ids}
        spread = z.spread if spread is None else spread
        z = z.pos
    Z = np.asarray(z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    if Z.ndim != 2:
        raise ValueError(f"fixes must be (T, D) positions, got shape {Z.shape}")
    if spread is None:
        spread = np.full(len(Z), np.nan)
    spread = np.broadcast_to(np.asarray(spread, dtype=np.float64), (len(Z),))
    return Z, spread, labels


class KalmanTracker(_GaussTracker):
    """Kalman tracker of a localizer's fixes with adaptive measurement noise and an RTS smoother.

    The state is position and velocity (``motion="cv"``) or position, velocity and
    acceleration (``"ca"``) per axis, driven by white noise (:func:`motion_model`). Each fix
    ``z_k`` is a direct measurement of the position with covariance ``r_k^2 I``, where
    ``r_k = max(spread_k * spread_scale, min_meas_std)`` when the fix carries a
    ``Prediction.spread`` (and ``use_spread``), else ``meas_std``. The first valid fix
    initialises the state (position = fix, velocity 0 with std ``init_vel_std``). A fix
    whose normalised innovation exceeds ``gate`` (in sigma units, ``sqrt(NIS) > gate``) is
    rejected as an outlier. :meth:`smooth` adds the Rauch-Tung-Striebel backward pass, the
    minimum-variance estimate given the whole recording.

    Parameters
    ----------
    motion : {"cv", "ca"}
        Constant velocity or constant acceleration.
    process_noise : float
        Noise intensity ``q``: PSD of the white acceleration (cv, m^2/s^3) or jerk (ca, m^2/s^5)
        for ``noise="continuous"``; per-step variance for ``noise="discrete"``. A walking
        person changes speed by roughly ``sqrt(q * dt)`` m/s over ``dt`` seconds.
    noise : {"continuous", "discrete"}
        Process-noise discretisation (see :func:`motion_model`).
    meas_std : float
        Fix noise std (coordinate units) when no spread is available or ``use_spread=False``.
    use_spread, spread_scale, min_meas_std
        Adaptive noise from ``Prediction.spread`` as described above (min applies to the
        spread-derived value only).
    init_vel_std : float
        Prior std of the velocity (m/s) and, for ``"ca"``, of the acceleration (m/s^2).
    gate : float or None
        Outlier gate in sigma units (e.g. 3.0); None accepts every fix.

    Online use: :meth:`reset` ``(t)``, then :meth:`update` ``(t, z, spread)`` per fix and
    :meth:`predict_to` ``(t)`` to extrapolate without one; every ``t`` is an absolute time in
    seconds (``predict(t)``, the older name, is also absolute, unlike ``ParticleFilter.predict(dt)``).

    Attributes (after :meth:`filter` / :meth:`smooth`): ``states_`` ``(T, S)``, ``covariances_``
    ``(T, S, S)``, ``nis_`` ``(T,)`` (NaN without a fix), ``accepted_`` ``(T,)``; the online
    state ``x_``, ``P_``, ``t_`` holds the last filtered step.

    References
    ----------
    R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems", Journal of
        Basic Engineering 82(1):35-45, 1960. DOI 10.1115/1.3662552.
    H. E. Rauch, F. Tung, C. T. Striebel, "Maximum likelihood estimates of linear dynamic
        systems", AIAA Journal 3(8):1445-1450, 1965. DOI 10.2514/3.3166.
    Y. Bar-Shalom, X. R. Li, T. Kirubarajan, "Estimation with Applications to Tracking and
        Navigation", Wiley, 2001. DOI 10.1002/0471221279.
    """

    def __init__(self, motion: str = "cv", process_noise: float = 0.5, noise: str = "continuous",
                 meas_std: float = 3.0, use_spread: bool = True, spread_scale: float = 1.0,
                 min_meas_std: float = 1.0, init_vel_std: float = 1.0, gate: float | None = None):
        self.motion = motion
        self.process_noise = process_noise
        self.noise = noise
        self.meas_std = meas_std
        self.use_spread = use_spread
        self.spread_scale = spread_scale
        self.min_meas_std = min_meas_std
        self.init_vel_std = init_vel_std
        self.gate = gate

    def _state_dim(self) -> int:
        return self._blocks() * self.dim_

    def _std(self, spread) -> float:
        if self.use_spread and spread is not None and np.isfinite(spread):
            return max(float(spread) * self.spread_scale, float(self.min_meas_std))
        return float(self.meas_std)

    def _initialize(self, z, spread, t) -> bool:
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(z)):
            return False
        d, n = len(z), self._blocks()
        self.dim_ = d
        r = self._std(spread)
        self.x_ = np.concatenate([z, np.zeros((n - 1) * d)])
        self.P_ = np.diag(np.concatenate([np.full(d, r ** 2), np.full((n - 1) * d, float(self.init_vel_std) ** 2)]))
        self.t_ = float(t)
        return True

    def _correct(self, z, spread):
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(z)):
            return np.nan, False
        if len(z) != self.dim_:
            raise ValueError(f"fix has {len(z)} coordinates, the track has {self.dim_}")
        H = np.eye(self.dim_, self._state_dim())
        R = self._std(spread) ** 2 * np.eye(self.dim_)
        y = z - H @ self.x_
        nis = _nis(y, H, self.P_, R)
        if self.gate is not None and nis > float(self.gate) ** 2:
            return nis, False
        self.x_, self.P_ = _kf_correct(self.x_, self.P_, y, H, R)
        return nis, True

    def update(self, t: float, z=None, spread=None) -> np.ndarray:
        """Online step: predict to ``t``, then correct with fix ``z`` (None/NaN = no fix).

        ``spread`` is the fix's ``Prediction.spread`` (sets the noise as described above).
        The first valid fix initialises the track. Returns the position ``(D,)``.
        """
        self._step(t, np.full(getattr(self, "dim_", 1), np.nan) if z is None else z, spread)
        return self.estimate()[0]

    def correct(self, z, spread=None) -> bool:
        """Measurement update at the current time; returns False if gated out or missing."""
        if getattr(self, "x_", None) is None:
            return bool(self._initialize(z, spread, getattr(self, "t_", 0.0)))
        return bool(self._correct(z, spread)[1])

    def filter(self, z, t=None, spread=None) -> Prediction:
        """Forward pass over a recording: ``z`` is ``(T, D)`` fixes (NaN rows = missing) or a
        ``Prediction`` (its ``spread``, floor, building and ids are used / passed through).
        ``t`` defaults to ``0, 1, 2, ...`` seconds. Rows before the first fix are NaN."""
        return self._offline(z, t, spread, smooth=False)

    def smooth(self, z, t=None, spread=None) -> Prediction:
        """Filter, then the RTS backward pass (same inputs and output as :meth:`filter`)."""
        return self._offline(z, t, spread, smooth=True)

    def _offline(self, z, t, spread, smooth: bool) -> Prediction:
        Z, spreads, labels = _unpack_fixes(z, spread)
        self.dim_ = Z.shape[1]
        pos, s = self._sequence(Z, t, spreads, smooth=smooth)
        return Prediction(pos, spread=s, **labels)


def multilaterate(ranges, anchors, *, height=None, iterations: int = 20) -> np.ndarray | None:
    """Least-squares position from ranges to anchors (NaN ranges are ignored), or None.

    Linear initial guess (differences of squared range equations), refined by Gauss-Newton.
    With ``height`` the anchors are 3-D and the target moves in the plane ``z = height``.
    Needs at least ``D + 1`` finite ranges (D = planar dimension), else returns None.
    """
    r = np.asarray(ranges, dtype=np.float64).reshape(-1)
    A = np.asarray(anchors, dtype=np.float64)
    ok = np.isfinite(r)
    r, A = r[ok], A[ok]
    if height is not None:
        dz2 = (A[:, 2] - float(height)) ** 2
        A, r = A[:, :2], np.sqrt(np.maximum(r ** 2 - dz2, 0.0))
    d = A.shape[1]
    if len(r) < d + 1:
        return None
    M = 2.0 * (A[1:] - A[0])
    b = (A[1:] ** 2).sum(1) - (A[0] ** 2).sum() - r[1:] ** 2 + r[0] ** 2
    if np.linalg.matrix_rank(M) < d:
        return None
    p, *_ = np.linalg.lstsq(M, b, rcond=None)
    for _ in range(int(iterations)):
        diff = p - A
        dist = np.maximum(np.linalg.norm(diff, axis=1), 1e-12)
        J = diff / dist[:, None]
        step, *_ = np.linalg.lstsq(J, r - dist, rcond=None)
        p = p + step
        if np.linalg.norm(step) < 1e-10:
            break
    return p


class ExtendedKalmanTracker(_GaussTracker):
    """Extended Kalman tracker fed directly by ranges to known anchors (tight coupling).

    State and motion model as in :class:`KalmanTracker`. The measurement of step k is the
    vector of ranges ``r_i = ||p - a_i||`` to the anchors (``(A,)``, NaN = anchor not heard),
    linearised at the predicted position (Jacobian rows ``(p - a_i)' / r_i``). Because each
    range updates the track on its own, one or two anchors still correct it between
    snapshots that a multilateration could not solve. The track starts at the first scan
    that :func:`multilaterate` can solve (at least D + 1 ranges), with covariance
    ``range_std^2 (J'J)^-1``. ``gate`` rejects single ranges whose normalised innovation
    exceeds it (NLOS outliers).

    Parameters
    ----------
    anchors : array-like (A, D_a)
        Anchor coordinates in the dataset frame.
    range_std : float or array-like (A,)
        Range noise std (metres), per anchor if an array.
    motion, process_noise, noise, init_vel_std
        As in :class:`KalmanTracker`.
    height : float or None
        With 3-D anchors, the fixed height of the tag: the track is planar ``(x, y)``.
    gate : float or None
        Per-range outlier gate in sigma units.

    References
    ----------
    Y. Bar-Shalom, X. R. Li, T. Kirubarajan, "Estimation with Applications to Tracking and
        Navigation", Wiley, 2001, Sec. 10.3 (extended Kalman filter). DOI 10.1002/0471221279.
    J. D. Hol, F. Dijkstra, H. Luinge, T. B. Schon, "Tightly coupled UWB/IMU pose estimation",
        IEEE Int. Conf. on Ultra-Wideband (ICUWB), 2009. DOI 10.1109/ICUWB.2009.5288724.
    """

    def __init__(self, anchors=None, range_std=0.3, motion: str = "cv", process_noise: float = 0.5,
                 noise: str = "continuous", height: float | None = None, init_vel_std: float = 1.0,
                 gate: float | None = None):
        self.anchors = anchors
        self.range_std = range_std
        self.motion = motion
        self.process_noise = process_noise
        self.noise = noise
        self.height = height
        self.init_vel_std = init_vel_std
        self.gate = gate

    def _anchors(self) -> np.ndarray:
        if self.anchors is None:
            raise ValueError("ExtendedKalmanTracker needs anchors=(A, D) coordinates")
        A = np.asarray(self.anchors, dtype=np.float64)
        if A.ndim != 2 or A.shape[1] not in (2, 3):
            raise ValueError(f"anchors must be (A, 2) or (A, 3), got shape {A.shape}")
        if self.height is not None and A.shape[1] != 3:
            raise ValueError("height= needs 3-D anchors")
        return A

    def _state_dim(self) -> int:
        return self._blocks() * self.dim_

    def _planar_dim(self) -> int:
        return 2 if self.height is not None else self._anchors().shape[1]

    def _stds(self, n: int) -> np.ndarray:
        return np.broadcast_to(np.asarray(self.range_std, dtype=np.float64), (n,))

    def _ranges(self, z) -> np.ndarray:
        A = self._anchors()
        r = np.asarray(z, dtype=np.float64).reshape(-1)
        if r.shape != (len(A),):
            raise ValueError(f"expected {len(A)} ranges (one per anchor), got shape {r.shape}")
        return r

    def _initialize(self, z, spread, t) -> bool:
        A = self._anchors()
        r = self._ranges(z)
        p = multilaterate(r, A, height=self.height)
        if p is None:
            return False
        d, n = len(p), self._blocks()
        self.dim_ = d
        ok = np.isfinite(r)
        _, J = self._h(p, A[ok])
        info = J.T @ (J / self._stds(len(A))[ok, None] ** 2)
        P_pos = np.linalg.pinv(info)
        P = np.diag(np.full(n * d, float(self.init_vel_std) ** 2))
        P[:d, :d] = P_pos + 1e-9 * np.eye(d)
        self.x_ = np.concatenate([p, np.zeros((n - 1) * d)])
        self.P_, self.t_ = P, float(t)
        return True

    def _h(self, p, A):
        """Predicted ranges and their Jacobian w.r.t. the planar/spatial position."""
        q = p if self.height is None else np.r_[p, float(self.height)]
        diff = q - A
        dist = np.maximum(np.linalg.norm(diff, axis=1), 1e-9)
        return dist, diff[:, :len(p)] / dist[:, None]

    def _correct(self, z, spread):
        A = self._anchors()
        r = self._ranges(z)
        ok = np.isfinite(r)
        if not ok.any():
            return np.nan, False
        d, S = self.dim_, self._state_dim()
        pred, J = self._h(self.x_[:d], A[ok])
        var = self._stds(len(A))[ok] ** 2
        H = np.zeros((ok.sum(), S))
        H[:, :d] = J
        y = r[ok] - pred
        if self.gate is not None:  # per-range test on the diagonal of S
            s_diag = np.einsum("ij,jk,ik->i", H, self.P_, H) + var
            keep = y ** 2 <= float(self.gate) ** 2 * s_diag
            if not keep.any():
                return _nis(y, H, self.P_, np.diag(var)), False
            H, y, var = H[keep], y[keep], var[keep]
        R = np.diag(var)
        nis = _nis(y, H, self.P_, R)
        self.x_, self.P_ = _kf_correct(self.x_, self.P_, y, H, R)
        return nis, True

    def update(self, t: float, ranges=None) -> np.ndarray:
        """Online step with one range vector ``(A,)`` (None or all-NaN = predict only)."""
        n = len(self._anchors())
        self.dim_ = getattr(self, "dim_", None) or self._planar_dim()
        self._step(t, np.full(n, np.nan) if ranges is None else ranges, None)
        return self.estimate()[0]

    def filter(self, ranges, t=None) -> Prediction:
        """Forward pass over ``(T, A)`` ranges; returns positions ``(T, D)`` (NaN before the start)."""
        return self._offline(ranges, t, smooth=False)

    def smooth(self, ranges, t=None) -> Prediction:
        """Extended RTS smoother (the motion model is linear, so the RTS pass is exact given
        the EKF linearisation points)."""
        return self._offline(ranges, t, smooth=True)

    def _offline(self, ranges, t, smooth: bool) -> Prediction:
        R = np.asarray(ranges, dtype=np.float64)
        if R.ndim != 2:
            raise ValueError(f"ranges must be (T, A), got shape {R.shape}")
        self.dim_ = self._planar_dim()
        pos, s = self._sequence(R, t, np.full(len(R), np.nan), smooth=smooth)
        return Prediction(pos, spread=s)


class ConstantVelocityKF(Estimator):
    """Minimal 2-D constant-velocity Kalman filter used by :func:`track` (the 0.2 sketch API).

    State ``[x, y, vx, vy]`` in the dataset frame; the measurement is a localizer's position
    with std ``r``. Process noise: piecewise-constant white acceleration of std ``accel_std``
    (``motion_model(dt, "cv", q=accel_std**2, noise="discrete")``). :class:`KalmanTracker` is the
    general version (CA model, gating, smoothing, missing fixes).

    References
    ----------
    Y. Bar-Shalom, X. R. Li, T. Kirubarajan, "Estimation with Applications to Tracking and
        Navigation", Wiley, 2001, Sec. 6.3.2 (discrete white noise acceleration model).
        DOI 10.1002/0471221279.
    """

    def __init__(self, accel_std: float = 0.5, min_meas_std: float = 1.0):
        self.accel_std = accel_std
        self.min_meas_std = min_meas_std

    def reset(self, z, r):
        self.x_ = np.r_[np.asarray(z, dtype=np.float64), 0.0, 0.0]
        self.P_ = np.diag([r ** 2, r ** 2, 1.0, 1.0])
        return self

    def step(self, z, r, dt):
        F, Q = motion_model(dt, "cv", dim=2, q=self.accel_std ** 2, noise="discrete")
        self.x_, self.P_ = _kf_predict(self.x_, self.P_, F, Q)
        H = np.eye(2, 4)
        self.x_, self.P_ = _kf_correct(self.x_, self.P_, np.asarray(z, dtype=np.float64) - H @ self.x_, H,
                                       r ** 2 * np.eye(2))
        return self.x_[:2].copy()


def track(localizer, kf, stream, adaptive: bool = True, fixed_std: float = 4.0):
    """stream yields (t_seconds, scan (F,) dBm with NaN). Yields (t, raw fix, filtered position).

    With ``adaptive`` the fix noise is ``max(spread, kf.min_meas_std)``; a localizer without a
    (finite) ``Prediction.spread`` falls back to ``fixed_std``. The general, missing-scan
    tolerant version is ``indoorloc.apps.streaming.OnlineLocalizer``.
    """
    t_prev = None
    for t, scan in stream:
        fix = localizer.localize(np.asarray(scan)[None])  # the first axis is samples: one scan = one row
        z = fix.pos[0]
        spread = np.nan if fix.spread is None else float(fix.spread[0])
        r = max(spread, kf.min_meas_std) if adaptive and np.isfinite(spread) else fixed_std
        xy = kf.reset(z, r).x_[:2].copy() if t_prev is None else kf.step(z, r, t - t_prev)
        t_prev = t
        yield t, z, xy


__all__ = ["ConstantVelocityKF", "ExtendedKalmanTracker", "KalmanTracker", "motion_model", "multilaterate",
           "rts_smooth", "track"]
