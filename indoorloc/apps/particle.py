"""L5 particle filter: random-walk or PDR motion, Gaussian likelihoods, wall constraints.

A cloud of weighted particles represents the position of one pedestrian or device. Motion
is a random walk (between the scans of a localizer) or one pedestrian step of given length and
heading (PDR). A particle whose move crosses a wall of the :class:`~indoorloc.apps.maps.FloorMap`
or leaves its bounds dies (weight 0), which is how floor plans correct heading drift. Fixes
from any localizer weight the particles with a Gaussian likelihood whose std comes from
``Prediction.spread``; ranges to known anchors use a product of Gaussians. The cloud is
resampled systematically when the effective sample size drops below a threshold.

Everything random draws from ``np.random.default_rng(random_state)``, created when the cloud
is initialised: two runs with the same seed and inputs give identical results.

References
----------
N. J. Gordon, D. J. Salmond, A. F. M. Smith, "Novel approach to nonlinear/non-Gaussian
    Bayesian state estimation", IEE Proceedings F 140(2):107-113, 1993. DOI 10.1049/ip-f-2.1993.0015.
G. Kitagawa, "Monte Carlo filter and smoother for non-Gaussian nonlinear state space models",
    Journal of Computational and Graphical Statistics 5(1):1-25, 1996. DOI 10.1080/10618600.1996.10474692.
O. Woodman, R. Harle, "Pedestrian localisation for indoor environments", Proc. UbiComp 2008,
    pp. 114-123. DOI 10.1145/1409635.1409651.
"""
from __future__ import annotations

import numpy as np

from ..core import Estimator, Prediction
from .tracking import _times

# bit generators whose state a saved model may restore (no arbitrary attribute lookup from a file)
_BIT_GENERATORS = ("PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64")


def effective_sample_size(weights) -> float:
    """``1 / sum(w_i^2)`` of normalised weights: N for uniform weights, 1 for a single particle."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return float(1.0 / np.dot(w, w))


def systematic_resample(weights, rng=None, *, u: float | None = None) -> np.ndarray:
    """Indices of a systematic resample (Kitagawa 1996): one uniform offset ``u``, points
    ``(k + u) / N``. Particle i is copied ``floor(N w_i)`` or ``ceil(N w_i)`` times; a zero
    weight is never selected. Pass ``u`` in ``[0, 1)`` for a deterministic draw."""
    w = np.asarray(weights, dtype=np.float64)
    n = len(w)
    if n == 0 or not np.all(w >= 0) or not w.sum() > 0:
        raise ValueError("weights must be non-negative with a positive sum")
    c = np.cumsum(w / w.sum())
    c[-1] = 1.0
    if u is None:
        u = np.random.default_rng(rng).random()  # a Generator passes through unchanged
    points = (np.arange(n) + float(u)) / n
    return np.minimum(np.searchsorted(c, points, side="right"), n - 1)


class ParticleFilter(Estimator):
    """Sequential Monte Carlo position filter with map constraints.

    State per particle: position ``(D,)`` and a heading bias (radians, used by PDR steps; it
    lets the cloud learn an unknown heading offset). Motion: :meth:`predict` (random walk,
    displacement ``N(0, motion_std^2 dt I)``) or :meth:`step` (one step: length
    ``+ N(0, step_length_std^2)``, heading ``+ bias + N(0, heading_std^2)``, bias random walk
    ``N(0, heading_drift_std^2)``). With a ``floor_map`` a particle whose move crosses a wall
    of ``floor`` (or ends outside the bounds) dies and the cloud is resampled at once, so no
    dead particle lingers behind a wall; if every particle would die, the move is cancelled
    for the whole cloud and counted in ``n_depleted_``. Measurements:
    :meth:`correct` (a fix, Gaussian with std from ``spread`` as in ``KalmanTracker``) and
    :meth:`correct_ranges` (ranges to ``anchors``). After every update the cloud is
    resampled (systematic) when ``ESS < resample_threshold * n_particles``.

    ``recovery=(alpha_slow, alpha_fast)`` enables the augmented-MCL recovery of Thrun et al.
    for fixes: short- and long-term averages of the fix's mean likelihood under the cloud
    (``sum w_i exp(-d_i^2 / 2 sigma^2)``, unnormalised so fixes of different precision
    compare) give an injection rate ``max(0, 1 - w_fast / w_slow)``; that fraction of the
    resampled cloud is redrawn around the fix (sensor resetting). Both averages start at 1/2
    (the expected value for a fix consistent with a tight cloud) rather than 0 as in the book,
    so a cloud started at a wrong fix also recovers once the fixes consistently disagree with
    it. Off by default (plain SIR).

    Parameters
    ----------
    n_particles : int
    motion_std : float
        Random-walk intensity, metres per sqrt(second).
    step_length_std, heading_std, heading_drift_std : float
        PDR step noise: metres, radians per step, radians per step.
    meas_std, use_spread, spread_scale, min_meas_std : float, bool, float, float
        Fix noise, as in :class:`~indoorloc.apps.tracking.KalmanTracker`.
    range_std : float or array-like (A,)
        Range noise std for :meth:`correct_ranges`.
    anchors : array-like (A, D) or None
        Anchor coordinates for range updates, in the frame of the particles; 3-D anchors
        with a planar cloud need ``height``.
    height : float or None
        Fixed height of the tag when ``anchors`` are 3-D and the cloud is planar (as in
        :class:`~indoorloc.apps.tracking.ExtendedKalmanTracker`).
    floor_map : FloorMap or None
        Walls and bounds; None = open space.
    floor : int or None
        Floor whose walls constrain the particles (None: every wall of the map).
    resample_threshold : float
        Fraction of ``n_particles``; 1.0 resamples after every update, 0 never.
    recovery : (float, float) or None
        Augmented-MCL averaging rates ``(alpha_slow, alpha_fast)`` with
        ``0 < alpha_slow << alpha_fast <= 1``, e.g. ``(0.05, 0.5)``; None disables.
    random_state : int, Generator or None

    Online use: :meth:`update` ``(t, z, spread)`` initialises the cloud around the first fix;
    or call :meth:`initialize` for an explicit prior (uniform over the map when ``mean`` is
    None). :meth:`predict_to` ``(t)`` moves the cloud to a time without a fix. Every ``t`` is
    an absolute time in seconds; only :meth:`predict` ``(dt)`` takes an increment (and leaves
    the clock alone), unlike ``KalmanTracker.predict(t)``: prefer :meth:`predict_to`.
    Offline: :meth:`filter` returns a ``Prediction`` for a sequence of fixes.
    ``save`` / ``load_model`` keep the cloud and the generator state (no pickle), so a loaded
    filter continues the same random sequence; the ``floor_map`` is saved with it
    (``FloorMap.to_dict``, no pickle).

    References
    ----------
    N. J. Gordon, D. J. Salmond, A. F. M. Smith, "Novel approach to nonlinear/non-Gaussian
        Bayesian state estimation", IEE Proceedings F 140(2):107-113, 1993. DOI 10.1049/ip-f-2.1993.0015.
    G. Kitagawa, "Monte Carlo filter and smoother for non-Gaussian nonlinear state space
        models", J. Computational and Graphical Statistics 5(1):1-25, 1996.
        DOI 10.1080/10618600.1996.10474692.
    O. Woodman, R. Harle, "Pedestrian localisation for indoor environments", Proc. UbiComp
        2008, pp. 114-123. DOI 10.1145/1409635.1409651.
    S. Thrun, W. Burgard, D. Fox, "Probabilistic Robotics", MIT Press, 2005, Sec. 8.3.5
        (augmented MCL); S. Lenser, M. Veloso, "Sensor resetting localization for poorly
        modelled mobile robots", IEEE ICRA 2000, pp. 1225-1232. DOI 10.1109/ROBOT.2000.844766.
    """

    _requires_fit = False

    def __init__(self, n_particles: int = 1000, motion_std: float = 1.0, step_length_std: float = 0.1,
                 heading_std: float = 0.1, heading_drift_std: float = 0.0, meas_std: float = 3.0,
                 use_spread: bool = True, spread_scale: float = 1.0, min_meas_std: float = 1.0,
                 range_std=0.5, anchors=None, height: float | None = None, floor_map=None,
                 floor: int | None = None, resample_threshold: float = 0.5, recovery=None, random_state=None):
        self.n_particles = n_particles
        self.motion_std = motion_std
        self.step_length_std = step_length_std
        self.heading_std = heading_std
        self.heading_drift_std = heading_drift_std
        self.meas_std = meas_std
        self.use_spread = use_spread
        self.spread_scale = spread_scale
        self.min_meas_std = min_meas_std
        self.range_std = range_std
        self.anchors = anchors
        self.height = height
        self.floor_map = floor_map
        self.floor = floor
        self.resample_threshold = resample_threshold
        self.recovery = recovery
        self.random_state = random_state

    # ---------------------------------------------------------------- state
    @property
    def initialized(self) -> bool:
        return getattr(self, "particles_", None) is not None

    def reset(self, t: float = 0.0):
        """Forget the cloud; the next fix given to :meth:`update` initialises it at time ``t``."""
        self.particles_ = self.weights_ = self.heading_bias_ = None
        self.t_ = float(t)
        self.n_depleted_ = 0
        return self

    def initialize(self, mean=None, std=1.0, *, heading_bias: float = 0.0, heading_bias_std: float | None = 0.0,
                   t: float = 0.0):
        """Draw a new cloud: ``N(mean, std^2 I)`` (``std`` scalar or per axis), or uniform over
        the map bounds when ``mean`` is None. Heading bias ``N(heading_bias, heading_bias_std^2)``;
        ``heading_bias_std=None`` draws it uniformly on ``[-pi, pi)`` (heading unknown)."""
        n = int(self.n_particles)
        if n < 1:
            raise ValueError(f"n_particles must be >= 1, got {self.n_particles}")
        self.rng_ = np.random.default_rng(self.random_state)
        if mean is None:
            bounds = None if self.floor_map is None else self.floor_map.bounds
            if bounds is None:
                raise ValueError("initialize(mean=None) needs a floor_map with bounds (uniform prior)")
            p = self.rng_.uniform(bounds[:2], bounds[2:], size=(n, 2))
        else:
            mean = np.asarray(mean, dtype=np.float64).reshape(-1)
            p = mean + np.asarray(std, dtype=np.float64) * self.rng_.standard_normal((n, len(mean)))
        if heading_bias_std is None:
            bias = self.rng_.uniform(-np.pi, np.pi, n)
        else:
            bias = float(heading_bias) + float(heading_bias_std) * self.rng_.standard_normal(n)
        w = np.full(n, 1.0 / n)
        if self.floor_map is not None:
            w[~self.floor_map.contains(p[:, :2])] = 0.0
            if w.sum() == 0:
                raise ValueError("every initial particle lies outside the map bounds")
            w /= w.sum()
        self.particles_, self.weights_, self.heading_bias_ = p, w, bias
        self.t_, self.n_depleted_ = float(t), 0
        # both averages start at 1/2: E[exp(-|e|^2 / 2 sigma^2)] for a fix error e ~ N(0, sigma^2 I2)
        # (a fix consistent with a tight cloud), so a cloud started at a wrong fix also recovers
        self.w_slow_ = self.w_fast_ = 0.5
        self.n_injected_ = 0
        return self

    def estimate(self) -> tuple[np.ndarray, float]:
        """Weighted mean position ``(D,)`` and spread ``sqrt(sum w |p - mean|^2)`` (NaN if empty)."""
        if not self.initialized:
            return np.full(2, np.nan), float("nan")
        mean = self.weights_ @ self.particles_
        d = self.particles_ - mean
        return mean, float(np.sqrt(self.weights_ @ np.einsum("ij,ij->i", d, d)))

    def _check(self):
        if not self.initialized:
            raise ValueError("the particle cloud is not initialised; call initialize() or update() first")

    def _std(self, spread) -> float:
        if self.use_spread and spread is not None and np.isfinite(spread):
            return max(float(spread) * self.spread_scale, float(self.min_meas_std))
        return float(self.meas_std)

    # ---------------------------------------------------------------- motion
    def _move(self, disp: np.ndarray) -> None:
        old = self.particles_
        new = old.copy()
        new[:, :disp.shape[1]] += disp
        if self.floor_map is not None:
            alive = self.weights_ > 0
            ok = np.zeros(len(new), dtype=bool)
            ok[alive] = self.floor_map.valid_moves(old[alive, :2], new[alive, :2], self.floor)
            if not ok.any():  # the whole cloud would die: cancel the move, keep the weights
                self.n_depleted_ += 1
                return
            w = np.where(ok, self.weights_, 0.0)
            self.weights_ = w / w.sum()
            if not ok.all():  # dead particles are replaced at once: none lingers behind a wall
                self.particles_ = new
                self._resample()
                return
        self.particles_ = new
        self._maybe_resample()

    def predict(self, dt: float = 1.0) -> np.ndarray:
        """Random-walk motion over an **increment** of ``dt`` seconds; returns the estimated position.

        ``dt`` is a duration, not a time stamp (``KalmanTracker.predict(t)`` takes an absolute
        time), and the filter clock ``t_`` used by :meth:`update` does not move. For an absolute
        time, as in the rest of the tracker protocol, use :meth:`predict_to`.
        """
        self._check()
        dt = float(dt)
        if dt < 0:
            raise ValueError(f"dt must be >= 0, got {dt}")
        if dt > 0 and self.motion_std > 0:
            self._move(float(self.motion_std) * np.sqrt(dt) * self.rng_.standard_normal(self.particles_.shape))
        return self.estimate()[0]

    def predict_to(self, t: float) -> np.ndarray:
        """Random walk to the **absolute** time ``t`` (seconds, the clock of :meth:`update`)
        without a fix, then set the clock to ``t``; returns the estimated position. Before the
        cloud exists it only moves the clock. The same name and meaning as
        :meth:`KalmanTracker.predict_to <indoorloc.apps.tracking.KalmanTracker.predict_to>`."""
        if not self.initialized:
            self.t_ = float(t)
            return self.estimate()[0]
        dt = float(t) - float(getattr(self, "t_", t))
        if dt < 0:
            raise ValueError(f"time went backwards: {t} < {self.t_}")
        self.predict(dt)
        self.t_ = float(t)
        return self.estimate()[0]

    def step(self, length: float, heading: float) -> np.ndarray:
        """One PDR step of ``length`` metres along ``heading`` (radians, counter-clockwise from
        +x); every particle adds its heading bias and noise. Returns the estimated position."""
        self._check()
        n = len(self.particles_)
        L = float(length) + float(self.step_length_std) * self.rng_.standard_normal(n)
        th = float(heading) + self.heading_bias_ + float(self.heading_std) * self.rng_.standard_normal(n)
        if self.heading_drift_std:
            self.heading_bias_ = self.heading_bias_ + float(self.heading_drift_std) * self.rng_.standard_normal(n)
        self._move(np.stack([L * np.cos(th), L * np.sin(th)], axis=1))
        return self.estimate()[0]

    # ---------------------------------------------------------------- measurements
    def _reweight(self, loglik: np.ndarray) -> None:
        with np.errstate(divide="ignore"):
            lw = np.log(self.weights_) + loglik
        top = lw.max()
        if not np.isfinite(top):
            return  # no particle alive to weight: keep the cloud as it is
        w = np.exp(lw - top)
        self.weights_ = w / w.sum()
        self._maybe_resample()

    def correct(self, z, spread=None) -> np.ndarray:
        """Weight by a position fix ``z`` (``(D,)``, NaN = ignore) with std from ``spread``."""
        self._check()
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if np.all(np.isfinite(z)):
            std = self._std(spread)
            d = self.particles_[:, :len(z)] - z
            loglik = -0.5 * np.einsum("ij,ij->i", d, d) / std ** 2
            inject = self._recovery_rate(loglik)
            self._reweight(loglik)
            if inject > 0:
                self._inject(z, std, inject)
        return self.estimate()[0]

    def _recovery_rate(self, loglik: np.ndarray) -> float:
        """Augmented MCL: update the slow/fast likelihood averages, return the injection rate."""
        if self.recovery is None:
            return 0.0
        slow, fast = (float(a) for a in self.recovery)
        w_avg = float(self.weights_ @ np.exp(loglik))
        self.w_slow_ += slow * (w_avg - self.w_slow_)
        self.w_fast_ += fast * (w_avg - self.w_fast_)
        return max(0.0, 1.0 - self.w_fast_ / self.w_slow_) if self.w_slow_ > 0 else 0.0

    def _inject(self, z: np.ndarray, std: float, rate: float) -> None:
        """Resample, then redraw a ``rate`` fraction of the particles from ``N(z, std^2 I)``."""
        self._resample()
        n = len(self.weights_)
        m = int(self.rng_.binomial(n, min(rate, 1.0)))
        if m == 0:
            return
        idx = self.rng_.choice(n, m, replace=False)
        new = self.particles_[idx].copy()
        new[:, :len(z)] = z + std * self.rng_.standard_normal((m, len(z)))
        if self.floor_map is not None:  # keep only draws inside the plan
            ok = self.floor_map.contains(new[:, :2])
            idx, new = idx[ok], new[ok]
        self.particles_[idx] = new
        self.heading_bias_[idx] = self.heading_bias_[self.rng_.integers(0, n, len(idx))]
        self.n_injected_ += len(idx)

    def correct_ranges(self, ranges) -> np.ndarray:
        """Weight by ranges ``(A,)`` to ``anchors`` (NaN = anchor not heard)."""
        self._check()
        if self.anchors is None:
            raise ValueError("correct_ranges needs anchors=(A, D)")
        A = np.asarray(self.anchors, dtype=np.float64)
        r = np.asarray(ranges, dtype=np.float64).reshape(-1)
        if A.ndim != 2 or r.shape != (len(A),):
            raise ValueError(f"expected anchors (A, D) and {len(A)} ranges, got shapes {A.shape} and {r.shape}")
        P, d = self.particles_, self.particles_.shape[1]
        if A.shape[1] == d + 1 and self.height is not None:  # planar cloud, 3-D anchors
            P = np.column_stack([P, np.full(len(P), float(self.height))])
        elif A.shape[1] != d:
            raise ValueError(f"anchors are {A.shape[1]}-D but the particles are {d}-D"
                             + ("; pass height= (the tag height) for 3-D anchors and a planar cloud"
                                if A.shape[1] == d + 1 else ""))
        ok = np.isfinite(r)
        if ok.any():
            std = np.broadcast_to(np.asarray(self.range_std, dtype=np.float64), (len(A),))[ok]
            dist = np.linalg.norm(P[:, None, :] - A[ok][None], axis=2)
            self._reweight(-0.5 * (((dist - r[ok]) / std) ** 2).sum(axis=1))
        return self.estimate()[0]

    def _maybe_resample(self) -> None:
        if effective_sample_size(self.weights_) < float(self.resample_threshold) * len(self.weights_):
            self._resample()

    def _resample(self) -> None:
        n = len(self.weights_)
        idx = systematic_resample(self.weights_, self.rng_)
        self.particles_ = self.particles_[idx]
        self.heading_bias_ = self.heading_bias_[idx]
        self.weights_ = np.full(n, 1.0 / n)

    # ---------------------------------------------------------------- persistence
    def _get_state(self) -> dict:
        """The cloud and the generator's state (a JSON dict), so a saved filter resumes the same
        random sequence: ``load_model(pf.save(path))`` continues exactly like ``pf``."""
        state = super()._get_state()
        if isinstance(state.get("rng_"), np.random.Generator):
            state["rng_"] = state["rng_"].bit_generator.state
        return state

    def _set_state(self, state: dict) -> None:
        state = dict(state)
        saved = state.get("rng_")
        if isinstance(saved, dict):
            name = saved.get("bit_generator")
            if name not in _BIT_GENERATORS:
                raise ValueError(f"unknown bit generator {name!r} in a saved ParticleFilter")
            bits = getattr(np.random, name)()
            bits.state = saved
            state["rng_"] = np.random.Generator(bits)
        super()._set_state(state)

    # ---------------------------------------------------------------- online / offline
    def update(self, t: float, z=None, spread=None) -> np.ndarray:
        """Online step (the tracker protocol of ``OnlineLocalizer``): random walk to time ``t``,
        then weight by fix ``z`` (None/NaN = no fix). The first valid fix initialises the cloud
        as ``N(z, std^2 I)``. Returns the estimated position."""
        valid = z is not None and np.all(np.isfinite(np.asarray(z, dtype=np.float64)))
        if not self.initialized:
            if not valid:
                self.t_ = float(t)
                return self.estimate()[0]
            z0 = np.array(z, dtype=np.float64).reshape(-1)
            bounds = None if self.floor_map is None else self.floor_map.bounds
            if bounds is not None:  # a fix outside the plan starts the cloud at the nearest edge
                z0[:2] = np.clip(z0[:2], bounds[:2], bounds[2:])
            self.initialize(z0, self._std(spread), t=t)
            return self.estimate()[0]
        self.predict_to(t)
        if valid:
            self.correct(z, spread)
        return self.estimate()[0]

    def filter(self, z, t=None, spread=None) -> Prediction:
        """Offline pass over fixes ``(T, D)`` or a ``Prediction`` (random-walk motion).
        Returns a Prediction with the weighted mean and spread per row (NaN before the first fix)."""
        if isinstance(z, Prediction):
            labels = {"floor": z.floor, "building": z.building, "ids": z.ids}
            spread, z = (z.spread if spread is None else spread), z.pos
        else:
            labels = {}
        Z = np.asarray(z, dtype=np.float64)
        Z = Z[:, None] if Z.ndim == 1 else Z
        T = len(Z)
        t = _times(t, T)
        spread = np.full(T, np.nan) if spread is None else np.broadcast_to(np.asarray(spread, np.float64), (T,))
        self.reset(t[0] if T else 0.0)
        pos, sp = np.full((T, Z.shape[1]), np.nan), np.full(T, np.nan)
        for k in range(T):
            self.update(t[k], Z[k], spread[k])
            if self.initialized:
                pos[k], sp[k] = self.estimate()
        return Prediction(pos, spread=sp, **labels)


__all__ = ["ParticleFilter", "effective_sample_size", "systematic_resample"]
