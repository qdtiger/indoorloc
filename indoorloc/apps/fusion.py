"""L5 sensor fusion: pedestrian dead reckoning + position fixes in a particle filter.

PDR is precise over a few metres but drifts (heading error grows, the initial heading is
unknown without a compass); fingerprinting or ranging fixes do not drift but scatter by
metres. :class:`PDRFusion` moves a particle cloud with every detected step (length and
heading from :mod:`indoorloc.apps.pdr`), weights it with every fix of any L3 localizer (noise
from ``Prediction.spread``) and lets the floor plan kill particles that walk through walls.
Each particle carries a heading bias, so the cloud also estimates the heading offset between
the gyro-integrated heading and the map.

References
----------
F. Evennou, F. Marx, "Advanced integration of WiFi and inertial navigation systems for indoor
    mobile positioning", EURASIP Journal on Advances in Signal Processing 2006:086706, 2006.
    DOI 10.1155/ASP/2006/86706.
O. Woodman, R. Harle, "Pedestrian localisation for indoor environments", Proc. UbiComp 2008,
    pp. 114-123. DOI 10.1145/1409635.1409651.
"""
from __future__ import annotations

import numpy as np

from ..core import Estimator, Prediction, clone
from .particle import ParticleFilter


def _default_filter() -> ParticleFilter:
    return ParticleFilter(n_particles=1000, step_length_std=0.1, heading_std=0.05, heading_drift_std=0.005,
                          motion_std=0.0, recovery=(0.05, 0.5))


class PDRFusion(Estimator):
    """Particle-filter fusion of PDR steps with position fixes from any localizer.

    Events are processed in time order (a step before a fix at the same time):

    * step ``(length, heading)`` -> :meth:`ParticleFilter.step` (per-particle length and heading
      noise, heading bias, wall constraint);
    * fix ``z`` with spread -> :meth:`ParticleFilter.correct` (Gaussian likelihood).

    The cloud starts at ``start`` (``start_std``), or else at the first fix with std
    ``init_std`` (default: the fix's own measurement std). Steps before the start are dropped.

    Parameters
    ----------
    particle_filter : ParticleFilter or None
        Template holding the particle count, noise levels, ``floor_map``/``floor`` and seed;
        it is cloned on every :meth:`run` / :meth:`reset` (the working copy is ``filter_``).
        Default: 1000 particles, step length std 0.1 m, heading std 0.05 rad/step, heading
        drift 0.005 rad/step, no random walk between steps, and augmented-MCL recovery
        ``(0.05, 0.5)``: fingerprinting fixes have gross outliers (a first fix tens of metres
        off), and on real traces the floor plan only helped together with recovery.
    localizer : fitted L3 localizer or None
        Turns raw scans into fixes in :meth:`run` when ``fixes`` is an array of scans.
    init_std : float or None
        Std (m) of the initial cloud around the first fix.
    heading_bias_std : float or None
        Prior std (rad) of the offset between PDR heading and map heading; None = unknown
        (uniform on ``[-pi, pi)``, the fixes resolve it).
    random_state : int, Generator or None
        Seed of the working filter; overrides the template's ``random_state`` when not None.
        With an int, two runs on the same inputs give identical results.

    References
    ----------
    F. Evennou, F. Marx, "Advanced integration of WiFi and inertial navigation systems for
        indoor mobile positioning", EURASIP J. Adv. Signal Process. 2006:086706, 2006.
        DOI 10.1155/ASP/2006/86706.
    O. Woodman, R. Harle, "Pedestrian localisation for indoor environments", UbiComp 2008.
        DOI 10.1145/1409635.1409651.
    """

    _requires_fit = False

    def __init__(self, particle_filter=None, localizer=None, init_std: float | None = None,
                 heading_bias_std: float | None = None, random_state=None):
        self.particle_filter = particle_filter
        self.localizer = localizer
        self.init_std = init_std
        self.heading_bias_std = heading_bias_std
        self.random_state = random_state

    # ---------------------------------------------------------------- online API
    def reset(self, t: float = 0.0, *, start=None, start_std: float = 1.0):
        """New working filter; with ``start`` the cloud is drawn around it now, otherwise the
        first fix will start it."""
        template = _default_filter() if self.particle_filter is None else self.particle_filter
        pf = clone(template)
        if self.random_state is not None:
            pf.set_params(random_state=self.random_state)
        self.filter_ = pf.reset(t)
        if start is not None:
            self.filter_.initialize(start, start_std, heading_bias_std=self.heading_bias_std, t=t)
        return self

    def update(self, t: float, *, step=None, fix=None, spread=None) -> np.ndarray:
        """One event at time ``t``: ``step=(length, heading)`` or ``fix=z`` (with its
        ``spread``). Returns the estimated position (NaN before the cloud starts)."""
        if not hasattr(self, "filter_"):
            self.reset(t)
        pf = self.filter_
        if step is not None and pf.initialized:
            pf.step(*step)
        if fix is not None:
            z = np.asarray(fix, dtype=np.float64).reshape(-1)
            if np.all(np.isfinite(z)):
                if pf.initialized:
                    pf.correct(z, spread)
                else:
                    std = pf._std(spread) if self.init_std is None else float(self.init_std)
                    pf.initialize(z, std, heading_bias_std=self.heading_bias_std, t=t)
        pf.t_ = float(t)
        return pf.estimate()[0]

    def estimate(self) -> tuple[np.ndarray, float]:
        return self.filter_.estimate() if hasattr(self, "filter_") else (np.full(2, np.nan), float("nan"))

    # ---------------------------------------------------------------- offline API
    def _fixes(self, fixes, fix_t):
        if isinstance(fixes, Prediction):
            pos, spread = fixes.pos, fixes.spread
        else:
            X = np.asarray(fixes)
            if self.localizer is not None:
                pred = self.localizer.localize(X)
                pos, spread = pred.pos, pred.spread
            else:
                pos, spread = np.asarray(X, dtype=np.float64), None
        pos = np.asarray(pos, dtype=np.float64).reshape(len(pos), -1)
        spread = np.full(len(pos), np.nan) if spread is None else np.asarray(spread, dtype=np.float64)
        fix_t = np.asarray(fix_t, dtype=np.float64).reshape(-1)
        if fix_t.shape != (len(pos),):
            raise ValueError(f"fix_t must have one time per fix ({len(pos)}), got shape {fix_t.shape}")
        return pos, spread, fix_t

    def run(self, steps, fixes=None, fix_t=None, *, start=None, start_std: float = 1.0,
            t0: float | None = None) -> tuple[np.ndarray, Prediction]:
        """Fuse a recording.

        ``steps`` is a ``StepTrack`` (from ``PDR.run``) or a tuple ``(t, length, heading)`` of
        ``(S,)`` arrays. ``fixes`` is a ``Prediction``, an array of positions ``(N, D)``, or,
        with a ``localizer``, an array of scans ``(N, F)``; ``fix_t`` ``(N,)`` their times.
        Returns ``(t, Prediction)``, not a bare ``Prediction``: the estimate (mean and spread of
        the cloud) after every event, steps and fixes merged in time order (NaN rows before the
        cloud starts). The rows are events, not the fixes, so score them against the truth at
        ``t`` (e.g. ``evaluate(truth_at(t), prediction)``).
        """
        if hasattr(steps, "length") and hasattr(steps, "heading"):
            s_t, s_len, s_head = steps.t, steps.length, steps.heading
        else:
            s_t, s_len, s_head = (np.asarray(a, dtype=np.float64).reshape(-1) for a in steps)
        s_t = np.asarray(s_t, dtype=np.float64)
        if fixes is None:
            f_pos, f_spread, f_t = np.zeros((0, 2)), np.zeros(0), np.zeros(0)
        else:
            f_pos, f_spread, f_t = self._fixes(fixes, fix_t)
        # event order: by time, a step before a fix at equal time (stable, deterministic)
        times = np.r_[s_t, f_t]
        kind = np.r_[np.zeros(len(s_t), np.int8), np.ones(len(f_t), np.int8)]
        order = np.lexsort((np.arange(len(times)), kind, times))
        first = float(times[order[0]]) if len(order) else 0.0
        self.reset(first if t0 is None else float(t0), start=start, start_std=start_std)
        out = np.full((len(order), 2), np.nan)
        spread = np.full(len(order), np.nan)
        for e, i in enumerate(order):
            if kind[i] == 0:
                self.update(times[i], step=(s_len[i], s_head[i]))
            else:
                j = i - len(s_t)
                self.update(times[i], fix=f_pos[j], spread=f_spread[j])
            if self.filter_.initialized:
                out[e], spread[e] = self.filter_.estimate()
        return times[order], Prediction(out, spread=spread)


__all__ = ["PDRFusion"]
