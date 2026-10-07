"""L5 streaming: run a fitted localizer (and optionally a tracker) on a live stream of scans.

The stream is any iterable of ``(t, scan)`` pairs: ``t`` in seconds, ``scan`` one ``(F,)``
measurement vector in physical units with NaN for missing readings (the L1 contract), or
None when a scan was lost. :class:`OnlineLocalizer` turns each pair into a
:class:`StreamEstimate` -- one scan becomes one row (``scan[None]``, the first axis is
samples), is localized, and is passed to the tracker as a fix whose noise comes from
``Prediction.spread``. Missing scans (None, or fewer than ``min_readings`` finite readings) do
not reach the localizer: the tracker only predicts, or the last estimate is held. Latency
(wall-clock time of localize + track per scan) is recorded for :meth:`OnlineLocalizer.latency_stats`.

No threads, no global state: the caller drives the loop (``for est in online.run(stream)``).
"""
from __future__ import annotations

import time
from collections import Counter, deque
from typing import Iterable, Iterator, NamedTuple

import numpy as np

from ..core import Estimator, Prediction, clone


class StreamEstimate(NamedTuple):
    """The output for one ``(t, scan)`` of the stream.

    t        time of the scan (seconds)
    pos      (D,) estimated position (tracked if a tracker is set, else the raw fix); NaN
             until the first valid scan
    fix      (D,) the localizer's own estimate for this scan (NaN if the scan was missing)
    floor    floor label (majority of the last ``floor_window`` fixes) or None
    spread   uncertainty scale of ``pos`` (tracker spread, else the fix's spread)
    latency  seconds spent on this scan (localize + track)
    missing  True if the scan was missing or had too few readings
    """

    t: float
    pos: np.ndarray
    fix: np.ndarray
    floor: int | None
    spread: float
    latency: float
    missing: bool


class OnlineLocalizer(Estimator):
    """Online localization of a stream of scans with an optional tracker.

    Parameters
    ----------
    localizer : fitted L3 localizer
        Anything with ``localize(X) -> Prediction`` (e.g. ``create_model("wknn").fit(train)``,
        with its preprocessing in a ``LocalizerPipeline``).
    tracker : KalmanTracker, ParticleFilter or None
        Any object with ``update(t, z, spread) -> position`` and ``estimate() -> (pos, spread)``,
        and ``reset(t)``. It is a template: :meth:`reset` / :meth:`run` work on a fresh clone,
        ``tracker_``, so the parameter is never modified and two online localizers never share
        state. None = raw fixes.
    min_readings : int
        A scan with fewer finite readings is treated as missing (not localized).
    floor_window : int
        Floor reported = majority vote of the last ``floor_window`` fixes (ties: the most
        recent), which suppresses single-scan floor flips.
    latency_window : int
        Number of recent latencies kept for :meth:`latency_stats`.
    """

    _requires_fit = False

    def __init__(self, localizer=None, tracker=None, min_readings: int = 1, floor_window: int = 1,
                 latency_window: int = 10000):
        self.localizer = localizer
        self.tracker = tracker
        self.min_readings = min_readings
        self.floor_window = floor_window
        self.latency_window = latency_window

    def reset(self, t: float = 0.0):
        """Forget the stream: tracker state, floor history, last estimate and latencies."""
        if self.localizer is None:
            raise ValueError("OnlineLocalizer needs a fitted localizer")
        self.tracker_ = None
        if self.tracker is not None:
            self.tracker_ = clone(self.tracker)
            self.tracker_.reset(t)
        self.floors_ = deque(maxlen=max(1, int(self.floor_window)))
        self.latencies_ = deque(maxlen=max(1, int(self.latency_window)))
        self.last_ = None  # (pos, spread) of the last output, held over missing scans without a tracker
        self.n_scans_ = self.n_missing_ = 0
        return self

    def _is_missing(self, scan) -> bool:
        if scan is None:
            return True
        x = np.asarray(scan)
        if x.size == 0:
            return True
        finite = np.isfinite(x).sum() if x.dtype.kind in "fc" else x.size
        return finite < int(self.min_readings)

    def _floor(self) -> int | None:
        if not self.floors_:
            return None
        counts = Counter(self.floors_)
        best = max(counts.values())
        for f in reversed(self.floors_):  # ties go to the most recent label
            if counts[f] == best:
                return int(f)
        return None

    def update(self, t: float, scan) -> StreamEstimate:
        """Process one scan (``(F,)`` array, or None if lost) observed at time ``t``."""
        if not hasattr(self, "latencies_"):
            self.reset(t)
        start = time.perf_counter()
        missing = self._is_missing(scan)
        self.n_scans_ += 1
        fix_spread = float("nan")
        if missing:
            self.n_missing_ += 1
            fix = None
        else:
            pred = self.localizer.localize(np.asarray(scan)[None])
            fix = np.asarray(pred.pos[0], dtype=np.float64)
            fix_spread = float(pred.spread[0]) if pred.spread is not None else float("nan")
            if pred.floor is not None:
                self.floors_.append(int(pred.floor[0]))
        if self.tracker_ is not None:
            self.tracker_.update(t, fix, None if np.isnan(fix_spread) else fix_spread)
            pos, spread = self.tracker_.estimate()
        elif fix is not None:
            pos, spread = fix, fix_spread
        elif self.last_ is not None:
            pos, spread = self.last_
        else:
            pos, spread = np.full(2, np.nan), float("nan")
        pos = np.asarray(pos, dtype=np.float64)
        self.last_ = (pos, spread)
        latency = time.perf_counter() - start
        self.latencies_.append(latency)
        fix_out = np.full(pos.shape, np.nan) if fix is None else fix
        return StreamEstimate(float(t), pos.copy(), fix_out, self._floor(), float(spread), latency, missing)

    def run(self, stream: Iterable) -> Iterator[StreamEstimate]:
        """Generator: reset, then one :class:`StreamEstimate` per ``(t, scan)`` of ``stream``."""
        first = True
        for t, scan in stream:
            if first:
                self.reset(t)
                first = False
            yield self.update(t, scan)

    def _get_state(self) -> dict:
        state = super()._get_state()  # the deques are saved as lists (arrays only, no pickle)
        return {k: list(v) if isinstance(v, deque) else v for k, v in state.items()}

    def _set_state(self, state: dict) -> None:
        state = dict(state)
        if "floors_" in state:
            state["floors_"] = deque(state["floors_"], maxlen=max(1, int(self.floor_window)))
        if "latencies_" in state:
            state["latencies_"] = deque(state["latencies_"], maxlen=max(1, int(self.latency_window)))
        super()._set_state(state)

    def latency_stats(self) -> dict:
        """Latency over the recent scans: ``n``, ``mean_ms``, ``p50_ms``, ``p95_ms``, ``max_ms``,
        plus ``n_scans`` and ``n_missing`` since the last reset."""
        lat = np.asarray(getattr(self, "latencies_", ()), dtype=np.float64) * 1e3
        stats = {"n": len(lat), "n_scans": getattr(self, "n_scans_", 0), "n_missing": getattr(self, "n_missing_", 0)}
        if len(lat):
            p50, p95 = np.percentile(lat, [50, 95])
            stats.update(mean_ms=float(lat.mean()), p50_ms=float(p50), p95_ms=float(p95), max_ms=float(lat.max()))
        else:
            stats.update(mean_ms=float("nan"), p50_ms=float("nan"), p95_ms=float("nan"), max_ms=float("nan"))
        return stats


def stack_estimates(estimates: Iterable[StreamEstimate]) -> tuple[np.ndarray, Prediction]:
    """``(t (N,), Prediction)`` from a sequence of stream estimates, for L4 evaluation
    (``evaluate(truth, pred)``). ``floor`` is kept when every estimate has one."""
    est = list(estimates)
    if not est:
        return np.zeros(0), Prediction(np.zeros((0, 2)))
    floors = [e.floor for e in est]
    return (np.array([e.t for e in est], dtype=np.float64),
            Prediction(np.stack([e.pos for e in est]), floor=None if None in floors else floors,
                       spread=np.array([e.spread for e in est], dtype=np.float64)))


__all__ = ["OnlineLocalizer", "StreamEstimate", "stack_estimates"]
