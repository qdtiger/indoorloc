"""Geometric measurements with noise: ToA / RTT / UWB ranges, TDoA and AoA bearings.

Every generator takes positions ``pos`` (N, D) and anchor positions ``anchors`` (A, D) in the
same frame and returns an (N, A) (TDoA: (N, A - 1)) float64 array in the layout of
``CONTRACTS.md`` section 2: metres for ``ranges`` / ``tdoa``, radians for ``aoa``.
With ``noise_std=0`` and no NLOS the output equals the geometry exactly.

Measurement models
------------------
range   ``r = ||p - a|| + n + b``, ``n ~ N(0, sigma^2)``; ``b = 0`` on line-of-sight links and
        ``b ~ Exp(mean)`` (always positive) on non-line-of-sight links, where the first
        detected path is a reflection or a slowed penetration. Ranges are clipped at 0.
        This is the usual UWB / RTT error model (Gezici et al. 2005; Alsindi et al. 2009).
tdoa    ``d_i = r_i - r_0`` for ``i = 1..A-1`` from per-anchor ToA ranges as above, so the
        differences share anchor 0's error (covariance ``sigma^2 (I + 1 1^T)``), as with a
        real reference anchor (Chan and Ho 1994).
aoa     ``theta = wrap(atan2(dy, dx) - orientation + n)``, the azimuth of the direction from
        the anchor to the device, relative to the anchor's boresight; NLOS links get extra
        Gaussian error ``nlos_std``.

``nlos`` is ``None`` (all LOS), an (N, A) bool mask (e.g. ``FloorPlan.nlos_mask``), or a
probability in [0, 1] drawn independently per link.

References
----------
S. Gezici, Z. Tian, G. B. Giannakis, H. Kobayashi, A. F. Molisch, H. V. Poor and
    Z. Sahinoglu, "Localization via ultra-wideband radios", IEEE Signal Processing Magazine
    22(4):70-84, 2005. DOI 10.1109/MSP.2005.1458289
N. Alsindi, B. Alavi and K. Pahlavan, "Measurement and modeling of ultrawideband TOA-based
    ranging in indoor multipath environments", IEEE Trans. Vehicular Technology
    58(3):1046-1058, 2009. DOI 10.1109/TVT.2008.926071
Y. T. Chan and K. C. Ho, "A simple and efficient estimator for hyperbolic location",
    IEEE Trans. Signal Processing 42(8):1905-1915, 1994. DOI 10.1109/78.301830
"""
from __future__ import annotations

import numpy as np

from .geometry import wrap_angle


def _geometry(pos, anchors) -> tuple[np.ndarray, np.ndarray]:
    pos = np.atleast_2d(np.asarray(pos, dtype=np.float64))
    anchors = np.atleast_2d(np.asarray(anchors, dtype=np.float64))
    if pos.shape[1] != anchors.shape[1]:
        raise ValueError(f"pos has D={pos.shape[1]} but anchors have D={anchors.shape[1]}")
    return pos, anchors


def _nlos_mask(nlos, shape, rng) -> np.ndarray:
    if nlos is None:
        return np.zeros(shape, dtype=bool)
    if np.ndim(nlos) == 0 and not isinstance(nlos, (bool, np.bool_)):
        p = float(nlos)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"an NLOS probability must be in [0, 1], got {p}")
        return rng.random(shape) < p
    return np.broadcast_to(np.asarray(nlos, dtype=bool), shape)


def distances(pos, anchors) -> np.ndarray:
    """(N, A) Euclidean distances between devices and anchors."""
    pos, anchors = _geometry(pos, anchors)
    return np.linalg.norm(pos[:, None, :] - anchors[None, :, :], axis=-1)


def bearings(pos, anchors, orientations=None) -> np.ndarray:
    """(N, A) azimuth (radians, ``[-pi, pi)``) of each device seen from each anchor,
    relative to the anchor's boresight ``orientations`` (A,) (0 = the +x axis)."""
    pos, anchors = _geometry(pos, anchors)
    diff = pos[:, None, :2] - anchors[None, :, :2]
    theta = np.arctan2(diff[..., 1], diff[..., 0])
    if orientations is not None:
        theta = theta - np.asarray(orientations, dtype=np.float64)[None, :]
    return wrap_angle(theta)


def simulate_ranges(pos, anchors, *, noise_std: float = 0.1, nlos=None, nlos_bias_mean: float = 0.5,
                    random_state=None) -> np.ndarray:
    """(N, A) noisy ToA / RTT / UWB ranges in metres (see the module docstring for the model)."""
    rng = np.random.default_rng(random_state)
    d = distances(pos, anchors)
    mask = _nlos_mask(nlos, d.shape, rng)
    noise = rng.normal(0.0, noise_std, d.shape) if noise_std > 0 else 0.0
    bias = np.where(mask, rng.exponential(nlos_bias_mean, d.shape), 0.0) if nlos_bias_mean > 0 else 0.0
    return np.maximum(d + noise + bias, 0.0)


def ranges_to_tdoa(ranges) -> np.ndarray:
    """(N, A - 1) range differences to anchor 0: ``r[:, 1:] - r[:, :1]``."""
    ranges = np.atleast_2d(np.asarray(ranges, dtype=np.float64))
    return ranges[:, 1:] - ranges[:, :1]


def simulate_tdoa(pos, anchors, *, noise_std: float = 0.1, nlos=None, nlos_bias_mean: float = 0.5,
                  random_state=None) -> np.ndarray:
    """(N, A - 1) TDoA range differences in metres, relative to anchor 0.

    ``noise_std`` is the per-anchor ToA error (in metres); the differences therefore have
    variance ``2 noise_std^2`` and a common term from anchor 0.
    """
    pos, anchors = _geometry(pos, anchors)
    if len(anchors) < 2:
        raise ValueError("TDoA needs at least two anchors")
    rng = np.random.default_rng(random_state)
    d = distances(pos, anchors)
    mask = _nlos_mask(nlos, d.shape, rng)
    noise = rng.normal(0.0, noise_std, d.shape) if noise_std > 0 else 0.0
    bias = np.where(mask, rng.exponential(nlos_bias_mean, d.shape), 0.0) if nlos_bias_mean > 0 else 0.0
    return ranges_to_tdoa(d + noise + bias)


def simulate_aoa(pos, anchors, orientations=None, *, noise_std: float = np.deg2rad(3.0), nlos=None,
                 nlos_std: float = np.deg2rad(10.0), random_state=None) -> np.ndarray:
    """(N, A) noisy azimuth angles of arrival in radians, ``[-pi, pi)``, relative to boresight."""
    rng = np.random.default_rng(random_state)
    theta = bearings(pos, anchors, orientations)
    mask = _nlos_mask(nlos, theta.shape, rng)
    noise = rng.normal(0.0, noise_std, theta.shape) if noise_std > 0 else 0.0
    extra = np.where(mask, rng.normal(0.0, nlos_std, theta.shape), 0.0) if nlos_std > 0 else 0.0
    return wrap_angle(theta + noise + extra)


__all__ = ["bearings", "distances", "ranges_to_tdoa", "simulate_aoa", "simulate_ranges", "simulate_tdoa"]
