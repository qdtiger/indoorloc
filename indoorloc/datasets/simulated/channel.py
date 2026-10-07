"""Narrowband multipath CSI for a uniform linear array (ULA) over OFDM subcarriers.

The channel between a single-antenna device and an M-element ULA at an access point is a
sum of P plane waves (line of sight, first-order specular wall reflections found with the
image method, and optional point scatterers):

``H[m, k] = sum_p  g_p * a_m(theta_p) * exp(-j 2 pi df_k tau_p)``,
``a_m(theta) = exp(+j 2 pi (d / lambda) (m - (M - 1) / 2) cos(el) sin(theta))``,

where ``g_p`` is the complex path gain at the carrier (it already contains the carrier
phase ``exp(-j 2 pi f_c tau_p)``), ``tau_p`` the propagation delay, ``df_k`` the baseband
offset of subcarrier ``k`` from the carrier, ``theta_p`` the azimuth of arrival measured from
the array boresight (positive towards the array axis ``m`` increases along), ``el`` the
elevation and ``d`` the element spacing. The array phase centre is the AP position. Each
path is a plane wave at the array (far field) and the steering vector uses the carrier
wavelength (narrowband array assumption); this is the model under SpotFi's joint
AoA/ToF estimation and under MUSIC (Schmidt 1986): because the paths have different delays,
the subcarriers act as snapshots with different inter-path phases, which decorrelates the
coherent multipath and lets subspace methods resolve it.

Path gains follow free-space spreading along the unfolded path length ``L``:
``|g| = lambda / (4 pi L) * Gamma * 10 ** (-penetration_dB / 20)`` with a reflection
coefficient ``Gamma`` per wall (1 for the direct path), wall penetration losses of every
other wall a leg crosses, and ``scatter_coef`` instead of ``Gamma`` for point scatterers
(a simplification of the bistatic radar equation). Heights enter through the 3-D path
length and the elevation; walls are vertical, so reflections are found in the plan view.

References
----------
R. O. Schmidt, "Multiple emitter location and signal parameter estimation", IEEE Trans.
    Antennas and Propagation 34(3):276-280, 1986. DOI 10.1109/TAP.1986.1143830
M. Kotaru, K. Joshi, D. Bharadia and S. Katti, "SpotFi: Decimeter level localization using
    WiFi", ACM SIGCOMM 2015. DOI 10.1145/2785956.2787487
J. B. Allen and D. A. Berkley, "Image method for efficiently simulating small-room
    acoustics", J. Acoust. Soc. Am. 65(4):943-950, 1979. DOI 10.1121/1.382599
Z. Yun and M. F. Iskander, "Ray tracing for radio propagation modeling: principles and
    applications", IEEE Access 3:1089-1100, 2015. DOI 10.1109/ACCESS.2015.2453991
IEEE Std 802.11-2020, clause 19 (HT PHY): 20 MHz channels use subcarriers -28..-1, 1..28
    spaced 312.5 kHz.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import count_crossings, mirror_points, paired_intersection_params, side_of
from .propagation import SPEED_OF_LIGHT

HT20_SUBCARRIERS = np.r_[-28:0, 1:29]  # IEEE 802.11n HT20: 52 data + 4 pilot subcarriers
OFDM_SPACING_HZ = 312.5e3


def subcarrier_offsets(indices=HT20_SUBCARRIERS, spacing_hz: float = OFDM_SPACING_HZ) -> np.ndarray:
    """Baseband frequency offsets (Hz) of OFDM subcarriers from the carrier."""
    return np.asarray(indices, dtype=np.float64) * float(spacing_hz)


def ula_steering(theta, n_antennas: int, spacing_wavelengths: float = 0.5, elevation=None) -> np.ndarray:
    """``(..., M)`` ULA response to plane waves arriving from ``theta`` (radians from boresight)."""
    theta = np.asarray(theta, dtype=np.float64)
    sin = np.sin(theta) if elevation is None else np.sin(theta) * np.cos(np.asarray(elevation, dtype=np.float64))
    m = np.arange(n_antennas, dtype=np.float64) - (n_antennas - 1) / 2.0
    return np.exp(2j * np.pi * spacing_wavelengths * sin[..., None] * m)


def ofdm_csi(gain, delay, theta, offsets_hz, *, n_antennas: int, spacing_wavelengths: float = 0.5,
             elevation=None) -> np.ndarray:
    """``(N, M, K)`` complex128 channel frequency response of a ULA (see the module docstring).

    ``gain`` (N, P) complex, ``delay`` (N, P) seconds, ``theta`` (N, P) radians from boresight,
    ``elevation`` (N, P) radians or None; entries that are NaN in any of them are absent paths
    (padding, as in DeepMIMO). ``offsets_hz`` (K,) are subcarrier offsets from the carrier.
    """
    gain = np.atleast_2d(np.asarray(gain, dtype=np.complex128))
    delay = np.atleast_2d(np.asarray(delay, dtype=np.float64))
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    el = None if elevation is None else np.atleast_2d(np.asarray(elevation, dtype=np.float64))
    valid = ~(np.isnan(gain) | np.isnan(delay) | np.isnan(theta))
    if el is not None:
        valid &= ~np.isnan(el)
        el = np.where(valid, el, 0.0)
    g = np.where(valid, gain, 0.0)
    tau = np.where(valid, delay, 0.0)
    theta = np.where(valid, theta, 0.0)
    offsets = np.asarray(offsets_hz, dtype=np.float64)
    n, p = g.shape
    out = np.empty((n, n_antennas, len(offsets)), dtype=np.complex128)
    step = max(1, (1 << 21) // max(1, p * (n_antennas + len(offsets))))      # bounded temporaries
    for lo in range(0, n, step):
        rows = slice(lo, lo + step)
        a = ula_steering(theta[rows], n_antennas, spacing_wavelengths, None if el is None else el[rows])  # (n, P, M)
        f = np.exp(-2j * np.pi * tau[rows, :, None] * offsets)                                        # (n, P, K)
        out[rows] = np.einsum("np,npm,npk->nmk", g[rows], a, f)
    return out


@dataclass(frozen=True, eq=False)
class Paths:
    """Propagation paths between N device/AP pairs, padded to P paths with NaN.

    gain (N, P) complex128 at the carrier (includes the carrier phase), delay (N, P) s,
    azimuth / elevation (N, P) radians of arrival at the AP in the global frame (azimuth from
    +x, counter-clockwise), departure_azimuth (N, P) at the device, kind (P,) one of
    ``"los"``, ``"wall"``, ``"scatterer"`` and index (P,) the wall or scatterer number.
    """

    gain: np.ndarray
    delay: np.ndarray
    azimuth: np.ndarray
    elevation: np.ndarray
    departure_azimuth: np.ndarray
    kind: np.ndarray
    index: np.ndarray

    @property
    def power_db(self) -> np.ndarray:
        """(N, P) path power ``20 log10 |g|`` in dB relative to the transmit power (NaN if absent)."""
        with np.errstate(divide="ignore"):
            return 20.0 * np.log10(np.abs(self.gain))


def multipath(device, ap, walls=None, *, frequency_hz: float, wall_loss_db=0.0, reflection_coef=0.5,
              scatterers=None, scatter_coef: float = 0.3, extra_loss_db=0.0) -> Paths:
    """Line-of-sight, first-order wall reflections and single-bounce scatterer paths.

    Parameters
    ----------
    device, ap : (N, 2) or (N, 3) positions (heights optional; missing heights are 0).
    walls : (W, 4) wall segments in plan view (all assumed to reach floor and ceiling).
    wall_loss_db, reflection_coef : scalars or (W,) per-wall penetration loss (dB) and
        reflection-coefficient magnitude.
    scatterers : (S, 2) or (S, 3) point scatterers, or None.
    extra_loss_db : (N,) or scalar loss applied to every path of a link (e.g. floors crossed).
    """
    dev = np.atleast_2d(np.asarray(device, dtype=np.float64))
    ap = np.atleast_2d(np.asarray(ap, dtype=np.float64))
    dev, ap = np.broadcast_arrays(dev, ap)
    n = len(dev)
    z = lambda v: v[:, 2] if v.shape[1] > 2 else np.zeros(len(v))  # noqa: E731
    dz = z(dev) - z(ap)                                                  # device above AP > 0
    walls = np.zeros((0, 4)) if walls is None else np.asarray(walls, dtype=np.float64).reshape(-1, 4)
    w = len(walls)
    loss = np.broadcast_to(np.asarray(wall_loss_db, dtype=np.float64), (w,))
    refl = np.broadcast_to(np.asarray(reflection_coef, dtype=np.float64), (w,))
    lam = SPEED_OF_LIGHT / frequency_hz
    extra = np.broadcast_to(np.asarray(extra_loss_db, dtype=np.float64), (n,))

    def gain(length, amp_db, coef):
        amp = coef * lam / (4.0 * np.pi * length) * 10.0 ** (-(amp_db + extra[:, None]) / 20.0)
        return amp * np.exp(-2j * np.pi * length / lam)

    def arrive(points):  # azimuth/elevation at the AP of waves coming from points (N, P, 2)
        vec = points - ap[:, None, :2]
        hor = np.hypot(vec[..., 0], vec[..., 1])
        return np.arctan2(vec[..., 1], vec[..., 0]), hor

    blocks = []
    # line of sight
    los_len2 = np.hypot(*(dev[:, :2] - ap[:, :2]).T)
    los_len = np.hypot(los_len2, dz)[:, None]
    az, _ = arrive(dev[:, None, :2])
    dep = np.arctan2(ap[:, 1] - dev[:, 1], ap[:, 0] - dev[:, 0])[:, None]
    pen = count_crossings(dev, ap, walls, loss)[:, None]
    blocks.append((gain(los_len, pen, 1.0), los_len, az, np.arctan2(dz, los_len2)[:, None], dep,
                   ["los"], [0]))
    # first-order specular reflections (image method)
    if w:
        image = mirror_points(dev[:, :2], walls)                                  # (N, W, 2)
        length2 = np.linalg.norm(image - ap[:, None, :2], axis=-1)
        length = np.hypot(length2, dz[:, None])
        same_side = side_of(dev[:, :2], walls) * side_of(ap[:, :2], walls) > 0
        idx = np.tile(np.arange(w), n)
        flat_ap = np.repeat(ap[:, :2], w, axis=0)
        t, _ = paired_intersection_params(image.reshape(-1, 2), flat_ap, walls[idx])
        valid = same_side & ~np.isnan(t).reshape(n, w)
        hit = image + np.nan_to_num(t).reshape(n, w)[..., None] * (ap[:, None, :2] - image)  # bounce points
        flat_hit = hit.reshape(-1, 2)
        pen = (count_crossings(np.repeat(dev[:, :2], w, axis=0), flat_hit, walls, loss, exclude=idx)
               + count_crossings(flat_hit, flat_ap, walls, loss, exclude=idx)).reshape(n, w)
        g = np.where(valid, gain(np.where(valid, length, 1.0), pen, refl[None, :]), np.nan)
        az, _ = arrive(hit)
        dep = np.arctan2(hit[..., 1] - dev[:, None, 1], hit[..., 0] - dev[:, None, 0])
        el = np.arctan2(dz[:, None], length2)
        nan = lambda a: np.where(valid, a, np.nan)  # noqa: E731
        blocks.append((g, nan(length), nan(az), nan(el), nan(dep), ["wall"] * w, list(range(w))))
    # single-bounce point scatterers
    if scatterers is not None and len(scatterers):
        sc = np.atleast_2d(np.asarray(scatterers, dtype=np.float64))
        s = len(sc)
        sz = sc[:, 2] if sc.shape[1] > 2 else np.zeros(s)
        d1 = np.hypot(np.linalg.norm(sc[None, :, :2] - dev[:, None, :2], axis=-1), sz[None] - z(dev)[:, None])
        d2_h = np.linalg.norm(sc[None, :, :2] - ap[:, None, :2], axis=-1)
        d2 = np.hypot(d2_h, sz[None] - z(ap)[:, None])
        length = d1 + d2
        flat_sc = np.tile(sc[:, :2], (n, 1))
        pen = (count_crossings(np.repeat(dev[:, :2], s, axis=0), flat_sc, walls, loss)
               + count_crossings(flat_sc, np.repeat(ap[:, :2], s, axis=0), walls, loss)).reshape(n, s)
        az, _ = arrive(np.broadcast_to(sc[None, :, :2], (n, s, 2)))
        dep = np.arctan2(sc[None, :, 1] - dev[:, None, 1], sc[None, :, 0] - dev[:, None, 0])
        el = np.arctan2(sz[None] - z(ap)[:, None], d2_h)
        blocks.append((gain(length, pen, scatter_coef), length, az, el, dep, ["scatterer"] * s, list(range(s))))

    cat = lambda i: np.concatenate([b[i] for b in blocks], axis=1)  # noqa: E731
    return Paths(gain=cat(0), delay=cat(1) / SPEED_OF_LIGHT, azimuth=cat(2), elevation=cat(3),
                 departure_azimuth=cat(4), kind=np.array([k for b in blocks for k in b[5]]),
                 index=np.array([i for b in blocks for i in b[6]], dtype=np.int64))


def paths_to_csi(paths: Paths, offsets_hz, *, n_antennas: int, orientation=0.0,
                 spacing_wavelengths: float = 0.5) -> np.ndarray:
    """(N, M, K) CSI of ``paths`` at a ULA whose boresight points along ``orientation`` (N,) or scalar."""
    theta = paths.azimuth - np.asarray(orientation, dtype=np.float64).reshape(-1, 1)
    return ofdm_csi(paths.gain, paths.delay, theta, offsets_hz, n_antennas=n_antennas,
                    spacing_wavelengths=spacing_wavelengths, elevation=paths.elevation)


__all__ = ["HT20_SUBCARRIERS", "OFDM_SPACING_HZ", "Paths", "multipath", "ofdm_csi", "paths_to_csi",
           "subcarrier_offsets", "ula_steering"]
