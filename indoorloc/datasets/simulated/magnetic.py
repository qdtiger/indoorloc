"""A synthetic indoor magnetic field: the Earth's field plus magnetic dipoles in the building structure.

Indoor magnetic fingerprinting relies on the distortion of the geomagnetic field by steel
(reinforcement bars, beams, ducts): a smooth, static anomaly of a few to tens of microtesla
(Li et al. 2012). This module builds a **synthetic** stand-in with exact physics but made-up
sources: the field at a point is

    b(p) = b_earth + sum_k (mu0 / 4 pi) (3 (m_k . r_hat) r_hat - m_k) / |r|^3,   r = p - s_k

(the field of point dipoles ``m_k`` at ``s_k``; Jackson 1999), in microtesla, with
``mu0 / 4 pi = 1e-7 T m / A``. The dipoles sit in the floor slabs (``z = k * floor_height``,
slightly below the surface), half of them under walls (beams and wall reinforcement) and
half anywhere under the footprint, with uniformly random directions and log-uniform
moment magnitudes. Because every source lies in a slab, at least ``device_height`` below
and ``floor_height - device_height`` above a device, the field in the walking space is
smooth and exactly curl- and divergence-free there. It is not fitted to any measured
building: amplitudes and spatial scales are set by the parameters. In ``SyntheticOffice``
(seed 0) the defaults give a noise-free ``|b|`` with a 1.9 uT standard deviation over the
floor at 1.2 m, and a 5-95 % range of 47.1-53.1 uT.

Realism: the synthetic field is static and perfectly repeatable. A reading differs from the
field only by white sensor noise and an optional constant device bias, and the device is
always held flat. Measured data are much less repeatable. In the ILC 2020 sample (site1/F1),
``|b|`` recorded by two different traces less than 0.5 m apart differed by a median 3.5 uT,
about half the 6.7 uT between random places. Accuracy obtained on this field is therefore an
upper bound for magnetic fingerprinting, not an estimate of it.

Frame: x east, y north, z up (the simulator's local frame is taken as East-North-Up), so the
Earth's field with total intensity ``F``, inclination ``I`` (positive downward) and
declination ``D`` (positive east) is ``F (cos I sin D, cos I cos D, -sin I)``.

References
----------
J. D. Jackson, "Classical Electrodynamics", 3rd ed., Wiley, 1999, section 5.6 (field of a
    magnetic dipole). ISBN 978-0-471-30932-1
B. Li, T. Gallagher, A. G. Dempster, C. Rizos, "How feasible is the use of magnetic field alone
    for indoor positioning?", IPIN 2012. DOI 10.1109/IPIN.2012.6418880
"""
from __future__ import annotations

import numpy as np

MU0_OVER_4PI_UT = 0.1  # mu0 / 4 pi = 1e-7 T m/A = 0.1 uT m^3 / (A m^2)


def earth_field(total_ut: float = 50.0, inclination_deg: float = 60.0, declination_deg: float = 0.0) -> np.ndarray:
    """``(3,)`` geomagnetic field in uT in the East-North-Up frame (see the module docstring)."""
    inc, dec = np.deg2rad(float(inclination_deg)), np.deg2rad(float(declination_deg))
    return float(total_ut) * np.array([np.cos(inc) * np.sin(dec), np.cos(inc) * np.cos(dec), -np.sin(inc)])


def dipole_field(points, positions, moments, *, chunk: int = 1 << 20) -> np.ndarray:
    """``(N, 3)`` field in uT at ``points`` ``(N, 3)`` m of dipoles ``moments`` ``(K, 3)`` A m^2 at
    ``positions`` ``(K, 3)`` m (summed). A point on a dipole gives inf/NaN: keep them apart."""
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))
    s = np.atleast_2d(np.asarray(positions, dtype=np.float64)).reshape(-1, 3)
    m = np.atleast_2d(np.asarray(moments, dtype=np.float64)).reshape(-1, 3)
    out = np.zeros((len(p), 3))
    if len(s) == 0:
        return out
    step = max(1, chunk // len(s))
    for a in range(0, len(p), step):
        r = p[a:a + step, None, :] - s[None]                            # (n, K, 3)
        d2 = np.sum(r * r, axis=2)
        d = np.sqrt(d2)
        mr = np.einsum("nkd,kd->nk", r, m)
        field = (3.0 * mr[..., None] * r / d2[..., None] - m[None]) / (d2 * d)[..., None]
        out[a:a + step] = MU0_OVER_4PI_UT * field.sum(axis=1)
    return out


def place_dipoles(plan, *, density: float = 0.1, moment_range=(20.0, 200.0), wall_fraction: float = 0.5,
                  depth: float = 0.1, random_state=None) -> tuple[np.ndarray, np.ndarray]:
    """Dipoles in every floor slab of ``plan`` (``n_floors + 1`` slabs: ground to roof).

    ``density`` dipoles per m^2 of footprint per slab; a ``wall_fraction`` of them under the
    walls of the adjacent storey (uniform along the walls, by length), the rest uniform over
    the footprint; ``depth`` metres below the slab surface; directions uniform on the sphere,
    magnitudes log-uniform in ``moment_range`` (A m^2). Returns ``(positions (K, 3), moments (K, 3))``.
    """
    rng = np.random.default_rng(random_state)
    x0, y0, x1, y1 = plan.bounds
    lo, hi = (float(v) for v in moment_range)
    if not 0 < lo <= hi or not density >= 0 or not 0 <= wall_fraction <= 1:
        raise ValueError("need 0 < moment_range[0] <= moment_range[1], density >= 0, 0 <= wall_fraction <= 1")
    n = int(round(float(density) * (x1 - x0) * (y1 - y0)))
    pos, mom = [], []
    for k in range(plan.n_floors + 1):
        storey = min(k, plan.n_floors - 1)
        walls = plan.walls_on(storey)
        n_wall = int(round(wall_fraction * n)) if len(walls) else 0
        xy = np.column_stack([rng.uniform(x0, x1, n - n_wall), rng.uniform(y0, y1, n - n_wall)])
        if n_wall:
            length = np.hypot(walls[:, 2] - walls[:, 0], walls[:, 3] - walls[:, 1])
            w = rng.choice(len(walls), n_wall, p=length / length.sum())
            t = rng.uniform(0.0, 1.0, n_wall)[:, None]
            xy = np.vstack([walls[w, :2] + t * (walls[w, 2:] - walls[w, :2]), xy])
        z = np.full(len(xy), k * plan.floor_height - float(depth))
        direction = rng.normal(size=(len(xy), 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        magnitude = np.exp(rng.uniform(np.log(lo), np.log(hi), len(xy)))
        pos.append(np.column_stack([xy, z]))
        mom.append(direction * magnitude[:, None])
    return np.concatenate(pos), np.concatenate(mom)


def device_readings(field_world, heading, *, noise_std: float = 0.0, bias=None, random_state=None) -> np.ndarray:
    """``(N, 3)`` magnetometer readings of a device held flat (z up) facing ``heading`` (radians,
    counter-clockwise from +x): the world field rotated into the device frame (x forward,
    y left, the frame of ``building.synthesize_imu``), plus a device ``bias`` (hard-iron
    residual, ``(3,)`` or ``(N, 3)`` uT) and white noise ``noise_std`` uT per axis. The
    compass heading of these readings is ``signals.magnetic.heading(readings,
    forward=(1, 0, 0))`` (its default forward axis is the device +y of a portrait phone)."""
    rng = np.random.default_rng(random_state)
    b = np.atleast_2d(np.asarray(field_world, dtype=np.float64))
    h = np.broadcast_to(np.asarray(heading, dtype=np.float64), (len(b),))
    c, s = np.cos(h), np.sin(h)
    dev = np.column_stack([c * b[:, 0] + s * b[:, 1], -s * b[:, 0] + c * b[:, 1], b[:, 2]])
    if bias is not None:
        dev = dev + np.asarray(bias, dtype=np.float64)
    if noise_std > 0:
        dev = dev + rng.normal(0.0, float(noise_std), dev.shape)
    return dev


def features(readings) -> np.ndarray:
    """``(N, 3)`` ``[B, B_h, B_v]`` of flat-device readings (the ``magnetic`` layout, z up)."""
    r = np.atleast_2d(np.asarray(readings, dtype=np.float64))
    return np.column_stack([np.linalg.norm(r, axis=1), np.hypot(r[:, 0], r[:, 1]), r[:, 2]])


__all__ = ["MU0_OVER_4PI_UT", "device_readings", "dipole_field", "earth_field", "features", "place_dipoles"]
