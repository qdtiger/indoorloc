"""Visible light: ceiling LED layouts, the Lambertian line-of-sight gain and photodiode noise.

The simulator's copy of the optical channel (L1 does not import L2; ``signals.vlc`` holds the
same formulas for users, and ``tests/datasets`` checks that both agree). Line-of-sight DC gain
of an LED of Lambertian order ``m`` at a photodiode of area ``A`` (Kahn & Barry 1997;
Komine & Nakagawa 2004)::

    H = (m + 1) A / (2 pi d^2) cos^m(phi) Ts g cos(psi)   for psi <= FOV, else 0

with ``g = n^2 / sin^2(FOV)`` for an ideal concentrator of refractive index ``n``. Walls and
floors are opaque: a link whose straight path crosses a wall or a floor slab has no gain
(light passes through doorways). Reflections are not modelled.

Noise (Komine & Nakagawa 2004): the photocurrent ``R P`` carries shot noise
``2 q R P B + 2 q I_bg I2 B`` and thermal noise
``8 pi k T eta A I2 B^2 / G + 16 pi^2 k T Gamma eta^2 A^2 I3 B^3 / g_m`` (A^2); a reading of
optical power has standard deviation ``sqrt(variance) / R`` watts.

References
----------
J. M. Kahn, J. R. Barry, "Wireless infrared communications", Proceedings of the IEEE
    85(2):265-298, 1997. DOI 10.1109/5.554222
T. Komine, M. Nakagawa, "Fundamental analysis for visible-light communication system using LED
    lights", IEEE Transactions on Consumer Electronics 50(1):100-107, 2004. DOI 10.1109/TCE.2004.1277847
"""
from __future__ import annotations

import numpy as np

ELEMENTARY_CHARGE = 1.602176634e-19  # C
BOLTZMANN = 1.380649e-23             # J/K


def lambertian_order(half_power_angle_rad: float) -> float:
    """``m = -ln 2 / ln cos(Phi_1/2)``."""
    phi = float(half_power_angle_rad)
    if not 0 < phi < np.pi / 2:
        raise ValueError(f"the half-power angle must be in (0, pi/2) radians, got {phi}")
    return float(-np.log(2.0) / np.log(np.cos(phi)))


def room_grid_leds(plan, spacing: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """LEDs on a regular grid in every room of every storey (the whole footprint if the plan
    has no rooms): ``max(1, round(width / spacing))`` by ``max(1, round(depth / spacing))``
    cells per room, one LED at each cell centre, ``height`` m above the storey floor.
    Returns ``(xyz (A, 3), floor (A,))``, storey-major, then room order, then row-major."""
    if not float(spacing) > 0:
        raise ValueError(f"led spacing must be positive, got {spacing}")
    rooms = plan.rooms if len(plan.rooms) else np.asarray(plan.bounds, dtype=np.float64)[None]
    room_floor = plan.room_floor if len(plan.rooms) else None
    xyz, fl = [], []
    for f in range(plan.n_floors):
        for r, (x0, y0, x1, y1) in enumerate(rooms):
            if room_floor is not None and room_floor[r] != f:
                continue
            nx = max(1, int(round((x1 - x0) / spacing)))
            ny = max(1, int(round((y1 - y0) / spacing)))
            gx, gy = np.meshgrid(x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx,
                                 y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny)
            n = gx.size
            xyz.append(np.column_stack([gx.ravel(), gy.ravel(), np.full(n, plan.height_of(f, height))]))
            fl.append(np.full(n, f, dtype=np.int64))
    return np.concatenate(xyz), np.concatenate(fl)


def lambertian_gain(positions, leds, *, order: float = 1.0, area: float = 1e-4, fov: float = np.pi / 2,
                    led_normal=(0.0, 0.0, -1.0), receiver_normal=(0.0, 0.0, 1.0), filter_gain: float = 1.0,
                    concentrator_gain: float = 1.0) -> np.ndarray:
    """``(N, A)`` line-of-sight gain (no occlusion test) for receivers ``(N, 3)`` and LEDs ``(A, 3)``."""
    p = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    a = np.atleast_2d(np.asarray(leds, dtype=np.float64))
    nt = np.asarray(led_normal, dtype=np.float64) / np.linalg.norm(led_normal)
    nr = np.asarray(receiver_normal, dtype=np.float64) / np.linalg.norm(receiver_normal)
    v = p[:, None, :] - a[None]
    d = np.sqrt(np.sum(v * v, axis=2))
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_phi = (v @ nt) / d
        cos_psi = -(v @ nr) / d
    seen = (d > 0) & (cos_phi > 0) & (cos_psi > 0) & (cos_psi >= np.cos(float(fov)) - 1e-15)
    cp, cs, dd = np.where(seen, cos_phi, 1.0), np.where(seen, cos_psi, 0.0), np.where(seen, d, 1.0)
    m = float(order)
    H = (m + 1.0) * float(area) / (2.0 * np.pi * dd * dd) * cp ** m * float(filter_gain) * float(concentrator_gain) * cs
    return np.where(seen, H, 0.0)


def noise_variance(power_w, *, responsivity: float, bandwidth: float, background_current: float, area: float,
                   temperature: float, open_loop_gain: float, capacitance_per_area: float,
                   channel_noise_factor: float, transconductance: float, i2: float, i3: float) -> np.ndarray:
    """Shot plus thermal noise variance (A^2) of the photocurrent at received power ``power_w``."""
    P = np.maximum(np.asarray(power_w, dtype=np.float64), 0.0)
    q, k, B, eta = ELEMENTARY_CHARGE, BOLTZMANN, float(bandwidth), float(capacitance_per_area)
    shot = 2.0 * q * responsivity * P * B + 2.0 * q * background_current * i2 * B
    thermal = (8.0 * np.pi * k * temperature * eta * area * i2 * B ** 2 / open_loop_gain
               + 16.0 * np.pi ** 2 * k * temperature * channel_noise_factor * eta ** 2 * area ** 2 * i3 * B ** 3
               / transconductance)
    return shot + thermal


__all__ = ["lambertian_gain", "lambertian_order", "noise_variance", "room_grid_leds"]
