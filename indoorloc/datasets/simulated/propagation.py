"""Large-scale radio propagation: path-loss models, shadowing and receiver sensitivity.

All losses are positive dB, distances metres, frequencies Hz, powers dBm. Nothing here
knows about floor plans: the multi-wall model takes wall/floor counts, which
``building.FloorPlan.wall_crossings`` computes from exact segment-wall intersections.

Models
------
``free_space_path_loss``      Friis (1946).
``log_distance_path_loss``    PL(d0) + 10 n log10(d / d0) (Rappaport 2002, section 4.9).
``multi_wall_path_loss``      COST 231 multi-wall model (Damosso 1999, section 4.7).
``inh_office_path_loss``      3GPP TR 38.901 InH-Office LOS / NLOS (Table 7.4.1-1),
``inh_office_los_probability``  with its LOS probability (Table 7.4.2-1) and
``INH_OFFICE_SHADOW_STD_DB``  shadow-fading standard deviations (Table 7.4.1-1).
``ShadowingMap``              spatially correlated log-normal shadowing with Gudmundson's
                              exponential autocorrelation, drawn exactly on a grid by
                              circulant embedding and read back by bilinear interpolation.
``apply_sensitivity``         readings weaker than the receiver sensitivity become NaN.

References
----------
H. T. Friis, "A note on a simple transmission formula", Proc. IRE 34(5):254-256, 1946.
    DOI 10.1109/JRPROC.1946.234568
T. S. Rappaport, "Wireless Communications: Principles and Practice", 2nd ed., Prentice Hall,
    2002, sections 4.9.1-4.9.2 (log-distance path loss, log-normal shadowing).
E. Damosso (ed.), "COST Action 231: Digital mobile radio towards future generation systems,
    Final report", European Commission EUR 18957, 1999, section 4.7 (multi-wall model).
J. M. Keenan and A. J. Motley, "Radio coverage in buildings", British Telecom Technology
    Journal 8(1):19-24, 1990 (the wall/floor-count model COST 231 extends).
3GPP TR 38.901 V17.0.0, "Study on channel model for frequencies from 0.5 to 100 GHz", 2022,
    Tables 7.4.1-1, 7.4.2-1 and 7.5-6. https://www.3gpp.org/ftp/Specs/archive/38_series/38.901/
M. Gudmundson, "Correlation model for shadow fading in mobile radio systems", Electronics
    Letters 27(23):2145-2146, 1991. DOI 10.1049/el:19911328
C. R. Dietrich and G. N. Newsam, "Fast and exact simulation of stationary Gaussian processes
    through circulant embedding of the covariance matrix", SIAM J. Sci. Comput.
    18(4):1088-1107, 1997. DOI 10.1137/S1064827592240555
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0  # m/s
BOLTZMANN = 1.380649e-23        # J/K

# COST 231 multi-wall model, parameters fitted at 1.8 GHz (Damosso 1999, section 4.7):
# light (plasterboard, light concrete) and heavy (load-bearing concrete, brick) walls, one floor, b.
COST231_WALL_LOSS_DB = {"light": 3.4, "heavy": 6.9}
COST231_FLOOR_LOSS_DB = 18.3
COST231_FLOOR_EXPONENT_B = 0.46

# 3GPP TR 38.901 Table 7.4.1-1, InH-Office shadow-fading standard deviations (dB).
INH_OFFICE_SHADOW_STD_DB = {"los": 3.0, "nlos": 8.03, "nlos_optional": 8.29}
# 3GPP TR 38.901 Table 7.5-6 (part 1), InH shadow-fading decorrelation distances (m).
INH_SHADOW_DECORRELATION_M = {"los": 10.0, "nlos": 6.0}


def _distance(d, minimum: float) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64)
    if np.any(d < 0):
        raise ValueError("distances must be non-negative")
    return np.maximum(d, minimum)


def wavelength(frequency_hz) -> np.ndarray:
    """Free-space wavelength in metres."""
    return SPEED_OF_LIGHT / np.asarray(frequency_hz, dtype=np.float64)


def free_space_path_loss(distance, frequency_hz, *, min_distance: float = 1e-3) -> np.ndarray:
    """Friis free-space path loss ``20 log10(4 pi d f / c)`` in dB (isotropic antennas).

    ``distance`` is clamped to ``min_distance`` (the far-field formula diverges at 0).
    Known value: 1 m at 2.4 GHz is 40.05 dB.
    """
    d = _distance(distance, min_distance)
    return 20.0 * np.log10(4.0 * np.pi * d * np.asarray(frequency_hz, dtype=np.float64) / SPEED_OF_LIGHT)


def log_distance_path_loss(distance, frequency_hz=2.4e9, *, exponent: float = 2.0, d0: float = 1.0,
                           pl0_db=None, shadowing_std_db: float = 0.0, random_state=None,
                           min_distance: float = 0.1) -> np.ndarray:
    """``PL(d) = PL(d0) + 10 n log10(d / d0) + X_sigma`` (Rappaport 2002, section 4.9).

    ``pl0_db`` defaults to the Friis loss at ``d0``; ``exponent`` n is 2 in free space and
    typically 1.6-1.8 (line of sight) to 4-6 (obstructed) inside buildings (Rappaport Table 4.2).
    ``X_sigma ~ N(0, shadowing_std_db^2)`` is drawn independently per entry from
    ``random_state`` (log-normal shadowing); use ``ShadowingMap`` for spatially correlated
    shadowing instead.
    """
    d = _distance(distance, min_distance)
    pl0 = free_space_path_loss(d0, frequency_hz) if pl0_db is None else np.asarray(pl0_db, dtype=np.float64)
    pl = pl0 + 10.0 * exponent * np.log10(d / d0)
    if shadowing_std_db > 0:
        pl = pl + np.random.default_rng(random_state).normal(0.0, shadowing_std_db, np.shape(pl))
    return pl


def multi_wall_path_loss(distance, frequency_hz, wall_counts, floor_counts=0, *,
                         wall_loss_db=(COST231_WALL_LOSS_DB["light"], COST231_WALL_LOSS_DB["heavy"]),
                         floor_loss_db: float = COST231_FLOOR_LOSS_DB, b: float = COST231_FLOOR_EXPONENT_B,
                         constant_db: float = 0.0, min_distance: float = 0.1) -> np.ndarray:
    """COST 231 multi-wall model.

    ``L = L_FS(d) + L_c + sum_i k_wi L_wi + k_f ** ((k_f + 2) / (k_f + 1) - b) * L_f``

    Parameters
    ----------
    distance : (...,) metres (3-D distance between the antennas).
    wall_counts : (..., I) number of penetrated walls of each type ``i`` (columns in the
        order of ``wall_loss_db``), or (...,) when there is a single wall type; exact counts
        come from ``building.FloorPlan.wall_crossings`` (per-wall losses: pass
        ``plan.crossed_walls(a, b) @ loss_db`` as ``wall_counts`` with ``wall_loss_db=1.0``).
    floor_counts : (...,) number of penetrated floors ``k_f``.
    wall_loss_db, floor_loss_db, b : per-type wall loss ``L_wi``, floor loss ``L_f`` and the
        empirical floor exponent; defaults are the COST 231 values at 1.8 GHz (light 3.4 dB,
        heavy 6.9 dB, floor 18.3 dB, b = 0.46).
    constant_db : ``L_c``, a fitted constant (0 with the default parameters).
    """
    losses = np.atleast_1d(np.asarray(wall_loss_db, dtype=np.float64))
    counts = np.asarray(wall_counts, dtype=np.float64)
    if counts.ndim == 0 or counts.shape[-1] != len(losses):
        if len(losses) != 1:
            raise ValueError(f"wall_counts must end in an axis of {len(losses)} wall types, got shape {counts.shape}")
        counts = counts[..., None]
    walls = counts @ losses
    kf = np.asarray(floor_counts, dtype=np.float64)
    safe = np.maximum(kf, 1.0)
    floors = np.where(kf > 0, safe ** ((safe + 2.0) / (safe + 1.0) - b) * floor_loss_db, 0.0)
    return free_space_path_loss(_distance(distance, min_distance), frequency_hz) + constant_db + walls + floors


def inh_office_path_loss(distance_3d, frequency_hz, los=True, *, optional_nlos: bool = False,
                         min_distance: float = 1.0) -> np.ndarray:
    """3GPP TR 38.901 InH-Office path loss (Table 7.4.1-1), fc in Hz, d3D in metres.

    ``PL_LOS  = 32.4 + 17.3 log10(d3D) + 20 log10(fc/GHz)``                     (sigma_SF 3 dB)
    ``PL'_NLOS = 17.30 + 38.3 log10(d3D) + 24.9 log10(fc/GHz)``,
    ``PL_NLOS = max(PL_LOS, PL'_NLOS)``                                           (sigma_SF 8.03 dB)
    optional: ``PL_NLOS = 32.4 + 20 log10(fc/GHz) + 31.9 log10(d3D)``             (sigma_SF 8.29 dB)

    Valid for 1 m <= d3D <= 150 m and 0.5-100 GHz; distances below ``min_distance`` are
    clamped to it. ``los`` is a bool or a bool array broadcastable with ``distance_3d``.
    """
    d = _distance(distance_3d, min_distance)
    fc = np.asarray(frequency_hz, dtype=np.float64) / 1e9
    pl_los = 32.4 + 17.3 * np.log10(d) + 20.0 * np.log10(fc)
    if optional_nlos:
        pl_nlos = 32.4 + 20.0 * np.log10(fc) + 31.9 * np.log10(d)
    else:
        pl_nlos = np.maximum(pl_los, 17.30 + 38.3 * np.log10(d) + 24.9 * np.log10(fc))
    return np.where(np.asarray(los, dtype=bool), pl_los, pl_nlos)


def inh_office_los_probability(distance_2d, *, variant: str = "mixed") -> np.ndarray:
    """3GPP TR 38.901 Table 7.4.2-1 line-of-sight probability for InH-Office.

    mixed office: 1 for d2D <= 1.2 m, exp(-(d2D - 1.2) / 4.7) up to 6.5 m, then
    0.32 exp(-(d2D - 6.5) / 32.6).  open office: 1 for d2D <= 5 m,
    exp(-(d2D - 5) / 70.8) up to 49 m, then 0.54 exp(-(d2D - 49) / 211.7).
    """
    d = np.asarray(distance_2d, dtype=np.float64)
    if variant == "mixed":
        return np.where(d <= 1.2, 1.0, np.where(d < 6.5, np.exp(-(d - 1.2) / 4.7),
                                                0.32 * np.exp(-(d - 6.5) / 32.6)))
    if variant == "open":
        return np.where(d <= 5.0, 1.0, np.where(d <= 49.0, np.exp(-(d - 5.0) / 70.8),
                                                0.54 * np.exp(-(d - 49.0) / 211.7)))
    raise ValueError(f"variant must be 'mixed' or 'open', not {variant!r}")


def thermal_noise_dbm(bandwidth_hz, *, noise_figure_db: float = 7.0, temperature_k: float = 290.0) -> np.ndarray:
    """Receiver noise power ``10 log10(k T B / 1 mW) + NF`` (-174 dBm/Hz at 290 K)."""
    return 10.0 * np.log10(BOLTZMANN * temperature_k * np.asarray(bandwidth_hz, dtype=np.float64) / 1e-3) \
        + noise_figure_db


def apply_sensitivity(power_dbm, sensitivity_dbm: float) -> np.ndarray:
    """Readings below the receiver sensitivity are not reported: they become NaN (never a sentinel)."""
    power = np.asarray(power_dbm)
    power = power.astype(power.dtype if power.dtype.kind == "f" else np.float64, copy=True)
    power[~(power >= sensitivity_dbm)] = np.nan
    return power


def exponential_correlation(distance, decorrelation_distance: float) -> np.ndarray:
    """Gudmundson's shadowing autocorrelation ``exp(-d / d_corr)``.

    Gudmundson (1991) writes it as ``eps_D ** (d / D)`` (correlation ``eps_D`` at distance
    ``D``); the two are equal for ``d_corr = -D / ln(eps_D)``. 3GPP TR 38.901 uses the same
    form with ``d_corr`` from Table 7.5-6.
    """
    return np.exp(-np.abs(np.asarray(distance, dtype=np.float64)) / decorrelation_distance)


def correlated_gaussian_field(shape, spacing: float, decorrelation_distance: float, *, n_fields: int = 1,
                              random_state=None) -> np.ndarray:
    """Zero-mean, unit-variance Gaussian fields on a regular grid with covariance
    ``exp(-|x - x'| / d_corr)`` (isotropic, Euclidean distance), drawn by circulant embedding.

    Returns an ``(n_fields, ny, nx)`` float64 array. The grid is embedded in a periodic grid
    at least twice as large plus ``4 d_corr`` of padding; the covariance on that torus is
    diagonalised by the 2-D FFT and each (real, imaginary) pair of one complex draw gives two
    independent fields (Dietrich and Newsam 1997). With this padding the embedding of the
    exponential covariance was non-negative definite for every grid we checked (the draw is
    then exact); should a negative eigenvalue appear it is clipped to zero.
    """
    ny, nx = (int(v) for v in shape)
    if ny < 1 or nx < 1:
        raise ValueError(f"shape must be positive, got {shape}")
    if spacing <= 0 or decorrelation_distance <= 0:
        raise ValueError("spacing and decorrelation_distance must be positive")
    rng = np.random.default_rng(random_state)
    pad = int(np.ceil(4.0 * decorrelation_distance / spacing))
    my, mx = 2 * (ny + pad), 2 * (nx + pad)
    iy = np.minimum(np.arange(my), my - np.arange(my)) * spacing
    ix = np.minimum(np.arange(mx), mx - np.arange(mx)) * spacing
    cov = exponential_correlation(np.hypot(iy[:, None], ix[None, :]), decorrelation_distance)
    eig = np.fft.fft2(cov).real
    eig = np.maximum(eig, 0.0)
    scale = np.sqrt(eig / (my * mx))
    fields = []
    for _ in range((n_fields + 1) // 2):
        z = rng.standard_normal((my, mx)) + 1j * rng.standard_normal((my, mx))
        f = np.fft.fft2(scale * z)
        fields.extend([f.real[:ny, :nx], f.imag[:ny, :nx]])
    return np.stack(fields[:n_fields])


@dataclass(frozen=True, eq=False)
class ShadowingMap:
    """Spatially correlated log-normal shadowing, in dB, as a function of position.

    One map is one realisation of the shadowing seen from one transmitter over a rectangle:
    a unit-variance Gaussian field with Gudmundson's exponential autocorrelation on a
    regular grid, multiplied by ``std_db`` and read at arbitrary points by bilinear
    interpolation (interpolation slightly lowers the variance between grid nodes; keep
    ``spacing`` well below ``decorrelation_distance``). Deterministic for a given seed.

    ``field`` (ny, nx) unit-variance values; ``origin`` (x0, y0) of node [0, 0]; ``spacing`` m.
    """

    field: np.ndarray
    origin: tuple[float, float]
    spacing: float
    std_db: float = 1.0
    decorrelation_distance: float = 1.0

    @classmethod
    def generate(cls, bounds, *, std_db: float, decorrelation_distance: float, spacing: float | None = None,
                 n_maps: int = 1, random_state=None) -> list[ShadowingMap]:
        """``n_maps`` independent maps covering ``bounds = (xmin, ymin, xmax, ymax)``."""
        xmin, ymin, xmax, ymax = (float(v) for v in bounds)
        spacing = float(spacing or min(0.5, decorrelation_distance / 4.0))
        nx = int(np.ceil((xmax - xmin) / spacing)) + 2
        ny = int(np.ceil((ymax - ymin) / spacing)) + 2
        fields = correlated_gaussian_field((ny, nx), spacing, decorrelation_distance, n_fields=n_maps,
                                           random_state=random_state)
        return [cls(f, (xmin, ymin), spacing, float(std_db), float(decorrelation_distance)) for f in fields]

    def __call__(self, points) -> np.ndarray:
        """Shadowing (dB) at ``points`` (M, 2+); points outside the grid use the nearest edge."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        ny, nx = self.field.shape
        gx = np.clip((pts[:, 0] - self.origin[0]) / self.spacing, 0.0, nx - 1.0)
        gy = np.clip((pts[:, 1] - self.origin[1]) / self.spacing, 0.0, ny - 1.0)
        x0 = np.minimum(np.floor(gx).astype(np.int64), nx - 2) if nx > 1 else np.zeros(len(gx), np.int64)
        y0 = np.minimum(np.floor(gy).astype(np.int64), ny - 2) if ny > 1 else np.zeros(len(gy), np.int64)
        fx, fy = gx - x0, gy - y0
        x1, y1 = np.minimum(x0 + 1, nx - 1), np.minimum(y0 + 1, ny - 1)
        f = self.field
        value = (f[y0, x0] * (1 - fx) * (1 - fy) + f[y0, x1] * fx * (1 - fy)
                 + f[y1, x0] * (1 - fx) * fy + f[y1, x1] * fx * fy)
        return self.std_db * value


__all__ = ["BOLTZMANN", "COST231_FLOOR_EXPONENT_B", "COST231_FLOOR_LOSS_DB", "COST231_WALL_LOSS_DB",
           "INH_OFFICE_SHADOW_STD_DB", "INH_SHADOW_DECORRELATION_M", "SPEED_OF_LIGHT", "ShadowingMap",
           "apply_sensitivity", "correlated_gaussian_field", "exponential_correlation", "free_space_path_loss",
           "inh_office_los_probability", "inh_office_path_loss", "log_distance_path_loss", "multi_wall_path_loss",
           "thermal_noise_dbm", "wavelength"]
