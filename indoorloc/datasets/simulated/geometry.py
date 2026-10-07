"""Planar geometry for floor plans: segment intersection, distances and mirror images.

Walls are 2-D segments stored as a ``(W, 4)`` float64 array of ``[x1, y1, x2, y2]`` rows
(vertical walls that span a whole storey, seen from above). Every function is vectorized
over links and walls and works in chunks, so a few hundred thousand links against a few
hundred walls stay within a few tens of megabytes.

The intersection test is the standard parametric one (two segments ``p + t r`` and
``a + u s`` meet when ``0 <= t, u <= 1``), see e.g. Schneider and Eberly, *Geometric Tools
for Computer Graphics*, Morgan Kaufmann, 2003, section 7.1. Collinear (grazing) contact is
not a crossing: a link running along a wall does not penetrate it.
"""
from __future__ import annotations

import numpy as np

_CHUNK = 1 << 20  # links x walls evaluated at once (~8 MB per float64 temporary)


def _as_walls(walls) -> np.ndarray:
    walls = np.asarray(walls, dtype=np.float64).reshape(-1, 4)
    return walls


def _pairs(p, q) -> tuple[np.ndarray, np.ndarray]:
    p = np.atleast_2d(np.asarray(p, dtype=np.float64))[..., :2]
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))[..., :2]
    p, q = np.broadcast_arrays(p, q)
    return p, q


def _chunks(n_rows: int, n_walls: int):
    step = max(1, _CHUNK // max(n_walls, 1))
    for start in range(0, n_rows, step):
        yield slice(start, min(start + step, n_rows))


def _hits(p, q, walls, eps: float = 1e-12):
    """Yield ``(rows, t, u)`` per chunk of links; ``t``/``u`` are NaN where there is no crossing."""
    a, s = walls[:, :2], walls[:, 2:] - walls[:, :2]
    for rows in _chunks(len(p), len(walls)):
        r = (q[rows] - p[rows])[:, None, :]                   # (m, 1, 2)
        ap = a[None, :, :] - p[rows][:, None, :]               # (m, W, 2)
        denom = r[..., 0] * s[None, :, 1] - r[..., 1] * s[None, :, 0]
        ok = np.abs(denom) > eps
        safe = np.where(ok, denom, 1.0)
        t = (ap[..., 0] * s[None, :, 1] - ap[..., 1] * s[None, :, 0]) / safe
        u = (ap[..., 0] * r[..., 1] - ap[..., 1] * r[..., 0]) / safe
        hit = ok & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
        yield rows, np.where(hit, t, np.nan), np.where(hit, u, np.nan)


def intersection_params(p, q, walls) -> tuple[np.ndarray, np.ndarray]:
    """Where the links ``p -> q`` meet the walls.

    Parameters
    ----------
    p, q : (M, 2) arrays (or broadcastable), link end points; extra columns (z) are ignored.
    walls : (W, 4) array of ``[x1, y1, x2, y2]``.

    Returns
    -------
    t, u : (M, W) float64 arrays. ``t`` is the position of the crossing along the link
        (0 at ``p``, 1 at ``q``) and ``u`` along the wall; both are NaN where the link does
        not cross the wall (parallel, collinear or outside either segment).
    """
    p, q = _pairs(p, q)
    walls = _as_walls(walls)
    t_out = np.full((len(p), len(walls)), np.nan)
    u_out = np.full((len(p), len(walls)), np.nan)
    if len(walls):
        for rows, t, u in _hits(p, q, walls):
            t_out[rows], u_out[rows] = t, u
    return t_out, u_out


def paired_intersection_params(p, q, walls, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Like ``intersection_params`` for link ``m`` against wall ``m`` only: ``t, u`` of shape (M,)."""
    p, q = _pairs(p, q)
    walls = np.broadcast_to(_as_walls(walls), (len(p), 4))
    r, s, ap = q - p, walls[:, 2:] - walls[:, :2], walls[:, :2] - p
    denom = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    ok = np.abs(denom) > eps
    safe = np.where(ok, denom, 1.0)
    t = (ap[:, 0] * s[:, 1] - ap[:, 1] * s[:, 0]) / safe
    u = (ap[:, 0] * r[:, 1] - ap[:, 1] * r[:, 0]) / safe
    hit = ok & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    return np.where(hit, t, np.nan), np.where(hit, u, np.nan)


def crossing_matrix(p, q, walls) -> np.ndarray:
    """``(M, W)`` bool: does link ``m`` cross wall ``w``?"""
    t, _ = intersection_params(p, q, walls)
    return ~np.isnan(t)


def count_crossings(p, q, walls, weights=None, *, exclude=None) -> np.ndarray:
    """Number of walls each link crosses, or the sum of ``weights`` (W,) over the crossed walls.

    ``exclude`` (M,) optionally names one wall per link to ignore (e.g. the wall a reflected
    path bounces off, which its legs touch at their end point). Computed chunk by chunk, so
    no (M, W) matrix is kept.
    """
    p, q = _pairs(p, q)
    walls = _as_walls(walls)
    w = np.ones(len(walls)) if weights is None else np.broadcast_to(np.asarray(weights, dtype=np.float64),
                                                                     (len(walls),))
    out = np.zeros(len(p))
    if len(walls) == 0:
        return out.astype(np.int64) if weights is None else out
    excl = None if exclude is None else np.broadcast_to(np.asarray(exclude, dtype=np.int64), (len(p),))
    for rows, t, _ in _hits(p, q, walls):
        hit = ~np.isnan(t)
        if excl is not None:
            hit[np.arange(len(hit)), excl[rows]] = False
        out[rows] = hit.astype(np.float64) @ w
    return np.rint(out).astype(np.int64) if weights is None else out


def point_segment_distance(points, walls) -> np.ndarray:
    """``(M, W)`` Euclidean distance from each point to each wall segment."""
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))[:, :2]
    walls = _as_walls(walls)
    out = np.empty((len(points), len(walls)))
    if len(walls) == 0:
        return out
    a, s = walls[:, :2], walls[:, 2:] - walls[:, :2]
    ss = np.maximum(np.einsum("wi,wi->w", s, s), 1e-300)
    for rows in _chunks(len(points), len(walls)):
        ap = points[rows][:, None, :] - a[None]                        # (m, W, 2)
        u = np.clip(np.einsum("mwi,wi->mw", ap, s) / ss, 0.0, 1.0)
        diff = ap - u[..., None] * s[None]
        out[rows] = np.hypot(diff[..., 0], diff[..., 1])
    return out


def segment_segment_distance(p, q, walls) -> np.ndarray:
    """``(M, W)`` minimum distance between the links ``p -> q`` and the wall segments (0 if they cross)."""
    p, q = _pairs(p, q)
    walls = _as_walls(walls)
    if len(walls) == 0:
        return np.empty((len(p), 0))
    d = np.minimum(point_segment_distance(p, walls), point_segment_distance(q, walls))
    links = np.concatenate([p, q], axis=1)
    d = np.minimum(d, point_segment_distance(walls[:, :2], links).T)
    d = np.minimum(d, point_segment_distance(walls[:, 2:], links).T)
    return np.where(crossing_matrix(p, q, walls), 0.0, d)


def mirror_points(points, walls) -> np.ndarray:
    """``(M, W, 2)`` mirror images of the points across the (infinite) line of each wall.

    The image method (Allen and Berkley, JASA 1979) builds a specular reflection path
    ``tx -> wall -> rx`` as the straight line from the image of ``tx`` to ``rx``.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))[:, :2]
    walls = _as_walls(walls)
    a, s = walls[:, :2], walls[:, 2:] - walls[:, :2]
    n = np.stack([-s[:, 1], s[:, 0]], axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    dist = np.einsum("mwi,wi->mw", points[:, None, :] - a[None], n)    # signed distance to the line
    return points[:, None, :] - 2.0 * dist[..., None] * n[None]


def side_of(points, walls) -> np.ndarray:
    """``(M, W)`` sign (-1, 0, +1) of each point relative to each wall's line."""
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))[:, :2]
    walls = _as_walls(walls)
    s = walls[:, 2:] - walls[:, :2]
    ap = points[:, None, :] - walls[None, :, :2]
    return np.sign(s[None, :, 0] * ap[..., 1] - s[None, :, 1] * ap[..., 0])


def inside_rects(points, rects) -> np.ndarray:
    """``(M, R)`` bool: point inside (or on the border of) axis-aligned rectangle ``[xmin, ymin, xmax, ymax]``."""
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))[:, :2]
    rects = np.asarray(rects, dtype=np.float64).reshape(-1, 4)
    x, y = points[:, :1], points[:, 1:2]
    return (x >= rects[:, 0]) & (x <= rects[:, 2]) & (y >= rects[:, 1]) & (y <= rects[:, 3])


def wrap_angle(angle) -> np.ndarray:
    """Wrap radians to ``[-pi, pi)``."""
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


__all__ = ["count_crossings", "crossing_matrix", "inside_rects", "intersection_params", "mirror_points",
           "paired_intersection_params",
           "point_segment_distance", "segment_segment_distance", "side_of", "wrap_angle"]
