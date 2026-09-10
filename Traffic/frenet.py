"""Task-aligned Frenet-style coordinates and resampling to fixed length."""

from __future__ import annotations

import numpy as np


def resample_trajectory_even_arc_length(xy: np.ndarray, n_pts: int) -> np.ndarray:
    """Resample polyline ``xy`` (N, 2) to ``n_pts`` points at uniform arc length."""
    if len(xy) < 2:
        return np.repeat(xy[:1], n_pts, axis=0)
    seg = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    u = np.concatenate([[0.0], np.cumsum(seg)])
    if u[-1] <= 0:
        return np.repeat(xy[:1], n_pts, axis=0)
    u_new = np.linspace(0.0, u[-1], n_pts)
    out = np.zeros((n_pts, 2))
    for dim in range(2):
        out[:, dim] = np.interp(u_new, u, xy[:, dim])
    return out


def task_aligned_frenet_profiles(
    xy: np.ndarray,
    dt: float,
    n_resample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Translate to segment origin, rotate so initial road direction is +s.

    Returns lateral displacement ``d(t)`` and lateral velocity ``v_d(t)`` along the
    resampled grid (``n_resample`` steps).
    """
    xy_r = resample_trajectory_even_arc_length(xy, n_resample)
    p0 = xy_r[0]
    k = min(5, xy_r.shape[0] - 1)
    dx = xy_r[k, 0] - p0[0]
    dy = xy_r[k, 1] - p0[1]
    theta = float(np.arctan2(dy, dx))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, s], [-s, c]])
    rel = (xy_r - p0) @ R.T
    d = rel[:, 1]
    v_d = np.gradient(d, dt)
    return d.astype(np.float32), v_d.astype(np.float32)


def zscore_per_segment(a: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = float(np.mean(a))
    sd = float(np.std(a))
    if sd < eps:
        return np.zeros_like(a)
    return ((a - m) / sd).astype(np.float32)
