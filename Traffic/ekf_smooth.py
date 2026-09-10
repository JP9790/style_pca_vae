"""
Extended Kalman filter on a discrete bicycle kinematic model (feet, seconds, rad).

State :math:`x = [p_x, p_y, \\psi, v]`. Yaw rate ``omega`` and longitudinal ``a``
are taken from raw finite-difference kinematics each step (exogenous), so the
filter fuses noisy positions while respecting bicycle propagation.
"""
from __future__ import annotations

import numpy as np

from Traffic.config import DT_S, EKF_MEASUREMENT_NOISE_POS_FT2, EKF_PROCESS_NOISE_VEL_FT2_S2


def _unwrap(a: np.ndarray) -> np.ndarray:
    return np.unwrap(a)


def smooth_track_bicycle_ekf(
    xy: np.ndarray,
    dt: float = DT_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    xy : (N, 2)
        Local_X, Local_Y (feet), time ascending, uniform ``dt``.

    Returns
    -------
    xy_filt, vel, acc, speed
        Filtered positions, velocity (ft/s), acceleration (ft/s²), speed magnitude.
    """
    n = xy.shape[0]
    if n < 3:
        z = np.zeros((n, 2))
        sp = np.zeros(n)
        return xy.copy(), z, z, sp

    dx = np.diff(xy[:, 0])
    dy = np.diff(xy[:, 1])
    psi_raw = np.arctan2(dy, dx)  # length n-1: heading of each segment
    psi_seg = _unwrap(psi_raw)
    psi = np.zeros(n)
    psi[:-1] = psi_seg
    psi[-1] = psi_seg[-1]

    spd = np.zeros(n)
    for i in range(n - 1):
        spd[i] = np.hypot(xy[i + 1, 0] - xy[i, 0], xy[i + 1, 1] - xy[i, 1]) / dt
    spd[-1] = spd[-2]

    omega = np.zeros(n)
    omega[1:-1] = (psi[2:] - psi[:-2]) / (2 * dt)
    omega[0] = (psi[1] - psi[0]) / dt
    omega[-1] = (psi[-1] - psi[-2]) / dt

    acc_long = np.zeros(n)
    acc_long[1:-1] = (spd[2:] - spd[:-2]) / (2 * dt)
    acc_long[0] = (spd[1] - spd[0]) / dt
    acc_long[-1] = (spd[-1] - spd[-2]) / dt

    Q = np.diag([0.25, 0.25, 0.02, EKF_PROCESS_NOISE_VEL_FT2_S2])
    R = np.eye(2) * EKF_MEASUREMENT_NOISE_POS_FT2
    H = np.zeros((2, 4))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    I4 = np.eye(4)

    x = np.array(
        [xy[0, 0], xy[0, 1], psi[0], max(float(spd[0]), 0.1)],
        dtype=np.float64,
    )
    P = np.eye(4) * 0.5

    out_xy = np.zeros_like(xy)
    out_v = np.zeros((n, 2))
    out_spd = np.zeros(n)

    for k in range(n):
        om = omega[k]
        ak = acc_long[k]
        px, py, ps, v = x

        # Predict
        px_p = px + v * np.cos(ps) * dt
        py_p = py + v * np.sin(ps) * dt
        ps_p = ps + om * dt
        v_p = max(v + ak * dt, 0.0)
        x_pred = np.array([px_p, py_p, ps_p, v_p])

        F = np.eye(4)
        F[0, 2] = -v * np.sin(ps) * dt
        F[0, 3] = np.cos(ps) * dt
        F[1, 2] = v * np.cos(ps) * dt
        F[1, 3] = np.sin(ps) * dt
        F[2, 2] = 1.0
        F[3, 3] = 1.0
        P_pred = F @ P @ F.T + Q

        # Update
        z = xy[k]
        h = x_pred[:2]
        y_innov = z - h
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ y_innov
        P = (I4 - K @ H) @ P_pred

        out_xy[k, 0], out_xy[k, 1] = x[0], x[1]
        out_spd[k] = x[3]
        out_v[k, 0] = x[3] * np.cos(x[2])
        out_v[k, 1] = x[3] * np.sin(x[2])

    acc = np.zeros_like(out_v)
    acc[1:-1] = (out_v[2:] - out_v[:-2]) / (2 * dt)
    acc[0] = (out_v[1] - out_v[0]) / dt
    acc[-1] = (out_v[-1] - out_v[-2]) / dt

    return out_xy, out_v, acc, out_spd
