import numpy as np


def compute_curvature(x_arr, y_arr):
    if len(x_arr) < 3:
        raise ValueError(f"curvature needs >=3 pts, got {len(x_arr)}")
    dx = np.diff(x_arr)
    dy = np.diff(y_arr)
    ds = np.sqrt(dx**2 + dy**2) + 1e-12
    ddx = np.diff(dx)
    ddy = np.diff(dy)
    kappa_inner = (dx[:-1] * ddy - dy[:-1] * ddx) / (ds[:-1]**3 + 1e-18)
    kappa = np.concatenate([[kappa_inner[0]], kappa_inner, [kappa_inner[-1]]])
    return kappa
