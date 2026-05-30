import numpy as np
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from utils.geometry import compute_curvature


# === Phase 1: Arc Length Parameterization ===

def _arc_length(x, y):
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return np.concatenate([[0.0], np.cumsum(ds)])


# === Phase 2: Resample ===

def _resample_by_arc(s, x, y, n_output):
    s_uniform = np.linspace(s[0], s[-1], n_output)
    x_new = np.interp(s_uniform, s, x)
    y_new = np.interp(s_uniform, s, y)
    return s_uniform, x_new, y_new


# === Phase 3: Gaussian Smooth ===

def _gaussian_kernel(sigma, radius):
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / max(sigma, 0.1))**2)
    return kernel / kernel.sum()


def _gaussian_smooth(arr, sigma):
    radius = int(3 * sigma + 0.5)
    if radius < 1:
        return arr.copy()
    kernel = _gaussian_kernel(sigma, radius)
    return np.convolve(arr, kernel, mode="same")


# === Phase 4: Curvature (imported from utils.geometry) ===


# === Phase 5: Smooth Path ===

def smooth_path(path_x, path_y, max_deviation=0.3, n_output=50):
    if len(path_x) < 3:
        raise ValueError(f"need >=3 pts, got {len(path_x)}")

    s = _arc_length(path_x, path_y)
    _, xr, yr = _resample_by_arc(s, path_x, path_y, n_output)

    best_sigma = 0.0
    path_x_arr = np.asarray(path_x)
    path_y_arr = np.asarray(path_y)
    for sigma_try in np.arange(0.5, 8.0, 0.5):
        sx = _gaussian_smooth(xr, sigma_try)
        sy = _gaussian_smooth(yr, sigma_try)
        dist_mat = np.sqrt((sx[:, None] - path_x_arr[None, :])**2 +
                           (sy[:, None] - path_y_arr[None, :])**2)
        dev_max = float(dist_mat.min(axis=0).max())
        if dev_max <= max_deviation:
            best_sigma = sigma_try
        else:
            break

    sx = _gaussian_smooth(xr, best_sigma)
    sy = _gaussian_smooth(yr, best_sigma)
    kappa = compute_curvature(sx, sy)
    return sx, sy, kappa


# === Phase 6: Test ===

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    t = np.linspace(0, 10, 30)
    x_clean = t
    y_clean = np.sin(t * 0.5)
    x_noisy = x_clean + rng.normal(0, 0.15, len(t))
    y_noisy = y_clean + rng.normal(0, 0.15, len(t))

    sx, sy, kappa = smooth_path(x_noisy, y_noisy, max_deviation=0.3, n_output=50)

    dev_max = 0.0
    for px, py in zip(x_noisy, y_noisy):
        dist = np.sqrt((sx - px)**2 + (sy - py)**2).min()
        dev_max = max(dev_max, dist)
    print(f"max deviation from input: {dev_max:.3f}m (limit 0.3m)")

    dev_clean = np.sqrt((sx - np.interp(np.linspace(0, 10, 50), t, x_clean))**2 +
                        (sy - np.interp(np.linspace(0, 10, 50), t, y_clean))**2)
    print(f"max deviation from clean: {dev_clean.max():.3f}m (reference)")
    print(f"curvature range: [{kappa.min():.4f}, {kappa.max():.4f}]")
    print(f"curvature jumps (diff): {np.abs(np.diff(kappa)).max():.6f} (should be small)")
