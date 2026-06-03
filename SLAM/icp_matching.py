import sys
import os
import numpy as np
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === Phase 1: Nearest Neighbor ===

def nearest_neighbor_association(prev_pts, curr_pts):
    n_prev = prev_pts.shape[1]
    n_curr = curr_pts.shape[1]
    if n_prev < 1 or n_curr < 1:
        raise ValueError(f"Need >=1 pts, got prev={n_prev}, curr={n_curr}")
    dx = curr_pts[0, :, None] - prev_pts[0, None, :]
    dy = curr_pts[1, :, None] - prev_pts[1, None, :]
    dist_sq = dx**2 + dy**2
    indices = np.argmin(dist_sq, axis=1)
    errors = np.sqrt(dist_sq[np.arange(len(indices)), indices])
    return indices, float(np.mean(errors))

# === Phase 2: SVD Motion Estimation ===

def svd_motion_estimation(prev_pts, curr_pts, weights=None):
    if weights is None:
        weights = np.ones(prev_pts.shape[1])
    w = weights / (weights.sum() + 1e-12)
    pm = (prev_pts * w[None, :]).sum(axis=1)
    cm = (curr_pts * w[None, :]).sum(axis=1)
    p_centered = prev_pts - pm[:, None]
    c_centered = curr_pts - cm[:, None]
    W = (c_centered * w[None, :]) @ p_centered.T
    u, s, vh = np.linalg.svd(W)
    R = (u @ vh).T
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = (u @ vh).T
    T = pm - R @ cm
    return R, T

# === Phase 3: ICP Match ===

def icp_match(source_pts, target_pts, max_iter=100, eps=1e-4, outlier_ratio=3.0,
              init_pose=None):
    if source_pts.shape[1] < 3 or target_pts.shape[1] < 3:
        raise ValueError(f"ICP needs >=3 pts, got source={source_pts.shape[1]}, target={target_pts.shape[1]}")
    if init_pose is None:
        H = np.eye(3)
        H[:2, 2] = target_pts.mean(axis=1) - source_pts.mean(axis=1)
    else:
        H = np.array(init_pose, dtype=float)
        if H.shape == (3, 3):
            pass
        elif H.shape == (3,):
            H_mat = np.eye(3)
            H_mat[0, 2] = H[0]
            H_mat[1, 2] = H[1]
            H_mat[:2, :2] = np.array([[np.cos(H[2]), -np.sin(H[2])],
                                       [np.sin(H[2]), np.cos(H[2])]])
            H = H_mat
    src_t = H[:2, :2] @ source_pts + H[:2, 2:3]
    error_hist = []
    prev_err = np.inf

    for _ in range(max_iter):
        indices, err = nearest_neighbor_association(target_pts, src_t)
        error_hist.append(err)

        if abs(prev_err - err) < eps:
            break
        prev_err = err

        dx = src_t[0, :] - target_pts[0, indices]
        dy = src_t[1, :] - target_pts[1, indices]
        dists = np.sqrt(dx**2 + dy**2)
        threshold = dists.mean() + outlier_ratio * dists.std()
        valid = dists < threshold
        if valid.sum() < 3:
            valid = np.ones(len(dists), dtype=bool)

        huber_delta = max(1.5 * np.median(dists[valid]), 1e-6)
        weights = np.ones(len(dists))
        large = dists > huber_delta
        weights[large] = huber_delta / dists[large]

        R, T = svd_motion_estimation(target_pts[:, indices[valid]], src_t[:, valid], weights[valid])
        src_t = (R @ src_t) + T[:, None]

        H_upd = np.eye(3)
        H_upd[:2, :2] = R
        H_upd[:2, 2] = T
        H = H_upd @ H

    return H[:2, :2], H[:2, 2], error_hist

show_animation = True

def main():
    rng = np.random.default_rng(42)
    n_pts = 50
    angle_deg = 15.0
    angle_rad = np.deg2rad(angle_deg)
    tx, ty = 1.5, 2.0

    source_pts = rng.uniform(-5, 5, (2, n_pts))
    R_true = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                        [np.sin(angle_rad), np.cos(angle_rad)]])
    T_true = np.array([tx, ty])
    target_pts = R_true @ source_pts + T_true[:, None]
    target_pts += rng.normal(0, 0.05, target_pts.shape)

    R_est, T_est, err_hist = icp_match(source_pts, target_pts, max_iter=50)
    angle_est = np.rad2deg(np.arctan2(R_est[1, 0], R_est[0, 0]))
    pos_err = np.linalg.norm(T_est - T_true)
    print(f"[ICP] True: angle={angle_deg:.1f}deg, tx={tx:.2f}, ty={ty:.2f}")
    print(f"[ICP] Est:  angle={angle_est:.1f}deg, tx={T_est[0]:.2f}, ty={T_est[1]:.2f}")
    print(f"[ICP] Error: pos={pos_err:.4f}m, angle={abs(angle_est - angle_deg):.2f}deg")
    print(f"[ICP] Convergence: {len(err_hist)} iters, err {err_hist[0]:.4f} -> {err_hist[-1]:.4f}")

    transformed = R_est @ source_pts + T_est[:, None]

    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    ax.plot(source_pts[0], source_pts[1], ".r", label="Source")
    ax.plot(target_pts[0], target_pts[1], ".b", label="Target")
    ax.plot(transformed[0], transformed[1], ".g", label="Transformed")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"ICP: angle={angle_est:.1f}deg, pos_err={pos_err:.3f}m")
    ax.legend(frameon=True, fancybox=True)
    ax.grid(True)
    ax.axis("equal")
    plt.show()

if __name__ == "__main__":
    main()
