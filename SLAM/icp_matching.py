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

def svd_motion_estimation(prev_pts, curr_pts):
    pm = prev_pts.mean(axis=1)
    cm = curr_pts.mean(axis=1)
    p_centered = prev_pts - pm[:, None]
    c_centered = curr_pts - cm[:, None]
    W = c_centered @ p_centered.T
    u, s, vh = np.linalg.svd(W)
    R = (u @ vh).T
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = (u @ vh).T
    T = pm - R @ cm
    return R, T

# === Phase 3: ICP Match ===

def icp_match(prev_pts, curr_pts, max_iter=100, eps=1e-4, outlier_ratio=3.0,
              init_pose=None):
    if prev_pts.shape[1] < 3 or curr_pts.shape[1] < 3:
        raise ValueError(f"ICP needs >=3 pts, got prev={prev_pts.shape[1]}, curr={curr_pts.shape[1]}")
    if init_pose is None:
        H = np.eye(3)
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
    curr_t = H[:2, :2] @ curr_pts + H[:2, 2:3]
    error_hist = []
    prev_err = np.inf

    for _ in range(max_iter):
        indices, err = nearest_neighbor_association(prev_pts, curr_t)
        error_hist.append(err)

        if abs(prev_err - err) < eps:
            break
        prev_err = err

        dx = curr_t[0, :] - prev_pts[0, indices]
        dy = curr_t[1, :] - prev_pts[1, indices]
        dists = np.sqrt(dx**2 + dy**2)
        threshold = dists.mean() + outlier_ratio * dists.std()
        valid = dists < threshold
        if valid.sum() < 3:
            valid = np.ones(len(dists), dtype=bool)

        huber_delta = max(1.5 * np.median(dists[valid]), 1e-6)
        weights = np.ones(len(dists))
        large = dists > huber_delta
        weights[large] = huber_delta / dists[large]

        R, T = svd_motion_estimation(prev_pts[:, indices[valid]], curr_t[:, valid])
        curr_t = (R @ curr_t) + T[:, None]

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

    if show_animation:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 10))

    H = np.eye(3)
    curr_pts = source_pts.copy()
    max_iter = 50
    eps = 1e-4
    prev_err = np.inf

    for iteration in range(max_iter):
        indices, err = nearest_neighbor_association(target_pts, curr_pts)
        if abs(prev_err - err) < eps:
            break
        prev_err = err

        R, T = svd_motion_estimation(target_pts[:, indices], curr_pts)
        curr_pts = (R @ curr_pts) + T[:, None]

        H_upd = np.eye(3)
        H_upd[:2, :2] = R
        H_upd[:2, 2] = T
        H = H_upd @ H

        if show_animation:
            plt.cla()
            ax.plot(source_pts[0], source_pts[1], ".r", label="Source")
            ax.plot(target_pts[0], target_pts[1], ".b", label="Target")
            ax.plot(curr_pts[0], curr_pts[1], ".g", label="Transformed")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title(f"ICP Iteration {iteration + 1}, Error: {err:.4f}")
            ax.legend(loc="upper left", frameon=True, fancybox=True)
            ax.grid(True)
            ax.axis("equal")
            plt.pause(0.001)

    if show_animation:
        plt.ioff()

    R_est = H[:2, :2]
    T_est = H[:2, 2]
    angle_est = np.rad2deg(np.arctan2(R_est[1, 0], R_est[0, 0]))
    print(f"[ICP] True: angle={angle_deg:.1f}deg, tx={tx:.2f}, ty={ty:.2f}")
    print(f"[ICP] Est:  angle={angle_est:.1f}deg, tx={T_est[0]:.2f}, ty={T_est[1]:.2f}")

    fig2, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    ax.plot(source_pts[0], source_pts[1], ".r", label="Source")
    ax.plot(target_pts[0], target_pts[1], ".b", label="Target")
    ax.plot(curr_pts[0], curr_pts[1], ".g", label="Transformed")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(frameon=True, fancybox=True)
    ax.grid(True)
    ax.axis("equal")

    plt.show()

if __name__ == "__main__":
    main()
