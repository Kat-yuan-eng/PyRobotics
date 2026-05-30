import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from SLAM.fast_slam import (fast_slam, estimate_from_particles, create_particle,
                              generate_slam_test)
from SLAM.icp_matching import icp_match

# === Phase 1: Pipeline Config ===

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True

# === Phase 2: Run SLAM Pipeline ===

def run_slam_pipeline(n_steps=300, dt=0.1, n_landmarks=10, n_particles=100):
    Q = np.diag([0.2, np.deg2rad(5.0)]) ** 2
    R_motion = np.diag([0.5, np.deg2rad(10.0)]) ** 2
    NTh = n_particles / 1.5

    true_traj, observations_seq, controls, landmarks = generate_slam_test(
        n_steps, dt, n_landmarks)

    particles = [create_particle(true_traj[0, 0], true_traj[0, 1], true_traj[0, 2],
                                  n_landmarks=n_landmarks)
                 for _ in range(n_particles)]

    est_traj = np.zeros((n_steps, 3))
    est_lm_hist = []
    rmse_hist = np.zeros(n_steps)

    for i in range(n_steps):
        particles = fast_slam(particles, controls[i], observations_seq[i],
                               Q, R_motion, dt, NTh)
        est_traj[i], lm_est = estimate_from_particles(particles)
        est_lm_hist.append(lm_est)

        err_sq = np.sum((est_traj[i, :2] - true_traj[i, :2])**2)
        rmse_hist[i] = np.sqrt(np.mean(np.sum((est_traj[:i+1, :2] - true_traj[:i+1, :2])**2, axis=1)))

    icp_errors = run_icp_test(true_traj, est_traj)

    final_lm = est_lm_hist[-1]
    if final_lm is not None and len(landmarks) > 0:
        valid = ~np.isnan(final_lm[:, 0])
        n_valid = valid.sum()
        if n_valid > 0:
            map_err = np.sqrt(np.mean((final_lm[valid] - landmarks[valid])**2))
        else:
            map_err = np.inf
    else:
        map_err = np.inf

    print(f"[SLAM Pipeline] Trajectory RMSE = {rmse_hist.mean():.3f}m")
    print(f"[SLAM Pipeline] Map Error = {map_err:.3f}m")
    print(f"[SLAM Pipeline] ICP Mean Error = {np.mean(icp_errors):.4f}m")

    fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figs')
    os.makedirs(fig_dir, exist_ok=True)
    visualize_slam(true_traj, est_traj, landmarks, final_lm, rmse_hist, fig_dir)

    return rmse_hist, map_err, est_traj

# === Phase 3: ICP Test ===

def run_icp_test(true_traj, est_traj, step=10):
    errors = []
    for i in range(step, len(true_traj), step):
        prev_true = true_traj[i - step:i, :2].T
        curr_true = true_traj[i - step + 1:i + 1, :2].T
        if prev_true.shape[1] < 3 or curr_true.shape[1] < 3:
            continue
        assert prev_true.shape[1] >= 3 and curr_true.shape[1] >= 3
        _, _, err_hist = icp_match(prev_true, curr_true)
        if len(err_hist) > 0:
            errors.append(err_hist[-1])
    return errors

# === Phase 4: Visualization ===

def visualize_slam(true_traj, est_traj, landmarks, est_landmarks, rmse_hist, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax1 = axes[0]
    ax1.plot(true_traj[:, 0], true_traj[:, 1], 'b-', linewidth=2, label='True')
    ax1.plot(est_traj[:, 0], est_traj[:, 1], 'r--', linewidth=1.5, label='Estimated')
    if landmarks is not None and len(landmarks) > 0:
        ax1.scatter(landmarks[:, 0], landmarks[:, 1], c='g', marker='*', s=100, label='Landmarks')
    if est_landmarks is not None and len(est_landmarks) > 0:
        valid = ~np.isnan(est_landmarks[:, 0])
        if valid.sum() > 0:
            ax1.scatter(est_landmarks[valid, 0], est_landmarks[valid, 1], c='r', marker='x', s=60, label='Est. Landmarks')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('SLAM Trajectory')
    ax1.legend(frameon=True, fancybox=True)
    ax1.axis('equal')

    ax2 = axes[1]
    pos_err = np.sqrt(np.sum((est_traj[:, :2] - true_traj[:, :2])**2, axis=1))
    ax2.plot(pos_err, 'r-', linewidth=1)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Position Error [m]')
    ax2.set_title('Position Error over Time')

    ax3 = axes[2]
    ax3.plot(rmse_hist, 'b-', linewidth=1)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('RMSE [m]')
    ax3.set_title('Cumulative RMSE')

    plt.tight_layout()
    save_path = os.path.join(fig_dir, 'slam_pipeline.png')
    plt.savefig(save_path, dpi=100)
    plt.show()

if __name__ == "__main__":
    run_slam_pipeline()
