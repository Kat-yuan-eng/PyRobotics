import sys
import os
import numpy as np
from Localization.ekf_localizer import ekf_localize, adaptive_Q
from Localization.pf_localizer import pf_localize, pf_estimate, pf_covariance

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === Phase 1: Covariance Fusion ===

def covariance_fusion(x_ekf, P_ekf, x_pf, P_pf):
    n = len(x_ekf)
    ekf_diverged = np.max(np.abs(np.diag(P_ekf[:n, :n]))) > 100.0
    pf_diverged = np.max(np.abs(np.diag(P_pf[:n, :n]))) > 100.0
    if ekf_diverged and pf_diverged:
        return x_ekf, P_ekf
    if ekf_diverged:
        return x_pf, P_pf
    if pf_diverged:
        return x_ekf, P_ekf
    P_ekf_reg = P_ekf[:n, :n] + np.eye(n) * 1e-9
    P_pf_reg = P_pf[:n, :n] + np.eye(n) * 1e-9
    W_ekf = np.linalg.inv(P_ekf_reg)
    W_pf = np.linalg.inv(P_pf_reg)
    W_sum = W_ekf + W_pf
    W_sum += np.eye(n) * 1e-12
    x_fused = np.linalg.solve(W_sum, W_ekf @ x_ekf[:n] + W_pf @ x_pf[:n])

    x_out = x_pf.copy()
    x_out[:2] = x_fused[:2]
    if len(x_fused) >= 3:
        w_ekf_yaw = 1.0 / max(P_ekf_reg[2, 2], 1e-12)
        w_pf_yaw = 1.0 / max(P_pf_reg[2, 2], 1e-12)
        w_yaw_sum = w_ekf_yaw + w_pf_yaw
        x_out[2] = np.arctan2(
            (w_ekf_yaw * np.sin(x_ekf[2]) + w_pf_yaw * np.sin(x_pf[2])) / w_yaw_sum,
            (w_ekf_yaw * np.cos(x_ekf[2]) + w_pf_yaw * np.cos(x_pf[2])) / w_yaw_sum
        )
    if len(x_fused) >= 4:
        x_out[3] = x_fused[3]

    P_fused = np.linalg.inv(W_sum)
    P_out = P_pf.copy()
    P_out[:n, :n] = P_fused
    P_out[:n, :n] = 0.5 * (P_out[:n, :n] + P_out[:n, :n].T)
    return x_out, P_out

# === Phase 2: Fusion Localize ===

def fusion_localize(ekf_state, pf_state, u, z_gps, z_landmark, landmarks,
                    Q_ekf, R_ekf, R_motion_pf, Q_obs_pf, dt, n_threshold):
    x_ekf, P_ekf = ekf_localize(ekf_state[0], ekf_state[1], u, z_gps,
                                  Q_ekf, R_ekf, dt, adaptive=True)

    x_pf, P_pf, px, pw = pf_localize(pf_state[0], pf_state[1], u, z_landmark,
                                        landmarks, R_motion_pf, Q_obs_pf, dt, n_threshold)

    x_fused, P_fused = covariance_fusion(x_ekf, P_ekf, x_pf, P_pf)

    new_ekf_state = (x_ekf, P_ekf)
    new_pf_state = (px, pw)

    return x_fused, P_fused, new_ekf_state, new_pf_state

# === Phase 3: Simulation ===

def run_fusion_demo():
    from Localization.ekf_localizer import generate_ekf_test
    from Localization.pf_localizer import generate_pf_test, pf_init

    dt = 0.1
    NP = 200
    NTh = NP / 2.0

    Q_ekf = np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0]) ** 2
    R_ekf = np.diag([1.0, 1.0]) ** 2
    R_motion_pf = np.diag([0.5, np.deg2rad(10.0)]) ** 2
    Q_obs_pf = np.diag([0.2, np.deg2rad(5.0)]) ** 2

    true_states, gps_obs, controls_ekf = generate_ekf_test(dt=dt)
    _, observations, _, landmarks = generate_pf_test(dt=dt)
    n_steps = len(true_states)

    x_init = true_states[0]
    x_ekf = x_init.copy()
    P_ekf = np.eye(4)
    px, pw = pf_init(NP, x_init, 0.5)

    est_ekf = np.zeros((n_steps, 4))
    est_pf = np.zeros((n_steps, 4))
    est_fused = np.zeros((n_steps, 4))

    for i in range(n_steps):
        z_gps = gps_obs[i] if not np.any(np.isnan(gps_obs[i])) else None
        z_lm = observations[i]

        x_fused, P_fused, (x_ekf, P_ekf), (px, pw) = fusion_localize(
            (x_ekf, P_ekf), (px, pw), controls_ekf[i], z_gps, z_lm, landmarks,
            Q_ekf, R_ekf, R_motion_pf, Q_obs_pf, dt, NTh
        )

        est_ekf[i] = x_ekf
        est_pf[i] = pf_estimate(px, pw) if i > 0 else x_init
        est_fused[i] = x_fused

    err_ekf = est_ekf[:, :2] - true_states[:, :2]
    err_pf = est_pf[:, :2] - true_states[:, :2]
    err_fused = est_fused[:, :2] - true_states[:, :2]

    rmse_ekf = np.sqrt(np.mean(err_ekf ** 2))
    rmse_pf = np.sqrt(np.mean(err_pf ** 2))
    rmse_fused = np.sqrt(np.mean(err_fused ** 2))

    print(f"[Fusion] EKF RMSE = {rmse_ekf:.3f}m")
    print(f"[Fusion] PF  RMSE = {rmse_pf:.3f}m")
    print(f"[Fusion] Fused RMSE = {rmse_fused:.3f}m")
    return rmse_ekf, rmse_pf, rmse_fused

show_animation = True

def main():
    import matplotlib.pyplot as plt
    from Localization.ekf_localizer import generate_ekf_test
    from Localization.pf_localizer import generate_pf_test, pf_init
    sys.path.insert(0, _PROJECT_ROOT)

    dt = 0.1
    NP = 200
    NTh = NP / 2.0
    Q_ekf = np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0]) ** 2
    R_ekf = np.diag([1.0, 1.0]) ** 2
    R_motion_pf = np.diag([0.5, np.deg2rad(10.0)]) ** 2
    Q_obs_pf = np.diag([0.2, np.deg2rad(5.0)]) ** 2

    true_states, gps_obs, controls_ekf = generate_ekf_test(dt=dt)
    _, observations, _, landmarks = generate_pf_test(dt=dt)
    n_steps = len(true_states)

    x_init = true_states[0]
    x_ekf = x_init.copy()
    P_ekf = np.eye(4)
    px, pw = pf_init(NP, x_init, 0.5)

    est_ekf = np.zeros((n_steps, 4))
    est_pf = np.zeros((n_steps, 4))
    est_fused = np.zeros((n_steps, 4))

    if show_animation:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(n_steps):
        z_gps = gps_obs[i] if not np.any(np.isnan(gps_obs[i])) else None
        z_lm = observations[i]

        x_fused, P_fused, (x_ekf, P_ekf), (px, pw) = fusion_localize(
            (x_ekf, P_ekf), (px, pw), controls_ekf[i], z_gps, z_lm, landmarks,
            Q_ekf, R_ekf, R_motion_pf, Q_obs_pf, dt, NTh
        )

        est_ekf[i] = x_ekf
        est_pf[i] = pf_estimate(px, pw) if i > 0 else x_init
        est_fused[i] = x_fused

        if show_animation and i % 5 == 0:
            plt.cla()
            ax.plot(true_states[:i + 1, 0], true_states[:i + 1, 1], "-b", label="Truth")
            ax.plot(est_ekf[:i + 1, 0], est_ekf[:i + 1, 1], "-k", label="EKF")
            ax.plot(est_pf[:i + 1, 0], est_pf[:i + 1, 1], "-g", label="PF")
            ax.plot(est_fused[:i + 1, 0], est_fused[:i + 1, 1], "-r", label="Fusion")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.legend(loc="upper left", frameon=True, fancybox=True)
            ax.grid(True)
            ax.axis("equal")
            plt.pause(0.001)

    if show_animation:
        plt.ioff()

    err_ekf = np.sqrt((est_ekf[:, 0] - true_states[:, 0]) ** 2 +
                       (est_ekf[:, 1] - true_states[:, 1]) ** 2)
    err_pf = np.sqrt((est_pf[:, 0] - true_states[:, 0]) ** 2 +
                      (est_pf[:, 1] - true_states[:, 1]) ** 2)
    err_fused = np.sqrt((est_fused[:, 0] - true_states[:, 0]) ** 2 +
                         (est_fused[:, 1] - true_states[:, 1]) ** 2)
    t = np.arange(n_steps) * dt

    fig2, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
    ax.plot(t, err_ekf, "-k", label="EKF")
    ax.plot(t, err_pf, "-g", label="PF")
    ax.plot(t, err_fused, "-r", label="Fusion")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error [m]")
    ax.legend(frameon=True, fancybox=True)
    ax.grid(True)

    rmse_ekf = np.sqrt(np.mean(err_ekf ** 2))
    rmse_pf = np.sqrt(np.mean(err_pf ** 2))
    rmse_fused = np.sqrt(np.mean(err_fused ** 2))
    print(f"[Fusion] EKF RMSE = {rmse_ekf:.3f}m")
    print(f"[Fusion] PF  RMSE = {rmse_pf:.3f}m")
    print(f"[Fusion] Fused RMSE = {rmse_fused:.3f}m")

    plt.show()

if __name__ == "__main__":
    main()
