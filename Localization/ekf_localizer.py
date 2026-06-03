import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from utils.angle import normalize_angle

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === Phase 1: Motion Model ===

def motion_model(x, u, dt):
    x_pred = x.copy()
    x_pred[0] = x[0] + x[3] * np.cos(x[2]) * dt
    x_pred[1] = x[1] + x[3] * np.sin(x[2]) * dt
    x_pred[2] = x[2] + u[1] * dt
    x_pred[3] = max(u[0], 0.0)
    return x_pred

def jacob_motion(x, u, dt):
    jF = np.eye(4)
    jF[0, 2] = -dt * x[3] * np.sin(x[2])
    jF[0, 3] = dt * np.cos(x[2])
    jF[1, 2] = dt * x[3] * np.cos(x[2])
    jF[1, 3] = dt * np.sin(x[2])
    jF[3, 3] = 0.0
    return jF

# === Phase 2: EKF Predict ===

def ekf_predict(x_est, P_est, u, Q, dt):
    x_pred = motion_model(x_est, u, dt)
    jF = jacob_motion(x_est, u, dt)
    P_pred = jF @ P_est @ jF.T + Q
    return x_pred, P_pred

# === Phase 3: EKF Update ===

def ekf_update(x_pred, P_pred, z, R, jH=None):
    if jH is None:
        jH = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])
    H = jH
    z_pred = H @ x_pred
    y = z - z_pred
    S = H @ P_pred @ H.T + R
    S += np.eye(len(S)) * 1e-9
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_est = x_pred + K @ y
    x_est[3] = max(x_est[3], 0.0)
    IKH = np.eye(4) - K @ H
    P_est = IKH @ P_pred @ IKH.T + K @ R @ K.T
    P_est = (P_est + P_est.T) / 2
    return x_est, P_est

# === Phase 4: Adaptive Q ===

def adaptive_Q(Q_base, u, accel_scale=0.5, curvature_scale=3.0):
    accel_factor = 1.0 + accel_scale * np.abs(u[0])
    curvature_factor = 1.0 + curvature_scale * np.abs(u[1])
    Q_adaptive = Q_base * accel_factor * curvature_factor
    return Q_adaptive

# === Phase 5: EKF Localize ===

def ekf_localize(x_est, P_est, u, z, Q, R, dt, adaptive=True):
    if adaptive:
        Q_eff = adaptive_Q(Q, u)
    else:
        Q_eff = Q
    x_pred, P_pred = ekf_predict(x_est, P_est, u, Q_eff, dt)
    if z is not None:
        z = np.asarray(z, dtype=float)
        if np.all(np.isfinite(z)):
            x_est, P_est = ekf_update(x_pred, P_pred, z, R)
        else:
            x_est, P_est = x_pred, P_pred
    else:
        x_est, P_est = x_pred, P_est
    eigvals = np.linalg.eigvalsh(P_est)
    if np.min(eigvals) < 1e-9:
        P_est += np.eye(4) * (1e-9 - np.min(eigvals) + 1e-9)
    if np.max(np.abs(np.diag(P_est))) > 100.0:
        P_est = np.eye(4) * 0.01
    if not np.all(np.isfinite(x_est)):
        x_est = x_pred
        P_est = P_pred
    return x_est, P_est

# === Phase 6: Simulation ===

def generate_ekf_test(n_steps=500, dt=0.1, v_noise=0.3, yaw_noise_deg=5.0):
    rng = np.random.default_rng(42)
    true_states = np.zeros((n_steps, 4))
    gps_obs = np.full((n_steps, 2), np.nan)
    controls = np.zeros((n_steps, 2))

    radius = 10.0
    v_const = 1.0
    yaw_rate = v_const / radius

    for i in range(n_steps):
        t = i * dt
        true_states[i, 0] = radius * np.sin(yaw_rate * t)
        true_states[i, 1] = radius * (1 - np.cos(yaw_rate * t))
        true_states[i, 2] = yaw_rate * t
        true_states[i, 3] = v_const
        controls[i, 0] = v_const + rng.normal(0, v_noise)
        controls[i, 1] = yaw_rate + rng.normal(0, np.deg2rad(yaw_noise_deg))
        if i % 5 == 0:
            gps_obs[i] = true_states[i, :2] + rng.normal(0, 1.0, 2)

    return true_states, gps_obs, controls

def run_ekf_demo():
    dt = 0.1
    Q = np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0]) ** 2
    R = np.diag([1.0, 1.0]) ** 2

    true_states, gps_obs, controls = generate_ekf_test(dt=dt)
    n_steps = len(true_states)

    x_est = true_states[0].copy()
    P_est = np.eye(4) * 0.01
    est_states = np.zeros((n_steps, 4))

    for i in range(n_steps):
        z = gps_obs[i] if not np.any(np.isnan(gps_obs[i])) else None
        x_est, P_est = ekf_localize(x_est, P_est, controls[i], z, Q, R, dt)
        est_states[i] = x_est

    err = est_states[:, :2] - true_states[:, :2]
    rmse = np.sqrt(np.mean(err ** 2))

    dr_states = np.zeros((n_steps, 4))
    for i in range(1, n_steps):
        dr_states[i] = motion_model(dr_states[i-1], controls[i], dt)
    dr_err = dr_states[:, :2] - true_states[:, :2]
    dr_rmse = np.sqrt(np.mean(dr_err ** 2))

    print(f"[EKF] RMSE = {rmse:.3f}m  (Dead Reckoning RMSE = {dr_rmse:.3f}m)")
    print(f"[EKF] Improvement = {(1 - rmse/dr_rmse)*100:.1f}%")
    return rmse, dr_rmse

SHOW_ANIMATION = True

def main():
    dt = 0.1
    Q = np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0]) ** 2
    R = np.diag([1.0, 1.0]) ** 2

    true_states, gps_obs, controls = generate_ekf_test(dt=dt)
    n_steps = len(true_states)

    x_est = true_states[0].copy()
    P_est = np.eye(4) * 0.01
    est_states = np.zeros((n_steps, 4))
    dr_states = np.zeros((n_steps, 4))
    dr_states[0] = true_states[0].copy()

    if SHOW_ANIMATION:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(n_steps):
        z = gps_obs[i] if not np.any(np.isnan(gps_obs[i])) else None
        x_est, P_est = ekf_localize(x_est, P_est, controls[i], z, Q, R, dt)
        est_states[i] = x_est
        if i > 0:
            dr_states[i] = motion_model(dr_states[i - 1], controls[i], dt)

        if SHOW_ANIMATION and i % 5 == 0:
            plt.cla()
            ax.plot(true_states[:i + 1, 0], true_states[:i + 1, 1], "-b", label="Truth")
            ax.plot(dr_states[:i + 1, 0], dr_states[:i + 1, 1], "-k", label="Dead Reckoning")
            ax.plot(est_states[:i + 1, 0], est_states[:i + 1, 1], "-r", label="EKF Estimate")
            if z is not None:
                ax.plot(gps_obs[i, 0], gps_obs[i, 1], "g+")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.legend(loc="upper left", frameon=True, fancybox=True)
            ax.grid(True)
            ax.axis("equal")
            plt.pause(0.001)

    if SHOW_ANIMATION:
        plt.ioff()

    err_pos = np.sqrt((est_states[:, 0] - true_states[:, 0]) ** 2 +
                       (est_states[:, 1] - true_states[:, 1]) ** 2)
    dr_err_pos = np.sqrt((dr_states[:, 0] - true_states[:, 0]) ** 2 +
                          (dr_states[:, 1] - true_states[:, 1]) ** 2)
    err_angle_raw = est_states[:, 2] - true_states[:, 2]
    err_angle = np.abs(np.rad2deg(normalize_angle(err_angle_raw)))
    t = np.arange(n_steps) * dt

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True)
    ax1.plot(t, err_pos, "-r", label="EKF")
    ax1.plot(t, dr_err_pos, "-k", label="Dead Reckoning")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Position Error [m]")
    ax1.legend(frameon=True, fancybox=True)
    ax1.grid(True)
    ax2.plot(t, err_angle, "-r")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Angle Error [deg]")
    ax2.grid(True)

    rmse = np.sqrt(np.mean(err_pos ** 2))
    dr_rmse = np.sqrt(np.mean(dr_err_pos ** 2))
    print(f"[EKF] RMSE = {rmse:.3f}m  (Dead Reckoning RMSE = {dr_rmse:.3f}m)")
    print(f"[EKF] Improvement = {(1 - rmse / max(dr_rmse, 1e-9)) * 100:.1f}%")

    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/ekf_localizer.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
