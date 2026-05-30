import sys
import os
import numpy as np
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.angle import normalize_angle

# === Phase 1: Particle Operations ===

STATE_SIZE = 3
LM_SIZE = 2
REG_EPS = 1e-9

def create_particle(x, y, yaw, n_landmarks=0, w=1.0, p_cov=0.01):
    return {
        "w": w,
        "x": x, "y": y, "yaw": yaw,
        "lm": np.full((n_landmarks, LM_SIZE), np.nan),
        "lmP": np.zeros((n_landmarks, LM_SIZE, LM_SIZE)),
        "P": np.eye(STATE_SIZE) * p_cov
    }

def particle_pose(p):
    return np.array([p["x"], p["y"], p["yaw"]])

def set_particle_pose(p, pose):
    p["x"] = float(pose[0])
    p["y"] = float(pose[1])
    p["yaw"] = float(pose[2])

# === Phase 2: Motion Model ===

def motion_model_slam(pose, u, dt):
    x, y, yaw = pose
    v, yaw_rate = u
    return np.array([
        x + v * np.cos(yaw) * dt,
        y + v * np.sin(yaw) * dt,
        yaw + yaw_rate * dt
    ])

# === Phase 3: Predict Particles ===

def adaptive_R_motion_slam(R_base, u, curvature_scale=3.0, accel_scale=0.5):
    yaw_factor = 1.0 + curvature_scale * np.abs(u[1])
    accel_factor = 1.0 + accel_scale * np.abs(u[0])
    return R_base * yaw_factor * accel_factor

def predict_particles(particles, u, R_motion, dt):
    rng = np.random.default_rng()
    v_nom, yaw_rate_nom = u
    for p in particles:
        yaw = p["yaw"]
        F = np.array([
            [1.0, 0.0, -v_nom * np.sin(yaw) * dt],
            [0.0, 1.0,  v_nom * np.cos(yaw) * dt],
            [0.0, 0.0, 1.0]
        ])
        G = np.array([
            [np.cos(yaw) * dt, 0.0],
            [np.sin(yaw) * dt, 0.0],
            [0.0, dt]
        ])
        p["P"] = F @ p["P"] @ F.T + G @ R_motion @ G.T
        p["P"] = 0.5 * (p["P"] + p["P"].T)
        noise = rng.multivariate_normal(np.zeros(2), R_motion)
        u_noised = u + noise
        new_pose = motion_model_slam(particle_pose(p), u_noised, dt)
        set_particle_pose(p, new_pose)
    return particles

# === Phase 4: Observation Model ===

def observation_model(pose, lm_pos):
    dx = lm_pos[0] - pose[0]
    dy = lm_pos[1] - pose[1]
    r = np.sqrt(dx**2 + dy**2)
    b = np.arctan2(dy, dx) - pose[2]
    return np.array([r, b])

def jacob_observation(pose, lm_pos):
    dx = lm_pos[0] - pose[0]
    dy = lm_pos[1] - pose[1]
    r2 = dx**2 + dy**2
    r = np.sqrt(max(r2, REG_EPS))
    return np.array([
        [dx / r, dy / r],
        [-dy / r2, dx / r2]
    ])

def jacob_observation_pose(pose, lm_pos):
    dx = lm_pos[0] - pose[0]
    dy = lm_pos[1] - pose[1]
    r2 = dx**2 + dy**2
    r = np.sqrt(max(r2, REG_EPS))
    return np.array([
        [-dx / r, -dy / r, 0.0],
        [dy / r2, -dx / r2, -1.0]
    ])

# === Phase 5: Compute Weight ===

def compute_weight(p, z, Q):
    w = 1.0
    pose = particle_pose(p)
    for obs in z:
        lm_id = int(obs[2])
        if lm_id >= len(p["lm"]) or np.isnan(p["lm"][lm_id, 0]):
            continue
        z_pred = observation_model(pose, p["lm"][lm_id])
        dz = np.array([obs[0] - z_pred[0], obs[1] - z_pred[1]])
        dz[1] = normalize_angle(dz[1])
        H = jacob_observation(pose, p["lm"][lm_id])
        S = H @ p["lmP"][lm_id] @ H.T + Q
        S += np.eye(LM_SIZE) * REG_EPS
        det_S = max(np.linalg.det(S), 1e-300)
        w *= np.exp(-0.5 * dz @ np.linalg.solve(S, dz)) / np.sqrt(2 * np.pi * det_S)
    return max(w, 1e-300)

# === Phase 6: Update Landmark ===

def update_landmark(p, z, Q):
    pose = particle_pose(p)
    for obs in z:
        lm_id = int(obs[2])
        if lm_id >= len(p["lm"]) or np.isnan(p["lm"][lm_id, 0]):
            continue
        z_pred = observation_model(pose, p["lm"][lm_id])
        dz = np.array([obs[0] - z_pred[0], obs[1] - z_pred[1]])
        dz[1] = normalize_angle(dz[1])
        H = jacob_observation(pose, p["lm"][lm_id])
        S = H @ p["lmP"][lm_id] @ H.T + Q
        S += np.eye(LM_SIZE) * REG_EPS
        K = p["lmP"][lm_id] @ H.T @ np.linalg.inv(S)
        p["lm"][lm_id] += K @ dz
        IKH = np.eye(LM_SIZE) - K @ H
        p["lmP"][lm_id] = IKH @ p["lmP"][lm_id] @ IKH.T + K @ Q @ K.T
    return p

# === Phase 7: Add New Landmark ===

def add_new_landmark(p, z, Q):
    pose = particle_pose(p)
    for obs in z:
        lm_id = int(obs[2])
        if lm_id < len(p["lm"]) and not np.isnan(p["lm"][lm_id, 0]):
            continue
        lx = pose[0] + obs[0] * np.cos(pose[2] + obs[1])
        ly = pose[1] + obs[0] * np.sin(pose[2] + obs[1])
        if lm_id >= len(p["lm"]):
            raise ValueError(f"lm_id={lm_id} >= n_landmarks={len(p['lm'])}")
        p["lm"][lm_id] = [lx, ly]
        H = jacob_observation(pose, np.array([lx, ly]))
        Q_reg = Q + np.eye(LM_SIZE) * REG_EPS
        HtQinvH = H.T @ np.linalg.inv(Q_reg) @ H
        HtQinvH += np.eye(LM_SIZE) * REG_EPS
        p["lmP"][lm_id] = np.linalg.inv(HtQinvH)
    return p

# === Phase 8: Proposal Sampling (FastSLAM 2.0) ===

def proposal_sampling(p, z, Q, alpha=1.0):
    if len(z) == 0:
        return p
    pose = particle_pose(p)
    P = p["P"].copy()

    for obs in z:
        lm_id = int(obs[2])
        if lm_id >= len(p["lm"]) or np.isnan(p["lm"][lm_id, 0]):
            continue
        z_pred = observation_model(pose, p["lm"][lm_id])
        dz = np.array([obs[0] - z_pred[0], obs[1] - z_pred[1]])
        dz[1] = normalize_angle(dz[1])
        Hv = jacob_observation_pose(pose, p["lm"][lm_id])
        Hlm = jacob_observation(pose, p["lm"][lm_id])
        Sf = Hlm @ p["lmP"][lm_id] @ Hlm.T + Q
        S = Hv @ P @ Hv.T + Sf
        S += np.eye(len(S)) * REG_EPS
        K = P @ Hv.T @ np.linalg.inv(S)
        pose = pose + alpha * K @ dz
        IKH = np.eye(STATE_SIZE) - alpha * K @ Hv
        P = IKH @ P @ IKH.T + alpha * K @ Q @ K.T
        P = 0.5 * (P + P.T)

    eigvals = np.linalg.eigvalsh(P)
    if np.min(eigvals) < REG_EPS:
        P += np.eye(STATE_SIZE) * (REG_EPS - np.min(eigvals) + REG_EPS)
    set_particle_pose(p, pose)
    p["P"] = P
    return p

# === Phase 9: Resample ===

def slam_roughening(particles, K=0.12):
    n = len(particles)
    if n == 0:
        return particles
    x_arr = np.array([p["x"] for p in particles])
    y_arr = np.array([p["y"] for p in particles])
    yaw_arr = np.array([p["yaw"] for p in particles])
    x_mean = np.mean(x_arr)
    y_mean = np.mean(y_arr)
    sin_mean = np.mean(np.sin(yaw_arr))
    cos_mean = np.mean(np.cos(yaw_arr))
    yaw_mean = np.arctan2(sin_mean, cos_mean)
    d_max = np.array([
        np.max(np.abs(x_arr - x_mean)),
        np.max(np.abs(y_arr - y_mean)),
        np.max(np.abs(np.arctan2(np.sin(yaw_arr - yaw_mean), np.cos(yaw_arr - yaw_mean))))
    ])
    sigma = K * d_max
    sigma = np.maximum(sigma, 1e-9)
    rng = np.random.default_rng()
    for p in particles:
        p["x"] += rng.normal(0, sigma[0])
        p["y"] += rng.normal(0, sigma[1])
        p["yaw"] += rng.normal(0, sigma[2])
        p["yaw"] = normalize_angle(p["yaw"])
    return particles

def resampling(particles, n_threshold):
    weights = np.array([p["w"] for p in particles])
    weights = weights / (weights.sum() + 1e-300)
    n_eff = 1.0 / (np.sum(weights**2) + 1e-300)
    did_resample = False

    if n_eff < n_threshold:
        did_resample = True
        n = len(particles)
        positions = (np.arange(n) + np.random.uniform()) / n
        cumsum = np.cumsum(weights)
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, n - 1)
        new_particles = []
        for idx in indices:
            new_particles.append({
                "w": 1.0 / n,
                "x": particles[idx]["x"],
                "y": particles[idx]["y"],
                "yaw": particles[idx]["yaw"],
                "lm": particles[idx]["lm"].copy(),
                "lmP": particles[idx]["lmP"].copy(),
                "P": particles[idx]["P"].copy()
            })
        particles = new_particles
        particles = slam_roughening(particles)

    return particles, did_resample, n_eff

# === Phase 10: FastSLAM ===

def fast_slam(particles, u, z, Q, R_motion, dt, n_threshold, adaptive=True):
    if adaptive:
        R_eff = adaptive_R_motion_slam(R_motion, u)
    else:
        R_eff = R_motion
    particles = predict_particles(particles, u, R_eff, dt)

    if z is not None and len(z) > 0:
        z = np.asarray(z, dtype=float)
        valid = np.all(np.isfinite(z[:, :2]), axis=1)
        if valid.sum() > 0:
            z_valid = z[valid]
            for p in particles:
                p = add_new_landmark(p, z_valid, Q)
                for obs in z_valid:
                    obs_single = obs.reshape(1, -1)
                    p["w"] *= compute_weight(p, obs_single, Q)
                    p = update_landmark(p, obs_single, Q)
                    p = proposal_sampling(p, obs_single, Q)

    w_sum = sum(p["w"] for p in particles)
    if w_sum > 1e-300:
        for p in particles:
            p["w"] /= w_sum
    else:
        n = len(particles)
        for p in particles:
            p["w"] = 1.0 / n

    particles, did_resample, n_eff_pre = resampling(particles, n_threshold)

    return particles, did_resample, n_eff_pre

def estimate_from_particles(particles):
    weights = np.array([p["w"] for p in particles])
    weights = weights / (weights.sum() + 1e-300)
    x_est = sum(p["x"] * w for p, w in zip(particles, weights))
    y_est = sum(p["y"] * w for p, w in zip(particles, weights))
    sin_sum = sum(np.sin(p["yaw"]) * w for p, w in zip(particles, weights))
    cos_sum = sum(np.cos(p["yaw"]) * w for p, w in zip(particles, weights))
    yaw_est = np.arctan2(sin_sum, cos_sum)

    lm_est = None
    lm_count = None
    for p in particles:
        n_lm = len(p["lm"])
        for j in range(n_lm):
            if np.isnan(p["lm"][j, 0]):
                continue
            if lm_est is None:
                lm_est = np.zeros((n_lm, LM_SIZE))
                lm_count = np.zeros(n_lm)
            lm_est[j] += p["lm"][j] * p["w"]
            lm_count[j] += p["w"]

    if lm_est is not None:
        observed = lm_count > 0
        lm_est[observed] /= lm_count[observed, None]
        lm_est[~observed] = np.nan

    return np.array([x_est, y_est, yaw_est]), lm_est

# === Phase 11: Simulation ===

def generate_slam_test(n_steps=300, dt=0.1, n_landmarks=10, seed=42):
    rng = np.random.default_rng(seed)
    landmarks = rng.uniform(-5, 25, (n_landmarks, 2))

    true_traj = np.zeros((n_steps, 3))
    observations_seq = []
    controls = np.zeros((n_steps, 2))

    radius = 10.0
    v_const = 1.0
    yaw_rate = v_const / radius
    max_range = 15.0
    sigma_r = 0.2
    sigma_b_deg = 5.0

    for i in range(n_steps):
        t = i * dt
        true_traj[i, 0] = radius * np.sin(yaw_rate * t)
        true_traj[i, 1] = radius * (1 - np.cos(yaw_rate * t))
        true_traj[i, 2] = yaw_rate * t
        controls[i, 0] = v_const
        controls[i, 1] = yaw_rate

        dx_lm = landmarks[:, 0] - true_traj[i, 0]
        dy_lm = landmarks[:, 1] - true_traj[i, 1]
        r_lm = np.sqrt(dx_lm**2 + dy_lm**2)
        mask = r_lm < max_range
        if np.any(mask):
            b_lm = np.arctan2(dy_lm[mask], dx_lm[mask]) - true_traj[i, 2]
            r_noisy = r_lm[mask] + rng.normal(0, sigma_r, size=mask.sum())
            b_noisy = b_lm + rng.normal(0, np.deg2rad(sigma_b_deg), size=mask.sum())
            lm_ids = np.arange(n_landmarks)[mask].astype(float)
            observations_seq.append(np.column_stack([r_noisy, b_noisy, lm_ids]))
        else:
            observations_seq.append(np.zeros((0, 3)))

    return true_traj, observations_seq, controls, landmarks

def run_fast_slam_demo():
    dt = 0.1
    NP = 200
    NTh = NP * 0.5
    Q = np.diag([0.2, np.deg2rad(5.0)]) ** 2
    R_motion = np.diag([0.2, np.deg2rad(5.0)]) ** 2

    true_traj, observations_seq, controls, landmarks = generate_slam_test(dt=dt)
    n_steps = len(true_traj)
    n_landmarks = len(landmarks)

    particles = [create_particle(true_traj[0, 0], true_traj[0, 1], true_traj[0, 2],
                                  n_landmarks=n_landmarks)
                 for _ in range(NP)]

    est_traj = np.zeros((n_steps, 3))
    resample_count = 0
    n_eff_min = float(NP)

    for i in range(n_steps):
        particles, did_resample, n_eff_pre = fast_slam(particles, controls[i], observations_seq[i], Q, R_motion, dt, NTh, adaptive=False)
        est_traj[i], P_est = estimate_from_particles(particles)
        n_eff_min = min(n_eff_min, n_eff_pre)
        if did_resample:
            resample_count += 1

    err = est_traj[:, :2] - true_traj[:, :2]
    rmse = np.sqrt(np.mean(err**2))
    print(f"[FastSLAM] RMSE = {rmse:.3f}m (NP={NP})")
    print(f"[FastSLAM] Resample={resample_count}/{n_steps}  N_eff_min={n_eff_min:.1f}")
    return rmse

show_animation = True

def main():
    dt = 0.1
    NP = 200
    NTh = NP * 0.5
    Q = np.diag([0.2, np.deg2rad(5.0)]) ** 2
    R_motion = np.diag([0.2, np.deg2rad(5.0)]) ** 2

    true_traj, observations_seq, controls, landmarks = generate_slam_test(dt=dt)
    n_steps = len(true_traj)
    n_landmarks = len(landmarks)

    particles = [create_particle(true_traj[0, 0], true_traj[0, 1], true_traj[0, 2],
                                  n_landmarks=n_landmarks)
                 for _ in range(NP)]
    est_traj = np.zeros((n_steps, 3))
    resample_count = 0
    n_eff_history = np.zeros(n_steps)

    if show_animation:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(n_steps):
        particles, did_resample, n_eff_pre = fast_slam(particles, controls[i], observations_seq[i],
                              Q, R_motion, dt, NTh, adaptive=False)
        est_traj[i], lm_est = estimate_from_particles(particles)
        n_eff_history[i] = n_eff_pre
        if did_resample:
            resample_count += 1

        if show_animation and i % 5 == 0:
            plt.cla()
            ax.plot(landmarks[:, 0], landmarks[:, 1], "xb", markersize=10,
                    label="True Landmarks")
            if lm_est is not None:
                observed = ~np.isnan(lm_est[:, 0])
                if np.any(observed):
                    ax.plot(lm_est[observed, 0], lm_est[observed, 1], "xr",
                            markersize=10, label="Est Landmarks")
            px_x = np.array([p["x"] for p in particles])
            px_y = np.array([p["y"] for p in particles])
            ax.plot(px_x, px_y, ".g", alpha=0.3, markersize=2)
            ax.plot(true_traj[:i + 1, 0], true_traj[:i + 1, 1], "-b", label="Truth")
            ax.plot(est_traj[:i + 1, 0], est_traj[:i + 1, 1], "-r", label="Estimate")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.legend(loc="upper left", frameon=True, fancybox=True)
            ax.grid(True)
            ax.axis("equal")
            plt.pause(0.001)

    if show_animation:
        plt.ioff()

    err_pos = np.sqrt((est_traj[:, 0] - true_traj[:, 0]) ** 2 +
                       (est_traj[:, 1] - true_traj[:, 1]) ** 2)
    err_angle_raw = est_traj[:, 2] - true_traj[:, 2]
    err_angle = np.abs(np.rad2deg(normalize_angle(err_angle_raw)))
    t = np.arange(n_steps) * dt

    n50 = min(50, n_steps)
    rmse_first50 = np.sqrt(np.mean(err_pos[:n50] ** 2))
    rmse_last50 = np.sqrt(np.mean(err_pos[-n50:] ** 2))
    rmse = np.sqrt(np.mean(err_pos ** 2))
    rmse_angle = np.sqrt(np.mean(err_angle ** 2))
    max_err = np.max(err_pos)
    n_eff_mean = np.mean(n_eff_history)
    n_eff_min = np.min(n_eff_history)

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True)
    ax1.plot(t, err_pos, "-r")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Position Error [m]")
    ax1.grid(True)
    ax2.plot(t, err_angle, "-r")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Angle Error [deg]")
    ax2.grid(True)

    print(f"[FastSLAM] RMSE_pos={rmse:.3f}m  RMSE_angle={rmse_angle:.2f}deg  Max={max_err:.3f}m")
    print(f"[FastSLAM] First50={rmse_first50:.3f}m  Last50={rmse_last50:.3f}m")
    print(f"[FastSLAM] Resample={resample_count}/{n_steps}  N_eff_mean={n_eff_mean:.1f}  N_eff_min={n_eff_min:.1f}")
    assert rmse < 0.5, f"RMSE={rmse:.3f}m exceeds 0.5m threshold"

    plt.show()

if __name__ == "__main__":
    main()
