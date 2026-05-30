import sys
import os
import numpy as np

from utils.angle import normalize_angle

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === Phase 1: Particle Init ===

def pf_init(n_particles, x_init, spread):
    rng = np.random.default_rng()
    px = np.tile(x_init, (n_particles, 1)).T
    px += rng.normal(0, spread, px.shape)
    pw = np.ones((1, n_particles)) / n_particles
    return px, pw

# === Phase 2: Particle Predict ===

def adaptive_R_motion(R_base, u, curvature_scale=3.0, accel_scale=0.5):
    yaw_factor = 1.0 + curvature_scale * np.abs(u[1])
    accel_factor = 1.0 + accel_scale * np.abs(u[0])
    R_adaptive = R_base * yaw_factor * accel_factor
    return R_adaptive

def pf_predict(px, u, R_motion, dt):
    rng = np.random.default_rng()
    noise = rng.multivariate_normal(np.zeros(2), R_motion, px.shape[1]).T
    u_noised = np.tile(u, (px.shape[1], 1)).T + noise

    px[0] += u_noised[0] * np.cos(px[2]) * dt
    px[1] += u_noised[0] * np.sin(px[2]) * dt
    px[2] += u_noised[1] * dt
    px[3] = np.maximum(u_noised[0], 0.0)
    return px

# === Phase 3: Particle Update ===

def pf_update(px, pw, z, landmarks, Q_obs):
    n_particles = px.shape[1]

    for obs_idx in range(z.shape[0]):
        r_obs = z[obs_idx, 0]
        b_obs = z[obs_idx, 1]
        lm_id = int(z[obs_idx, 2])

        if lm_id >= landmarks.shape[0]:
            continue

        lm_pos = landmarks[lm_id]
        dx = lm_pos[0] - px[0]
        dy = lm_pos[1] - px[1]
        r_pred = np.sqrt(dx**2 + dy**2)
        b_pred = np.arctan2(dy, dx) - px[2]

        r_err = r_obs - r_pred
        b_err = b_obs - b_pred
        b_err = normalize_angle(b_err)

        w = np.exp(-0.5 * (r_err**2 / max(Q_obs[0, 0], 1e-12) + b_err**2 / max(Q_obs[1, 1], 1e-12)))
        w = np.maximum(w, 1e-300)
        pw = pw * w.reshape(1, -1)

    pw_sum = pw.sum()
    if pw_sum > 1e-300:
        pw = pw / pw_sum
    else:
        pw = np.ones((1, n_particles)) / n_particles

    return px, pw

# === Phase 4: Resample ===

def pf_roughening(px, pw, K=0.12):
    n_particles = px.shape[1]
    n_dim = px.shape[0]
    x_est = pf_estimate(px, pw)
    diff = px - x_est.reshape(-1, 1)
    d_max = np.max(np.abs(diff), axis=1)
    d_max = np.maximum(d_max, 1e-6)
    sigma = K * d_max
    sigma = np.maximum(sigma, 1e-9)
    rng = np.random.default_rng()
    noise = rng.normal(0, 1, px.shape) * sigma.reshape(-1, 1)
    px = px + noise
    return px

def pf_partial_reset(px, pw, x_est, reset_frac=0.05, reset_spread=1.0):
    n_particles = px.shape[1]
    n_reset = max(1, int(n_particles * reset_frac))
    rng = np.random.default_rng()
    reset_particles = np.tile(x_est, (n_reset, 1)).T
    reset_particles += rng.normal(0, reset_spread, reset_particles.shape)
    idx = rng.choice(n_particles, n_reset, replace=False)
    px[:, idx] = reset_particles
    pw = np.ones((1, n_particles)) / n_particles
    return px, pw

def pf_resample(px, pw, n_threshold):
    n_eff = 1.0 / max((pw @ pw.T)[0, 0], 1e-300)
    n_particles = px.shape[1]
    if n_eff < n_threshold:
        indices = systematic_resample(pw.flatten(), n_particles)
        px = px[:, indices]
        pw = np.ones((1, n_particles)) / n_particles
        px = pf_roughening(px, pw)
        if n_eff < n_threshold * 0.3:
            x_est = pf_estimate(px, pw)
            px, pw = pf_partial_reset(px, pw, x_est, reset_frac=0.1, reset_spread=0.5)
    return px, pw

def systematic_resample(weights, n):
    positions = (np.arange(n) + np.random.uniform()) / n
    cumsum = np.cumsum(weights)
    indices = np.searchsorted(cumsum, positions)
    indices = np.clip(indices, 0, n - 1)
    return indices

# === Phase 5: Estimate ===

def pf_estimate(px, pw):
    x_est = (px * pw).sum(axis=1)
    x_est[2] = np.arctan2(
        (np.sin(px[2]) * pw).sum(),
        (np.cos(px[2]) * pw).sum()
    )
    return x_est

def pf_covariance(px, pw):
    x_est = pf_estimate(px, pw)
    diff = px - x_est.reshape(-1, 1)
    P = (diff * pw) @ diff.T
    return P

# === Phase 6: PF Localize ===

def pf_localize(px, pw, u, z, landmarks, R_motion, Q_obs, dt, n_threshold,
                 adaptive=True):
    if adaptive:
        R_eff = adaptive_R_motion(R_motion, u)
    else:
        R_eff = R_motion
    px = pf_predict(px, u, R_eff, dt)
    if z is not None and len(z) > 0:
        z = np.asarray(z, dtype=float)
        valid = np.all(np.isfinite(z[:, :2]), axis=1)
        if valid.sum() > 0:
            px, pw = pf_update(px, pw, z[valid], landmarks, Q_obs)
    px, pw = pf_resample(px, pw, n_threshold)
    x_est = pf_estimate(px, pw)
    P_est = pf_covariance(px, pw)
    P_est = 0.5 * (P_est + P_est.T)
    eigvals = np.linalg.eigvalsh(P_est)
    if np.min(eigvals) < 1e-9:
        P_est += np.eye(P_est.shape[0]) * (1e-9 - np.min(eigvals) + 1e-9)
    return x_est, P_est, px, pw

# === Phase 7: Simulation ===

def generate_pf_test(n_steps=500, dt=0.1, n_landmarks=10):
    rng = np.random.default_rng(42)
    landmarks = rng.uniform(-5, 25, (n_landmarks, 2))

    true_states = np.zeros((n_steps, 4))
    observations = [None] * n_steps
    controls = np.zeros((n_steps, 2))

    radius = 10.0
    v_const = 1.0
    yaw_rate = v_const / radius
    max_range = 15.0

    for i in range(n_steps):
        t = i * dt
        true_states[i, 0] = radius * np.sin(yaw_rate * t)
        true_states[i, 1] = radius * (1 - np.cos(yaw_rate * t))
        true_states[i, 2] = yaw_rate * t
        true_states[i, 3] = v_const
        controls[i, 0] = v_const + rng.normal(0, 0.1)
        controls[i, 1] = yaw_rate + rng.normal(0, np.deg2rad(2))

        obs_list = []
        for lm_id in range(n_landmarks):
            dx = landmarks[lm_id, 0] - true_states[i, 0]
            dy = landmarks[lm_id, 1] - true_states[i, 1]
            r = np.sqrt(dx**2 + dy**2)
            if r < max_range:
                b = np.arctan2(dy, dx) - true_states[i, 2]
                r_noisy = r + rng.normal(0, 0.2)
                b_noisy = b + rng.normal(0, np.deg2rad(5))
                obs_list.append([r_noisy, b_noisy, lm_id])
        if obs_list:
            observations[i] = np.array(obs_list)

    return true_states, observations, controls, landmarks

def run_pf_demo():
    dt = 0.1
    NP = 200
    NTh = NP / 2.0
    R_motion = np.diag([0.5, np.deg2rad(10.0)]) ** 2
    Q_obs = np.diag([0.2, np.deg2rad(5.0)]) ** 2

    true_states, observations, controls, landmarks = generate_pf_test(dt=dt)
    n_steps = len(true_states)

    x_init = true_states[0]
    px, pw = pf_init(NP, x_init, 0.5)

    est_states = np.zeros((n_steps, 4))

    for i in range(n_steps):
        z = observations[i]
        x_est, P_est, px, pw = pf_localize(px, pw, controls[i], z, landmarks,
                                             R_motion, Q_obs, dt, NTh)
        est_states[i] = x_est

    err = est_states[:, :2] - true_states[:, :2]
    rmse = np.sqrt(np.mean(err ** 2))
    print(f"[PF] RMSE = {rmse:.3f}m (NP={NP})")
    return rmse

show_animation = True

def main():
    import matplotlib.pyplot as plt
    sys.path.insert(0, _PROJECT_ROOT)

    dt = 0.1
    NP = 200
    NTh = NP / 2.0
    R_motion = np.diag([0.5, np.deg2rad(10.0)]) ** 2
    Q_obs = np.diag([0.2, np.deg2rad(5.0)]) ** 2

    true_states, observations, controls, landmarks = generate_pf_test(dt=dt)
    n_steps = len(true_states)

    x_init = true_states[0]
    px, pw = pf_init(NP, x_init, 0.5)
    est_states = np.zeros((n_steps, 4))

    if show_animation:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(n_steps):
        z = observations[i]
        x_est, P_est, px, pw = pf_localize(px, pw, controls[i], z, landmarks,
                                             R_motion, Q_obs, dt, NTh)
        est_states[i] = x_est

        if show_animation and i % 5 == 0:
            plt.cla()
            ax.plot(px[0], px[1], ".g", alpha=0.3, markersize=2)
            ax.plot(true_states[:i + 1, 0], true_states[:i + 1, 1], "-b", label="Truth")
            ax.plot(est_states[:i + 1, 0], est_states[:i + 1, 1], "-r", label="PF Estimate")
            ax.plot(landmarks[:, 0], landmarks[:, 1], "xk", markersize=8)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.legend(loc="upper left", frameon=True, fancybox=True)
            ax.grid(True)
            ax.axis("equal")
            plt.pause(0.001)

    if show_animation:
        plt.ioff()

    err_pos = np.sqrt((est_states[:, 0] - true_states[:, 0]) ** 2 +
                       (est_states[:, 1] - true_states[:, 1]) ** 2)
    t = np.arange(n_steps) * dt

    fig2, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
    ax.plot(t, err_pos, "-r")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error [m]")
    ax.grid(True)

    rmse = np.sqrt(np.mean(err_pos ** 2))
    print(f"[PF] RMSE = {rmse:.3f}m (NP={NP})")

    plt.show()

if __name__ == "__main__":
    main()
