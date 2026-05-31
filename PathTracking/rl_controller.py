import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
from utils.plot import generate_serpentine_course, plot_vehicle

show_animation = True

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "generated"))
sys.path.insert(0, _PROJECT_ROOT)

from generated import (DecisionOutput, Behavior, ControlOutput, ControlMode,
                       GearPosition)


# === Phase 1: Neural Network (Pure NumPy) ===

def _relu(x):
    return np.maximum(0.0, x)


def _relu_grad(x):
    return (x > 0.0).astype(np.float64)


def _forward(x, W1, b1, W2, b2, W3, b3):
    h1 = _relu(x @ W1 + b1)
    h2 = _relu(h1 @ W2 + b2)
    q = h2 @ W3 + b3
    return q, h1, h2


# === Phase 2: Environment ===

class _PathTrackEnv:
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng(42)
        self.L = 2.7
        self.dt = 0.1
        self.reset()

    def _generate_path(self):
        n_pts = 80
        t = np.linspace(0, 1, n_pts)
        self.cx = t * 50.0
        amp = self.rng.uniform(3.0, 8.0)
        freq = self.rng.uniform(1.0, 3.0)
        phase = self.rng.uniform(0, 2 * np.pi)
        self.cy = 15.0 + amp * np.sin(freq * 2 * np.pi * t + phase)
        dx = np.diff(self.cx)
        dy = np.diff(self.cy)
        self.cyaw = np.arctan2(dy, dx)
        self.cyaw = np.concatenate([self.cyaw, [self.cyaw[-1]]])
        ds = np.sqrt(dx**2 + dy**2)
        self.s_cum = np.concatenate([[0], np.cumsum(ds)])

    def reset(self):
        self._generate_path()
        self.x = self.cx[0]
        self.y = self.cy[0]
        self.theta = self.cyaw[0]
        self.v = 0.0
        self.done = False
        self.steps = 0
        self.idx_nearest = 0
        return self._obs()

    def _find_nearest(self):
        dx = self.cx - self.x
        dy = self.cy - self.y
        dist2 = dx**2 + dy**2
        self.idx_nearest = int(np.argmin(dist2))

    def _obs(self):
        self._find_nearest()
        cos_v = np.cos(self.theta)
        sin_v = np.sin(self.theta)
        dx = self.cx - self.x
        dy = self.cy - self.y
        path_x_veh = dx * cos_v + dy * sin_v
        path_y_veh = -dx * sin_v + dy * cos_v

        angles = np.linspace(-np.pi / 2, np.pi / 2, 16)
        scans = np.full(16, 10.0)
        for i, a in enumerate(angles):
            ray_dx = np.cos(a)
            ray_dy = np.sin(a)
            proj = path_x_veh * ray_dx + path_y_veh * ray_dy
            perp = np.abs(-path_x_veh * ray_dy + path_y_veh * ray_dx)
            valid = (proj > 0) & (proj < 10.0) & (perp < 2.0)
            if valid.any():
                scans[i] = float(np.min(proj[valid]))

        lookahead = min(self.idx_nearest + 15, len(self.cx) - 1)
        goal_dir = np.arctan2(self.cy[lookahead] - self.y,
                              self.cx[lookahead] - self.x) - self.theta

        lat_err = float(path_y_veh[self.idx_nearest]) / 5.0
        hdg_err = float(self.cyaw[self.idx_nearest] - self.theta)
        hdg_err = np.arctan2(np.sin(hdg_err), np.cos(hdg_err)) / np.pi

        k_idx = min(self.idx_nearest + 5, len(self.cx) - 1)
        dx2 = self.cx[min(k_idx + 1, len(self.cx) - 1)] - self.cx[max(k_idx - 1, 0)]
        dy2 = self.cy[min(k_idx + 1, len(self.cy) - 1)] - self.cy[max(k_idx - 1, 0)]
        curvature = np.clip(dy2 / max(dx2, 0.01), -1, 1)

        dist_to_goal = np.sqrt((self.cx[-1] - self.x)**2 + (self.cy[-1] - self.y)**2)
        goal_dir_final = np.arctan2(self.cy[-1] - self.y, self.cx[-1] - self.x) - self.theta

        return np.concatenate([scans / 10.0, [self.v / 10.0],
                               [np.sin(goal_dir), np.cos(goal_dir)],
                               [lat_err, hdg_err, curvature],
                               [dist_to_goal / 50.0, np.sin(goal_dir_final), np.cos(goal_dir_final)]])

    def step(self, action):
        dist_to_goal = np.sqrt((self.cx[-1] - self.x)**2 + (self.cy[-1] - self.y)**2)
        if dist_to_goal < 5.0:
            steer_options = np.radians([-10, -5, -2, 0, 2, 5, 10])
        else:
            steer_options = np.radians([-20, -13, -7, 0, 7, 13, 20])
        throttle_options = [0.2, 0.5, 0.8]
        steer = steer_options[action // 3]
        thr = throttle_options[action % 3]

        accel = thr * 5.0 - 0.5
        self.v = max(self.v + accel * self.dt, 0.0)
        self.v = min(self.v, 15.0)
        self.theta += self.v * np.tan(steer) / self.L * self.dt
        self.x += self.v * np.cos(self.theta) * self.dt
        self.y += self.v * np.sin(self.theta) * self.dt
        self.steps += 1

        self._find_nearest()
        cos_v = np.cos(self.theta)
        sin_v = np.sin(self.theta)
        dx = self.cx - self.x
        dy = self.cy - self.y
        path_y_veh = -dx * sin_v + dy * cos_v
        lat_err = float(path_y_veh[self.idx_nearest])

        hdg_err = self.cyaw[self.idx_nearest] - self.theta
        hdg_err = np.arctan2(np.sin(hdg_err), np.cos(hdg_err))

        progress = self.s_cum[self.idx_nearest]
        prev_progress = self.s_cum[max(self.idx_nearest - 1, 0)]
        reward = (progress - prev_progress) * 0.5
        reward -= abs(lat_err) * 0.3
        reward -= abs(hdg_err) * 0.5
        reward -= 0.05
        if self.v > 10.0:
            reward -= (self.v - 10.0) * 0.2

        dist_to_goal = np.sqrt((self.cx[-1] - self.x)**2 + (self.cy[-1] - self.y)**2)
        if dist_to_goal < 3.0:
            reward += 50.0 + (3.0 - dist_to_goal) * 20.0

        if self.idx_nearest >= len(self.cx) - 3:
            reward += 100.0
            self.done = True
        elif abs(lat_err) > 8.0:
            reward -= 50.0
            self.done = True
        elif self.steps >= 600:
            self.done = True

        return self._obs(), reward, self.done


# === Phase 3: DQN Training ===

def train_dqn(n_episodes=800, save_path=None, lr=0.0003, batch_size=128, gamma=0.99):
    if save_path is None:
        save_path = os.path.join(_PROJECT_ROOT, "control", "dqn_weights.npz")

    rng = np.random.default_rng(42)
    env = _PathTrackEnv(rng)

    n_obs = 25
    n_act = 21
    n_hidden1 = 128
    n_hidden2 = 64

    W1 = rng.normal(0, np.sqrt(2.0 / n_obs), (n_obs, n_hidden1))
    b1 = np.zeros(n_hidden1)
    W2 = rng.normal(0, np.sqrt(2.0 / n_hidden1), (n_hidden1, n_hidden2))
    b2 = np.zeros(n_hidden2)
    W3 = rng.normal(0, np.sqrt(2.0 / n_hidden2), (n_hidden2, n_act))
    b3 = np.zeros(n_act)

    W1_t, b1_t = W1.copy(), b1.copy()
    W2_t, b2_t = W2.copy(), b2.copy()
    W3_t, b3_t = W3.copy(), b3.copy()

    replay_max = 50000
    replay = deque(maxlen=replay_max)
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    target_update = 500
    total_steps = 0
    best_reward = -1e9
    best_W1, best_b1 = W1.copy(), b1.copy()
    best_W2, best_b2 = W2.copy(), b2.copy()
    best_W3, best_b3 = W3.copy(), b3.copy()

    for ep in range(n_episodes):
        state = env.reset()
        ep_reward = 0.0

        while not env.done:
            if rng.random() < epsilon:
                action = rng.integers(0, n_act)
            else:
                q, _, _ = _forward(state, W1, b1, W2, b2, W3, b3)
                action = int(np.argmax(q))

            next_state, reward, done = env.step(action)
            ep_reward += reward

            replay.append((state, action, reward, next_state, done))

            if len(replay) >= batch_size:
                idx = rng.choice(len(replay), batch_size, replace=False)
                s_batch = np.array([replay[i][0] for i in idx])
                a_batch = np.array([replay[i][1] for i in idx])
                r_batch = np.array([replay[i][2] for i in idx])
                s2_batch = np.array([replay[i][3] for i in idx])
                d_batch = np.array([replay[i][4] for i in idx])

                q2, _, _ = _forward(s2_batch, W1_t, b1_t, W2_t, b2_t, W3_t, b3_t)
                q_online_next, _, _ = _forward(s2_batch, W1, b1, W2, b2, W3, b3)
                best_actions = np.argmax(q_online_next, axis=1)
                target = r_batch + gamma * q2[np.arange(batch_size), best_actions] * (1.0 - d_batch)

                q_pred, h1, h2 = _forward(s_batch, W1, b1, W2, b2, W3, b3)
                td_error = q_pred.copy()
                td_error[np.arange(batch_size), a_batch] = target

                d3 = (q_pred - td_error) / batch_size
                dW3 = h2.T @ d3
                db3 = d3.sum(axis=0)

                d2 = d3 @ W3.T * _relu_grad(h1 @ W2 + b2)
                dW2 = h1.T @ d2
                db2 = d2.sum(axis=0)

                d1 = d2 @ W2.T * _relu_grad(s_batch @ W1 + b1)
                dW1 = s_batch.T @ d1
                db1 = d1.sum(axis=0)

                W1 -= lr * np.clip(dW1, -5, 5)
                b1 -= lr * np.clip(db1, -5, 5)
                W2 -= lr * np.clip(dW2, -5, 5)
                b2 -= lr * np.clip(db2, -5, 5)
                W3 -= lr * np.clip(dW3, -5, 5)
                b3 -= lr * np.clip(db3, -5, 5)

            state = next_state
            total_steps += 1

            if total_steps % target_update == 0:
                W1_t, b1_t = W1.copy(), b1.copy()
                W2_t, b2_t = W2.copy(), b2.copy()
                W3_t, b3_t = W3.copy(), b3.copy()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_W1, best_b1 = W1.copy(), b1.copy()
            best_W2, best_b2 = W2.copy(), b2.copy()
            best_W3, best_b3 = W3.copy(), b3.copy()

        if (ep + 1) % 50 == 0:
            print(f"[DQN] ep={ep + 1}/{n_episodes}  reward={ep_reward:.1f}  eps={epsilon:.3f}  best={best_reward:.1f}")

    np.savez(save_path, W1=best_W1, b1=best_b1, W2=best_W2, b2=best_b2, W3=best_W3, b3=best_b3)
    print(f"[DQN] Weights saved to {save_path}")
    return save_path


# === Phase 4: RL Control (Inference) ===

def rl_control(decision_output, speed_actual=0.0,
               wheelbase=2.7, max_steer_deg=30.0,
               weights=None, rl_state=None, dt=0.02,
               vehicle_x=0.0, vehicle_y=0.0, vehicle_theta=0.0):
    control = ControlOutput()
    control.header.timestamp_ns = int(time.time() * 1e9)
    control.header.seq = decision_output.header.seq
    control.header.frame_id = "control_rl"

    behavior = decision_output.behavior

    if behavior == Behavior.Value("BEHAVIOR_EMERGENCY_STOP"):
        control.mode = ControlMode.Value("MODE_EMERGENCY")
        control.steering.steering_angle = 0.0
        control.steering.steering_valid = True
        control.throttle_brake.throttle = 0.0
        control.throttle_brake.brake = 1.0
        control.throttle_brake.throttle_valid = True
        control.throttle_brake.brake_valid = True
        control.gear = GearPosition.Value("GEAR_DRIVE")
        control.target_speed = 0.0
        control.control_valid = True
        return control, rl_state

    control.mode = ControlMode.Value("MODE_AUTO")

    if weights is None:
        weights_path = os.path.join(_PROJECT_ROOT, "control", "dqn_weights.npz")
        if os.path.exists(weights_path):
            data = np.load(weights_path)
            weights = (data["W1"], data["b1"], data["W2"], data["b2"],
                       data["W3"], data["b3"])
        else:
            control.steering.steering_angle = 0.0
            control.steering.steering_valid = True
            control.throttle_brake.throttle = 0.5
            control.throttle_brake.throttle_valid = True
            control.throttle_brake.brake_valid = True
            control.gear = GearPosition.Value("GEAR_DRIVE")
            control.target_speed = decision_output.target_speed
            control.control_valid = False
            return control, rl_state

    W1, b1, W2, b2, W3, b3 = weights

    if rl_state is None:
        rl_state = {"prev_x": 0.0, "prev_y": 0.0, "prev_theta": 0.0}

    n_path = len(decision_output.target_path)
    if n_path < 1:
        obs = np.zeros(25)
    else:
        path_x_global = np.array([p.pose.x for p in decision_output.target_path])
        path_y_global = np.array([p.pose.y for p in decision_output.target_path])
        dx = path_x_global - vehicle_x
        dy = path_y_global - vehicle_y
        cos_v = np.cos(vehicle_theta)
        sin_v = np.sin(vehicle_theta)
        path_x_veh = dx * cos_v + dy * sin_v
        path_y_veh = -dx * sin_v + dy * cos_v

        angles = np.linspace(-np.pi / 2, np.pi / 2, 16)
        scans = np.full(16, 10.0)
        for i, a in enumerate(angles):
            ray_dx = np.cos(a)
            ray_dy = np.sin(a)
            proj = path_x_veh * ray_dx + path_y_veh * ray_dy
            perp = np.abs(-path_x_veh * ray_dy + path_y_veh * ray_dx)
            valid = (proj > 0) & (proj < 10.0) & (perp < 2.0)
            if valid.any():
                scans[i] = float(np.min(proj[valid]))

        idx_nearest = int(np.argmin(dx**2 + dy**2))
        lookahead = min(idx_nearest + 15, n_path - 1)
        goal_dir = np.arctan2(path_y_veh[lookahead], path_x_veh[lookahead])

        lat_err = float(path_y_veh[idx_nearest]) / 5.0
        hdg_err_raw = np.arctan2(
            path_y_global[min(idx_nearest + 1, n_path - 1)] - path_y_global[max(idx_nearest - 1, 0)],
            path_x_global[min(idx_nearest + 1, n_path - 1)] - path_x_global[max(idx_nearest - 1, 0)]
        ) - vehicle_theta
        hdg_err = np.arctan2(np.sin(hdg_err_raw), np.cos(hdg_err_raw)) / np.pi

        k_idx = min(idx_nearest + 5, n_path - 1)
        dx2 = path_x_global[min(k_idx + 1, n_path - 1)] - path_x_global[max(k_idx - 1, 0)]
        dy2 = path_y_global[min(k_idx + 1, n_path - 1)] - path_y_global[max(k_idx - 1, 0)]
        curvature = np.clip(dy2 / max(dx2, 0.01), -1, 1)

        dist_to_goal = np.sqrt((path_x_global[-1] - vehicle_x)**2 + (path_y_global[-1] - vehicle_y)**2)
        goal_dir_final = np.arctan2(path_y_global[-1] - vehicle_y, path_x_global[-1] - vehicle_x) - vehicle_theta

        obs = np.concatenate([scans / 10.0, [speed_actual / 10.0],
                              [np.sin(goal_dir), np.cos(goal_dir)],
                              [lat_err, hdg_err, curvature],
                              [dist_to_goal / 50.0, np.sin(goal_dir_final), np.cos(goal_dir_final)]])

    q, _, _ = _forward(obs, W1, b1, W2, b2, W3, b3)
    action = int(np.argmax(q))

    steer_options = np.radians([-20, -13, -7, 0, 7, 13, 20])
    throttle_options = [0.2, 0.5, 0.8]
    if n_path >= 1:
        dg = np.sqrt((path_x_global[-1] - vehicle_x)**2 + (path_y_global[-1] - vehicle_y)**2)
        if dg < 5.0:
            steer_options = np.radians([-10, -5, -2, 0, 2, 5, 10])
    steer_deg = np.degrees(steer_options[action // 3])
    throttle = throttle_options[action % 3]

    if rl_state is not None and "steer_ema" in rl_state:
        steer_ema = rl_state["steer_ema"]
        delta = np.clip(steer_deg - steer_ema, -10.0, 10.0)
        steer_deg = steer_ema + delta
        steer_deg = 0.5 * steer_deg + 0.5 * steer_ema
    if rl_state is not None:
        rl_state["steer_ema"] = steer_deg

    max_steer_rad = np.radians(max_steer_deg)
    steer_deg = np.clip(steer_deg, -max_steer_deg, max_steer_deg)

    brake = 0.0
    if behavior == Behavior.Value("BEHAVIOR_STOP"):
        throttle = 0.0
        brake = max(0.3, float(np.clip(speed_actual * 0.5, 0.0, 1.0)))

    control.steering.steering_angle = float(steer_deg)
    control.steering.steering_valid = True
    control.throttle_brake.throttle = float(throttle)
    control.throttle_brake.brake = float(brake)
    control.throttle_brake.throttle_valid = True
    control.throttle_brake.brake_valid = True
    control.gear = GearPosition.Value("GEAR_DRIVE")
    control.target_speed = float(decision_output.target_speed)
    control.lateral_error = 0.0
    control.heading_error = 0.0
    control.control_valid = True

    return control, rl_state


def main():
    sys.path.insert(0, _PROJECT_ROOT)
    # === Phase 5: DQN Training ===
    n_episodes = 800
    rng = np.random.default_rng(42)
    env = _PathTrackEnv(rng)

    n_obs_dim = 25
    n_act_dim = 21
    n_h1, n_h2 = 128, 64

    W1 = rng.normal(0, np.sqrt(2.0 / n_obs_dim), (n_obs_dim, n_h1))
    b1 = np.zeros(n_h1)
    W2 = rng.normal(0, np.sqrt(2.0 / n_h1), (n_h1, n_h2))
    b2 = np.zeros(n_h2)
    W3 = rng.normal(0, np.sqrt(2.0 / n_h2), (n_h2, n_act_dim))
    b3 = np.zeros(n_act_dim)

    W1_t, b1_t = W1.copy(), b1.copy()
    W2_t, b2_t = W2.copy(), b2.copy()
    W3_t, b3_t = W3.copy(), b3.copy()

    replay_max = 50000
    replay = deque(maxlen=replay_max)
    batch_size = 64
    gamma = 0.99
    lr_dqn = 0.0005
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    target_update = 500
    total_steps = 0
    reward_hist = []

    for ep in range(n_episodes):
        state = env.reset()
        ep_reward = 0.0
        while not env.done:
            if rng.random() < epsilon:
                action = rng.integers(0, n_act_dim)
            else:
                q, _, _ = _forward(state, W1, b1, W2, b2, W3, b3)
                action = int(np.argmax(q))
            next_state, reward, done = env.step(action)
            ep_reward += reward
            replay.append((state, action, reward, next_state, done))
            if len(replay) >= batch_size:
                idx = rng.choice(len(replay), batch_size, replace=False)
                s_b = np.array([replay[i][0] for i in idx])
                a_b = np.array([replay[i][1] for i in idx])
                r_b = np.array([replay[i][2] for i in idx])
                s2_b = np.array([replay[i][3] for i in idx])
                d_b = np.array([replay[i][4] for i in idx])
                q2, _, _ = _forward(s2_b, W1_t, b1_t, W2_t, b2_t, W3_t, b3_t)
                q_online_next, _, _ = _forward(s2_b, W1, b1, W2, b2, W3, b3)
                best_actions = np.argmax(q_online_next, axis=1)
                target = r_b + gamma * q2[np.arange(batch_size), best_actions] * (1.0 - d_b)
                q_pred, h1, h2 = _forward(s_b, W1, b1, W2, b2, W3, b3)
                td = q_pred.copy()
                td[np.arange(batch_size), a_b] = target
                d3 = (q_pred - td) / batch_size
                dW3 = h2.T @ d3
                db3 = d3.sum(axis=0)
                d2 = d3 @ W3.T * _relu_grad(h1 @ W2 + b2)
                dW2 = h1.T @ d2
                db2 = d2.sum(axis=0)
                d1 = d2 @ W2.T * _relu_grad(s_b @ W1 + b1)
                dW1 = s_b.T @ d1
                db1 = d1.sum(axis=0)
                W1 -= lr_dqn * np.clip(dW1, -5, 5)
                b1 -= lr_dqn * np.clip(db1, -5, 5)
                W2 -= lr_dqn * np.clip(dW2, -5, 5)
                b2 -= lr_dqn * np.clip(db2, -5, 5)
                W3 -= lr_dqn * np.clip(dW3, -5, 5)
                b3 -= lr_dqn * np.clip(db3, -5, 5)
            state = next_state
            total_steps += 1
            if total_steps % target_update == 0:
                W1_t, b1_t = W1.copy(), b1.copy()
                W2_t, b2_t = W2.copy(), b2.copy()
                W3_t, b3_t = W3.copy(), b3.copy()
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        reward_hist.append(ep_reward)
        if (ep + 1) % 50 == 0:
            print(f"[DQN] ep={ep + 1}/{n_episodes} reward={ep_reward:.1f} eps={epsilon:.3f}")

    # === Phase 6: Reward Curve ===
    plt.figure(figsize=(8, 4))
    plt.plot(reward_hist, "-b")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.title("DQN Training Reward")
    plt.tight_layout()
    plt.show()

    # === Phase 7: Path Tracking Demo ===
    cx, cy, cyaw, ck = generate_serpentine_course(ds=0.1)
    dt_rl = 0.1
    L = 2.7
    max_sim_time = 60.0
    target_speed = 20.0 / 3.6

    x, y, yaw, v = cx[0], cy[0], cyaw[0], 0.0
    x_hist, y_hist, v_hist, t_hist, lat_err_hist = [], [], [], [], []
    t = 0.0

    while t < max_sim_time:
        dx = cx - x
        dy = cy - y
        cos_v = np.cos(yaw)
        sin_v = np.sin(yaw)
        path_x = dx * cos_v + dy * sin_v
        path_y = -dx * sin_v + dy * cos_v

        idx_nearest = int(np.argmin(path_x**2 + path_y**2))
        e_lat = float(path_y[idx_nearest])

        angles = np.linspace(-np.pi / 2, np.pi / 2, 16)
        ray_dx = np.cos(yaw + angles)
        ray_dy = np.sin(yaw + angles)
        p_dx = cx - x
        p_dy = cy - y
        proj = np.outer(ray_dx, p_dx) + np.outer(ray_dy, p_dy)
        perp = np.abs(-np.outer(ray_dx, p_dy) + np.outer(ray_dy, p_dx))
        valid = (proj > 0) & (proj < 10.0) & (perp < 2.0)
        proj_masked = np.where(valid, proj, 10.0)
        scans = np.min(proj_masked, axis=1)

        lookahead_idx = min(idx_nearest + 20, len(cx) - 1)
        goal_dir = np.arctan2(cy[lookahead_idx] - y, cx[lookahead_idx] - x) - yaw

        lat_err_norm = e_lat / 5.0
        hdg_err = cyaw[idx_nearest] - yaw
        hdg_err = np.arctan2(np.sin(hdg_err), np.cos(hdg_err)) / np.pi
        k_idx = min(idx_nearest + 5, len(cx) - 1)
        dx2 = cx[min(k_idx + 1, len(cx) - 1)] - cx[max(k_idx - 1, 0)]
        dy2 = cy[min(k_idx + 1, len(cy) - 1)] - cy[max(k_idx - 1, 0)]
        curvature = np.clip(dy2 / max(dx2, 0.01), -1, 1)

        dist_to_goal = np.sqrt((cx[-1] - x)**2 + (cy[-1] - y)**2)
        goal_dir_final = np.arctan2(cy[-1] - y, cx[-1] - x) - yaw

        obs = np.concatenate([scans / 10.0, [v / 10.0],
                              [np.sin(goal_dir), np.cos(goal_dir)],
                              [lat_err_norm, hdg_err, curvature],
                              [dist_to_goal / 50.0, np.sin(goal_dir_final), np.cos(goal_dir_final)]])

        q, _, _ = _forward(obs, W1, b1, W2, b2, W3, b3)
        action = int(np.argmax(q))

        steer_options = np.radians([-20, -13, -7, 0, 7, 13, 20])
        throttle_options = [0.2, 0.5, 0.8]
        dg_demo = np.sqrt((cx[-1] - x)**2 + (cy[-1] - y)**2)
        if dg_demo < 5.0:
            steer_options = np.radians([-10, -5, -2, 0, 2, 5, 10])
        steer_rad = steer_options[action // 3]
        throttle = throttle_options[action % 3]

        a_cmd = throttle * 5.0 - 0.5
        x += v * np.cos(yaw) * dt_rl
        y += v * np.sin(yaw) * dt_rl
        yaw += v / L * np.tan(steer_rad) * dt_rl
        v += a_cmd * dt_rl
        v = max(v, 0.0)

        x_hist.append(x)
        y_hist.append(y)
        v_hist.append(v)
        t_hist.append(t)
        lat_err_hist.append(e_lat)
        t += dt_rl

        dist_to_end = np.sqrt((x - cx[-1])**2 + (y - cy[-1])**2)
        if dist_to_end < 2.0:
            break

        if show_animation:
            plt.cla()
            plt.plot(cx, cy, ".r", label="course")
            plt.plot(x_hist, y_hist, "-b", label="trajectory")
            plt.plot(cx[idx_nearest], cy[idx_nearest], "xg", label="target")
            plot_vehicle(x, y, yaw)
            plt.axis("equal")
            plt.grid(True)
            plt.title(f"DQN: Speed[km/h]:{v * 3.6:.1f}")
            plt.pause(0.001)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    axes[0].plot(cx, cy, ".r", label="course")
    axes[0].plot(x_hist, y_hist, "-b", label="trajectory")
    axes[0].legend()
    axes[0].set_xlabel("x[m]")
    axes[0].set_ylabel("y[m]")
    axes[0].axis("equal")
    axes[0].grid(True)
    axes[1].plot(t_hist, [iv * 3.6 for iv in v_hist], "-r")
    axes[1].set_xlabel("Time[s]")
    axes[1].set_ylabel("Speed[km/h]")
    axes[1].grid(True)
    axes[2].plot(t_hist, lat_err_hist, "-b")
    axes[2].set_xlabel("Time[s]")
    axes[2].set_ylabel("Lateral error[m]")
    axes[2].grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
