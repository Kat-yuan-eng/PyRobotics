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

class _GridWorld:
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng(42)
        self.reset()

    def reset(self):
        self.x = 0.5
        self.y = 5.0
        self.theta = 0.0
        self.v = 0.0
        self.n_obs = self.rng.integers(3, 6)
        self.obs_x = self.rng.uniform(2.0, 8.0, self.n_obs)
        self.obs_y = self.rng.uniform(1.0, 9.0, self.n_obs)
        self.obs_w = self.rng.uniform(0.3, 0.8, self.n_obs)
        self.obs_h = self.rng.uniform(0.3, 0.8, self.n_obs)
        self.done = False
        self.steps = 0
        return self._obs()

    def _obs(self):
        angles = np.linspace(-np.pi / 2, np.pi / 2, 16)
        scans = np.full(16, 5.0)
        for i, a in enumerate(angles):
            for d in np.arange(0.1, 5.0, 0.1):
                px = self.x + d * np.cos(self.theta + a)
                py = self.y + d * np.sin(self.theta + a)
                if px < 0 or px > 10 or py < 0 or py > 10:
                    scans[i] = d
                    break
                hit = False
                for j in range(self.n_obs):
                    if (abs(px - self.obs_x[j]) < self.obs_w[j] / 2 and
                            abs(py - self.obs_y[j]) < self.obs_h[j] / 2):
                        scans[i] = d
                        hit = True
                        break
                if hit:
                    break

        goal_dir = np.arctan2(5.0 - self.y, 9.0 - self.x) - self.theta
        return np.concatenate([scans / 5.0, [self.v / 5.0], [np.sin(goal_dir), np.cos(goal_dir)]])

    def step(self, action):
        steer_options = np.radians([-15, 0, 15])
        throttle_options = [0.3, 0.7]
        steer = steer_options[action // 2]
        thr = throttle_options[action % 2]

        dt = 0.1
        L = 2.7
        accel = thr * 3.0 - 0.5
        self.v = max(self.v + accel * dt, 0.0)
        self.theta += self.v * np.tan(steer) / L * dt
        x_old = self.x
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.steps += 1

        reward = (self.x - x_old) * 1.0
        reward -= 0.1 * abs(np.degrees(steer)) / 15.0
        reward -= 0.1

        if self.x >= 9.0:
            reward += 50.0
            self.done = True
        elif self._collision():
            reward -= 100.0
            self.done = True
        elif self.x < 0 or self.y < 0 or self.y > 10:
            reward -= 100.0
            self.done = True
        elif self.steps >= 500:
            self.done = True

        return self._obs(), reward, self.done

    def _collision(self):
        for j in range(self.n_obs):
            if (abs(self.x - self.obs_x[j]) < self.obs_w[j] / 2 + 0.2 and
                    abs(self.y - self.obs_y[j]) < self.obs_h[j] / 2 + 0.2):
                return True
        return False


# === Phase 3: DQN Training ===

def train_dqn(n_episodes=500, save_path=None):
    if save_path is None:
        save_path = os.path.join(_PROJECT_ROOT, "control", "dqn_weights.npz")

    rng = np.random.default_rng(42)
    env = _GridWorld(rng)

    n_obs = 18
    n_act = 6
    n_hidden1 = 64
    n_hidden2 = 32

    W1 = rng.normal(0, 0.1, (n_obs, n_hidden1))
    b1 = np.zeros(n_hidden1)
    W2 = rng.normal(0, 0.1, (n_hidden1, n_hidden2))
    b2 = np.zeros(n_hidden2)
    W3 = rng.normal(0, 0.1, (n_hidden2, n_act))
    b3 = np.zeros(n_act)

    W1_t, b1_t = W1.copy(), b1.copy()
    W2_t, b2_t = W2.copy(), b2.copy()
    W3_t, b3_t = W3.copy(), b3.copy()

    replay_max = 10000
    replay = deque(maxlen=replay_max)
    batch_size = 32
    gamma = 0.99
    lr = 0.001
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = (1.0 - epsilon_min) / max(n_episodes, 1)
    target_update = 100
    total_steps = 0

    for ep in range(n_episodes):
        state = env.reset()
        ep_reward = 0.0

        while not env.done:
            if rng.random() < epsilon:
                action = rng.integers(0, n_act)
            else:
                q, _, _ = _forward(state, W1, b1, W2, b2, W3, b3)
                action = int(np.clip(np.argmax(q), 0, 5))

            next_state, reward, done = env.step(action)
            ep_reward += reward

            if len(replay) >= replay_max:
                replay.popleft()
            replay.append((state, action, reward, next_state, done))

            if len(replay) >= batch_size:
                idx = rng.choice(len(replay), batch_size, replace=False)
                s_batch = np.array([replay[i][0] for i in idx])
                a_batch = np.array([replay[i][1] for i in idx])
                r_batch = np.array([replay[i][2] for i in idx])
                s2_batch = np.array([replay[i][3] for i in idx])
                d_batch = np.array([replay[i][4] for i in idx])

                q2, _, _ = _forward(s2_batch, W1_t, b1_t, W2_t, b2_t, W3_t, b3_t)
                target = r_batch + gamma * np.max(q2, axis=1) * (1.0 - d_batch)

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

                W1 -= lr * np.clip(dW1, -1, 1)
                b1 -= lr * np.clip(db1, -1, 1)
                W2 -= lr * np.clip(dW2, -1, 1)
                b2 -= lr * np.clip(db2, -1, 1)
                W3 -= lr * np.clip(dW3, -1, 1)
                b3 -= lr * np.clip(db3, -1, 1)

            state = next_state
            total_steps += 1

            if total_steps % target_update == 0:
                W1_t, b1_t = W1.copy(), b1.copy()
                W2_t, b2_t = W2.copy(), b2.copy()
                W3_t, b3_t = W3.copy(), b3.copy()

        epsilon = max(epsilon - epsilon_decay, epsilon_min)

        if (ep + 1) % 50 == 0:
            print(f"[DQN] ep={ep + 1}/{n_episodes}  reward={ep_reward:.1f}  eps={epsilon:.3f}")

    np.savez(save_path, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
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
        obs = np.zeros(18)
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
        scans = np.full(16, 5.0)
        for i, a in enumerate(angles):
            ray_dx = np.cos(a)
            ray_dy = np.sin(a)
            proj = path_x_veh * ray_dx + path_y_veh * ray_dy
            perp = np.abs(-path_x_veh * ray_dy + path_y_veh * ray_dx)
            valid = (proj > 0) & (proj < 5.0) & (perp < 1.0)
            if valid.any():
                scans[i] = float(np.min(proj[valid]))

        lookahead_idx = min(15, n_path - 1)
        goal_dir = np.arctan2(path_y_veh[lookahead_idx], path_x_veh[lookahead_idx])
        obs = np.concatenate([scans / 5.0, [speed_actual / 5.0], [np.sin(goal_dir), np.cos(goal_dir)]])

    q, _, _ = _forward(obs, W1, b1, W2, b2, W3, b3)
    action = int(np.argmax(q))

    steer_options = np.radians([-15, 0, 15])
    throttle_options = [0.3, 0.7]
    steer_deg = np.degrees(steer_options[action // 2])
    throttle = throttle_options[action % 2]

    if rl_state is not None and "steer_ema" in rl_state:
        steer_ema = rl_state["steer_ema"]
        steer_deg = 0.3 * steer_deg + 0.7 * steer_ema
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
    n_episodes = 200
    rng = np.random.default_rng(42)
    env = _GridWorld(rng)

    n_obs_dim = 18
    n_act_dim = 6
    n_h1, n_h2 = 64, 32

    W1 = rng.normal(0, 0.1, (n_obs_dim, n_h1))
    b1 = np.zeros(n_h1)
    W2 = rng.normal(0, 0.1, (n_h1, n_h2))
    b2 = np.zeros(n_h2)
    W3 = rng.normal(0, 0.1, (n_h2, n_act_dim))
    b3 = np.zeros(n_act_dim)

    W1_t, b1_t = W1.copy(), b1.copy()
    W2_t, b2_t = W2.copy(), b2.copy()
    W3_t, b3_t = W3.copy(), b3.copy()

    replay_max = 10000
    replay = deque(maxlen=replay_max)
    batch_size = 32
    gamma = 0.99
    lr_dqn = 0.001
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = (1.0 - epsilon_min) / max(n_episodes, 1)
    target_update = 100
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
                target = r_b + gamma * np.max(q2, axis=1) * (1.0 - d_b)
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
                W1 -= lr_dqn * np.clip(dW1, -1, 1)
                b1 -= lr_dqn * np.clip(db1, -1, 1)
                W2 -= lr_dqn * np.clip(dW2, -1, 1)
                b2 -= lr_dqn * np.clip(db2, -1, 1)
                W3 -= lr_dqn * np.clip(dW3, -1, 1)
                b3 -= lr_dqn * np.clip(db3, -1, 1)
            state = next_state
            total_steps += 1
            if total_steps % target_update == 0:
                W1_t, b1_t = W1.copy(), b1.copy()
                W2_t, b2_t = W2.copy(), b2.copy()
                W3_t, b3_t = W3.copy(), b3.copy()
        epsilon = max(epsilon - epsilon_decay, epsilon_min)
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
        valid = (proj > 0) & (proj < 5.0) & (perp < 1.0)
        proj_masked = np.where(valid, proj, 5.0)
        scans = np.min(proj_masked, axis=1)

        lookahead_idx = min(idx_nearest + 20, len(cx) - 1)
        goal_dir = np.arctan2(cy[lookahead_idx] - y, cx[lookahead_idx] - x) - yaw
        obs = np.concatenate([scans / 5.0, [v / 5.0], [np.sin(goal_dir), np.cos(goal_dir)]])

        q, _, _ = _forward(obs, W1, b1, W2, b2, W3, b3)
        action = int(np.argmax(q))

        steer_options = np.radians([-15, 0, 15])
        throttle_options = [0.3, 0.7]
        steer_rad = steer_options[action // 2]
        throttle = throttle_options[action % 2]

        a_cmd = throttle * 3.0 - 0.5
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
