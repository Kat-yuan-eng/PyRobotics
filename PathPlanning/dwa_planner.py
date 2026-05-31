import numpy as np
import time

# === Phase 1: Dynamic Window ===

def calc_dynamic_window(x_state, max_speed, min_speed, max_yaw_rate,
                        max_accel, max_delta_yaw_rate, dt, v_resolution):
    vs = [min_speed, max_speed, -max_yaw_rate, max_yaw_rate]
    vd = [x_state[3] - max_accel * dt, x_state[3] + max_accel * dt,
          x_state[4] - max_delta_yaw_rate * dt, x_state[4] + max_delta_yaw_rate * dt]
    dw = [max(vs[0], vd[0]), min(vs[1], vd[1]),
          max(vs[2], vd[2]), min(vs[3], vd[3])]
    if dw[1] <= dw[0]:
        dw[1] = dw[0] + v_resolution
    if dw[3] <= dw[2]:
        dw[3] = dw[2] + np.deg2rad(1.0)
    return dw

# === Phase 2: Trajectory Prediction ===

def predict_trajectory(x_init, v, yaw_rate, dt, predict_time):
    n_steps = max(int(np.ceil(predict_time / dt)), 1)
    traj = np.zeros((n_steps + 1, 5))
    traj[0] = x_init.copy()
    thetas = x_init[2] + yaw_rate * dt * np.arange(1, n_steps + 1)
    traj[1:, 0] = x_init[0] + v * np.cumsum(np.cos(np.concatenate([[x_init[2]], thetas[:-1]]))) * dt
    traj[1:, 1] = x_init[1] + v * np.cumsum(np.sin(np.concatenate([[x_init[2]], thetas[:-1]]))) * dt
    traj[1:, 2] = thetas
    traj[1:, 3] = v
    traj[1:, 4] = yaw_rate
    return traj

# === Phase 3: Cost Functions ===

def calc_to_goal_cost(trajectory, goal, dist_to_goal=None):
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    dist = np.hypot(dx, dy)
    angle_to_goal = np.arctan2(dy, dx)
    angle_diff = trajectory[-1, 2] - angle_to_goal
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
    angle_weight = 0.5
    if dist_to_goal is not None and dist_to_goal < 10.0:
        angle_weight = 0.5 + 2.0 * max(0.0, 1.0 - dist_to_goal / 10.0)
    return dist + abs(angle_diff) * angle_weight

def calc_path_cost(trajectory, global_path):
    if len(global_path) == 0:
        return 0.0
    gp = np.asarray(global_path)
    sampled = trajectory[::3]
    dists = np.sqrt((sampled[:, 0, None] - gp[None, :, 0])**2 +
                    (sampled[:, 1, None] - gp[None, :, 1])**2)
    return dists.min(axis=1).mean()

def calc_obstacle_cost(trajectory, ob, robot_radius):
    if len(ob) == 0:
        return 0.0, np.inf
    safe_radius = robot_radius * 1.2
    dists = np.sqrt((trajectory[:, 0, None] - ob[None, :, 0])**2 +
                    (trajectory[:, 1, None] - ob[None, :, 1])**2)
    min_dists = dists.min(axis=1)
    min_dist_overall = min_dists.min()
    if min_dist_overall < safe_radius:
        return np.inf, min_dist_overall
    return 1.0 / (min_dist_overall - robot_radius + 1e-9), min_dist_overall

# === Phase 4: DWA Plan ===

def dwa_plan(x_state, goal, ob, max_speed=1.0, min_speed=-0.5,
             max_yaw_rate=40.0, max_accel=0.5, max_delta_yaw_rate=100.0,
             v_resolution=0.01, yaw_rate_resolution=1.0,
             predict_time=3.0, dt=0.1, robot_radius=1.0,
             to_goal_cost_gain=5.0, speed_cost_gain=1.0, obstacle_cost_gain=10.0,
             path_cost_gain=1.0, global_path=None, goal_threshold=None):
    t0 = time.perf_counter()
    max_yaw_rate_rad = np.deg2rad(max_yaw_rate)
    max_delta_yaw_rate_rad = np.deg2rad(max_delta_yaw_rate)
    yaw_rate_res_rad = np.deg2rad(yaw_rate_resolution)

    if goal_threshold is None:
        goal_threshold = max(robot_radius * 2.0, max_speed * dt * 5.0)

    dist_to_goal = np.hypot(x_state[0] - goal[0], x_state[1] - goal[1])

    if dist_to_goal < goal_threshold:
        stats = {"planner": "DWA", "path_length": 0.0, "planning_time_ms": 0.0,
                 "nodes_explored": 0, "path_points": 1}
        return np.array([0.0, 0.0]), x_state.reshape(1, -1), stats

    dw = calc_dynamic_window(x_state, max_speed, min_speed, max_yaw_rate_rad,
                             max_accel, max_delta_yaw_rate_rad, dt, v_resolution)

    best_u = np.array([0.0, 0.0])
    best_traj = np.array([x_state])
    min_cost = np.inf
    best_u_emergency = np.array([0.0, 0.0])
    best_traj_emergency = np.array([x_state])
    max_min_dist = -1.0

    predict_dist = max_speed * predict_time
    approach_ratio = min(dist_to_goal / max(predict_dist, 1e-9), 1.0)
    adaptive_goal_gain = to_goal_cost_gain * (1.0 + 2.0 * (1.0 - approach_ratio))
    adaptive_speed_gain = speed_cost_gain * approach_ratio

    for v in np.arange(dw[0], dw[1], v_resolution):
        for yw in np.arange(dw[2], dw[3], yaw_rate_res_rad):
            traj = predict_trajectory(x_state, v, yw, dt, predict_time)

            goal_cost = adaptive_goal_gain * calc_to_goal_cost(traj, goal, dist_to_goal)
            speed_cost = adaptive_speed_gain * abs(max_speed - abs(traj[-1, 3]))
            ob_cost, min_dist = calc_obstacle_cost(traj, ob, robot_radius)
            ob_cost = obstacle_cost_gain * ob_cost
            p_cost = path_cost_gain * calc_path_cost(traj, global_path) if global_path is not None else 0.0

            cost = goal_cost + speed_cost + ob_cost + p_cost

            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_u_emergency = np.array([v, yw])
                best_traj_emergency = traj

            if cost < min_cost:
                min_cost = cost
                best_u = np.array([v, yw])
                best_traj = traj

    if min_cost == np.inf:
        best_u = best_u_emergency
        best_traj = best_traj_emergency

    elapsed_ms = (time.perf_counter() - t0) * 1000
    path_length = np.sum(np.hypot(np.diff(best_traj[:, 0]), np.diff(best_traj[:, 1]))) if len(best_traj) > 1 else 0.0
    n_v = max(int((dw[1] - dw[0]) / v_resolution), 1)
    n_yw = max(int((dw[3] - dw[2]) / yaw_rate_res_rad), 1)
    stats = {"planner": "DWA", "path_length": path_length, "planning_time_ms": elapsed_ms,
             "nodes_explored": n_v * n_yw, "path_points": len(best_traj)}

    return best_u, best_traj, stats
