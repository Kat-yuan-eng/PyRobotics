import numpy as np
import time
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from decision.path_smoother import smooth_path

# === Phase 1: Collision Check ===

def _check_collision(x, y, obstacle_list, robot_radius):
    for ox, oy, size in obstacle_list:
        if np.hypot(x - ox, y - oy) <= size + robot_radius:
            return True
    return False

def _check_path_collision(x0, y0, x1, y1, obstacle_list, robot_radius, path_resolution=0.5):
    dist = np.hypot(x1 - x0, y1 - y0)
    n_steps = max(int(np.ceil(dist / path_resolution)), 1)
    ts = np.linspace(0, 1, n_steps + 1)
    px = x0 + ts * (x1 - x0)
    py = y0 + ts * (y1 - y0)
    for ox, oy, size in obstacle_list:
        dists = np.hypot(px - ox, py - oy)
        if dists.min() <= size + robot_radius:
            return True
    return False

# === Phase 2: Path Extraction ===

def _extract_path(nodes_x, nodes_y, parents):
    path_x, path_y = [nodes_x[-1]], [nodes_y[-1]]
    idx = len(nodes_x) - 1
    while parents[idx] >= 0:
        idx = parents[idx]
        path_x.append(nodes_x[idx])
        path_y.append(nodes_y[idx])
    path_x.reverse()
    path_y.reverse()
    return path_x, path_y

# === Phase 3: RRT Planning ===

def rrt_plan(sx, sy, gx, gy, obstacle_list, rand_area,
             expand_dis=3.0, path_resolution=0.5, goal_sample_rate=5,
             max_iter=500, robot_radius=0.8):
    t0 = time.perf_counter()
    rng = np.random.default_rng()

    nodes_x = [sx]
    nodes_y = [sy]
    parents = [-1]

    min_rand, max_rand = rand_area

    for iteration in range(max_iter):
        if rng.integers(0, 100) > goal_sample_rate:
            rnd_x = rng.uniform(min_rand, max_rand)
            rnd_y = rng.uniform(min_rand, max_rand)
        else:
            rnd_x, rnd_y = gx, gy

        dists = np.hypot(np.array(nodes_x) - rnd_x, np.array(nodes_y) - rnd_y)
        nearest_idx = int(np.argmin(dists))

        angle = np.arctan2(rnd_y - nodes_y[nearest_idx], rnd_x - nodes_x[nearest_idx])
        new_x = nodes_x[nearest_idx] + expand_dis * np.cos(angle)
        new_y = nodes_y[nearest_idx] + expand_dis * np.sin(angle)

        if _check_collision(new_x, new_y, obstacle_list, robot_radius):
            continue

        if _check_path_collision(nodes_x[nearest_idx], nodes_y[nearest_idx],
                                 new_x, new_y, obstacle_list, robot_radius, path_resolution):
            continue

        nodes_x.append(new_x)
        nodes_y.append(new_y)
        parents.append(nearest_idx)

        dist_to_goal = np.hypot(new_x - gx, new_y - gy)
        if dist_to_goal <= expand_dis:
            if not _check_path_collision(new_x, new_y, gx, gy,
                                         obstacle_list, robot_radius, path_resolution):
                nodes_x.append(gx)
                nodes_y.append(gy)
                parents.append(len(nodes_x) - 2)

                path_x, path_y = _extract_path(nodes_x, nodes_y, parents)
                path_x_arr, path_y_arr = np.array(path_x), np.array(path_y)
                if len(path_x_arr) >= 3:
                    try:
                        sx, sy, kappa = smooth_path(path_x_arr, path_y_arr, max_deviation=0.3, n_output=len(path_x_arr))
                        path_x_arr, path_y_arr = sx, sy
                    except Exception:
                        kappa = np.zeros(len(path_x_arr))
                else:
                    kappa = np.zeros(len(path_x_arr))
                path_length = np.sum(np.hypot(np.diff(path_x_arr), np.diff(path_y_arr))) if len(path_x_arr) > 1 else 0.0
                elapsed_ms = (time.perf_counter() - t0) * 1000
                stats = {"planner": "RRT", "path_length": path_length, "planning_time_ms": elapsed_ms,
                         "nodes_explored": len(nodes_x), "path_points": len(path_x_arr), "curvature": kappa}
                return path_x_arr, path_y_arr, stats

    elapsed_ms = (time.perf_counter() - t0) * 1000
    stats = {"planner": "RRT", "path_length": 0.0, "planning_time_ms": elapsed_ms,
             "nodes_explored": len(nodes_x), "path_points": 0}
    return np.array([]), np.array([]), stats
