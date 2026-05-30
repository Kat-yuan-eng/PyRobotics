import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import time
from PathPlanning.a_star_planner import a_star_plan
from PathPlanning.rrt_planner import rrt_plan
from PathPlanning.dwa_planner import dwa_plan

# === Phase 1: Benchmark Scenario ===

def create_benchmark_scenarios():
    scenarios = []

    ox1 = np.array([10, 20, 30, 40, 10, 20, 30, 40])
    oy1 = np.array([5, 5, 5, 5, 15, 15, 15, 15])
    scenarios.append({"name": "sparse_obstacles", "sx": 0, "sy": 0, "gx": 50, "gy": 50,
                       "ox": ox1, "oy": oy1, "resolution": 2.0, "robot_radius": 1.0,
                       "obstacle_list": [(10, 5, 1), (20, 5, 1), (30, 5, 1), (40, 5, 1),
                                         (10, 15, 1), (20, 15, 1), (30, 15, 1), (40, 15, 1)]})

    ox2 = np.concatenate([np.linspace(10, 10, 20), np.linspace(10, 10, 20)])
    oy2 = np.concatenate([np.linspace(0, 8, 20), np.linspace(12, 20, 20)])
    scenarios.append({"name": "narrow_passage", "sx": 0, "sy": 10, "gx": 30, "gy": 10,
                       "ox": ox2, "oy": oy2, "resolution": 0.5, "robot_radius": 0.3,
                       "obstacle_list": [(10, y, 0.5) for y in list(np.linspace(0, 8, 5)) + list(np.linspace(12, 20, 5))]})

    ox3 = np.array([5, 5, 5, 5, 5, 15, 15, 15, 15, 15, 25, 25, 25, 25, 25])
    oy3 = np.array([0, 5, 10, 15, 20, 0, 5, 10, 15, 20, 0, 5, 10, 15, 20])
    scenarios.append({"name": "maze", "sx": 0, "sy": 0, "gx": 30, "gy": 20,
                       "ox": ox3, "oy": oy3, "resolution": 1.0, "robot_radius": 0.5,
                       "obstacle_list": [(5, 0, 1), (5, 5, 1), (5, 10, 1), (5, 15, 1), (5, 20, 1),
                                         (15, 0, 1), (15, 5, 1), (15, 10, 1), (15, 15, 1), (15, 20, 1),
                                         (25, 0, 1), (25, 5, 1), (25, 10, 1), (25, 15, 1), (25, 20, 1)]})

    return scenarios

# === Phase 2: Compare Planners ===

def compare_planners(scenarios=None):
    if scenarios is None:
        scenarios = create_benchmark_scenarios()

    results = []
    for sc in scenarios:
        print(f"\n--- Scenario: {sc['name']} ---")

        rx, ry, stats_a = a_star_plan(sc['sx'], sc['sy'], sc['gx'], sc['gy'],
                                       sc['ox'], sc['oy'],
                                       resolution=sc.get('resolution', 2.0),
                                       robot_radius=sc.get('robot_radius', 1.0))
        stats_a["scenario"] = sc['name']
        stats_a["success"] = len(rx) > 0
        results.append(stats_a)
        print(f"  A*: len={stats_a['path_length']:.1f}m  time={stats_a['planning_time_ms']:.1f}ms  nodes={stats_a['nodes_explored']}")

        rx, ry, stats_r = rrt_plan(sc['sx'], sc['sy'], sc['gx'], sc['gy'],
                                    sc['obstacle_list'],
                                    rand_area=(min(sc['sx'], sc['gx']) - 5, max(sc['sx'], sc['gx']) + 5))
        stats_r["scenario"] = sc['name']
        stats_r["success"] = len(rx) > 0
        results.append(stats_r)
        print(f"  RRT: len={stats_r.get('path_length', 0):.1f}m  time={stats_r['planning_time_ms']:.1f}ms  nodes={stats_r['nodes_explored']}")

        x_state = np.array([sc['sx'], sc['sy'], 0.0, 0.0, 0.0])
        goal = np.array([sc['gx'], sc['gy']])
        ob = np.column_stack([sc['ox'], sc['oy']])
        u, traj, stats_d = dwa_plan(x_state, goal, ob)
        stats_d["scenario"] = sc['name']
        stats_d["success"] = len(traj) > 0
        results.append(stats_d)
        print(f"  DWA: len={stats_d['path_length']:.1f}m  time={stats_d['planning_time_ms']:.1f}ms")

    return results

# === Phase 3: Select Planner ===

def select_planner(scenario_type):
    selection = {
        "static_sparse": "A*",
        "dynamic": "DWA",
        "narrow_passage": "A*",
        "large_map": "RRT",
        "realtime": "DWA",
    }
    return selection.get(scenario_type, "A*")

if __name__ == "__main__":
    compare_planners()
