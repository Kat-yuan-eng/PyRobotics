import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

IMG_DIR = os.path.join(PROJECT_ROOT, 'docs', 'images')
os.makedirs(IMG_DIR, exist_ok=True)


def save_fig(name, fig_idx):
    fig_idx[0] += 1
    suffix = "" if fig_idx[0] == 1 else f"_{fig_idx[0]}"
    path = os.path.join(IMG_DIR, f"{name}{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  -> {os.path.relpath(path, PROJECT_ROOT)}")
    return path


def capture_module(rel_path, name):
    file_path = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(file_path):
        print(f"SKIP {name}: {rel_path} not found")
        return

    fig_idx = [0]
    saved = []

    def capture_show(*args, **kwargs):
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            p = save_fig(name, fig_idx)
            saved.append(p)
        plt.close('all')

    original_pause = plt.pause

    def fast_pause(interval):
        pass

    plt.pause = fast_pause
    plt.show = capture_show

    print(f"Capturing {name}...", end=" ", flush=True)
    t0 = time.time()

    try:
        with open(file_path, encoding='utf-8') as f:
            code = f.read()

        ns = {'__name__': '__main__', '__file__': file_path}
        exec(compile(code, file_path, 'exec'), ns)

        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            p = save_fig(name, fig_idx)
            saved.append(p)
        plt.close('all')

        elapsed = time.time() - t0
        print(f"OK ({len(saved)} fig, {elapsed:.1f}s)")
    except Exception as e:
        print(f"FAIL: {e}")
        plt.close('all')
    finally:
        plt.pause = original_pause


def capture_pathplanning():
    from PathPlanning.a_star_planner import a_star_plan
    from PathPlanning.rrt_planner import rrt_plan
    from PathPlanning.dwa_planner import dwa_plan

    ox = np.array([10, 20, 30, 10, 20, 30])
    oy = np.array([5, 5, 5, 15, 15, 15])
    obstacle_list = [(10, 5, 1.5), (20, 5, 1.5), (30, 5, 1.5),
                     (10, 15, 1.5), (20, 15, 1.5), (30, 15, 1.5)]

    print("Capturing a_star...", end=" ", flush=True)
    t0 = time.time()
    rx, ry, stats = a_star_plan(0, 0, 40, 20, ox, oy, resolution=1.0, robot_radius=0.8)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ox, oy, ".k", markersize=8)
    for (obx, oby, obr) in obstacle_list:
        theta = np.linspace(0, 2 * np.pi, 30)
        ax.plot(obx + obr * np.cos(theta), oby + obr * np.sin(theta), "-b")
    ax.plot(rx, ry, "-r", linewidth=2, label="A* Path")
    ax.plot(0, 0, "og", markersize=10, label="Start")
    ax.plot(40, 20, "xr", markersize=10, label="Goal")
    ax.set_title(f"A* Path Planning ({stats['path_length']:.1f}m)")
    ax.legend()
    ax.axis("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)
    plt.tight_layout()
    save_fig("a_star", [0])
    print(f"OK ({time.time() - t0:.1f}s)")

    print("Capturing rrt...", end=" ", flush=True)
    t0 = time.time()
    rx, ry, stats = rrt_plan(0, 0, 40, 20, obstacle_list,
                              rand_area=(-5, 45), expand_dis=3.0, max_iter=500)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ox, oy, ".k", markersize=8)
    for (obx, oby, obr) in obstacle_list:
        theta = np.linspace(0, 2 * np.pi, 30)
        ax.plot(obx + obr * np.cos(theta), oby + obr * np.sin(theta), "-b")
    ax.plot(rx, ry, "-r", linewidth=2, label="RRT Path")
    ax.plot(0, 0, "og", markersize=10, label="Start")
    ax.plot(40, 20, "xr", markersize=10, label="Goal")
    ax.set_title(f"RRT Path Planning ({stats.get('path_length', 0):.1f}m)")
    ax.legend()
    ax.axis("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)
    plt.tight_layout()
    save_fig("rrt", [0])
    print(f"OK ({time.time() - t0:.1f}s)")

    print("Capturing dwa...", end=" ", flush=True)
    t0 = time.time()
    x_state = np.array([0.0, 0.0, 0.0, 0.5, 0.0])
    goal = np.array([40.0, 20.0])
    ob = np.column_stack([ox, oy])
    traj_all = [x_state[:2].copy()]
    x = x_state.copy()
    for _ in range(200):
        u, traj, _ = dwa_plan(x, goal, ob, max_speed=2.0)
        x[0] += u[0] * math.cos(x[2]) * 0.1
        x[1] += u[0] * math.sin(x[2]) * 0.1
        x[2] += u[1] * 0.1
        x[3] = u[0]
        traj_all.append(x[:2].copy())
        if np.sqrt((x[0] - goal[0])**2 + (x[1] - goal[1])**2) < 2.0:
            break
    traj_arr = np.array(traj_all)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ox, oy, ".k", markersize=8)
    for (obx, oby, obr) in obstacle_list:
        theta = np.linspace(0, 2 * np.pi, 30)
        ax.plot(obx + obr * np.cos(theta), oby + obr * np.sin(theta), "-b")
    ax.plot(traj_arr[:, 0], traj_arr[:, 1], "-r", linewidth=2, label="DWA Path")
    ax.plot(0, 0, "og", markersize=10, label="Start")
    ax.plot(40, 20, "xr", markersize=10, label="Goal")
    ax.set_title("DWA Path Planning")
    ax.legend()
    ax.axis("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)
    plt.tight_layout()
    save_fig("dwa", [0])
    print(f"OK ({time.time() - t0:.1f}s)")


MODULES = [
    ("perception/lane_pixel_detector.py", "lane_detection"),
    ("perception/obstacle_detector.py", "obstacle_detection"),
    ("decision/obstacle_avoidance.py", "obstacle_avoidance"),
    ("decision/task_scheduler.py", "task_scheduler"),
    ("PathTracking/stanley_controller.py", "stanley"),
    ("PathTracking/pure_pursuit_controller.py", "pure_pursuit"),
    ("PathTracking/fuzzy_controller.py", "fuzzy"),
    ("PathTracking/mpc_controller.py", "mpc"),
    ("Localization/ekf_localizer.py", "ekf"),
    ("Localization/pf_localizer.py", "particle_filter"),
    ("Localization/fusion_localizer.py", "fusion"),
    ("SLAM/fast_slam.py", "fastslam"),
    ("SLAM/icp_matching.py", "icp"),
    ("SLAM/slam_pipeline.py", "slam_pipeline"),
]

for rel_path, name in MODULES:
    capture_module(rel_path, name)

capture_pathplanning()

print("\nDone. Images saved to docs/images/")
