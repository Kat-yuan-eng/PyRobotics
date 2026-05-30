import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import math

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "generated"))
sys.path.insert(0, _PROJECT_ROOT)

show_animation = True

FIGS_DIR = os.path.join(_PROJECT_ROOT, "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.grid'] = True

COLORS = {
    "blue": "#3A86FF",
    "orange": "#FF9E00",
    "green": "#38B000",
    "purple": "#9D4EDD",
    "gray": "#6C757D",
    "red": "#E63946",
}


# === Phase 0: Drawing Utilities ===

def plot_vehicle(ax, x, y, yaw, steer_deg=0.0, color="-k", vehicle_len=2.7, vehicle_wid=1.8):
    c, s = math.cos(yaw), math.sin(yaw)
    Rot = np.array([[c, s], [-s, c]])
    hl, hw = vehicle_len / 2, vehicle_wid / 2
    outline = np.array([[-hl, hl, hl, -hl, -hl],
                        [hw, hw, -hw, -hw, hw]])
    outline = Rot @ outline
    outline[0] += x
    outline[1] += y
    ax.plot(outline[0], outline[1], color, linewidth=1.5)

    wl, ww = 0.4, 0.2
    wheel_offsets = [(hl * 0.7, hw - 0.15), (hl * 0.7, -hw + 0.15),
                     (-hl * 0.7, hw - 0.15), (-hl * 0.7, -hw + 0.15)]
    steer_rad = math.radians(steer_deg)
    for idx, (dx, dy) in enumerate(wheel_offsets):
        is_front = idx < 2
        w_yaw = yaw + (steer_rad if is_front else 0.0)
        wc, ws = math.cos(w_yaw), math.sin(w_yaw)
        WRot = np.array([[wc, ws], [-ws, wc]])
        wheel = np.array([[-wl / 2, wl / 2, wl / 2, -wl / 2, -wl / 2],
                          [-ww / 2, -ww / 2, ww / 2, ww / 2, -ww / 2]])
        wheel = WRot @ wheel
        wheel[0] += x + dx * c - dy * s
        wheel[1] += y + dx * s + dy * c
        ax.plot(wheel[0], wheel[1], color, linewidth=1.0)

    arrow_len = vehicle_len * 0.6
    ax.arrow(x, y, arrow_len * c, arrow_len * s,
             head_width=0.3, fc="r", ec="r", alpha=0.5)


def plot_obstacle_circle(ax, ox, oy, radius, color="-b"):
    deg = np.arange(0, 360 + 5, 5)
    xl = ox + radius * np.cos(np.deg2rad(deg))
    yl = oy + radius * np.sin(np.deg2rad(deg))
    ax.plot(xl, yl, color)


def plot_cov_ellipse(ax, x, y, P, chi2=3.0, color="-r"):
    eig_val, eig_vec = np.linalg.eig(P[:2, :2])
    eig_val = np.real(eig_val)
    eig_val = np.maximum(eig_val, 1e-12)
    big_ind = np.argmax(eig_val)
    small_ind = 1 - big_ind
    a = math.sqrt(chi2 * eig_val[big_ind])
    b = math.sqrt(chi2 * eig_val[small_ind])
    angle = math.atan2(eig_vec[1, big_ind], eig_vec[0, big_ind])
    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    px = a * np.cos(t)
    py = b * np.sin(t)
    R2 = np.array([[math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)]])
    fx = R2 @ np.array([px, py])
    ax.plot(fx[0] + x, fx[1] + y, color, linewidth=0.8)


# === Phase 1: Complex Path Generators ===

def generate_figure8_path(n_pts=500, R=15.0):
    t = np.linspace(0, 2 * np.pi, n_pts)
    x = R * np.sin(t)
    y = R * np.sin(2 * t) / 2
    dx = np.gradient(x)
    dy = np.gradient(y)
    yaw = np.arctan2(dy, dx)
    return x, y, yaw


def generate_slalom_path(n_pts=300, spacing=8.0, amplitude=3.0, n_poles=6):
    x = np.linspace(0, spacing * n_poles, n_pts)
    y = amplitude * np.sin(2 * np.pi * x / (2 * spacing))
    dx = np.gradient(x)
    dy = np.gradient(y)
    yaw = np.arctan2(dy, dx)
    pole_x = np.arange(1, n_poles + 1) * spacing
    pole_y = amplitude * np.sin(2 * np.pi * pole_x / (2 * spacing))
    return x, y, yaw, pole_x, pole_y


def generate_city_path(n_pts=400):
    segs = []
    segs.append((np.linspace(0, 20, 50), np.zeros(50)))
    r_turn = 5.0
    angles = np.linspace(0, np.pi / 2, 30)
    x0, y0 = 20.0, 0.0
    segs.append((x0 + r_turn * np.sin(angles), y0 + r_turn * (1 - np.cos(angles))))
    x1, y1 = segs[-1][0][-1], segs[-1][1][-1]
    segs.append((np.full(50, x1), np.linspace(y1, y1 + 15, 50)))
    x2, y2 = segs[-1][0][-1], segs[-1][1][-1]
    angles2 = np.linspace(np.pi / 2, np.pi, 30)
    segs.append((x2 + r_turn * np.cos(angles2), y2 - r_turn * np.sin(angles2) + r_turn))
    x3, y3 = segs[-1][0][-1], segs[-1][1][-1]
    segs.append((np.linspace(x3, x3 - 20, 50), np.full(50, y3)))

    x_all = np.concatenate([s[0] for s in segs])
    y_all = np.concatenate([s[1] for s in segs])
    dx = np.gradient(x_all)
    dy = np.gradient(y_all)
    yaw = np.arctan2(dy, dx)
    obstacles = [(10, 1.5, 0.8), (25, 5, 0.6), (x1 - 2, y1 + 8, 0.7)]
    return x_all, y_all, yaw, obstacles


# === Phase 2: Scene 1 — Camera Lane Detection ===

def scene1_camera_lane():
    import cv2
    from perception.lane_pixel_detector import detect_lane_pixels, _lane_color_mask
    from perception.sensor_fusion import _pixel_to_vehicle, default_camera_params

    print("\n[Scene 1] Camera Lane Detection (press 'q' to exit)")
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Cannot open camera"

    K, R_cam, t_cam = default_camera_params()
    K_inv = np.linalg.inv(K)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Scene 1: Camera Lane Detection", fontsize=14, fontweight='bold')

    frame_count = 0
    while show_animation:
        ret, frame = cap.read()
        if not ret:
            break

        scan_rows, center_x = detect_lane_pixels(frame)
        mask = _lane_color_mask(frame)

        for ax in axes.flat:
            ax.cla()

        axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("(a) Camera Frame")

        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title("(b) HLS Color Mask")

        overlay = frame.copy()
        valid = ~np.isnan(center_x)
        if valid.sum() > 0:
            for row, cx in zip(scan_rows[valid], center_x[valid]):
                cv2.circle(overlay, (int(cx), int(row)), 3, (0, 255, 0), -1)
            axes[1, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        else:
            axes[1, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("(c) Detected Center Line")

        veh_pts = []
        for row, cx in zip(scan_rows[valid], center_x[valid]):
            pt = _pixel_to_vehicle(cx, row, K_inv, R_cam, t_cam)
            if pt is not None:
                veh_pts.append(pt)
        if len(veh_pts) >= 2:
            vp = np.array(veh_pts)
            axes[1, 1].plot(vp[:, 0], vp[:, 1], "-g", linewidth=2, label="Center Line")
            axes[1, 1].plot(vp[:, 0], vp[:, 1] - 1.75, "--b", linewidth=1, label="Left Bound")
            axes[1, 1].plot(vp[:, 0], vp[:, 1] + 1.75, "--b", linewidth=1, label="Right Bound")
            axes[1, 1].legend(fontsize=8)
        axes[1, 1].set_xlabel("x [m]")
        axes[1, 1].set_ylabel("y [m]")
        axes[1, 1].set_title("(d) Bird's Eye View")
        axes[1, 1].axis("equal")

        plt.tight_layout()
        plt.pause(0.03)

        frame_count += 1
        if frame_count >= 100:
            break

    cap.release()
    save_path = os.path.join(FIGS_DIR, "visual_perception_lane.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Scene 1] Saved: {save_path}")
    plt.close(fig)


# === Phase 3: Scene 2 — Obstacle + Sign Detection ===

def scene2_obstacle_sign():
    from perception.obstacle_detector import detect_obstacles, generate_test_point_cloud
    from perception.sign_recognizer import recognize_signs, generate_test_sign_image

    print("\n[Scene 2] Obstacle + Sign Detection")

    pc = generate_test_point_cloud()
    obstacles = detect_obstacles(pc)
    sign_img = generate_test_sign_image()
    signs = recognize_signs(sign_img)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Scene 2: Obstacle + Sign Detection", fontsize=14, fontweight='bold')

    ax1.scatter(pc[:, 0], pc[:, 1], c=pc[:, 2], cmap='viridis', s=1, alpha=0.5)
    for obs in obstacles:
        cx, cy = obs["center"]
        l, w = obs["length"], obs["width"]
        heading = obs["heading"]
        corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2], [-l/2, -w/2]])
        R = np.array([[math.cos(heading), -math.sin(heading)],
                       [math.sin(heading), math.cos(heading)]])
        corners = (R @ corners.T).T + np.array([cx, cy])
        ax1.plot(corners[:, 0], corners[:, 1], "-r", linewidth=1.5)
        ax1.plot(cx, cy, "xr", markersize=8)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("(a) Point Cloud + Obstacle OBB")
    ax1.axis("equal")

    import cv2
    overlay = sign_img.copy()
    for sign in signs:
        x, y, w, h = sign["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(overlay, sign["category"], (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    ax2.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax2.set_title("(b) Sign Recognition")

    plt.tight_layout()
    save_path = os.path.join(FIGS_DIR, "visual_perception_obstacle_sign.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Scene 2] Saved: {save_path}")
    plt.show()
    plt.close(fig)


# === Phase 4: Scene 3 — Planner Comparison ===

def scene3_planners():
    from PathPlanning.a_star_planner import a_star_plan
    from PathPlanning.rrt_planner import rrt_plan
    from PathPlanning.dwa_planner import dwa_plan

    print("\n[Scene 3] Planner Comparison (dynamic)")

    ox = np.array([10, 20, 30, 10, 20, 30])
    oy = np.array([5, 5, 5, 15, 15, 15])
    obstacle_list = [(10, 5, 1.5), (20, 5, 1.5), (30, 5, 1.5),
                     (10, 15, 1.5), (20, 15, 1.5), (30, 15, 1.5)]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Scene 3: Planner Comparison", fontsize=14, fontweight='bold')

    rx_a, ry_a, stats_a = a_star_plan(0, 0, 40, 20, ox, oy, resolution=1.0, robot_radius=0.8)
    ax1.plot(ox, oy, ".k", markersize=8)
    for (obx, oby, obr) in obstacle_list:
        plot_obstacle_circle(ax1, obx, oby, obr, color="-b")
    ax1.plot(rx_a, ry_a, "-r", linewidth=2, label="A* Path")
    ax1.plot(0, 0, "og", markersize=10, label="Start")
    ax1.plot(40, 20, "xr", markersize=10, label="Goal")
    ax1.set_title(f"A* ({stats_a['path_length']:.1f}m, {stats_a['planning_time_ms']:.1f}ms)")
    ax1.legend(fontsize=8)
    ax1.axis("equal")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")

    rx_r, ry_r, stats_r = rrt_plan(0, 0, 40, 20, obstacle_list,
                                    rand_area=(-5, 45), expand_dis=3.0, max_iter=500)
    ax2.plot(ox, oy, ".k", markersize=8)
    for (obx, oby, obr) in obstacle_list:
        plot_obstacle_circle(ax2, obx, oby, obr, color="-b")
    ax2.plot(rx_r, ry_r, "-r", linewidth=2, label="RRT Path")
    ax2.plot(0, 0, "og", markersize=10, label="Start")
    ax2.plot(40, 20, "xr", markersize=10, label="Goal")
    ax2.set_title(f"RRT ({stats_r.get('path_length', 0):.1f}m, {stats_r['planning_time_ms']:.1f}ms)")
    ax2.legend(fontsize=8)
    ax2.axis("equal")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")

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
    ax3.plot(ox, oy, ".k", markersize=8)
    for (obx, oby, obr) in obstacle_list:
        plot_obstacle_circle(ax3, obx, oby, obr, color="-b")
    ax3.plot(traj_arr[:, 0], traj_arr[:, 1], "-r", linewidth=2, label="DWA Path")
    ax3.plot(0, 0, "og", markersize=10, label="Start")
    ax3.plot(40, 20, "xr", markersize=10, label="Goal")
    ax3.set_title("DWA (dynamic window)")
    ax3.legend(fontsize=8)
    ax3.axis("equal")
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")

    plt.tight_layout()
    save_path = os.path.join(FIGS_DIR, "visual_decision_planners.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Scene 3] Saved: {save_path}")
    plt.show()
    plt.close(fig)


# === Phase 5: Scene 4 — Controller Comparison ===

def scene4_controllers():
    from generated import DecisionOutput, PathPoint, Behavior, DecisionStatus, Header
    from PathTracking.controller_selector import (select_controller, CTRL_STANLEY, CTRL_FUZZY,
                                              CTRL_MPC, CTRL_RL, CTRL_PURE_PURSUIT)
    from system.vehicle_sim import simulate_vehicle

    print("\n[Scene 4] Controller Comparison (S-curve slalom)")

    ref_x, ref_y, ref_yaw, pole_x, pole_y = generate_slalom_path()
    ctrl_types = [CTRL_STANLEY, CTRL_FUZZY, CTRL_MPC, CTRL_RL, CTRL_PURE_PURSUIT]
    ctrl_labels = ["Stanley", "Fuzzy", "MPC", "RL", "PurePursuit"]
    ctrl_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"],
                   COLORS["purple"], COLORS["gray"]]

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle("Scene 4: Controller Comparison — S-curve Slalom", fontsize=14, fontweight='bold')

    ax.plot(ref_x, ref_y, "--r", linewidth=2, label="Reference Path", alpha=0.5)
    for px, py in zip(pole_x, pole_y):
        plot_obstacle_circle(ax, px, py, 0.5, color="-k")

    for ct, label, color in zip(ctrl_types, ctrl_labels, ctrl_colors):
        v_actual = 0.0
        x_actual = ref_x[0]
        y_actual = ref_y[0]
        theta_actual = ref_yaw[0]
        ctrl_state = None
        dt = 0.02
        traj_x, traj_y = [x_actual], [y_actual]

        for i in range(1, len(ref_x)):
            decision = DecisionOutput()
            decision.header.seq = i
            decision.behavior = Behavior.Value("BEHAVIOR_LANE_KEEP")
            decision.status = DecisionStatus.Value("DECISION_NORMAL")
            decision.target_speed = 5.0

            look_ahead = min(i + 10, len(ref_x) - 1)
            for j in range(i, look_ahead + 1):
                pp = decision.target_path.add()
                pp.pose.x = float(ref_x[j])
                pp.pose.y = float(ref_y[j])
                pp.pose.theta = float(ref_yaw[j])
                pp.curvature = 0.0
                pp.speed = 5.0

            try:
                ctrl_out, ctrl_state = select_controller(ct, decision, v_actual, ctrl_state, dt=dt)
                steer = ctrl_out.steering.steering_angle
                throttle = ctrl_out.throttle_brake.throttle
                brake = ctrl_out.throttle_brake.brake
            except Exception:
                steer, throttle, brake = 0.0, 0.3, 0.0

            v_actual, x_actual, y_actual, theta_actual = simulate_vehicle(
                steer, throttle, brake, v_actual, x_actual, y_actual, theta_actual, dt)
            traj_x.append(x_actual)
            traj_y.append(y_actual)

        ax.plot(traj_x, traj_y, "-", color=color, linewidth=1.5, label=label)

    ax.legend(fontsize=9)
    ax.axis("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    plt.tight_layout()
    save_path = os.path.join(FIGS_DIR, "visual_control_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Scene 4] Saved: {save_path}")
    plt.show()
    plt.close(fig)


# === Phase 6: Scene 5 — Localization ===

def scene5_localization():
    from Localization.ekf_localizer import ekf_localize, generate_ekf_test
    from Localization.pf_localizer import pf_localize, pf_init, pf_estimate
    from Localization.pf_localizer import generate_pf_test as gen_pf_test
    from Localization.fusion_localizer import fusion_localize

    print("\n[Scene 5] Localization Comparison (figure-8 path)")

    dt = 0.1
    n_steps = 200
    Q_ekf = np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0]) ** 2
    R_ekf = np.diag([1.0, 1.0]) ** 2
    NP = 100
    NTh = NP / 2.0
    R_motion_pf = np.diag([0.5, np.deg2rad(10.0)]) ** 2
    Q_obs_pf = np.diag([0.2, np.deg2rad(5.0)]) ** 2

    true_states, gps_obs, controls = generate_ekf_test(n_steps=n_steps, dt=dt)
    _, observations, _, landmarks = gen_pf_test(n_steps=n_steps, dt=dt, n_landmarks=5)

    x_ekf = true_states[0].copy()
    P_ekf = np.eye(4) * 0.01
    px, pw = pf_init(NP, true_states[0], 0.5)
    x_fused_init = true_states[0].copy()
    P_fused = np.eye(4) * 0.01

    ekf_traj = np.zeros((n_steps, 4))
    pf_traj = np.zeros((n_steps, 4))
    fused_traj = np.zeros((n_steps, 4))
    ekf_P_hist = np.zeros((n_steps, 4, 4))
    pf_P_hist = np.zeros((n_steps, 4, 4))

    for i in range(n_steps):
        z_gps = gps_obs[i] if not np.any(np.isnan(gps_obs[i])) else None
        x_ekf, P_ekf = ekf_localize(x_ekf, P_ekf, controls[i], z_gps, Q_ekf, R_ekf, dt)
        ekf_traj[i] = x_ekf
        ekf_P_hist[i] = P_ekf

        z_lm = observations[i]
        x_pf, P_pf, px, pw = pf_localize(px, pw, controls[i], z_lm, landmarks,
                                            R_motion_pf, Q_obs_pf, dt, NTh)
        pf_traj[i] = x_pf
        pf_P_hist[i] = P_pf

        x_fused, P_fused, (x_ekf, P_ekf), (px, pw) = fusion_localize(
            (x_ekf, P_ekf), (px, pw), controls[i], z_gps, z_lm, landmarks,
            Q_ekf, R_ekf, R_motion_pf, Q_obs_pf, dt, NTh)
        fused_traj[i] = x_fused

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Scene 5: Localization — Figure-8 Path", fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(true_states[:, 0], true_states[:, 1], "-b", linewidth=2, label="True")
    ax.plot(ekf_traj[:, 0], ekf_traj[:, 1], "--", color=COLORS["orange"], linewidth=1.5, label="EKF")
    ax.plot(pf_traj[:, 0], pf_traj[:, 1], "--", color=COLORS["green"], linewidth=1.5, label="PF")
    ax.plot(fused_traj[:, 0], fused_traj[:, 1], "--", color=COLORS["purple"], linewidth=1.5, label="Fusion")
    gps_mask = ~np.isnan(gps_obs[:, 0])
    ax.plot(gps_obs[gps_mask, 0], gps_obs[gps_mask, 1], "xg", markersize=4, alpha=0.5, label="GPS")
    for i in range(0, n_steps, 20):
        plot_cov_ellipse(ax, ekf_traj[i, 0], ekf_traj[i, 1], ekf_P_hist[i], color=COLORS["orange"])
    ax.legend(fontsize=8)
    ax.axis("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("(a) Trajectory Comparison")

    for idx, (traj, label, color) in enumerate([
        (ekf_traj, "EKF", COLORS["orange"]),
        (pf_traj, "PF", COLORS["green"]),
        (fused_traj, "Fusion", COLORS["purple"])
    ]):
        ax = axes.flat[idx + 1]
        err = np.sqrt(np.sum((traj[:, :2] - true_states[:, :2])**2, axis=1))
        ax.plot(err, "-", color=color, linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("Position Error [m]")
        ax.set_title(f"({chr(98+idx)}) {label} Error (RMSE={np.sqrt(np.mean(err**2)):.3f}m)")

    plt.tight_layout()
    save_path = os.path.join(FIGS_DIR, "visual_localization_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Scene 5] Saved: {save_path}")
    plt.show()
    plt.close(fig)


# === Phase 7: Scene 6 — SLAM Pipeline ===

def scene6_slam():
    from SLAM.fast_slam import (fast_slam, estimate_from_particles, create_particle,
                                  generate_slam_test)
    from SLAM.icp_matching import icp_match

    print("\n[Scene 6] SLAM Pipeline (dynamic)")

    n_steps = 200
    dt = 0.1
    n_landmarks = 8
    n_particles = 50
    Q = np.diag([0.2, np.deg2rad(5.0)]) ** 2
    R_motion = np.diag([0.5, np.deg2rad(10.0)]) ** 2
    NTh = n_particles / 1.5

    true_traj, obs_seq, controls, landmarks = generate_slam_test(n_steps, dt, n_landmarks)
    particles = [create_particle(true_traj[0, 0], true_traj[0, 1], true_traj[0, 2],
                                  n_landmarks=n_landmarks) for _ in range(n_particles)]

    est_traj = np.zeros((n_steps, 3))
    rmse_hist = np.zeros(n_steps)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Scene 6: SLAM Pipeline", fontsize=14, fontweight='bold')

    for i in range(n_steps):
        particles, _, _ = fast_slam(particles, controls[i], obs_seq[i], Q, R_motion, dt, NTh)
        est_traj[i], _ = estimate_from_particles(particles)
        err = est_traj[i, :2] - true_traj[i, :2]
        rmse_hist[i] = np.sqrt(np.mean(err**2))

        if show_animation and i % 5 == 0:
            for ax in axes.flat:
                ax.cla()

            ax = axes[0, 0]
            ax.plot(true_traj[:i+1, 0], true_traj[:i+1, 1], "-b", linewidth=2, label="True")
            ax.plot(est_traj[:i+1, 0], est_traj[:i+1, 1], "--r", linewidth=1.5, label="Estimated")
            if landmarks is not None and len(landmarks) > 0:
                ax.scatter(landmarks[:, 0], landmarks[:, 1], c="g", marker="*", s=100, label="Landmarks")
            px_arr = np.array([p["x"] for p in particles])
            py_arr = np.array([p["y"] for p in particles])
            ax.scatter(px_arr, py_arr, c="gray", s=3, alpha=0.3, label="Particles")
            ax.legend(fontsize=7)
            ax.axis("equal")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title("(a) Trajectory + Particles")

            ax = axes[0, 1]
            pos_err = np.sqrt(np.sum((est_traj[:i+1, :2] - true_traj[:i+1, :2])**2, axis=1))
            ax.plot(pos_err, "-r", linewidth=1)
            ax.set_xlabel("Step")
            ax.set_ylabel("Position Error [m]")
            ax.set_title("(b) Position Error")

            ax = axes[1, 0]
            ax.plot(rmse_hist[:i+1], "-b", linewidth=1)
            ax.set_xlabel("Step")
            ax.set_ylabel("RMSE [m]")
            ax.set_title("(c) Cumulative RMSE")

            ax = axes[1, 1]
            if i >= 20:
                step = 10
                prev = true_traj[i-step:i, :2].T
                curr = true_traj[i-step+1:i+1, :2].T
                if prev.shape[1] >= 3:
                    _, _, err_h = icp_match(prev, curr, max_iter=30)
                    ax.plot(err_h, "-g", linewidth=1)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("ICP Error [m]")
            ax.set_title("(d) ICP Convergence")

            plt.tight_layout()
            plt.pause(0.001)

    save_path = os.path.join(FIGS_DIR, "visual_slam_pipeline.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Scene 6] Saved: {save_path}")
    plt.show()
    plt.close(fig)


# === Phase 8: Scene 7 — Main Loop City Drive ===

def scene7_mainloop():
    from generated import (PerceptionOutput, DecisionOutput, Behavior,
                           DecisionStatus, ControlOutput, ControlMode, Header, PathPoint)
    from perception.lane_pixel_detector import detect_lane_pixels, generate_test_image as gen_lane_img
    from perception.obstacle_detector import detect_obstacles, generate_test_point_cloud
    from perception.sign_recognizer import recognize_signs, generate_test_sign_image
    from perception.sensor_fusion import fuse_to_perception, default_camera_params
    from decision.task_scheduler import schedule, TASK_PATROL
    from PathTracking.controller_selector import select_controller, CTRL_STANLEY
    from system.vehicle_sim import simulate_vehicle

    print("\n[Scene 7] Main Loop — City Intersection Drive")

    ref_x, ref_y, ref_yaw, obstacles_city = generate_city_path()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Scene 7: Main Loop — City Drive", fontsize=14, fontweight='bold')

    test_img = gen_lane_img()
    test_pc = generate_test_point_cloud()
    test_sign_img = generate_test_sign_image()
    K, R_cam, t_cam = default_camera_params()
    from perception.sign_recognizer import _get_template_db
    _get_template_db()
    signs = recognize_signs(test_sign_img)
    obstacles = detect_obstacles(test_pc)

    v_actual = 0.0
    x_actual = ref_x[0]
    y_actual = ref_y[0]
    theta_actual = ref_yaw[0]
    pid_state = (0.0, 0.0)
    ctrl_state = None
    scheduler_state = {"current_task": TASK_PATROL, "task_state": {}}

    traj_x, traj_y = [x_actual], [y_actual]
    v_hist, steer_hist = [0.0], [0.0]
    beh_hist = []

    dt = 0.02
    n_steps = len(ref_x)

    for i in range(1, n_steps):
        scan_rows, center_x = detect_lane_pixels(test_img)
        perception_out = fuse_to_perception(scan_rows, center_x, obstacles, signs, K, R_cam, t_cam)
        perception_out.header.seq = i

        decision_out, scheduler_state = schedule(perception_out, scheduler_state, v_nominal=5.0)

        look_ahead = min(i + 10, n_steps - 1)
        for j in range(i, look_ahead + 1):
            pp = decision_out.target_path.add()
            pp.pose.x = float(ref_x[j])
            pp.pose.y = float(ref_y[j])
            pp.pose.theta = float(ref_yaw[j])

        ctrl_out, ctrl_state = select_controller(CTRL_STANLEY, decision_out, v_actual, ctrl_state, dt=dt)

        v_actual, x_actual, y_actual, theta_actual = simulate_vehicle(
            ctrl_out.steering.steering_angle,
            ctrl_out.throttle_brake.throttle,
            ctrl_out.throttle_brake.brake,
            v_actual, x_actual, y_actual, theta_actual, dt)

        traj_x.append(x_actual)
        traj_y.append(y_actual)
        v_hist.append(v_actual)
        steer_hist.append(ctrl_out.steering.steering_angle)
        beh_hist.append(Behavior.Name(decision_out.behavior))

        if show_animation and i % 3 == 0:
            for ax in axes.flat:
                ax.cla()

            ax = axes[0, 0]
            ax.plot(ref_x, ref_y, "--r", linewidth=1.5, alpha=0.5, label="Reference")
            ax.plot(traj_x, traj_y, "-b", linewidth=1.5, label="Actual")
            for (ox, oy, or_) in obstacles_city:
                plot_obstacle_circle(ax, ox, oy, or_, color="-k")
            plot_vehicle(ax, x_actual, y_actual, theta_actual,
                         steer_deg=ctrl_out.steering.steering_angle)
            ax.legend(fontsize=7)
            ax.axis("equal")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title("(a) Vehicle Trajectory")

            ax = axes[0, 1]
            ax.plot(v_hist, "-", color=COLORS["blue"], linewidth=1)
            ax.set_xlabel("Step")
            ax.set_ylabel("Speed [m/s]")
            ax.set_title("(b) Speed")

            ax = axes[1, 0]
            ax.plot(steer_hist, "-", color=COLORS["orange"], linewidth=1)
            ax.set_xlabel("Step")
            ax.set_ylabel("Steering [deg]")
            ax.set_title("(c) Steering Angle")

            ax = axes[1, 1]
            if beh_hist:
                unique_beh = list(set(beh_hist))
                beh_colors = {b: COLORS[k] for b, k in zip(unique_beh,
                              ["blue", "orange", "green", "red", "purple", "gray"][:len(unique_beh)])}
                for bi, bh in enumerate(beh_hist):
                    ax.bar(bi, 1, color=beh_colors.get(bh, COLORS["gray"]), width=1.0)
                from matplotlib.patches import Patch
                legend_patches = [Patch(color=beh_colors[b], label=b) for b in unique_beh]
                ax.legend(handles=legend_patches, fontsize=7)
            ax.set_xlabel("Step")
            ax.set_title("(d) Behavior State")

            plt.tight_layout()
            plt.pause(0.001)

    save_path = os.path.join(FIGS_DIR, "visual_mainloop_city.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Scene 7] Saved: {save_path}")
    plt.show()
    plt.close(fig)


# === Phase 9: Main Entry ===

if __name__ == "__main__":
    print("=" * 70)
    print("  Comprehensive Visual Test — PythonRobotics Style")
    print("  7 Scenes: Perception / Decision / Control / Localization / SLAM / MainLoop")
    print("=" * 70)

    scene1_camera_lane()
    scene2_obstacle_sign()
    scene3_planners()
    scene4_controllers()
    scene5_localization()
    scene6_slam()
    scene7_mainloop()

    print("\n" + "=" * 70)
    print("  All 7 scenes completed! Check figs/ for saved images.")
    print("=" * 70)
