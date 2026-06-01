import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io, os, sys, math, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

IMG_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(IMG_DIR, exist_ok=True)

import imageio.v2 as imageio
from PIL import Image

_frames = []
_target_size = (640, 480)

def _capture_frame(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    buf.seek(0)
    img = imageio.imread(buf)
    buf.close()
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(_target_size, Image.LANCZOS)
    _frames.append(np.array(pil_img))

def _save_gif(name, duration=80, loop=0):
    if len(_frames) == 0:
        print(f"  [GIF] {name}: no frames!")
        return
    path = os.path.join(IMG_DIR, f"{name}.gif")
    imageio.mimsave(path, _frames, duration=duration, loop=loop)
    print(f"  -> {path} ({len(_frames)} frames)")
    _frames.clear()

def _save_png(name, fig):
    path = os.path.join(IMG_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  -> {path}")
    plt.close(fig)


# === 0. Obstacle Detection PNG ===
def gen_obstacle_detection():
    from perception.obstacle_detector import (detect_obstacles, generate_test_point_cloud,
                                              _fit_plane_ransac, _voxel_downsample,
                                              _dbscan, _adaptive_dbscan_params,
                                              _OBSTACLE_TYPE_VEHICLE, _OBSTACLE_TYPE_PEDESTRIAN,
                                              _OBSTACLE_TYPE_STATIC, _OBSTACLE_TYPE_UNKNOWN)
    pc = generate_test_point_cloud()
    rng = np.random.default_rng(42)
    ground_mask = _fit_plane_ransac(pc, 100, 0.1, rng)
    non_ground = pc[~ground_mask]
    downsampled = _voxel_downsample(non_ground, 0.1) if len(non_ground) > 5000 else non_ground
    eps, min_pts = _adaptive_dbscan_params(downsampled, 0.5, 5)
    labels = _dbscan(downsampled, eps, min_pts)
    obs = detect_obstacles(pc)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax1 = axes[0, 0]
    ax1.scatter(pc[:, 0], pc[:, 1], s=1, c='0.6', alpha=0.3)
    ax1.set_title("Raw Point Cloud (Top View)")
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
    ax1.set_aspect('equal'); ax1.grid(True)

    ax2 = axes[0, 1]
    ax2.scatter(pc[ground_mask, 0], pc[ground_mask, 1], s=1, c='0.8', alpha=0.3, label='ground')
    ax2.scatter(pc[~ground_mask, 0], pc[~ground_mask, 1], s=1, c='r', alpha=0.5, label='non-ground')
    ax2.set_title("RANSAC Ground Removal")
    ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
    ax2.legend(frameon=True, fancybox=True); ax2.set_aspect('equal'); ax2.grid(True)

    ax3 = axes[1, 0]
    unique_labels = sorted(set(labels.tolist()) - {-1})
    cmap = plt.cm.Set1(np.linspace(0, 1, max(len(unique_labels), 1)))
    noise_mask = labels == -1
    ax3.scatter(downsampled[noise_mask, 0], downsampled[noise_mask, 1], s=1, c='0.7', alpha=0.3, label='noise')
    for lbl, c in zip(unique_labels, cmap):
        mask = labels == lbl
        ax3.scatter(downsampled[mask, 0], downsampled[mask, 1], s=3, c=[c], alpha=0.6, label=f'cluster {lbl}')
    ax3.set_title("DBSCAN Clustering (Adaptive)")
    ax3.set_xlabel("x [m]"); ax3.set_ylabel("y [m]")
    ax3.legend(frameon=True, fancybox=True); ax3.set_aspect('equal'); ax3.grid(True)

    ax4 = axes[1, 1]
    ax4.scatter(downsampled[:, 0], downsampled[:, 1], s=1, c='0.6', alpha=0.3)
    type_colors = {
        _OBSTACLE_TYPE_VEHICLE: 'b', _OBSTACLE_TYPE_PEDESTRIAN: 'g',
        _OBSTACLE_TYPE_STATIC: 'orange', _OBSTACLE_TYPE_UNKNOWN: 'r',
    }
    for i, o in enumerate(obs):
        color = type_colors.get(o['type'], 'r')
        cx, cy = o['center']
        L, W, hdg = o['length'], o['width'], o['heading']
        corners_local = np.array([
            [-L/2, -W/2], [L/2, -W/2], [L/2, W/2], [-L/2, W/2], [-L/2, -W/2]
        ])
        c_h, s_h = np.cos(hdg), np.sin(hdg)
        R = np.array([[c_h, -s_h], [s_h, c_h]])
        corners_global = corners_local @ R.T + np.array([cx, cy])
        ax4.plot(corners_global[:, 0], corners_global[:, 1], '-', color=color, linewidth=2)
        ax4.plot(cx, cy, 'xk', markersize=10)
        ax4.text(cx, cy + 0.3, f"#{i} {o['type'].replace('OBSTACLE_', '')}", ha='center', fontsize=8)
    ax4.set_title("OBB Bounding Boxes (Typed)")
    ax4.set_xlabel("x [m]"); ax4.set_ylabel("y [m]")
    ax4.set_aspect('equal'); ax4.grid(True)

    plt.tight_layout()
    _save_png("obstacle_detection", fig)


# === 1. Obstacle Tracking GIF ===
def gen_obstacle_tracking():
    from perception.obstacle_tracker import track_obstacles, get_confirmed_tracks
    rng = np.random.default_rng(42)
    n_frames = 40
    tracks = []
    free_ids = set()
    history = {}
    obj_types = ["OBSTACLE_VEHICLE", "OBSTACLE_PEDESTRIAN", "OBSTACLE_STATIC"]
    for frame_i in range(n_frames):
        detections = []
        for d in range(3):
            cx = 5.0 + d * 5.0 + rng.normal(0, 0.3)
            cy = 5.0 + 2.0 * np.sin(frame_i * 0.15 + d) + rng.normal(0, 0.2)
            detections.append({
                "center": np.array([cx, cy]),
                "length": 1.5 + rng.normal(0, 0.1),
                "width": 0.8 + rng.normal(0, 0.1),
                "heading": rng.normal(0, 0.1),
                "type": obj_types[d],
            })
        tracks = track_obstacles(detections, tracks, dt=0.1, free_ids=free_ids)
        confirmed = get_confirmed_tracks(tracks)
        for t in tracks:
            tid = t["id"]
            if tid not in history:
                history[tid] = []
            history[tid].append((t["x"], t["y"]))
        fig, ax = plt.subplots(figsize=(6, 5))
        for tid, pts in history.items():
            arr = np.array(pts)
            color = plt.cm.tab10(tid % 10)
            ax.plot(arr[:, 0], arr[:, 1], '-', color=color, alpha=0.5, linewidth=1)
            ax.plot(arr[-1, 0], arr[-1, 1], 'o', color=color, markersize=8)
            label = f'ID:{tid}'
            for t in tracks:
                if t["id"] == tid:
                    label = f'ID:{tid} {t.get("type","").replace("OBSTACLE_","")}'
                    break
            ax.annotate(label, (arr[-1, 0], arr[-1, 1]), fontsize=8)
        for det in detections:
            ax.plot(det["center"][0], det["center"][1], 'x', color='gray', markersize=6, alpha=0.5)
        n_conf = len(confirmed)
        ax.set_title(f'Obstacle Tracking (Hungarian+KF) — Frame {frame_i}  confirmed={n_conf}')
        ax.set_xlim(-2, 25)
        ax.set_ylim(-2, 15)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.grid(True)
        ax.axis('equal')
        _capture_frame(fig)
        plt.close(fig)
    _save_gif("obstacle_tracking", duration=100)


# === 2. Sign Recognition PNG ===
def gen_sign_recognition():
    from perception.sign_recognizer import recognize_signs, generate_test_sign_image
    import cv2
    img = generate_test_sign_image()
    results = recognize_signs(img)
    fig, ax = plt.subplots(figsize=(8, 6))
    img_rgb = img.copy()
    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb]*3, axis=-1)
    ax.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    color_map = {"red": "lime", "blue": "cyan", "yellow": "orange"}
    for r in results:
        x, y, w, h = r["bbox"]
        edge_color = color_map.get(r.get("color", "red"), "lime")
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=edge_color, facecolor='none')
        ax.add_patch(rect)
        label = f'{r["category"]} ({r["confidence"]:.2f}) [{r.get("color","?")}]'
        ax.text(x, y - 5, label, color=edge_color, fontsize=9, fontweight='bold')
    ax.set_title('Sign Recognition — HOG Template Matching')
    ax.axis('off')
    _save_png("sign_recognition", fig)


# === 3. Sensor Fusion PNG ===
def gen_sensor_fusion():
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cam_pts = rng.normal(0, 1, (20, 2))
    cam_pts[:, 0] = cam_pts[:, 0] * 3 + 5
    cam_pts[:, 1] = cam_pts[:, 1] * 2 + 3
    axes[0].scatter(cam_pts[:, 0], cam_pts[:, 1], c='blue', s=30, label='Camera 2D')
    axes[0].set_title('Camera Detections')
    axes[0].set_xlabel('x [m]'); axes[0].set_ylabel('y [m]')
    axes[0].legend(); axes[0].grid(True); axes[0].axis('equal')
    lidar_pts = rng.normal(0, 1, (50, 2))
    lidar_pts[:, 0] = lidar_pts[:, 0] * 4 + 5
    lidar_pts[:, 1] = lidar_pts[:, 1] * 3 + 3
    axes[1].scatter(lidar_pts[:, 0], lidar_pts[:, 1], c='red', s=10, alpha=0.6, label='LiDAR 3D')
    axes[1].set_title('LiDAR Point Cloud')
    axes[1].set_xlabel('x [m]'); axes[1].set_ylabel('y [m]')
    axes[1].legend(); axes[1].grid(True); axes[1].axis('equal')
    axes[2].scatter(lidar_pts[:, 0], lidar_pts[:, 1], c='red', s=10, alpha=0.3, label='LiDAR')
    axes[2].scatter(cam_pts[:, 0], cam_pts[:, 1], c='blue', s=40, alpha=0.7, label='Camera')
    fused_x = np.concatenate([cam_pts[:, 0], lidar_pts[:, 0]])
    fused_y = np.concatenate([cam_pts[:, 1], lidar_pts[:, 1]])
    axes[2].scatter(fused_x, fused_y, c='green', s=15, alpha=0.4, label='Fused')
    axes[2].set_title('Sensor Fusion Result')
    axes[2].set_xlabel('x [m]'); axes[2].set_ylabel('y [m]')
    axes[2].legend(); axes[2].grid(True); axes[2].axis('equal')
    plt.tight_layout()
    _save_png("sensor_fusion", fig)


# === 4. Path Smoothing GIF ===
def gen_path_smoothing():
    from decision.path_smoother import smooth_path
    rng = np.random.default_rng(42)
    t = np.linspace(0, 10, 30)
    x_raw = t + rng.normal(0, 0.3, len(t))
    y_raw = np.sin(t * 0.5) * 3 + rng.normal(0, 0.3, len(t))
    sigmas = np.arange(0.0, 5.5, 0.5)
    for sigma_val in sigmas:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_raw, y_raw, 'o-', color='gray', alpha=0.5, linewidth=1, label='Raw')
        if sigma_val > 0:
            try:
                sx, sy, kappa = smooth_path(x_raw, y_raw, max_deviation=2.0, n_output=50)
                ax.plot(sx, sy, '-r', linewidth=2, label=f'Smoothed (σ={sigma_val:.1f})')
            except Exception:
                ax.plot(x_raw, y_raw, '-r', linewidth=2, label='Smoothed (failed)')
        else:
            ax.plot(x_raw, y_raw, '-r', linewidth=2, label='Original')
        ax.set_title(f'Path Smoothing — σ = {sigma_val:.1f}')
        ax.legend(frameon=True, fancybox=True)
        ax.axis('equal')
        ax.grid(True)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        _capture_frame(fig)
        plt.close(fig)
    _save_gif("path_smoothing", duration=200)


# === 5. Multi-agent PNG ===
def gen_multi_agent():
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(8, 6))
    n_agents = 4
    id_ranges = [(i * 100, (i + 1) * 100 - 1) for i in range(n_agents)]
    colors = ['blue', 'red', 'green', 'orange']
    for i in range(n_agents):
        t_vals = np.linspace(0, 10, 50)
        x_vals = t_vals * 2
        y_vals = 2.0 * i + np.sin(t_vals + i) * 0.5
        ax.plot(x_vals, y_vals, '-', color=colors[i], linewidth=1.5,
                label=f'Agent {i} (ID {id_ranges[i][0]}-{id_ranges[i][1]})')
        ax.plot(x_vals[-1], y_vals[-1], 'o', color=colors[i], markersize=8)
    ax.set_title('Multi-agent Coordination — ID Ranges & Trajectories')
    ax.legend(frameon=True, fancybox=True)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.grid(True)
    ax.axis('equal')
    _save_png("multi_agent", fig)


# === 6. DQN Control GIF ===
def gen_dqn_control():
    from PathTracking.rl_controller import _PathTrackEnv, _forward
    weights_path = os.path.join(os.path.dirname(__file__), "..", "control", "dqn_weights.npz")
    if not os.path.exists(weights_path):
        print("  [DQN] No trained weights found, skipping")
        return
    data = np.load(weights_path)
    W1, b1, W2, b2, W3, b3 = data['W1'], data['b1'], data['W2'], data['b2'], data['W3'], data['b3']
    rng = np.random.default_rng(42)
    env = _PathTrackEnv(rng)
    state = env.reset()
    x_hist, y_hist = [env.x], [env.y]
    for step in range(500):
        q, _, _ = _forward(state, W1, b1, W2, b2, W3, b3)
        action = int(np.argmax(q))
        state, reward, done = env.step(action)
        x_hist.append(env.x)
        y_hist.append(env.y)
        if step % 3 == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(env.cx, env.cy, '-r', linewidth=1, alpha=0.5, label='Reference')
            ax.plot(x_hist, y_hist, '-b', linewidth=1.5, label='DQN Path')
            ax.plot(env.x, env.y, 'ko', markersize=8)
            ax.plot(env.cx[-1], env.cy[-1], 'g*', markersize=12, label='Goal')
            ax.set_title(f'DQN Path Tracking — Step {step}')
            ax.legend(frameon=True, fancybox=True)
            ax.grid(True)
            ax.axis('equal')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            _capture_frame(fig)
            plt.close(fig)
        if done:
            for _ in range(5):
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(env.cx, env.cy, '-r', linewidth=1, alpha=0.5, label='Reference')
                ax.plot(x_hist, y_hist, '-b', linewidth=1.5, label='DQN Path')
                ax.plot(env.x, env.y, 'ko', markersize=8)
                ax.plot(env.cx[-1], env.cy[-1], 'g*', markersize=12, label='Goal')
                ax.set_title('DQN Path Tracking — Goal Reached!')
                ax.legend(frameon=True, fancybox=True)
                ax.grid(True)
                ax.axis('equal')
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
                _capture_frame(fig)
                plt.close(fig)
            break
    _save_gif("dqn_control", duration=100)


# === 7. Controller Selection GIF ===
def gen_controller_selection():
    from PathTracking.controller_selector import auto_select_controller, ctrl_name
    from utils.plot import generate_serpentine_course
    from generated import DecisionOutput, Behavior

    cx, cy, cyaw, ck = generate_serpentine_course(ds=0.1)
    dt = 0.1
    L = 2.9
    x, y, yaw, v = cx[0], cy[0], cyaw[0], 2.0
    x_hist, y_hist = [x], [y]
    ctrl_hist = []
    state = None

    for step in range(500):
        decision = DecisionOutput()
        decision.header.seq = step
        decision.behavior = Behavior.Value("BEHAVIOR_LANE_KEEP")
        decision.target_speed = 5.0
        n_path = len(cx)
        for i in range(n_path):
            pp = decision.target_path.add()
            pp.pose.x = float(cx[i])
            pp.pose.y = float(cy[i])
            pp.pose.theta = float(cyaw[i])
            pp.curvature = float(ck[i])

        ctrl_out, state = auto_select_controller(
            decision, v, state,
            vehicle_x=x, vehicle_y=y, vehicle_theta=yaw, dt=dt
        )
        steer = np.radians(ctrl_out.steering.steering_angle)
        throttle = ctrl_out.throttle_brake.throttle
        brake = ctrl_out.throttle_brake.brake

        accel = throttle * 3.0 - brake * 5.0
        v = max(v + accel * dt, 0.0)
        v = min(v, 8.0)
        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt
        yaw += v / L * np.tan(steer) * dt
        x_hist.append(x)
        y_hist.append(y)

        current_ctrl = ctrl_name(state.get("type", 0))
        ctrl_hist.append(current_ctrl)

        if step % 5 == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(cx, cy, '.r', markersize=1, alpha=0.3, label='Reference')
            ax.plot(x_hist, y_hist, '-b', linewidth=1.5, label='Trajectory')
            ax.plot(x, y, 'ko', markersize=8)
            color_map = {"PurePursuit": "green", "Stanley": "blue",
                         "Fuzzy": "orange", "MPC": "red", "RL": "purple"}
            ax.set_title(f'Controller Selection — {current_ctrl}',
                         color=color_map.get(current_ctrl, "black"))
            ax.legend(frameon=True, fancybox=True)
            ax.grid(True)
            ax.axis('equal')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            _capture_frame(fig)
            plt.close(fig)

        idx = int(np.argmin((cx - x)**2 + (cy - y)**2))
        if idx >= len(cx) - 5:
            for _ in range(5):
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(cx, cy, '.r', markersize=1, alpha=0.3, label='Reference')
                ax.plot(x_hist, y_hist, '-b', linewidth=1.5, label='Trajectory')
                ax.plot(x, y, 'ko', markersize=8)
                ax.plot(cx[-1], cy[-1], 'g*', markersize=12, label='Goal')
                ax.set_title(f'Controller Selection — Goal Reached! ({current_ctrl})',
                             color=color_map.get(current_ctrl, "black"))
                ax.legend(frameon=True, fancybox=True)
                ax.grid(True)
                ax.axis('equal')
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
                _capture_frame(fig)
                plt.close(fig)
            break

    _save_gif("controller_selection", duration=80)


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else [
        "obstacle_detection", "obstacle_tracking", "sign_recognition", "sensor_fusion",
        "path_smoothing", "multi_agent", "dqn_control", "controller_selection"
    ]
    funcs = {
        "obstacle_detection": gen_obstacle_detection,
        "obstacle_tracking": gen_obstacle_tracking,
        "sign_recognition": gen_sign_recognition,
        "sensor_fusion": gen_sensor_fusion,
        "path_smoothing": gen_path_smoothing,
        "multi_agent": gen_multi_agent,
        "dqn_control": gen_dqn_control,
        "controller_selection": gen_controller_selection,
    }
    for name in targets:
        if name not in funcs:
            print(f"Unknown: {name}")
            continue
        print(f"Generating {name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            funcs[name]()
            print(f"OK ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
