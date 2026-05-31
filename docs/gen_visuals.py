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


# === 1. Obstacle Tracking GIF ===
def gen_obstacle_tracking():
    from perception.obstacle_tracker import track_obstacles, get_confirmed_tracks
    rng = np.random.default_rng(42)
    n_frames = 40
    tracks = []
    history = {}
    for frame_i in range(n_frames):
        n_det = rng.integers(2, 5)
        detections = []
        for d in range(n_det):
            cx = 5.0 + d * 5.0 + rng.normal(0, 0.3)
            cy = 5.0 + 2.0 * np.sin(frame_i * 0.15 + d) + rng.normal(0, 0.2)
            detections.append({
                "center": np.array([cx, cy]),
                "length": 1.5 + rng.normal(0, 0.1),
                "width": 0.8 + rng.normal(0, 0.1),
                "heading": rng.normal(0, 0.1),
            })
        tracks = track_obstacles(detections, tracks, dt=0.1)
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
            ax.annotate(f'ID:{tid}', (arr[-1, 0], arr[-1, 1]), fontsize=8)
        for det in detections:
            ax.plot(det["center"][0], det["center"][1], 'x', color='gray', markersize=6, alpha=0.5)
        ax.set_title(f'Obstacle Tracking — Frame {frame_i}')
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
    img = generate_test_sign_image()
    results = recognize_signs(img)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if 'cv2' in dir() else img)
    img_rgb = img.copy()
    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb]*3, axis=-1)
    ax.imshow(img_rgb)
    for r in results:
        x, y, w, h = r["bbox"]
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        label = f'{r["category"]} ({r["confidence"]:.2f})'
        ax.text(x, y - 5, label, color='lime', fontsize=10, fontweight='bold')
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
    from PathTracking.rl_controller import _GridWorld, _forward, _relu_grad
    rng = np.random.default_rng(42)
    env = _GridWorld(rng)
    test_obs = env.reset()
    n_obs_dim = len(test_obs)
    n_act_dim, n_h1, n_h2 = 6, 64, 32
    W1 = rng.normal(0, 0.1, (n_obs_dim, n_h1))
    b1 = np.zeros(n_h1)
    W2 = rng.normal(0, 0.1, (n_h1, n_h2))
    b2 = np.zeros(n_h2)
    W3 = rng.normal(0, 0.1, (n_h2, n_act_dim))
    b3 = np.zeros(n_act_dim)
    for _ in range(50):
        state = env.reset()
        while not env.done:
            q, _, _ = _forward(state, W1, b1, W2, b2, W3, b3)
            action = int(np.argmax(q))
            next_state, reward, done = env.step(action)
            state = next_state
    state = env.reset()
    x_hist, y_hist = [env.x], [env.y]
    for step in range(100):
        q, _, _ = _forward(state, W1, b1, W2, b2, W3, b3)
        action = int(np.argmax(q))
        state, reward, done = env.step(action)
        x_hist.append(env.x)
        y_hist.append(env.y)
        if step % 2 == 0:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.set_xlim(-0.5, 10.5)
            ax.set_ylim(-0.5, 10.5)
            for j in range(env.n_obs):
                rect = plt.Rectangle(
                    (env.obs_x[j] - env.obs_w[j]/2, env.obs_y[j] - env.obs_h[j]/2),
                    env.obs_w[j], env.obs_h[j], color='gray', alpha=0.5)
                ax.add_patch(rect)
            ax.plot(x_hist, y_hist, '-b', linewidth=1.5, label='DQN Path')
            ax.plot(env.x, env.y, 'ro', markersize=8)
            ax.plot(9.0, 5.0, 'g*', markersize=12, label='Goal')
            ax.set_title(f'DQN Control — Step {step}')
            ax.legend(frameon=True, fancybox=True)
            ax.grid(True)
            ax.axis('equal')
            _capture_frame(fig)
            plt.close(fig)
        if done:
            break
    _save_gif("dqn_control", duration=100)


# === 7. Controller Selection GIF ===
def gen_controller_selection():
    from utils.plot import generate_serpentine_course
    cx, cy, cyaw, ck = generate_serpentine_course(ds=0.1)
    dt = 0.1
    L = 2.9
    x, y, yaw, v = cx[0], cy[0], cyaw[0], 0.0
    x_hist, y_hist = [x], [y]
    ctrl_hist = []
    for step in range(200):
        dx = cx - x
        dy = cy - y
        dists = dx**2 + dy**2
        idx = int(np.argmin(dists))
        kappa_local = abs(ck[min(idx, len(ck)-1)])
        if v > 5.0:
            ctrl_name = "Stanley"
            steer = np.radians(-5.0)
        elif kappa_local > 0.05:
            ctrl_name = "Fuzzy"
            steer = np.radians(-8.0)
        else:
            ctrl_name = "PurePursuit"
            steer = np.radians(-3.0)
        throttle = 0.5
        v += throttle * dt
        v = min(v, 8.0)
        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt
        yaw += v / L * np.tan(steer) * dt
        x_hist.append(x)
        y_hist.append(y)
        ctrl_hist.append(ctrl_name)
        if step % 5 == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(cx, cy, '.r', markersize=1, alpha=0.3, label='Reference')
            ax.plot(x_hist, y_hist, '-b', linewidth=1.5, label='Trajectory')
            ax.plot(x, y, 'ko', markersize=8)
            color_map = {"Stanley": "blue", "Fuzzy": "red", "PurePursuit": "green"}
            ax.set_title(f'Controller Selection — {ctrl_name}', color=color_map.get(ctrl_name, "black"))
            ax.legend(frameon=True, fancybox=True)
            ax.grid(True)
            ax.axis('equal')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            _capture_frame(fig)
            plt.close(fig)
        if idx >= len(cx) - 5:
            break
    _save_gif("controller_selection", duration=80)


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else [
        "obstacle_tracking", "sign_recognition", "sensor_fusion",
        "path_smoothing", "multi_agent", "dqn_control", "controller_selection"
    ]
    funcs = {
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
