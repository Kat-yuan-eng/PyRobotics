import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import os
import sys
import time
import imageio.v2 as imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

GIF_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(GIF_DIR, exist_ok=True)

_frames = []
_target_size = (640, 480)

def _capture_frame(fig=None):
    if fig is None:
        fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    buf.seek(0)
    img = imageio.imread(buf)
    buf.close()
    from PIL import Image
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(_target_size, Image.LANCZOS)
    _frames.append(np.array(pil_img))

def _mock_show(*args, **kwargs):
    _capture_frame()

def _mock_pause(interval=0.001, *args, **kwargs):
    _capture_frame()

def save_gif(name, frames=None, duration=80, loop=0):
    if frames is None:
        frames = _frames
    if len(frames) == 0:
        print(f"  [GIF] {name}: no frames captured!")
        return
    path = os.path.join(GIF_DIR, f"{name}.gif")
    imageio.mimsave(path, frames, duration=duration, loop=loop)
    print(f"  -> {path} ({len(frames)} frames)")
    _frames.clear()

def patch_plt():
    plt.show = _mock_show
    plt.pause = _mock_pause

patch_plt()

# === DWA ===
def capture_dwa():
    import math
    from PathPlanning.dwa_planner import dwa_plan
    ox = [10, 20, 30, 10, 20, 30]
    oy = [5, 5, 5, 15, 15, 15]
    ob = np.column_stack([ox, oy])
    goal = np.array([40.0, 20.0])
    x_state = np.array([0.0, 0.0, 0.0, 0.5, 0.0])
    x = x_state.copy()
    traj_all = [x[:2].copy()]
    for step in range(500):
        u, traj, _ = dwa_plan(x, goal, ob, max_speed=2.0, max_accel=1.0,
                              max_delta_yaw_rate=100.0, yaw_rate_resolution=1.0)
        x[0] += u[0] * math.cos(x[2]) * 0.1
        x[1] += u[0] * math.sin(x[2]) * 0.1
        x[2] += u[1] * 0.1
        x[3] = u[0]
        traj_all.append(x[:2].copy())
        if step % 5 == 0:
            fig, ax = plt.subplots(figsize=(6, 5))
            arr = np.array(traj_all)
            ax.plot(ox, oy, ".k", markersize=8)
            ax.plot(arr[:, 0], arr[:, 1], "-r", linewidth=1.5)
            ax.plot(0, 0, "og", markersize=8)
            ax.plot(40, 20, "xr", markersize=10)
            ax.set_title(f"DWA Step {step}")
            ax.axis("equal")
            ax.grid(True)
            _capture_frame(fig)
            plt.close(fig)
        if u[0] == 0.0 and u[1] == 0.0:
            break
    save_gif("dwa", duration=60)

# === FastSLAM ===
def capture_fastslam():
    from SLAM.fast_slam import (fast_slam, estimate_from_particles, create_particle,
                                 generate_slam_test, normalize_angle)
    dt = 0.1; NP = 200; NTh = NP * 0.5
    Q = np.diag([0.2, np.deg2rad(5.0)])**2
    R_motion = np.diag([0.2, np.deg2rad(5.0)])**2
    rng = np.random.default_rng(42)
    true_traj, obs_seq, controls, landmarks = generate_slam_test(dt=dt)
    n_steps = len(true_traj); n_lm = len(landmarks)
    particles = [create_particle(true_traj[0,0], true_traj[0,1], true_traj[0,2], n_landmarks=n_lm) for _ in range(NP)]
    est_traj = np.zeros((n_steps, 3))
    for i in range(n_steps):
        particles, _, _ = fast_slam(particles, controls[i], obs_seq[i], Q, R_motion, dt, NTh, adaptive=False, rng=rng)
        est_traj[i], _ = estimate_from_particles(particles)
        if i % 10 == 0:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(landmarks[:,0], landmarks[:,1], "xb", markersize=8)
            ax.plot(true_traj[:i+1,0], true_traj[:i+1,1], "-b", linewidth=1.5, label="Truth")
            ax.plot(est_traj[:i+1,0], est_traj[:i+1,1], "-r", linewidth=1.5, label="Est")
            ax.set_title(f"FastSLAM Step {i}")
            ax.legend(frameon=True, fancybox=True)
            ax.axis("equal"); ax.grid(True)
            _capture_frame(fig)
            plt.close(fig)
    save_gif("fastslam", duration=60)

# === EKF ===
def capture_ekf():
    from Localization.ekf_localizer import main
    main()
    save_gif("ekf", duration=80)

# === Stanley ===
def capture_stanley():
    from PathTracking.stanley_controller import main
    main()
    save_gif("stanley", duration=80)

# === Pure Pursuit ===
def capture_pure_pursuit():
    from PathTracking.pure_pursuit_controller import main
    main()
    save_gif("pure_pursuit", duration=80)

# === MPC ===
def capture_mpc():
    from PathTracking.mpc_controller import main
    main()
    save_gif("mpc", duration=80)

# === Particle Filter ===
def capture_pf():
    from Localization.pf_localizer import main
    main()
    save_gif("particle_filter", duration=80)

# === ICP ===
def capture_icp():
    from SLAM.icp_matching import main
    main()
    save_gif("icp", duration=100)

# === Fusion ===
def capture_fusion():
    from Localization.fusion_localizer import main
    main()
    save_gif("fusion", duration=80)

# === Fuzzy ===
def capture_fuzzy():
    from PathTracking.fuzzy_controller import main
    main()
    save_gif("fuzzy", duration=80)

CAPTURES = {
    "dwa": capture_dwa,
    "fastslam": capture_fastslam,
    "ekf": capture_ekf,
    "stanley": capture_stanley,
    "pure_pursuit": capture_pure_pursuit,
    "mpc": capture_mpc,
    "particle_filter": capture_pf,
    "icp": capture_icp,
    "fusion": capture_fusion,
    "fuzzy": capture_fuzzy,
}

if __name__ == "__main__":
    import sys
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(CAPTURES.keys())
    for name in targets:
        if name not in CAPTURES:
            print(f"Unknown module: {name}")
            continue
        print(f"Capturing {name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            CAPTURES[name]()
            print(f"OK ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
        _frames.clear()
