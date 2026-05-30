import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import math
import json

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "generated"))
sys.path.insert(0, _PROJECT_ROOT)

FIGS_DIR = os.path.join(_PROJECT_ROOT, "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "axes.grid": True,
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#a0a0a0",
    "ytick.color": "#a0a0a0",
})

COLORS = {
    "track": "#4a90d9", "vehicle": "#ff6b35", "trail": "#ffd166",
    "obstacle": "#e63946", "intersection": "#06d6a0", "wet": "#48cae4",
    "rough": "#f4a261", "normal": "#52b788", "speed": "#ff6b35",
    "curvature": "#e63946", "heading": "#06d6a0",
}

TOTAL_FRAMES = 600
FPS = 5
TOTAL_TIME_S = TOTAL_FRAMES / FPS


def load_or_generate_track():
    json_path = os.path.join(_PROJECT_ROOT, "racing_track_design.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            design = json.load(f)
        return (np.array(design["track"]["x"]), np.array(design["track"]["y"]),
                np.array(design["track"]["yaw"]), np.array(design["track"]["curvature"]),
                np.array(design["track"]["surface_friction"]),
                design["track"]["surface_type"],
                design.get("intersections", []), design.get("obstacles", []))
    from racing_benchmark import (generate_racing_track, generate_intersections,
                                   generate_obstacles, generate_surface_conditions)
    tx, ty, tyaw, curv, stypes = generate_racing_track()
    inter = generate_intersections()
    obs = generate_obstacles()
    surf, stype, hv = generate_surface_conditions(tx, ty)
    return tx, ty, tyaw, curv, surf, stype, inter, obs


def compute_cum_dist(x, y):
    return np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))])


def sample_track_positions(track_x, track_y, track_yaw, curvature, surface,
                            surface_type, n_frames):
    cum_d = compute_cum_dist(track_x, track_y)
    total_len = cum_d[-1]
    target_d = np.linspace(0, total_len, n_frames + 1)[:-1]

    idx_x = np.interp(target_d, cum_d, track_x)
    idx_y = np.interp(target_d, cum_d, track_y)
    idx_yaw = np.interp(target_d, cum_d, track_yaw)
    idx_curv = np.interp(target_d, cum_d, curvature)
    idx_surf = np.interp(target_d, cum_d, surface)

    idx_stype = []
    for d in target_d:
        i = int(np.searchsorted(cum_d, d)) - 1
        i = max(0, min(i, len(surface_type) - 1))
        idx_stype.append(surface_type[i])

    return idx_x, idx_y, idx_yaw, idx_curv, idx_surf, idx_stype, target_d


def get_phase_label(dist_ratio, stype):
    if dist_ratio < 0.10:
        return "PHASE 1: Start Straight", "#52b788"
    elif dist_ratio < 0.25:
        return "PHASE 2: Gentle Curve Entry", "#4a90d9"
    elif stype == "wet":
        return "PHASE 3: Wet Surface Zone", "#48cae4"
    elif dist_ratio < 0.50:
        return "PHASE 4: Sharp Turn + Obstacle", "#e63946"
    elif stype == "rough":
        return "PHASE 5: Rough Surface Zone", "#f4a261"
    elif dist_ratio < 0.85:
        return "PHASE 6: Intersection Passage", "#06d6a0"
    else:
        return "PHASE 7: Final Sprint", "#ff6b35"


def make_vehicle_artists(ax, color=COLORS["vehicle"]):
    body = plt.Polygon(np.zeros((4, 2)), closed=True, facecolor=color,
                        edgecolor='white', linewidth=0.8, alpha=0.9, zorder=10)
    ax.add_patch(body)
    arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
                         zorder=12)
    return body, arrow


def update_vehicle(body, arrow, x, y, yaw, vlen=4.0, vwid=2.0):
    c, s = math.cos(yaw), math.sin(yaw)
    Rot = np.array([[c, s], [-s, c]])
    hl, hw = vlen / 2, vwid / 2
    corners = np.array([[-hl, hl, hl, -hl], [hw, hw, -hw, -hw]])
    corners = Rot @ corners
    corners[0] += x
    corners[1] += y
    body.set_xy(corners.T)

    arrow_len = vlen * 0.6
    arrow.xy = (x + arrow_len * c, y + arrow_len * s)
    arrow.xyann = (x, y)


def draw_obstacle(ax, obs):
    if obs["type"] == "static_vehicle":
        rect = patches.FancyBboxPatch(
            (obs["x"] - obs["length"]/2, obs["y"] - obs["width"]/2),
            obs["length"], obs["width"],
            boxstyle="round,pad=0.3", facecolor=COLORS["obstacle"],
            alpha=0.7, edgecolor='white', linewidth=0.8, zorder=5)
        ax.add_patch(rect)
        ax.text(obs["x"], obs["y"] + obs["width"]/2 + 3, "Vehicle",
                ha='center', fontsize=7, color=COLORS["obstacle"], zorder=6)
    elif obs["type"] == "construction":
        rect = patches.FancyBboxPatch(
            (obs["x"] - obs["length"]/2, obs["y"] - obs["width"]/2),
            obs["length"], obs["width"],
            boxstyle="round,pad=0.5", facecolor="#f4a261",
            alpha=0.5, edgecolor='#e76f51', linewidth=1.0,
            linestyle='--', zorder=5)
        ax.add_patch(rect)
        ax.text(obs["x"], obs["y"] + obs["width"]/2 + 3, "Construction",
                ha='center', fontsize=7, color="#f4a261", zorder=6)


def draw_intersection(ax, inter):
    marker_size = 12 if inter["type"] == "four_way" else 10
    shape = 's' if inter["type"] == "four_way" else '^'
    ax.plot(inter["x"], inter["y"], shape, color=COLORS["intersection"],
            markersize=marker_size, alpha=0.7, zorder=4)
    label = "4-Way" if inter["type"] == "four_way" else "T-Junc"
    ax.text(inter["x"], inter["y"] + 8, label,
            ha='center', fontsize=6, color=COLORS["intersection"], zorder=4)


def main():
    print("Loading track data...", flush=True)
    (track_x, track_y, track_yaw, curvature, surface,
     surface_type, intersections, obstacles) = load_or_generate_track()

    cum_d = compute_cum_dist(track_x, track_y)
    total_len = cum_d[-1]

    print(f"Track: {len(track_x)} pts, {total_len:.0f}m total", flush=True)
    print(f"Animation: {TOTAL_FRAMES} frames @ {FPS}fps = {TOTAL_TIME_S:.0f}s", flush=True)

    pos_x, pos_y, pos_yaw, pos_curv, pos_surf, pos_stype, pos_dist = \
        sample_track_positions(track_x, track_y, track_yaw, curvature,
                               surface, surface_type, TOTAL_FRAMES)

    fig = plt.figure(figsize=(16, 9), facecolor="#1a1a2e")

    ax_main = fig.add_axes([0.05, 0.15, 0.62, 0.78])
    ax_speed = fig.add_axes([0.72, 0.70, 0.25, 0.22])
    ax_curv = fig.add_axes([0.72, 0.40, 0.25, 0.22])
    ax_info = fig.add_axes([0.72, 0.05, 0.25, 0.30])
    ax_progress = fig.add_axes([0.05, 0.03, 0.62, 0.06])

    for ax in [ax_main, ax_speed, ax_curv, ax_info, ax_progress]:
        ax.set_facecolor("#16213e")

    ax_main.set_aspect('equal')
    ax_main.set_xlim(track_x.min() - 30, track_x.max() + 30)
    ax_main.set_ylim(track_y.min() - 50, track_y.max() + 50)
    ax_main.set_xlabel("x [m]")
    ax_main.set_ylabel("y [m]")

    friction_colors = {"normal": COLORS["normal"], "wet": COLORS["wet"],
                       "rough": COLORS["rough"]}
    for stype in ["normal", "wet", "rough"]:
        mask = np.array([s == stype for s in surface_type])
        if mask.any():
            ax_main.scatter(track_x[mask], track_y[mask], c=friction_colors[stype],
                           s=0.3, alpha=0.4, zorder=1)

    ax_main.plot(track_x, track_y, '-', color=COLORS["track"], linewidth=1.5,
                 alpha=0.6, zorder=2)

    for obs in obstacles:
        draw_obstacle(ax_main, obs)
    for inter in intersections:
        draw_intersection(ax_main, inter)

    trail_line, = ax_main.plot([], [], '-', color=COLORS["trail"], linewidth=2.0,
                                alpha=0.8, zorder=8)
    vehicle_body, vehicle_arrow = make_vehicle_artists(ax_main)

    ax_speed.set_xlim(0, TOTAL_FRAMES)
    ax_speed.set_ylim(0, 8)
    ax_speed.set_ylabel("Speed [m/s]")
    ax_speed.set_title("Speed Profile", fontsize=9, color="#e0e0e0")
    speed_line, = ax_speed.plot([], [], '-', color=COLORS["speed"], linewidth=1.2)

    ax_curv.set_xlim(0, TOTAL_FRAMES)
    ax_curv.set_ylim(0, max(curvature) * 1.2 + 0.001)
    ax_curv.set_ylabel("Curvature [1/m]")
    ax_curv.set_title("Curvature", fontsize=9, color="#e0e0e0")
    curv_line, = ax_curv.plot([], [], '-', color=COLORS["curvature"], linewidth=1.0)
    curv_marker, = ax_curv.plot([], [], 'o', color='white', markersize=4, zorder=5)

    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis('off')
    info_texts = {}
    info_labels = [
        ("phase", 0.5, 0.88, 11, "bold"),
        ("dist", 0.5, 0.72, 10, "normal"),
        ("surface", 0.5, 0.56, 10, "normal"),
        ("friction", 0.5, 0.40, 10, "normal"),
        ("steer", 0.5, 0.24, 10, "normal"),
        ("time", 0.5, 0.08, 10, "normal"),
    ]
    for key, x, y, fs, fw in info_labels:
        info_texts[key] = ax_info.text(x, y, "", ha='center', va='center',
                                        fontsize=fs, fontweight=fw, color="#e0e0e0")

    ax_progress.set_xlim(0, 1)
    ax_progress.set_ylim(0, 1)
    ax_progress.axis('off')
    progress_bg = patches.Rectangle((0, 0.2), 1, 0.6, facecolor="#2a2a4a",
                                     edgecolor="#4a4a6a", linewidth=0.5)
    ax_progress.add_patch(progress_bg)
    progress_fill = patches.Rectangle((0, 0.2), 0, 0.6, facecolor=COLORS["speed"],
                                       alpha=0.8)
    ax_progress.add_patch(progress_fill)
    progress_text = ax_progress.text(0.5, 0.5, "0%", ha='center', va='center',
                                      fontsize=9, color='white', fontweight='bold')

    title_text = fig.text(0.36, 0.96, "Intelligent Vehicle Racing Track — Live Demo",
                          ha='center', fontsize=14, fontweight='bold', color="#ffd166")

    speed_history = []
    curv_history = []
    trail_x, trail_y = [], []

    def init():
        trail_line.set_data([], [])
        speed_line.set_data([], [])
        curv_line.set_data([], [])
        curv_marker.set_data([], [])
        return []

    def update(frame):
        nonlocal speed_history, curv_history, trail_x, trail_y

        x = pos_x[frame]
        y = pos_y[frame]
        yaw = pos_yaw[frame]
        curv_val = pos_curv[frame]
        surf_val = pos_surf[frame]
        stype = pos_stype[frame]
        dist_val = pos_dist[frame]
        dist_ratio = dist_val / total_len

        if frame > 0:
            dx = pos_x[frame] - pos_x[frame - 1]
            dy = pos_y[frame] - pos_y[frame - 1]
            dt = TOTAL_TIME_S / TOTAL_FRAMES
            speed = math.hypot(dx, dy) / dt
        else:
            speed = 0.0

        speed_history.append(speed)
        curv_history.append(curv_val)
        trail_x.append(x)
        trail_y.append(y)

        trail_line.set_data(trail_x, trail_y)

        update_vehicle(vehicle_body, vehicle_arrow, x, y, yaw)

        view_w, view_h = 200, 120
        cx = x + view_w * 0.3 * math.cos(yaw)
        cy = y + view_w * 0.3 * math.sin(yaw)
        ax_main.set_xlim(cx - view_w / 2, cx + view_w / 2)
        ax_main.set_ylim(cy - view_h / 2, cy + view_h / 2)

        frames_arr = np.arange(len(speed_history))
        speed_line.set_data(frames_arr, speed_history)
        curv_line.set_data(frames_arr, curv_history)
        curv_marker.set_data([frame], [curv_val])

        phase_label, phase_color = get_phase_label(dist_ratio, stype)
        info_texts["phase"].set_text(phase_label)
        info_texts["phase"].set_color(phase_color)
        info_texts["dist"].set_text(f"Distance: {dist_val:.0f} / {total_len:.0f} m")
        info_texts["surface"].set_text(f"Surface: {stype.upper()}")
        info_texts["surface"].set_color(friction_colors.get(stype, "#e0e0e0"))
        info_texts["friction"].set_text(f"Friction: mu = {surf_val:.2f}")
        info_texts["steer"].set_text(f"Curvature: {curv_val:.5f} 1/m")
        elapsed = frame / FPS
        info_texts["time"].set_text(f"Time: {elapsed:.1f}s / {TOTAL_TIME_S:.0f}s")

        progress_fill.set_width(dist_ratio)
        progress_text.set_text(f"{dist_ratio*100:.0f}%")

        if frame % 50 == 0:
            print(f"  Frame {frame}/{TOTAL_FRAMES} ({dist_ratio*100:.0f}%)", flush=True)

        return []

    print("Starting animation...", flush=True)
    for frame_idx in range(TOTAL_FRAMES):
        update(frame_idx)
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{TOTAL_FRAMES} ({frame_idx/TOTAL_FRAMES*100:.0f}%)", flush=True)

    save_path = os.path.join(FIGS_DIR, "racing_demo_final.png")
    fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"[Demo] Final frame saved: {save_path}")

    for phase_name, phase_color in [("Full Track", COLORS["track"])]:
        fig2, ax2 = plt.subplots(figsize=(16, 9), facecolor="#1a1a2e")
        ax2.set_facecolor("#16213e")
        ax2.set_aspect('equal')
        ax2.plot(track_x, track_y, '-', color=COLORS["track"], linewidth=2, alpha=0.8)
        ax2.plot(pos_x, pos_y, '-', color=COLORS["trail"], linewidth=1.5, alpha=0.6)
        for obs in obstacles:
            draw_obstacle(ax2, obs)
        for inter in intersections:
            draw_intersection(ax2, inter)
        ax2.set_xlabel("x [m]")
        ax2.set_ylabel("y [m]")
        ax2.set_title("Full Racing Track with Vehicle Trajectory", color="#e0e0e0", fontsize=13)
        overview_path = os.path.join(FIGS_DIR, "racing_demo_overview.png")
        fig2.savefig(overview_path, dpi=150, facecolor=fig2.get_facecolor())
        plt.close(fig2)
        print(f"[Demo] Overview saved: {overview_path}")
        break


if __name__ == "__main__":
    main()
