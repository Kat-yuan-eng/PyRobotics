import sys
import os
import numpy as np
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# === Phase 1: Project Obstacle onto Path ===

def _project_obstacle(path_x, path_y, obs_center):
    dx = path_x - obs_center[0]
    dy = path_y - obs_center[1]
    dist_sq = dx**2 + dy**2
    idx = int(np.argmin(dist_sq))

    ds = np.sqrt(np.diff(path_x)**2 + np.diff(path_y)**2)
    s_cum = np.concatenate([[0.0], np.cumsum(ds)])
    return idx, s_cum[idx], dist_sq[idx]**0.5


# === Phase 2: Path Normal ===

def _path_normals(path_x, path_y):
    dx = np.gradient(path_x)
    dy = np.gradient(path_y)
    length = np.sqrt(dx**2 + dy**2) + 1e-12
    nx = -dy / length
    ny = dx / length
    return nx, ny


# === Phase 3: Check Lateral Clearance ===

def _check_clearance(path_x, path_y, nx, ny, obs_list, offset, margin, skip_center=None):
    shifted_x = path_x + offset * nx
    shifted_y = path_y + offset * ny

    for obs in obs_list:
        if skip_center is not None and np.allclose(obs["center"], skip_center, atol=0.01):
            continue
        ox, oy = obs["center"]
        half_l = obs.get("length", 1.0) / 2.0 + margin
        half_w = obs.get("width", 0.5) / 2.0 + margin
        dx = shifted_x - ox
        dy = shifted_y - oy
        cos_h = np.cos(obs.get("heading", 0.0))
        sin_h = np.sin(obs.get("heading", 0.0))
        along = dx * cos_h + dy * sin_h
        across = -dx * sin_h + dy * cos_h
        collision = (np.abs(along) < half_l) & (np.abs(across) < half_w)
        if collision.any():
            return False
    return True


# === Phase 4: Sinusoidal Lateral Offset ===

def _apply_lateral_offset(path_x, path_y, nx, ny, s_cum,
                          s_start, s_end, offset):
    new_x = path_x.copy()
    new_y = path_y.copy()
    span = s_end - s_start
    if span < 1e-6:
        return new_x, new_y

    mask = (s_cum >= s_start) & (s_cum <= s_end)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return new_x, new_y

    t_norm = (s_cum[idx] - s_start) / span
    lateral = offset * np.sin(np.pi * t_norm)

    new_x[idx] += lateral * nx[idx]
    new_y[idx] += lateral * ny[idx]
    return new_x, new_y


# === Phase 5: Avoid Obstacles ===

def avoid_obstacles(path_x, path_y, obstacles, max_lateral=1.5,
                    safety_margin=0.5, priority="left"):
    if len(path_x) < 2:
        raise ValueError(f"need >=2 path pts, got {len(path_x)}")

    if len(obstacles) == 0:
        return path_x.copy(), path_y.copy(), "detour"

    nx, ny = _path_normals(path_x, path_y)
    ds = np.sqrt(np.diff(path_x)**2 + np.diff(path_y)**2)
    s_cum = np.concatenate([[0.0], np.cumsum(ds)])

    new_x = path_x.copy()
    new_y = path_y.copy()

    for obs in obstacles:
        if not np.all(np.isfinite(obs["center"])):
            continue
        if obs.get("width", 0.5) <= 0 or obs.get("length", 1.0) <= 0:
            continue
        idx_near, s_obs, d_obs = _project_obstacle(new_x, new_y, obs["center"])

        half_l = obs.get("length", 1.0) / 2.0 + safety_margin
        s_start = max(s_obs - half_l - max_lateral, 0.0)
        s_end = min(s_obs + half_l + max_lateral, s_cum[-1])

        dx_obs = obs["center"][0] - new_x[idx_near]
        dy_obs = obs["center"][1] - new_y[idx_near]
        d_lateral = dx_obs * nx[idx_near] + dy_obs * ny[idx_near]
        d_lateral_abs = abs(d_lateral)

        offset_needed = obs.get("width", 0.5) / 2.0 + safety_margin - d_lateral_abs
        if offset_needed <= 0:
            continue
        if offset_needed > max_lateral:
            return new_x, new_y, "stop"

        away_side = -1.0 if d_lateral > 0 else 1.0
        toward_side = -away_side

        away_ok = _check_clearance(new_x, new_y, nx, ny, obstacles, away_side * offset_needed, safety_margin, skip_center=obs["center"])
        toward_ok = _check_clearance(new_x, new_y, nx, ny, obstacles, toward_side * offset_needed, safety_margin, skip_center=obs["center"])

        if priority == "left":
            left_offset = offset_needed
            right_offset = -offset_needed
            if away_side > 0:
                if away_ok:
                    new_x, new_y = _apply_lateral_offset(
                        new_x, new_y, nx, ny, s_cum, s_start, s_end, left_offset)
                elif toward_ok:
                    new_x, new_y = _apply_lateral_offset(
                        new_x, new_y, nx, ny, s_cum, s_start, s_end, right_offset)
                else:
                    return new_x, new_y, "stop"
            else:
                if toward_ok:
                    new_x, new_y = _apply_lateral_offset(
                        new_x, new_y, nx, ny, s_cum, s_start, s_end, left_offset)
                elif away_ok:
                    new_x, new_y = _apply_lateral_offset(
                        new_x, new_y, nx, ny, s_cum, s_start, s_end, right_offset)
                else:
                    return new_x, new_y, "stop"
        else:
            right_offset = -offset_needed
            left_offset = offset_needed
            if away_side < 0:
                if away_ok:
                    new_x, new_y = _apply_lateral_offset(
                        new_x, new_y, nx, ny, s_cum, s_start, s_end, right_offset)
                elif toward_ok:
                    new_x, new_y = _apply_lateral_offset(
                        new_x, new_y, nx, ny, s_cum, s_start, s_end, left_offset)
                else:
                    return new_x, new_y, "stop"
            else:
                if toward_ok:
                    new_x, new_y = _apply_lateral_offset(
                        new_x, new_y, nx, ny, s_cum, s_start, s_end, right_offset)
                elif away_ok:
                    new_x, new_y = _apply_lateral_offset(
                        new_x, new_y, nx, ny, s_cum, s_start, s_end, left_offset)
                else:
                    return new_x, new_y, "stop"

    return new_x, new_y, "detour"


def _draw_obs_rect(ax, obs, color='k', linewidth=1.5):
    cx, cy = obs["center"]
    half_l = obs.get("length", 1.0) / 2.0
    half_w = obs.get("width", 0.5) / 2.0
    heading = obs.get("heading", 0.0)
    corners_local = np.array([
        [-half_l, -half_w], [half_l, -half_w],
        [half_l, half_w], [-half_l, half_w], [-half_l, -half_w]
    ])
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    corners_global = corners_local @ R.T + np.array([cx, cy])
    ax.plot(corners_global[:, 0], corners_global[:, 1], '-', color=color, linewidth=linewidth)
    ax.fill(corners_global[:-1, 0], corners_global[:-1, 1], color=color, alpha=0.2)


# === Phase 6: Test ===

if __name__ == "__main__":
    show_animation = True

    path_x = np.linspace(0, 20, 50)
    path_y = np.zeros(50)

    obstacles = [
        {"center": np.array([8.0, 0.0]), "length": 1.0, "width": 1.0, "heading": 0.0},
    ]

    new_x, new_y, beh = avoid_obstacles(path_x, path_y, obstacles)
    print(f"behavior: {beh}")
    max_offset = np.max(np.abs(new_y))
    print(f"max lateral offset: {max_offset:.3f}m (limit 1.5m)")

    obs2 = [
        {"center": np.array([8.0, 1.0]), "length": 1.0, "width": 1.0, "heading": 0.0},
        {"center": np.array([8.0, -1.0]), "length": 1.0, "width": 1.0, "heading": 0.0},
    ]
    _, _, beh2 = avoid_obstacles(path_x, path_y, obs2)
    print(f"blocked both sides: behavior={beh2} (expect stop)")

    if show_animation:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax1 = axes[0]
        ax1.plot(path_x, path_y, '.r', label='reference path')
        for obs in obstacles:
            _draw_obs_rect(ax1, obs)
        ax1.plot(new_x, new_y, '-b', label='avoidance path')
        ax1.set_title("Single Obstacle Avoidance")
        ax1.set_xlabel("x[m]")
        ax1.set_ylabel("y[m]")
        ax1.legend(frameon=True, fancybox=True)
        ax1.set_aspect('equal')
        ax1.grid(True)

        ax2 = axes[1]
        obs_multi = [
            {"center": np.array([6.0, 0.5]), "length": 1.0, "width": 0.8, "heading": 0.0},
            {"center": np.array([12.0, -0.3]), "length": 1.2, "width": 0.6, "heading": 0.3},
        ]
        new_x2, new_y2, beh2_full = avoid_obstacles(path_x, path_y, obs_multi)
        ax2.plot(path_x, path_y, '.r', label='reference path')
        for obs in obs_multi:
            _draw_obs_rect(ax2, obs)
        ax2.plot(new_x2, new_y2, '-b', label='avoidance path')
        ax2.set_title(f"Multi-Obstacle Avoidance ({beh2_full})")
        ax2.set_xlabel("x[m]")
        ax2.set_ylabel("y[m]")
        ax2.legend(frameon=True, fancybox=True)
        ax2.set_aspect('equal')
        ax2.grid(True)

        ax3 = axes[2]
        obs_blocked = [
            {"center": np.array([8.0, 1.0]), "length": 1.0, "width": 1.0, "heading": 0.0},
            {"center": np.array([8.0, -1.0]), "length": 1.0, "width": 1.0, "heading": 0.0},
        ]
        new_x3, new_y3, beh3 = avoid_obstacles(path_x, path_y, obs_blocked)
        ax3.plot(path_x, path_y, '.r', label='reference path')
        for obs in obs_blocked:
            _draw_obs_rect(ax3, obs)
        ax3.plot(new_x3, new_y3, '-b', label='avoidance path')
        ax3.set_title(f"Blocked Both Sides ({beh3})")
        ax3.set_xlabel("x[m]")
        ax3.set_ylabel("y[m]")
        ax3.legend(frameon=True, fancybox=True)
        ax3.set_aspect('equal')
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

        n_frames = 50
        path_x_anim = np.linspace(0, 30, 100)
        path_y_anim = np.zeros(100)
        obs_anim = [
            {"center": np.array([15.0, 0.0]), "length": 1.5, "width": 1.0, "heading": 0.0},
        ]
        new_x_a, new_y_a, _ = avoid_obstacles(path_x_anim, path_y_anim, obs_anim)

        plt.figure(figsize=(12, 5))
        for frame in range(n_frames):
            plt.cla()
            idx = min(frame * 2 + 1, len(path_x_anim) - 1)
            plt.plot(path_x_anim, path_y_anim, '.r', label='reference')
            plt.plot(new_x_a, new_y_a, '-b', label='avoidance')
            for obs in obs_anim:
                _draw_obs_rect(plt, obs)
            plt.plot(new_x_a[idx], new_y_a[idx], 'og', markersize=12, label='vehicle')
            plt.xlim(-1, 31)
            plt.ylim(-3, 3)
            plt.title(f"Obstacle Avoidance Animation  step={frame + 1}/{n_frames}")
            plt.xlabel("x[m]")
            plt.ylabel("y[m]")
            plt.legend(frameon=True, fancybox=True)
            plt.grid(True)
            plt.pause(0.05)
        plt.show()
