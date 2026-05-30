import sys
import os
import numpy as np
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "generated"))
sys.path.insert(0, _PROJECT_ROOT)

from generated import (PerceptionOutput, DecisionOutput, PathPoint,
                       TargetObject, Behavior, DecisionStatus, Header, Pose2D)
from decision.path_smoother import smooth_path
from decision.obstacle_avoidance import avoid_obstacles
from utils.geometry import compute_curvature
import matplotlib.pyplot as plt
from perception.lane_pixel_detector import generate_test_image, detect_lane_pixels
from perception.obstacle_detector import generate_test_point_cloud, detect_obstacles
from perception.sensor_fusion import fuse_to_perception, default_camera_params


TASK_PATROL = 0
TASK_AVOID = 1
TASK_PARK = 2
OBSTACLE_TYPE_PARKING_SIGN = 4

_TASK_NAMES = {TASK_PATROL: "PATROL", TASK_AVOID: "AVOID", TASK_PARK: "PARK"}


# === Phase 0: Lane Keep (inlined from lane_keeper) ===

def _assign_speed_profile(kappa, v_nominal, v_min, kappa_threshold):
    speed = np.full_like(kappa, v_nominal)
    mask = np.abs(kappa) > kappa_threshold
    speed[mask] = np.clip(
        v_nominal / (1.0 + np.abs(kappa[mask]) / kappa_threshold),
        v_min, v_nominal
    )
    return speed


def _analyze_obstacles(perception_output, v_nominal):
    has_ahead = False
    min_dist = float('inf')
    lead_id = 0
    for obs in perception_output.obstacles:
        if not (np.isfinite(obs.center.x) and np.isfinite(obs.center.y)):
            continue
        d = np.sqrt(obs.center.x**2 + obs.center.y**2)
        if obs.center.y > 0 and abs(obs.center.x) < 2.0:
            has_ahead = True
            if d < min_dist:
                min_dist = d
                lead_id = obs.id
    ttc = min_dist / (v_nominal + 1e-9)
    return has_ahead, min_dist, lead_id, ttc


def _lane_keep_decide(perception_output, v_nominal=5.0, v_min=1.0,
                      stop_dist_threshold=5.0, kappa_threshold=0.1,
                      ttc_threshold=3.0, global_route=None):
    decision = DecisionOutput()
    decision.header.timestamp_ns = int(time.time() * 1e9)
    decision.header.seq = perception_output.header.seq
    decision.header.frame_id = "decision"

    if not perception_output.road_valid:
        decision.behavior = Behavior.Value("BEHAVIOR_EMERGENCY_STOP")
        decision.status = DecisionStatus.Value("DECISION_EMERGENCY")
        decision.target_speed = 0.0
        decision.stop_distance = 0.0
        return decision

    n_center = len(perception_output.road.center_line)
    if n_center < 3:
        if global_route is not None and len(global_route) >= 3:
            cx_percep = np.array([p[0] for p in global_route])
            cy_percep = np.array([p[1] for p in global_route])
        else:
            decision.behavior = Behavior.Value("BEHAVIOR_EMERGENCY_STOP")
            decision.status = DecisionStatus.Value("DECISION_EMERGENCY")
            decision.target_speed = 0.0
            decision.stop_distance = 0.0
            return decision
    else:
        cx_percep = np.array([p.x for p in perception_output.road.center_line])
        cy_percep = np.array([p.y for p in perception_output.road.center_line])

    if global_route is not None and len(global_route) >= 3 and n_center >= 3:
        last_percep = np.array([cx_percep[-1], cy_percep[-1]])
        gr_arr = np.array(global_route)
        dists = np.sqrt((gr_arr[:, 0] - last_percep[0])**2 + (gr_arr[:, 1] - last_percep[1])**2)
        attach_idx = np.argmin(dists)
        if attach_idx < len(gr_arr) - 2:
            n_append = min(len(gr_arr) - attach_idx - 1, 30)
            cx_percep = np.concatenate([cx_percep, gr_arr[attach_idx + 1:attach_idx + 1 + n_append, 0]])
            cy_percep = np.concatenate([cy_percep, gr_arr[attach_idx + 1:attach_idx + 1 + n_append, 1]])

    if len(cx_percep) > 1 and cx_percep[0] > cx_percep[-1]:
        cx_percep = cx_percep[::-1]
        cy_percep = cy_percep[::-1]

    kappa = compute_curvature(cx_percep, cy_percep)
    speed_profile = _assign_speed_profile(kappa, v_nominal, v_min, kappa_threshold)

    ds = np.sqrt(np.diff(cx_percep)**2 + np.diff(cy_percep)**2)
    s_cum = np.concatenate([[0.0], np.cumsum(ds)])
    t_rel = np.zeros_like(s_cum)
    t_rel[1:] = np.cumsum(ds / (speed_profile[1:] + 1e-9))

    n_path = len(cx_percep)
    for i in range(n_path):
        pp = decision.target_path.add()
        pp.pose.x = float(cx_percep[i])
        pp.pose.y = float(cy_percep[i])
        i_lo = max(i - 1, 0)
        i_hi = min(i + 1, n_path - 1)
        pp.pose.theta = float(np.arctan2(cy_percep[i_hi] - cy_percep[i_lo],
                                          cx_percep[i_hi] - cx_percep[i_lo]))
        pp.curvature = float(kappa[i])
        pp.speed = float(speed_profile[i])
        pp.acceleration = 0.0
        pp.relative_time = float(t_rel[i])

    has_ahead, min_dist, lead_id, ttc = _analyze_obstacles(perception_output, v_nominal)

    if has_ahead and (min_dist < stop_dist_threshold or ttc < ttc_threshold):
        decision.behavior = Behavior.Value("BEHAVIOR_STOP")
        decision.status = DecisionStatus.Value("DECISION_NORMAL")
        decision.target_speed = 0.0
        decision.stop_distance = min_dist
        decision.lead_object.obstacle_id = lead_id
        decision.lead_object.distance = min_dist
        decision.lead_object.is_lead = True
        decision.lead_object.time_to_collision = ttc
    elif has_ahead and (min_dist < stop_dist_threshold * 3 or ttc < ttc_threshold * 3):
        decision.behavior = Behavior.Value("BEHAVIOR_FOLLOW")
        decision.status = DecisionStatus.Value("DECISION_NORMAL")
        decision.target_speed = max(v_nominal * 0.5, v_min)
        decision.stop_distance = min_dist
        decision.lead_object.obstacle_id = lead_id
        decision.lead_object.distance = min_dist
        decision.lead_object.is_lead = True
        decision.lead_object.time_to_collision = ttc
    else:
        decision.behavior = Behavior.Value("BEHAVIOR_LANE_KEEP")
        decision.status = DecisionStatus.Value("DECISION_NORMAL")
        decision.target_speed = v_nominal

    decision.target_lane = 0
    decision.has_target_lane = True
    decision.has_target_speed = True
    decision.path_length = float(s_cum[-1])

    return decision

def _detect_park_trigger(perception_output):
    for obs in perception_output.obstacles:
        if hasattr(obs, "type") and obs.type == OBSTACLE_TYPE_PARKING_SIGN:
            d = np.sqrt(obs.center.x**2 + obs.center.y**2)
            if d < 5.0:
                return True
    return False


def _detect_avoid_trigger(perception_output, trigger_dist):
    for obs in perception_output.obstacles:
        d = np.sqrt(obs.center.x**2 + obs.center.y**2)
        if obs.center.x > 0 and d < trigger_dist:
            return True
    return False


# === Phase 2: Task Implementations ===

def _task_patrol(perception_output, task_state, v_nominal, global_route=None):
    return _lane_keep_decide(perception_output, v_nominal=v_nominal, global_route=global_route), task_state


def _task_avoid(perception_output, task_state, v_nominal, avoid_trigger_dist=8.0):
    if not perception_output.road_valid or len(perception_output.road.center_line) < 3:
        decision = DecisionOutput()
        decision.header.timestamp_ns = int(time.time() * 1e9)
        decision.header.frame_id = "decision"
        decision.behavior = Behavior.Value("BEHAVIOR_EMERGENCY_STOP")
        decision.status = DecisionStatus.Value("DECISION_EMERGENCY")
        decision.target_speed = 0.0
        return decision, task_state

    cx = np.array([p.x for p in perception_output.road.center_line])
    cy = np.array([p.y for p in perception_output.road.center_line])

    if len(cx) > 1 and cx[0] > cx[-1]:
        cx = cx[::-1]
        cy = cy[::-1]

    sx, sy, kappa = smooth_path(cx, cy, max_deviation=0.3, n_output=len(cx))

    obs_list = []
    for obs in perception_output.obstacles:
        obs_list.append({
            "center": np.array([obs.center.x, obs.center.y]),
            "length": obs.length,
            "width": obs.width,
            "heading": obs.heading,
        })

    new_x, new_y, beh = avoid_obstacles(sx, sy, obs_list)

    kappa_new = compute_curvature(new_x, new_y) if len(new_x) >= 3 else np.zeros(len(new_x))

    decision = DecisionOutput()
    decision.header.timestamp_ns = int(time.time() * 1e9)
    decision.header.seq = perception_output.header.seq
    decision.header.frame_id = "decision"

    n = len(new_x)
    ds = np.sqrt(np.diff(new_x)**2 + np.diff(new_y)**2)
    s_cum = np.concatenate([[0.0], np.cumsum(ds)])
    speed_profile = np.full(n, v_nominal)

    for i in range(n):
        pp = decision.target_path.add()
        pp.pose.x = float(new_x[i])
        pp.pose.y = float(new_y[i])
        i_lo = max(i - 1, 0)
        i_hi = min(i + 1, n - 1)
        pp.pose.theta = float(np.arctan2(new_y[i_hi] - new_y[i_lo],
                                          new_x[i_hi] - new_x[i_lo]))
        pp.curvature = float(kappa_new[i]) if i < len(kappa_new) else 0.0
        pp.speed = float(speed_profile[i])
        pp.acceleration = 0.0
        pp.relative_time = float(s_cum[i] / max(v_nominal, 1e-9))

    if beh == "stop":
        decision.behavior = Behavior.Value("BEHAVIOR_STOP")
        decision.target_speed = 0.0
    else:
        decision.behavior = Behavior.Value("BEHAVIOR_OVERTAKE")
        decision.target_speed = v_nominal

    decision.status = DecisionStatus.Value("DECISION_NORMAL")
    decision.path_length = float(s_cum[-1])
    decision.target_lane = 0
    decision.has_target_lane = True
    decision.has_target_speed = True

    has_obstacle_ahead = any(
        obs.center.x > 0 and np.sqrt(obs.center.x**2 + obs.center.y**2) < avoid_trigger_dist + 3.0
        for obs in perception_output.obstacles
        if np.isfinite(obs.center.x) and np.isfinite(obs.center.y)
    )
    task_state["obstacle_passed"] = not has_obstacle_ahead
    return decision, task_state


def _task_park(perception_output, task_state, v_nominal):
    decision = DecisionOutput()
    decision.header.timestamp_ns = int(time.time() * 1e9)
    decision.header.seq = perception_output.header.seq
    decision.header.frame_id = "decision"
    decision.behavior = Behavior.Value("BEHAVIOR_PARK")
    decision.status = DecisionStatus.Value("DECISION_NORMAL")
    decision.target_speed = 0.0
    decision.stop_distance = 0.0

    if perception_output.road_valid and len(perception_output.road.center_line) > 0:
        cx = np.array([p.x for p in perception_output.road.center_line])
        cy = np.array([p.y for p in perception_output.road.center_line])
        ctheta = np.arctan2(np.gradient(cy), np.gradient(cx))
        for i in range(min(len(cx), 10)):
            pp = decision.target_path.add()
            pp.pose.x = float(cx[i])
            pp.pose.y = float(cy[i])
            pp.pose.theta = float(ctheta[i])
            pp.speed = 0.5
            pp.curvature = 0.0
            pp.acceleration = -1.0
            pp.relative_time = float(i * 0.1)

    task_state["park_complete"] = task_state.get("park_steps", 0) > 50
    task_state["park_steps"] = task_state.get("park_steps", 0) + 1
    return decision, task_state


# === Phase 3: Scheduler ===

def schedule(perception_output, scheduler_state, v_nominal=5.0,
             avoid_trigger_dist=8.0, global_route=None):
    current_task = scheduler_state.get("current_task", TASK_PATROL)
    task_state = scheduler_state.get("task_state", {})

    park_trigger = _detect_park_trigger(perception_output)
    avoid_trigger = _detect_avoid_trigger(perception_output, avoid_trigger_dist)

    new_task = current_task

    if current_task == TASK_AVOID:
        if task_state.get("obstacle_passed", False):
            new_task = TASK_PATROL
            task_state = {}
    elif current_task == TASK_PARK:
        if avoid_trigger:
            new_task = TASK_AVOID
            task_state = {}
        elif task_state.get("park_complete", False):
            new_task = TASK_PATROL
            task_state = {}
    else:
        if avoid_trigger:
            new_task = TASK_AVOID
        elif park_trigger:
            new_task = TASK_PARK

    if new_task > current_task:
        task_state = {}

    current_task = new_task
    scheduler_state["current_task"] = current_task
    scheduler_state["task_state"] = task_state

    v_effective = v_nominal

    if current_task == TASK_PATROL:
        decision, task_state = _task_patrol(perception_output, task_state, v_effective, global_route=global_route)
    elif current_task == TASK_AVOID:
        decision, task_state = _task_avoid(perception_output, task_state, v_effective, avoid_trigger_dist)
    elif current_task == TASK_PARK:
        decision, task_state = _task_park(perception_output, task_state, v_effective)
    else:
        decision, task_state = _task_patrol(perception_output, task_state, v_effective, global_route=global_route)

    scheduler_state["task_state"] = task_state
    scheduler_state["task_name"] = _TASK_NAMES.get(current_task, "UNKNOWN")
    return decision, scheduler_state


# === Phase 4: Test ===

if __name__ == "__main__":
    show_animation = True

    img = generate_test_image()
    pc = generate_test_point_cloud()
    K, R, t = default_camera_params()

    rows, cx = detect_lane_pixels(img)
    obs = detect_obstacles(pc)
    perc = fuse_to_perception(rows, cx, obs, [], K, R, t)

    state = {"current_task": TASK_PATROL, "task_state": {}}
    dec, state = schedule(perc, state, v_nominal=5.0)
    print(f"task={state['task_name']}  behavior={Behavior.Name(dec.behavior)}  "
          f"path_pts={len(dec.target_path)}  speed={dec.target_speed:.1f}")

    if show_animation:
        n_steps = 60
        task_history = []
        speed_history = []
        behavior_history = []

        state = {"current_task": TASK_PATROL, "task_state": {}}
        rng = np.random.default_rng(42)

        for step in range(n_steps):
            perc_step = fuse_to_perception(rows, cx, [], [], K, R, t)

            if 15 <= step <= 35:
                extra = PerceptionOutput()
                extra.header.seq = step
                extra.header.timestamp_ns = int(time.time() * 1e9)
                extra.header.frame_id = "perception"
                extra.road_valid = True
                for r, x in zip(rows, cx):
                    if not np.isnan(x):
                        pp = extra.road.center_line.add()
                        pp.x = float(x)
                        pp.y = float(r)
                o = extra.obstacles.add()
                o.id = 100
                o.center.x = 5.0
                o.center.y = 0.5
                o.length = 1.0
                o.width = 0.8
                o.heading = 0.0
                perc_step = extra

            dec, state = schedule(perc_step, state, v_nominal=5.0)
            task_history.append(state["current_task"])
            speed_history.append(dec.target_speed)
            behavior_history.append(dec.behavior)

        task_names_arr = np.array([_TASK_NAMES.get(t, "?") for t in task_history])
        task_colors = {"PATROL": "green", "AVOID": "orange", "PARK": "red"}
        color_arr = [task_colors.get(n, "gray") for n in task_names_arr]

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        ax1 = axes[0]
        for i in range(n_steps):
            ax1.bar(i, 1, color=color_arr[i], edgecolor='none')
        ax1.set_title("Task Scheduler State Machine")
        ax1.set_xlabel("step")
        ax1.set_ylabel("task")
        ax1.set_yticks([])
        patches = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in task_colors.values()]
        ax1.legend(patches, task_colors.keys(), frameon=True, fancybox=True)
        ax1.grid(True, axis='x')

        ax2 = axes[1]
        ax2.plot(speed_history, '-b', linewidth=1.5)
        ax2.set_title("Target Speed Profile")
        ax2.set_xlabel("step")
        ax2.set_ylabel("speed[m/s]")
        ax2.grid(True)

        ax3 = axes[2]
        beh_names = []
        beh_map = {}
        beh_idx = 0
        beh_int = []
        for b in behavior_history:
            name = Behavior.Name(b)
            if name not in beh_map:
                beh_map[name] = beh_idx
                beh_idx += 1
            beh_int.append(beh_map[name])
        ax3.step(range(n_steps), beh_int, '-r', where='mid', linewidth=1.5)
        ax3.set_yticks(list(beh_map.values()))
        ax3.set_yticklabels(list(beh_map.keys()))
        ax3.set_title("Behavior Output")
        ax3.set_xlabel("step")
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

        state_diagram = {
            "PATROL": {"AVOID": "obstacle ahead", "PARK": "parking sign"},
            "AVOID": {"PATROL": "obstacle passed"},
            "PARK": {"PATROL": "parking complete"},
        }
        pos = {"PATROL": (0.3, 0.5), "AVOID": (0.7, 0.8), "PARK": (0.7, 0.2)}

        fig2, ax_d = plt.subplots(figsize=(8, 6))
        for name, (x, y) in pos.items():
            circle = plt.Circle((x, y), 0.08, color=task_colors[name], alpha=0.7)
            ax_d.add_patch(circle)
            ax_d.text(x, y, name, ha='center', va='center', fontsize=11, fontweight='bold')

        for src, transitions in state_diagram.items():
            for dst, label in transitions.items():
                sx, sy = pos[src]
                dx, dy = pos[dst]
                ax_d.annotate("", xy=(dx, dy), xytext=(sx, sy),
                              arrowprops=dict(arrowstyle="->", color='black', lw=1.5))
                mx, my = (sx + dx) / 2, (sy + dy) / 2
                ax_d.text(mx, my + 0.03, label, ha='center', fontsize=8,
                          bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', alpha=0.8))

        ax_d.set_xlim(0, 1)
        ax_d.set_ylim(0, 1)
        ax_d.set_title("Task Scheduler State Diagram")
        ax_d.set_aspect('equal')
        ax_d.axis('off')
        plt.tight_layout()
        plt.show()
