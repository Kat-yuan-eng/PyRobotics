import sys
import os
import numpy as np
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "generated"))
sys.path.insert(0, _PROJECT_ROOT)

from generated import AgentState
from decision.task_scheduler import schedule


# === Phase 1: Agent State to Moving Obstacle ===

def _agent_to_obstacle(agent, current_time):
    if len(agent.planned_path) < 2:
        if hasattr(agent, 'pose') and (abs(agent.pose.x) > 0.01 or abs(agent.pose.y) > 0.01):
            return {
                "center": np.array([agent.pose.x, agent.pose.y]),
                "length": getattr(agent, 'occupied_length', 4.5),
                "width": getattr(agent, 'occupied_width', 2.0),
                "heading": agent.pose.theta,
            }
        return None

    path_x = np.array([p.x for p in agent.planned_path])
    path_y = np.array([p.y for p in agent.planned_path])

    ds = np.sqrt(np.diff(path_x)**2 + np.diff(path_y)**2)
    s_cum = np.concatenate([[0.0], np.cumsum(ds)])

    dt = max(min(current_time - agent.timestamp, 10.0), 0.0)
    speed_clamped = min(agent.planned_speed, 50.0)
    s_now = speed_clamped * dt
    s_now = min(s_now, s_cum[-1])

    idx = int(np.searchsorted(s_cum, s_now, side="right")) - 1
    idx = max(0, min(idx, len(path_x) - 2))
    frac = (s_now - s_cum[idx]) / max(s_cum[idx + 1] - s_cum[idx], 1e-12)
    frac = np.clip(frac, 0.0, 1.0)

    cx = path_x[idx] + frac * (path_x[idx + 1] - path_x[idx])
    cy = path_y[idx] + frac * (path_y[idx + 1] - path_y[idx])

    i_lo = max(idx - 1, 0)
    i_hi = min(idx + 1, len(path_x) - 1)
    heading = np.arctan2(path_y[i_hi] - path_y[i_lo], path_x[i_hi] - path_x[i_lo])

    return {
        "center": np.array([cx, cy]),
        "length": agent.occupied_length,
        "width": agent.occupied_width,
        "heading": heading,
    }


# === Phase 2: Build Ego AgentState ===

def build_ego_state(ego_id, decision_output, x_actual, y_actual, theta_actual,
                    v_actual, occupied_length=4.5, occupied_width=2.0):
    agent = AgentState()
    agent.agent_id = ego_id
    agent.pose.x = x_actual
    agent.pose.y = y_actual
    agent.pose.theta = theta_actual
    agent.planned_speed = v_actual
    agent.occupied_length = occupied_length
    agent.occupied_width = occupied_width
    agent.occupied_center.x = x_actual
    agent.occupied_center.y = y_actual
    agent.timestamp = time.time()

    for pp in decision_output.target_path:
        p = agent.planned_path.add()
        p.x = pp.pose.x
        p.y = pp.pose.y

    return agent


# === Phase 3: Coordinate Avoidance ===

def coordinate_avoidance(perception_output, agent_states, ego_id,
                         scheduler_state, v_nominal=5.0):
    current_time = time.time()

    moving_obstacles = []
    for agent in agent_states:
        if agent.agent_id == ego_id:
            continue
        obs = _agent_to_obstacle(agent, current_time)
        if obs is not None:
            moving_obstacles.append(obs)

    from generated import PerceptionOutput as PO, Obstacle, ObstacleType, ObstacleShape
    merged_perception = PO()
    merged_perception.CopyFrom(perception_output)

    existing_ids = [obs.id for obs in perception_output.obstacles if obs.id > 0]
    id_offset = max(existing_ids, default=0) + 1
    for i, mobs in enumerate(moving_obstacles):
        o = merged_perception.obstacles.add()
        o.id = id_offset + i
        o.type = ObstacleType.Value("OBSTACLE_DYNAMIC")
        o.shape = ObstacleShape.Value("SHAPE_RECT")
        o.center.x = float(mobs["center"][0])
        o.center.y = float(mobs["center"][1])
        o.length = float(mobs["length"])
        o.width = float(mobs["width"])
        o.heading = float(mobs["heading"])
        o.confidence = 0.7

    merged_perception.obstacle_count = len(merged_perception.obstacles)

    return schedule(merged_perception, scheduler_state, v_nominal=v_nominal)


# === Phase 4: Test ===

def _make_test_agent(agent_id, cx, cy, speed=3.0):
    agent = AgentState()
    agent.agent_id = agent_id
    agent.pose.x = cx
    agent.pose.y = cy
    agent.pose.theta = 0.0
    agent.planned_speed = speed
    agent.occupied_length = 4.5
    agent.occupied_width = 2.0
    agent.occupied_center.x = cx
    agent.occupied_center.y = cy
    agent.timestamp = time.time()

    for i in range(10):
        p = agent.planned_path.add()
        p.x = cx + i * 0.5
        p.y = cy

    return agent


if __name__ == "__main__":
    from perception.lane_pixel_detector import detect_lane_pixels, generate_test_image
    from perception.sensor_fusion import fuse_to_perception, default_camera_params

    img = generate_test_image()
    K, R, t = default_camera_params()
    rows, cx = detect_lane_pixels(img)
    perc = fuse_to_perception(rows, cx, [], [], K, R, t)

    teammate = _make_test_agent(2, 5.0, 1.5, speed=3.0)
    state = {"current_task": 0, "task_state": {}}
    dec, state = coordinate_avoidance(perc, [teammate], ego_id=1, scheduler_state=state)
    print(f"task={state['task_name']}  behavior={dec.behavior}  "
          f"path_pts={len(dec.target_path)}")
