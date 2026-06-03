import sys
import os
import numpy as np
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "generated"))
sys.path.insert(0, _PROJECT_ROOT)

from generated import PerceptionOutput, Obstacle, ObstacleType, ObstacleShape, Header, Vec2


_OBSTACLE_TYPE_MAP = {
    "OBSTACLE_UNKNOWN": "OBSTACLE_UNKNOWN",
    "OBSTACLE_VEHICLE": "OBSTACLE_VEHICLE",
    "OBSTACLE_PEDESTRIAN": "OBSTACLE_PEDESTRIAN",
    "OBSTACLE_CYCLIST": "OBSTACLE_CYCLIST",
    "OBSTACLE_STATIC": "OBSTACLE_STATIC",
}


# === Phase 1: Pixel to Vehicle Coordinate ===

def _pixel_to_vehicle(u, v, K_inv, R, t):
    ray_cam = K_inv @ np.array([u, v, 1.0])
    ray_veh = R @ ray_cam
    if abs(ray_veh[2]) < 1e-9:
        return None
    depth = -t[2] / ray_veh[2]
    if depth < 0:
        return None
    p_veh = R @ (ray_cam * depth) + t
    return p_veh[:2]


# === Phase 2: Build RoadBoundary ===

def _build_road(output, center_veh, lane_width, vehicle_x=0.0, vehicle_y=0.0, vehicle_theta=0.0):
    cos_v = np.cos(vehicle_theta)
    sin_v = np.sin(vehicle_theta)
    for x_veh, y_veh in center_veh:
        x_g = x_veh * cos_v - y_veh * sin_v + vehicle_x
        y_g = x_veh * sin_v + y_veh * cos_v + vehicle_y
        p = output.road.center_line.add()
        p.x = float(x_g)
        p.y = float(y_g)

    half_w = lane_width / 2.0
    for x_veh, y_veh in center_veh:
        x_g = x_veh * cos_v - (y_veh + half_w) * sin_v + vehicle_x
        y_g = x_veh * sin_v + (y_veh + half_w) * cos_v + vehicle_y
        p = output.road.left_bound.add()
        p.x = float(x_g)
        p.y = float(y_g)
    for x_veh, y_veh in center_veh:
        x_g = x_veh * cos_v - (y_veh - half_w) * sin_v + vehicle_x
        y_g = x_veh * sin_v + (y_veh - half_w) * cos_v + vehicle_y
        p = output.road.right_bound.add()
        p.x = float(x_g)
        p.y = float(y_g)

    output.road.lane_width = float(lane_width)
    output.road.lane_count = 1


# === Phase 3: Build Obstacles ===

def _build_obstacles(output, obstacles, signs, R, t, K_inv, image_h,
                     vehicle_x=0.0, vehicle_y=0.0, vehicle_theta=0.0):
    cos_v = np.cos(vehicle_theta)
    sin_v = np.sin(vehicle_theta)
    obs_id = 0

    for obs in obstacles:
        o = output.obstacles.add()
        o.id = obs.get("id", obs_id)

        obs_type_str = obs.get("type", "OBSTACLE_UNKNOWN")
        proto_type = _OBSTACLE_TYPE_MAP.get(obs_type_str, "OBSTACLE_UNKNOWN")
        o.type = ObstacleType.Value(proto_type)
        o.shape = ObstacleShape.Value("SHAPE_RECT")

        x_veh, y_veh = obs["center"]
        x_g = x_veh * cos_v - y_veh * sin_v + vehicle_x
        y_g = x_veh * sin_v + y_veh * cos_v + vehicle_y
        c = o.center
        c.x = float(x_g)
        c.y = float(y_g)
        o.length = obs.get("length", 1.0)
        o.width = obs.get("width", 1.0)
        o.heading = obs.get("heading", 0.0)

        if "confidence" in obs:
            o.confidence = float(obs["confidence"])
        else:
            o.confidence = min(obs.get("n_points", 25) / 50.0, 1.0) * (1.0 - obs.get("residual", 0.1))

        if "vx" in obs and "vy" in obs:
            v = o.velocity
            v.x = float(obs["vx"])
            v.y = float(obs["vy"])

        obs_id = max(obs_id, o.id + 1)

    for sign in signs:
        x, y, w, h = sign["bbox"]
        u = x + w / 2.0
        v_coord = y + h
        veh_pt = _pixel_to_vehicle(u, v_coord, K_inv, R, t)
        if veh_pt is None:
            continue

        o = output.obstacles.add()
        o.id = obs_id
        o.type = ObstacleType.Value("OBSTACLE_STATIC")
        o.shape = ObstacleShape.Value("SHAPE_RECT")
        x_veh, y_veh = veh_pt
        x_g = x_veh * cos_v - y_veh * sin_v + vehicle_x
        y_g = x_veh * sin_v + y_veh * cos_v + vehicle_y
        c = o.center
        c.x = float(x_g)
        c.y = float(y_g)
        o.length = 0.6
        o.width = 0.6
        o.heading = 0.0
        o.confidence = sign["confidence"]

        sign_cat = sign.get("category", "unknown")
        if sign_cat != "unknown":
            cat_idx = 0
            from perception.sign_recognizer import SIGN_CATEGORIES
            if sign_cat in SIGN_CATEGORIES:
                cat_idx = SIGN_CATEGORIES.index(sign_cat)
            o.heading = 0.0
            o.confidence = float(cat_idx)

        obs_id += 1


# === Phase 4: Fuse ===

def fuse_to_perception(scan_rows, center_x, obstacles, signs,
                       K, R_cam2veh, t_cam2veh,
                       lane_width=3.5, image_w=640, image_h=480,
                       vehicle_x=0.0, vehicle_y=0.0, vehicle_theta=0.0):
    output = PerceptionOutput()
    output.header.timestamp_ns = int(time.time() * 1e9)
    output.header.frame_id = "fusion"
    output.header.seq = 0
    output.ego_pose.x = vehicle_x
    output.ego_pose.y = vehicle_y
    output.ego_pose.theta = vehicle_theta

    if abs(np.linalg.det(K)) < 1e-12:
        K_inv = np.linalg.pinv(K)
    else:
        K_inv = np.linalg.inv(K)

    center_veh = []
    valid_mask = ~np.isnan(center_x)
    for row, cx in zip(scan_rows[valid_mask], center_x[valid_mask]):
        pt = _pixel_to_vehicle(cx, row, K_inv, R_cam2veh, t_cam2veh)
        if pt is not None:
            center_veh.append(pt)

    road_valid = len(center_veh) >= 3
    output.road_valid = road_valid

    if road_valid:
        _build_road(output, center_veh, lane_width, vehicle_x, vehicle_y, vehicle_theta)

    _build_obstacles(output, obstacles, signs, R_cam2veh, t_cam2veh, K_inv, image_h,
                     vehicle_x, vehicle_y, vehicle_theta)
    output.obstacle_count = len(output.obstacles)

    return output


# === Phase 5: Default Camera Params ===

def default_camera_params(image_w=640, image_h=480):
    fx = fy = 600.0
    cx = image_w / 2.0
    cy = image_h / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1.0]])
    R = np.array([[0.0, 0.0, 1.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0]])
    t = np.array([0.0, 0.0, 1.5])
    return K, R, t
