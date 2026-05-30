import sys
import os
import numpy as np
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "generated"))
sys.path.insert(0, _PROJECT_ROOT)

from generated import (DecisionOutput, Behavior, ControlOutput, ControlMode,
                       GearPosition)
import matplotlib.pyplot as plt
from utils.plot import generate_serpentine_course, plot_vehicle


# === Phase 1: Stanley Steering ===

def _find_nearest(path_x, path_y):
    dist_sq = path_x**2 + path_y**2
    idx = int(np.argmin(dist_sq))
    return idx


def _find_preview_idx(path_x, path_y, nearest_idx, preview_dist):
    ds = np.sqrt(np.diff(path_x)**2 + np.diff(path_y)**2)
    s_cum = np.concatenate([[0.0], np.cumsum(ds)])
    s_target = s_cum[nearest_idx] + preview_dist
    idx = int(np.searchsorted(s_cum, s_target))
    return min(idx, len(path_x) - 1)


def _stanley_steer(path_x, path_y, path_theta, speed_actual, k_e, k_v,
                   wheelbase, k_preview, steer_prev, max_steer_rate_dps, dt,
                   dead_zone, v_damping, kappa_arr):
    front_x = path_x - wheelbase
    front_y = path_y.copy()
    dist_sq_front = front_x**2 + front_y**2
    idx_nearest = int(np.argmin(dist_sq_front))

    preview_dist = k_preview * max(speed_actual, 0.5)
    idx_preview = _find_preview_idx(path_x, path_y, idx_nearest, preview_dist)

    e_heading = path_theta[idx_preview]

    dx = path_x[idx_nearest] - wheelbase
    dy = path_y[idx_nearest]
    cos_h = np.cos(path_theta[idx_nearest])
    sin_h = np.sin(path_theta[idx_nearest])
    e_lat_front = -dx * sin_h + dy * cos_h

    e_lat_eff = e_lat_front if abs(e_lat_front) > dead_zone else 0.0

    k_e_eff = k_e * min(1.0, speed_actual / v_damping) if v_damping > 0 else k_e

    steer_fb = e_heading + np.arctan2(k_e_eff * e_lat_eff, speed_actual + k_v)

    kappa_near = kappa_arr[idx_nearest] if idx_nearest < len(kappa_arr) else 0.0
    steer_ff = np.arctan(kappa_near * wheelbase)

    steer_raw = steer_fb + steer_ff

    if steer_prev is not None and dt > 0:
        max_delta = np.radians(max_steer_rate_dps) * dt
        steer_raw = np.clip(steer_raw, steer_prev - max_delta, steer_prev + max_delta)

    e_lat_rear = -(path_x[idx_nearest]) * np.sin(path_theta[idx_nearest]) + path_y[idx_nearest] * np.cos(path_theta[idx_nearest])
    return steer_raw, e_lat_rear, e_heading, idx_nearest


# === Phase 2: Cascaded PID ===

def _speed_pid_outer(target_v, v_actual, kp, ki, kd, integral, error_prev,
                     d_filtered_prev, d_alpha, dt):
    error = target_v - v_actual
    integral = integral + error * dt
    integral = np.clip(integral, -3.0, 3.0)
    derivative = (error - error_prev) / max(dt, 1e-9)
    d_filtered = d_alpha * derivative + (1.0 - d_alpha) * d_filtered_prev
    a_target = kp * error + ki * integral + kd * d_filtered
    return float(a_target), float(integral), float(error), float(d_filtered)


def _accel_pi(a_target, a_actual, kp, ki, integral, dt):
    error = a_target - a_actual
    integral = integral + error * dt
    integral = np.clip(integral, -2.0, 2.0)
    output = kp * error + ki * integral
    throttle = float(np.clip(output, 0.0, 1.0))
    brake = float(np.clip(-output, 0.0, 1.0))
    return throttle, brake, float(integral)


# === Phase 3: Curvature-Limited Speed ===

def _curvature_limited_speed(kappa, v_nominal, a_lat_max):
    v_limit = np.sqrt(a_lat_max / (np.abs(kappa) + 1e-6))
    return min(v_nominal, v_limit)


# === Phase 4: Control ===

def stanley_control(decision_output, speed_actual=0.0,
                    vehicle_x=0.0, vehicle_y=0.0, vehicle_theta=0.0,
                    wheelbase=2.7, max_steer_deg=30.0,
                    k_e=0.5, k_v=1.0, a_lat_max=2.0,
                    k_preview=0.6, max_steer_rate_dps=100.0,
                    dead_zone=0.05, v_damping=2.0,
                    jerk_max=5.0, brake_ramp_rate=0.5, d_alpha=0.3,
                    pid_state=None, dt=0.02):
    control = ControlOutput()
    control.header.timestamp_ns = int(time.time() * 1e9)
    control.header.seq = decision_output.header.seq
    control.header.frame_id = "control_stanley"

    behavior = decision_output.behavior
    n_path = len(decision_output.target_path)

    if behavior == Behavior.Value("BEHAVIOR_EMERGENCY_STOP"):
        control.mode = ControlMode.Value("MODE_EMERGENCY")
        control.steering.steering_angle = 0.0
        control.steering.steering_valid = True
        control.throttle_brake.throttle = 0.0
        control.throttle_brake.brake = 1.0
        control.throttle_brake.throttle_valid = True
        control.throttle_brake.brake_valid = True
        control.gear = GearPosition.Value("GEAR_DRIVE")
        control.target_speed = 0.0
        control.target_acceleration = -5.0
        control.lateral_error = 0.0
        control.heading_error = 0.0
        control.control_valid = True
        return control, {"outer": (0.0, 0.0), "inner": 0.0, "steer_prev": None, "d_filtered": 0.0, "a_prev": 0.0, "brake_prev": 0.0}

    control.mode = ControlMode.Value("MODE_AUTO")

    if n_path < 2:
        control.steering.steering_angle = 0.0
        control.steering.steering_valid = True
        control.throttle_brake.throttle = 0.0
        control.throttle_brake.brake = 0.5
        control.throttle_brake.throttle_valid = True
        control.throttle_brake.brake_valid = True
        control.gear = GearPosition.Value("GEAR_DRIVE")
        control.target_speed = 0.0
        control.control_valid = False
        return control, {"outer": (0.0, 0.0), "inner": 0.0, "steer_prev": None, "d_filtered": 0.0, "a_prev": 0.0, "brake_prev": 0.0}

    path_x_global = np.array([p.pose.x for p in decision_output.target_path])
    path_y_global = np.array([p.pose.y for p in decision_output.target_path])
    path_theta_global = np.array([p.pose.theta for p in decision_output.target_path])
    kappa = np.array([p.curvature for p in decision_output.target_path])

    # Transform global path to vehicle-relative coordinates
    dx = path_x_global - vehicle_x
    dy = path_y_global - vehicle_y
    cos_v = np.cos(vehicle_theta)
    sin_v = np.sin(vehicle_theta)
    path_x = dx * cos_v + dy * sin_v
    path_y = -dx * sin_v + dy * cos_v
    path_theta = path_theta_global - vehicle_theta

    if pid_state is None:
        pid_state = {"outer": (0.0, 0.0), "inner": 0.0, "steer_prev": None,
                     "d_filtered": 0.0, "a_prev": 0.0, "brake_prev": 0.0}

    steer_prev = pid_state.get("steer_prev", None)
    d_filtered_prev = pid_state.get("d_filtered", 0.0)
    a_prev = pid_state.get("a_prev", 0.0)
    brake_prev = pid_state.get("brake_prev", 0.0)

    steer_rad, e_lat, e_heading, idx_nearest = _stanley_steer(
        path_x, path_y, path_theta, speed_actual, k_e, k_v,
        wheelbase, k_preview, steer_prev, max_steer_rate_dps, dt,
        dead_zone, v_damping, kappa
    )
    max_steer_rad = np.radians(max_steer_deg)
    steer_rad = np.clip(steer_rad, -max_steer_rad, max_steer_rad)
    steer_deg = np.degrees(steer_rad)

    v_nominal = decision_output.target_speed
    kappa_near = kappa[idx_nearest] if len(kappa) > 0 else 0.0
    v_target = _curvature_limited_speed(kappa_near, v_nominal, a_lat_max)

    a_target, outer_int, outer_err, d_filtered = _speed_pid_outer(
        v_target, speed_actual, 1.0, 0.05, 0.1,
        pid_state["outer"][0], pid_state["outer"][1],
        d_filtered_prev, d_alpha, dt
    )
    a_target = np.clip(a_target, -5.0, 3.0)

    if dt > 0 and jerk_max > 0:
        max_a_delta = jerk_max * dt
        a_target = np.clip(a_target, a_prev - max_a_delta, a_prev + max_a_delta)

    a_actual_est = (pid_state["outer"][1] - outer_err) / max(dt, 1e-9)
    throttle, brake, inner_int = _accel_pi(
        a_target, a_actual_est, 0.8, 0.1, pid_state["inner"], dt
    )

    if dt > 0 and brake_ramp_rate > 0:
        max_brake_inc = brake_ramp_rate * dt
        brake = max(brake, brake_prev - max_brake_inc)
        brake = min(brake, brake_prev + max_brake_inc * 3.0)

    if behavior == Behavior.Value("BEHAVIOR_STOP"):
        throttle = 0.0
        brake = max(brake, 0.3)

    if brake > 0.01:
        throttle = 0.0

    control.steering.steering_angle = float(steer_deg)
    steer_prev_deg = np.degrees(steer_prev) if steer_prev is not None else 0.0
    steer_rate_raw = (steer_deg - steer_prev_deg) / max(dt, 1e-9)
    control.steering.steering_angle_rate = float(np.clip(steer_rate_raw, -max_steer_rate_dps, max_steer_rate_dps))
    control.steering.steering_valid = True
    control.throttle_brake.throttle = throttle
    control.throttle_brake.brake = brake
    control.throttle_brake.throttle_valid = True
    control.throttle_brake.brake_valid = True
    control.gear = GearPosition.Value("GEAR_DRIVE")
    control.target_speed = float(v_target)
    control.target_acceleration = float(a_target)
    control.lateral_error = float(e_lat)
    control.heading_error = float(np.degrees(e_heading))
    control.control_valid = True

    new_state = {"outer": (outer_int, outer_err), "inner": inner_int,
                 "steer_prev": float(steer_rad), "d_filtered": float(d_filtered),
                 "a_prev": float(a_target), "brake_prev": float(brake)}
    return control, new_state


show_animation = True


def main():
    cx, cy, cyaw, ck = generate_serpentine_course(ds=0.1)

    target_speed = 30.0 / 3.6
    dt = 0.1
    L = 2.7
    max_sim_time = 60.0

    x = cx[0]
    y = cy[0]
    yaw = cyaw[0]
    v = 0.0

    x_hist, y_hist, v_hist, t_hist, lat_err_hist = [], [], [], [], []
    t = 0.0
    steer_prev = None

    while t < max_sim_time:
        dx = cx - x
        dy = cy - y
        cos_v = np.cos(yaw)
        sin_v = np.sin(yaw)
        path_x = dx * cos_v + dy * sin_v
        path_y = -dx * sin_v + dy * cos_v
        path_theta = cyaw - yaw

        idx_nearest = int(np.argmin(path_x**2 + path_y**2))
        e_heading = path_theta[idx_nearest]
        cos_h = np.cos(path_theta[idx_nearest])
        sin_h = np.sin(path_theta[idx_nearest])
        e_lat = -path_x[idx_nearest] * sin_h + path_y[idx_nearest] * cos_h

        steer_rad = e_heading + np.arctan2(0.5 * e_lat, v + 1.0)
        if steer_prev is not None:
            max_delta = np.radians(100.0) * dt
            steer_rad = np.clip(steer_rad, steer_prev - max_delta, steer_prev + max_delta)
        steer_rad = np.clip(steer_rad, -np.radians(30.0), np.radians(30.0))
        steer_prev = steer_rad

        a_lat_max = 2.0
        kappa_near = ck[idx_nearest] if idx_nearest < len(ck) else 0.0
        v_target = min(target_speed, np.sqrt(a_lat_max / (abs(kappa_near) + 1e-6)))
        a_cmd = 1.0 * (v_target - v)
        a_cmd = np.clip(a_cmd, -5.0, 3.0)

        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt
        yaw += v / L * np.tan(steer_rad) * dt
        v += a_cmd * dt
        v = max(v, 0.0)

        x_hist.append(x)
        y_hist.append(y)
        v_hist.append(v)
        t_hist.append(t)
        lat_err_hist.append(e_lat)
        t += dt

        dist_to_end = np.sqrt((x - cx[-1])**2 + (y - cy[-1])**2)
        if dist_to_end < 2.0:
            break

        if show_animation:
            plt.cla()
            plt.plot(cx, cy, ".r", label="course")
            plt.plot(x_hist, y_hist, "-b", label="trajectory")
            plt.plot(cx[idx_nearest], cy[idx_nearest], "xg", label="target")
            plot_vehicle(x, y, yaw)
            plt.axis("equal")
            plt.grid(True)
            plt.title("Stanley: Speed[km/h]:" + str(v * 3.6)[:5])
            plt.pause(0.001)

    if show_animation:
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(x_hist, y_hist, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(t_hist, [iv * 3.6 for iv in v_hist], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(t_hist, lat_err_hist, "-b")
        plt.xlabel("Time[s]")
        plt.ylabel("Lateral error[m]")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main()
