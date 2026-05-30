import sys
import os
import numpy as np
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "generated"))
sys.path.insert(0, _PROJECT_ROOT)

from generated import (DecisionOutput, Behavior, ControlOutput, SteeringCommand,
                       ThrottleCommand, ControlMode, GearPosition, Header)
import matplotlib.pyplot as plt
from utils.plot import generate_serpentine_course, plot_vehicle


# === Phase 1: Pure Pursuit Steering ===

def _find_lookahead_point(path_x, path_y, ld):
    ds = np.sqrt(np.diff(path_x)**2 + np.diff(path_y)**2)
    s_cum = np.concatenate([[0.0], np.cumsum(ds)])
    dist = np.sqrt(path_x**2 + path_y**2)
    idx_nearest = int(np.argmin(dist))
    s_target = s_cum[idx_nearest] + ld
    idx = int(np.searchsorted(s_cum, s_target))
    idx = min(idx, len(path_x) - 1)
    if idx <= idx_nearest:
        return path_x[idx_nearest], path_y[idx_nearest], dist[idx_nearest]
    alpha = (s_target - s_cum[idx - 1]) / (s_cum[idx] - s_cum[idx - 1] + 1e-12)
    lx = path_x[idx - 1] + alpha * (path_x[idx] - path_x[idx - 1])
    ly = path_y[idx - 1] + alpha * (path_y[idx] - path_y[idx - 1])
    return lx, ly, ld


def _pure_pursuit_steer(path_x, path_y, ld, wheelbase):
    lx, ly, ld_actual = _find_lookahead_point(path_x, path_y, ld)
    alpha = np.arctan2(ly, lx)
    kappa = 2.0 * np.sin(alpha) / max(ld_actual, 1e-9)
    steer_rad = np.arctan(kappa * wheelbase)
    return steer_rad, alpha


# === Phase 2: PID Speed Control ===

def _speed_pid(target_v, v_actual, kp, ki, kd, integral_prev, error_prev, dt):
    error = target_v - v_actual
    integral = integral_prev + error * dt
    integral = np.clip(integral, -5.0, 5.0)
    derivative = (error - error_prev) / max(dt, 1e-9)
    output = kp * error + ki * integral + kd * derivative
    throttle = float(np.clip(output, 0.0, 1.0))
    brake = float(np.clip(-output, 0.0, 1.0))
    return throttle, brake, float(integral), float(error)


# === Phase 3: Control Output ===

def pure_pursuit_control(decision_output, speed_actual=0.0,
                         vehicle_x=0.0, vehicle_y=0.0, vehicle_theta=0.0,
                         wheelbase=2.7, max_steer_deg=40.0,
                         lookahead_gain=0.6, lookahead_min=2.0,
                         kp=0.5, ki=0.01, kd=0.05,
                         pid_state=None, dt=0.02):
    control = ControlOutput()
    control.header.timestamp_ns = int(time.time() * 1e9)
    control.header.seq = decision_output.header.seq
    control.header.frame_id = "control"

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
        return control, (0.0, 0.0)

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
        control.target_acceleration = 0.0
        control.lateral_error = 0.0
        control.heading_error = 0.0
        control.control_valid = False
        return control, (0.0, 0.0)

    path_x_global = np.array([p.pose.x for p in decision_output.target_path])
    path_y_global = np.array([p.pose.y for p in decision_output.target_path])

    dx = path_x_global - vehicle_x
    dy = path_y_global - vehicle_y
    cos_v = np.cos(vehicle_theta)
    sin_v = np.sin(vehicle_theta)
    path_x = dx * cos_v + dy * sin_v
    path_y = -dx * sin_v + dy * cos_v

    ld = max(lookahead_gain * speed_actual, lookahead_min)
    steer_rad, heading_err = _pure_pursuit_steer(path_x, path_y, ld, wheelbase)
    max_steer_rad = np.radians(max_steer_deg)
    steer_rad = np.clip(steer_rad, -max_steer_rad, max_steer_rad)
    steer_deg = np.degrees(steer_rad)

    lateral_err = float(path_y[np.argmin(path_x**2 + path_y**2)])

    target_v = decision_output.target_speed

    if pid_state is None:
        pid_state = (0.0, 0.0)

    throttle, brake, integral_new, error_new = _speed_pid(
        target_v, speed_actual, kp, ki, kd,
        pid_state[0], pid_state[1], dt
    )

    if behavior == Behavior.Value("BEHAVIOR_STOP"):
        throttle = 0.0
        brake = max(brake, 0.3)

    if brake > 0.01:
        throttle = 0.0

    control.steering.steering_angle = float(steer_deg)
    control.steering.steering_angle_rate = 0.0
    control.steering.steering_valid = True
    control.throttle_brake.throttle = throttle
    control.throttle_brake.brake = brake
    control.throttle_brake.throttle_valid = True
    control.throttle_brake.brake_valid = True
    control.gear = GearPosition.Value("GEAR_DRIVE")
    control.target_speed = float(target_v)
    accel_est = (pid_state[1] - error_new) / max(dt, 1e-9)
    control.target_acceleration = float(np.clip(accel_est, -5.0, 3.0))
    control.lateral_error = lateral_err
    control.heading_error = float(np.degrees(heading_err))
    control.control_valid = True

    return control, (integral_new, error_new)


show_animation = True


def main():
    cx, cy, cyaw, ck = generate_serpentine_course(ds=0.1)

    target_speed = 30.0 / 3.6
    dt = 0.1
    L = 2.7
    max_sim_time = 60.0
    lookahead_gain = 0.6
    lookahead_min = 2.0

    x, y, yaw, v = cx[0], cy[0], cyaw[0], 0.0
    x_hist, y_hist, v_hist, t_hist, lat_err_hist = [], [], [], [], []
    t = 0.0
    pid_state = (0.0, 0.0)

    while t < max_sim_time:
        dx = cx - x
        dy = cy - y
        cos_v = np.cos(yaw)
        sin_v = np.sin(yaw)
        path_x = dx * cos_v + dy * sin_v
        path_y = -dx * sin_v + dy * cos_v

        ld = max(lookahead_gain * v, lookahead_min)
        steer_rad, heading_err = _pure_pursuit_steer(path_x, path_y, ld, L)
        steer_rad = np.clip(steer_rad, -np.radians(40.0), np.radians(40.0))

        idx_nearest = int(np.argmin(path_x**2 + path_y**2))
        e_lat = float(path_y[idx_nearest])

        throttle, brake, integral_new, error_new = _speed_pid(
            target_speed, v, 0.5, 0.01, 0.05, pid_state[0], pid_state[1], dt
        )
        pid_state = (integral_new, error_new)
        a_cmd = throttle * 3.0 - brake * 5.0

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
            lx, ly, _ = _find_lookahead_point(path_x, path_y, ld)
            gx = x + lx * np.cos(yaw) - ly * np.sin(yaw)
            gy = y + lx * np.sin(yaw) + ly * np.cos(yaw)
            plt.plot(gx, gy, "xg", label="target")
            plot_vehicle(x, y, yaw)
            plt.axis("equal")
            plt.grid(True)
            plt.title(f"PurePursuit: Speed[km/h]:{v * 3.6:.1f}")
            plt.pause(0.001)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    axes[0].plot(cx, cy, ".r", label="course")
    axes[0].plot(x_hist, y_hist, "-b", label="trajectory")
    axes[0].legend()
    axes[0].set_xlabel("x[m]")
    axes[0].set_ylabel("y[m]")
    axes[0].axis("equal")
    axes[0].grid(True)
    axes[1].plot(t_hist, [iv * 3.6 for iv in v_hist], "-r")
    axes[1].set_xlabel("Time[s]")
    axes[1].set_ylabel("Speed[km/h]")
    axes[1].grid(True)
    axes[2].plot(t_hist, lat_err_hist, "-b")
    axes[2].set_xlabel("Time[s]")
    axes[2].set_ylabel("Lateral error[m]")
    axes[2].grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
