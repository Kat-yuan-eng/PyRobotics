import sys
import os
import numpy as np
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "generated"))
sys.path.insert(0, _PROJECT_ROOT)

from generated import (DecisionOutput, Behavior, ControlOutput, ControlMode,
                       GearPosition)
from PathTracking.stanley_controller import _stanley_steer
import matplotlib.pyplot as plt
from utils.plot import generate_serpentine_course, plot_vehicle


# === Phase 1: Membership Functions ===

def _trimf(x, a, b, c):
    return np.maximum(0.0, np.minimum((x - a) / (b - a + 1e-12),
                                       (c - x) / (c - b + 1e-12)))


# === Phase 2: Fuzzy Sets ===

_KAPPA_Z = (-0.01, 0.0, 0.12)
_KAPPA_S = (0.06, 0.20, 0.45)
_KAPPA_L = (0.30, 0.60, 1.00)
_KAPPA_VL = (0.70, 1.50, 3.00)

_DV_NL = (-9.0, -4.0, -1.0)
_DV_NS = (-3.0, -1.0, 0.5)
_DV_Z = (-0.8, 0.0, 0.8)
_DV_PS = (-0.5, 1.0, 3.0)
_DV_PL = (1.0, 4.0, 9.0)

_DU_NL = -0.4
_DU_NS = -0.15
_DU_Z = 0.0
_DU_PS = 0.15
_DU_PL = 0.4

_RULES = [
    (_KAPPA_Z, _DV_NL, _DU_PL),
    (_KAPPA_Z, _DV_NS, _DU_PS),
    (_KAPPA_Z, _DV_Z, _DU_Z),
    (_KAPPA_Z, _DV_PS, _DU_NS),
    (_KAPPA_Z, _DV_PL, _DU_NL),
    (_KAPPA_S, _DV_NL, _DU_PL),
    (_KAPPA_S, _DV_NS, _DU_PS),
    (_KAPPA_S, _DV_Z, _DU_Z),
    (_KAPPA_S, _DV_PS, _DU_NS),
    (_KAPPA_S, _DV_PL, _DU_NL),
    (_KAPPA_L, _DV_NL, _DU_PS),
    (_KAPPA_L, _DV_NS, _DU_Z),
    (_KAPPA_L, _DV_Z, _DU_NS),
    (_KAPPA_L, _DV_PS, _DU_NL),
    (_KAPPA_L, _DV_PL, _DU_NL),
    (_KAPPA_VL, _DV_NL, _DU_PS),
    (_KAPPA_VL, _DV_NS, _DU_Z),
    (_KAPPA_VL, _DV_Z, _DU_NS),
    (_KAPPA_VL, _DV_PS, _DU_NL),
    (_KAPPA_VL, _DV_PL, _DU_NL),
]


# === Phase 3: Fuzzy Inference ===

def _fuzzy_infer(kappa, dv):
    dv = np.clip(dv, -8.0, 8.0)
    num = 0.0
    den = 0.0
    for k_set, v_set, du_val in _RULES:
        w = min(_trimf(kappa, *k_set), _trimf(dv, *v_set))
        num += w * du_val
        den += w
    if den < 1e-12:
        return 0.0
    return num / den


# === Phase 4: Control ===

def fuzzy_control(decision_output, speed_actual=0.0,
                  wheelbase=2.7, max_steer_deg=30.0,
                  k_e=0.5, k_v=1.0,
                  k_preview=0.6, max_steer_rate_dps=100.0,
                  dead_zone=0.05, v_damping=1.0,
                  steer_state=None, fuzzy_state=None, dt=0.02,
                  vehicle_x=0.0, vehicle_y=0.0, vehicle_theta=0.0):
    control = ControlOutput()
    control.header.timestamp_ns = int(time.time() * 1e9)
    control.header.seq = decision_output.header.seq
    control.header.frame_id = "control_fuzzy"

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
        control.control_valid = True
        return control, {"steer": None, "fuzzy": (0.0, decision_output.target_speed, 0.0)}

    control.mode = ControlMode.Value("MODE_AUTO")

    if n_path < 2:
        control.steering.steering_angle = 0.0
        control.steering.steering_valid = True
        control.throttle_brake.brake = 0.5
        control.throttle_brake.throttle_valid = True
        control.throttle_brake.brake_valid = True
        control.gear = GearPosition.Value("GEAR_DRIVE")
        control.control_valid = False
        return control, {"steer": None, "fuzzy": (0.0, decision_output.target_speed, 0.0)}

    path_x_global = np.array([p.pose.x for p in decision_output.target_path])
    path_y_global = np.array([p.pose.y for p in decision_output.target_path])
    path_theta_global = np.array([p.pose.theta for p in decision_output.target_path])
    dx = path_x_global - vehicle_x
    dy = path_y_global - vehicle_y
    cos_v = np.cos(vehicle_theta)
    sin_v = np.sin(vehicle_theta)
    path_x = dx * cos_v + dy * sin_v
    path_y = -dx * sin_v + dy * cos_v
    path_theta = path_theta_global - vehicle_theta
    kappa = np.array([p.curvature for p in decision_output.target_path])

    steer_prev = steer_state.get("steer_prev", None) if steer_state else None
    e_heading_prev = steer_state.get("e_heading_prev", None) if steer_state else None

    steer_rad, e_lat, e_heading, idx_near = _stanley_steer(
        path_x, path_y, path_theta, speed_actual, k_e, k_v,
        wheelbase, k_preview, steer_prev, max_steer_rate_dps, dt,
        dead_zone, v_damping, kappa, e_heading_prev
    )
    max_steer_rad = np.radians(max_steer_deg)
    steer_rad = np.clip(steer_rad, -max_steer_rad, max_steer_rad)
    steer_deg = np.degrees(steer_rad)

    if fuzzy_state is None:
        fuzzy_state = (0.0, decision_output.target_speed, 0.0)

    throttle_prev = fuzzy_state[0]
    v_integral = fuzzy_state[2]

    v_target = decision_output.target_speed
    kappa_near = abs(kappa[idx_near]) if len(kappa) > 0 else 0.0
    dv = speed_actual - v_target

    du = _fuzzy_infer(kappa_near, dv)

    v_err = v_target - speed_actual
    v_integral = np.clip(v_integral + v_err * dt, -3.0, 3.0)
    throttle_pi = 0.1 * v_err + 0.02 * v_integral
    throttle_new = np.clip(throttle_prev + du + throttle_pi * dt, 0.0, 1.0)

    brake_preview = float(np.clip(0.3 * kappa_near * speed_actual, 0.0, 0.5))

    if dv > 1.0:
        brake = float(np.clip(dv * 0.3, 0.0, 1.0))
        throttle_new = max(throttle_new - brake * 0.5, 0.0)
    elif kappa_near > 0.3 and dv > -1.0:
        brake = brake_preview
        throttle_new = max(throttle_new - brake_preview * 0.3, 0.0)
    else:
        brake = 0.0

    if behavior == Behavior.Value("BEHAVIOR_STOP"):
        throttle_new = 0.0
        brake = max(brake, 0.3)

    if brake > 0.01:
        throttle_new = 0.0

    control.steering.steering_angle = float(steer_deg)
    control.steering.steering_valid = True
    control.throttle_brake.throttle = float(throttle_new)
    control.throttle_brake.brake = float(brake)
    control.throttle_brake.throttle_valid = True
    control.throttle_brake.brake_valid = True
    control.gear = GearPosition.Value("GEAR_DRIVE")
    control.target_speed = float(v_target)
    control.lateral_error = float(e_lat)
    control.heading_error = float(np.degrees(e_heading))
    control.steering.steering_angle_rate = 0.0
    control.target_acceleration = 0.0
    control.control_valid = True

    new_state = {"steer": {"steer_prev": float(steer_rad), "e_heading_prev": float(e_heading)},
                 "fuzzy": (float(throttle_new), float(v_target), float(v_integral))}
    return control, new_state


show_animation = True


def main():
    cx, cy, cyaw, ck = generate_serpentine_course(ds=0.1)

    target_speed = 30.0 / 3.6
    dt = 0.1
    L = 2.7
    max_sim_time = 60.0
    k_e = 0.5
    k_v = 1.0
    k_preview = 0.6
    max_steer_rate_dps = 100.0
    dead_zone = 0.05
    v_damping = 1.0

    x, y, yaw, v = cx[0], cy[0], cyaw[0], 0.0
    x_hist, y_hist, v_hist, t_hist, lat_err_hist = [], [], [], [], []
    t = 0.0
    steer_prev = None
    e_heading_prev = None
    throttle_prev = 0.0
    v_integral = 0.0

    while t < max_sim_time:
        dx = cx - x
        dy = cy - y
        cos_v = np.cos(yaw)
        sin_v = np.sin(yaw)
        path_x = dx * cos_v + dy * sin_v
        path_y = -dx * sin_v + dy * cos_v
        path_theta = cyaw - yaw

        steer_rad, e_lat, e_heading, idx_nearest = _stanley_steer(
            path_x, path_y, path_theta, v, k_e, k_v,
            L, k_preview, steer_prev, max_steer_rate_dps, dt,
            dead_zone, v_damping, ck, e_heading_prev
        )
        steer_rad = np.clip(steer_rad, -np.radians(30.0), np.radians(30.0))
        steer_prev = steer_rad
        e_heading_prev = e_heading

        kappa_near = abs(ck[idx_nearest]) if idx_nearest < len(ck) else 0.0
        dv = v - target_speed
        du = _fuzzy_infer(kappa_near, dv)

        v_err = target_speed - v
        v_integral = np.clip(v_integral + v_err * dt, -3.0, 3.0)
        throttle_pi = 0.1 * v_err + 0.02 * v_integral
        throttle_new = np.clip(throttle_prev + du + throttle_pi * dt, 0.0, 1.0)
        throttle_prev = throttle_new

        brake_preview = float(np.clip(0.3 * kappa_near * v, 0.0, 0.5))

        if dv > 1.0:
            brake = float(np.clip(dv * 0.3, 0.0, 1.0))
            throttle_new = max(throttle_new - brake * 0.5, 0.0)
        elif kappa_near > 0.3 and dv > -1.0:
            brake = brake_preview
            throttle_new = max(throttle_new - brake_preview * 0.3, 0.0)
        else:
            brake = 0.0

        a_cmd = throttle_new * 3.0 - brake * 5.0

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
            plt.title(f"Fuzzy: Speed[km/h]:{v * 3.6:.1f}")
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
