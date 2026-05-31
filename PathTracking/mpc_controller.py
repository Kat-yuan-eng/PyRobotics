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


# === Phase 1: Bicycle Model Rollout ===

def _rollout(x0, y0, theta0, v0, deltas, accels, L, dt_mpc, max_speed=30.0):
    n = len(deltas)
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    theta = np.zeros(n + 1)
    v = np.zeros(n + 1)
    x[0], y[0], theta[0], v[0] = x0, y0, theta0, v0

    for k in range(n):
        x[k + 1] = x[k] + v[k] * np.cos(theta[k]) * dt_mpc
        y[k + 1] = y[k] + v[k] * np.sin(theta[k]) * dt_mpc
        theta[k + 1] = theta[k] + v[k] * np.tan(deltas[k]) / L * dt_mpc
        v[k + 1] = min(max(v[k] + accels[k] * dt_mpc, 0.0), max_speed)

    return x, y, theta, v


# === Phase 2: Cost Function ===

def _compute_cost(x, y, theta, v, deltas, accels,
                  ref_x, ref_y, ref_theta, ref_v,
                  w_y, w_theta, w_v, w_delta, w_a, w_deltadot, dt_mpc,
                  w_terminal=5.0):
    n = len(deltas)
    cost = 0.0
    for k in range(1, n + 1):
        dists_sq = (ref_x - x[k])**2 + (ref_y - y[k])**2
        idx = int(np.argmin(dists_sq))
        cost += w_y * (y[k] - ref_y[idx])**2
        cost += w_theta * (theta[k] - ref_theta[idx])**2
        cost += w_v * (v[k] - ref_v[idx])**2
        cost += w_delta * deltas[k - 1]**2
        cost += w_a * accels[k - 1]**2

    for k in range(1, n):
        ddot = (deltas[k] - deltas[k - 1]) / dt_mpc
        cost += w_deltadot * ddot**2

    dists_sq_t = (ref_x - x[n])**2 + (ref_y - y[n])**2
    idx_terminal = int(np.argmin(dists_sq_t))
    cost += w_terminal * ((y[n] - ref_y[idx_terminal])**2 +
                          (theta[n] - ref_theta[idx_terminal])**2 +
                          (v[n] - ref_v[idx_terminal])**2)

    return cost


# === Phase 3: Gradient Descent ===

def _numerical_gradient(x0, y0, theta0, v0, deltas, accels, L, dt_mpc,
                        ref_x, ref_y, ref_theta, ref_v,
                        w_y, w_theta, w_v, w_delta, w_a, w_deltadot, eps):
    grad_d = np.zeros_like(deltas)
    grad_a = np.zeros_like(accels)

    for i in range(len(deltas)):
        d_plus = deltas.copy()
        d_plus[i] += eps
        x_p, y_p, t_p, v_p = _rollout(x0, y0, theta0, v0, d_plus, accels, L, dt_mpc)
        c_p = _compute_cost(x_p, y_p, t_p, v_p, d_plus, accels,
                            ref_x, ref_y, ref_theta, ref_v,
                            w_y, w_theta, w_v, w_delta, w_a, w_deltadot, dt_mpc)

        d_minus = deltas.copy()
        d_minus[i] -= eps
        x_m, y_m, t_m, v_m = _rollout(x0, y0, theta0, v0, d_minus, accels, L, dt_mpc)
        c_m = _compute_cost(x_m, y_m, t_m, v_m, d_minus, accels,
                            ref_x, ref_y, ref_theta, ref_v,
                            w_y, w_theta, w_v, w_delta, w_a, w_deltadot, dt_mpc)

        grad_d[i] = (c_p - c_m) / (2.0 * eps)

    for i in range(len(accels)):
        a_plus = accels.copy()
        a_plus[i] += eps
        x_p, y_p, t_p, v_p = _rollout(x0, y0, theta0, v0, deltas, a_plus, L, dt_mpc)
        c_p = _compute_cost(x_p, y_p, t_p, v_p, deltas, a_plus,
                            ref_x, ref_y, ref_theta, ref_v,
                            w_y, w_theta, w_v, w_delta, w_a, w_deltadot, dt_mpc)

        a_minus = accels.copy()
        a_minus[i] -= eps
        x_m, y_m, t_m, v_m = _rollout(x0, y0, theta0, v0, deltas, a_minus, L, dt_mpc)
        c_m = _compute_cost(x_m, y_m, t_m, v_m, deltas, a_minus,
                            ref_x, ref_y, ref_theta, ref_v,
                            w_y, w_theta, w_v, w_delta, w_a, w_deltadot, dt_mpc)

        grad_a[i] = (c_p - c_m) / (2.0 * eps)

    return grad_d, grad_a


# === Phase 4: Control ===

def mpc_control(decision_output, speed_actual=0.0,
                wheelbase=2.7, max_steer_deg=30.0,
                N=10, dt_mpc=0.1,
                w_y=3.0, w_theta=2.0, w_v=1.0,
                w_delta=0.1, w_a=0.1, w_deltadot=0.5,
                mpc_state=None, dt=0.02,
                vehicle_x=0.0, vehicle_y=0.0, vehicle_theta=0.0,
                lr=0.05, n_grad_iter=5, grad_eps=1e-4):
    control = ControlOutput()
    control.header.timestamp_ns = int(time.time() * 1e9)
    control.header.seq = decision_output.header.seq
    control.header.frame_id = "control_mpc"

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
        return control, None

    control.mode = ControlMode.Value("MODE_AUTO")

    if n_path < 2:
        control.steering.steering_angle = 0.0
        control.steering.steering_valid = True
        control.throttle_brake.brake = 0.5
        control.throttle_brake.throttle_valid = True
        control.throttle_brake.brake_valid = True
        control.gear = GearPosition.Value("GEAR_DRIVE")
        control.control_valid = False
        return control, None

    path_x_global = np.array([p.pose.x for p in decision_output.target_path])
    path_y_global = np.array([p.pose.y for p in decision_output.target_path])
    path_theta_global = np.array([p.pose.theta for p in decision_output.target_path])
    dx = path_x_global - vehicle_x
    dy = path_y_global - vehicle_y
    cos_v = np.cos(vehicle_theta)
    sin_v = np.sin(vehicle_theta)
    ref_x = dx * cos_v + dy * sin_v
    ref_y = -dx * sin_v + dy * cos_v
    ref_theta = path_theta_global - vehicle_theta
    ref_v = np.full(n_path, decision_output.target_speed)

    max_steer_rad = np.radians(max_steer_deg)
    max_accel = 3.0

    if mpc_state is not None and "deltas" in mpc_state:
        deltas = mpc_state["deltas"].copy()
        accels = mpc_state["accels"].copy()
    else:
        deltas = np.zeros(N)
        accels = np.zeros(N)

    for _ in range(n_grad_iter):
        grad_d, grad_a = _numerical_gradient(
            0.0, 0.0, 0.0, speed_actual, deltas, accels,
            wheelbase, dt_mpc, ref_x, ref_y, ref_theta, ref_v,
            w_y, w_theta, w_v, w_delta, w_a, w_deltadot, grad_eps
        )
        deltas -= lr * grad_d
        accels -= lr * grad_a
        deltas = np.clip(deltas, -max_steer_rad, max_steer_rad)
        accels = np.clip(accels, -max_accel, max_accel)

    steer_deg = np.degrees(deltas[0])
    a_cmd = accels[0]

    throttle = float(np.clip(a_cmd / 3.0, 0.0, 1.0))
    brake = float(np.clip(-a_cmd / 5.0, 0.0, 1.0))

    if behavior == Behavior.Value("BEHAVIOR_STOP"):
        throttle = 0.0
        brake = max(brake, 0.3)

    if brake > 0.01:
        throttle = 0.0

    control.steering.steering_angle = float(steer_deg)
    control.steering.steering_valid = True
    control.throttle_brake.throttle = throttle
    control.throttle_brake.brake = brake
    control.throttle_brake.throttle_valid = True
    control.throttle_brake.brake_valid = True
    control.gear = GearPosition.Value("GEAR_DRIVE")
    control.target_speed = float(decision_output.target_speed)
    control.target_acceleration = float(a_cmd)
    idx_near = int(np.argmin(ref_x**2 + ref_y**2))
    control.lateral_error = float(ref_y[idx_near])
    control.heading_error = float(np.degrees(ref_theta[idx_near]))
    control.control_valid = True

    new_state = {
        "deltas": np.roll(deltas, -1),
        "accels": np.roll(accels, -1),
    }
    new_state["deltas"][-1] = deltas[-1]
    new_state["accels"][-1] = accels[-1]

    return control, new_state


show_animation = True


def main():
    cx, cy, cyaw, ck = generate_serpentine_course(ds=0.1)

    target_speed = 30.0 / 3.6
    dt = 0.1
    L = 2.7
    max_sim_time = 50.0
    N = 20
    dt_mpc = 0.1
    max_steer_rad = np.radians(30.0)
    max_accel = 3.0
    w_y = 10.0
    w_theta = 6.0
    w_v = 1.0
    w_delta = 0.05
    w_a = 0.05
    w_deltadot = 0.5
    lr_mpc = 0.1
    n_grad_iter = 10

    x, y, yaw, v = cx[0], cy[0], cyaw[0], 2.0
    x_hist, y_hist, v_hist, t_hist, lat_err_hist = [], [], [], [], []
    t = 0.0
    deltas = np.zeros(N)
    accels = np.full(N, 1.5)

    while t < max_sim_time:
        dx = cx - x
        dy = cy - y
        cos_v = np.cos(yaw)
        sin_v = np.sin(yaw)
        ref_x = dx * cos_v + dy * sin_v
        ref_y = -dx * sin_v + dy * cos_v
        ref_theta = cyaw - yaw
        ref_v = np.full(len(cx), target_speed)

        for _ in range(n_grad_iter):
            grad_d, grad_a = _numerical_gradient(
                0.0, 0.0, 0.0, v, deltas, accels,
                L, dt_mpc, ref_x, ref_y, ref_theta, ref_v,
                w_y, w_theta, w_v, w_delta, w_a, w_deltadot, 1e-4
            )
            deltas -= lr_mpc * grad_d
            accels -= lr_mpc * grad_a
            deltas = np.clip(deltas, -max_steer_rad, max_steer_rad)
            accels = np.clip(accels, -max_accel, max_accel)

        pred_x, pred_y, pred_theta, pred_v = _rollout(
            0.0, 0.0, 0.0, v, deltas, accels, L, dt_mpc
        )
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        pred_x_g = x + pred_x * cos_y - pred_y * sin_y
        pred_y_g = y + pred_x * sin_y + pred_y * cos_y

        steer_rad = deltas[0]
        a_cmd = accels[0]
        throttle = float(np.clip(a_cmd / 3.0, 0.0, 1.0))
        brake = float(np.clip(-a_cmd / 5.0, 0.0, 1.0))
        a_actual = throttle * 3.0 - brake * 5.0

        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt
        yaw += v / L * np.tan(steer_rad) * dt
        v += a_actual * dt
        v = max(v, 0.0)

        idx_nearest = int(np.argmin(ref_x**2 + ref_y**2))
        e_lat = float(ref_y[idx_nearest])

        x_hist.append(x)
        y_hist.append(y)
        v_hist.append(v)
        t_hist.append(t)
        lat_err_hist.append(e_lat)
        t += dt

        deltas = np.roll(deltas, -1)
        accels = np.roll(accels, -1)
        deltas[-1] = deltas[-2]
        accels[-1] = accels[-2]

        dist_to_end = np.sqrt((x - cx[-1])**2 + (y - cy[-1])**2)
        if dist_to_end < 2.0:
            break

        if show_animation:
            plt.cla()
            plt.plot(cx, cy, ".r", label="course")
            plt.plot(x_hist, y_hist, "-b", label="trajectory")
            plt.plot(pred_x_g, pred_y_g, "-r", alpha=0.3, label="MPC pred")
            plt.plot(cx[idx_nearest], cy[idx_nearest], "xg", label="target")
            plot_vehicle(x, y, yaw)
            plt.axis("equal")
            plt.grid(True)
            plt.title(f"MPC: Speed[km/h]:{v * 3.6:.1f}")
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

    final_lat_err = np.sqrt(np.mean(np.array(lat_err_hist[-20:])**2)) if len(lat_err_hist) >= 20 else float('inf')
    assert final_lat_err < 1.0, f"MPC lateral error too large: {final_lat_err:.2f}m (threshold: 1.0m)"
    print(f"MPC validation PASSED: final_lateral_error_RMS={final_lat_err:.2f}m, "
          f"max_speed={max(v_hist)*3.6:.1f}km/h")


if __name__ == '__main__':
    main()
