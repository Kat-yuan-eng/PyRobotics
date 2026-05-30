import threading
import time
import numpy as np
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "generated"))
sys.path.insert(0, _PROJECT_ROOT)

from system.realtime.latest_value import LatestResult
from system.vehicle_sim import simulate_vehicle
from generated import (PerceptionOutput, DecisionOutput, Behavior,
                       DecisionStatus, ControlOutput, ControlMode, Header)

from perception.lane_pixel_detector import detect_lane_pixels, generate_test_image as gen_lane_img
from perception.obstacle_detector import detect_obstacles, generate_test_point_cloud
from perception.sign_recognizer import recognize_signs, generate_test_sign_image
from perception.sensor_fusion import fuse_to_perception, default_camera_params
from decision.task_scheduler import schedule, TASK_PATROL
from PathTracking.controller_selector import select_controller, CTRL_STANLEY, ctrl_name
from Localization.ekf_localizer import ekf_localize


CONTROL_FREQ_HZ = 50
CONTROL_PERIOD_S = 1.0 / CONTROL_FREQ_HZ
PERCEPTION_PERIOD_S = 0.03
V_NOMINAL = 5.0
GPS_INTERVAL = 5
GPS_NOISE_STD = 2.0
PERCEPTION_MAX_AGE_S = 0.1


# === Phase 1: Perception Thread ===

def _perception_loop(latest_perception, stop_event, test_img, test_pc, test_sign_img, K, R_cam, t_cam, loc_state, loc_lock):
    obstacles = []
    signs = []
    frame = 0
    while not stop_event.is_set():
        t0 = time.perf_counter()

        scan_rows, center_x = detect_lane_pixels(test_img)
        if frame % 2 == 0:
            obstacles = detect_obstacles(test_pc)
            signs = recognize_signs(test_sign_img)

        with loc_lock:
            vx, vy, vtheta = loc_state
        perception_out = fuse_to_perception(
            scan_rows, center_x, obstacles, signs,
            K, R_cam, t_cam,
            vehicle_x=vx, vehicle_y=vy, vehicle_theta=vtheta
        )
        perception_out.header.timestamp_ns = int(t0 * 1e9)
        latest_perception.write(perception_out)

        frame += 1
        elapsed = time.perf_counter() - t0
        sleep_time = PERCEPTION_PERIOD_S - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# === Phase 2: Control Thread ===

def _control_loop(latest_perception, stop_event, loc_state, loc_lock, v_nominal, duration_s):
    v_actual = 0.0
    x_actual = 0.0
    y_actual = 0.0
    theta_actual = 0.0
    prev_theta = theta_actual
    ctrl_state = None
    scheduler_state = {"current_task": TASK_PATROL, "task_state": {}}
    seq = 0

    x_ekf = np.array([0.0, 0.0, 0.0, 0.0])
    P_ekf = np.eye(4) * 0.01
    Q_ekf = np.diag([0.1, 0.1, np.deg2rad(1.0), 0.5]) ** 2
    R_ekf = np.diag([GPS_NOISE_STD, GPS_NOISE_STD]) ** 2

    rng = np.random.default_rng(42)

    t_start = time.perf_counter()
    t_next = t_start

    while not stop_event.is_set():
        t_now = time.perf_counter()
        if t_now - t_start >= duration_s:
            break

        sleep_margin = t_next - time.perf_counter() - 0.001
        if sleep_margin > 0:
            time.sleep(sleep_margin)
        while time.perf_counter() < t_next:
            pass

        perception_out, percep_seq = latest_perception.read()

        percep_age = 0.0
        if perception_out is not None:
            percep_age = (time.perf_counter() - perception_out.header.timestamp_ns / 1e9)
            if percep_age > PERCEPTION_MAX_AGE_S:
                perception_out = None

        if perception_out is not None:
            perception_out.header.seq = seq

            yaw_rate_odom = (theta_actual - prev_theta) / CONTROL_PERIOD_S if seq > 0 else 0.0
            u_odom = np.array([v_actual, yaw_rate_odom])

            z_gps = None
            if seq % GPS_INTERVAL == 0 and seq > 0:
                z_gps = np.array([x_actual, y_actual]) + rng.normal(0, GPS_NOISE_STD, 2)

            x_ekf, P_ekf = ekf_localize(x_ekf, P_ekf, u_odom, z_gps, Q_ekf, R_ekf, CONTROL_PERIOD_S)
            prev_theta = theta_actual

            with loc_lock:
                loc_state[0] = x_ekf[0]
                loc_state[1] = x_ekf[1]
                loc_state[2] = x_ekf[2]

            decision_out, scheduler_state = schedule(
                perception_out, scheduler_state, v_nominal=v_nominal
            )

            control_out, ctrl_state = select_controller(
                CTRL_STANLEY, decision_out, v_actual, ctrl_state,
                vehicle_x=x_ekf[0], vehicle_y=x_ekf[1], vehicle_theta=x_ekf[2]
            )

            v_actual, x_actual, y_actual, theta_actual = simulate_vehicle(
                control_out.steering.steering_angle,
                control_out.throttle_brake.throttle,
                control_out.throttle_brake.brake,
                v_actual, x_actual, y_actual, theta_actual,
                CONTROL_PERIOD_S
            )

            if seq % 10 == 0:
                beh_name = Behavior.Name(decision_out.behavior)
                mode_name = ControlMode.Name(control_out.mode)
                ctrl_label = ctrl_name(ctrl_state.get("type", CTRL_STANLEY)) if ctrl_state else "Stanley"
                loc_err = np.sqrt((x_ekf[0] - x_actual)**2 + (x_ekf[1] - y_actual)**2)
                print(
                    f"[{seq:4d}] "
                    f"steer={control_out.steering.steering_angle:+6.2f}deg  "
                    f"thr={control_out.throttle_brake.throttle:.2f}  "
                    f"brk={control_out.throttle_brake.brake:.2f}  "
                    f"v={v_actual:.2f}m/s  "
                    f"loc_err={loc_err:.3f}m  "
                    f"percep_age={percep_age*1000:.0f}ms  "
                    f"{beh_name}  {ctrl_label}"
                )

        seq += 1
        t_next = t_start + seq * CONTROL_PERIOD_S


# === Phase 3: Pipeline Entry ===

def run_realtime_pipeline(duration_s=10.0):
    print(f"[RealtimePipeline] Starting concurrent perception({PERCEPTION_PERIOD_S*1000:.0f}ms) + control({CONTROL_PERIOD_S*1000:.0f}ms)")
    print(f"[RealtimePipeline] With EKF localization + perception staleness check")
    print("=" * 78)

    test_img = gen_lane_img()
    test_pc = generate_test_point_cloud()
    test_sign_img = generate_test_sign_image()
    K, R_cam, t_cam = default_camera_params()

    from perception.sign_recognizer import _get_template_db
    _get_template_db()

    latest_perception = LatestResult()
    stop_event = threading.Event()
    loc_state = np.array([0.0, 0.0, 0.0])
    loc_lock = threading.Lock()

    t_percep = threading.Thread(
        target=_perception_loop,
        args=(latest_perception, stop_event, test_img, test_pc, test_sign_img, K, R_cam, t_cam, loc_state, loc_lock),
        daemon=True
    )
    t_ctrl = threading.Thread(
        target=_control_loop,
        args=(latest_perception, stop_event, loc_state, loc_lock, V_NOMINAL, duration_s),
        daemon=True
    )

    t_percep.start()
    t_ctrl.start()

    time.sleep(duration_s)

    stop_event.set()
    t_percep.join(timeout=1.0)
    t_ctrl.join(timeout=1.0)

    print("=" * 78)
    print("[RealtimePipeline] Stopped")


if __name__ == "__main__":
    run_realtime_pipeline()
