import sys
import os
import numpy as np
import time

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "generated"))
sys.path.insert(0, _PROJECT_ROOT)

from generated import (PerceptionOutput, DecisionOutput, Behavior,
                       DecisionStatus, ControlOutput, ControlMode, Header)

from perception.lane_pixel_detector import detect_lane_pixels, generate_test_image as gen_lane_img
from perception.obstacle_detector import detect_obstacles, generate_test_point_cloud
from perception.obstacle_tracker import track_obstacles, get_confirmed_tracks
from perception.sign_recognizer import recognize_signs, generate_test_sign_image, warmup_template_db
from perception.sensor_fusion import fuse_to_perception, default_camera_params
from decision.task_scheduler import schedule, TASK_PATROL
from PathTracking.controller_selector import auto_select_controller, CTRL_STANLEY, ctrl_name
from system.vehicle_sim import simulate_vehicle
from Localization.ekf_localizer import ekf_localize


CONTROL_FREQ_HZ = 50
CONTROL_PERIOD_S = 1.0 / CONTROL_FREQ_HZ
V_NOMINAL = 5.0
SIM_DURATION_S = 5.0
GPS_INTERVAL = 5
GPS_NOISE_STD = 2.0


# === Phase 1: Main Loop ===

def run_main_loop(duration_s=SIM_DURATION_S):
    print(f"[MainLoop] Starting {CONTROL_FREQ_HZ}Hz control loop for {duration_s}s")
    print(f"[MainLoop] Perception(Detect+Track+Fusion) -> Localization(EKF) -> Decision -> Control pipeline")
    print("=" * 78)

    test_img = gen_lane_img()
    test_pc = generate_test_point_cloud()
    test_sign_img = generate_test_sign_image()
    K, R_cam, t_cam = default_camera_params()

    warmup_template_db()

    signs = recognize_signs(test_sign_img)
    obstacles = detect_obstacles(test_pc)
    tracks = []
    free_ids = set()

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

    while True:
        t_now = time.perf_counter()
        if t_now - t_start >= duration_s:
            break

        sleep_margin = t_next - time.perf_counter() - 0.001
        if sleep_margin > 0:
            time.sleep(sleep_margin)
        while time.perf_counter() < t_next:
            pass

        scan_rows, center_x = detect_lane_pixels(test_img)
        if seq % 5 == 0:
            obstacles = detect_obstacles(test_pc)
            signs = recognize_signs(test_sign_img)
        tracks = track_obstacles(obstacles, tracks, dt=CONTROL_PERIOD_S,
                                 free_ids=free_ids)
        confirmed = get_confirmed_tracks(tracks)
        perception_out = fuse_to_perception(
            scan_rows, center_x, confirmed, signs,
            K, R_cam, t_cam,
            vehicle_x=x_ekf[0], vehicle_y=x_ekf[1], vehicle_theta=x_ekf[2]
        )
        perception_out.header.seq = seq

        yaw_rate_odom = (theta_actual - prev_theta) / CONTROL_PERIOD_S if seq > 0 else 0.0
        u_odom = np.array([v_actual, yaw_rate_odom])

        z_gps = None
        if seq % GPS_INTERVAL == 0 and seq > 0:
            z_gps = np.array([x_actual, y_actual]) + rng.normal(0, GPS_NOISE_STD, 2)

        x_ekf, P_ekf = ekf_localize(x_ekf, P_ekf, u_odom, z_gps, Q_ekf, R_ekf, CONTROL_PERIOD_S)

        prev_theta = theta_actual

        decision_out, scheduler_state = schedule(
            perception_out, scheduler_state, v_nominal=V_NOMINAL
        )

        control_out, ctrl_state = auto_select_controller(
            decision_out, v_actual, ctrl_state,
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
            task_name = scheduler_state.get("task_name", "?")
            ctrl_label = ctrl_name(ctrl_state.get("type", CTRL_STANLEY)) if ctrl_state else "Stanley"
            loc_err = np.sqrt((x_ekf[0] - x_actual)**2 + (x_ekf[1] - y_actual)**2)
            n_confirmed = len(confirmed)
            n_tracks = len(tracks)
            print(
                f"[{seq:4d}] "
                f"steer={control_out.steering.steering_angle:+6.2f}deg  "
                f"thr={control_out.throttle_brake.throttle:.2f}  "
                f"brk={control_out.throttle_brake.brake:.2f}  "
                f"v={v_actual:.2f}m/s  "
                f"loc_err={loc_err:.3f}m  "
                f"trk={n_confirmed}/{n_tracks}  "
                f"{beh_name}  {mode_name}  {task_name}  {ctrl_label}"
            )

        seq += 1
        t_next = t_start + seq * CONTROL_PERIOD_S

    t_elapsed = time.perf_counter() - t_start
    loc_err_final = np.sqrt((x_ekf[0] - x_actual)**2 + (x_ekf[1] - y_actual)**2)
    print("=" * 78)
    print(f"[MainLoop] Finished: {seq} cycles in {t_elapsed:.2f}s "
          f"(target {seq * CONTROL_PERIOD_S:.2f}s)")
    print(f"[MainLoop] True: pos=({x_actual:.2f}, {y_actual:.2f})  "
          f"theta={np.degrees(theta_actual):.1f}deg")
    print(f"[MainLoop] EKF:  pos=({x_ekf[0]:.2f}, {x_ekf[1]:.2f})  "
          f"theta={np.degrees(x_ekf[2]):.1f}deg")
    print(f"[MainLoop] Final localization error: {loc_err_final:.3f}m")
    print(f"[MainLoop] Avg freq: {seq / t_elapsed:.1f}Hz")


if __name__ == "__main__":
    run_main_loop()
