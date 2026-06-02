import sys
import os
import traceback
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "generated"))
sys.path.insert(0, ROOT)

results = {}

def test_module(name, fn):
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - t0
        results[name] = ("PASS", elapsed)
        print(f"  [PASS] {name} ({elapsed*1000:.1f}ms)")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        results[name] = ("FAIL", elapsed)
        print(f"  [FAIL] {name} ({elapsed*1000:.1f}ms)")
        traceback.print_exc()
        print()

# === Perception Layer ===

def test_lane_pixel_detector():
    from perception.lane_pixel_detector import detect_lane_pixels, generate_test_image
    img = generate_test_image()
    scan_rows, center_x = detect_lane_pixels(img)
    assert len(center_x) > 0, "no center_x detected"
    valid_cx = [c for c in center_x if not np.isnan(c)]
    assert len(valid_cx) > 0, "no valid cx detected"
    mean_cx = np.mean(valid_cx)
    assert 250 < mean_cx < 390, f"mean_cx={mean_cx} out of range"

def test_obstacle_detector():
    from perception.obstacle_detector import detect_obstacles, generate_test_point_cloud
    pc = generate_test_point_cloud()
    obs = detect_obstacles(pc)
    assert len(obs) >= 2, f"expected >=2 obstacles, got {len(obs)}"
    for o in obs:
        assert "center" in o, "missing center"
        assert "center_z" in o, "missing center_z"
        assert "length" in o, "missing length"
        assert "width" in o, "missing width"
        assert "n_points" in o, "missing n_points"
        assert "type" in o, "missing type"
        assert "height" in o, "missing height"
    has_vehicle = any(o['type'] == 'OBSTACLE_VEHICLE' for o in obs)
    assert has_vehicle, "expected at least one OBSTACLE_VEHICLE"

def test_obstacle_tracker():
    from perception.obstacle_tracker import track_obstacles, get_confirmed_tracks
    import numpy as np
    rng = np.random.default_rng(42)
    tracks = []
    free_ids = set()
    for frame_i in range(15):
        detections = []
        for d in range(3):
            cx = 5.0 + d * 5.0 + rng.normal(0, 0.3)
            cy = 5.0 + 2.0 * np.sin(frame_i * 0.15 + d) + rng.normal(0, 0.2)
            detections.append({
                "center": np.array([cx, cy]),
                "length": 1.5, "width": 0.8,
                "heading": 0.0, "type": "OBSTACLE_UNKNOWN",
            })
        tracks = track_obstacles(detections, tracks, dt=0.1, free_ids=free_ids)
    confirmed = get_confirmed_tracks(tracks)
    assert len(confirmed) >= 1, f"expected >=1 confirmed, got {len(confirmed)}"
    for t in confirmed:
        assert "id" in t, "missing id"
        assert "vx" in t, "missing vx"
        assert "vy" in t, "missing vy"
        assert "speed" in t, "missing speed"
        assert "type" in t, "missing type"
    track_ids = [t['id'] for t in confirmed]
    assert len(track_ids) == len(set(track_ids)), f"ID collision: {track_ids}"

def test_sign_recognizer():
    from perception.sign_recognizer import recognize_signs, generate_test_sign_image
    img = generate_test_sign_image()
    signs = recognize_signs(img)
    assert isinstance(signs, list), "signs not list"
    if len(signs) > 0:
        assert "color" in signs[0], "missing color"
        assert "shape" in signs[0], "missing shape"
        assert "category" in signs[0], "missing category"
    has_red = any(s['color'] == 'red' for s in signs)
    has_blue = any(s['color'] == 'blue' for s in signs)
    has_yellow = any(s['color'] == 'yellow' for s in signs)
    assert has_red or has_blue or has_yellow, "no colored signs detected"

def test_sensor_fusion():
    from perception.lane_pixel_detector import detect_lane_pixels, generate_test_image as gen_lane
    from perception.obstacle_detector import detect_obstacles, generate_test_point_cloud
    from perception.sign_recognizer import recognize_signs, generate_test_sign_image
    from perception.sensor_fusion import fuse_to_perception, default_camera_params
    img = gen_lane()
    pc = generate_test_point_cloud()
    sign_img = generate_test_sign_image()
    rows, cx = detect_lane_pixels(img)
    obs = detect_obstacles(pc)
    signs = recognize_signs(sign_img)
    K, R, t = default_camera_params()
    out = fuse_to_perception(rows, cx, obs, signs, K, R, t)
    assert out.road_valid, "road not valid"
    assert out.obstacle_count >= 0, "negative obstacle count"

# === Decision Layer ===

def test_path_smoother():
    from decision.path_smoother import smooth_path
    px = np.linspace(0, 10, 20)
    py = np.sin(px * 0.5)
    sx, sy, kappa = smooth_path(px, py, max_deviation=0.3, n_output=50)
    assert len(sx) == 50, f"expected 50, got {len(sx)}"
    assert len(kappa) == 50, f"expected 50 kappa, got {len(kappa)}"
    diffs = np.sqrt((sx[:, None] - px[None, :])**2 + (sy[:, None] - py[None, :])**2)
    max_dev = diffs.min(axis=0).max()
    assert max_dev <= 0.31, f"max_dev={max_dev:.3f} > 0.3"

def test_obstacle_avoidance():
    from decision.obstacle_avoidance import avoid_obstacles
    px = np.linspace(0, 10, 20)
    py = np.zeros(20)
    obs = [{"center": np.array([5.0, 0.0]), "length": 1.0, "width": 1.0, "heading": 0.0}]
    nx, ny, beh = avoid_obstacles(px, py, obs)
    assert len(nx) == 20, f"expected 20, got {len(nx)}"
    assert beh in ("left", "right", "none", "detour"), f"unexpected behavior: {beh}"

def test_task_scheduler():
    from decision.task_scheduler import schedule, TASK_PATROL, TASK_AVOID
    from perception.sensor_fusion import PerceptionOutput
    from generated import Vec2, Pose2D
    percep = PerceptionOutput()
    percep.road_valid = True
    percep.obstacle_count = 0
    for i in range(5):
        pt = percep.road.center_line.add()
        pt.x = 2.0 + i * 0.5
        pt.y = 0.0
    percep.ego_pose.theta = 0.0
    state = {"current_task": TASK_PATROL, "task_state": {}}
    dec, state = schedule(percep, state, v_nominal=5.0)
    assert dec.target_speed >= 0, f"target_speed={dec.target_speed}"

    state = {"current_task": TASK_PATROL, "task_state": {}}
    speed_hist = []
    for step in range(30):
        dec, state = schedule(percep, state, v_nominal=5.0)
        speed_hist.append(dec.target_speed)
    changes = [abs(speed_hist[i] - speed_hist[i - 1]) for i in range(1, len(speed_hist))]
    max_change = max(changes) if changes else 0
    assert max_change < 2.0, f"speed jump {max_change:.2f} > 2.0 m/s per step"

def test_multi_agent():
    from decision.multi_agent import build_ego_state
    from generated import DecisionOutput
    dec = DecisionOutput()
    dec.target_speed = 5.0
    agent = build_ego_state(0, dec, 1.0, 2.0, 0.5, 3.0, 0.0)
    assert agent.agent_id == 0, "agent_id mismatch"

# === Control Layer ===

def test_stanley():
    from PathTracking.stanley_controller import stanley_control
    from generated import DecisionOutput, PathPoint, Behavior
    dec = DecisionOutput()
    dec.behavior = Behavior.Value("BEHAVIOR_LANE_KEEP")
    dec.target_speed = 5.0
    for i in range(10):
        pp = dec.target_path.add()
        pp.pose.x = 2.0 + i * 0.5
        pp.pose.y = 0.0
        pp.pose.theta = 0.0
        pp.curvature = 0.0
    ctrl, state = stanley_control(dec, speed_actual=2.0, pid_state=None, dt=0.02)
    assert ctrl.control_valid, "control not valid"
    assert "steer_prev" in state, "missing steer_prev in state"
    assert "d_filtered" in state, "missing d_filtered in state"
    assert "a_prev" in state, "missing a_prev in state"
    assert "brake_prev" in state, "missing brake_prev in state"

def test_fuzzy():
    from PathTracking.fuzzy_controller import fuzzy_control, _fuzzy_infer
    from generated import DecisionOutput, PathPoint, Behavior
    dec = DecisionOutput()
    dec.behavior = Behavior.Value("BEHAVIOR_LANE_KEEP")
    dec.target_speed = 5.0
    for i in range(10):
        pp = dec.target_path.add()
        pp.pose.x = 2.0 + i * 0.5
        pp.pose.y = 0.0
        pp.pose.theta = 0.0
        pp.curvature = 0.0
    ctrl, state = fuzzy_control(dec, speed_actual=2.0, steer_state=None, fuzzy_state=None, dt=0.02)
    assert ctrl.control_valid, "control not valid"

    du_accel = _fuzzy_infer(0.0, -3.0)
    assert du_accel > 0, f"kappa=0 dv=-3 should accelerate, got du={du_accel}"
    du_decel = _fuzzy_infer(0.0, 3.0)
    assert du_decel < 0, f"kappa=0 dv=3 should decelerate, got du={du_decel}"
    du_curve = _fuzzy_infer(0.5, 0.0)
    assert du_curve < 0, f"kappa=0.5 dv=0 should decelerate for safety, got du={du_curve}"

def test_mpc():
    from PathTracking.mpc_controller import mpc_control
    from generated import DecisionOutput, PathPoint, Behavior
    dec = DecisionOutput()
    dec.behavior = Behavior.Value("BEHAVIOR_LANE_KEEP")
    dec.target_speed = 3.0
    for i in range(10):
        pp = dec.target_path.add()
        pp.pose.x = 2.0 + i * 0.5
        pp.pose.y = 0.0
        pp.pose.theta = 0.0
        pp.curvature = 0.0
    ctrl, state = mpc_control(dec, speed_actual=2.0, mpc_state=None, dt=0.02)
    assert ctrl.control_valid, "control not valid"

def test_rl():
    from PathTracking.rl_controller import rl_control
    from generated import DecisionOutput, PathPoint, Behavior
    dec = DecisionOutput()
    dec.behavior = Behavior.Value("BEHAVIOR_LANE_KEEP")
    dec.target_speed = 3.0
    for i in range(10):
        pp = dec.target_path.add()
        pp.pose.x = 2.0 + i * 0.5
        pp.pose.y = 0.0
        pp.pose.theta = 0.0
        pp.curvature = 0.0
    ctrl, state = rl_control(dec, speed_actual=2.0, weights=None, rl_state=None, dt=0.02)
    assert ctrl.steering.steering_valid, "steering not valid"
    assert ctrl.throttle_brake.throttle_valid, "throttle not valid"

def test_controller_selector():
    from PathTracking.controller_selector import select_controller, CTRL_STANLEY, CTRL_FUZZY, ctrl_name
    from generated import DecisionOutput, PathPoint, Behavior
    dec = DecisionOutput()
    dec.behavior = Behavior.Value("BEHAVIOR_LANE_KEEP")
    dec.target_speed = 5.0
    for i in range(10):
        pp = dec.target_path.add()
        pp.pose.x = 2.0 + i * 0.5
        pp.pose.y = 0.0
        pp.pose.theta = 0.0
        pp.curvature = 0.0
    for ctype in [CTRL_STANLEY, CTRL_FUZZY]:
        ctrl, state = select_controller(ctype, dec, speed_actual=2.0, state=None, dt=0.02)
        assert ctrl.control_valid, f"{ctrl_name(ctype)} not valid"

# === System Layer ===

def test_embedded_c():
    import subprocess
    r = subprocess.run(
        ["gcc", "-Wall", "-Wextra", "-std=c99", "-I", "inc",
         "-c", "src/lookup_table.c", "-o", "build/lookup_table.o"],
        capture_output=True, text=True, cwd=os.path.join(ROOT, "system", "embedded")
    )
    assert r.returncode == 0, f"lookup_table.c compile failed: {r.stderr}"
    r = subprocess.run(
        ["gcc", "-Wall", "-Wextra", "-std=c99", "-I", "inc",
         "-c", "src/stanley.c", "-o", "build/stanley.o"],
        capture_output=True, text=True, cwd=os.path.join(ROOT, "system", "embedded")
    )
    assert r.returncode == 0, f"stanley.c compile failed: {r.stderr}"
    r = subprocess.run(
        ["gcc", "-Wall", "-Wextra", "-std=c99", "-I", "inc",
         "-c", "src/pid.c", "-o", "build/pid.o"],
        capture_output=True, text=True, cwd=os.path.join(ROOT, "system", "embedded")
    )
    assert r.returncode == 0, f"pid.c compile failed: {r.stderr}"
    r = subprocess.run(
        ["gcc", "-Wall", "-Wextra", "-std=c99", "-I", "inc",
         "-c", "src/ekf_localize.c", "-o", "build/ekf_localize.o"],
        capture_output=True, text=True, cwd=os.path.join(ROOT, "system", "embedded")
    )
    assert r.returncode == 0, f"ekf_localize.c compile failed: {r.stderr}"
    r = subprocess.run(
        ["gcc", "-Wall", "-Wextra", "-std=c99", "-I", "inc",
         "-c", "src/slam_core.c", "-o", "build/slam_core.o"],
        capture_output=True, text=True, cwd=os.path.join(ROOT, "system", "embedded")
    )
    assert r.returncode == 0, f"slam_core.c compile failed: {r.stderr}"
    r = subprocess.run(
        ["gcc", "-Wall", "-Wextra", "-std=c99", "-I", "inc",
         "-c", "src/vehicle_control.c", "-o", "build/vehicle_control.o"],
        capture_output=True, text=True, cwd=os.path.join(ROOT, "system", "embedded")
    )
    assert r.returncode == 0, f"vehicle_control.c compile failed: {r.stderr}"

def test_realtime():
    from system.realtime.latest_value import LatestResult
    lr = LatestResult()
    lr.write("test_data")
    data, seq = lr.read()
    assert data == "test_data", f"expected test_data, got {data}"
    assert seq == 1, f"expected seq=1, got {seq}"
    lr.write("new_data")
    data, seq = lr.read()
    assert data == "new_data", f"expected new_data, got {data}"
    assert seq == 2, f"expected seq=2, got {seq}"

def test_ros2_framework():
    ros2_dir = os.path.join(ROOT, "system", "ros2")
    assert os.path.isdir(os.path.join(ros2_dir, "smart_car_interfaces")), "missing smart_car_interfaces"
    assert os.path.isdir(os.path.join(ros2_dir, "smart_car_control_system")), "missing smart_car_control_system"
    msg_dir = os.path.join(ros2_dir, "smart_car_interfaces", "msg")
    expected_msgs = ["Header.msg", "Vec2.msg", "Vec3.msg", "Pose2D.msg", "Twist2D.msg",
                     "Obstacle.msg", "RoadBoundary.msg", "PerceptionOutput.msg",
                     "PathPoint.msg", "TargetObject.msg", "DecisionOutput.msg",
                     "SteeringCommand.msg", "ThrottleCommand.msg", "ControlOutput.msg",
                     "VehicleFeedback.msg", "SystemCommand.msg"]
    for msg in expected_msgs:
        assert os.path.isfile(os.path.join(msg_dir, msg)), f"missing {msg}"
    node_dir = os.path.join(ros2_dir, "smart_car_control_system", "smart_car_control_system")
    for node in ["perception_node.py", "planning_node.py", "control_node.py"]:
        assert os.path.isfile(os.path.join(node_dir, node)), f"missing {node}"

# === Localization Layer ===

def test_ekf_localizer():
    from Localization.ekf_localizer import ekf_localize, generate_ekf_test
    dt = 0.1
    Q = np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0]) ** 2
    R = np.diag([1.0, 1.0]) ** 2
    true_states, gps_obs, controls = generate_ekf_test(n_steps=100, dt=dt)
    x_est = true_states[0].copy()
    P_est = np.eye(4) * 0.01
    est_states = np.zeros((100, 4))
    for i in range(100):
        z = gps_obs[i] if not np.any(np.isnan(gps_obs[i])) else None
        x_est, P_est = ekf_localize(x_est, P_est, controls[i], z, Q, R, dt)
        est_states[i] = x_est
    err = est_states[:, :2] - true_states[:, :2]
    rmse = np.sqrt(np.mean(err ** 2))
    assert rmse < 2.0, f"EKF RMSE={rmse:.3f}m too high"

def test_pf_localizer():
    from Localization.pf_localizer import pf_localize, pf_init, pf_estimate, generate_pf_test
    dt = 0.1
    NP = 100
    NTh = NP / 2.0
    R_motion = np.diag([0.5, np.deg2rad(10.0)]) ** 2
    Q_obs = np.diag([0.2, np.deg2rad(5.0)]) ** 2
    true_states, observations, controls, landmarks = generate_pf_test(n_steps=100, dt=dt, n_landmarks=5)
    x_init = true_states[0]
    px, pw = pf_init(NP, x_init, 0.5)
    est_states = np.zeros((100, 4))
    for i in range(100):
        z = observations[i]
        x_est, P_est, px, pw = pf_localize(px, pw, controls[i], z, landmarks, R_motion, Q_obs, dt, NTh)
        est_states[i] = x_est
    err = est_states[:, :2] - true_states[:, :2]
    rmse = np.sqrt(np.mean(err ** 2))
    assert rmse < 2.0, f"PF RMSE={rmse:.3f}m too high"

def test_fusion_localizer():
    from Localization.fusion_localizer import fusion_localize
    from Localization.ekf_localizer import generate_ekf_test
    from Localization.pf_localizer import pf_init, generate_pf_test
    dt = 0.1
    NP = 100
    NTh = NP / 2.0
    Q_ekf = np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0]) ** 2
    R_ekf = np.diag([1.0, 1.0]) ** 2
    R_motion_pf = np.diag([0.5, np.deg2rad(10.0)]) ** 2
    Q_obs_pf = np.diag([0.2, np.deg2rad(5.0)]) ** 2
    true_states, gps_obs, controls_ekf = generate_ekf_test(n_steps=100, dt=dt)
    _, observations, _, landmarks = generate_pf_test(n_steps=100, dt=dt, n_landmarks=5)
    x_init = true_states[0]
    x_ekf = x_init.copy()
    P_ekf = np.eye(4)
    px, pw = pf_init(NP, x_init, 0.5)
    est_fused = np.zeros((100, 4))
    for i in range(100):
        z_gps = gps_obs[i] if not np.any(np.isnan(gps_obs[i])) else None
        z_lm = observations[i]
        x_fused, P_fused, (x_ekf, P_ekf), (px, pw) = fusion_localize(
            (x_ekf, P_ekf), (px, pw), controls_ekf[i], z_gps, z_lm, landmarks,
            Q_ekf, R_ekf, R_motion_pf, Q_obs_pf, dt, NTh)
        est_fused[i] = x_fused
    err = est_fused[:, :2] - true_states[:, :2]
    rmse = np.sqrt(np.mean(err ** 2))
    assert rmse < 2.0, f"Fusion RMSE={rmse:.3f}m too high"

# === Planning Layer ===

def test_a_star_planner():
    from PathPlanning.a_star_planner import a_star_plan
    ox = np.array([10, 20, 30])
    oy = np.array([5, 5, 5])
    rx, ry, stats = a_star_plan(0, 0, 40, 20, ox, oy, resolution=2.0, robot_radius=1.0)
    assert len(rx) > 0, "A* found no path"
    assert stats["path_length"] > 0, "A* path length = 0"
    assert stats["planner"] == "A*", "planner name mismatch"

def test_rrt_planner():
    from PathPlanning.rrt_planner import rrt_plan
    obstacle_list = [(10, 5, 1), (20, 5, 1)]
    rx, ry, stats = rrt_plan(0, 0, 30, 15, obstacle_list,
                              rand_area=(-5, 35), expand_dis=3.0, max_iter=300)
    assert stats["planner"] == "RRT", "planner name mismatch"
    assert stats["nodes_explored"] > 0, "RRT explored 0 nodes"

def test_dwa_planner():
    from PathPlanning.dwa_planner import dwa_plan
    x_state = np.array([0.0, 0.0, 0.0, 0.5, 0.0])
    goal = np.array([10.0, 10.0])
    ob = np.array([[5.0, 5.0], [6.0, 6.0]])
    best_u, best_traj, stats = dwa_plan(x_state, goal, ob, max_speed=1.0)
    assert stats["planner"] == "DWA", "planner name mismatch"
    assert len(best_traj) > 0, "DWA produced empty trajectory"
    assert len(best_u) == 2, "DWA control dimension mismatch"

def test_planner_selector():
    from PathPlanning.planner_selector import select_planner, compare_planners, create_benchmark_scenarios
    assert select_planner("static_sparse") == "A*", "static_sparse should select A*"
    assert select_planner("dynamic") == "DWA", "dynamic should select DWA"
    assert select_planner("realtime") == "DWA", "realtime should select DWA"
    scenarios = create_benchmark_scenarios()
    assert len(scenarios) == 3, f"expected 3 scenarios, got {len(scenarios)}"
    results = compare_planners(scenarios[:1])
    assert len(results) == 3, f"expected 3 results (A*/RRT/DWA), got {len(results)}"

# === SLAM Layer ===

def test_icp_matching():
    from SLAM.icp_matching import icp_match
    rng = np.random.default_rng(42)
    n_pts = 50
    angle = np.deg2rad(15)
    R_true = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    T_true = np.array([0.5, -0.3])
    pts = rng.uniform(-5, 5, (2, n_pts))
    transformed = R_true @ pts + T_true[:, None]
    R_est, T_est, err_hist = icp_match(pts, transformed, max_iter=50)
    pos_err = np.linalg.norm(T_est - T_true)
    angle_est = np.rad2deg(np.arctan2(R_est[1, 0], R_est[0, 0]))
    angle_true = np.rad2deg(np.arctan2(R_true[1, 0], R_true[0, 0]))
    assert pos_err < 0.5, f"ICP translation error={pos_err:.3f}m too large"
    assert abs(angle_est - angle_true) < 5.0, f"ICP angle error={abs(angle_est - angle_true):.1f}deg too large"
    assert len(err_hist) > 0, "ICP no error history"
    assert err_hist[-1] < err_hist[0], "ICP did not converge"

    R_est2, T_est2, err_hist2 = icp_match(pts, transformed, max_iter=50,
                                            init_pose=np.eye(3))
    pos_err2 = np.linalg.norm(T_est2 - T_true)
    assert pos_err2 < 1.0, f"ICP with init_pose error={pos_err2:.3f}m too large"

def test_fast_slam():
    from SLAM.fast_slam import fast_slam, estimate_from_particles, create_particle, generate_slam_test
    dt = 0.1
    NP = 30
    NTh = NP / 1.5
    Q = np.diag([0.2, np.deg2rad(5.0)]) ** 2
    R_motion = np.diag([0.5, np.deg2rad(10.0)]) ** 2
    true_traj, obs_seq, controls, landmarks = generate_slam_test(n_steps=50, dt=dt, n_landmarks=5)
    particles = [create_particle(true_traj[0, 0], true_traj[0, 1], true_traj[0, 2],
                                  n_landmarks=5) for _ in range(NP)]
    est_traj = np.zeros((50, 3))
    for i in range(50):
        particles, _, _ = fast_slam(particles, controls[i], obs_seq[i], Q, R_motion, dt, NTh)
        est_traj[i], _ = estimate_from_particles(particles)
    err = est_traj[:, :2] - true_traj[:, :2]
    rmse = np.sqrt(np.mean(err ** 2))
    assert rmse < 3.0, f"FastSLAM RMSE={rmse:.3f}m too high"

def test_slam_pipeline():
    from SLAM.slam_pipeline import run_slam_pipeline
    rmse_hist, map_err, est_traj = run_slam_pipeline(n_steps=50, dt=0.1, n_landmarks=5, n_particles=30)
    assert len(rmse_hist) == 50, f"expected 50 rmse steps, got {len(rmse_hist)}"
    assert est_traj.shape == (50, 3), f"est_traj shape={est_traj.shape}"
    assert map_err < 5.0, f"map error={map_err:.3f}m too high"

# === Main Loop ===

def test_main_loop():
    from main_loop import run_main_loop
    run_main_loop(duration_s=2.0)

# === Run All ===

print("=" * 70)
print("全模块调试测试")
print("=" * 70)

print("\n--- 感知层 ---")
test_module("lane_pixel_detector", test_lane_pixel_detector)
test_module("obstacle_detector", test_obstacle_detector)
test_module("obstacle_tracker", test_obstacle_tracker)
test_module("sign_recognizer", test_sign_recognizer)
test_module("sensor_fusion", test_sensor_fusion)

print("\n--- 决策层 ---")
test_module("path_smoother", test_path_smoother)
test_module("obstacle_avoidance", test_obstacle_avoidance)
test_module("task_scheduler", test_task_scheduler)
test_module("multi_agent", test_multi_agent)

print("\n--- 控制层 ---")
test_module("stanley_controller", test_stanley)
test_module("fuzzy_controller", test_fuzzy)
test_module("mpc_controller", test_mpc)
test_module("rl_controller", test_rl)
test_module("controller_selector", test_controller_selector)

print("\n--- 系统层 ---")
test_module("embedded_c_compile", test_embedded_c)
test_module("realtime_latest_value", test_realtime)
test_module("ros2_framework", test_ros2_framework)

print("\n--- 定位层 ---")
test_module("ekf_localizer", test_ekf_localizer)
test_module("pf_localizer", test_pf_localizer)
test_module("fusion_localizer", test_fusion_localizer)

print("\n--- 规划层 ---")
test_module("a_star_planner", test_a_star_planner)
test_module("rrt_planner", test_rrt_planner)
test_module("dwa_planner", test_dwa_planner)
test_module("planner_selector", test_planner_selector)

print("\n--- SLAM层 ---")
test_module("icp_matching", test_icp_matching)
test_module("fast_slam", test_fast_slam)
test_module("slam_pipeline", test_slam_pipeline)

print("\n--- 主循环 ---")
test_module("main_loop_2s", test_main_loop)

print("\n" + "=" * 70)
n_pass = sum(1 for v in results.values() if v[0] == "PASS")
n_fail = sum(1 for v in results.values() if v[0] == "FAIL")
total_time = sum(v[1] for v in results.values())
print(f"结果: {n_pass} PASS / {n_fail} FAIL / {len(results)} TOTAL ({total_time:.1f}s)")
if n_fail > 0:
    print("失败模块:")
    for name, (status, _) in results.items():
        if status == "FAIL":
            print(f"  - {name}")
print("=" * 70)
