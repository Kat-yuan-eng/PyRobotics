# PyRobotics API 参考

## perception 模块

### lane_pixel_detector

```python
detect_lane_pixels(image, scan_rows=None, peak_margin=80, jump_threshold=100)
```

**参数**：
- `image` (ndarray): BGR 图像，形状 (H, W, 3)
- `scan_rows` (ndarray, optional): 扫描行坐标，默认从图像底部 1/2 区域均匀采样 20 行
- `peak_margin` (int): 峰值搜索半宽（像素），默认 80
- `jump_threshold` (float): 跳变滤波阈值（像素），默认 100

**返回**：
- `scan_rows` (ndarray): 扫描行 y 坐标
- `center_x` (ndarray): 车道中心 x 坐标，无效值为 NaN

```python
generate_test_image(img_w=640, img_h=480)
```

**返回**：合成测试图像 (ndarray, uint8, BGR)

---

### obstacle_detector

```python
detect_obstacles(points, ransac_iter=100, ransac_thresh=0.1,
                 voxel_size=0.1, dbscan_eps=0.5, dbscan_min=5,
                 min_height=0.2, max_aspect_ratio=10.0,
                 min_area=0.1, min_points=8)
```

**参数**：
- `points` (ndarray): 3D 点云，形状 (N, 3)
- `ransac_iter` (int): RANSAC 迭代次数，默认 100
- `ransac_thresh` (float): RANSAC 内点距离阈值 [m]，默认 0.1
- `voxel_size` (float): 体素下采样边长 [m]，默认 0.1
- `dbscan_eps` (float): DBSCAN 邻域半径 [m]，默认 0.5
- `dbscan_min` (int): DBSCAN 最小邻居数，默认 5
- `min_height` (float): 障碍物最小高度 [m]，默认 0.2
- `max_aspect_ratio` (float): 最大纵横比，默认 10.0
- `min_area` (float): 最小面积 [m²]，默认 0.1
- `min_points` (int): 最小点数，默认 8

**返回**：
- `list[dict]`: 障碍物列表，每个元素含 `center`(ndarray[2]), `center_z`(float), `length`(float), `width`(float), `heading`(float), `n_points`(int)

```python
generate_test_point_cloud()
```

**返回**：合成测试点云 (ndarray, float64, 形状 (N, 3))

---

### obstacle_tracker

```python
track_obstacles(current_obstacles, prev_tracks, max_dist=2.0, dt=0.1)
```

**参数**：
- `current_obstacles` (list[dict]): 当前帧障碍物列表
- `prev_tracks` (list[dict]): 上一帧跟踪列表
- `max_dist` (float): 最大关联距离 [m]，默认 2.0
- `dt` (float): 时间步长 [s]，默认 0.1

**返回**：
- `list[dict]`: 更新后的跟踪列表，含 `id`, `center`, `velocity`, `age`

---

### sign_recognizer

```python
recognize_signs(image, min_area=500, score_threshold=0.3)
```

**参数**：
- `image` (ndarray): BGR 图像
- `min_area` (int): 最小标志面积（像素²），默认 500
- `score_threshold` (float): HOG 匹配最低分数，默认 0.3

**返回**：
- `list[dict]`: 标志列表，每个元素含 `category`(str), `center`(tuple), `score`(float)

---

### sensor_fusion

```python
fuse_to_perception(rows, cx, obstacles, signs, K, R, t,
                   vehicle_x=0.0, vehicle_y=0.0, vehicle_theta=0.0)
```

**参数**：
- `rows` (ndarray): 扫描行 y 坐标
- `cx` (ndarray): 车道中心 x 坐标
- `obstacles` (list[dict]): 障碍物列表
- `signs` (list[dict]): 标志列表
- `K` (ndarray): 3×3 相机内参矩阵
- `R` (ndarray): 3×3 旋转矩阵（相机→车辆）
- `t` (ndarray): 3×1 平移向量
- `vehicle_x/y/theta` (float): 车辆全局位姿，用于坐标变换

**返回**：
- `PerceptionOutput`: Protobuf 消息

```python
default_camera_params()
```

**返回**：`(K, R, t)` 默认相机参数元组

---

## decision 模块

### task_scheduler

```python
schedule(perception_output, scheduler_state, v_nominal=5.0,
         avoid_trigger_dist=8.0, global_route=None)
```

**参数**：
- `perception_output` (PerceptionOutput): 感知输出
- `scheduler_state` (dict): 调度状态，含 `current_task`, `task_state`
- `v_nominal` (float): 标称速度 [m/s]，默认 5.0
- `avoid_trigger_dist` (float): 避障触发距离 [m]，默认 8.0
- `global_route` (list, optional): 全局路由坐标列表

**返回**：
- `(DecisionOutput, dict)`: 决策输出 + 更新后的调度状态

---

### path_smoother

```python
smooth_path(cx, cy, max_deviation=0.3, n_output=None)
```

**参数**：
- `cx/cy` (ndarray): 路径 x/y 坐标
- `max_deviation` (float): 最大允许偏差 [m]，默认 0.3
- `n_output` (int, optional): 输出点数，默认与输入相同

**返回**：
- `(sx, sy, kappa)`: 平滑路径 x/y 坐标 + 曲率数组

---

### obstacle_avoidance

```python
avoid_obstacles(path_x, path_y, obstacles, max_lateral=1.5,
                safety_margin=0.5, priority="left")
```

**参数**：
- `path_x/path_y` (ndarray): 参考路径坐标
- `obstacles` (list[dict]): 障碍物列表，每个含 `center`(ndarray), `length`, `width`, `heading`
- `max_lateral` (float): 最大横向偏移 [m]，默认 1.5
- `safety_margin` (float): 安全间距 [m]，默认 0.5
- `priority` (str): 优先避障方向，"left" 或 "right"

**返回**：
- `(new_x, new_y, behavior)`: 避障路径 + 行为字符串（"detour"/"stop"）

---

## PathTracking 模块

### stanley_controller

```python
stanley_control(state, cx, cy, cyaw, ck, target_speed, wheelbase,
                k_e=0.5, k_v=1.0, k_preview=0.0, max_steer_deg=45.0,
                dead_zone=0.5, v_damping=0.5, vehicle_x=0.0,
                vehicle_y=0.0, vehicle_theta=0.0)
```

**参数**：
- `state` (ndarray): [x, y, yaw, v] 车辆状态
- `cx/cy/cyaw/ck` (ndarray): 参考路径坐标/航向/曲率
- `target_speed` (float): 目标速度 [m/s]
- `wheelbase` (float): 轴距 [m]
- `k_e` (float): 横向误差增益，默认 0.5
- `k_v` (float): 速度柔化因子，默认 1.0
- `k_preview` (float): 前视距离增益，默认 0.0
- `max_steer_deg` (float): 最大转向角 [deg]，默认 45.0
- `dead_zone` (float): 转向死区 [deg]，默认 0.5
- `v_damping` (float): 低速阻尼，默认 0.5
- `vehicle_x/y/theta` (float): 车辆全局位姿

**返回**：
- `(steer_deg, accel, target_idx)`: 转向角 [deg] + 加速度 [m/s²] + 目标路径索引

---

### pure_pursuit_controller

```python
pure_pursuit_control(state, cx, cy, target_speed, wheelbase,
                     lookahead_gain=0.1, lookahead_min=2.0)
```

**返回**：`(steer_deg, accel, target_idx)`

---

### fuzzy_controller

```python
fuzzy_control(state, cx, cy, cyaw, ck, target_speed, wheelbase,
              vehicle_x=0.0, vehicle_y=0.0, vehicle_theta=0.0)
```

**返回**：`(steer_deg, accel, target_idx)`

---

### mpc_controller

```python
mpc_control(state, cx, cy, cyaw, target_speed, wheelbase,
            N=10, dt_mpc=0.1, vehicle_x=0.0, vehicle_y=0.0,
            vehicle_theta=0.0)
```

**返回**：`(steer_deg, accel, target_idx, predicted_traj)`

---

### rl_controller

```python
rl_control(state, cx, cy, target_speed, wheelbase, weights=None)
```

**返回**：`(steer_deg, accel, target_idx)`

```python
train_dqn(n_episodes=500, save_path="dqn_weights.npy")
```

---

## PathPlanning 模块

### a_star_planner

```python
a_star_plan(sx, sy, gx, gy, obstacles, resolution=0.5,
            robot_radius=0.5)
```

**参数**：
- `sx/sy` (float): 起点
- `gx/gy` (float): 终点
- `obstacles` (list[tuple]): 障碍物中心坐标列表
- `resolution` (float): 栅格分辨率 [m]
- `robot_radius` (float): 机器人半径 [m]

**返回**：`(path_x, path_y, stats)` 路径坐标 + 统计信息字典

---

### rrt_planner

```python
rrt_plan(sx, sy, gx, gy, obstacles, expand_dis=3.0,
         path_resolution=0.5, goal_sample_rate=5, max_iter=500,
         robot_radius=0.5)
```

**返回**：`(path_x, path_y, stats)`

---

### dwa_planner

```python
dwa_plan(state, goal, obstacles, config=None)
```

**返回**：`(v, omega, trajectory, stats)`

---

## Localization 模块

### ekf_localizer

```python
ekf_localize(x_est, P_est, u, z, dt, Q=None, R=None)
```

**参数**：
- `x_est` (ndarray): [x, y, yaw, v] 状态估计
- `P_est` (ndarray): 4×4 协方差矩阵
- `u` (ndarray): [v, omega] 控制输入
- `z` (ndarray): [x_gps, y_gps] GPS 观测（可为 None）
- `dt` (float): 时间步长 [s]

**返回**：`(x_est, P_est)`

---

### pf_localizer

```python
pf_localize(particles, u, z_landmarks, dt, R_motion, Q_obs)
```

**参数**：
- `particles` (list[dict]): 粒子列表，每个含 `pose`(ndarray[3]), `weight`(float)
- `u` (ndarray): [v, omega] 控制输入
- `z_landmarks` (list): 路标观测列表
- `dt` (float): 时间步长

**返回**：`particles`（更新后的粒子列表）

---

### fusion_localizer

```python
fusion_localize(x_ekf, P_ekf, x_pf, P_pf)
```

**返回**：`(x_fused, P_fused)`

---

## SLAM 模块

### fast_slam

```python
fast_slam(particles, u, z, n_landmarks, dt, R_motion, Q_obs)
```

**返回**：`particles`（更新后的粒子列表）

---

### icp_matching

```python
icp_match(prev_points, curr_points, max_iter=100, eps=1e-4,
          outlier_ratio=3.0, init_pose=None)
```

**参数**：
- `prev_points` (ndarray): 前帧点云 (N, 2)
- `curr_points` (ndarray): 当前帧点云 (M, 2)
- `max_iter` (int): 最大迭代次数
- `eps` (float): 收敛阈值
- `outlier_ratio` (float): 异常值比率阈值，默认 3.0
- `init_pose` (ndarray, optional): 初始位姿猜测 [x, y, theta]

**返回**：`(R, T, error)` 旋转矩阵 + 平移向量 + 残差

---

## system 模块

### vehicle_sim

```python
simulate_vehicle(state, steer_deg, throttle, brake, dt, wheelbase)
```

**参数**：
- `state` (ndarray): [x, y, yaw, v]
- `steer_deg` (float): 转向角 [deg]
- `throttle` (float): 油门 [m/s²]
- `brake` (float): 制动力 [m/s²]
- `dt` (float): 时间步长 [s]
- `wheelbase` (float): 轴距 [m]

**返回**：`state`（更新后的状态）

---

## utils 模块

### plot

```python
generate_serpentine_course(n_points=200, amplitude=3.0, length=50.0)
```

**返回**：`(cx, cy, cyaw, ck)` 蛇形路径坐标/航向/曲率

```python
generate_circle_course(radius=20.0, n_points=200)
```

**返回**：`(cx, cy, cyaw, ck)` 圆形路径

```python
cubic_spline_course(waypoints_x, waypoints_y, ds=0.1)
```

**返回**：`(cx, cy, cyaw, ck, s)` 三次样条路径 + 弧长

### angle

```python
normalize_angle(angle)
```

**返回**：归一化到 [-π, π] 的角度
