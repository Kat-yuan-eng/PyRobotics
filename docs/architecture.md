# PyRobotics 架构文档

## 系统总览

PyRobotics 是一个智能汽车自动驾驶软件栈，按四层架构组织：**感知→决策→控制→系统**，所有模块通过 Protobuf 消息解耦，形成严格的线性数据流水线。

```
┌─────────────────────────────────────────────────────────────┐
│                      PyRobotics 系统架构                      │
├─────────┬──────────┬──────────┬──────────┬──────────────────┤
│  感知层  │  决策层   │  控制层   │  定位层   │     SLAM层       │
│Perception│ Decision │PathTrack │Localization│     SLAM        │
├─────────┼──────────┼──────────┼──────────┼──────────────────┤
│车道线检测│任务调度   │Stanley   │EKF       │FastSLAM 2.0     │
│障碍物检测│路径平滑   │PurePursuit│粒子滤波  │ICP配准          │
│标志识别  │障碍物避障 │模糊控制   │协方差融合 │SLAM流水线       │
│传感器融合│多智能体   │MPC       │          │                 │
│障碍物跟踪│          │DQN       │          │                 │
├─────────┴──────────┴──────────┴──────────┴──────────────────┤
│                    系统基础设施层                               │
│  vehicle_sim │ realtime_pipeline │ embedded C │ ROS2          │
└─────────────────────────────────────────────────────────────┘
```

## 目录结构

```
PyRobotics/
├── perception/          # 感知层：车道线/障碍物/标志检测与融合
│   ├── lane_pixel_detector.py
│   ├── obstacle_detector.py
│   ├── obstacle_tracker.py
│   ├── sign_recognizer.py
│   └── sensor_fusion.py
├── decision/            # 决策层：任务调度/路径平滑/避障
│   ├── task_scheduler.py
│   ├── path_smoother.py
│   ├── obstacle_avoidance.py
│   └── multi_agent.py
├── PathTracking/        # 控制层：5种路径跟踪算法
│   ├── stanley_controller.py
│   ├── pure_pursuit_controller.py
│   ├── fuzzy_controller.py
│   ├── mpc_controller.py
│   ├── rl_controller.py
│   └── controller_selector.py
├── PathPlanning/        # 规划层：3种路径规划算法
│   ├── a_star_planner.py
│   ├── rrt_planner.py
│   ├── dwa_planner.py
│   └── planner_selector.py
├── Localization/        # 定位层：3种定位算法
│   ├── ekf_localizer.py
│   ├── pf_localizer.py
│   └── fusion_localizer.py
├── SLAM/                # SLAM层：同时定位与地图构建
│   ├── fast_slam.py
│   ├── icp_matching.py
│   └── slam_pipeline.py
├── Mapping/             # 建图（预留）
├── MissionPlanning/     # 任务规划（预留）
├── system/              # 系统基础设施
│   ├── vehicle_sim.py
│   ├── realtime/
│   ├── embedded/
│   └── ros2/
├── utils/               # 工具函数
│   ├── plot.py
│   ├── angle.py
│   └── geometry.py
├── generated/           # Protobuf 生成代码
├── proto/               # Protobuf 消息定义
├── tests/               # 单元测试
├── docs/                # 文档
└── requirements/        # 依赖
```

## 数据流

### 主循环数据流

```
摄像头/激光雷达
      │
      ▼
┌─────────────┐
│  Perception  │  lane_pixel_detector + obstacle_detector + sign_recognizer + sensor_fusion
└──────┬──────┘
       │ PerceptionOutput (protobuf)
       ▼
┌─────────────┐
│  Decision    │  task_scheduler (FSM: PATROL→AVOID→PARK) + obstacle_avoidance
└──────┬──────┘
       │ DecisionOutput (protobuf)
       ▼
┌─────────────┐
│ PathTracking │  stanley / pure_pursuit / fuzzy / mpc / rl → controller_selector
└──────┬──────┘
       │ ControlOutput (protobuf)
       ▼
┌─────────────┐
│  System      │  vehicle_sim (自行车模型) → 状态更新
└─────────────┘
```

### 定位数据流

```
GPS观测 ──────→ EKF ──→ (x_ekf, P_ekf) ──┐
                                            ├──→ Fusion ──→ (x_fused, P_fused)
路标观测 ─────→ PF  ──→ (x_pf, P_pf) ────┘
```

### Protobuf 消息体系

| 消息类型 | 定义文件 | 生产者 | 消费者 |
|----------|---------|--------|--------|
| PerceptionOutput | perception.proto | sensor_fusion | task_scheduler |
| DecisionOutput | decision.proto | task_scheduler | controller_selector |
| ControlOutput | control.proto | controller_selector | vehicle_sim |
| AgentState | agent.proto | multi_agent | coordinate_avoidance |
| SystemCommand | system.proto | 外部输入 | realtime_pipeline |

## 坐标系约定

- **车辆坐标系**：原点在后轴中心，x 轴朝前，y 轴朝左，z 轴朝上
- **全局坐标系**：原点在起始位置，x 轴朝东，y 轴朝北
- **坐标变换**：`x_g = x_v*cos(θ) - y_v*sin(θ) + X`，`y_g = x_v*sin(θ) + y_v*cos(θ) + Y`
- **传感器融合**：图像像素→车辆坐标（针孔模型反投影）
- **控制器**：全局坐标→车辆坐标（用于横向误差计算）

## 编码规范

- 纯函数 + 数组数据流，禁 class（仅 `LatestResult` 和 `_GridWorld` 例外）
- 变量链式后缀标记加工状态（_raw→_u→_det→_smooth）
- 按 Phase 组织代码，以 `# === Phase N: 描述 ===` 分隔
- 模块间通过 Protobuf 消息解耦，禁止跨脚本内存 import
- 全向量化运算，禁 for 遍历数组元素（收敛性外循环例外）
- 动画可视化遵循 PythonRobotics 规范：show_animation 模块级变量控制，plt.cla()+plt.pause(0.001) 逐帧刷新
