# PyRobotics 快速入门

## 环境要求

- Python >= 3.9
- NumPy
- SciPy
- Matplotlib
- OpenCV (cv2)
- scikit-learn (DBSCAN聚类)
- protobuf (消息序列化)
- pytest (单元测试)

## 安装

```bash
cd PyRobotics
pip install -r requirements/requirements.txt
```

### Protobuf 编译

如需修改 `.proto` 文件后重新生成 Python 代码：

```bash
python -m grpc_tools.protoc -I=proto --python_out=generated proto/common.proto proto/control.proto proto/perception.proto proto/decision.proto proto/system.proto proto/agent.proto
```

## 运行方式

### 1. 单模块可视化

每个算法模块都可以直接运行，展示 PythonRobotics 风格的交互式动画：

```bash
# 路径跟踪
python PathTracking/stanley_controller.py
python PathTracking/pure_pursuit_controller.py
python PathTracking/fuzzy_controller.py
python PathTracking/mpc_controller.py
python PathTracking/rl_controller.py

# 路径规划
python PathPlanning/a_star_planner.py
python PathPlanning/rrt_planner.py
python PathPlanning/dwa_planner.py

# 定位
python Localization/ekf_localizer.py
python Localization/pf_localizer.py
python Localization/fusion_localizer.py

# SLAM
python SLAM/fast_slam.py
python SLAM/icp_matching.py

# 感知
python perception/lane_pixel_detector.py
python perception/obstacle_detector.py

# 决策
python decision/obstacle_avoidance.py
python decision/task_scheduler.py
```

所有模块通过 `show_animation = True` 控制动画开关。设为 `False` 则仅输出数值结果。

### 2. 主循环仿真

```bash
python main_loop.py
```

单线程 50Hz 闭环仿真：感知→决策→控制→车辆仿真，EKF 定位接入主循环。

### 3. 实时流水线

```bash
python -m system.realtime.realtime_pipeline
```

双线程实时执行：感知线程 30ms 周期 + 控制线程 50Hz 硬实时。

### 4. 单元测试

```bash
python -m pytest tests/ -v
```

### 5. 规划器对比

```bash
python -m PathPlanning.planner_selector
```

A*/RRT/DWA 三种规划器基准测试。

## 动画规范

所有可视化遵循 PythonRobotics 动画规范：

| 元素 | 颜色 | 标记 | 说明 |
|------|------|------|------|
| 参考路径 | 红色 | `.r` | 目标跟踪路径 |
| 实际轨迹 | 蓝色 | `-b` | 车辆实际行驶轨迹 |
| 目标点 | 绿色 | `xg` | 当前目标位置 |
| 障碍物 | 黑色 | `.k` | 障碍物位置 |
| 地面点 | 灰色 | `0.8` | RANSAC地面内点 |
| 非地面点 | 红色 | `r` | RANSAC非地面点 |
| EKF估计 | 红色 | `-r` | EKF定位结果 |
| 真实轨迹 | 蓝色 | `-b` | 真实位姿轨迹 |
| 航位推算 | 黑色 | `-k` | 纯运动模型预测 |

动画刷新方式：`plt.cla()` + `plt.pause(0.001)` 逐帧刷新，不使用 FuncAnimation。

## 模块选择指南

### 路径跟踪控制器

| 场景 | 推荐控制器 | 理由 |
|------|-----------|------|
| 低速简单场景 | Pure Pursuit | 实现简单，计算量最小 |
| 中高速结构化道路 | Stanley | 前轮反馈，横向精度高 |
| 弯道频繁 | 模糊控制 | 曲率自适应速度控制 |
| 复杂约束场景 | MPC | 显式约束处理，预测优化 |
| 未知环境探索 | DQN | 端到端学习，无需模型 |

### 路径规划器

| 场景 | 推荐规划器 | 理由 |
|------|-----------|------|
| 静态环境全局规划 | A* | 最优性保证 |
| 连续空间探索 | RRT | 概率完备，无需离散化 |
| 动态环境实时避障 | DWA | 速度空间搜索，实时性好 |

### 定位算法

| 场景 | 推荐算法 | 理由 |
|------|---------|------|
| GPS信号良好 | EKF | 计算量最小，高斯噪声最优 |
| 非高斯/多模态 | PF | 粒子采样，非参数化 |
| 多传感器互补 | Fusion | 协方差交叉融合，保守一致 |
