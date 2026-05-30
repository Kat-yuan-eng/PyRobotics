# PyRobotics — 智能汽车自动驾驶软件栈

本项目是一个完整的自动驾驶软件栈，按四层架构组织：**感知→决策→控制→系统**，所有模块通过 Protobuf 消息解耦，形成严格的线性数据流水线。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.9-blue.svg)](pyproject.toml)

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    PyRobotics 系统架构                        │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│  感知层   │  决策层   │  控制层   │  定位层   │    SLAM层      │
│Perception│ Decision │PathTrack │Localization│     SLAM       │
├──────────┼──────────┼──────────┼──────────┼────────────────┤
│车道线检测 │任务调度   │Stanley   │EKF       │FastSLAM 2.0    │
│障碍物检测 │路径平滑   │PurePursuit│粒子滤波  │ICP配准         │
│标志识别   │障碍物避障 │模糊控制   │协方差融合 │SLAM流水线      │
│传感器融合 │多智能体   │MPC / DQN │          │                │
├──────────┴──────────┴──────────┴──────────┴────────────────┤
│                    系统基础设施层                              │
│  vehicle_sim │ realtime_pipeline │ embedded C │ ROS2         │
└─────────────────────────────────────────────────────────────┘
```

## 子系统概览

| 子系统 | 目录 | 核心功能 | 关键算法 |
|--------|------|---------|---------|
| 感知 | perception/ | 车道线/障碍物/标志检测与融合 | HLS颜色分割, RANSAC+DBSCAN, HOG模板匹配 |
| 决策 | decision/ | 任务调度、路径平滑、避障 | FSM任务调度, 曲率约束平滑, 横向避障 |
| 规划 | PathPlanning/ | 全局路径规划与局部避障 | A*, RRT(含平滑), DWA(含全局路径对齐) |
| 控制 | PathTracking/ | 横向+纵向车辆控制 | Stanley, PurePursuit, 模糊, MPC, DQN, 选择器 |
| 定位 | Localization/ | 车辆位姿估计 | EKF(Joseph形式), 粒子滤波, 协方差交叉融合 |
| SLAM | SLAM/ | 同时定位与地图构建 | FastSLAM 2.0(自适应阈值), ICP(Huber核) |
| 系统 | system/ | 车辆仿真与实时框架 | 自行车模型, 双线程流水线, 嵌入式C, ROS2 |

## 安全特性

- 油门/制动互斥：`if brake > 0.01: throttle = 0.0`
- EKF协方差爆炸检测：`max(diag(P)) > 100` 时重置
- NaN守卫：所有估计器回退至预测状态
- 障碍物坐标验证（有限性检查）
- 多智能体/跟踪器ID冲突预防
- Joseph形式协方差更新（保证正定性）

## 环境要求

- Python >= 3.9
- NumPy, SciPy, Matplotlib
- OpenCV (cv2), scikit-learn
- protobuf + grpcio-tools
- pytest

## 安装

```bash
cd PyRobotics
pip install -r requirements/requirements.txt

# 如需重新生成 Protobuf 代码
python -m grpc_tools.protoc -I=proto --python_out=generated proto/*.proto
```

## 运行方式

### 单模块可视化

每个算法模块都可以直接运行，展示 PythonRobotics 风格的交互式动画：

```bash
python PathTracking/stanley_controller.py
python PathPlanning/dwa_planner.py
python Localization/ekf_localizer.py
python SLAM/fast_slam.py
```

### 主循环仿真

```bash
python main_loop.py
```

单线程 50Hz 闭环仿真：感知→决策→控制→车辆仿真。

### 单元测试

```bash
python tests/test_all_modules.py
```

## 致谢

本项目灵感来源于 [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)（Atsushi Sakai 等人），该项目提供了优秀的机器人学算法实现集合。

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。
