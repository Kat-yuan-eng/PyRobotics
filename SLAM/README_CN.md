# SLAM -- 同时定位与地图构建

本模块实现了 FastSLAM 2.0 和 ICP 点云配准两种核心 SLAM 算法，并通过流水线模块将二者组合为完整的 SLAM 系统，用于在未知环境中同时估计机器人位姿与构建路标地图。

## 模块列表

| 算法 | 文件 | 核心函数 | 输入 | 输出 | 适用场景 |
|------|------|---------|------|------|---------|
| FastSLAM 2.0 | fast_slam.py | `fast_slam()` | 控制输入u, 观测z | 粒子集(位姿+路标地图) | 中大规模路标、非线性环境 |
| ICP 配准 | icp_matching.py | `icp_match()` | 前帧点云, 当前帧点云 | 刚体变换(R, T) | 帧间点云配准 |
| SLAM流水线 | slam_pipeline.py | `run_slam_pipeline()` | 仿真参数 | 轨迹+地图+RMSE曲线 | 完整SLAM系统验证与可视化 |

---

## 算法详细说明

### 1. FastSLAM 2.0 (fast_slam.py)

基于 Rao-Blackwellized 粒子滤波的 SLAM 算法，将高维联合 SLAM 后验分解为机器人轨迹的粒子滤波与每个路标的独立 EKF。每个粒子维护一组路标 EKF（含位置均值 `lm` 与协方差 `lmP`）。与 FastSLAM 1.0 的核心区别在于引入了 `proposal_sampling()` 函数，利用当前观测对粒子位姿进行修正——以观测残差驱动均值偏移，以粒子协方差逆与观测信息矩阵之和的逆作为提议分布协方差，生成更接近真实后验的提议分布。

预测步对控制输入加噪采样生成粒子提议分布；更新步对已知路标计算权重并更新 EKF，对新路标通过反投影初始化（`add_new_landmark`）；当有效粒子数低于阈值时触发系统重采样（`resampling`）。`estimate_from_particles()` 输出加权平均的位姿估计和路标地图。

粒子数据结构为字典：`{pose: ndarray(3), weight: float, landmarks: ndarray(N,2), lmP: ndarray(N,2,2)}`。关键参数：`n_landmarks`（路标数）、`R_motion`（运动噪声）、`Q`（观测噪声）、`n_threshold`（重采样阈值）、`alpha`（提议分布缩放因子）。

### 2. ICP 配准 (icp_matching.py)

迭代最近点（Iterative Closest Point）算法，用于两帧点云之间的刚体变换估计。每轮迭代：首先通过最近邻关联（`nearest_neighbor_association`）寻找点对，然后基于 SVD 分解（`svd_motion_estimation`）求解最优旋转与平移，更新当前点云并累积齐次变换矩阵。迭代直到残差变化量小于阈值 `eps` 或达到最大迭代次数 `max_iter`。

SVD 求解步骤：去质心 → 计算互协方差矩阵 H → SVD 分解 H = UΣV^T → R = VU^T，T = μ_curr - R @ μ_prev。ICP 收敛性依赖初始对齐精度，容易陷入局部极小值。

关键参数：`max_iter`（最大迭代次数，默认100）、`eps`（收敛阈值，默认1e-4）。

### 3. SLAM 流水线 (slam_pipeline.py)

将 FastSLAM 和 ICP 组合为完整的验证流水线。`run_slam_pipeline()` 生成仿真数据（圆形轨迹+随机路标），逐帧运行 FastSLAM，同时用 ICP 对真实轨迹和估计轨迹做配准验证。输出包含真实轨迹、估计轨迹、路标地图、估计路标和 RMSE 历史曲线，通过三子图可视化展示。

---

## 算法对比总表

| 维度 | FastSLAM 2.0 | ICP 配准 |
|------|-------------|---------|
| 方法类型 | 粒子滤波+路标EKF | 点云配准 |
| 状态表示 | 粒子集+路标高斯 | 齐次变换矩阵 |
| 计算复杂度 | O(M log N) | O(NM) 每轮 |
| 非线性处理 | 采样+观测修正 | 迭代优化 |
| 回环处理 | 不支持 | 不直接支持 |
| 输出性质 | 在线增量 | 帧间增量 |
| 路标数量 | 中大规模 | 无路标概念 |
| 收敛保证 | 局部 | 局部（依赖初始对齐） |

---

## 可视化

所有算法模块支持 `show_animation = True` 交互式可视化，遵循 PythonRobotics 动画规范：

- 动画方式：`plt.cla()` + `plt.pause(0.001)` 逐帧刷新
- 颜色语义：红(.r)=参考路径、蓝(-b)=实际轨迹、绿(xg)=目标点、黑(.k)=障碍物
- 运行方式：直接执行各模块脚本，如 `python SLAM/fast_slam.py`

### 模块动画说明

| 模块 | 动画内容 |
|------|---------|
| FastSLAM | 粒子分布+路标估计动画，蓝色真实轨迹+红色FastSLAM估计 |
| ICP | 迭代配准动画，逐帧显示点云对齐过程 |

---

## 模块间关联关系

```
控制输入u + 观测z ──→ fast_slam ──→ 位姿估计 + 路标地图
                                    │
真实轨迹 + 估计轨迹 ──→ icp_match ──→ 配准验证
                                    │
                    slam_pipeline ──→ 可视化 + RMSE评估
```

FastSLAM 是核心 SLAM 引擎，ICP 作为验证工具评估轨迹估计精度。`slam_pipeline` 将二者串联为完整的测试流程。

---
