# PathPlanning -- 路径规划

本模块实现了3种路径规划算法，覆盖全局搜索、随机采样和局部避障三类规划范式，通过规划器选择器和基准测试框架统一管理，为决策模块提供从起点到终点的可行路径。

## 模块列表

| 算法 | 文件 | 核心函数 | 搜索空间 | 输出 | 适用场景 |
|------|------|---------|---------|------|---------|
| A*搜索 | a_star_planner.py | `a_star_plan()` | 2D栅格 | 路径坐标+统计信息 | 静态环境全局最优路径 |
| RRT采样 | rrt_planner.py | `rrt_plan()` | 连续2D空间 | 路径坐标+统计信息 | 高维/连续空间快速探索 |
| DWA局部 | dwa_planner.py | `dwa_plan()` | 速度空间(v,ω) | 控制输入+预测轨迹+统计信息 | 动态环境实时避障 |
| 规划器选择 | planner_selector.py | `select_planner()` / `compare_planners()` | — | 推荐规划器/对比结果 | 统一入口与基准测试 |

---

## 算法详细说明

### 1. A* 搜索 (a_star_planner.py)

经典启发式搜索算法，在2D栅格地图上寻找从起点到终点的最短路径。评估函数 `f(n) = g(n) + h(n)`，其中 `g(n)` 为起点到当前节点的实际代价，`h(n) = ||n - goal||` 为欧几里得距离启发式。支持8方向移动（4正交+4对角，对角代价 √2），使用二叉堆（`heapq`）维护开放集。

栅格地图构建（`create_grid_map`）将障碍物点膨胀 `robot_radius` 个栅格单元，确保路径与障碍物保持安全距离。A* 保证在栅格空间中找到最短路径，但搜索节点数与地图规模成正比，不适合大规模或动态变化的环境。

关键参数：`resolution`（栅格分辨率，越小路径越精细但计算量越大）、`robot_radius`（机器人半径，控制障碍物膨胀范围）。输出包含路径坐标、路径长度、规划耗时和搜索节点数。

### 2. RRT 采样 (rrt_planner.py)

快速探索随机树（Rapidly-exploring Random Tree），在连续2D空间中通过随机采样构建搜索树。每轮迭代：以 `(1 - goal_sample_rate/100)` 的概率在采样区域内随机取点，否则直接采样目标点；找到树中最近节点，向采样方向扩展固定距离 `expand_dis`；经碰撞检测（点碰撞+路径碰撞）后加入树。当新节点到目标距离小于 `expand_dis` 且路径无碰撞时，连接目标点完成规划。

碰撞检测分两级：`_check_collision` 检查单点是否在障碍物膨胀范围内，`_check_path_collision` 沿线段等间距采样检查路径碰撞。RRT 概率完备但不保证最优，路径通常曲折需要后处理平滑。

关键参数：`expand_dis`（扩展步长）、`path_resolution`（路径碰撞检测分辨率）、`goal_sample_rate`（目标采样率，加速收敛）、`max_iter`（最大迭代次数）、`robot_radius`（机器人半径）。

### 3. DWA 局部规划 (dwa_planner.py)

动态窗口法（Dynamic Window Approach），在速度空间 (v, yaw_rate) 中搜索最优控制输入。首先根据机器人运动限制和当前速度计算动态窗口 [v_min, v_max, ω_min, ω_max]；然后在窗口内离散采样，对每组 (v, ω) 前向模拟预测轨迹（预测时间 `predict_time`）；以加权代价函数评估：`cost = to_goal_cost_gain × 角度偏差 + speed_cost_gain × 速度损失 + obstacle_cost_gain × 障碍物距离倒数`。碰撞轨迹代价设为无穷大。

DWA 天然考虑了运动学约束和速度限制，适合动态环境中的实时避障，但可能陷入局部极小值。与 A* 和 RRT 不同，DWA 不生成全局路径，而是逐帧输出控制指令。

关键参数：`max_speed/min_speed`（速度范围）、`max_yaw_rate`（最大角速度）、`max_accel/max_delta_yaw_rate`（加速度限制）、`predict_time`（预测时间）、`v_resolution/yaw_rate_resolution`（采样分辨率）、`to_goal_cost_gain/speed_cost_gain/obstacle_cost_gain`（代价权重）。

### 4. 规划器选择与基准测试 (planner_selector.py)

提供两个核心功能：`select_planner()` 根据场景类型（static_sparse/dynamic/narrow_passage/large_map/realtime）推荐最合适的规划器；`compare_planners()` 在3个预设基准场景（稀疏障碍、狭窄通道、迷宫）上运行 A*/RRT/DWA 三种规划器，输出路径长度、规划耗时和搜索节点数的对比结果。

---

## 算法对比总表

| 维度 | A* | RRT | DWA |
|------|-----|-----|-----|
| 方法类型 | 全局搜索 | 随机采样 | 局部规划 |
| 搜索空间 | 2D栅格 | 连续2D | 速度空间(v,ω) |
| 最优性 | 栅格最优 | 不保证 | 局部最优 |
| 完备性 | 是 | 概率完备 | 是(局部) |
| 计算复杂度 | O(N² log N) | O(N_iter) | O(N_v × N_ω) |
| 动态环境 | 不支持 | 不支持 | 支持 |
| 运动学约束 | 不考虑 | 不考虑 | 考虑 |
| 输出类型 | 路径点序列 | 路径点序列 | 控制输入+轨迹 |
| 适用阶段 | 全局规划 | 全局规划 | 局部规划 |

---

## 可视化

所有算法模块支持 `show_animation = True` 交互式可视化，遵循 PythonRobotics 动画规范：

- 动画方式：`plt.cla()` + `plt.pause(0.001)` 逐帧刷新
- 颜色语义：红(.r)=参考路径、蓝(-b)=实际轨迹、绿(xg)=目标点、黑(.k)=障碍物
- 运行方式：直接执行各模块脚本，如 `python PathPlanning/a_star_planner.py`

### 模块动画说明

| 模块 | 动画内容 |
|------|---------|
| A* | 搜索过程动画，青色已搜索节点+红色最终路径 |
| RRT | 树扩展动画，绿色搜索树+红色最终路径 |
| DWA | 轨迹预测动画，多候选轨迹+最优轨迹高亮 |

---

## 模块间关联关系

```
场景类型 ──→ planner_selector.select_planner() ──→ 推荐规划器
                                                      │
                    ┌─────────────────────────────────┤
                    │                                 │
                    ▼                                 ▼
             a_star_plan()                    rrt_plan()
             (静态全局路径)                   (连续空间路径)
                    │                                 │
                    └──────────┐    ┌─────────────────┘
                               │    │
                               ▼    ▼
                         decision.task_scheduler
                               │
                               ▼
                         dwa_plan() (实时局部避障)
```

A* 和 RRT 为全局规划器，在决策层任务调度时生成参考路径；DWA 为局部规划器，在控制层实时避障时使用。`planner_selector` 根据场景特征选择合适的全局规划器。三种规划器的输出均可被 `decision/path_smoother.py` 平滑处理后供控制模块跟踪。
