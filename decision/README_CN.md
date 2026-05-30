# Decision -- 决策

本模块实现了自动驾驶的决策层，包含任务调度、路径平滑、障碍物避障和多智能体协调四个核心功能，将感知输出转换为带速度分配的规划路径和决策指令，供控制模块执行。

## 模块列表

| 功能 | 文件 | 核心函数 | 输入 | 输出 | 适用场景 |
|------|------|---------|------|------|---------|
| 任务调度 | task_scheduler.py | `schedule()` | PerceptionOutput, 调度状态 | DecisionOutput protobuf | 巡逻/避障/泊车任务切换 |
| 路径平滑 | path_smoother.py | `smooth_path()` | 原始路径点 | 平滑路径点 | 消除路径锯齿和曲率突变 |
| 障碍物避障 | obstacle_avoidance.py | `avoid_obstacles()` | 路径+障碍物列表 | 横向偏移路径 | 前方障碍物绕行 |
| 多智能体协调 | multi_agent.py | `coordinate_avoidance()` | 多Agent状态+感知输出 | 协调后DecisionOutput | 多车协同避障 |

---

## 算法详细说明

### 1. 任务调度 (task_scheduler.py)

基于有限状态机（FSM）的任务调度器，管理三种驾驶任务的切换：巡逻（PATROL）、避障（AVOID）、泊车（PARK）。巡逻任务为默认状态，沿参考路径行驶并执行车道保持决策；当检测到前方障碍物距离小于触发阈值时切换到避障任务；当检测到泊车标志时切换到泊车任务。

车道保持决策（`_lane_keep_decide`）包含两个核心逻辑：曲率自适应速度分配（`_assign_speed_profile`）根据路径曲率 `κ` 分配目标速度——直道（κ < 阈值）分配标称速度，弯道分配 `v = v_nominal × (1 - κ/κ_max)`；前方障碍物分析（`_analyze_obstacles`）计算到最近障碍物的距离和碰撞时间 TTC，当距离小于停车阈值时输出停车行为。避障任务调用 `path_smoother` 和 `obstacle_avoidance` 生成绕行路径。泊车任务输出低速+精确转向指令。

关键参数：`v_nominal`（标称速度）、`v_min`（最低速度）、`stop_dist_threshold`（停车距离阈值）、`kappa_threshold`（弯道曲率阈值）、`avoid_trigger_dist`（避障触发距离）。

### 2. 路径平滑 (path_smoother.py)

采用弧长参数化 + 自适应高斯平滑的两阶段路径处理。第一阶段，计算路径累积弧长 `_arc_length()`，按弧长均匀重采样 `_resample_by_arc()` 到指定点数，消除原始路径点间距不均匀的问题。第二阶段，在最大偏差约束下搜索最大平滑尺度：从 `sigma=0.5` 开始递增，对路径 x/y 坐标分别做高斯平滑 `_gaussian_smooth()`（边缘镜像填充），计算平滑后路径与原始路径的最大偏差，若偏差超过 `max_deviation` 则回退到上一个 sigma 值。

自适应 sigma 的核心思想：直道区域允许大 sigma（强平滑），弯道区域自动降 sigma（保形状）。最终输出均匀采样+适度平滑的路径及其曲率信息。

关键参数：`max_deviation`（最大允许偏差，控制平滑程度与路径保真度的权衡）、`n_output`（输出点数）。

### 3. 障碍物避障 (obstacle_avoidance.py)

基于横向偏移的局部避障算法。首先将每个障碍物投影到最近路径点（`_project_obstacle`），计算路径法向量（`_path_normals`）。然后对每个障碍物计算所需横向偏移量 = 障碍物到路径距离 + 安全间距，优先尝试左侧偏移，若左侧碰撞检测（`_check_clearance`）不通过则尝试右侧。偏移采用正弦半波过渡（`_apply_lateral_offset`），在障碍物投影区间内平滑地从零偏移过渡到目标偏移再回归零偏移，避免路径折角。

关键参数：`max_lateral`（最大横向偏移量）、`safety_margin`（安全间距）、`priority`（优先避障方向，'left' 或 'right'）。该算法适用于结构化道路上的静态障碍物绕行，不处理动态障碍物预测。

### 4. 多智能体协调 (multi_agent.py)

将其他智能体（Agent）的规划路径按时间插值为动态障碍物，合并到感知输出后调用单智能体调度器。`_agent_to_obstacle()` 根据当前时间在 Agent 路径上线性插值得到其瞬时位置，作为动态障碍物添加到 `PerceptionOutput` 中。`build_ego_state()` 构建自车的 `AgentState` protobuf 消息，包含当前位置、速度、规划路径等信息。

关键参数：通过 `task_scheduler.schedule()` 的参数间接控制。该模块是分布式多车协同的基础——每个 Agent 独立运行调度器，通过共享 AgentState 实现去中心化协调。

---

## 可视化

所有算法模块支持 `show_animation = True` 交互式可视化，遵循 PythonRobotics 动画规范：

- 动画方式：`plt.cla()` + `plt.pause(0.001)` 逐帧刷新
- 颜色语义：红(.r)=参考路径、蓝(-b)=实际轨迹、绿(xg)=目标点、黑(.k)=障碍物
- 运行方式：直接执行各模块脚本，如 `python decision/task_scheduler.py`

### 模块动画说明

| 模块 | 动画内容 |
|------|---------|
| task_scheduler | 状态机时序图（任务切换+速度曲线+行为输出）+状态转移图（PATROL↔AVOID↔PARK） |
| obstacle_avoidance | 3子图避障场景（单障碍/多障碍/双侧阻塞）+车辆沿避障路径行驶动画 |

---

## 模块间关联关系

```
PerceptionOutput (protobuf)
       │
       ├──→ task_scheduler.schedule()
       │        │
       │        ├── PATROL ──→ _lane_keep_decide() ──→ DecisionOutput
       │        ├── AVOID  ──→ path_smoother + obstacle_avoidance ──→ DecisionOutput
       │        └── PARK   ──→ 泊车指令 ──→ DecisionOutput
       │
       └──→ multi_agent.coordinate_avoidance()
                │
                └──→ 其他Agent路径插值 → 合并障碍物 → schedule()

path_smoother.smooth_path() ←── 被 task_scheduler 调用
obstacle_avoidance.avoid_obstacles() ←── 被 task_scheduler 调用
```

`task_scheduler` 是决策层的核心调度器，`path_smoother` 和 `obstacle_avoidance` 作为其内部工具被调用。`multi_agent` 在调度器之上增加多车协调层，将其他 Agent 转化为动态障碍物后复用单 Agent 调度逻辑。
