# PathTracking -- 路径跟踪控制

本模块实现了5种车辆横向+纵向控制算法，覆盖从经典几何跟踪到强化学习的完整控制谱系，通过控制器选择器统一分发，输出 ControlOutput protobuf 消息供车辆仿真或实车执行。

## 模块列表

| 算法 | 文件 | 核心函数 | 横向控制 | 纵向控制 | 适用场景 |
|------|------|---------|---------|---------|---------|
| Pure Pursuit | pure_pursuit_controller.py | `pure_pursuit_control()` | 前视点曲率转角 | PID速度控制 | 低速大曲率路径跟踪 |
| Stanley | stanley_controller.py | `stanley_control()` | 航向误差+横向误差+前视+曲率前馈 | 级联PID | 中高速结构化道路 |
| 模糊控制 | fuzzy_controller.py | `fuzzy_control()` | Stanley转向 | 模糊推理速度 | 曲率变化频繁的弯道 |
| MPC | mpc_controller.py | `mpc_control()` | 自行车模型预测 | 代价函数优化 | 需要前瞻约束的复杂场景 |
| DQN强化学习 | rl_controller.py | `rl_control()` | 3层MLP推理 | 6离散动作 | 未知环境探索式控制 |
| 控制器选择 | controller_selector.py | `select_controller()` | 分发 | 分发 | 统一入口，策略模式 |

---

## 算法详细说明

### 1. Pure Pursuit (pure_pursuit_controller.py)

经典几何跟踪算法，在参考路径上搜索前视距离 `ld` 处的目标点，计算从后轴到目标点的圆弧曲率 `κ = 2sin(α)/ld`，再由 `δ = arctan(κL)` 得到转向角。前视距离自适应：`ld = lookahead_gain × v + lookahead_min`，速度越快前视越远，兼顾跟踪精度与稳定性。纵向采用 PID 速度控制，目标速度由决策模块给定。

关键参数：`lookahead_gain`（前视距离速度增益）、`lookahead_min`（最小前视距离）、`kp/ki/kd`（速度PID参数）。Pure Pursuit 在大曲率路径上跟踪精度高，但高速时因前视距离增大导致内切现象，适合低速场景。

### 2. Stanley (stanley_controller.py)

Stanford 大学提出的横向控制算法，转向角由三部分组成：航向误差修正 `δ_yaw = θ_path - θ_actual`、横向误差修正 `δ_cte = arctan(k_e × cte / (k_v + v))`、前视补偿 `δ_preview`。额外引入曲率前馈 `κ × L` 提前补偿弯道转向，死区 `dead_zone` 避免直道微振，低速阻尼 `v_damping` 防止低速转向过灵。纵向采用级联PID：外环速度PID输出目标加速度，内环加速度PI跟踪目标加速度。曲率限速 `v = sqrt(a_lat_max / κ)` 约束弯道最大横向加速度。Jerk限制和制动斜坡保证控制平滑性。

关键参数：`k_e`（横向误差增益，越大横向纠偏越快）、`k_v`（速度柔化因子，防止低速时横向修正过大）、`k_preview`（前视距离）、`max_steer_deg`（最大转向角）、`dead_zone`（转向死区角度）。Stanley 是本项目中功能最完整的控制器，适合中高速结构化道路。

### 3. 模糊控制 (fuzzy_controller.py)

横向沿用 Stanley 转向，纵向用模糊推理替代 PID。输入为路径曲率 `κ`（3个模糊集：Z/S/L）和速度差 `Δv`（5个模糊集：NL/NS/Z/PS/PL），输出为油门增量 `Δu`（5个模糊集）。15条模糊规则覆盖"弯道减速、直道加速"的驾驶经验，例如：κ=S 且 Δv=PL → Δu=NL（弯道超速则强减速）。隶属度函数为三角函数 `_trimf`，去模糊化采用加权平均法。

关键参数：`_KAPPA_Z/S/L`（曲率模糊集边界）、`_DV_NL/NS/Z/PS/PL`（速度差模糊集边界）、`_RULES`（15条规则表）。模糊控制的优势在于将驾驶经验编码为可解释的规则，无需精确模型参数，但规则设计和隶属度函数调优依赖经验。

### 4. MPC (mpc_controller.py)

基于自行车运动学模型的模型预测控制。在预测时域 N 步内，用 `_rollout()` 前向仿真车辆状态序列，计算代价函数：`J = w_y×Σy² + w_θ×Σθ² + w_v×ΣΔv² + w_δ×Σδ² + w_a×Σa² + w_δ̇×Σδ̇²`，分别惩罚横向偏差、航向偏差、速度偏差、转向角、加速度和转向变化率。通过中心差分数值梯度 `_numerical_gradient()` 计算代价对控制序列的梯度，执行5步梯度下降迭代优化。

关键参数：`N`（预测步数，越大前瞻越远但计算量越大）、`dt_mpc`（预测步长）、`w_y/w_θ/w_v/w_delta/w_a/w_deltadot`（各代价权重）。MPC 的核心优势在于可显式处理约束（转向角限制、加速度限制），但数值梯度法收敛较慢，实时性受预测步数限制。

### 5. DQN 强化学习 (rl_controller.py)

基于 Deep Q-Network 的端到端控制。状态空间为18维：16束激光扫描距离 + 目标方向角。动作空间为6个离散动作：{加速, 减速, 左转, 右转, 加速+左转, 加速+右转}。网络为3层全连接 MLP(18→64→32→6)，纯 NumPy 实现（无 PyTorch/TensorFlow 依赖）。训练采用经验回放（replay buffer）+ 目标网络（软更新）+ ε-贪心衰减。

`train_dqn()` 为离线训练入口，`rl_control()` 为在线推理入口（加载预训练权重）。推理时前向传播一次 MLP 即可输出动作，延迟极低。关键参数：`n_episodes`（训练轮数）、`weights`（预训练权重路径）。DQN 适合未知环境中的探索式控制，但离散动作空间限制了控制精度，且训练需要大量仿真交互。

### 6. 控制器选择器 (controller_selector.py)

策略模式分发器，根据 `ctrl_type` 常量（0-4）调用对应控制器函数，管理各控制器的内部状态（PID积分项、前次误差等）。`ctrl_name()` 返回控制器名称字符串，用于日志和可视化。

---

## 算法对比总表

| 维度 | Pure Pursuit | Stanley | 模糊控制 | MPC | DQN |
|------|-------------|---------|---------|-----|-----|
| 横向方法 | 前视曲率 | 航向+横向+前视+前馈 | Stanley | 模型预测 | MLP推理 |
| 纵向方法 | PID | 级联PID | 模糊推理 | 代价优化 | 离散动作 |
| 需要模型 | 是(自行车) | 是(自行车) | 否(规则) | 是(自行车) | 否(学习) |
| 计算复杂度 | O(N_path) | O(N_path) | O(N_path+规则) | O(N×5迭代) | O(前向传播) |
| 实时性 | 优 | 优 | 优 | 中 | 优(推理) |
| 参数调优 | 少 | 中 | 中(规则+隶属度) | 多(权重) | 大量训练 |
| 可解释性 | 高 | 高 | 中 | 中 | 低 |
| 适用速度 | 低速 | 中高速 | 弯道频繁 | 复杂约束 | 未知环境 |

---

## 可视化

所有算法模块支持 `show_animation = True` 交互式可视化，遵循 PythonRobotics 动画规范：

- 动画方式：`plt.cla()` + `plt.pause(0.001)` 逐帧刷新
- 颜色语义：红(.r)=参考路径、蓝(-b)=实际轨迹、绿(xg)=目标点、黑(.k)=障碍物
- 运行方式：直接执行各模块脚本，如 `python PathTracking/stanley_controller.py`

### 模块动画说明

| 模块 | 动画内容 |
|------|---------|
| Stanley | 蛇形路径跟踪动画，红色参考路径+蓝色实际轨迹+绿色目标点+车辆矩形，跟踪后显示轨迹/速度/横向误差三子图 |
| Pure Pursuit | 圆形路径跟踪动画，前视点可视化 |
| Fuzzy | 弯道路径跟踪动画，模糊推理速度控制可视化 |
| MPC | 路径跟踪动画，半透明红色显示MPC预测轨迹 |
| DQN | 训练奖励曲线 + 跟踪演示动画 |

---

## 模块间关联关系

```
DecisionOutput (protobuf)
       │
       ▼
controller_selector ──→ ctrl_type=0 ──→ pure_pursuit_control ──┐
                  ──→ ctrl_type=1 ──→ stanley_control ─────────┤
                  ──→ ctrl_type=2 ──→ fuzzy_control ───────────┤──→ ControlOutput (protobuf)
                  ──→ ctrl_type=3 ──→ mpc_control ─────────────┤
                  ──→ ctrl_type=4 ──→ rl_control ──────────────┘
```

所有控制器共享相同的输入接口（`DecisionOutput` protobuf + 实际速度 + 轮距等参数）和输出接口（`ControlOutput` protobuf），通过选择器统一调度。模糊控制器内部复用 Stanley 的横向控制函数 `_stanley_steer`。下游 `system/vehicle_sim.py` 消费 `ControlOutput` 执行车辆仿真。
