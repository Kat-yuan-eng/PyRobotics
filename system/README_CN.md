# System -- 系统仿真与实时框架

本模块实现了车辆运动学仿真和实时控制流水线，是整个自动驾驶软件栈的运行基础设施，提供从控制指令到车辆状态的单步仿真以及多线程实时执行框架。

## 模块列表

| 功能 | 文件 | 核心函数 | 输入 | 输出 | 适用场景 |
|------|------|---------|------|------|---------|
| 车辆仿真 | vehicle_sim.py | `simulate_vehicle()` | 转向角+油门+制动+当前状态 | 更新后状态(x,y,θ,v) | 自行车模型单步仿真 |
| 实时流水线 | realtime/realtime_pipeline.py | `run_realtime_pipeline()` | 运行时长 | 实时感知+控制循环 | 双线程硬实时执行 |
| 最新值容器 | realtime/latest_value.py | `LatestResult` | 写入数据 | 读取数据+序列号 | 线程安全数据传递 |
| 嵌入式C代码 | embedded/ | — | — | — | 嵌入式平台算法移植（EKF/Stanley/PID/SLAM/查表） |
| ROS2框架 | ros2/ | — | — | — | ROS2节点化部署（感知/规划/控制节点） |

---

## 算法详细说明

### 1. 车辆仿真 (vehicle_sim.py)

基于自行车运动学模型的单步仿真器。给定转向角 `steer_deg`、油门 `throttle`、制动力 `brake` 和当前状态，计算下一时刻的车辆状态。运动学方程：

```
x_{k+1} = x_k + v_k * cos(θ_k) * dt
y_{k+1} = y_k + v_k * sin(θ_k) * dt
θ_{k+1} = θ_k + v_k * tan(δ) / L * dt
v_{k+1} = v_k + (throttle - brake * sign(v)) * dt
```

其中 `L = WHEELBASE = 2.7m` 为轴距，`δ` 为前轮转向角。油门和制动分别产生正/负加速度，速度被裁剪到非负值（不允许倒车）。该模型忽略了轮胎侧滑、质量转移等动力学效应，适用于低速和中速场景的快速验证。

### 2. 实时流水线 (realtime/realtime_pipeline.py)

双线程实时执行框架，将感知和控制解耦为独立线程。感知线程以 30ms 周期运行（`_perception_loop`）：读取测试图像/点云 → 运行4个感知模块 → 输出 `PerceptionOutput` 到 `LatestResult`。控制线程以 50Hz（20ms）硬实时节拍运行（`_control_loop`）：从 `LatestResult` 读取最新感知结果 → 决策调度 → 控制器计算 → 车辆仿真 → 输出控制指令。

两个线程通过 `LatestResult` 容器传递数据，感知线程写入、控制线程读取，互不阻塞。控制线程使用 `time.sleep()` 精确控制节拍，保证 50Hz 的控制频率。

关键参数：`CONTROL_FREQ_HZ=50`（控制频率）、`PERCEPTION_PERIOD_S=0.03`（感知周期）、`V_NOMINAL=5.0`（标称速度）。

### 3. 最新值容器 (realtime/latest_value.py)

线程安全的最新值读写容器，核心思想是"生产者-消费者"模式下的无阻塞数据传递。`write(data)` 在互斥锁保护下写入数据并递增序列号；`read()` 在互斥锁保护下读取数据和序列号。消费者通过比较序列号判断数据是否更新，避免重复处理同一帧数据。

该容器保证数据一致性（读写不会交错），但不保证消费者能读到每一帧数据（如果生产者写入速度慢于消费者读取速度，消费者会读到相同数据；反之，中间帧会被跳过）。这种"最新值语义"适合实时控制系统——宁可跳过旧数据，也不能因等待而延迟控制。

---

## 模块间关联关系

```
感知线程 (30ms周期)                    控制线程 (50Hz硬实时)
  │                                      │
  ├── lane_pixel_detector                ├── task_scheduler.schedule()
  ├── obstacle_detector                  ├── controller_selector
  ├── sign_recognizer                    └── vehicle_sim.simulate_vehicle()
  └── sensor_fusion                           │
       │                                      │
       └──── LatestResult ◄───────────────────┘
              (线程安全)
```

感知线程和控制线程通过 `LatestResult` 解耦。感知线程是数据生产者，控制线程是数据消费者。`vehicle_sim` 在控制线程内被调用，将控制指令转换为车辆状态更新，形成闭环。
