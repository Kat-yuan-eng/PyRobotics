# Perception -- 感知

本模块实现了4个核心感知功能，覆盖车道线检测、障碍物检测、交通标志识别与多传感器融合，将原始传感器数据转换为结构化的 PerceptionOutput protobuf 消息，供下游决策模块消费。

## 模块列表

| 功能 | 文件 | 核心函数 | 输入 | 输出 | 适用场景 |
|------|------|---------|------|------|---------|
| 车道线像素检测 | lane_pixel_detector.py | `detect_lane_pixels()` | BGR图像 | scan_rows(行扫描坐标), center_x(车道中心x) | 结构化道路白色/黄色车道线 |
| 障碍物检测 | obstacle_detector.py | `detect_obstacles()` | 3D点云(N×3) | 障碍物列表(位置+尺寸+形状) | 激光雷达点云中的静态障碍物 |
| 交通标志识别 | sign_recognizer.py | `recognize_signs()` | BGR图像 | 标志列表(类别+位置+置信度) | 红色交通标志(限速/箭头/禁行/禁停) |
| 传感器融合 | sensor_fusion.py | `fuse_to_perception()` | 车道线+障碍物+标志+相机参数 | PerceptionOutput protobuf | 统一融合输出供决策模块消费 |
| 障碍物跟踪 | obstacle_tracker.py | `track_obstacles()` | 当前帧障碍物+上一帧跟踪列表 | 更新后跟踪列表 | 障碍物ID关联与速度估计 |

---

## 算法详细说明

### 1. 车道线像素检测 (lane_pixel_detector.py)

采用 HLS 颜色空间分割 + 逐行峰值扫描的两阶段检测流程。第一阶段，将 BGR 图像转换到 HLS 空间，对白色（L 通道高亮度）和黄色（H 通道黄色范围 + S 通道高饱和度）分别生成二值掩码并合并。第二阶段，对掩码图逐行扫描：以图像水平中点为界，在左右半区分别搜索峰值像素位置作为车道线候选点，取左右峰值的中点作为车道中心。最后通过跳变滤波（`_filter_jumps`）去除相邻行中心点突变超过阈值的异常值。

关键参数：`scan_rows`（扫描行数，默认从图像底部1/3区域均匀采样）、`peak_margin`（峰值搜索半宽，控制左右搜索范围）、`jump_threshold`（跳变滤波阈值，相邻行中心点最大允许跳变像素数）。辅助函数 `generate_test_image()` 可生成含左右实线+中心虚线的合成测试图，无需真实摄像头即可验证检测流程。

### 2. 障碍物检测 (obstacle_detector.py)

采用 RANSAC 地面移除 → 体素下采样 → DBSCAN 聚类 → PCA-OBB 包围盒 → 多条件过滤的五级管线。第一级，RANSAC 拟合地面平面（法向量近似竖直），剔除地面点。第二级，体素下采样（`voxel_size` 控制分辨率）降低点密度。第三级，DBSCAN（`eps` 邻域半径，`min_samples` 最小点数）将剩余点聚类为独立障碍物。第四级，对每个聚类用 PCA 求主方向，构建有向包围盒（OBB），提取中心位置、尺寸和朝向。第五级，多条件过滤：最小高度（剔除矮小误检）、最大纵横比（剔除墙面等薄片状物体）、最小面积、最小点数。

关键参数：`ransac_iter`（RANSAC 迭代次数）、`ransac_thresh`（内点距离阈值）、`voxel_size`（体素边长，越小保留细节越多但计算量越大）、`dbscan_eps`（DBSCAN 邻域半径）、`dbscan_min`（DBSCAN 最小邻居数）。辅助函数 `generate_test_point_cloud()` 生成含地面平面+两个盒子+路沿的合成点云。

### 3. 交通标志识别 (sign_recognizer.py)

采用 HSV 红色区域定位 + HOG 特征模板匹配的两阶段识别流程。第一阶段，在 HSV 空间检测红色区域（H 通道红色范围 + S 通道高饱和度 + V 通道高亮度），提取轮廓并按面积筛选候选标志。第二阶段，对每个候选区域缩放到 64×64，提取 HOG 特征（9 方向直方图，8×8 cell，2×2 block），与预构建的模板 HOG 库（4 旋转角 × 14 类别 = 56 个模板）计算余弦相似度，取最高分类别作为识别结果。遮挡检测通过暗像素比例判断，遮挡超过阈值时降低置信度。

支持 15 类标志：限速 30/50/60/80/100、左转/右转/直行/掉头箭头、禁止驶入/禁止停车/停车让行/让行、行人注意/施工。模板由 `_generate_template()` 程序化生成（无需外部图像），支持 0°/90°/180°/270° 四个旋转角。关键参数：`min_area`（最小标志面积，过滤小误检）、`score_threshold`（HOG 匹配最低分数阈值）。

### 4. 传感器融合 (sensor_fusion.py)

将车道线、障碍物、交通标志三类感知结果统一融合为 `PerceptionOutput` protobuf 消息。核心操作为坐标变换：利用针孔相机模型反投影（`_pixel_to_vehicle`），将图像像素坐标 (u, v) 通过相机内参矩阵 K 的逆和相机-车辆外参 (R, t) 转换为车辆坐标系下的三维位置。车道中心线点逐点反投影后构建 `RoadBoundary`（含中心线、左右边界、车道宽度）。障碍物和交通标志分别反投影后构建对应的 protobuf 消息，交通标志额外附带标志类型和图像位置信息。

`default_camera_params()` 提供默认相机内参（焦距、光心）和外参（旋转矩阵、平移向量），可直接用于测试。融合输出是整个感知模块的唯一出口，下游决策模块仅依赖 `PerceptionOutput` protobuf，与具体传感器解耦。

---

## 算法对比总表

| 维度 | 车道线检测 | 障碍物检测 | 标志识别 | 传感器融合 |
|------|-----------|-----------|---------|-----------|
| 传感器 | 摄像头 | 激光雷达 | 摄像头 | 多传感器 |
| 处理域 | 图像像素 | 3D点云 | 图像像素 | 坐标变换 |
| 核心算法 | HLS颜色+峰值扫描 | RANSAC+DBSCAN+PCA | HSV颜色+HOG模板 | 针孔模型反投影 |
| 输出类型 | 行坐标+中心x | 障碍物列表 | 标志列表 | Protobuf |
| 计算复杂度 | O(H×W) | O(N×logN) | O(M×T) | O(N_pts) |
| 依赖 | cv2, numpy | numpy | cv2, numpy | numpy, protobuf |

---

## 可视化

所有算法模块支持 `show_animation = True` 交互式可视化，遵循 PythonRobotics 动画规范：

- 动画方式：`plt.cla()` + `plt.pause(0.001)` 逐帧刷新
- 颜色语义：红(.r)=参考路径、蓝(-b)=实际轨迹、绿(xg)=目标点、黑(.k)=障碍物
- 运行方式：直接执行各模块脚本，如 `python perception/lane_pixel_detector.py`

### 模块动画说明

| 模块 | 动画内容 |
|------|---------|
| lane_pixel_detector | 4子图可视化（原图/HLS掩码/检测结果/车道边界） |
| obstacle_detector | RANSAC迭代动画+4子图管线结果（原始点云/地面分离/DBSCAN聚类/OBB包围盒） |

---

## 模块间关联关系

```
摄像头图像 ──→ lane_pixel_detector ──→ scan_rows, center_x ──┐
                                                              │
摄像头图像 ──→ sign_recognizer ──────→ 标志列表 ──────────────┤──→ sensor_fusion ──→ PerceptionOutput
                                                              │
激光雷达 ────→ obstacle_detector ────→ 障碍物列表 ────────────┘
```

三个检测模块各自独立运行，输出原始检测结果。`sensor_fusion` 作为汇聚节点，将所有结果统一到车辆坐标系并封装为 protobuf。下游 `decision` 模块仅消费 `PerceptionOutput`，不直接访问任何检测模块。
