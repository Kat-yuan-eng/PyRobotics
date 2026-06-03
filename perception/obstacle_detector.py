import sys
import os
import numpy as np
from sklearn.cluster import DBSCAN

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# === Phase 1: RANSAC Ground Removal ===

def _fit_plane_ransac(points, n_iter, thresh, rng):
    n = len(points)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0

    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]

        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-12:
            continue
        normal /= norm_len
        d = -np.dot(normal, p0)

        dists = np.abs(points @ normal + d)
        inliers = dists < thresh
        count = inliers.sum()
        if count > best_count:
            best_count = count
            best_inliers = inliers

    return best_inliers


# === Phase 2: Voxel Downsample ===

def _voxel_downsample(points, voxel_size):
    voxel_idx = np.floor(points / voxel_size).astype(np.int64)
    _, unique_map = np.unique(voxel_idx, axis=0, return_inverse=True)

    n_unique = unique_map.max() + 1
    sums = np.zeros((n_unique, 3), dtype=np.float64)
    counts = np.zeros(n_unique, dtype=np.float64)
    np.add.at(sums, unique_map, points)
    np.add.at(counts, unique_map, 1)
    counts = np.maximum(counts, 1.0)
    centroids = sums / counts[:, None]
    return centroids


# === Phase 3: DBSCAN Clustering ===

def _dbscan(points, eps, min_samples):
    n = len(points)
    if n == 0:
        return np.array([], dtype=np.int32)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, algorithm="auto").fit(points)
    return clustering.labels_.astype(np.int32)


# === Phase 4: PCA-OBB Bounding Box ===

def _fit_obb(cluster_pts):
    mean = cluster_pts.mean(axis=0)
    centered = cluster_pts - mean
    cov = centered[:, :2].T @ centered[:, :2] / max(len(centered) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if eigvals[-1] < 1e-6:
        dx_pts = centered[:, 0]
        dy_pts = centered[:, 1]
        if np.std(dx_pts) > np.std(dy_pts):
            eigvecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        else:
            eigvecs = np.array([[0.0, 1.0], [1.0, 0.0]])

    projected = centered[:, :2] @ eigvecs
    half_extents = (projected.max(axis=0) - projected.min(axis=0)) / 2.0
    heading = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

    return mean[:2], float(mean[2]), half_extents[0] * 2, half_extents[1] * 2, heading


# === Phase 5: Obstacle Type Classification ===

_OBSTACLE_TYPE_VEHICLE = "OBSTACLE_VEHICLE"
_OBSTACLE_TYPE_PEDESTRIAN = "OBSTACLE_PEDESTRIAN"
_OBSTACLE_TYPE_CYCLIST = "OBSTACLE_CYCLIST"
_OBSTACLE_TYPE_STATIC = "OBSTACLE_STATIC"
_OBSTACLE_TYPE_UNKNOWN = "OBSTACLE_UNKNOWN"

def _classify_obstacle_type(height, center_z, length, width, n_points):
    area = length * width
    max_dim = max(length, width)
    min_dim = min(length, width) + 1e-9

    if height > 1.2 and area > 2.0 and max_dim > 2.0:
        return _OBSTACLE_TYPE_VEHICLE
    if height > 1.2 and height < 2.2 and area > 0.1 and area < 1.5 and min_dim < 0.8:
        return _OBSTACLE_TYPE_PEDESTRIAN
    if height > 1.0 and height < 2.0 and area > 0.5 and area < 3.0 and min_dim < 1.0 and max_dim > 1.2:
        return _OBSTACLE_TYPE_CYCLIST
    if center_z < 0.5 and area < 1.0:
        return _OBSTACLE_TYPE_STATIC
    if height < 1.2 and area < 2.0:
        return _OBSTACLE_TYPE_STATIC
    return _OBSTACLE_TYPE_UNKNOWN


# === Phase 6: Obstacle Filters ===

def _filter_obstacles(obstacles, min_height, min_height_low, max_aspect_ratio,
                      max_length, min_area, min_area_low, min_points, min_points_low):
    filtered = []
    for obs in obstacles:
        center_z = obs.get("center_z", 0.0)
        is_low = min_height_low <= center_z < min_height
        is_static = obs.get("type") == _OBSTACLE_TYPE_STATIC

        if not is_static and center_z < min_height_low:
            continue
        if is_static and center_z < min_height_low:
            continue

        length = obs["length"]
        width = obs["width"]
        aspect = max(length, width) / (min(length, width) + 1e-9)
        if not is_static and aspect > max_aspect_ratio:
            continue

        if not is_static and max(length, width) > max_length:
            continue

        area = length * width
        area_thresh = min_area_low if is_low else min_area
        if is_static:
            area_thresh = min_area_low
        if area < area_thresh:
            continue

        pts_thresh = min_points_low if is_low else min_points
        if is_static:
            pts_thresh = min_points_low
        if obs["n_points"] < pts_thresh:
            continue

        filtered.append(obs)
    return filtered


# === Phase 7: Distance-Adaptive DBSCAN ===

def _adaptive_dbscan_params(non_ground, base_eps, base_min_pts):
    if len(non_ground) == 0:
        return base_eps, base_min_pts

    x_mean = non_ground[:, 0].mean()
    x_std = max(non_ground[:, 0].std(), 1.0)
    far_thresh = x_mean + x_std
    near_thresh = x_mean - x_std

    far_mask = non_ground[:, 0] > far_thresh
    far_ratio = far_mask.sum() / max(len(non_ground), 1)

    eps = base_eps * (1.0 + 0.5 * far_ratio)
    min_pts = max(int(base_min_pts * (1.0 - 0.3 * far_ratio)), 3)
    return eps, min_pts


# === Phase 8: Detect ===

def detect_obstacles(points, ransac_iter=100, ransac_thresh=0.1,
                     voxel_size=0.1, dbscan_eps=0.5, dbscan_min=5,
                     min_height=0.2, min_height_low=0.05,
                     max_aspect_ratio=6.0, max_length=5.0,
                     min_area=0.1, min_area_low=0.05,
                     min_points=8, min_points_low=4,
                     rng_seed=42, adaptive_dbscan=True,
                     near_ground_thresh=0.2, near_ground_dbscan_eps=0.8,
                     near_ground_dbscan_min=3):
    assert points.ndim == 2 and points.shape[1] == 3, \
        f"points must be (N,3), got {points.shape}"

    points = points[np.all(np.isfinite(points), axis=1)]

    if len(points) < 10:
        return []

    rng = np.random.default_rng(rng_seed)
    ground_mask = _fit_plane_ransac(points, ransac_iter, ransac_thresh, rng)
    non_ground = points[~ground_mask]

    ground_pts = points[ground_mask]
    if len(ground_pts) > 0:
        ground_z_mean = ground_pts[:, 2].mean()
    else:
        ground_z_mean = 0.0

    near_ground_mask = (ground_mask) & (points[:, 2] > ground_z_mean + 0.03) & \
                       (points[:, 2] < ground_z_mean + near_ground_thresh)
    near_ground_pts = points[near_ground_mask]

    if len(non_ground) < dbscan_min and len(near_ground_pts) < near_ground_dbscan_min:
        return []

    downsampled = _voxel_downsample(non_ground, voxel_size) \
        if len(non_ground) > 5000 else non_ground

    if adaptive_dbscan:
        eps, min_pts = _adaptive_dbscan_params(downsampled, dbscan_eps, dbscan_min)
    else:
        eps, min_pts = dbscan_eps, dbscan_min

    labels = _dbscan(downsampled, eps, min_pts)

    obstacles = []
    for lbl in range(labels.max() + 1):
        cluster = downsampled[labels == lbl]
        if len(cluster) < 3:
            continue
        center, center_z, length, width, heading = _fit_obb(cluster)
        height = float(cluster[:, 2].max() - cluster[:, 2].min())
        obs_type = _classify_obstacle_type(height, center_z, length, width, len(cluster))
        obstacles.append({
            "center": center,
            "center_z": center_z,
            "height": height,
            "length": float(length),
            "width": float(width),
            "heading": float(heading),
            "n_points": int(len(cluster)),
            "type": obs_type,
        })

    if len(near_ground_pts) >= near_ground_dbscan_min:
        ng_labels = _dbscan(near_ground_pts, near_ground_dbscan_eps, near_ground_dbscan_min)
        for lbl in range(ng_labels.max() + 1):
            cluster = near_ground_pts[ng_labels == lbl]
            if len(cluster) < 3:
                continue
            center, center_z, length, width, heading = _fit_obb(cluster)
            height = float(cluster[:, 2].max() - cluster[:, 2].min())
            obstacles.append({
                "center": center,
                "center_z": center_z,
                "height": height,
                "length": float(length),
                "width": float(width),
                "heading": float(heading),
                "n_points": int(len(cluster)),
                "type": _OBSTACLE_TYPE_STATIC,
            })

    obstacles = _filter_obstacles(obstacles, min_height, min_height_low,
                                  max_aspect_ratio, max_length,
                                  min_area, min_area_low,
                                  min_points, min_points_low)
    return obstacles


# === Phase 9: Test ===

def generate_test_point_cloud():
    rng = np.random.default_rng(42)

    ground_x = rng.uniform(-5, 5, 2000)
    ground_y = rng.uniform(-3, 3, 2000)
    ground_z = rng.normal(0, 0.02, 2000)
    ground = np.column_stack([ground_x, ground_y, ground_z])

    box1_center = np.array([3.0, 1.0, 0.5])
    box1 = box1_center + rng.uniform([-0.5, -0.3, 0], [0.5, 0.3, 1.0], (200, 3))

    box2_center = np.array([6.0, -1.5, 0.4])
    box2 = box2_center + rng.uniform([-0.4, -0.2, 0], [0.4, 0.2, 0.8], (150, 3))

    curb_x = rng.uniform(-2, 8, 80)
    curb_y = np.full(80, 2.8)
    curb_z = rng.uniform(0.05, 0.12, 80)
    curb = np.column_stack([curb_x, curb_y, curb_z])

    ped_center = np.array([-2.0, 0.5, 0.85])
    ped = ped_center + rng.uniform([-0.2, -0.15, 0], [0.2, 0.15, 1.7], (120, 3))

    veh_center = np.array([8.0, 0.0, 0.75])
    veh_x = rng.uniform(-2.25, 2.25, 300)
    veh_y = rng.uniform(-0.9, 0.9, 300)
    veh_z = rng.uniform(0, 1.5, 300)
    veh = veh_center + np.column_stack([veh_x, veh_y, veh_z])

    return np.vstack([ground, box1, box2, curb, ped, veh])


def _draw_obb(ax, center, length, width, heading, color='r', linewidth=2):
    corners_local = np.array([
        [-length / 2, -width / 2], [length / 2, -width / 2],
        [length / 2, width / 2], [-length / 2, width / 2], [-length / 2, -width / 2]
    ])
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    corners_global = corners_local @ R.T + center
    ax.plot(corners_global[:, 0], corners_global[:, 1], '-', color=color, linewidth=linewidth)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    SHOW_ANIMATION = True

    pc = generate_test_point_cloud()

    obs = detect_obstacles(pc)
    print(f"Detected {len(obs)} obstacles:")
    for i, o in enumerate(obs):
        print(f"  [{i}] center=({o['center'][0]:.2f}, {o['center'][1]:.2f})  "
              f"z={o['center_z']:.2f}m  L={o['length']:.2f}m  W={o['width']:.2f}m  "
              f"hdg={np.degrees(o['heading']):.1f}deg  pts={o['n_points']}  type={o['type']}")

    if SHOW_ANIMATION:
        rng = np.random.default_rng(42)
        ground_mask = _fit_plane_ransac(pc, 100, 0.1, rng)
        non_ground = pc[~ground_mask]
        downsampled = _voxel_downsample(non_ground, 0.1) if len(non_ground) > 5000 else non_ground
        eps, min_pts = _adaptive_dbscan_params(downsampled, 0.5, 5)
        labels = _dbscan(downsampled, eps, min_pts)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ax1 = axes[0, 0]
        ax1.scatter(pc[:, 0], pc[:, 1], s=1, c='0.6', alpha=0.3)
        ax1.set_title("Raw Point Cloud (Top View)")
        ax1.set_xlabel("x[m]")
        ax1.set_ylabel("y[m]")
        ax1.set_aspect('equal')
        ax1.grid(True)

        ax2 = axes[0, 1]
        ax2.scatter(pc[ground_mask, 0], pc[ground_mask, 1],
                    s=1, c='0.8', alpha=0.3, label='ground')
        ax2.scatter(pc[~ground_mask, 0], pc[~ground_mask, 1],
                    s=1, c='r', alpha=0.5, label='non-ground')
        ax2.set_title("RANSAC Ground Removal")
        ax2.set_xlabel("x[m]")
        ax2.set_ylabel("y[m]")
        ax2.legend(frameon=True, fancybox=True)
        ax2.set_aspect('equal')
        ax2.grid(True)

        ax3 = axes[1, 0]
        unique_labels = sorted(set(labels.tolist()) - {-1})
        cmap = plt.cm.Set1(np.linspace(0, 1, max(len(unique_labels), 1)))
        noise_mask = labels == -1
        ax3.scatter(downsampled[noise_mask, 0], downsampled[noise_mask, 1],
                    s=1, c='0.7', alpha=0.3, label='noise')
        for lbl, c in zip(unique_labels, cmap):
            mask = labels == lbl
            ax3.scatter(downsampled[mask, 0], downsampled[mask, 1],
                        s=3, c=[c], alpha=0.6, label=f'cluster {lbl}')
        ax3.set_title("DBSCAN Clustering (Adaptive)")
        ax3.set_xlabel("x[m]")
        ax3.set_ylabel("y[m]")
        ax3.legend(frameon=True, fancybox=True)
        ax3.set_aspect('equal')
        ax3.grid(True)

        ax4 = axes[1, 1]
        ax4.scatter(downsampled[:, 0], downsampled[:, 1], s=1, c='0.6', alpha=0.3)
        type_colors = {
            _OBSTACLE_TYPE_VEHICLE: 'b',
            _OBSTACLE_TYPE_PEDESTRIAN: 'g',
            _OBSTACLE_TYPE_CYCLIST: 'm',
            _OBSTACLE_TYPE_STATIC: 'orange',
            _OBSTACLE_TYPE_UNKNOWN: 'r',
        }
        for i, o in enumerate(obs):
            color = type_colors.get(o['type'], 'r')
            _draw_obb(ax4, o['center'], o['length'], o['width'], o['heading'], color=color)
            ax4.plot(o['center'][0], o['center'][1], 'xk', markersize=10)
            ax4.text(o['center'][0], o['center'][1] + 0.3,
                     f"#{i} {o['type'].replace('OBSTACLE_', '')}",
                     ha='center', fontsize=8)
        ax4.set_title("OBB Bounding Boxes (Typed)")
        ax4.set_xlabel("x[m]")
        ax4.set_ylabel("y[m]")
        ax4.set_aspect('equal')
        ax4.grid(True)

        plt.tight_layout()
        os.makedirs("figs", exist_ok=True)
        plt.savefig("figs/obstacle_detector.png", dpi=150)
        plt.show()
