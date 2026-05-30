import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# === Phase 1: RANSAC Ground Removal ===

def _fit_plane_ransac(points, n_iter, thresh):
    n = len(points)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0

    for _ in range(n_iter):
        idx = np.random.choice(n, 3, replace=False)
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
    voxel_idx = np.floor(points / voxel_size).astype(np.int32)
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


# === Phase 5: Obstacle Filters ===

def _filter_obstacles(obstacles, min_height, max_aspect_ratio, min_area, min_points):
    filtered = []
    for obs in obstacles:
        center_z = obs.get("center_z", 0.0)
        if center_z < min_height:
            continue

        length = obs["length"]
        width = obs["width"]
        aspect = max(length, width) / (min(length, width) + 1e-9)
        if aspect > max_aspect_ratio:
            continue

        area = length * width
        if area < min_area:
            continue

        if obs["n_points"] < min_points:
            continue

        filtered.append(obs)
    return filtered


# === Phase 6: Detect ===

def detect_obstacles(points, ransac_iter=100, ransac_thresh=0.1,
                     voxel_size=0.1, dbscan_eps=0.5, dbscan_min=5,
                     min_height=0.2, max_aspect_ratio=10.0,
                     min_area=0.1, min_points=8):
    if not (points.ndim == 2 and points.shape[1] == 3):
        raise ValueError(f"points must be (N,3), got {points.shape}")

    points = points[np.all(np.isfinite(points), axis=1)]

    if len(points) < 10:
        return []

    ground_mask = _fit_plane_ransac(points, ransac_iter, ransac_thresh)
    non_ground = points[~ground_mask]

    if len(non_ground) < dbscan_min:
        return []

    downsampled = _voxel_downsample(non_ground, voxel_size) \
        if len(non_ground) > 5000 else non_ground

    labels = _dbscan(downsampled, dbscan_eps, dbscan_min)

    obstacles = []
    for lbl in range(labels.max() + 1):
        cluster = downsampled[labels == lbl]
        if len(cluster) < 3:
            continue
        center, center_z, length, width, heading = _fit_obb(cluster)
        obstacles.append({
            "center": center,
            "center_z": center_z,
            "length": float(length),
            "width": float(width),
            "heading": float(heading),
            "n_points": int(len(cluster)),
        })

    obstacles = _filter_obstacles(obstacles, min_height, max_aspect_ratio,
                                  min_area, min_points)
    return obstacles


# === Phase 7: Test ===

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

    return np.vstack([ground, box1, box2, curb])


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
    show_animation = True

    pc = generate_test_point_cloud()

    if show_animation:
        n_iter = 100
        thresh = 0.1
        n_pts = len(pc)
        best_inliers = np.zeros(n_pts, dtype=bool)
        best_count = 0

        plt.figure(figsize=(10, 6))
        for it in range(n_iter):
            idx = np.random.choice(n_pts, 3, replace=False)
            p0, p1, p2 = pc[idx]
            normal = np.cross(p1 - p0, p2 - p0)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-12:
                continue
            normal /= norm_len
            d = -np.dot(normal, p0)
            dists = np.abs(pc @ normal + d)
            inliers = dists < thresh
            count = inliers.sum()
            if count > best_count:
                best_count = count
                best_inliers = inliers

            if it % 10 == 0 or it == n_iter - 1:
                plt.cla()
                plt.scatter(pc[best_inliers, 0], pc[best_inliers, 1],
                            s=1, c='0.8', alpha=0.3, label='ground')
                plt.scatter(pc[~best_inliers, 0], pc[~best_inliers, 1],
                            s=1, c='r', alpha=0.5, label='non-ground')
                plt.title(f"RANSAC Ground Removal  iter={it + 1}/{n_iter}  inliers={best_count}")
                plt.xlabel("x[m]")
                plt.ylabel("y[m]")
                plt.legend(frameon=True, fancybox=True)
                plt.grid(True)
                plt.pause(0.001)

    obs = detect_obstacles(pc)
    print(f"Detected {len(obs)} obstacles (expected 2):")
    for i, o in enumerate(obs):
        print(f"  [{i}] center=({o['center'][0]:.2f}, {o['center'][1]:.2f})  "
              f"L={o['length']:.2f}m  W={o['width']:.2f}m  "
              f"hdg={np.degrees(o['heading']):.1f}deg  pts={o['n_points']}")

    if show_animation:
        ground_mask = _fit_plane_ransac(pc, 100, 0.1)
        non_ground = pc[~ground_mask]
        downsampled = _voxel_downsample(non_ground, 0.1) if len(non_ground) > 5000 else non_ground
        labels = _dbscan(downsampled, 0.5, 5)

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
        ax3.set_title("DBSCAN Clustering")
        ax3.set_xlabel("x[m]")
        ax3.set_ylabel("y[m]")
        ax3.legend(frameon=True, fancybox=True)
        ax3.set_aspect('equal')
        ax3.grid(True)

        ax4 = axes[1, 1]
        ax4.scatter(downsampled[:, 0], downsampled[:, 1], s=1, c='0.6', alpha=0.3)
        for i, o in enumerate(obs):
            _draw_obb(ax4, o['center'], o['length'], o['width'], o['heading'], color='r')
            ax4.plot(o['center'][0], o['center'][1], 'xg', markersize=10)
            ax4.text(o['center'][0], o['center'][1] + 0.3, f"#{i}", ha='center', fontsize=9)
        ax4.set_title("OBB Bounding Boxes")
        ax4.set_xlabel("x[m]")
        ax4.set_ylabel("y[m]")
        ax4.set_aspect('equal')
        ax4.grid(True)

        plt.tight_layout()
        plt.show()
