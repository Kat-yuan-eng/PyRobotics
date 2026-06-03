import numpy as np
import heapq
import time

# === Phase 1: Grid Map ===

def create_grid_map(ox, oy, resolution, robot_radius, sx=0.0, sy=0.0, gx=0.0, gy=0.0):
    min_x = int(np.floor(min(ox.min(), sx, gx) / resolution)) - 1
    min_y = int(np.floor(min(oy.min(), sy, gy) / resolution)) - 1
    max_x = int(np.ceil(max(ox.max(), sx, gx) / resolution)) + 1
    max_y = int(np.ceil(max(oy.max(), sy, gy) / resolution)) + 1

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    obstacle_map = np.zeros((width, height), dtype=bool)
    r_cells = int(np.ceil(robot_radius / resolution))
    for i in range(len(ox)):
        obs_gx = int(np.round(ox[i] / resolution)) - min_x
        obs_gy = int(np.round(oy[i] / resolution)) - min_y
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                nx, ny = obs_gx + dx, obs_gy + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if dx * dx + dy * dy <= r_cells * r_cells:
                        obstacle_map[nx, ny] = True

    return obstacle_map, min_x, min_y, width, height

# === Phase 2: A* Search ===

MOTION_8 = [(1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

def a_star_plan(sx, sy, gx, gy, ox, oy, resolution=2.0, robot_radius=1.0):
    t0 = time.perf_counter()

    obs_map, min_x, min_y, width, height = create_grid_map(ox, oy, resolution, robot_radius, sx, sy, gx, gy)

    si = int(np.round(sx / resolution)) - min_x
    sj = int(np.round(sy / resolution)) - min_y
    gi = int(np.round(gx / resolution)) - min_x
    gj = int(np.round(gy / resolution)) - min_y

    si = int(np.clip(si, 0, width - 1))
    sj = int(np.clip(sj, 0, height - 1))
    gi = int(np.clip(gi, 0, width - 1))
    gj = int(np.clip(gj, 0, height - 1))

    g_score = np.full((width, height), np.inf)
    g_score[si, sj] = 0.0
    parent = np.full((width, height, 2), -1, dtype=int)

    open_set = [(np.hypot(gi - si, gj - sj), 0, si, sj)]
    closed = np.zeros((width, height), dtype=bool)
    nodes_explored = 0

    while open_set:
        f, _, ci, cj = heapq.heappop(open_set)
        if closed[ci, cj]:
            continue
        closed[ci, cj] = True
        nodes_explored += 1

        if ci == gi and cj == gj:
            break

        for di, dj, cost in MOTION_8:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < width and 0 <= nj < height and not obs_map[ni, nj] and not closed[ni, nj]:
                new_g = g_score[ci, cj] + cost
                if new_g < g_score[ni, nj]:
                    g_score[ni, nj] = new_g
                    h = np.hypot(gi - ni, gj - nj)
                    heapq.heappush(open_set, (new_g + h, nodes_explored, ni, nj))
                    parent[ni, nj] = [ci, cj]

    path_i, path_j = [], []
    if closed[gi, gj] or (gi == si and gj == sj):
        ci, cj = gi, gj
        path_i.append(ci)
        path_j.append(cj)
        while parent[ci, cj, 0] >= 0:
            ci, cj = int(parent[ci, cj, 0]), int(parent[ci, cj, 1])
            path_i.append(ci)
            path_j.append(cj)
        path_i.reverse()
        path_j.reverse()

    rx = np.array(path_i, dtype=float) * resolution + min_x * resolution
    ry = np.array(path_j, dtype=float) * resolution + min_y * resolution

    path_length = 0.0
    if len(rx) > 1:
        path_length = np.sum(np.hypot(np.diff(rx), np.diff(ry)))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    stats = {"planner": "A*", "path_length": path_length, "planning_time_ms": elapsed_ms,
             "nodes_explored": nodes_explored, "path_points": len(rx)}

    return rx, ry, stats
