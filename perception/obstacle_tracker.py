import numpy as np
from scipy.optimize import linear_sum_assignment


# === Phase 1: Hungarian Assignment ===

def _hungarian_assign(cost_mat, max_cost):
    n_det, n_trk = cost_mat.shape
    if n_det == 0 or n_trk == 0:
        return np.full(n_det, -1, dtype=int), list(range(n_det))

    row_ind, col_ind = linear_sum_assignment(cost_mat)

    assigned_det = np.full(n_det, -1, dtype=int)
    for r, c in zip(row_ind, col_ind):
        if cost_mat[r, c] < max_cost:
            assigned_det[r] = c

    unmatched_det = [d for d in range(n_det) if assigned_det[d] == -1]
    return assigned_det, unmatched_det


# === Phase 2: Kalman Filter ===

_KF_F = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], dtype=np.float64)

_KF_H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
], dtype=np.float64)

_KF_Q_BASE = np.diag([0.1, 0.1, 0.5, 0.5]) ** 2
_KF_R_BASE = np.diag([0.5, 0.5]) ** 2

def _kf_predict(x, P, dt):
    F = _KF_F.copy()
    F[0, 2] = dt
    F[1, 3] = dt
    x_pred = F @ x
    P_pred = F @ P @ F.T + _KF_Q_BASE
    return x_pred, P_pred

def _kf_update(x, P, z):
    y = z - _KF_H @ x
    S = _KF_H @ P @ _KF_H.T + _KF_R_BASE
    K = P @ _KF_H.T @ np.linalg.inv(S)
    x_upd = x + K @ y
    P_upd = (np.eye(4) - K @ _KF_H) @ P
    return x_upd, P_upd


# === Phase 3: Track State ===

def _predict_track(track, dt, use_kalman):
    if use_kalman and "kf_x" in track:
        kf_x, kf_P = _kf_predict(track["kf_x"], track["kf_P"], dt)
        track["kf_x"] = kf_x
        track["kf_P"] = kf_P
        track["x"] = float(kf_x[0])
        track["y"] = float(kf_x[1])
        track["vx"] = float(kf_x[2])
        track["vy"] = float(kf_x[3])
    else:
        track["x"] += track["vx"] * dt
        track["y"] += track["vy"] * dt

    track["age"] += 1
    track["hits_since_update"] += 1
    return track


def _update_track(track, det, dt, alpha, use_kalman):
    det_x = float(det["center"][0])
    det_y = float(det["center"][1])

    if use_kalman and "kf_x" in track:
        z = np.array([det_x, det_y])
        kf_x, kf_P = _kf_update(track["kf_x"], track["kf_P"], z)
        track["kf_x"] = kf_x
        track["kf_P"] = kf_P
        track["x"] = float(kf_x[0])
        track["y"] = float(kf_x[1])
        track["vx"] = float(kf_x[2])
        track["vy"] = float(kf_x[3])
    else:
        track["x_prev"] = track["x"]
        track["y_prev"] = track["y"]
        track["x"] = alpha * det_x + (1 - alpha) * track["x"]
        track["y"] = alpha * det_y + (1 - alpha) * track["y"]

        if track["hits"] >= 2 and dt > 1e-9:
            vx_meas = (det_x - track["x_prev"]) / dt
            vy_meas = (det_y - track["y_prev"]) / dt
            track["vx"] = alpha * vx_meas + (1 - alpha) * track["vx"]
            track["vy"] = alpha * vy_meas + (1 - alpha) * track["vy"]

    if "vx" in det and "vy" in det:
        track["vx"] = alpha * det["vx"] + (1 - alpha) * track["vx"]
        track["vy"] = alpha * det["vy"] + (1 - alpha) * track["vy"]

    track["length"] = det.get("length", track["length"])
    track["width"] = det.get("width", track["width"])
    track["heading"] = det.get("heading", track["heading"])
    track["type"] = det.get("type", track.get("type", "OBSTACLE_UNKNOWN"))
    track["hits"] += 1
    track["hits_since_update"] = 0
    track["age"] = 0
    track["occluded"] = False
    return track


# === Phase 4: Track Obstacles ===

def track_obstacles(detections, tracks, dt=0.1,
                    max_dist=3.0, min_hits=3, max_age=5,
                    max_age_occluded=15, alpha=0.3,
                    use_kalman=True, free_ids=None):
    if free_ids is None:
        free_ids = set()

    for track in tracks:
        _predict_track(track, dt, use_kalman)

    if len(detections) == 0:
        surviving = []
        for t in tracks:
            age_limit = max_age_occluded if t.get("confirmed", False) else max_age
            if t["age"] < age_limit:
                t["occluded"] = True
                surviving.append(t)
            else:
                free_ids.add(t["id"])
        return surviving

    if len(tracks) == 0:
        next_id_init = _alloc_id(free_ids, 0)
        for det in detections:
            new_track = _create_track(det, next_id_init, dt, use_kalman)
            tracks.append(new_track)
            next_id_init = _alloc_id(free_ids, next_id_init + 1)
        return tracks

    n_det = len(detections)
    n_trk = len(tracks)

    det_xy = np.array([d["center"] for d in detections])
    trk_xy = np.array([[t["x"], t["y"]] for t in tracks])
    cost_mat = np.sqrt(((det_xy[:, None, :] - trk_xy[None, :, :]) ** 2).sum(axis=2))

    assigned_det, unmatched_det = _hungarian_assign(cost_mat, max_dist)

    next_id = max((t["id"] for t in tracks), default=0) + 1
    unmatched_set = set(unmatched_det)

    for d in range(n_det):
        t = assigned_det[d]
        if t >= 0 and cost_mat[d, t] < max_dist:
            _update_track(tracks[t], detections[d], dt, alpha, use_kalman)
        else:
            unmatched_set.add(d)

    for d in unmatched_set:
        if d < n_det:
            new_id = _alloc_id(free_ids, next_id)
            new_track = _create_track(detections[d], new_id, dt, use_kalman)
            tracks.append(new_track)
            if new_id >= next_id:
                next_id = new_id + 1

    for track in tracks:
        if track["hits"] >= min_hits:
            track["confirmed"] = True

    surviving = []
    for t in tracks:
        age_limit = max_age_occluded if t.get("confirmed", False) else max_age
        if t["age"] < age_limit:
            surviving.append(t)
        else:
            free_ids.add(t["id"])
    return surviving


def _alloc_id(free_ids, fallback):
    if free_ids:
        return free_ids.pop()
    return fallback


def _create_track(det, track_id, dt, use_kalman):
    det_x = float(det["center"][0])
    det_y = float(det["center"][1])
    track = {
        "id": track_id,
        "x": det_x,
        "y": det_y,
        "x_prev": det_x,
        "y_prev": det_y,
        "vx": 0.0,
        "vy": 0.0,
        "length": det.get("length", 1.0),
        "width": det.get("width", 1.0),
        "heading": det.get("heading", 0.0),
        "type": det.get("type", "OBSTACLE_UNKNOWN"),
        "hits": 1,
        "hits_since_update": 0,
        "age": 0,
        "confirmed": False,
        "occluded": False,
    }
    if use_kalman:
        track["kf_x"] = np.array([det_x, det_y, 0.0, 0.0], dtype=np.float64)
        track["kf_P"] = np.diag([0.5, 0.5, 2.0, 2.0]) ** 2
    return track


# === Phase 5: Get Confirmed ===

def get_confirmed_tracks(tracks):
    results = []
    for t in tracks:
        if t["confirmed"]:
            results.append({
                "id": t["id"],
                "center": np.array([t["x"], t["y"]]),
                "vx": t["vx"],
                "vy": t["vy"],
                "speed": np.sqrt(t["vx"]**2 + t["vy"]**2),
                "length": t["length"],
                "width": t["width"],
                "heading": t["heading"],
                "type": t.get("type", "OBSTACLE_UNKNOWN"),
                "confidence": min(t["hits"] / 10.0, 1.0),
                "occluded": t.get("occluded", False),
            })
    return results
