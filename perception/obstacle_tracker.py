import numpy as np

# === Phase 1: Hungarian Assignment ===

def _hungarian_assign(cost_mat):
    n_det, n_trk = cost_mat.shape
    if n_det == 0 or n_trk == 0:
        return list(range(n_det)), []
    assigned_det = [-1] * n_det
    assigned_trk = [-1] * n_trk
    used_det = set()
    used_trk = set()
    flat_idx = np.argsort(cost_mat, axis=None)
    for idx in flat_idx:
        d = int(idx // n_trk)
        t = int(idx % n_trk)
        if d in used_det or t in used_trk:
            continue
        if cost_mat[d, t] > 1e6:
            break
        assigned_det[d] = t
        assigned_trk[t] = d
        used_det.add(d)
        used_trk.add(t)
    unmatched_det = [d for d in range(n_det) if assigned_det[d] == -1]
    return assigned_det, unmatched_det


# === Phase 2: Track State ===

def _predict_track(track, dt):
    track["x"] += track["vx"] * dt
    track["y"] += track["vy"] * dt
    track["age"] += 1
    track["hits_since_update"] += 1
    return track


def _update_track(track, det):
    alpha = 0.3
    track["x"] = alpha * det["center"][0] + (1 - alpha) * track["x"]
    track["y"] = alpha * det["center"][1] + (1 - alpha) * track["y"]
    if "vx" in det and "vy" in det:
        track["vx"] = alpha * det["vx"] + (1 - alpha) * track["vx"]
        track["vy"] = alpha * det["vy"] + (1 - alpha) * track["vy"]
    track["length"] = det.get("length", track["length"])
    track["width"] = det.get("width", track["width"])
    track["heading"] = det.get("heading", track["heading"])
    track["hits"] += 1
    track["hits_since_update"] = 0
    track["age"] = 0
    return track


# === Phase 3: Track Obstacles ===

def track_obstacles(detections, tracks, dt=0.1,
                    max_dist=3.0, min_hits=3, max_age=5):
    for track in tracks:
        _predict_track(track, dt)

    if len(detections) == 0:
        tracks = [t for t in tracks if t["age"] < max_age]
        return tracks

    if len(tracks) == 0:
        next_id_init = 0
        for det in detections:
            tracks.append({
                "id": next_id_init,
                "x": float(det["center"][0]),
                "y": float(det["center"][1]),
                "vx": 0.0,
                "vy": 0.0,
                "length": det.get("length", 1.0),
                "width": det.get("width", 1.0),
                "heading": det.get("heading", 0.0),
                "hits": 1,
                "hits_since_update": 0,
                "age": 0,
                "confirmed": False,
            })
            next_id_init += 1
        return tracks

    n_det = len(detections)
    n_trk = len(tracks)
    cost_mat = np.full((n_det, n_trk), 1e9)
    for d in range(n_det):
        for t in range(n_trk):
            dx = detections[d]["center"][0] - tracks[t]["x"]
            dy = detections[d]["center"][1] - tracks[t]["y"]
            cost_mat[d, t] = np.sqrt(dx**2 + dy**2)

    assigned_det, unmatched_hungarian = _hungarian_assign(cost_mat)
    unmatched_set = set(unmatched_hungarian)

    next_id = max((t["id"] for t in tracks), default=0) + 1
    for d in range(n_det):
        t = assigned_det[d]
        if t >= 0 and cost_mat[d, t] < max_dist:
            _update_track(tracks[t], detections[d])
        else:
            unmatched_set.add(d)

    for d in unmatched_set:
        if d < n_det:
            tracks.append({
                "id": next_id,
                "x": float(detections[d]["center"][0]),
                "y": float(detections[d]["center"][1]),
                "vx": 0.0,
                "vy": 0.0,
                "length": detections[d].get("length", 1.0),
                "width": detections[d].get("width", 1.0),
                "heading": detections[d].get("heading", 0.0),
                "hits": 1,
                "hits_since_update": 0,
                "age": 0,
                "confirmed": False,
            })
            next_id += 1

    for track in tracks:
        if track["hits"] >= min_hits:
            track["confirmed"] = True

    tracks = [t for t in tracks if t["age"] < max_age]
    return tracks


# === Phase 4: Get Confirmed ===

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
            })
    return results
