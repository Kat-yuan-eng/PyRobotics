import sys
import os
import numpy as np
import cv2
import functools

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# === Phase 1: Sign Categories & Templates ===

SIGN_CATEGORIES = [
    "speed_20", "speed_30", "speed_40", "speed_50",
    "speed_60", "speed_80", "speed_100", "speed_120",
    "arrow_left", "arrow_right", "arrow_forward", "arrow_uturn",
    "no_entry", "no_parking",
    "warning_curve_left", "warning_curve_right", "warning_intersection",
    "mandatory_left", "mandatory_right", "mandatory_forward",
    "unknown",
]

_SIGN_COLOR_RED = "red"
_SIGN_COLOR_BLUE = "blue"
_SIGN_COLOR_YELLOW = "yellow"

_SIGN_SHAPE_CIRCLE = "circle"
_SIGN_SHAPE_TRIANGLE = "triangle"
_SIGN_SHAPE_RECT = "rect"

_CATEGORY_META = {
    "speed_20":            {"color": _SIGN_COLOR_RED,    "shape": _SIGN_SHAPE_CIRCLE},
    "speed_30":            {"color": _SIGN_COLOR_RED,    "shape": _SIGN_SHAPE_CIRCLE},
    "speed_40":            {"color": _SIGN_COLOR_RED,    "shape": _SIGN_SHAPE_CIRCLE},
    "speed_50":            {"color": _SIGN_COLOR_RED,    "shape": _SIGN_SHAPE_CIRCLE},
    "speed_60":            {"color": _SIGN_COLOR_RED,    "shape": _SIGN_SHAPE_CIRCLE},
    "speed_80":            {"color": _SIGN_COLOR_RED,    "shape": _SIGN_SHAPE_CIRCLE},
    "speed_100":           {"color": _SIGN_COLOR_RED,    "shape": _SIGN_SHAPE_CIRCLE},
    "speed_120":           {"color": _SIGN_COLOR_RED,    "shape": _SIGN_SHAPE_CIRCLE},
    "arrow_left":          {"color": _SIGN_COLOR_RED,    "shape": _SIGN_SHAPE_CIRCLE},
    "arrow_right":         {"color": _SIGN_COLOR_RED,    "shape": _SIGN_SHAPE_CIRCLE},
    "arrow_forward":       {"color": _SIGN_COLOR_RED,    "shape": _SIGN_SHAPE_CIRCLE},
    "arrow_uturn":         {"color": _SIGN_COLOR_RED,    "shape": _SIGN_SHAPE_CIRCLE},
    "no_entry":            {"color": _SIGN_COLOR_RED,    "shape": _SIGN_SHAPE_CIRCLE},
    "no_parking":          {"color": _SIGN_COLOR_RED,    "shape": _SIGN_SHAPE_CIRCLE},
    "warning_curve_left":  {"color": _SIGN_COLOR_YELLOW, "shape": _SIGN_SHAPE_TRIANGLE},
    "warning_curve_right": {"color": _SIGN_COLOR_YELLOW, "shape": _SIGN_SHAPE_TRIANGLE},
    "warning_intersection":{"color": _SIGN_COLOR_YELLOW, "shape": _SIGN_SHAPE_TRIANGLE},
    "mandatory_left":      {"color": _SIGN_COLOR_BLUE,   "shape": _SIGN_SHAPE_CIRCLE},
    "mandatory_right":     {"color": _SIGN_COLOR_BLUE,   "shape": _SIGN_SHAPE_CIRCLE},
    "mandatory_forward":   {"color": _SIGN_COLOR_BLUE,   "shape": _SIGN_SHAPE_CIRCLE},
}

_TEMPLATE_SIZE = 128
_HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2))
_ROTATION_ANGLES = list(range(0, 360, 15))


def _generate_template(category, size=_TEMPLATE_SIZE):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    r = int(size * 0.42)
    meta = _CATEGORY_META.get(category, {})
    color = meta.get("color", _SIGN_COLOR_RED)
    shape = meta.get("shape", _SIGN_SHAPE_CIRCLE)

    if color == _SIGN_COLOR_RED:
        border_color = (0, 0, 255)
    elif color == _SIGN_COLOR_BLUE:
        border_color = (255, 100, 0)
    else:
        border_color = (0, 200, 255)

    if shape == _SIGN_SHAPE_CIRCLE:
        cv2.circle(img, (cx, cy), r, border_color, 3)
        cv2.circle(img, (cx, cy), r - 4, (255, 255, 255), -1)

        if category.startswith("speed_"):
            num = category.split("_")[1]
            font_scale = 2.4 if len(num) <= 2 else 1.8
            text_size = cv2.getTextSize(num, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
            text_x = cx - text_size[0] // 2
            text_y = cy + text_size[1] // 2
            cv2.putText(img, num, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3)
        elif category.startswith("arrow_"):
            if category == "arrow_left":
                pts = np.array([[cx + 15, cy - 10], [cx - 10, cy], [cx + 15, cy + 10]])
            elif category == "arrow_right":
                pts = np.array([[cx - 15, cy - 10], [cx + 10, cy], [cx - 15, cy + 10]])
            elif category == "arrow_forward":
                pts = np.array([[cx - 10, cy + 15], [cx, cy - 10], [cx + 10, cy + 15]])
            else:
                pts = np.array([[cx + 10, cy + 5], [cx - 5, cy - 10],
                                [cx - 5, cy], [cx - 15, cy],
                                [cx - 15, cy + 10], [cx - 5, cy + 10]])
            cv2.fillPoly(img, [pts], (0, 0, 0))
        elif category == "no_entry":
            cv2.circle(img, (cx, cy), r, (0, 0, 255), -1)
            cv2.circle(img, (cx, cy), r - 3, (255, 255, 255), 3)
            cv2.rectangle(img, (cx - r + 8, cy - 3), (cx + r - 8, cy + 3),
                          (0, 0, 255), -1)
        elif category == "no_parking":
            cv2.circle(img, (cx, cy), r, (0, 0, 200), 3)
            cv2.circle(img, (cx, cy), r - 4, (255, 255, 255), -1)
            cv2.putText(img, "P", (cx - 8, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.line(img, (cx - r + 6, cy + r - 6),
                     (cx + r - 6, cy - r + 6), (0, 0, 200), 3)
        elif category.startswith("mandatory_"):
            if category == "mandatory_left":
                pts = np.array([[cx + 15, cy - 10], [cx - 10, cy], [cx + 15, cy + 10]])
            elif category == "mandatory_right":
                pts = np.array([[cx - 15, cy - 10], [cx + 10, cy], [cx - 15, cy + 10]])
            else:
                pts = np.array([[cx - 10, cy + 15], [cx, cy - 10], [cx + 10, cy + 15]])
            cv2.fillPoly(img, [pts], (255, 255, 255))

    elif shape == _SIGN_SHAPE_TRIANGLE:
        h_tri = int(r * 1.7)
        pts = np.array([
            [cx, cy - h_tri // 2],
            [cx - h_tri // 2, cy + h_tri // 2],
            [cx + h_tri // 2, cy + h_tri // 2],
        ])
        cv2.fillPoly(img, [pts], border_color)
        inner_margin = int(h_tri * 0.15)
        pts_inner = np.array([
            [cx, cy - h_tri // 2 + inner_margin * 2],
            [cx - h_tri // 2 + inner_margin * 2, cy + h_tri // 2 - inner_margin],
            [cx + h_tri // 2 - inner_margin * 2, cy + h_tri // 2 - inner_margin],
        ])
        cv2.fillPoly(img, [pts_inner], (255, 255, 255))

        if category == "warning_curve_left":
            pts_arrow = np.array([[cx + 8, cy - 5], [cx - 8, cy], [cx + 8, cy + 5]])
            cv2.fillPoly(img, [pts_arrow], (0, 0, 0))
        elif category == "warning_curve_right":
            pts_arrow = np.array([[cx - 8, cy - 5], [cx + 8, cy], [cx - 8, cy + 5]])
            cv2.fillPoly(img, [pts_arrow], (0, 0, 0))
        elif category == "warning_intersection":
            cv2.putText(img, "+", (cx - 6, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return img


def _augment_template(img, rng):
    aug_type = rng.integers(0, 5)
    if aug_type == 0:
        return img
    elif aug_type == 1:
        noise = rng.normal(0, 10, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif aug_type == 2:
        return cv2.GaussianBlur(img, (3, 3), 0.5)
    elif aug_type == 3:
        factor = 0.7 + rng.random() * 0.6
        return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    else:
        M = np.float32([[1, 0, rng.integers(-2, 3)], [0, 1, rng.integers(-2, 3)]])
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def _compute_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (_TEMPLATE_SIZE, _TEMPLATE_SIZE))
    win_size = (_TEMPLATE_SIZE, _TEMPLATE_SIZE)
    block_size = (32, 32)
    block_stride = (16, 16)
    cell_size = (16, 16)
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
                            cell_size, _HOG_PARAMS["orientations"])
    feat = hog.compute(resized)
    feat = feat.flatten()
    norm = np.linalg.norm(feat) + 1e-12
    feat = feat / norm

    margin = _TEMPLATE_SIZE // 4
    roi = resized[margin:_TEMPLATE_SIZE - margin, margin:_TEMPLATE_SIZE - margin]
    if roi.size > 0:
        roi_resized = cv2.resize(roi, (32, 32))
        roi_feat = roi_resized.flatten().astype(np.float64) / 255.0
        roi_norm = np.linalg.norm(roi_feat) + 1e-12
        roi_feat = roi_feat / roi_norm
        feat = np.concatenate([feat, roi_feat * 0.3])

    return feat


def _build_template_db():
    db = []
    rng = np.random.default_rng(123)
    for cat in SIGN_CATEGORIES:
        if cat == "unknown":
            continue
        tmpl = _generate_template(cat)
        for angle in _ROTATION_ANGLES:
            M = cv2.getRotationMatrix2D((_TEMPLATE_SIZE / 2, _TEMPLATE_SIZE / 2),
                                        angle, 1.0)
            rotated = cv2.warpAffine(tmpl, M, (_TEMPLATE_SIZE, _TEMPLATE_SIZE))
            feat = _compute_hog(rotated)
            db.append((cat, feat))
            for _ in range(2):
                aug = _augment_template(rotated, rng)
                feat_aug = _compute_hog(aug)
                db.append((cat, feat_aug))
    return db


@functools.lru_cache(maxsize=1)
def _get_template_db():
    _build_template_db.cache_clear = None
    return _build_template_db()


# === Phase 2: Sign Localization (Multi-Color) ===

def _localize_signs(image, min_area):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask_r1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)

    mask_blue = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))

    mask_yellow = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    candidates = []
    for mask, color in [(mask_red, _SIGN_COLOR_RED),
                        (mask_blue, _SIGN_COLOR_BLUE),
                        (mask_yellow, _SIGN_COLOR_YELLOW)]:
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            if aspect < 0.4 or aspect > 2.5:
                continue
            shape = _classify_shape(cnt, aspect)
            candidates.append((x, y, w, h, cnt, color, shape))
    return candidates


def _classify_shape(cnt, aspect):
    perimeter = cv2.arcLength(cnt, True) + 1e-9
    area = cv2.contourArea(cnt) + 1e-9
    circularity = 4 * np.pi * area / (perimeter ** 2)

    epsilon = 0.04 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    n_vertices = len(approx)

    if circularity > 0.6:
        return _SIGN_SHAPE_CIRCLE
    if n_vertices == 3:
        return _SIGN_SHAPE_TRIANGLE
    if n_vertices >= 4 and 0.8 < aspect < 1.2:
        return _SIGN_SHAPE_RECT
    if circularity > 0.4:
        return _SIGN_SHAPE_CIRCLE
    if 0.7 < aspect < 1.4:
        return _SIGN_SHAPE_CIRCLE
    return _SIGN_SHAPE_CIRCLE


# === Phase 3: Classify ===

def _classify_sign(patch, score_threshold, candidate_color, candidate_shape):
    feat = _compute_hog(patch)
    db = _get_template_db()

    best_cat = "unknown"
    best_score = -1.0
    for cat, tmpl_feat in db:
        meta = _CATEGORY_META.get(cat, {})
        if candidate_color and meta.get("color") != candidate_color:
            continue
        if candidate_shape and meta.get("shape") != candidate_shape:
            continue
        n_hog = min(len(feat), len(tmpl_feat))
        score = float(np.dot(feat[:n_hog], tmpl_feat[:n_hog]))
        if score > best_score:
            best_score = score
            best_cat = cat

    if best_score < score_threshold:
        best_cat = "unknown"

    if best_cat.startswith("speed_") and candidate_shape == _SIGN_SHAPE_CIRCLE:
        best_cat, best_score = _refine_speed_class(patch, best_cat, best_score)

    return best_cat, best_score


def _refine_speed_class(patch, hog_cat, hog_score):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    resized = cv2.resize(gray, (_TEMPLATE_SIZE, _TEMPLATE_SIZE))
    margin = _TEMPLATE_SIZE // 4
    roi = resized[margin:_TEMPLATE_SIZE - margin, margin:_TEMPLATE_SIZE - margin]
    roi_f = roi.astype(np.float64)
    roi_norm = (roi_f - roi_f.mean()) / (roi_f.std() + 1e-9)

    speed_cats = [c for c in SIGN_CATEGORIES if c.startswith("speed_")]
    best_ncc = -2.0
    best_cat = hog_cat
    for cat in speed_cats:
        tmpl = _generate_template(cat)
        tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        tmpl_resized = cv2.resize(tmpl_gray, (_TEMPLATE_SIZE, _TEMPLATE_SIZE))
        tmpl_roi = tmpl_resized[margin:_TEMPLATE_SIZE - margin, margin:_TEMPLATE_SIZE - margin]
        tmpl_f = tmpl_roi.astype(np.float64)
        tmpl_norm = (tmpl_f - tmpl_f.mean()) / (tmpl_f.std() + 1e-9)
        ncc = float(np.mean(roi_norm * tmpl_norm))
        if ncc > best_ncc:
            best_ncc = ncc
            best_cat = cat

    if best_ncc > 0.3:
        return best_cat, max(hog_score, best_ncc)
    return hog_cat, hog_score


# === Phase 4: Occlusion Check ===

def _check_occlusion(patch, threshold=0.5):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    valid_ratio = np.count_nonzero(gray > 20) / max(gray.size, 1)
    return valid_ratio < threshold


def _nms(results, iou_thresh=0.3):
    if len(results) == 0:
        return results
    results_sorted = sorted(results, key=lambda r: r["confidence"], reverse=True)
    kept = []
    for r in results_sorted:
        x1, y1, w1, h1 = r["bbox"]
        x1_end, y1_end = x1 + w1, y1 + h1
        area1 = w1 * h1
        suppress = False
        for k in kept:
            x2, y2, w2, h2 = k["bbox"]
            x2_end, y2_end = x2 + w2, y2 + h2
            area2 = w2 * h2
            inter_x1 = max(x1, x2)
            inter_y1 = max(y1, y2)
            inter_x2 = min(x1_end, x2_end)
            inter_y2 = min(y1_end, y2_end)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            iou = inter_area / max(area1 + area2 - inter_area, 1)
            if iou > iou_thresh:
                suppress = True
                break
        if not suppress:
            kept.append(r)
    return kept


# === Phase 5: Recognize ===

def recognize_signs(image, min_area=3000, score_threshold=0.35):
    candidates = _localize_signs(image, min_area)
    results = []

    for x, y, w, h, cnt, color, shape in candidates:
        x2 = min(x + w, image.shape[1])
        y2 = min(y + h, image.shape[0])
        patch = image[y:y2, x:x2]
        if patch.size == 0:
            continue

        patch_resized = cv2.resize(patch, (_TEMPLATE_SIZE, _TEMPLATE_SIZE))
        category, confidence = _classify_sign(patch_resized, score_threshold,
                                              color, shape)
        occluded = _check_occlusion(patch_resized)
        if occluded:
            confidence *= 0.8

        results.append({
            "bbox": (x, y, w, h),
            "category": category,
            "confidence": float(confidence),
            "occluded": occluded,
            "color": color,
            "shape": shape,
        })

    results = _nms(results)
    return results


# === Phase 6: Test ===

def generate_test_sign_image(img_w=640, img_h=480):
    img = np.full((img_h, img_w, 3), (60, 60, 60), dtype=np.uint8)

    sign_r = 55
    cx1, cy1 = 120, 120
    cv2.circle(img, (cx1, cy1), sign_r, (0, 0, 255), 6)
    cv2.circle(img, (cx1, cy1), sign_r - 6, (255, 255, 255), -1)
    tmpl_50 = _generate_template("speed_50")
    tmpl_50_gray = cv2.cvtColor(tmpl_50, cv2.COLOR_BGR2GRAY)
    inner_size = (sign_r - 6) * 2
    tmpl_50_inner = tmpl_50_gray[16:112, 16:112]
    tmpl_50_resized = cv2.resize(tmpl_50_inner, (inner_size, inner_size))
    mask_50 = tmpl_50_resized < 200
    y1, y2 = cy1 - sign_r + 6, cy1 + sign_r - 6
    x1, x2 = cx1 - sign_r + 6, cx1 + sign_r - 6
    roi_50 = img[y1:y2, x1:x2, 0]
    roi_50[mask_50] = tmpl_50_resized[mask_50]

    cx2, cy2 = 350, 150
    sign_r2 = 60
    cv2.circle(img, (cx2, cy2), sign_r2, (0, 0, 255), 6)
    cv2.circle(img, (cx2, cy2), sign_r2 - 6, (255, 255, 255), -1)
    tmpl_30 = _generate_template("speed_30")
    tmpl_30_gray = cv2.cvtColor(tmpl_30, cv2.COLOR_BGR2GRAY)
    inner_size2 = (sign_r2 - 6) * 2
    tmpl_30_inner = tmpl_30_gray[16:112, 16:112]
    tmpl_30_resized = cv2.resize(tmpl_30_inner, (inner_size2, inner_size2))
    mask_30 = tmpl_30_resized < 200
    y1, y2 = cy2 - sign_r2 + 6, cy2 + sign_r2 - 6
    x1, x2 = cx2 - sign_r2 + 6, cx2 + sign_r2 - 6
    roi_30 = img[y1:y2, x1:x2, 0]
    roi_30[mask_30] = tmpl_30_resized[mask_30]

    cx3, cy3 = 540, 140
    sign_r3 = 55
    cv2.circle(img, (cx3, cy3), sign_r3, (255, 100, 0), 8)
    cv2.circle(img, (cx3, cy3), sign_r3 - 8, (255, 255, 255), -1)
    tmpl_mf = _generate_template("mandatory_forward")
    tmpl_mf_gray = cv2.cvtColor(tmpl_mf, cv2.COLOR_BGR2GRAY)
    inner_size3 = (sign_r3 - 8) * 2
    tmpl_mf_inner = tmpl_mf_gray[20:108, 20:108]
    tmpl_mf_resized = cv2.resize(tmpl_mf_inner, (inner_size3, inner_size3))
    mask_mf = tmpl_mf_resized < 200
    y1, y2 = cy3 - sign_r3 + 8, cy3 + sign_r3 - 8
    x1, x2 = cx3 - sign_r3 + 8, cx3 + sign_r3 - 8
    roi_mf = img[y1:y2, x1:x2, 0]
    roi_mf[mask_mf] = tmpl_mf_resized[mask_mf]

    h_tri = 90
    cx_t, cy_t = 250, 370
    tri_half = h_tri // 2
    cv2.fillPoly(img, [np.array([
        [cx_t, cy_t - tri_half],
        [cx_t - tri_half, cy_t + tri_half],
        [cx_t + tri_half, cy_t + tri_half],
    ])], (0, 200, 255))
    margin_t = int(h_tri * 0.15)
    cv2.fillPoly(img, [np.array([
        [cx_t, cy_t - tri_half + margin_t * 2],
        [cx_t - tri_half + margin_t * 2, cy_t + tri_half - margin_t],
        [cx_t + tri_half - margin_t * 2, cy_t + tri_half - margin_t],
    ])], (255, 255, 255))
    pts_arrow = np.array([[cx_t + 12, cy_t - 6], [cx_t - 12, cy_t], [cx_t + 12, cy_t + 6]])
    cv2.fillPoly(img, [pts_arrow], (0, 0, 0))

    return img


if __name__ == "__main__":
    test_img = generate_test_sign_image()
    signs = recognize_signs(test_img)
    print(f"Detected {len(signs)} signs:")
    for s in signs:
        print(f"  bbox={s['bbox']}  cat={s['category']}  "
              f"conf={s['confidence']:.3f}  occ={s['occluded']}  "
              f"color={s['color']}  shape={s['shape']}")
