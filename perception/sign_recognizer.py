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
    "no_entry", "no_parking", "unknown",
]

_TEMPLATE_SIZE = 64
_HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2))


def _generate_template(category, size=_TEMPLATE_SIZE):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    r = int(size * 0.42)

    if category.startswith("speed_"):
        cv2.circle(img, (cx, cy), r, (0, 0, 255), 3)
        cv2.circle(img, (cx, cy), r - 4, (255, 255, 255), -1)
        num = category.split("_")[1]
        font_scale = 1.2 if len(num) <= 2 else 0.9
        cv2.putText(img, num, (cx - 12, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
    elif category.startswith("arrow_"):
        cv2.circle(img, (cx, cy), r, (0, 0, 255), 3)
        cv2.circle(img, (cx, cy), r - 4, (255, 255, 255), -1)
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

    return img


def _compute_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (_TEMPLATE_SIZE, _TEMPLATE_SIZE))
    win_size = (_TEMPLATE_SIZE, _TEMPLATE_SIZE)
    hog = cv2.HOGDescriptor(win_size, (16, 16), (8, 8),
                            (8, 8), _HOG_PARAMS["orientations"])
    feat = hog.compute(resized)
    feat = feat.flatten()
    norm = np.linalg.norm(feat) + 1e-12
    return feat / norm


def _build_template_db():
    db = []
    angles = [0, 90, 180, 270]
    for cat in SIGN_CATEGORIES:
        if cat == "unknown":
            continue
        tmpl = _generate_template(cat)
        for angle in angles:
            M = cv2.getRotationMatrix2D((_TEMPLATE_SIZE / 2, _TEMPLATE_SIZE / 2),
                                        angle, 1.0)
            rotated = cv2.warpAffine(tmpl, M, (_TEMPLATE_SIZE, _TEMPLATE_SIZE))
            feat = _compute_hog(rotated)
            db.append((cat, feat))
    return db


@functools.lru_cache(maxsize=1)
def _get_template_db():
    return _build_template_db()


# === Phase 2: Sign Localization ===

def _localize_signs(image, min_area):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_r1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask_r1, mask_r2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        if aspect < 0.5 or aspect > 2.0:
            continue
        candidates.append((x, y, w, h, cnt))
    return candidates


# === Phase 3: Classify ===

def _classify_sign(patch, score_threshold):
    feat = _compute_hog(patch)
    db = _get_template_db()

    best_cat = "unknown"
    best_score = -1.0
    for cat, tmpl_feat in db:
        score = float(np.dot(feat, tmpl_feat))
        if score > best_score:
            best_score = score
            best_cat = cat

    if best_score < score_threshold:
        best_cat = "unknown"
    return best_cat, best_score


# === Phase 4: Occlusion Check ===

def _check_occlusion(patch, threshold=0.6):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    valid_ratio = np.count_nonzero(gray > 20) / max(gray.size, 1)
    return valid_ratio < threshold


# === Phase 5: Recognize ===

def recognize_signs(image, min_area=500, score_threshold=0.5):
    candidates = _localize_signs(image, min_area)
    results = []

    for x, y, w, h, cnt in candidates:
        x2 = min(x + w, image.shape[1])
        y2 = min(y + h, image.shape[0])
        patch = image[y:y2, x:x2]
        if patch.size == 0:
            continue

        patch_resized = cv2.resize(patch, (_TEMPLATE_SIZE, _TEMPLATE_SIZE))
        category, confidence = _classify_sign(patch_resized, score_threshold)
        occluded = _check_occlusion(patch_resized)
        if occluded:
            confidence *= 0.7

        results.append({
            "bbox": (x, y, w, h),
            "category": category,
            "confidence": float(confidence),
            "occluded": occluded,
        })

    return results


# === Phase 6: Test ===

def generate_test_sign_image(img_w=640, img_h=480):
    img = np.full((img_h, img_w, 3), (60, 60, 60), dtype=np.uint8)

    cv2.circle(img, (100, 100), 35, (0, 0, 255), 3)
    cv2.circle(img, (100, 100), 31, (255, 255, 255), -1)
    cv2.putText(img, "50", (82, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.circle(img, (300, 150), 40, (0, 0, 255), 3)
    cv2.circle(img, (300, 150), 36, (255, 255, 255), -1)
    cv2.putText(img, "30", (280, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return img


if __name__ == "__main__":
    test_img = generate_test_sign_image()
    signs = recognize_signs(test_img)
    print(f"Detected {len(signs)} signs:")
    for s in signs:
        print(f"  bbox={s['bbox']}  cat={s['category']}  "
              f"conf={s['confidence']:.3f}  occ={s['occluded']}")
