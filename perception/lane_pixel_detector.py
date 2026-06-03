import sys
import os
import numpy as np
import cv2

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# === Phase 1: Color Filter ===

def _lane_color_mask(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    mask_w = cv2.inRange(hls, np.array([0, 180, 0]), np.array([180, 255, 255]))
    mask_y = cv2.inRange(hls, np.array([10, 80, 100]), np.array([40, 255, 255]))
    return cv2.bitwise_or(mask_w, mask_y)


# === Phase 2: Row Scan ===

def _scan_row_peaks(row_profile, midpoint, margin):
    left_lo = max(0, midpoint - margin)
    left_hi = midpoint
    right_lo = midpoint
    right_hi = min(len(row_profile), midpoint + margin)

    left_region = row_profile[left_lo:left_hi]
    right_region = row_profile[right_lo:right_hi]

    left_x = left_lo + int(np.argmax(left_region)) if left_region.max() > 0 else np.nan
    right_x = right_lo + int(np.argmax(right_region)) if right_region.max() > 0 else np.nan

    return left_x, right_x


# === Phase 3: Jump Filter ===

def _filter_jumps(center_x, threshold):
    valid = ~np.isnan(center_x)
    if valid.sum() < 2:
        return center_x
    idx = np.where(valid)[0]
    last_valid_val = center_x[idx[0]]
    for i in range(1, len(idx)):
        if abs(center_x[idx[i]] - last_valid_val) > threshold:
            center_x[idx[i]] = np.nan
        else:
            last_valid_val = center_x[idx[i]]
    return center_x


# === Phase 4: Detect ===

def detect_lane_pixels(image, scan_rows=None, peak_margin=80, jump_threshold=100):
    img_h, img_w = image.shape[:2]
    if scan_rows is None:
        scan_rows = np.linspace(img_h // 2, img_h - 1, 20, dtype=int)

    mask = _lane_color_mask(image)
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)

    midpoint = img_w // 2
    n_rows = len(scan_rows)
    center_x = np.full(n_rows, np.nan, dtype=np.float64)

    profiles = blurred[scan_rows, :]

    left_lo = max(0, midpoint - peak_margin)
    left_hi = midpoint
    right_lo = midpoint
    right_hi = min(img_w, midpoint + peak_margin)

    left_regions = profiles[:, left_lo:left_hi]
    right_regions = profiles[:, right_lo:right_hi]

    left_max_vals = left_regions.max(axis=1)
    right_max_vals = right_regions.max(axis=1)
    left_argmax = left_regions.argmax(axis=1)
    right_argmax = right_regions.argmax(axis=1)

    left_x = left_lo + left_argmax
    right_x = right_lo + right_argmax

    left_valid = left_max_vals > 0
    right_valid = right_max_vals > 0

    both_valid = left_valid & right_valid
    center_x[both_valid] = (left_x[both_valid] + right_x[both_valid]) / 2.0

    left_only = left_valid & ~right_valid
    center_x[left_only] = left_x[left_only] + peak_margin / 2.0

    right_only = right_valid & ~left_valid
    center_x[right_only] = right_x[right_only] - peak_margin / 2.0

    center_x = _filter_jumps(center_x, jump_threshold)
    return scan_rows, center_x


# === Phase 5: Test ===

def generate_test_image(img_w=640, img_h=480):
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    img[:] = (60, 60, 60)

    left_pts = np.array([
        [int(img_w * 0.30), img_h],
        [int(img_w * 0.35), int(img_h * 0.7)],
        [int(img_w * 0.40), int(img_h * 0.5)],
        [int(img_w * 0.44), int(img_h * 0.35)],
    ], np.int32)
    right_pts = np.array([
        [int(img_w * 0.70), img_h],
        [int(img_w * 0.65), int(img_h * 0.7)],
        [int(img_w * 0.60), int(img_h * 0.5)],
        [int(img_w * 0.56), int(img_h * 0.35)],
    ], np.int32)

    cv2.polylines(img, [left_pts], False, (0, 255, 255), 4)
    cv2.polylines(img, [right_pts], False, (0, 255, 255), 4)

    for y in range(img_h // 2, img_h, 30):
        cx = int(img_w * 0.50)
        cv2.line(img, (cx - 2, y), (cx + 2, y), (255, 255, 255), 2)

    return img


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    SHOW_ANIMATION = True

    test_img = generate_test_image()
    rows, cx = detect_lane_pixels(test_img)

    print(f"scan_rows: {rows}")
    print(f"center_x:  {cx}")
    valid = ~np.isnan(cx)
    print(f"valid: {valid.sum()}/{len(cx)}")
    if valid.sum() > 0:
        print(f"center_x mean: {np.nanmean(cx):.1f} px (expected ~320)")
        print(f"center_x std:  {np.nanstd(cx):.1f} px")

    if SHOW_ANIMATION:
        mask = _lane_color_mask(test_img)

        plt.subplots(2, 2, figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.xlabel("x[px]")
        plt.ylabel("y[px]")

        plt.subplot(2, 2, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("HLS Color Mask")
        plt.xlabel("x[px]")
        plt.ylabel("y[px]")

        plt.subplot(2, 2, 3)
        vis = test_img.copy()
        for row, x in zip(rows, cx):
            if not np.isnan(x):
                cv2.circle(vis, (int(x), int(row)), 3, (0, 255, 0), -1)
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("Detected Center Line")
        plt.xlabel("x[px]")
        plt.ylabel("y[px]")

        plt.subplot(2, 2, 4)
        valid_mask = ~np.isnan(cx)
        if valid_mask.sum() > 0:
            plt.plot(cx[valid_mask], rows[valid_mask], ".r", label="center")
            plt.plot(cx[valid_mask] - 40, rows[valid_mask], "-b", label="left")
            plt.plot(cx[valid_mask] + 40, rows[valid_mask], "-g", label="right")
            plt.legend()
        plt.title("Lane Boundaries")
        plt.xlabel("x[px]")
        plt.ylabel("y[px]")
        plt.grid(True)

        plt.tight_layout()
        os.makedirs("figs", exist_ok=True)
        plt.savefig("figs/lane_pixel_detector.png", dpi=150)
        plt.show()
