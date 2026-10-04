import sys

import cv2
import numpy as np

from road_seg_Generalized import segment_road
from scene_Generalized import load_calib, save_calib, H_PATH, H_RAW_PATH

IMAGE_PATH = "calibration_frame.jpg"
SHOW = "--no-show" not in sys.argv

TOP_ROW_TARGET_FRAC = 0.42
TOP_ROW_SEARCH_RANGE = 60
MIN_ROAD_WIDTH_PX = 80
CORNER_INSET_PX = 8

STANDARD_ROAD_WIDTH_M = 7.0
PIXELS_PER_METER = 120

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not found!")
    exit()

img_h, img_w = image.shape[:2]

road_mask, hull, contour = segment_road(image)

def row_bounds(mask, y):
    row = mask[y, :]
    xs = np.where(row > 0)[0]
    if len(xs) == 0:
        return None
    return int(xs.min()), int(xs.max())

def find_good_row(mask, target_y, search_range, min_width):
    for dy in range(search_range + 1):
        for y in {target_y - dy, target_y + dy}:
            if 0 <= y < mask.shape[0]:
                bounds = row_bounds(mask, y)
                if bounds and (bounds[1] - bounds[0]) >= min_width:
                    return y, bounds
    return None, None

def fit_edge_line(mask, y_start, y_end, side):
    ys, xs = [], []
    for y in range(y_start, y_end + 1):
        b = row_bounds(mask, y)
        if b is None:
            continue
        ys.append(y)
        xs.append(b[0] if side == "left" else b[1])
    if len(ys) < 10:
        return None
    ys_arr = np.array(ys, dtype=np.float64)
    xs_arr = np.array(xs, dtype=np.float64)
    m, c = np.polyfit(ys_arr, xs_arr, 1)
    resid = xs_arr - (m * ys_arr + c)
    std = resid.std()
    if std > 0:
        keep = np.abs(resid) < 2 * std
        if keep.sum() >= 10:
            m, c = np.polyfit(ys_arr[keep], xs_arr[keep], 1)
    return m, c

bottom_y, bottom_bounds = None, None
for y in range(img_h - 1, int(img_h * 0.6), -1):
    bounds = row_bounds(road_mask, y)
    if bounds and (bounds[1] - bounds[0]) >= 200:
        bottom_y, bottom_bounds = y, bounds
        break

top_y, top_bounds = find_good_row(
    road_mask, int(img_h * TOP_ROW_TARGET_FRAC), TOP_ROW_SEARCH_RANGE, MIN_ROAD_WIDTH_PX
)

if bottom_y is None or top_y is None:
    print("Could not find a usable road span in the calibration image.")
    exit()

left_line = fit_edge_line(road_mask, top_y, bottom_y, "left")
right_line = fit_edge_line(road_mask, top_y, bottom_y, "right")

if left_line is None or right_line is None:
    print("Could not fit road edge lines; falling back to raw row bounds.")
    tl_x, tr_x = top_bounds
    bl_x, br_x = bottom_bounds
else:
    lm, lc = left_line
    rm, rc = right_line
    tl_x, bl_x = lm * top_y + lc, lm * bottom_y + lc
    tr_x, br_x = rm * top_y + rc, rm * bottom_y + rc

points = [
    (int(tl_x + CORNER_INSET_PX), top_y),
    (int(tr_x - CORNER_INSET_PX), top_y),
    (int(br_x - CORNER_INSET_PX), bottom_y),
    (int(bl_x + CORNER_INSET_PX), bottom_y),
]

print("Auto-selected calibration points (fitted road-edge lines):")
for i, p in enumerate(points, start=1):
    print(f"  {i}: {p}")

display = image.copy()
for i, p in enumerate(points):
    cv2.circle(display, p, 6, (0, 0, 255), -1)
    cv2.putText(display, str(i + 1), (p[0] + 10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
cv2.line(display, points[0], points[1], (0, 255, 255), 2)
cv2.line(display, points[1], points[2], (0, 255, 255), 2)
cv2.line(display, points[2], points[3], (0, 255, 255), 2)
cv2.line(display, points[3], points[0], (0, 255, 255), 2)
if SHOW:
    cv2.imshow("Calibration (auto)", display)

src = np.float32(points)

OUTPUT_WIDTH = int(PIXELS_PER_METER * STANDARD_ROAD_WIDTH_M)

width_top = np.linalg.norm(src[1] - src[0])
width_bottom = np.linalg.norm(src[2] - src[3])
height_left = np.linalg.norm(src[3] - src[0])
height_right = np.linalg.norm(src[2] - src[1])
aspect_ratio = max(height_left, height_right) / max(width_top, width_bottom)
OUTPUT_HEIGHT = int(OUTPUT_WIDTH * aspect_ratio)

dst = np.float32([
    [0, 0],
    [OUTPUT_WIDTH, 0],
    [OUTPUT_WIDTH, OUTPUT_HEIGHT],
    [0, OUTPUT_HEIGHT]
])

H = cv2.getPerspectiveTransform(src, dst)

np.save(H_PATH, H)
np.save(H_RAW_PATH, H)

calib = load_calib()
calib.update({"raw_width": OUTPUT_WIDTH, "raw_height": OUTPUT_HEIGHT, "sx": 1.0, "sy": 1.0})
save_calib(calib)

bird = cv2.warpPerspective(
    image,
    H,
    (OUTPUT_WIDTH, OUTPUT_HEIGHT),
    flags=cv2.INTER_LANCZOS4,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0)
)

if SHOW:
    cv2.imshow("Bird Eye", bird)

print(f"\nHomography saved as {H_PATH} and {H_RAW_PATH}; depth scale reset, rerun depth_calib_Generalized.py --solve")
print(f"Output Size : {OUTPUT_WIDTH} x {OUTPUT_HEIGHT}")
print(f"Scale: {PIXELS_PER_METER} px/m (assumed road width {STANDARD_ROAD_WIDTH_M}m; "
      f"width axis is anchored to this, depth axis uses the same px/m but is an estimate)")

if SHOW:
    cv2.waitKey(0)
    cv2.destroyAllWindows()
