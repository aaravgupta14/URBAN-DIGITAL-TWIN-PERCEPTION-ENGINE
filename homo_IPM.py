import cv2
import numpy as np

from road_seg import segment_road

IMAGE_PATH = r"C:\Users\Aarav Gupta\OneDrive\Desktop\DIGITAL_TWIN\calibration_frame.jpg"

# How far down the frame (as a fraction of height) the top pair should sit.
# Kept away from the horizon/vanishing-point zone, where a few pixels of
# detection noise translate into a large error after the perspective warp.
TOP_ROW_TARGET_FRAC = 0.42
TOP_ROW_SEARCH_RANGE = 60   # rows to search above/below the target if it's blocked
MIN_ROAD_WIDTH_PX = 80      # minimum clean mask width to accept a row
CORNER_INSET_PX = 8         # pull points in slightly from the raw mask edge

# Standard-assumption road width, used to anchor real-world scale. Not a
# measured value -- correct this if you ever get an actual measurement of
# this road/lane. Chosen as a round mid-range figure for an urban arterial
# carriageway + shoulder (~3.5m/lane plus a parking/shoulder strip).
STANDARD_ROAD_WIDTH_M = 7.0
PIXELS_PER_METER = 120  # canvas resolution; tune for detail vs. size

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
    """Search outward from target_y for a row with a clean, wide road span."""
    for dy in range(search_range + 1):
        for y in {target_y - dy, target_y + dy}:
            if 0 <= y < mask.shape[0]:
                bounds = row_bounds(mask, y)
                if bounds and (bounds[1] - bounds[0]) >= min_width:
                    return y, bounds
    return None, None


def fit_edge_line(mask, y_start, y_end, side):
    """Robust line fit x = m*y + c to one road edge across many rows, so a
    single noisy/obstructed row (a parked vehicle bulging the mask locally)
    can't throw off a corner the way sampling just one row can. This is what
    actually keeps the two road edges honestly parallel/straight in the
    warped output, since the corners get placed ON the fitted true edge
    line rather than on a possibly-noisy single-row mask boundary."""
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


# Bottom row: scan up from the very bottom of the frame for the first
# row with a wide, clean stretch of road (as low/near as possible).
bottom_y, bottom_bounds = None, None
for y in range(img_h - 1, int(img_h * 0.6), -1):
    bounds = row_bounds(road_mask, y)
    if bounds and (bounds[1] - bounds[0]) >= 200:
        bottom_y, bottom_bounds = y, bounds
        break

# Top row: target row out of the noisy near-horizon zone.
top_y, top_bounds = find_good_row(
    road_mask, int(img_h * TOP_ROW_TARGET_FRAC), TOP_ROW_SEARCH_RANGE, MIN_ROAD_WIDTH_PX
)

if bottom_y is None or top_y is None:
    print("Could not find a usable road span in the calibration image.")
    exit()

# Fit the true left/right road-edge lines across the whole span between the
# two rows, then place all 4 corners ON those fitted lines -- not on raw
# per-row mask edges -- so the quad's sides track the road's real edges
# instead of picking up local segmentation noise at just two sample rows.
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
    (int(tl_x + CORNER_INSET_PX), top_y),      # 1: top-left
    (int(tr_x - CORNER_INSET_PX), top_y),      # 2: top-right
    (int(br_x - CORNER_INSET_PX), bottom_y),   # 3: bottom-right
    (int(bl_x + CORNER_INSET_PX), bottom_y),   # 4: bottom-left
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
cv2.imshow("Calibration (auto)", display)

src = np.float32(points)

# Anchor scale to the standard road-width assumption, and -- importantly --
# use the SAME pixels-per-meter for both axes. Only the width axis has an
# actual real-world reference (the assumed road width); the depth axis has
# no independent measurement, so instead of inventing a false "verified"
# number for it, its height is estimated by preserving the aspect ratio the
# raw pixel geometry already implied, just rescaled onto the same, single,
# consistent pixels-per-meter as the width. Using two different, unrelated
# scales for X and Y (the old behavior) rotates/skews any diagonal
# real-world direction on the canvas -- this keeps both axes isotropic.
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

np.save("homography.npy", H)

bird = cv2.warpPerspective(
    image,
    H,
    (OUTPUT_WIDTH, OUTPUT_HEIGHT),
    flags=cv2.INTER_LANCZOS4,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0)
)

cv2.imshow("Bird Eye", bird)

print("\nHomography saved as homography.npy")
print(f"Output Size : {OUTPUT_WIDTH} x {OUTPUT_HEIGHT}")
print(f"Scale: {PIXELS_PER_METER} px/m (assumed road width {STANDARD_ROAD_WIDTH_M}m; "
      f"width axis is anchored to this, depth axis uses the same px/m but is an estimate)")

cv2.waitKey(0)
cv2.destroyAllWindows()
