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
CANVAS_SCALE = 2.0          # uniform upscale of the output canvas resolution

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


# Bottom pair: scan up from the very bottom of the frame for the first
# row with a wide, clean stretch of road (as low/near as possible).
bottom_y, bottom_bounds = None, None
for y in range(img_h - 1, int(img_h * 0.6), -1):
    bounds = row_bounds(road_mask, y)
    if bounds and (bounds[1] - bounds[0]) >= 200:
        bottom_y, bottom_bounds = y, bounds
        break

# Top pair: target row out of the noisy near-horizon zone.
top_y, top_bounds = find_good_row(
    road_mask, int(img_h * TOP_ROW_TARGET_FRAC), TOP_ROW_SEARCH_RANGE, MIN_ROAD_WIDTH_PX
)

if bottom_y is None or top_y is None:
    print("Could not find a usable road span in the calibration image.")
    exit()

points = [
    (top_bounds[0] + CORNER_INSET_PX, top_y),      # 1: top-left
    (top_bounds[1] - CORNER_INSET_PX, top_y),      # 2: top-right
    (bottom_bounds[1] - CORNER_INSET_PX, bottom_y),  # 3: bottom-right
    (bottom_bounds[0] + CORNER_INSET_PX, bottom_y),  # 4: bottom-left
]

print("Auto-selected calibration points (from road mask):")
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

width_top = np.linalg.norm(src[1] - src[0])
width_bottom = np.linalg.norm(src[2] - src[3])

height_left = np.linalg.norm(src[3] - src[0])
height_right = np.linalg.norm(src[2] - src[1])

OUTPUT_WIDTH = int(max(width_top, width_bottom) * CANVAS_SCALE)
OUTPUT_HEIGHT = int(max(height_left, height_right) * CANVAS_SCALE)

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

cv2.waitKey(0)
cv2.destroyAllWindows()
