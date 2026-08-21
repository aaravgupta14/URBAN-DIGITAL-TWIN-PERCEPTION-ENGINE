import cv2
import numpy as np

from road_seg import segment_road
from road_boundary import extract_road_boundaries
from homography import HomographyEstimator

video_path = r"C:\Users\Aarav Gupta\OneDrive\Desktop\DIGITAL_TWIN\dataset\Weast (1).mp4"

cap = cv2.VideoCapture(video_path)

estimator = HomographyEstimator()

H = None

while True:

    ret, frame = cap.read()

    if not ret:
        break

    output = frame.copy()

    # -------------------------------------------------
    # Road Segmentation
    # -------------------------------------------------

    road_mask, hull, contour = segment_road(frame)

    # -------------------------------------------------
    # Boundary Extraction
    # -------------------------------------------------

    left_pts, right_pts = extract_road_boundaries(road_mask)

    # -------------------------------------------------
    # Draw Boundaries
    # -------------------------------------------------

    if len(left_pts) > 1:

        cv2.polylines(
            output,
            [np.array(left_pts, dtype=np.int32)],
            False,
            (0,255,0),
            3
        )

    if len(right_pts) > 1:

        cv2.polylines(
            output,
            [np.array(right_pts, dtype=np.int32)],
            False,
            (0,0,255),
            3
        )

    # -------------------------------------------------
    # Draw Boundary Points
    # -------------------------------------------------

    for p in left_pts:

        cv2.circle(
            output,
            (int(p[0]), int(p[1])),
            2,
            (0,255,255),
            -1
        )

    for p in right_pts:

        cv2.circle(
            output,
            (int(p[0]), int(p[1])),
            2,
            (255,255,0),
            -1
        )

    # -------------------------------------------------
    # Fit Curves
    # -------------------------------------------------

    left_curve = estimator.fit_boundary(left_pts)
    right_curve = estimator.fit_boundary(right_pts)

    if left_curve is not None and right_curve is not None:

        image_height = frame.shape[0]

        top_y = int(image_height * 0.35)
        bottom_y = int(image_height * 0.95)

        # -------- Polynomial Curves --------

        for y in range(top_y, bottom_y, 5):

            lx = int(np.polyval(left_curve, y))
            rx = int(np.polyval(right_curve, y))

            cv2.circle(output, (lx, y), 2, (255,0,255), -1)
            cv2.circle(output, (rx, y), 2, (255,0,255), -1)

        # -------- Four Homography Points --------

        tl = (int(np.polyval(left_curve, top_y)), top_y)
        tr = (int(np.polyval(right_curve, top_y)), top_y)

        bl = (int(np.polyval(left_curve, bottom_y)), bottom_y)
        br = (int(np.polyval(right_curve, bottom_y)), bottom_y)

        # Draw Points

        cv2.circle(output, tl, 10, (0,0,255), -1)
        cv2.circle(output, tr, 10, (0,0,255), -1)

        cv2.circle(output, bl, 10, (255,0,0), -1)
        cv2.circle(output, br, 10, (255,0,0), -1)

        # Draw Quadrilateral

        cv2.line(output, tl, tr, (0,255,255), 3)
        cv2.line(output, tr, br, (0,255,255), 3)
        cv2.line(output, br, bl, (0,255,255), 3)
        cv2.line(output, bl, tl, (0,255,255), 3)

    # -------------------------------------------------
    # Show Windows
    # -------------------------------------------------

    cv2.imshow("Road Mask", road_mask)
    cv2.imshow("Road Geometry Debug", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()