import cv2
import numpy as np

from road_seg import segment_road
from road_boundary import extract_road_boundaries

video_path = r"C:\Users\Aarav Gupta\OneDrive\Desktop\DIGITAL_TWIN\dataset\Weast (1).mp4"

cap = cv2.VideoCapture(video_path)

while True:

    ret, frame = cap.read()

    if not ret:
        break
    road_mask, hull, contour = segment_road(frame)
    left_pts, right_pts = extract_road_boundaries(road_mask)

    output = frame.copy()
    if len(left_pts) > 1:
        cv2.polylines(
            output,
            [np.array(left_pts, dtype=np.int32)],
            False,
            (0, 0, 255),
            3
        )

    # Draw right boundary
    if len(right_pts) > 1:
        cv2.polylines(
            output,
            [np.array(right_pts, dtype=np.int32)],
            False,
            (255, 0, 0),
            3
        )
    for x, y in left_pts:
        cv2.circle(output, (x, y), 2, (0, 255, 255), -1)

    for x, y in right_pts:
        cv2.circle(output, (x, y), 2, (0, 255, 255), -1)

    cv2.imshow("Road Mask", road_mask)
    cv2.imshow("Road Boundaries", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
