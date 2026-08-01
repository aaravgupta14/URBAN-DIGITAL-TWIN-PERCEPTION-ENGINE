import cv2
import numpy as np

from road_seg import segment_road

video_path = r"C:\Users\Aarav Gupta\OneDrive\Desktop\DIGITAL_TWIN\dataset\Weast (1).mp4"

cap = cv2.VideoCapture(video_path)

best_score = -1
best_frame = None
best_mask = None
best_frame_number = 0

frame_number = 0
FRAME_SKIP = 10

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if frame_number % FRAME_SKIP != 0:
        frame_number += 1
        continue

    road_mask, hull, contour = segment_road(frame)

    road_area = cv2.countNonZero(road_mask)

    if road_area > best_score:

        best_score = road_area
        best_frame = frame.copy()
        best_mask = road_mask.copy()
        best_frame_number = frame_number

    frame_number += 1

cap.release()

print("----------------------------------")
print("Best Frame :", best_frame_number)
print("Road Area  :", best_score)
print("----------------------------------")

cv2.imwrite("calibration_frame.jpg", best_frame)

cv2.imshow("Best Calibration Frame", best_frame)
cv2.imshow("Road Mask", best_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()