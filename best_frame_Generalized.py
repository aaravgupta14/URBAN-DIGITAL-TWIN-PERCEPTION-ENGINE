import argparse
import os

import cv2
import numpy as np

from road_seg_Generalized import segment_road
from scene_Generalized import load_calib, save_calib

parser = argparse.ArgumentParser()
parser.add_argument("--video", default="dataset/Weast (1).mp4")
parser.add_argument("--start", type=int, default=0, help="first frame to search (0-based)")
parser.add_argument("--end", type=int, default=None, help="last frame to search (0-based)")
parser.add_argument("--frame", type=int, default=None, help="use exactly this frame")
parser.add_argument("--no-show", action="store_true")
args = parser.parse_args()
if args.frame is not None:
    args.start = args.end = args.frame

video_path = args.video

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

best_score = -1
best_frame = None
best_mask = None
best_frame_number = 0

frame_number = args.start
FRAME_SKIP = 10

while True:

    ret, frame = cap.read()

    if args.end is not None and frame_number > args.end:
        break

    if not ret:
        break

    if (frame_number - args.start) % FRAME_SKIP != 0:
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

calib = load_calib()
calib.update({"video": os.path.abspath(video_path), "ref_frame": best_frame_number,
              "frame_range": [args.start, args.end]})
save_calib(calib)

if args.no_show:
    raise SystemExit

cv2.imshow("Best Calibration Frame", best_frame)
cv2.imshow("Road Mask", best_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
