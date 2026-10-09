import json
import os

import cv2
import numpy as np

CALIB_PATH = "calibration.json"
H_PATH = "homography.npy"
H_RAW_PATH = "homography_raw.npy"
STAB_PATH = "stabilization.npz"

UNITS_PER_METER = 120

PAD_LEFT = 16
PAD_RIGHT = 96
PAD_TOP = 80
PAD_BOTTOM = 16

DEFAULT_CALIB = {
    "fps": 30.0,
    "raw_width": 840,
    "raw_height": 306,
    "sx": 1.0,
    "sy": 1.0,
}


def load_calib():
    calib = dict(DEFAULT_CALIB)
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH, encoding="utf-8") as f:
            calib.update(json.load(f))
    return calib


def save_calib(calib):
    with open(CALIB_PATH, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)


def output_size(calib=None):
    calib = calib or load_calib()
    return (
        int(round(calib["raw_width"] * calib["sx"])),
        int(round(calib["raw_height"] * calib["sy"])),
    )


def load_raw_homography():
    if os.path.exists(H_RAW_PATH):
        return np.load(H_RAW_PATH)
    H = np.load(H_PATH)
    np.save(H_RAW_PATH, H)
    return H


def scaled_homography(H_raw, sx, sy):
    return np.diag([sx, sy, 1.0]) @ H_raw


def load_transforms(path=STAB_PATH):
    if not os.path.exists(path):
        return None
    data = np.load(path)
    ref_frame = load_calib().get("ref_frame")
    if "ref_frame" in data and ref_frame is not None and int(data["ref_frame"]) != int(ref_frame):
        print(f"{path} was built for calibration frame {int(data['ref_frame'])}, "
              f"current is {ref_frame}; ignoring it")
        return None
    return data["T"]


def _apply(T, frame_no, pts, inverse):
    pts = np.float32(pts).reshape(-1, 1, 2)
    if T is None:
        return pts.reshape(-1, 2)
    if frame_no < 0 or frame_no >= len(T) or np.isnan(T[frame_no]).any():
        return None
    M = np.linalg.inv(T[frame_no]) if inverse else T[frame_no]
    return cv2.perspectiveTransform(pts, M).reshape(-1, 2)


def to_reference(T, frame_no, pts):
    return _apply(T, frame_no, pts, inverse=False)


def from_reference(T, frame_no, pts):
    return _apply(T, frame_no, pts, inverse=True)


def in_bounds(X, Y, width, height):
    return -PAD_LEFT <= X < width + PAD_RIGHT and -PAD_TOP <= Y < height + PAD_BOTTOM
