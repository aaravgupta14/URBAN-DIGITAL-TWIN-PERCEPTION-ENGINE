import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

from scene_Generalized import (
    load_calib, save_calib, load_transforms, to_reference, H_PATH, H_RAW_PATH, UNITS_PER_METER,
)

LOG_PATH = "tracking_log.csv"
POINTS_PATH = "scale_points.json"
REF_PATH = "calibration_frame.jpg"

CLASS_HEIGHT_M = {2: 1.55, 3: 1.65}
MIN_CONF = 0.5
EDGE_MARGIN_PX = 5
HORIZON_MARGIN_PX = 25
MAX_DEPTH_M = 30.0
LATERAL_HALF_M = 10.0
DEFAULT_HFOV_DEG = 75.0
CAMERA_HEIGHT_RANGE_M = (1.0, 2.5)
HFOV_RANGE_DEG = (50.0, 130.0)


def robust_line(x, y, iters=3):
    keep = np.ones(len(x), bool)
    for _ in range(iters):
        A = np.c_[x[keep], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        resid = y - (coef[0] * x + coef[1])
        keep = np.abs(resid) < 2 * resid[keep].std()
    return coef, keep


def estimate_horizon_and_height(df, img_w, img_h):
    d = df[df["class_id"].isin(list(CLASS_HEIGHT_M)) & (df["conf"] >= MIN_CONF)
           & (df["x1"] > EDGE_MARGIN_PX) & (df["x2"] < img_w - EDGE_MARGIN_PX)
           & (df["y1"] > EDGE_MARGIN_PX) & (df["y2"] < img_h - EDGE_MARGIN_PX)]
    if len(d) < 30:
        raise SystemExit(f"only {len(d)} usable boxes; need more detections to find the horizon")
    v = d["gy"].to_numpy()
    norm_h = (d["y2"] - d["y1"]).to_numpy() / d["class_id"].map(CLASS_HEIGHT_M).to_numpy()
    (slope, icept), keep = robust_line(v, norm_h)
    v_h = -icept / slope
    cam_h = 1.0 / slope
    width = (d["x2"] - d["x1"]).to_numpy()
    (ws, wi), _ = robust_line(v, width)
    return v_h, cam_h, -wi / ws, int(keep.sum())


def model_matrix(cam_h, f, v_h, cx):
    return np.array([
        [cam_h, 0.0, -cam_h * cx],
        [0.0, 0.0, f * cam_h],
        [0.0, 1.0, -v_h],
    ])


def ground_xz(M, pts):
    return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), M).reshape(-1, 2)


def fit_focal(pairs, cam_h, v_h, cx):
    M1 = model_matrix(cam_h, 1.0, v_h, cx)
    num = den = 0.0
    for p in pairs:
        a, b = ground_xz(M1, [p["p1"], p["p2"]])
        dx, dz1 = b[0] - a[0], b[1] - a[1]
        L2 = p["distance_m"] ** 2
        w = 1.0 / L2
        num += w * dz1 ** 2 * (L2 - dx ** 2)
        den += w * dz1 ** 4
    return float(np.sqrt(max(num / den, 1e-9)))


def load_fit_pairs():
    if not os.path.exists(POINTS_PATH):
        return []
    with open(POINTS_PATH, encoding="utf-8") as f:
        pairs = [p for p in json.load(f) if p["kind"] in ("fit", "check")]
    T = load_transforms()
    out = []
    for p in pairs:
        ref = to_reference(T, p.get("frame", 0), [p["p1"], p["p2"]])
        if ref is not None:
            out.append({**p, "p1": ref[0].tolist(), "p2": ref[1].tolist()})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hfov", type=float, default=None, help="force horizontal field of view (deg)")
    parser.add_argument("--focal", type=float, default=None, help="force focal length in pixels (same camera elsewhere)")
    parser.add_argument("--camera-height", type=float, default=None,
                        help="known mounting height in metres; with --pitch and --focal/--hfov skips estimation")
    parser.add_argument("--pitch", type=float, default=None, help="known downward tilt of the camera in degrees")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("W", "H"))
    parser.add_argument("--max-depth", type=float, default=MAX_DEPTH_M)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    if args.image_size:
        img_w, img_h = args.image_size
    else:
        ref = cv2.imread(REF_PATH)
        if ref is None:
            raise SystemExit(f"{REF_PATH} not found; pass --image-size W H")
        img_h, img_w = ref.shape[:2]
    cx, cy = img_w / 2.0, img_h / 2.0

    from_specs = args.camera_height is not None and args.pitch is not None
    print("=== Ground-plane camera model ===")
    if from_specs:
        if args.focal is None and args.hfov is None:
            raise SystemExit("with --camera-height and --pitch also give --focal or --hfov")
        f_spec = args.focal or cx / np.tan(np.radians(args.hfov) / 2)
        cam_h = args.camera_height
        v_h = cy - f_spec * np.tan(np.radians(args.pitch))
        print(f"from camera specs            : height {cam_h:.2f} m, tilt {args.pitch:.1f} deg, "
              f"focal {f_spec:.0f} px -> horizon row {v_h:.1f}")
    else:
        df = pd.read_csv(LOG_PATH)
        v_h, cam_h, v_h_width, n = estimate_horizon_and_height(df, img_w, img_h)
        print(f"horizon row from box heights : {v_h:.1f}  ({n} boxes)")
        print(f"horizon row from box widths  : {v_h_width:.1f}  (independent check)")
        print(f"camera height                : {cam_h:.2f} m  (assumes car {CLASS_HEIGHT_M[2]} m, "
              f"motorcycle+rider {CLASS_HEIGHT_M[3]} m)")
        if not CAMERA_HEIGHT_RANGE_M[0] <= cam_h <= CAMERA_HEIGHT_RANGE_M[1]:
            print("WARNING: camera height is outside the plausible handheld range")

    all_pairs = load_fit_pairs()
    fit = [p for p in all_pairs if p["kind"] == "fit"]
    check = [p for p in all_pairs if p["kind"] == "check"]
    if from_specs:
        f = f_spec
        hfov = 2 * np.degrees(np.arctan(cx / f))
    elif args.focal is not None:
        f = args.focal
        hfov = 2 * np.degrees(np.arctan(cx / f))
        print(f"focal length                 : {f:.0f} px fixed from another calibration (HFOV {hfov:.0f} deg)")
    elif args.hfov is not None or not fit:
        hfov = args.hfov or DEFAULT_HFOV_DEG
        f = cx / np.tan(np.radians(hfov) / 2)
        print(f"focal length                 : {f:.0f} px from assumed HFOV {hfov:.0f} deg")
    else:
        f = fit_focal(fit, cam_h, v_h, cx)
        hfov = 2 * np.degrees(np.arctan(cx / f))
        print(f"focal length                 : {f:.0f} px from {len(fit)} ruler pair(s) "
              f"(HFOV {hfov:.0f} deg)")
        if not HFOV_RANGE_DEG[0] <= hfov <= HFOV_RANGE_DEG[1]:
            print("WARNING: implied field of view is implausible; check the ruler clicks")

    M = model_matrix(cam_h, f, v_h, cx)
    for label, ps in (("fit", fit), ("check", check)):
        for p in ps:
            a, b = ground_xz(M, [p["p1"], p["p2"]])
            meas = float(np.hypot(*(b - a)))
            print(f"{label:5s} pair: known {p['distance_m']:.2f} m  model {meas:.2f} m  "
                  f"err {100 * (meas - p['distance_m']) / p['distance_m']:+.1f}%")

    z_near = float(ground_xz(M, [[cx, img_h - 1]])[0, 1])
    z_far = float(min(args.max_depth, ground_xz(M, [[cx, v_h + HORIZON_MARGIN_PX]])[0, 1]))
    print(f"ground covered               : {z_near:.1f} m to {z_far:.1f} m ahead, "
          f"+-{LATERAL_HALF_M:.0f} m sideways")

    U = UNITS_PER_METER
    C = np.array([[U, 0, LATERAL_HALF_M * U], [0, -U, z_far * U], [0, 0, 1]])
    H = C @ M
    width = int(round(2 * LATERAL_HALF_M * U))
    height = int(round((z_far - z_near) * U))

    if not args.write:
        print("Dry run; pass --write to save the homography and reproject the log.")
        return

    np.save(H_PATH, H)
    np.save(H_RAW_PATH, H)
    calib = load_calib()
    calib.update({
        "raw_width": width, "raw_height": height, "sx": 1.0, "sy": 1.0,
        "method": "ground_model", "horizon_row": round(v_h, 1), "camera_height_m": round(cam_h, 3),
        "focal_px": round(f, 1), "hfov_deg": round(hfov, 1), "depth_range_m": [round(z_near, 2), round(z_far, 2)],
    })
    save_calib(calib)
    print(f"Saved {H_PATH}, {H_RAW_PATH} and calibration.json (twin {width} x {height} units)")

    from depth_calib_Generalized import reproject_log
    reproject_log(H, H, width, height)


if __name__ == "__main__":
    main()
