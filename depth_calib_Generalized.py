import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

from scene_Generalized import (
    load_calib, save_calib, load_raw_homography, scaled_homography, in_bounds,
    load_transforms, to_reference, from_reference,
    H_PATH, UNITS_PER_METER,
)

VIDEO_PATH = "dataset/Weast (1).mp4"
POINTS_PATH = "scale_points.json"
LOG_PATH = "tracking_log.csv"

VEHICLE_PRIORS_M = {
    "hatchback": (3.7, 4.0),
    "erickshaw": (2.6, 3.0),
    "motorcycle": (1.9, 2.2),
    "bus": (11.5, 12.5),
}

MIN_AXIS_SHARE = 0.15
TREND_WARN_PCT = 10.0

KIND_COLORS = {"fit": (0, 255, 0), "check": (0, 255, 255), "vehicle": (255, 0, 255)}


def project(H, pts):
    arr = np.float32(pts).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, H).reshape(-1, 2)


def stabilize_pairs(pairs):
    T = load_transforms()
    out = []
    for p in pairs:
        ref = to_reference(T, p.get("frame", 0), [p["p1"], p["p2"]])
        if ref is None:
            print(f"skipping pair on frame {p.get('frame')}: no stabilization transform")
            continue
        out.append({**p, "p1": ref[0].tolist(), "p2": ref[1].tolist()})
    if T is None:
        print("No stabilization.npz; treating all clicks as calibration-frame pixels")
    return out


def pair_delta(H, pair):
    a, b = project(H, [pair["p1"], pair["p2"]])
    return b - a


def known_length(pair):
    if pair["kind"] == "vehicle":
        lo, hi = VEHICLE_PRIORS_M[pair["class"]]
        return (lo + hi) / 2
    return pair["distance_m"]


def fit_scales(deltas, lengths_m, depth_only=False):
    A = deltas ** 2
    b = (lengths_m * UNITS_PER_METER) ** 2
    w = 1.0 / b
    share_x = float(np.mean(A[:, 0] / A.sum(axis=1)))

    if depth_only:
        share_x = 0.0
    elif len(b) >= 2 and MIN_AXIS_SHARE <= share_x <= 1 - MIN_AXIS_SHARE:
        sol, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
        if (sol > 0).all():
            return float(np.sqrt(sol[0])), float(np.sqrt(sol[1])), "width+depth"

    if share_x < 0.5:
        sy2 = (w * A[:, 1] * (b - A[:, 0])).sum() / (w * A[:, 1] ** 2).sum()
        return 1.0, float(np.sqrt(max(sy2, 1e-9))), "depth only (width kept at assumed road width)"
    sx2 = (w * A[:, 0] * (b - A[:, 1])).sum() / (w * A[:, 0] ** 2).sum()
    return float(np.sqrt(max(sx2, 1e-9))), 1.0, "width only (depth unchanged)"


def measured_m(delta, sx, sy):
    return float(np.hypot(sx * delta[0], sy * delta[1]) / UNITS_PER_METER)


def report(title, pairs, deltas, sx, sy):
    if not pairs:
        return []
    print(f"--- {title} ---")
    print(f"{'#':>3} {'kind':8s} {'img_y':>6} {'known_m':>8} {'meas_m':>8} {'err_%':>7}")
    errs = []
    for i, (p, d) in enumerate(zip(pairs, deltas)):
        known = known_length(p)
        meas = measured_m(d, sx, sy)
        err = 100.0 * (meas - known) / known
        img_y = (p["p1"][1] + p["p2"][1]) / 2
        label = p["kind"] if p["kind"] != "vehicle" else p["class"]
        extra = ""
        if p["kind"] == "vehicle":
            lo, hi = VEHICLE_PRIORS_M[p["class"]]
            extra = "  in range" if lo <= meas <= hi else f"  OUT of {lo}-{hi} m"
        print(f"{i:3d} {label:8s} {img_y:6.0f} {known:8.2f} {meas:8.2f} {err:+7.1f}{extra}")
        errs.append((img_y, err))
    abs_errs = [abs(e) for _, e in errs]
    print(f"mean |err| = {np.mean(abs_errs):.1f}%   max |err| = {np.max(abs_errs):.1f}%   "
          f"bias = {np.mean([e for _, e in errs]):+.1f}%")
    print()
    return errs


def depth_trend(errs):
    if len(errs) < 3:
        return
    ys = np.array([y for y, _ in errs])
    es = np.array([e for _, e in errs])
    if np.ptp(ys) < 1:
        return
    slope = np.polyfit(ys, es, 1)[0]
    swing = slope * np.ptp(ys)
    print(f"Error trend from far to near rows: {swing:+.1f} percentage points")
    if abs(swing) > TREND_WARN_PCT:
        print("WARNING: error changes strongly with depth. A single scale cannot fix this;")
        print("the road is probably not flat or the fitted edges are not parallel in reality.")
    print()


def reproject_log(H, H_raw, width, height):
    if not os.path.exists(LOG_PATH):
        print(f"{LOG_PATH} not found; skipping log reprojection")
        return
    df = pd.read_csv(LOG_PATH)
    if "gx" not in df.columns:
        image_pts = project(np.linalg.inv(H_raw), df[["X", "Y"]].to_numpy())
        pos = df.columns.get_loc("X")
        df.insert(pos, "gy", image_pts[:, 1])
        df.insert(pos, "gx", image_pts[:, 0])
        print("Log had no gx/gy columns; recovered them through the raw homography")
    world = project(H, df[["gx", "gy"]].to_numpy())
    df["X"] = world[:, 0]
    df["Y"] = world[:, 1]
    df["in_bounds"] = [in_bounds(x, y, width, height) for x, y in world]
    if "stab_ok" in df.columns:
        df["in_bounds"] &= df["stab_ok"].astype(str).str.lower() == "true"
    df.to_csv(LOG_PATH, index=False)
    print(f"Reprojected {len(df)} rows in {LOG_PATH} "
          f"({int(df['in_bounds'].sum())} in bounds)")


def solve(pairs, write, depth_only=False):
    calib = load_calib()
    H_raw = load_raw_homography()
    pairs = stabilize_pairs(pairs)

    fit = [p for p in pairs if p["kind"] == "fit"]
    check = [p for p in pairs if p["kind"] == "check"]
    vehicles = [p for p in pairs if p["kind"] == "vehicle"]

    used = fit
    if not fit:
        if not vehicles:
            print("Need at least one 'fit' or 'vehicle' pair to solve.")
            return
        print("No marking pairs; fitting scale to vehicle-length priors (weaker).")
        used = vehicles

    def deltas(ps):
        return np.array([pair_delta(H_raw, p) for p in ps]) if ps else np.zeros((0, 2))

    lengths = np.array([known_length(p) for p in used])
    sx, sy, mode = fit_scales(deltas(used), lengths, depth_only)

    raw_w, raw_h = calib["raw_width"], calib["raw_height"]
    print(f"\nFit mode: {mode}")
    print(f"Previous scale: sx={calib['sx']:.4f} sy={calib['sy']:.4f}  "
          f"-> twin {raw_w * calib['sx'] / UNITS_PER_METER:.1f} x {raw_h * calib['sy'] / UNITS_PER_METER:.1f} m")
    print(f"New scale     : sx={sx:.4f} sy={sy:.4f}  "
          f"-> twin {raw_w * sx / UNITS_PER_METER:.1f} x {raw_h * sy / UNITS_PER_METER:.1f} m\n")

    print("Before correction (raw homography):")
    report("fit pairs", used, deltas(used), 1.0, 1.0)
    print("After correction:")
    fit_errs = report("fit pairs (in-sample)", used, deltas(used), sx, sy)
    check_errs = report("check pairs (held out = scale error)", check, deltas(check), sx, sy)
    if used is not vehicles:
        report("vehicle lengths vs class priors", vehicles, deltas(vehicles), sx, sy)
    depth_trend(fit_errs + check_errs)

    if not write:
        print("Dry run; pass --write to save homography.npy and reproject the log.")
        return

    H = scaled_homography(H_raw, sx, sy)
    np.save(H_PATH, H)
    calib.update({"sx": sx, "sy": sy})
    save_calib(calib)
    width, height = int(round(raw_w * sx)), int(round(raw_h * sy))
    print(f"Saved {H_PATH} and calibration.json (twin {width} x {height} units)")
    reproject_log(H, H_raw, width, height)


def load_pairs():
    if not os.path.exists(POINTS_PATH):
        return []
    with open(POINTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_pairs(pairs):
    with open(POINTS_PATH, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)


def parse_label(text):
    parts = text.strip().split()
    if not parts:
        return None
    key = parts[0].lower()
    if key in VEHICLE_PRIORS_M and len(parts) == 1:
        return {"kind": "vehicle", "class": key}
    if key in ("f", "fit", "c", "check") and len(parts) == 2:
        return {"kind": "fit" if key.startswith("f") else "check", "distance_m": float(parts[1])}
    if key in ("v", "vehicle") and len(parts) == 2 and parts[1] in VEHICLE_PRIORS_M:
        return {"kind": "vehicle", "class": parts[1]}
    return None


def draw(frame, pairs, pending, frame_no, quad):
    out = frame.copy()
    if quad is None:
        cv2.putText(out, "NO STABILIZATION FOR THIS FRAME - pairs here are skipped", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.polylines(out, [quad.astype(np.int32)], True, (255, 128, 0), 1)
    for p in pairs:
        if p.get("frame") != frame_no:
            continue
        color = KIND_COLORS[p["kind"]]
        a, b = tuple(map(int, p["p1"])), tuple(map(int, p["p2"]))
        cv2.line(out, a, b, color, 2)
        cv2.circle(out, a, 4, color, -1)
        cv2.circle(out, b, 4, color, -1)
        label = f"{p['distance_m']}m" if p["kind"] != "vehicle" else p["class"]
        cv2.putText(out, label, (b[0] + 6, b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for q in pending:
        cv2.circle(out, tuple(map(int, q)), 5, (0, 0, 255), -1)
    help_text = f"frame {frame_no}  a/d +-1  j/l +-30  click 2 pts  u undo  s solve  q quit"
    cv2.putText(out, help_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def collect(video_path, start_frame):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    calib = load_calib()
    H_raw = load_raw_homography()
    T = load_transforms()
    ref_quad = project(np.linalg.inv(H_raw), [
        [0, 0], [calib["raw_width"], 0],
        [calib["raw_width"], calib["raw_height"]], [0, calib["raw_height"]],
    ])

    pairs = load_pairs()
    pending = []
    frame_no = start_frame

    def read(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, f = cap.read()
        return f if ok else None

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending.append([float(x), float(y)])

    win = "depth calibration"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("After two clicks, type in this terminal:")
    print("  f <metres>   fit pair, e.g. one dash + gap, or a lane width")
    print("  c <metres>   held-out check pair, not used to fit")
    print(f"  v <class>    vehicle front-to-rear ground points, class in {list(VEHICLE_PRIORS_M)}")
    print("  (blank)      discard")

    frame = read(frame_no)
    while True:
        if frame is None:
            frame_no = max(0, min(frame_no, total - 1))
            frame = read(frame_no)
        quad = from_reference(T, frame_no, ref_quad)
        cv2.imshow(win, draw(frame, pairs, pending, frame_no, quad))
        key = cv2.waitKey(30) & 0xFF

        if len(pending) == 2:
            cv2.imshow(win, draw(frame, pairs, pending, frame_no, quad))
            cv2.waitKey(1)
            label = None
            try:
                label = parse_label(input("label for this pair: "))
            except ValueError:
                pass
            if label:
                label.update({"frame": frame_no, "p1": pending[0], "p2": pending[1]})
                pairs.append(label)
                save_pairs(pairs)
            else:
                print("discarded (expected e.g. 'f 7.5', 'c 3.5' or 'v erickshaw')")
            pending.clear()
            continue

        step = {ord("d"): 1, ord("a"): -1, ord("l"): 30, ord("j"): -30}.get(key)
        if step:
            frame_no = max(0, min(frame_no + step, total - 1))
            frame = read(frame_no)
        elif key == ord("u"):
            if pending:
                pending.clear()
            elif pairs:
                pairs.pop()
                save_pairs(pairs)
        elif key == ord("s"):
            save_pairs(pairs)
            solve(pairs, write=False)
        elif key in (ord("q"), 27):
            break

    save_pairs(pairs)
    cap.release()
    cv2.destroyAllWindows()
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=VIDEO_PATH)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--solve", action="store_true", help="skip clicking, solve from saved pairs")
    parser.add_argument("--write", action="store_true", help="save corrected homography and reproject log")
    parser.add_argument("--depth-only", action="store_true", help="keep the road-width scale, fit depth only")
    args = parser.parse_args()

    pairs = load_pairs() if args.solve else collect(args.video, args.frame)
    print(f"{len(pairs)} pairs in {POINTS_PATH}")
    solve(pairs, write=args.write, depth_only=args.depth_only)


if __name__ == "__main__":
    main()
