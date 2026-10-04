import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from log_loader_Generalized import load_rows
from motion_Generalized import build_kinematics
from plausibility_Generalized import (
    is_urban_speed, is_implausible_accel, is_implausible_speed,
    URBAN_SPEED_RANGE_KMH, MAX_PLAUSIBLE_ACCEL_MS2,
)

LOG_PATH = "tracking_log.csv"
FIG_PATH = "sanity_checks.png"
CLASS_NAMES = {2: "car", 7: "truck", 5: "bus", 3: "motorcycle"}
MIN_TRACK_FRAMES = 10
WORST_N = 10


def collect(rows):
    kin = build_kinematics(rows)
    per_track = defaultdict(list)
    cls = {}
    for r in rows:
        k = kin.get((r["track_id"], r["frame_idx"]))
        if k is None:
            continue
        speed = (k["vx"] ** 2 + k["vy"] ** 2) ** 0.5
        accel = (k["ax"] ** 2 + k["ay"] ** 2) ** 0.5
        per_track[r["track_id"]].append((speed, accel, r["low_conf"]))
        cls[r["track_id"]] = CLASS_NAMES.get(r["class_id"], "other")
    return {t: v for t, v in per_track.items() if len(v) >= MIN_TRACK_FRAMES}, cls


def speed_report(per_track, cls):
    lo, hi = URBAN_SPEED_RANGE_KMH
    medians = {t: statistics.median(s for s, _, _ in v) for t, v in per_track.items()}
    moving = {t: m for t, m in medians.items() if not is_implausible_speed(m)}
    print(f"=== Speed (per-track median, tracks >= {MIN_TRACK_FRAMES} frames) ===")
    print(f"tracks: {len(medians)}   moving: {len(moving)}")
    if not moving:
        print()
        return medians
    kmh = np.array(list(moving.values())) * 3.6
    p5, p50, p95 = np.percentile(kmh, [5, 50, 95])
    inside = sum(is_urban_speed(m) for m in moving.values())
    print(f"km/h  p5 {p5:5.1f}  median {p50:5.1f}  p95 {p95:5.1f}  max {kmh.max():5.1f}")
    print(f"inside {lo:.0f}-{hi:.0f} km/h: {inside}/{len(moving)} ({100 * inside / len(moving):.0f}%)")
    by_class = defaultdict(list)
    for t, m in moving.items():
        by_class[cls[t]].append(m * 3.6)
    for name, vals in sorted(by_class.items()):
        print(f"  {name:11s} n={len(vals):3d}  median {statistics.median(vals):5.1f} km/h")
    if p50 > hi or p50 < lo:
        print("Median is outside the urban range: suspect the scale (depth_calib_Generalized.py) or fps (fps_check_video_calb.py).")
    print()
    return medians


def accel_report(per_track):
    all_acc = [a for v in per_track.values() for _, a, _ in v]
    near = [a for v in per_track.values() for _, a, lc in v if not lc]
    far = [a for v in per_track.values() for _, a, lc in v if lc]
    print(f"=== Acceleration (Kalman, |a| > {MAX_PLAUSIBLE_ACCEL_MS2} m/s^2 is implausible) ===")
    for name, vals in (("all", all_acc), ("near-field", near), ("far-field", far)):
        if not vals:
            continue
        bad = sum(is_implausible_accel(a) for a in vals)
        print(f"{name:11s} readings {len(vals):6d}  implausible {bad:6d} ({100 * bad / len(vals):5.1f}%)  "
              f"p95 {np.percentile(vals, 95):5.2f} m/s^2")
    worst = sorted(
        ((t, sum(is_implausible_accel(a) for _, a, _ in v) / len(v)) for t, v in per_track.items()),
        key=lambda x: -x[1],
    )[:WORST_N]
    print("tracks with most implausible readings: "
          + ", ".join(f"{t} ({100 * f:.0f}%)" for t, f in worst if f > 0))
    print()
    return all_acc


def plot(medians, all_acc):
    lo, hi = URBAN_SPEED_RANGE_KMH
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.hist(np.array(list(medians.values())) * 3.6, bins=30, color="tab:blue")
    ax1.axvspan(lo, hi, color="tab:green", alpha=0.15)
    ax1.set_xlabel("per-track median speed (km/h)")
    ax1.set_ylabel("tracks")
    ax2.hist(np.clip(all_acc, 0, 3 * MAX_PLAUSIBLE_ACCEL_MS2), bins=40, color="tab:orange")
    ax2.axvline(MAX_PLAUSIBLE_ACCEL_MS2, color="tab:red", linestyle="--")
    ax2.set_xlabel("|acceleration| (m/s^2, clipped)")
    ax2.set_ylabel("readings")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=120)
    print(f"Saved {FIG_PATH}")


if __name__ == "__main__":
    rows = load_rows(LOG_PATH)
    rows = [r for r in rows if r["in_bounds"] and r["stable"]]
    per_track, cls = collect(rows)
    medians = speed_report(per_track, cls)
    all_acc = accel_report(per_track)
    if medians and all_acc:
        plot(medians, all_acc)
