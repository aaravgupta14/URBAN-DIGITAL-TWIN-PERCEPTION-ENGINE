import statistics
from collections import defaultdict

from log_loader_Generalized import load_rows
from motion_Generalized import build_kinematics, UNITS_PER_METER, FPS
from plausibility_Generalized import is_implausible_speed

LOG_PATH = "tracking_log.csv"
CLASS_NAMES = {2: "car", 7: "truck", 5: "bus", 3: "motorcycle", 1: "bicycle", 0: "pedestrian"}
LANE_BIN_WIDTH_M = 1.0
TIME_BIN_SEC = 5.0

def kept(rows):
    return [r for r in rows if r["in_bounds"] and r["stable"]]

def lane_assignment_report(rows):
    xs_m = [r["X"] / UNITS_PER_METER for r in kept(rows)]
    if not xs_m:
        return
    lo = min(xs_m)
    hi = max(xs_m)
    n_bins = max(1, int((hi - lo) / LANE_BIN_WIDTH_M) + 1)
    counts = [0] * n_bins
    for x in xs_m:
        idx = min(n_bins - 1, int((x - lo) / LANE_BIN_WIDTH_M))
        counts[idx] += 1

    print("=== Lane Assignment (1D histogram over X, meters) ===")
    peak = max(counts) if counts else 1
    for i, c in enumerate(counts):
        bin_lo = lo + i * LANE_BIN_WIDTH_M
        bar = "#" * int(40 * c / peak) if peak else ""
        print(f"{bin_lo:6.1f}m  {c:5d}  {bar}")
    print()

def speed_distribution_report(rows):
    kinematics = build_kinematics(rows)
    by_class = defaultdict(list)
    for r in kept(rows):
        k = kinematics.get((r["track_id"], r["frame_idx"]))
        if k is None:
            continue
        speed = (k["vx"]**2 + k["vy"]**2) ** 0.5
        name = CLASS_NAMES.get(r["class_id"], "unknown")
        by_class[name].append(speed)

    print("=== Speed Distribution (m/s, Kalman-smoothed) ===")
    print("all readings, then readings above the sub-walking-pace plausibility floor")
    for name, speeds in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
        if len(speeds) < 2:
            continue
        print(f"{name:12s} n={len(speeds):5d}  "
              f"mean={statistics.mean(speeds):5.2f}  "
              f"median={statistics.median(speeds):5.2f}  "
              f"stdev={statistics.stdev(speeds):5.2f}  "
              f"min={min(speeds):5.2f}  max={max(speeds):5.2f}")

        plausible = [s for s in speeds if not is_implausible_speed(s)]
        implausible_count = len(speeds) - len(plausible)
        if len(plausible) >= 2:
            print(f"{'':12s} implausible(<2km/h)={implausible_count:5d}  "
                  f"plausible_mean={statistics.mean(plausible):5.2f}  "
                  f"plausible_median={statistics.median(plausible):5.2f}")
    print()

def find_crossing_times(rows, detector_y):
    by_track = defaultdict(list)
    for r in kept(rows):
        by_track[r["track_id"]].append(r)

    crossings = []
    for track_id, track_rows in by_track.items():
        track_rows.sort(key=lambda r: r["frame_idx"])
        for prev, cur in zip(track_rows, track_rows[1:]):
            y0, y1 = prev["Y"], cur["Y"]
            if (y0 - detector_y) * (y1 - detector_y) > 0:
                continue
            if y1 == y0:
                continue
            frac = (detector_y - y0) / (y1 - y0)
            t0, t1 = prev["frame_idx"] / FPS, cur["frame_idx"] / FPS
            crossings.append(t0 + frac * (t1 - t0))
    return sorted(crossings)

def headway_report(rows):
    ys = [r["Y"] for r in kept(rows)]
    if not ys:
        return
    detector_y = statistics.median(ys)
    crossings = find_crossing_times(rows, detector_y)

    print(f"=== Headway/Gap at Virtual Detector Line (Y={detector_y:.1f}) ===")
    print(f"vehicles crossing : {len(crossings)}")
    if len(crossings) >= 2:
        gaps = [b - a for a, b in zip(crossings, crossings[1:])]
        print(f"headway (sec) mean={statistics.mean(gaps):.2f}  "
              f"median={statistics.median(gaps):.2f}  "
              f"min={min(gaps):.2f}  max={max(gaps):.2f}")
    print()

def temporal_profile_report(rows):
    kinematics = build_kinematics(rows)
    bins = defaultdict(list)
    for r in kept(rows):
        t = r["frame_idx"] / FPS
        bin_idx = int(t // TIME_BIN_SEC)
        k = kinematics.get((r["track_id"], r["frame_idx"]))
        speed = (k["vx"]**2 + k["vy"]**2) ** 0.5 if k else None
        bins[bin_idx].append((r["track_id"], speed))

    print(f"=== Temporal Profile ({TIME_BIN_SEC:.0f}s bins) ===")
    for bin_idx in sorted(bins):
        entries = bins[bin_idx]
        track_count = len(set(tid for tid, _ in entries))
        speeds = [s for _, s in entries if s is not None]
        avg_speed = statistics.mean(speeds) if speeds else 0.0
        start_sec = bin_idx * TIME_BIN_SEC
        print(f"t={start_sec:6.0f}s  active_tracks={track_count:4d}  "
              f"detections={len(entries):5d}  avg_speed={avg_speed:5.2f} m/s")
    print()

def fundamental_diagram_report(rows):
    kinematics = build_kinematics(rows)
    ys_m = [r["Y"] / UNITS_PER_METER for r in kept(rows)]
    if not ys_m:
        return
    seg_lo = statistics.median(ys_m) - 5.0
    seg_hi = statistics.median(ys_m) + 5.0
    seg_len = seg_hi - seg_lo

    by_track = defaultdict(list)
    for r in kept(rows):
        by_track[r["track_id"]].append(r)

    bin_td = defaultdict(float)
    bin_tt = defaultdict(float)

    for track_id, track_rows in by_track.items():
        track_rows.sort(key=lambda r: r["frame_idx"])
        for prev, cur in zip(track_rows, track_rows[1:]):
            y_mid = ((prev["Y"] + cur["Y"]) / 2) / UNITS_PER_METER
            if not (seg_lo <= y_mid <= seg_hi):
                continue
            dt = (cur["frame_idx"] - prev["frame_idx"]) / FPS
            if dt <= 0 or dt > 1.0:
                continue
            k = kinematics.get((track_id, cur["frame_idx"]))
            if k is None:
                continue
            speed = (k["vx"]**2 + k["vy"]**2) ** 0.5
            bin_idx = int((cur["frame_idx"] / FPS) // TIME_BIN_SEC)
            bin_td[bin_idx] += speed * dt
            bin_tt[bin_idx] += dt

    print(f"=== Fundamental Diagram (Edie's measures, segment {seg_len:.1f}m wide) ===")
    print(f"{'t_start':>8s} {'flow(veh/s)':>12s} {'density(veh/m)':>15s} {'speed(m/s)':>11s}")
    for bin_idx in sorted(set(bin_td) | set(bin_tt)):
        td = bin_td.get(bin_idx, 0.0)
        tt = bin_tt.get(bin_idx, 0.0)
        area = seg_len * TIME_BIN_SEC
        flow = td / area if area else 0.0
        density = tt / area if area else 0.0
        speed = td / tt if tt > 0 else 0.0
        start_sec = bin_idx * TIME_BIN_SEC
        print(f"{start_sec:8.0f} {flow:12.4f} {density:15.4f} {speed:11.2f}")
    print()

if __name__ == "__main__":
    rows = load_rows(LOG_PATH)
    lane_assignment_report(rows)
    speed_distribution_report(rows)
    headway_report(rows)
    temporal_profile_report(rows)
    fundamental_diagram_report(rows)
