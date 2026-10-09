from collections import Counter, defaultdict

from log_loader_Generalized import load_rows
from motion_Generalized import build_frame_index, build_kinematics, UNITS_PER_METER
from plausibility_Generalized import is_implausible_distance, MIN_PLAUSIBLE_DIST_M, URBAN_SPEED_RANGE_KMH
from scene_Generalized import load_calib

LOG_PATH = "tracking_log.csv"
TTC_WARNING_SEC = 3.0
COLLISION_MARGIN_M = 0.5
DUPLICATE_DIST_M = 0.5
DUPLICATE_MIN_FRAMES = 3
MADR_MS2 = 3.4

PERSIST_MIN_FRAMES = 5
MAX_GAP_FRAMES = 2
STATIONARY_KMH = 2.0
APPROACH_COS = 0.7
FAR_FIELD_DEPTH_M = 15.0
PARALLEL_COS = 0.9
CROSSING_SLACK_SEC = 1.0
MAX_PLAUSIBLE_KMH = 1.5 * URBAN_SPEED_RANGE_KMH[1]

VEHICLE_SIZE_M = {
    2: (4.3, 1.75),
    3: (2.0, 0.8),
    5: (11.0, 2.5),
    7: (5.0, 2.0),
}
DEFAULT_SIZE_M = (4.0, 1.8)


def find_duplicate_pairs(frames):
    close_counts = {}
    for frame_idx, vehicles in frames.items():
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                a, b = vehicles[i], vehicles[j]
                dx = (a["X"] - b["X"]) / UNITS_PER_METER
                dy = (a["Y"] - b["Y"]) / UNITS_PER_METER
                dist = (dx**2 + dy**2) ** 0.5
                if dist <= DUPLICATE_DIST_M:
                    key = tuple(sorted((a["track_id"], b["track_id"])))
                    close_counts[key] = close_counts.get(key, 0) + 1

    return {pair for pair, count in close_counts.items() if count >= DUPLICATE_MIN_FRAMES}


def solve_mttc(d0, r_prime, r_double_prime):
    if abs(r_double_prime) < 1e-6:
        if r_prime >= 0:
            return None
        return -d0 / r_prime

    a = 0.5 * r_double_prime
    b = r_prime
    c = d0
    disc = b * b - 4 * a * c
    if disc < 0:
        return None

    sqrt_disc = disc ** 0.5
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    candidates = [t for t in (t1, t2) if t > 0]
    if not candidates:
        return None
    return min(candidates)


def track_classes(rows):
    votes = defaultdict(Counter)
    for r in rows:
        votes[r["track_id"]][r["class_id"]] += 1
    return {tid: c.most_common(1)[0][0] for tid, c in votes.items()}


def depth_function():
    calib = load_calib()
    depth_range = calib.get("depth_range_m")
    if not depth_range:
        return None
    z_far = depth_range[1]
    return lambda y_m: z_far - y_m


def unit(x, y):
    n = (x * x + y * y) ** 0.5
    return (x / n, y / n) if n > 1e-9 else None


def speed(k):
    return (k["vx"] ** 2 + k["vy"] ** 2) ** 0.5


def heading(k, fallback):
    return unit(k["vx"], k["vy"]) if speed(k) * 3.6 >= STATIONARY_KMH else fallback


def extent_along(size, head, direction):
    half_len, half_wid = size[0] / 2, size[1] / 2
    along = abs(head[0] * direction[0] + head[1] * direction[1])
    across = abs(-head[1] * direction[0] + head[0] * direction[1])
    return half_len * along + half_wid * across


def lateral_offset(origin, head, point):
    dx, dy = point[0] - origin[0], point[1] - origin[1]
    return abs(-head[1] * dx + head[0] * dy)


def paths_can_meet(ka, kb, size_a, size_b, head_a, head_b):
    moving_a = speed(ka) * 3.6 >= STATIONARY_KMH
    moving_b = speed(kb) * 3.6 >= STATIONARY_KMH
    if not moving_a and not moving_b:
        return False, "both stationary"
    lane_half = (size_a[1] + size_b[1]) / 2 + COLLISION_MARGIN_M
    pa, pb = (ka["x"], ka["y"]), (kb["x"], kb["y"])

    if not moving_a or not moving_b:
        mover, target, head = (ka, pb, head_a) if moving_a else (kb, pa, head_b)
        origin = (mover["x"], mover["y"])
        to_target = unit(target[0] - origin[0], target[1] - origin[1])
        if to_target is None:
            return True, ""
        if head[0] * to_target[0] + head[1] * to_target[1] < APPROACH_COS:
            return False, "not heading toward stationary vehicle"
        if lateral_offset(origin, head, target) > lane_half:
            return False, "stationary vehicle outside path"
        return True, ""

    cos = head_a[0] * head_b[0] + head_a[1] * head_b[1]
    if abs(cos) >= PARALLEL_COS:
        if lateral_offset(pa, head_a, pb) > lane_half:
            return False, "parallel in different lanes"
        return True, ""

    va, vb = (ka["vx"], ka["vy"]), (kb["vx"], kb["vy"])
    det = va[0] * (-vb[1]) - va[1] * (-vb[0])
    if abs(det) < 1e-9:
        return False, "paths do not cross"
    rx, ry = pb[0] - pa[0], pb[1] - pa[1]
    s = (rx * (-vb[1]) - ry * (-vb[0])) / det
    t = (va[0] * ry - va[1] * rx) / det
    horizon = TTC_WARNING_SEC + CROSSING_SLACK_SEC
    if not (0 <= s <= horizon and 0 <= t <= horizon):
        return False, "paths do not cross"
    return True, ""


def time_to_contact(dpx, dpy, dvx, dvy, reach):
    a = dvx * dvx + dvy * dvy
    b = 2 * (dpx * dvx + dpy * dvy)
    c = dpx * dpx + dpy * dpy - reach * reach
    if c <= 0:
        return 0.0
    if a < 1e-9 or b >= 0:
        return None
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    return (-b - disc ** 0.5) / (2 * a)


def frame_conflicts(rows):
    kinematics = build_kinematics(rows)
    frames = build_frame_index(rows)
    duplicate_pairs = find_duplicate_pairs(frames)
    classes = track_classes(rows)
    depth_of = depth_function()
    last_heading = {}
    rejected = Counter()
    hits = []

    for frame_idx in sorted(frames):
        vehicles = frames[frame_idx]
        for v in vehicles:
            k = kinematics.get((v["track_id"], frame_idx))
            if k is not None and speed(k) * 3.6 >= STATIONARY_KMH:
                last_heading[v["track_id"]] = unit(k["vx"], k["vy"])

        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                a, b = vehicles[i], vehicles[j]
                pair_key = tuple(sorted((a["track_id"], b["track_id"])))
                if pair_key in duplicate_pairs:
                    continue

                ka = kinematics.get((a["track_id"], frame_idx))
                kb = kinematics.get((b["track_id"], frame_idx))
                if ka is None or kb is None:
                    continue

                dpx = ka["x"] - kb["x"]
                dpy = ka["y"] - kb["y"]
                dvx = ka["vx"] - kb["vx"]
                dvy = ka["vy"] - kb["vy"]
                d0 = (dpx**2 + dpy**2) ** 0.5
                if is_implausible_distance(d0):
                    rejected["implausible distance"] += 1
                    continue
                if dpx * dvx + dpy * dvy >= 0:
                    continue
                if max(speed(ka), speed(kb)) * 3.6 > MAX_PLAUSIBLE_KMH:
                    rejected["implausible speed (tracking glitch)"] += 1
                    continue

                size_a = VEHICLE_SIZE_M.get(classes[a["track_id"]], DEFAULT_SIZE_M)
                size_b = VEHICLE_SIZE_M.get(classes[b["track_id"]], DEFAULT_SIZE_M)
                head_a = heading(ka, last_heading.get(a["track_id"], (0.0, -1.0)))
                head_b = heading(kb, last_heading.get(b["track_id"], (0.0, -1.0)))

                ok, reason = paths_can_meet(ka, kb, size_a, size_b, head_a, head_b)
                if not ok:
                    rejected[reason] += 1
                    continue

                direction = (dpx / d0, dpy / d0)
                reach = extent_along(size_a, head_a, direction) + extent_along(size_b, head_b, direction)
                if d0 <= reach:
                    rejected["footprints already overlap"] += 1
                    continue
                ttc = time_to_contact(dpx, dpy, dvx, dvy, reach + COLLISION_MARGIN_M)
                if ttc is None or ttc > TTC_WARNING_SEC:
                    rejected["no footprint contact within horizon"] += 1
                    continue

                gap = max(d0 - reach, MIN_PLAUSIBLE_DIST_M)
                r_prime = (dpx * dvx + dpy * dvy) / d0
                dax = ka["ax"] - kb["ax"]
                day = ka["ay"] - kb["ay"]
                r_double_prime = (dvx**2 + dvy**2 + dpx * dax + dpy * day - r_prime**2) / d0
                mttc = solve_mttc(gap, r_prime, r_double_prime)
                drac = r_prime**2 / (2 * gap)

                if depth_of is not None:
                    depth = max(depth_of(ka["y"]), depth_of(kb["y"]))
                    far = depth > FAR_FIELD_DEPTH_M
                else:
                    depth = None
                    far = bool(a["low_conf"] or b["low_conf"])

                hits.append({
                    "pair": pair_key,
                    "frame_idx": frame_idx,
                    "ttc_sec": ttc,
                    "mttc_sec": mttc,
                    "gap_m": gap,
                    "drac_ms2": drac,
                    "depth_m": depth,
                    "far_field": far,
                    "speeds_kmh": (round(speed(ka) * 3.6, 1), round(speed(kb) * 3.6, 1)),
                })

    return hits, rejected, len(duplicate_pairs)


def group_episodes(hits):
    by_pair = defaultdict(list)
    for h in hits:
        by_pair[h["pair"]].append(h)

    episodes = []
    for pair, pair_hits in by_pair.items():
        pair_hits.sort(key=lambda h: h["frame_idx"])
        current = [pair_hits[0]]
        for h in pair_hits[1:]:
            if h["frame_idx"] - current[-1]["frame_idx"] <= MAX_GAP_FRAMES + 1:
                current.append(h)
            else:
                episodes.append((pair, current))
                current = [h]
        episodes.append((pair, current))

    out = []
    for pair, ep in episodes:
        worst = min(ep, key=lambda h: h["ttc_sec"])
        mttcs = [h["mttc_sec"] for h in ep if h["mttc_sec"] is not None]
        out.append({
            "track_a": pair[0],
            "track_b": pair[1],
            "frames": f"{ep[0]['frame_idx']}-{ep[-1]['frame_idx']}",
            "n_frames": len(ep),
            "min_ttc_sec": round(worst["ttc_sec"], 2),
            "min_mttc_sec": round(min(mttcs), 2) if mttcs else None,
            "min_gap_m": round(min(h["gap_m"] for h in ep), 2),
            "max_drac_ms2": round(max(h["drac_ms2"] for h in ep), 2),
            "depth_m": round(worst["depth_m"], 1) if worst["depth_m"] is not None else None,
            "speeds_kmh": worst["speeds_kmh"],
            "far_field": worst["far_field"],
            "exceeds_madr": max(h["drac_ms2"] for h in ep) > MADR_MS2,
        })
    return out


def ttc_report(rows):
    hits, rejected, n_duplicates = frame_conflicts(rows)
    episodes = group_episodes(hits)
    conflicts = [e for e in episodes if e["n_frames"] >= PERSIST_MIN_FRAMES]
    flickers = len(episodes) - len(conflicts)

    print("=== TTC / MTTC / DRAC Conflicts ===")
    print(f"excluded likely-duplicate track pairs : {n_duplicates}")
    print(f"excluded implausible (<{100*MIN_PLAUSIBLE_DIST_M:.0f}cm) single-frame distances : "
          f"{rejected.pop('implausible distance', 0)}")
    print("candidate frames rejected by gates:")
    for reason, count in rejected.most_common():
        print(f"  {reason:38s}: {count}")
    print(f"frame-level hits after gating : {len(hits)}")
    print(f"episodes shorter than {PERSIST_MIN_FRAMES} frames (dropped as flicker) : {flickers}")
    print(f"conflicts (persistent, per pair) : {len(conflicts)}")

    near = [c for c in conflicts if not c["far_field"]]
    far = [c for c in conflicts if c["far_field"]]
    print(f"near-field conflicts (<= {FAR_FIELD_DEPTH_M:.0f} m) : {len(near)}")
    print(f"far-field conflicts (> {FAR_FIELD_DEPTH_M:.0f} m, lower position confidence) : {len(far)}")
    exceeds = sum(1 for c in conflicts if c["exceeds_madr"])
    print(f"conflicts exceeding MADR ({MADR_MS2} m/s^2) : {exceeds}")
    print()

    for c in sorted(conflicts, key=lambda c: c["min_ttc_sec"]):
        print(c)
    return conflicts


if __name__ == "__main__":
    rows = load_rows(LOG_PATH)
    ttc_report(rows)
