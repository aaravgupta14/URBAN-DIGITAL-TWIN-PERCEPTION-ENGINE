from collections import defaultdict

from log_loader import load_rows
from motion import build_frame_index, build_kinematics, UNITS_PER_METER
from plausibility import is_implausible_distance, MIN_PLAUSIBLE_DIST_M

LOG_PATH = "tracking_log.csv"
TTC_WARNING_SEC = 3.0
COLLISION_RADIUS_M = 2.5
DUPLICATE_DIST_M = 0.5
DUPLICATE_MIN_FRAMES = 3
MADR_MS2 = 3.4

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

def ttc_report(rows):
    kinematics = build_kinematics(rows)
    frames = build_frame_index(rows)
    duplicate_pairs = find_duplicate_pairs(frames)
    warnings = []
    implausible_skipped = 0

    for frame_idx, vehicles in frames.items():
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
                dax = ka["ax"] - kb["ax"]
                day = ka["ay"] - kb["ay"]

                d0 = (dpx**2 + dpy**2) ** 0.5
                if is_implausible_distance(d0):
                    implausible_skipped += 1
                    continue

                dv_sq = dvx**2 + dvy**2
                dot = dpx * dvx + dpy * dvy
                if dot >= 0 or dv_sq < 1e-6:
                    continue

                ttc = -dot / dv_sq
                closest_x = dpx + ttc * dvx
                closest_y = dpy + ttc * dvy
                closest_dist = (closest_x**2 + closest_y**2) ** 0.5

                if not (0 < ttc <= TTC_WARNING_SEC and closest_dist <= COLLISION_RADIUS_M):
                    continue

                r_prime = dot / d0
                dp_dot_da = dpx * dax + dpy * day
                r_double_prime = (dv_sq + dp_dot_da - r_prime**2) / d0
                mttc = solve_mttc(d0, r_prime, r_double_prime)

                v_close = -r_prime
                drac = (v_close**2) / (2 * d0)

                warnings.append({
                    "frame_idx": frame_idx,
                    "track_a": a["track_id"],
                    "track_b": b["track_id"],
                    "far_field": bool(a["low_conf"] or b["low_conf"]),
                    "ttc_sec": round(ttc, 2),
                    "mttc_sec": round(mttc, 2) if mttc is not None else None,
                    "closest_dist_m": round(closest_dist, 2),
                    "drac_ms2": round(drac, 2),
                    "exceeds_madr": drac > MADR_MS2,
                })

    print("=== TTC / MTTC / DRAC Warnings ===")
    print(f"excluded likely-duplicate track pairs : {len(duplicate_pairs)}")
    print(f"excluded implausible (<{100*MIN_PLAUSIBLE_DIST_M:.0f}cm) single-frame distances : {implausible_skipped}")
    print(f"total collision-risk events : {len(warnings)}")

    near = [w for w in warnings if not w["far_field"]]
    far = [w for w in warnings if w["far_field"]]
    print(f"near-field events : {len(near)}")
    print(f"far-field events (lower position confidence) : {len(far)}")

    exceeds = sum(1 for w in warnings if w["exceeds_madr"])
    print(f"events exceeding MADR ({MADR_MS2} m/s^2) : {exceeds}")
    print()

    for w in sorted(warnings, key=lambda w: w["ttc_sec"])[:10]:
        print(w)

if __name__ == "__main__":
    rows = load_rows(LOG_PATH)
    ttc_report(rows)
