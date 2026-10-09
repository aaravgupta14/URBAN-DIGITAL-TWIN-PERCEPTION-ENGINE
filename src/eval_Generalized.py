import statistics
from collections import Counter, defaultdict

from log_loader_Generalized import load_rows
from scene_Generalized import output_size

LOG_PATH = "tracking_log.csv"
CLASS_NAMES = {2: "car", 7: "truck", 5: "bus", 3: "motorcycle", 1: "bicycle", 0: "pedestrian"}
SHORT_TRACK_FRAMES = 5
UNPADDED_WIDTH, UNPADDED_HEIGHT = output_size()

def unpadded_in_bounds(r):
    return 0 <= r["X"] < UNPADDED_WIDTH and 0 <= r["Y"] < UNPADDED_HEIGHT

def headline_report(rows):
    total = len(rows)
    kept_padded = sum(1 for r in rows if r["in_bounds"] and r["stable"])
    kept_unpadded = sum(1 for r in rows if unpadded_in_bounds(r) and r["stable"])
    near = [r for r in rows if not r["low_conf"]]
    far = [r for r in rows if r["low_conf"]]
    near_kept = sum(1 for r in near if r["in_bounds"] and r["stable"])
    far_kept = sum(1 for r in far if r["in_bounds"] and r["stable"])

    print("=== Layer 2 Headline: Coverage ===")
    print(f"coverage vs padded canvas   : {kept_padded}/{total} ({100*kept_padded/total:.1f}%)")
    print(f"coverage vs original {UNPADDED_WIDTH}x{UNPADDED_HEIGHT} canvas : "
          f"{kept_unpadded}/{total} ({100*kept_unpadded/total:.1f}%)")
    if near:
        print(f"near-field coverage (padded) : {near_kept}/{len(near)} ({100*near_kept/len(near):.1f}%)")
    if far:
        print(f"far-field (low_conf) coverage (padded) : {far_kept}/{len(far)} ({100*far_kept/len(far):.1f}%)")
    print()

def completeness_report(rows):
    total = len(rows)
    kept = sum(1 for r in rows if r["in_bounds"] and r["stable"])
    oob_only = sum(1 for r in rows if not r["in_bounds"] and r["stable"])
    unstable_only = sum(1 for r in rows if r["in_bounds"] and not r["stable"])
    both_failed = sum(1 for r in rows if not r["in_bounds"] and not r["stable"])

    print("=== Completeness ===")
    print(f"Total rows: {total}")
    print(f"Kept rows: {kept} ({kept/total:.2%})")
    print(f"Out of bounds only: {oob_only} ({oob_only/total:.2%})")
    print(f"Unstable only: {unstable_only} ({unstable_only/total:.2%})")
    print(f"Both failed: {both_failed} ({both_failed/total:.2%})")
    print()

def track_length_report(rows):
    lengths = Counter(r["track_id"] for r in rows)
    values = list(lengths.values())

    print("=== Track Lengths ===")
    print(f"unique tracks : {len(values)}")
    print(f"min / median / mean / max : "
          f"{min(values)} / {statistics.median(values):.1f} / "
          f"{statistics.mean(values):.1f} / {max(values)}")

    short = [tid for tid, n in lengths.items() if n < SHORT_TRACK_FRAMES]
    print(f"short tracks (<{SHORT_TRACK_FRAMES} frames, possible fragments): "
          f"{len(short)} / {len(values)}  ({100*len(short)/len(values):.1f}%)")
    print()

def class_report(rows):
    by_class = defaultdict(list)
    for r in rows:
        name = CLASS_NAMES.get(r["class_id"], "unknown")
        by_class[name].append(r["conf"])

    print("=== Per-Class Detections ===")
    for name, confs in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
        print(f"{name:12s} rows={len(confs):5d}  avg_conf={statistics.mean(confs):.3f}")
    print()

def class_by_condition_report(rows):
    by_key = defaultdict(list)
    for r in rows:
        name = CLASS_NAMES.get(r["class_id"], "unknown")
        field = "far" if r["low_conf"] else "near"
        by_key[(name, field)].append(r)

    print("=== Per-Class x Near/Far Breakdown ===")
    for (name, field), group in sorted(by_key.items()):
        kept = sum(1 for r in group if r["in_bounds"] and r["stable"])
        avg_conf = statistics.mean(r["conf"] for r in group)
        print(f"{name:12s} {field:5s} rows={len(group):5d}  "
              f"kept={kept:5d} ({100*kept/len(group):5.1f}%)  avg_conf={avg_conf:.3f}")
    print()

def frame_report(rows):
    frames = sorted(set(r["frame_idx"] for r in rows))
    span = frames[-1] - frames[0] + 1
    print("=== Frame Coverage ===")
    print(f"frames with at least one detection : {len(frames)} / {span} "
          f"({100*len(frames)/span:.1f}%)")
    print()

if __name__ == "__main__":
    rows = load_rows(LOG_PATH)
    headline_report(rows)
    completeness_report(rows)
    track_length_report(rows)
    class_report(rows)
    class_by_condition_report(rows)
    frame_report(rows)
