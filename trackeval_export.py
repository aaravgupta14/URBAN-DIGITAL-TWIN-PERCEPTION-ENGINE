from log_loader import load_rows

LOG_PATH = "tracking_log.csv"
OUTPUT_PATH = "predictions_mot.txt"

def export_mot_format(rows, output_path):
    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            if "x1" not in r:
                continue
            bb_left = r["x1"]
            bb_top = r["y1"]
            bb_width = r["x2"] - r["x1"]
            bb_height = r["y2"] - r["y1"]
            f.write(f"{r['frame_idx']},{r['track_id']},{bb_left},{bb_top},"
                    f"{bb_width},{bb_height},{r['conf']:.4f},-1,-1,-1\n")
            written += 1
    return written

if __name__ == "__main__":
    rows = load_rows(LOG_PATH)
    written = export_mot_format(rows, OUTPUT_PATH)
    print(f"wrote {written} rows to {OUTPUT_PATH}")
    if written == 0:
        print("no rows had x1/y1/x2/y2 - rerun digital_twin_log.py to regenerate the CSV with bbox columns")
