import csv

def load_rows(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            r["frame_idx"] = int(r["frame_idx"])
            r["track_id"] = int(r["track_id"])
            r["class_id"] = int(r["class_id"])
            r["conf"] = float(r["conf"])
            if "x1" in r:
                r["x1"] = int(r["x1"])
                r["y1"] = int(r["y1"])
                r["x2"] = int(r["x2"])
                r["y2"] = int(r["y2"])
            r["X"] = float(r["X"])
            r["Y"] = float(r["Y"])
            r["in_bounds"] = r["in_bounds"].strip().lower() == "true"
            r["stable"] = r["stable"].strip().lower() == "true"
            r["low_conf"] = r["low_conf"].strip().lower() == "true"
            rows.append(r)
    return rows
