import argparse
import json
import os
import sqlite3
from datetime import datetime

DB_PATH = "twin_state.db"
FLUSH_EVERY = 500

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT, video TEXT, frame_start INTEGER, frame_end INTEGER, fps REAL, calibration TEXT
);
CREATE TABLE IF NOT EXISTS detections (
    run_id INTEGER, frame_idx INTEGER, track_id INTEGER, class_id INTEGER, conf REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL, gx REAL, gy REAL, X REAL, Y REAL,
    low_conf INTEGER, in_bounds INTEGER, stable INTEGER, stab_ok INTEGER
);
CREATE TABLE IF NOT EXISTS kinematics (
    run_id INTEGER, frame_idx INTEGER, track_id INTEGER,
    x_m REAL, y_m REAL, vx REAL, vy REAL, ax REAL, ay REAL, speed_kmh REAL
);
CREATE TABLE IF NOT EXISTS alerts (
    run_id INTEGER, alert_id INTEGER, type TEXT, severity TEXT,
    frame_start INTEGER, frame_end INTEGER, time_s REAL, track_a INTEGER, track_b INTEGER,
    far_field INTEGER, speed_kmh REAL, rise_kmh REAL, ttc_sec REAL, gap_m REAL, message TEXT,
    PRIMARY KEY (run_id, alert_id)
);
CREATE INDEX IF NOT EXISTS idx_det_frame ON detections (run_id, frame_idx);
CREATE INDEX IF NOT EXISTS idx_det_track ON detections (run_id, track_id);
CREATE INDEX IF NOT EXISTS idx_kin_track ON kinematics (run_id, track_id);
"""

DETECTION_COLS = ["frame_idx", "track_id", "class_id", "conf", "x1", "y1", "x2", "y2", "gx", "gy",
                  "X", "Y", "low_conf", "in_bounds", "stable", "stab_ok"]
KINEMATIC_COLS = ["frame_idx", "track_id", "x_m", "y_m", "vx", "vy", "ax", "ay", "speed_kmh"]


class TwinDB:
    def __init__(self, path=DB_PATH, overwrite=False):
        if overwrite:
            for suffix in ("", "-wal", "-shm"):
                if os.path.exists(path + suffix):
                    os.remove(path + suffix)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.run_id = None
        self.det_buf = []
        self.kin_buf = []

    def start_run(self, video, frame_start, frame_end, fps, calibration):
        cur = self.conn.execute(
            "INSERT INTO runs (created, video, frame_start, frame_end, fps, calibration) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), video, frame_start, frame_end, fps,
             json.dumps(calibration)))
        self.conn.commit()
        self.run_id = cur.lastrowid
        return self.run_id

    def add_detection(self, row):
        self.det_buf.append((self.run_id, *[_plain(row.get(c)) for c in DETECTION_COLS]))
        self._maybe_flush()

    def add_kinematics(self, frame_idx, track_id, k):
        self.kin_buf.append((self.run_id, frame_idx, track_id, k["x"], k["y"], k["vx"], k["vy"],
                             k["ax"], k["ay"], k["speed_kmh"]))
        self._maybe_flush()

    def save_alert(self, a):
        tracks = list(a["tracks"]) + [None]
        self.conn.execute(
            "INSERT OR REPLACE INTO alerts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (self.run_id, a["alert_id"], a["type"], a["severity"], a["frame_start"], a["frame_end"],
             a["time_s"], tracks[0], tracks[1], int(a["far_field"]), a.get("speed_kmh"),
             a.get("rise_kmh"), a.get("ttc_sec"), a.get("gap_m"), a["message"]))

    def _maybe_flush(self):
        if len(self.det_buf) + len(self.kin_buf) >= FLUSH_EVERY:
            self.flush()

    def flush(self):
        if self.det_buf:
            self.conn.executemany(
                f"INSERT INTO detections VALUES ({', '.join('?' * (len(DETECTION_COLS) + 1))})", self.det_buf)
            self.det_buf = []
        if self.kin_buf:
            self.conn.executemany(
                f"INSERT INTO kinematics VALUES ({', '.join('?' * (len(KINEMATIC_COLS) + 1))})", self.kin_buf)
            self.kin_buf = []
        self.conn.commit()

    def close(self):
        self.flush()
        self.conn.close()


def _plain(v):
    if isinstance(v, bool):
        return int(v)
    if hasattr(v, "item"):
        return v.item()
    return v


def import_csv(db, log_path, video, fps):
    import pandas as pd
    df = pd.read_csv(log_path)
    for col in ("gx", "gy", "stab_ok"):
        if col not in df.columns:
            df[col] = None
    for col in ("low_conf", "in_bounds", "stable", "stab_ok"):
        df[col] = df[col].map(lambda v: None if v is None or v != v else int(str(v).lower() == "true"))
    db.start_run(video, int(df["frame_idx"].min()), int(df["frame_idx"].max()), fps,
                 {"imported_from": os.path.abspath(log_path)})
    for row in df[DETECTION_COLS].to_dict("records"):
        db.add_detection(row)
    db.flush()
    return db.run_id, len(df)


def summary(db):
    q = db.conn.execute
    for run_id, created, video, fs, fe in q("SELECT run_id, created, video, frame_start, frame_end FROM runs"):
        n_det, n_tracks = q("SELECT COUNT(*), COUNT(DISTINCT track_id) FROM detections WHERE run_id=?",
                            (run_id,)).fetchone()
        print(f"run {run_id}  {created}  frames {fs}-{fe}  {n_det} detections, {n_tracks} tracks  "
              f"({os.path.basename(video or '')})")
        for kind, sev, n in q("SELECT type, severity, COUNT(*) FROM alerts WHERE run_id=? "
                              "GROUP BY type, severity ORDER BY type, severity", (run_id,)):
            print(f"    {kind:20s} {sev:9s} {n}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--import-csv", dest="import_csv")
    parser.add_argument("--video", default="")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    db = TwinDB(args.db)
    if args.import_csv:
        run_id, n = import_csv(db, args.import_csv, args.video, args.fps)
        print(f"Imported {n} rows from {args.import_csv} as run {run_id} into {args.db}")
    if args.summary or not args.import_csv:
        summary(db)
    db.close()


if __name__ == "__main__":
    main()
