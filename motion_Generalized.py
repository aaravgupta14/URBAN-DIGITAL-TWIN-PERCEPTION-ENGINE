from kalman_Generalized import constant_acceleration_kf
from scene_Generalized import load_calib, UNITS_PER_METER

FPS = float(load_calib()["fps"])
KF_Q_VAR = 0.1
KF_R_VAR = 0.25

def build_frame_index(rows):
    frames = {}
    for r in rows:
        if r["in_bounds"] and r["stable"]:
            frames.setdefault(r["frame_idx"], []).append(r)
    return frames

def build_kinematics(rows):
    by_track = {}
    for r in rows:
        if r["in_bounds"] and r["stable"]:
            by_track.setdefault(r["track_id"], []).append(r)

    kinematics = {}
    for track_id, track_rows in by_track.items():
        track_rows.sort(key=lambda r: r["frame_idx"])
        times = [r["frame_idx"] / FPS for r in track_rows]
        xs = [r["X"] / UNITS_PER_METER for r in track_rows]
        ys = [r["Y"] / UNITS_PER_METER for r in track_rows]

        x_states = constant_acceleration_kf(times, xs, KF_Q_VAR, KF_R_VAR)
        y_states = constant_acceleration_kf(times, ys, KF_Q_VAR, KF_R_VAR)

        for r, (px, vx, ax), (py, vy, ay) in zip(track_rows, x_states, y_states):
            kinematics[(track_id, r["frame_idx"])] = {
                "x": px, "y": py,
                "vx": vx, "vy": vy,
                "ax": ax, "ay": ay,
            }
    return kinematics
