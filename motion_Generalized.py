from kalman_Generalized import constant_acceleration_kf
from plausibility_Generalized import URBAN_SPEED_RANGE_KMH
from scene_Generalized import load_calib, UNITS_PER_METER

FPS = float(load_calib()["fps"])
KF_Q_VAR = 0.1
KF_R_VAR = 0.25
MAX_JUMP_SPEED_MS = 1.5 * URBAN_SPEED_RANGE_KMH[1] / 3.6
MIN_JUMP_M = 2.0

def build_frame_index(rows):
    frames = {}
    for r in rows:
        if r["in_bounds"] and r["stable"]:
            frames.setdefault(r["frame_idx"], []).append(r)
    return frames

def split_at_jumps(track_rows):
    segments = [[track_rows[0]]]
    for prev, cur in zip(track_rows, track_rows[1:]):
        dt = (cur["frame_idx"] - prev["frame_idx"]) / FPS
        dist = ((cur["X"] - prev["X"]) ** 2 + (cur["Y"] - prev["Y"]) ** 2) ** 0.5 / UNITS_PER_METER
        if dist > max(MAX_JUMP_SPEED_MS * dt, MIN_JUMP_M):
            segments.append([cur])
        else:
            segments[-1].append(cur)
    return segments

def build_kinematics(rows):
    by_track = {}
    for r in rows:
        if r["in_bounds"] and r["stable"]:
            by_track.setdefault(r["track_id"], []).append(r)

    kinematics = {}
    for track_id, track_rows in by_track.items():
        track_rows.sort(key=lambda r: r["frame_idx"])
        for segment in split_at_jumps(track_rows):
            times = [r["frame_idx"] / FPS for r in segment]
            xs = [r["X"] / UNITS_PER_METER for r in segment]
            ys = [r["Y"] / UNITS_PER_METER for r in segment]

            x_states = constant_acceleration_kf(times, xs, KF_Q_VAR, KF_R_VAR)
            y_states = constant_acceleration_kf(times, ys, KF_Q_VAR, KF_R_VAR)

            for r, (px, vx, ax), (py, vy, ay) in zip(segment, x_states, y_states):
                kinematics[(track_id, r["frame_idx"])] = {
                    "x": px, "y": py,
                    "vx": vx, "vy": vy,
                    "ax": ax, "ay": ay,
                }
    return kinematics
