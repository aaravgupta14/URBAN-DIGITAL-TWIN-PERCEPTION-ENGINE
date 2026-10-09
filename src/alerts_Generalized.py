from collections import Counter, deque

import numpy as np

from ttc_Generalized import (
    COLLISION_MARGIN_M, DEFAULT_SIZE_M, DUPLICATE_DIST_M, DUPLICATE_MIN_FRAMES, FAR_FIELD_DEPTH_M,
    MAX_GAP_FRAMES, MAX_PLAUSIBLE_KMH, PERSIST_MIN_FRAMES, STATIONARY_KMH, TTC_WARNING_SEC,
    VEHICLE_SIZE_M, depth_function, extent_along, heading, paths_can_meet, time_to_contact, unit,
)
from plausibility_Generalized import is_implausible_distance

SPEED_LIMIT_KMH = 30.0
SEVERE_OVER_LIMIT_KMH = 15.0
OVERSPEED_MIN_FRAMES = 10
SUDDEN_INCREASE_KMH = 15.0
SUDDEN_WINDOW_SEC = 2.0
SUDDEN_MIN_FRAMES = 5
SUDDEN_EDGE_FRAMES = 10
CRITICAL_TTC_SEC = 1.0
WARMUP_FRAMES = 15
ALERT_COOLDOWN_SEC = 3.0

LIVE_Q_VAR = 0.5
LIVE_R_VAR = 0.25
JUMP_SPEED_MS = MAX_PLAUSIBLE_KMH / 3.6
MIN_JUMP_M = 2.0


class OnlineTrack:
    def __init__(self, x, y, t):
        self.s = np.array([[x, 0.0, 0.0], [y, 0.0, 0.0]])
        self.P = [np.eye(3) * 10.0, np.eye(3) * 10.0]
        self.t = t
        self.age = 1

    def update(self, x, y, t):
        dt = max(t - self.t, 1e-3)
        F = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        Q = LIVE_Q_VAR * np.array([
            [dt**4 / 4, dt**3 / 2, dt**2 / 2],
            [dt**3 / 2, dt**2, dt],
            [dt**2 / 2, dt, 1.0],
        ])
        H = np.array([1.0, 0.0, 0.0])
        for axis, z in ((0, x), (1, y)):
            s = F @ self.s[axis]
            P = F @ self.P[axis] @ F.T + Q
            S = H @ P @ H + LIVE_R_VAR
            K = P @ H / S
            self.s[axis] = s + K * (z - H @ s)
            self.P[axis] = (np.eye(3) - np.outer(K, H)) @ P
        self.t = t
        self.age += 1

    def state(self):
        return {"x": self.s[0, 0], "y": self.s[1, 0], "vx": self.s[0, 1], "vy": self.s[1, 1],
                "ax": self.s[0, 2], "ay": self.s[1, 2]}


class AlertEngine:
    def __init__(self, fps):
        self.fps = fps
        self.depth_of = depth_function()
        self.tracks = {}
        self.classes = {}
        self.speed_history = {}
        self.last_heading = {}
        self.close_counts = Counter()
        self.duplicates = set()
        self.episodes = {}
        self.over_streak = Counter()
        self.sudden_streak = Counter()
        self.last_alert = {}
        self.next_id = 1
        self.alerts = []

    def _speed_kmh(self, tid):
        st = self.tracks[tid].state()
        return 3.6 * (st["vx"] ** 2 + st["vy"] ** 2) ** 0.5

    def _depth(self, y_m):
        return None if self.depth_of is None else self.depth_of(y_m)

    def _far(self, y_m, low_conf):
        d = self._depth(y_m)
        return low_conf if d is None else d > FAR_FIELD_DEPTH_M

    def _new_alert(self, frame_idx, kind, severity, tracks, message, far, **values):
        alert = {"alert_id": self.next_id, "type": kind, "severity": severity, "frame_start": frame_idx,
                 "frame_end": frame_idx, "time_s": round(frame_idx / self.fps, 2), "tracks": tuple(tracks),
                 "far_field": bool(far), "message": message, **values}
        self.next_id += 1
        self.alerts.append(alert)
        return alert

    def _cooled(self, key, frame_idx):
        last = self.last_alert.get(key)
        return last is None or frame_idx - last >= ALERT_COOLDOWN_SEC * self.fps

    def _update_tracks(self, frame_idx, vehicles):
        t = frame_idx / self.fps
        for v in vehicles:
            tid = v["track_id"]
            self.classes.setdefault(tid, Counter())[v["class_id"]] += 1
            tr = self.tracks.get(tid)
            if tr is not None:
                st = tr.state()
                dist = ((v["x"] - st["x"]) ** 2 + (v["y"] - st["y"]) ** 2) ** 0.5
                if dist > max(JUMP_SPEED_MS * (t - tr.t), MIN_JUMP_M):
                    tr = None
                    self.speed_history.pop(tid, None)
            if tr is None:
                self.tracks[tid] = OnlineTrack(v["x"], v["y"], t)
            else:
                tr.update(v["x"], v["y"], t)
            sp = self._speed_kmh(tid)
            if self.tracks[tid].age >= WARMUP_FRAMES:
                self.speed_history.setdefault(tid, deque(maxlen=int(SUDDEN_WINDOW_SEC * self.fps))).append(sp)
            if sp >= STATIONARY_KMH:
                st = self.tracks[tid].state()
                self.last_heading[tid] = unit(st["vx"], st["vy"])

    def _speed_alerts(self, frame_idx, vehicles):
        new = []
        for v in vehicles:
            tid = v["track_id"]
            if self.tracks[tid].age < WARMUP_FRAMES:
                continue
            sp = self._speed_kmh(tid)
            if sp > MAX_PLAUSIBLE_KMH:
                continue
            far = self._far(v["y"], v["low_conf"])

            if sp > SPEED_LIMIT_KMH:
                self.over_streak[tid] += 1
            else:
                self.over_streak[tid] = 0
            if self.over_streak[tid] == OVERSPEED_MIN_FRAMES:
                severe = sp > SPEED_LIMIT_KMH + SEVERE_OVER_LIMIT_KMH
                key = ("overspeed", tid)
                if self._cooled(key, frame_idx):
                    self.last_alert[key] = frame_idx
                    new.append(self._new_alert(
                        frame_idx, "overspeed", "critical" if severe else "warning", [tid],
                        f"track {tid} at {sp:.0f} km/h (limit {SPEED_LIMIT_KMH:.0f})", far,
                        speed_kmh=round(sp, 1)))

            hist = list(self.speed_history.get(tid, []))
            rise = 0.0
            if len(hist) == int(SUDDEN_WINDOW_SEC * self.fps):
                rise = float(np.median(hist[-SUDDEN_EDGE_FRAMES:]) - np.median(hist[:SUDDEN_EDGE_FRAMES]))
            if rise >= SUDDEN_INCREASE_KMH:
                self.sudden_streak[tid] += 1
            else:
                self.sudden_streak[tid] = 0
            if self.sudden_streak[tid] == SUDDEN_MIN_FRAMES:
                key = ("sudden", tid)
                if self._cooled(key, frame_idx):
                    self.last_alert[key] = frame_idx
                    new.append(self._new_alert(
                        frame_idx, "sudden_acceleration", "warning", [tid],
                        f"track {tid} sped up {rise:.0f} km/h in {SUDDEN_WINDOW_SEC:.0f} s (now {sp:.0f} km/h)",
                        far, speed_kmh=round(sp, 1), rise_kmh=round(rise, 1)))
        return new

    def _conflict_hits(self, vehicles):
        hits = {}
        ready = [v for v in vehicles if self.tracks[v["track_id"]].age >= WARMUP_FRAMES]
        for i in range(len(ready)):
            for j in range(i + 1, len(ready)):
                a, b = ready[i], ready[j]
                key = tuple(sorted((a["track_id"], b["track_id"])))
                if key in self.duplicates:
                    continue
                ka, kb = self.tracks[a["track_id"]].state(), self.tracks[b["track_id"]].state()
                dpx, dpy = ka["x"] - kb["x"], ka["y"] - kb["y"]
                dvx, dvy = ka["vx"] - kb["vx"], ka["vy"] - kb["vy"]
                d0 = (dpx**2 + dpy**2) ** 0.5
                if d0 <= DUPLICATE_DIST_M:
                    self.close_counts[key] += 1
                    if self.close_counts[key] >= DUPLICATE_MIN_FRAMES:
                        self.duplicates.add(key)
                    continue
                if is_implausible_distance(d0) or dpx * dvx + dpy * dvy >= 0:
                    continue
                if max(self._speed_kmh(a["track_id"]), self._speed_kmh(b["track_id"])) > MAX_PLAUSIBLE_KMH:
                    continue
                size_a = VEHICLE_SIZE_M.get(self.classes[a["track_id"]].most_common(1)[0][0], DEFAULT_SIZE_M)
                size_b = VEHICLE_SIZE_M.get(self.classes[b["track_id"]].most_common(1)[0][0], DEFAULT_SIZE_M)
                head_a = heading(ka, self.last_heading.get(a["track_id"], (0.0, -1.0)))
                head_b = heading(kb, self.last_heading.get(b["track_id"], (0.0, -1.0)))
                ok, _ = paths_can_meet(ka, kb, size_a, size_b, head_a, head_b)
                if not ok:
                    continue
                direction = (dpx / d0, dpy / d0)
                reach = extent_along(size_a, head_a, direction) + extent_along(size_b, head_b, direction)
                if d0 <= reach:
                    continue
                ttc = time_to_contact(dpx, dpy, dvx, dvy, reach + COLLISION_MARGIN_M)
                if ttc is None or ttc > TTC_WARNING_SEC:
                    continue
                far = self._far(a["y"], a["low_conf"]) or self._far(b["y"], b["low_conf"])
                hits[key] = {"ttc": ttc, "gap": d0 - reach, "far": far}
        return hits

    def _conflict_alerts(self, frame_idx, vehicles):
        new, ended = [], []
        for key, h in self._conflict_hits(vehicles).items():
            ep = self.episodes.get(key)
            if ep is None or frame_idx - ep["last"] > MAX_GAP_FRAMES + 1:
                if ep is not None and ep.get("alert"):
                    ended.append(ep["alert"])
                ep = {"first": frame_idx, "hits": 0, "min_ttc": h["ttc"], "min_gap": h["gap"], "alert": None}
                self.episodes[key] = ep
            ep["last"] = frame_idx
            ep["hits"] += 1
            ep["min_ttc"] = min(ep["min_ttc"], h["ttc"])
            ep["min_gap"] = min(ep["min_gap"], h["gap"])
            if ep["alert"] is None and ep["hits"] >= PERSIST_MIN_FRAMES:
                severity = "critical" if ep["min_ttc"] < CRITICAL_TTC_SEC else "warning"
                ep["alert"] = self._new_alert(
                    ep["first"], "conflict", severity, key,
                    f"tracks {key[0]}-{key[1]} TTC {ep['min_ttc']:.1f} s, gap {ep['min_gap']:.1f} m",
                    h["far"], ttc_sec=round(ep["min_ttc"], 2), gap_m=round(ep["min_gap"], 2))
                new.append(ep["alert"])
            if ep["alert"] is not None:
                ep["alert"].update({"frame_end": frame_idx, "ttc_sec": round(ep["min_ttc"], 2),
                                    "gap_m": round(ep["min_gap"], 2)})

        for key, ep in list(self.episodes.items()):
            if frame_idx - ep["last"] > MAX_GAP_FRAMES + 1:
                if ep.get("alert"):
                    ended.append(ep["alert"])
                del self.episodes[key]
        return new, ended

    def update(self, frame_idx, vehicles):
        self._update_tracks(frame_idx, vehicles)
        new = self._speed_alerts(frame_idx, vehicles)
        conflict_new, ended = self._conflict_alerts(frame_idx, vehicles)
        return new + conflict_new, ended

    def kinematics(self, tid):
        st = self.tracks[tid].state()
        return {**st, "speed_kmh": 3.6 * (st["vx"] ** 2 + st["vy"] ** 2) ** 0.5}

    def active(self, frame_idx, hold_frames=None):
        hold = hold_frames if hold_frames is not None else int(self.fps)
        return [a for a in self.alerts if frame_idx - a["frame_end"] <= hold]

    def finish(self):
        ended = [ep["alert"] for ep in self.episodes.values() if ep.get("alert")]
        self.episodes.clear()
        return ended
