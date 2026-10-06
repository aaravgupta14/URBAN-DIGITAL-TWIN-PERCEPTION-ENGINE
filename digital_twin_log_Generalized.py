
import argparse
import sys
import tempfile
import time

import cv2
import numpy as np
import torch
import pandas as pd
import yaml
import matplotlib
if "--no-show" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty, Full
from threading import Event, Thread
from collections import deque
import os

from scene_Generalized import (
    load_calib, output_size, in_bounds as world_in_bounds, H_PATH,
    load_transforms, to_reference,
    UNITS_PER_METER, PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM,
)
from alerts_Generalized import AlertEngine
from twin_db_Generalized import TwinDB
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cv2.setNumThreads(1)
torch.set_num_threads(1)
torch.set_float32_matmul_precision("high")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument("--video", default="dataset/Weast (1).mp4")
parser.add_argument("--start", type=int, default=0, help="first frame to process (0-based)")
parser.add_argument("--end", type=int, default=None, help="last frame to process (0-based)")
parser.add_argument("--log", default="tracking_log.csv")
parser.add_argument("--model", default="yolo11s.pt")
parser.add_argument("--db", default="twin_state.db")
parser.add_argument("--alerts", default="alerts.csv")
parser.add_argument("--fast", action="store_true", help="BoT-SORT without camera-motion compensation (~30%% faster)")
parser.add_argument("--plot3d-every", type=int, default=2, help="draw the 3D view every N frames, 0 disables it")
parser.add_argument("--no-show", action="store_true", help="no live windows (batch runs)")
ARGS = parser.parse_args()

VIDEO_PATH = ARGS.video
HOMOGRAPHY_PATH = H_PATH

CALIB = load_calib()
OUTPUT_WIDTH, OUTPUT_HEIGHT = output_size(CALIB)
WORLD_WIDTH = OUTPUT_WIDTH + PAD_LEFT + PAD_RIGHT
WORLD_HEIGHT = OUTPUT_HEIGHT + PAD_TOP + PAD_BOTTOM

VIEW_MAX_WIDTH = 1200
VIEW_MAX_HEIGHT = 720
VIEW_MIN_WIDTH = 520
VIEW_HEADER = 40
VIEW_SCALE = min(1.0, VIEW_MAX_WIDTH / WORLD_WIDTH, VIEW_MAX_HEIGHT / WORLD_HEIGHT)
CANVAS_WIDTH = max(int(WORLD_WIDTH * VIEW_SCALE), VIEW_MIN_WIDTH)
CANVAS_HEIGHT = int(WORLD_HEIGHT * VIEW_SCALE) + VIEW_HEADER
VIEW_OFFSET_X = (CANVAS_WIDTH - int(WORLD_WIDTH * VIEW_SCALE)) // 2
PROXIMITY_THRESHOLD = int(2.5 * UNITS_PER_METER)
ARROW_LENGTH = 20
MIN_MOVEMENT_FOR_ARROW = 2
MIN_TRACK_AGE = 2
STEP_HISTORY_LEN = 8
JUMP_ANOMALY_MULT = 6
JUMP_ANOMALY_MIN_ABS = int(0.125 * UNITS_PER_METER)
SMOOTHING_ALPHA = 0.4
GROUND_MAX_RISE_FRAC = 0.2
EDGE_MARGIN_PX = 3

FRAME_QUEUE_MAX = 4

RECORD = True
SOURCE_OUTPUT_PATH = "output_tracking_source.avi"
TWIN_OUTPUT_PATH = "output_digital_twin.avi"
RECORD_FPS = float(CALIB["fps"])

LOG_PATH = ARGS.log
VIDEO_LABEL = "weast"

CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CLASS_COLORS = {"car": "tab:blue", "motorcycle": "tab:orange", "bus": "tab:green", "truck": "tab:red"}

Z_PLANE = 0.0
Z_LIM = 1.0

CLASS_RGBA = {name: to_rgba(c) for name, c in CLASS_COLORS.items()}
DEFAULT_RGBA = to_rgba("gray")

UPDATE_EVERY = ARGS.plot3d_every
GRAPH_OUTPUT_PATH = "output_digital_twin_3d.avi"

ALERT_COLORS = {"critical": (0, 0, 255), "warning": (0, 165, 255)}
ALERT_TYPE_COLORS = {"sudden_acceleration": (255, 0, 255)}
ALERT_LABELS = {"conflict": "CONFLICT", "overspeed": "OVERSPEED", "sudden_acceleration": "SUDDEN ACCEL"}
BANNER_MAX = 4

H = np.load(HOMOGRAPHY_PATH)
STAB_T = load_transforms()
print("Stabilization: " + ("loaded, ground points mapped to the calibration frame" if STAB_T is not None
                           else "not found, assuming a static camera (run stabilize_video_calb.py)"))
H_inv = np.linalg.inv(H)
print(f"Twin world: {OUTPUT_WIDTH / UNITS_PER_METER:.1f} m x {OUTPUT_HEIGHT / UNITS_PER_METER:.1f} m "
      f"(sx={CALIB['sx']:.3f}, sy={CALIB['sy']:.3f}), view scale {VIEW_SCALE:.3f}")
_canvas_corners = np.float32(
    [[0, 0], [OUTPUT_WIDTH, 0], [OUTPUT_WIDTH, OUTPUT_HEIGHT], [0, OUTPUT_HEIGHT]]
).reshape(-1, 1, 2)
_src_corners = cv2.perspectiveTransform(_canvas_corners, H_inv).reshape(-1, 2)
CALIB_TOP_Y = float(np.min(_src_corners[:, 1]))
CALIB_BOTTOM_Y = float(np.max(_src_corners[:, 1]))
LOW_CONF_Y = CALIB_TOP_Y + 0.35 * (CALIB_BOTTOM_Y - CALIB_TOP_Y)
print(f"Calibrated image-y range: {CALIB_TOP_Y:.0f} - {CALIB_BOTTOM_Y:.0f} "
      f"(low-confidence below y={LOW_CONF_Y:.0f})")

USE_HALF = device == "cuda"
TRACK_DEVICE = 0 if device == "cuda" else "cpu"


def tracker_config():
    if not ARGS.fast:
        return "botsort.yaml"
    import ultralytics
    with open(os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers", "botsort.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["gmc_method"] = "none"
    path = os.path.join(tempfile.gettempdir(), "botsort_no_gmc.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


TRACKER_CFG = tracker_config()
print(f"Tracker: {'BoT-SORT without GMC (fast)' if ARGS.fast else 'BoT-SORT'}, "
      f"{'fp16' if USE_HALF else 'fp32'} on {device}")


class AsyncWriter:
    def __init__(self):
        self.queue = Queue(maxsize=64)
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            writer, image = item
            writer.write(image)

    def write(self, writer, image):
        self.queue.put((writer, image))

    def close(self):
        self.queue.put(None)
        self.thread.join()


async_writer = AsyncWriter()

model = YOLO(ARGS.model)
model.to(device)
dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
print("Warming up GPU...")
for _ in range(5):
    with torch.inference_mode():
        model.predict(dummy, device=TRACK_DEVICE, half=USE_HALF, verbose=False)
if device == "cuda":
    torch.cuda.synchronize()
print("Warm-up complete.")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")
cap.set(cv2.CAP_PROP_POS_FRAMES, ARGS.start)

source_writer = None
twin_writer = None
if RECORD:
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    source_writer = cv2.VideoWriter(SOURCE_OUTPUT_PATH, fourcc, RECORD_FPS, (src_w, src_h))
    twin_writer = cv2.VideoWriter(TWIN_OUTPUT_PATH, fourcc, RECORD_FPS, (CANVAS_WIDTH, CANVAS_HEIGHT))
    print(f"Recording to {SOURCE_OUTPUT_PATH} and {TWIN_OUTPUT_PATH}")

frame_queue: "Queue" = Queue(maxsize=FRAME_QUEUE_MAX)
stop_event = Event()

def video_reader():
    pos = ARGS.start
    while not stop_event.is_set():
        if ARGS.end is not None and pos > ARGS.end:
            break
        ret, frame = cap.read()
        if not ret:
            break
        pos += 1
        while not stop_event.is_set():
            try:
                frame_queue.put(frame, timeout=0.5)
                break
            except Full:
                continue
    frame_queue.put(None)

def compute_close_pairs(X, Y, threshold):
    n = len(X)
    if n < 2:
        return []
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    dx = X[:, None] - X[None, :]
    dy = Y[:, None] - Y[None, :]
    dist_matrix = np.hypot(dx, dy)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dists = dist_matrix[i_idx, j_idx]
    mask = dists < threshold
    return list(zip(i_idx[mask].tolist(), j_idx[mask].tolist(), dists[mask].tolist()))

def format_distance(d):
    if UNITS_PER_METER:
        return f"{d / UNITS_PER_METER:.1f} m"
    return f"{int(d)} u"

def to_view(X, Y):
    return int(X * VIEW_SCALE) + VIEW_OFFSET_X, int(Y * VIEW_SCALE) + VIEW_HEADER

def refine_ground_point(frame, x1, y1, x2, y2):
    h = y2 - y1
    if h < 10:
        return (x1 + x2) // 2, y2
    band_top = int(y1 + h * 0.6)
    band_bottom = min(y2 + int(h * 0.15), frame.shape[0] - 1)
    x1c, x2c = max(x1, 0), min(x2, frame.shape[1])
    crop = frame[band_top:band_bottom, x1c:x2c]
    if crop.size == 0 or crop.shape[1] < 4:
        return (x1 + x2) // 2, y2

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    w = gray.shape[1]
    lo, hi = int(w * 0.3), int(w * 0.7)
    if hi <= lo:
        return (x1 + x2) // 2, y2
    strip = gray[:, lo:hi]
    row_brightness = strip.mean(axis=1)
    dark_rows = np.where(row_brightness < row_brightness.mean())[0]
    ground_row = int(dark_rows.max()) if len(dark_rows) else strip.shape[0] - 1
    refined_y = band_top + ground_row
    if refined_y < y2 - GROUND_MAX_RISE_FRAC * h:
        refined_y = y2
    return (x1 + x2) // 2, refined_y

executor = ThreadPoolExecutor(max_workers=1)
reader_future = executor.submit(video_reader)

paused = False
last_frame = None
prev_positions = {}
track_streak = {}
step_history = {}
smoothed_positions = {}
track_class = {}
log_rows = []
frame_idx = ARGS.start
frame_times = deque(maxlen=300)

engine = AlertEngine(RECORD_FPS)
db = TwinDB(ARGS.db)
db.start_run(VIDEO_PATH, ARGS.start + 1, None if ARGS.end is None else ARGS.end + 1, RECORD_FPS, CALIB)
print(f"Alerts: limit {__import__('alerts_Generalized').SPEED_LIMIT_KMH:.0f} km/h; "
      f"writing {ARGS.alerts} and run {db.run_id} in {ARGS.db}")


def alert_color(a):
    return ALERT_TYPE_COLORS.get(a["type"], ALERT_COLORS[a["severity"]])

plt.ion()
fig3d = plt.figure(figsize=(6, 5))
ax3d = fig3d.add_subplot(111, projection="3d")

ax3d.set_xlabel("X (across road)")
ax3d.set_ylabel("Y (0 = far)")
ax3d.set_zlabel("Z (road plane)")
ax3d.set_xlim(0, WORLD_WIDTH)
ax3d.set_ylim(0, WORLD_HEIGHT)
ax3d.set_zlim(-Z_LIM, Z_LIM)
ax3d.set_zticks([Z_PLANE])
title3d = ax3d.set_title("Live 3D Digital Twin")

scat3d = ax3d.scatter([0.0], [0.0], [Z_PLANE], s=25, depthshade=False)

graph_writer = None
while True:
    if not paused:
        try:
            frame = frame_queue.get(timeout=1.0)
        except Empty:
            continue
        if frame is None:
            break
        last_frame = frame
        frame_idx += 1
        new_frame = True
    else:
        frame = last_frame
        if frame is None:
            continue
        new_frame = False

    with torch.no_grad():
        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER_CFG,
            classes=[2, 3, 5, 7],
            imgsz=1280,
            conf=0.1,
            device=TRACK_DEVICE,
            half=USE_HALF,
            verbose=False,
        )

    boxes = results[0].boxes
    annotated_frame = frame.copy()
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), (40, 40, 40), dtype=np.uint8)

    vehicles = []
    live_vehicles = []
    frame_boxes = {}
    total_tracked = 0

    if boxes.id is not None:
        ids = boxes.id.cpu().numpy().astype(np.int32)
        total_tracked = len(ids)
        xyxy = boxes.xyxy.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy().astype(np.int32)
        conf_arr = boxes.conf.cpu().numpy()
        x1 = xyxy[:, 0]
        y1 = xyxy[:, 1]
        x2 = xyxy[:, 2]
        y2 = xyxy[:, 3]

        ground_pts = [
            refine_ground_point(frame, int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]))
            for i in range(len(ids))
        ]
        cx_arr = np.array([p[0] for p in ground_pts], dtype=np.float64)
        cy_arr = np.array([p[1] for p in ground_pts], dtype=np.float64)
        image_pts = np.stack([cx_arr, cy_arr], axis=1)
        ref_pts = to_reference(STAB_T, frame_idx - 1, image_pts)
        stab_ok = ref_pts is not None
        if not stab_ok:
            ref_pts = image_pts
        world_pts = cv2.perspectiveTransform(
            ref_pts.astype(np.float32).reshape(-1, 1, 2), H
        ).reshape(-1, 2)

        for tid in ids:
            track_streak[int(tid)] = track_streak.get(int(tid), 0) + 1

        for idx, track_id in enumerate(ids):
            track_id = int(track_id)
            tx1, ty1 = int(x1[idx]), int(y1[idx])
            tx2, ty2 = int(x2[idx]), int(y2[idx])
            cx, cy = int(cx_arr[idx]), int(cy_arr[idx])
            X, Y = world_pts[idx]
            low_conf = ref_pts[idx, 1] < LOW_CONF_Y

            cv2.rectangle(annotated_frame, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
            cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                annotated_frame, f"ID:{track_id}", (tx1, ty1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

            truncated = (tx1 <= EDGE_MARGIN_PX or tx2 >= frame.shape[1] - EDGE_MARGIN_PX
                         or ty2 >= frame.shape[0] - EDGE_MARGIN_PX)
            in_bounds = stab_ok and not truncated and world_in_bounds(X, Y, OUTPUT_WIDTH, OUTPUT_HEIGHT)
            stable = track_streak[track_id] >= MIN_TRACK_AGE

            log_rows.append({
                "source": VIDEO_LABEL,
                "frame_idx": frame_idx,
                "track_id": track_id,
                "class_id": int(cls_arr[idx]),
                "conf": float(conf_arr[idx]),
                "x1": tx1,
                "y1": ty1,
                "x2": tx2,
                "y2": ty2,
                "gx": float(ref_pts[idx, 0]),
                "gy": float(ref_pts[idx, 1]),
                "stab_ok": bool(stab_ok),
                "truncated": bool(truncated),
                "X": float(X),
                "Y": float(Y),
                "low_conf": bool(low_conf),
                "in_bounds": bool(in_bounds),
                "stable": bool(stable),
            })
            frame_boxes[track_id] = (tx1, ty1, tx2, ty2)
            if new_frame:
                db.add_detection(log_rows[-1])

            if in_bounds and stable:
                prev_smooth = smoothed_positions.get(track_id, (X, Y))
                sX = SMOOTHING_ALPHA * X + (1 - SMOOTHING_ALPHA) * prev_smooth[0]
                sY = SMOOTHING_ALPHA * Y + (1 - SMOOTHING_ALPHA) * prev_smooth[1]
                smoothed_positions[track_id] = (sX, sY)
                cls_name = CLASS_NAMES.get(int(cls_arr[idx]), "vehicle")
                track_class[track_id] = cls_name
                live_vehicles.append({
                    "track_id": track_id, "class_id": int(cls_arr[idx]),
                    "x": float(X) / UNITS_PER_METER, "y": float(Y) / UNITS_PER_METER,
                    "low_conf": bool(low_conf),
                })
                vehicles.append({
                    "id": track_id,
                    "X": sX + PAD_LEFT,
                    "Y": sY + PAD_TOP,
                    "Z": Z_PLANE,
                    "low_conf": low_conf,
                })

    for v in vehicles:

        color = (0, 165, 255) if v["low_conf"] else (0, 255, 0)
        vx, vy = to_view(v["X"], v["Y"])
        cv2.circle(canvas, (vx, vy), 7, color, -1)
        cv2.putText(
            canvas, f"ID:{v['id']}", (vx + 10, vy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        prev = prev_positions.get(v["id"])
        if prev is not None:
            dx, dy = v["X"] - prev[0], v["Y"] - prev[1]
            mag = (dx ** 2 + dy ** 2) ** 0.5
            history = step_history.setdefault(v["id"], deque(maxlen=STEP_HISTORY_LEN))

            anomalous = (
                len(history) >= 3
                and mag > JUMP_ANOMALY_MIN_ABS
                and mag > JUMP_ANOMALY_MULT * (sorted(history)[len(history) // 2])
            )

            if not anomalous:
                if mag > MIN_MOVEMENT_FOR_ARROW:
                    ux, uy = dx / mag, dy / mag
                    tip = (int(vx + ux * ARROW_LENGTH), int(vy + uy * ARROW_LENGTH))
                    cv2.arrowedLine(canvas, (vx, vy), tip, (255, 255, 0), 2, tipLength=0.35)
                history.append(mag)

    prev_positions = {v["id"]: (v["X"], v["Y"]) for v in vehicles}

    if len(vehicles) >= 2:
        Xs = [v["X"] for v in vehicles]
        Ys = [v["Y"] for v in vehicles]
        for i, j, dist in compute_close_pairs(Xs, Ys, PROXIMITY_THRESHOLD):
            v1, v2 = vehicles[i], vehicles[j]
            p1 = to_view(v1["X"], v1["Y"])
            p2 = to_view(v2["X"], v2["Y"])
            cv2.line(canvas, p1, p2, (0, 0, 255), 2)
            mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            cv2.putText(
                canvas, format_distance(dist), (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

    if new_frame:
        new_alerts, ended_alerts = engine.update(frame_idx, live_vehicles)
        for a in new_alerts:
            print(f"[{a['severity'].upper()}] frame {frame_idx} ({a['time_s']:.1f} s) "
                  f"{ALERT_LABELS[a['type']]}: {a['message']}{' (far field)' if a['far_field'] else ''}")
            db.save_alert(a)
        for a in ended_alerts:
            db.save_alert(a)
        for lv in live_vehicles:
            db.add_kinematics(frame_idx, lv["track_id"], engine.kinematics(lv["track_id"]))

    view_pos = {v["id"]: to_view(v["X"], v["Y"]) for v in vehicles}
    active_alerts = engine.active(frame_idx)
    for a in active_alerts:
        color = alert_color(a)
        for tid in a["tracks"]:
            if tid in frame_boxes:
                bx1, by1, bx2, by2 = frame_boxes[tid]
                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, 4)
                cv2.putText(annotated_frame, ALERT_LABELS[a["type"]], (bx1, by2 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if tid in view_pos:
                cv2.circle(canvas, view_pos[tid], 13, color, 2)
        if a["type"] == "conflict" and all(t in view_pos for t in a["tracks"]):
            cv2.line(canvas, view_pos[a["tracks"][0]], view_pos[a["tracks"][1]], color, 3)
    for lv in live_vehicles:
        tid = lv["track_id"]
        if tid in frame_boxes and engine.tracks[tid].age >= 15:
            bx1, by1 = frame_boxes[tid][0], frame_boxes[tid][1]
            cv2.putText(annotated_frame, f"{engine.kinematics(tid)['speed_kmh']:.0f} km/h",
                        (bx1 + 70, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for i, a in enumerate(active_alerts[-BANNER_MAX:]):
        text = f"{ALERT_LABELS[a['type']]} {a['message']}"
        y = 60 + 28 * i
        cv2.rectangle(annotated_frame, (annotated_frame.shape[1] - 640, y - 20),
                      (annotated_frame.shape[1] - 10, y + 6), (0, 0, 0), -1)
        cv2.putText(annotated_frame, text, (annotated_frame.shape[1] - 632, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, alert_color(a), 2)

    plotted = len(vehicles)
    dropped = total_tracked - plotted
    if new_frame:
        frame_times.append(time.perf_counter())
    proc_fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0.0
    info_text = f"Tracked: {total_tracked}  Plotted: {plotted}  Dropped: {dropped}  {proc_fps:.1f} fps"
    cv2.putText(annotated_frame, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(canvas, info_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    if RECORD:
        async_writer.write(source_writer, annotated_frame)
        async_writer.write(twin_writer, canvas)

    if not ARGS.no_show:
        cv2.imshow("Tracking (source)", annotated_frame)
        cv2.imshow("Digital Twin", canvas)

    if UPDATE_EVERY and new_frame and frame_idx % UPDATE_EVERY == 0:
        xs = [v["X"] for v in vehicles]
        ys = [v["Y"] for v in vehicles]
        zs = [Z_PLANE] * len(vehicles)
        scat3d._offsets3d = (xs, ys, zs)
        scat3d.set_facecolor(
            [CLASS_RGBA.get(track_class.get(v["id"], ""), DEFAULT_RGBA) for v in vehicles]
        )
        title3d.set_text(f"Live 3D Digital Twin (frame {frame_idx})")

        fig3d.canvas.draw()
        fig3d.canvas.flush_events()

        graph_rgba = np.asarray(fig3d.canvas.buffer_rgba())
        graph_bgr = cv2.cvtColor(graph_rgba, cv2.COLOR_RGBA2BGR)
        if graph_writer is None:
            gh, gw = graph_bgr.shape[:2]
            graph_writer = cv2.VideoWriter(
                GRAPH_OUTPUT_PATH,
                cv2.VideoWriter_fourcc(*"XVID"),
                RECORD_FPS / UPDATE_EVERY,
                (gw, gh),
            )
            print(f"Recording 3D view to {GRAPH_OUTPUT_PATH} ({gw}x{gh})")
        async_writer.write(graph_writer, graph_bgr)
    elif UPDATE_EVERY and not ARGS.no_show:
        fig3d.canvas.flush_events()

    key = 255 if ARGS.no_show else cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        stop_event.set()
        break
    elif key == ord(" "):
        paused = not paused
    elif key == ord("n") and paused:
        try:
            frame = frame_queue.get(timeout=1.0)
            if frame is not None:
                last_frame = frame
        except Empty:
            pass

stop_event.set()
try:
    reader_future.result(timeout=3.0)
except Exception:
    pass
executor.shutdown(wait=False)
cap.release()
async_writer.close()
if len(frame_times) > 1:
    print(f"Processing speed: {(len(frame_times) - 1) / (frame_times[-1] - frame_times[0]):.1f} frames/s "
          f"over {len(frame_times)} frames")
if RECORD:
    source_writer.release()
    twin_writer.release()
    print(f"Saved {SOURCE_OUTPUT_PATH} and {TWIN_OUTPUT_PATH}")
if graph_writer is not None:
    graph_writer.release()
    print(f"Saved {GRAPH_OUTPUT_PATH}")

pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)
print(f"Logged {len(log_rows)} rows to {LOG_PATH}")

for a in engine.finish():
    db.save_alert(a)
for a in engine.alerts:
    db.save_alert(a)
db.close()
alert_rows = [{**a, "tracks": "-".join(map(str, a["tracks"]))} for a in engine.alerts]
pd.DataFrame(alert_rows).to_csv(ARGS.alerts, index=False)
counts = pd.Series([f"{a['type']}/{a['severity']}" for a in engine.alerts]).value_counts()
print(f"Raised {len(engine.alerts)} alerts -> {ARGS.alerts} and {ARGS.db}")
for k, n in counts.items():
    print(f"  {k:32s} {n}")

cv2.destroyAllWindows()
plt.close(fig3d)
if device == "cuda":
    torch.cuda.empty_cache()
