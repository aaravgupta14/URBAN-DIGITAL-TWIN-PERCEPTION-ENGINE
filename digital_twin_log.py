
import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty, Full
from threading import Event
from collections import deque
import os
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

VIDEO_PATH = r"C:\Users\Aarav Gupta\OneDrive\Desktop\DIGITAL_TWIN\dataset\Weast (1).mp4"
HOMOGRAPHY_PATH = "homography.npy"

OUTPUT_WIDTH = 840
OUTPUT_HEIGHT = 306
PAD_LEFT = 16
PAD_RIGHT = 96
PAD_TOP = 80
PAD_BOTTOM = 16
CANVAS_WIDTH = OUTPUT_WIDTH + PAD_LEFT + PAD_RIGHT
CANVAS_HEIGHT = OUTPUT_HEIGHT + PAD_TOP + PAD_BOTTOM
UNITS_PER_METER = 120
PROXIMITY_THRESHOLD = int(2.5 * UNITS_PER_METER)
ARROW_LENGTH = 20
MIN_MOVEMENT_FOR_ARROW = 2
MIN_TRACK_AGE = 2
STEP_HISTORY_LEN = 8
JUMP_ANOMALY_MULT = 6
JUMP_ANOMALY_MIN_ABS = int(0.125 * UNITS_PER_METER)
SMOOTHING_ALPHA = 0.4

FRAME_QUEUE_MAX = 4

RECORD = True
SOURCE_OUTPUT_PATH = "output_tracking_source.avi"
TWIN_OUTPUT_PATH = "output_digital_twin.avi"
RECORD_FPS = 30.0

LOG_PATH = "tracking_log.csv"
VIDEO_LABEL = "weast"

CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CLASS_COLORS = {"car": "tab:blue", "motorcycle": "tab:orange", "bus": "tab:green", "truck": "tab:red"}

Z_PLANE = 0.0
Z_LIM = 1.0

CLASS_RGBA = {name: to_rgba(c) for name, c in CLASS_COLORS.items()}
DEFAULT_RGBA = to_rgba("gray")

UPDATE_EVERY = 2
GRAPH_OUTPUT_PATH = "output_digital_twin_3d.avi"

H = np.load(HOMOGRAPHY_PATH)
H_inv = np.linalg.inv(H)
_canvas_corners = np.float32(
    [[0, 0], [OUTPUT_WIDTH, 0], [OUTPUT_WIDTH, OUTPUT_HEIGHT], [0, OUTPUT_HEIGHT]]
).reshape(-1, 1, 2)
_src_corners = cv2.perspectiveTransform(_canvas_corners, H_inv).reshape(-1, 2)
CALIB_TOP_Y = float(np.min(_src_corners[:, 1]))
CALIB_BOTTOM_Y = float(np.max(_src_corners[:, 1]))
LOW_CONF_Y = CALIB_TOP_Y + 0.35 * (CALIB_BOTTOM_Y - CALIB_TOP_Y)
print(f"Calibrated image-y range: {CALIB_TOP_Y:.0f} - {CALIB_BOTTOM_Y:.0f} "
      f"(low-confidence below y={LOW_CONF_Y:.0f})")

model = YOLO("yolo11s.pt")
model.to(device)
dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
print("Warming up GPU...")
for _ in range(5):
    with torch.inference_mode():
        model.predict(dummy, device=0, half=True, verbose=False)
if device == "cuda":
    torch.cuda.synchronize()
print("Warm-up complete.")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

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
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
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
frame_idx = 0

plt.ion()
fig3d = plt.figure(figsize=(6, 5))
ax3d = fig3d.add_subplot(111, projection="3d")

ax3d.set_xlabel("X (across road)")
ax3d.set_ylabel("Y (0 = far)")
ax3d.set_zlabel("Z (road plane)")
ax3d.set_xlim(0, CANVAS_WIDTH)
ax3d.set_ylim(0, CANVAS_HEIGHT)
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
            tracker="tracktrack_reid_loose.yaml",
            classes=[2, 3, 5, 7],
            imgsz=1280,
            conf=0.1,
            device=0,
            half=False,
            verbose=False,
        )

    boxes = results[0].boxes
    annotated_frame = frame.copy()
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), (40, 40, 40), dtype=np.uint8)

    vehicles = []
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
        pts = np.stack([cx_arr, cy_arr], axis=1).astype(np.float32).reshape(-1, 1, 2)
        world_pts = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

        for tid in ids:
            track_streak[int(tid)] = track_streak.get(int(tid), 0) + 1

        for idx, track_id in enumerate(ids):
            track_id = int(track_id)
            tx1, ty1 = int(x1[idx]), int(y1[idx])
            tx2, ty2 = int(x2[idx]), int(y2[idx])
            cx, cy = int(cx_arr[idx]), int(cy_arr[idx])
            X, Y = world_pts[idx]
            low_conf = cy_arr[idx] < LOW_CONF_Y

            cv2.rectangle(annotated_frame, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
            cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                annotated_frame, f"ID:{track_id}", (tx1, ty1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

            in_bounds = (
                -PAD_LEFT <= X < OUTPUT_WIDTH + PAD_RIGHT
                and -PAD_TOP <= Y < OUTPUT_HEIGHT + PAD_BOTTOM
            )
            stable = track_streak[track_id] >= MIN_TRACK_AGE

            log_rows.append({
                "source": VIDEO_LABEL,
                "frame_idx": frame_idx,
                "track_id": track_id,
                "class_id": int(cls_arr[idx]),
                "conf": float(conf_arr[idx]),
                "X": float(X),
                "Y": float(Y),
                "low_conf": bool(low_conf),
                "in_bounds": bool(in_bounds),
                "stable": bool(stable),
            })

            if in_bounds and stable:
                prev_smooth = smoothed_positions.get(track_id, (X, Y))
                sX = SMOOTHING_ALPHA * X + (1 - SMOOTHING_ALPHA) * prev_smooth[0]
                sY = SMOOTHING_ALPHA * Y + (1 - SMOOTHING_ALPHA) * prev_smooth[1]
                smoothed_positions[track_id] = (sX, sY)
                cls_name = CLASS_NAMES.get(int(cls_arr[idx]), "vehicle")
                track_class[track_id] = cls_name
                vehicles.append({
                    "id": track_id,
                    "X": sX + PAD_LEFT,
                    "Y": sY + PAD_TOP,
                    "Z": Z_PLANE,
                    "low_conf": low_conf,
                })

    for v in vehicles:

        color = (0, 165, 255) if v["low_conf"] else (0, 255, 0)
        cv2.circle(canvas, (int(v["X"]), int(v["Y"])), 7, color, -1)
        cv2.putText(
            canvas, f"ID:{v['id']}", (int(v["X"]) + 10, int(v["Y"]) - 10),
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
                    tip = (int(v["X"] + ux * ARROW_LENGTH), int(v["Y"] + uy * ARROW_LENGTH))
                    cv2.arrowedLine(canvas, (int(v["X"]), int(v["Y"])), tip, (255, 255, 0), 2, tipLength=0.35)
                history.append(mag)

    prev_positions = {v["id"]: (v["X"], v["Y"]) for v in vehicles}

    if len(vehicles) >= 2:
        Xs = [v["X"] for v in vehicles]
        Ys = [v["Y"] for v in vehicles]
        for i, j, dist in compute_close_pairs(Xs, Ys, PROXIMITY_THRESHOLD):
            v1, v2 = vehicles[i], vehicles[j]
            p1 = (int(v1["X"]), int(v1["Y"]))
            p2 = (int(v2["X"]), int(v2["Y"]))
            cv2.line(canvas, p1, p2, (0, 0, 255), 2)
            mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            cv2.putText(
                canvas, format_distance(dist), (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

    plotted = len(vehicles)
    dropped = total_tracked - plotted
    info_text = f"Tracked: {total_tracked}  Plotted: {plotted}  Dropped: {dropped}"
    cv2.putText(annotated_frame, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(canvas, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if RECORD:
        source_writer.write(annotated_frame)
        twin_writer.write(canvas)

    cv2.imshow("Tracking (source)", annotated_frame)
    cv2.imshow("Digital Twin", canvas)

    if new_frame and frame_idx % UPDATE_EVERY == 0:
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
        graph_writer.write(graph_bgr)
    else:
        fig3d.canvas.flush_events()

    key = cv2.waitKey(1) & 0xFF
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
if RECORD:
    source_writer.release()
    twin_writer.release()
    print(f"Saved {SOURCE_OUTPUT_PATH} and {TWIN_OUTPUT_PATH}")
if graph_writer is not None:
    graph_writer.release()
    print(f"Saved {GRAPH_OUTPUT_PATH}")

pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)
print(f"Logged {len(log_rows)} rows to {LOG_PATH}")

cv2.destroyAllWindows()
plt.close(fig3d)
if device == "cuda":
    torch.cuda.empty_cache()
