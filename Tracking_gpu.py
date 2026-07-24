import cv2
import numpy as np
import torch
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty, Full
from threading import Event
from collections import deque

cv2.setNumThreads(1)
torch.set_num_threads(1)
torch.set_float32_matmul_precision("high")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    torch.backends.cudnn.benchmark = True

model = YOLO("yolo11s.pt")
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
model.to(device)
dummy = np.zeros((720, 1200, 3), dtype=np.uint8)

print("Warming up GPU...")

for _ in range(5):
    with torch.inference_mode():
        model.predict(
            dummy,
            device=0,
            half=True,
            verbose=False
        )

torch.cuda.synchronize()

print("Warm-up complete.")

video_path = r"C:\Users\Aarav Gupta\OneDrive\Desktop\DIGITAL_TWIN\dataset\Weast (1).mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {video_path}")

FRAME_QUEUE_MAX = 4
DISTANCE_THRESHOLD_PX = 300  # on the resized 1200x720 frame

frame_queue: "Queue" = Queue(maxsize=FRAME_QUEUE_MAX)
stop_event = Event()
track_history = {}


def video_reader():
    """Decode frames on a separate thread so it doesn't block GPU inference."""
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        # Block until the consumer has room, so the reader can't race ahead
        # and dump the whole file before inference even gets going.
        while not stop_event.is_set():
            try:
                frame_queue.put(frame, timeout=0.5)
                break
            except Full:
                continue
    frame_queue.put(None)  # signal end of stream


def preprocess_frame(frame):
    return frame


def compute_close_pairs(cx, cy, threshold):
    """Vectorized pairwise Euclidean distance. Returns (i, j, dist) for every pair under threshold."""
    n = len(cx)
    if n < 2:
        return []
    cx = np.asarray(cx, dtype=np.float64)
    cy = np.asarray(cy, dtype=np.float64)
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist_matrix = np.hypot(dx, dy)
    i_idx, j_idx = np.triu_indices(n, k=1)  
    dists = dist_matrix[i_idx, j_idx]
    mask = dists < threshold
    return list(zip(i_idx[mask].tolist(), j_idx[mask].tolist(), dists[mask].tolist()))


executor = ThreadPoolExecutor(max_workers=1)
reader_future = executor.submit(video_reader)

while True:
    try:
        frame = frame_queue.get(timeout=1.0)
        print("Frame received")
    except Empty:
        continue
    if frame is None:
        break

    frame = preprocess_frame(frame)

    with torch.no_grad():

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[2,3,5,7],
            imgsz=1280,
            conf=0.25,
            device=0,
            half=False,
            verbose=False
        )

    boxes = results[0].boxes
    annotated_frame = frame.copy()
    vehicles = []

    if boxes.id is not None:
        ids = boxes.id.cpu().numpy().astype(np.int32)
        xyxy = boxes.xyxy.cpu().numpy()
        x1 = xyxy[:, 0].astype(np.int32)
        y1 = xyxy[:, 1].astype(np.int32)
        x2 = xyxy[:, 2].astype(np.int32)
        y2 = xyxy[:, 3].astype(np.int32)

        cx_arr = (x1 + x2) // 2
        cy_arr = y2  # bottom-center

        for idx, track_id in enumerate(ids):
            cx, cy = (cx_arr[idx]), int(cy_arr[idx])
            tx1, ty1 = int(x1[idx]), int(y1[idx])

            vehicles.append({"id": track_id, "cx": cx, "cy": cy})            
            history = track_history.setdefault(
                track_id,
                deque(maxlen=30)
        )

            history.append((cx,cy))

            cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                annotated_frame, f"ID:{track_id}", (tx1, ty1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

        if len(vehicles) >= 2:
            close_pairs = compute_close_pairs(cx_arr, cy_arr, DISTANCE_THRESHOLD_PX)
            for i, j, distance in close_pairs:
                v1, v2 = vehicles[i], vehicles[j]
                cv2.line(annotated_frame, (v1["cx"], v1["cy"]), (v2["cx"], v2["cy"]), (0, 0, 255), 2)
                mx = (v1["cx"] + v2["cx"]) // 2
                my = (v1["cy"] + v2["cy"]) // 2
                cv2.putText(
                    annotated_frame, f"{int(distance)} px", (mx, my),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                )

    cv2.imshow("Tracking + Pixel Distance", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        stop_event.set()
        break

stop_event.set()
try:
    reader_future.result(timeout=3.0)
except Exception:
    pass
executor.shutdown(wait=False)
cap.release()
cv2.destroyAllWindows()

if device == "cuda":
    torch.cuda.empty_cache()