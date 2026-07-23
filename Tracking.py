import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# Load Model
# -----------------------------
model = YOLO("yolo11s.pt")

# -----------------------------
# Open Video
# -----------------------------
cap = cv2.VideoCapture(
    r"C:\Users\Aarav Gupta\OneDrive\Desktop\DIGITAL_TWIN\dataset\Weast (1).mp4"
)

track_history = {}

# Create CLAHE once
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (1200, 720))

    # -----------------------------
    # CLAHE Enhancement
    # -----------------------------
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # -----------------------------
    # Tracking
    # -----------------------------
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[2, 3, 5, 7],
        imgsz=1280,
        conf=0.25,
        verbose=False
    )

    boxes = results[0].boxes
    annotated_frame = results[0].plot()

    vehicles = []

    if boxes.id is not None:

        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy()

        for box, track_id in zip(xyxy, ids):

            x1, y1, x2, y2 = map(int, box)

            # Bottom-center point
            cx = (x1 + x2) // 2
            cy = y2

            vehicles.append({
                "id": track_id,
                "cx": cx,
                "cy": cy
            })

            # Keep history if needed later
            if track_id not in track_history:
                track_history[track_id] = []

            track_history[track_id].append((cx, cy))

            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)

            # Bottom-center point
            cv2.circle(
                annotated_frame,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                annotated_frame,
                f"ID:{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

    # -----------------------------
    # Pixel Distance Between Vehicles
    # -----------------------------
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):

            v1 = vehicles[i]
            v2 = vehicles[j]

            distance = np.sqrt(
                (v1["cx"] - v2["cx"]) ** 2 +
                (v1["cy"] - v2["cy"]) ** 2
            )

            if distance < 300:

                cv2.line(
                    annotated_frame,
                    (v1["cx"], v1["cy"]),
                    (v2["cx"], v2["cy"]),
                    (0, 0, 255),
                    2
                )

                mx = (v1["cx"] + v2["cx"]) // 2
                my = (v1["cy"] + v2["cy"]) // 2

                cv2.putText(
                    annotated_frame,
                    f"{int(distance)} px",
                    (mx, my),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

    # -----------------------------
    # Display
    # -----------------------------
    cv2.imshow("Tracking + Pixel Distance", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
