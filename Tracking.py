import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open video
cap = cv2.VideoCapture(r"C:\Users\Aarav Gupta\OneDrive\Desktop\DIGITAL_TWIN\dataset\Weast (1).mp4"
)

# Dictionary to store trajectories
track_history = {}

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # Run tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
    )

    # Get detections
    boxes = results[0].boxes

    # Draw results
    annotated_frame = results[0].plot()

    if boxes.id is not None:

        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy()

        for box, track_id in zip(xyxy, ids):

            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if track_id not in track_history:
                track_history[track_id] = []

            track_history[track_id].append((cx, cy))
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)
            points = track_history[track_id]

            cv2.putText(
                annotated_frame,
                f"ID:{track_id}",
                (x1, y1 -30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    cv2.imshow("ByteTrack", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()