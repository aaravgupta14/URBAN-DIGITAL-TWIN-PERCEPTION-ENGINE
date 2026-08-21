from ultralytics import YOLO
import cv2

video_path = r"C:\Users\Aarav Gupta\OneDrive\Desktop\DIGITAL_TWIN\dataset\Weast (1).mp4"

cap = cv2.VideoCapture(video_path)
model =YOLO("yolo11n.pt")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Detection", annotated_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()