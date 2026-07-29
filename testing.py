import cv2

from road_seg import segment_road
from road_polygon import extract_road_polygon

video_path = r"C:\Users\Aarav Gupta\OneDrive\Desktop\DIGITAL_TWIN\dataset\Weast (1).mp4"

cap = cv2.VideoCapture(video_path)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    clean_mask, hull = segment_road(frame)

    corners, polygon = extract_road_polygon(
        clean_mask,
        frame
    )

    cv2.imshow("Road Mask", clean_mask)

    cv2.imshow("Road Polygon", polygon)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()