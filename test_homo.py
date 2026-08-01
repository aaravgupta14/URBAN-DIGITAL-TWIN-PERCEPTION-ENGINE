import cv2
import numpy as np

VIDEO_PATH = r"C:\Users\Aarav Gupta\OneDrive\Desktop\DIGITAL_TWIN\dataset\Weast (1).mp4"

H = np.load("homography.npy")

OUTPUT_WIDTH = 950
OUTPUT_HEIGHT = 750

cap = cv2.VideoCapture(VIDEO_PATH)

paused = False
frame = None

canvas = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
canvas[:] = (40, 40, 40)


def mouse(event, x, y, flags, param):

    global canvas, frame

    if event == cv2.EVENT_LBUTTONDOWN:

        display = frame.copy()

        cv2.circle(display, (x, y), 6, (0, 0, 255), -1)

        pt = np.array([[[x, y]]], dtype=np.float32)

        warped = cv2.perspectiveTransform(pt, H)

        X = int(warped[0][0][0])
        Y = int(warped[0][0][1])

        canvas = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
        canvas[:] = (40, 40, 40)

        if 0 <= X < OUTPUT_WIDTH and 0 <= Y < OUTPUT_HEIGHT:

            cv2.circle(canvas, (X, Y), 8, (0, 255, 0), -1)

            cv2.putText(
                canvas,
                f"({X}, {Y})",
                (X + 10, Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        cv2.imshow("Original", display)
        cv2.imshow("Digital Twin", canvas)


cv2.namedWindow("Original")
cv2.setMouseCallback("Original", mouse)

while True:

    if not paused:

        ret, frame = cap.read()

        if not ret:
            break

    cv2.imshow("Original", frame)
    cv2.imshow("Digital Twin", canvas)

    key = cv2.waitKey(30) & 0xFF

    if key == 27:
        break

    elif key == ord(' '):
        paused = not paused

    elif key == ord('n') and paused:
        ret, frame = cap.read()
        if not ret:
            break

cap.release()
cv2.destroyAllWindows()