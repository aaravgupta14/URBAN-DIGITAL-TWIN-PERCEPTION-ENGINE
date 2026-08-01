import cv2
import numpy as np

IMAGE_PATH = r"C:\Users\Aarav Gupta\OneDrive\Desktop\DIGITAL_TWIN\calibration_frame.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not found!")
    exit()

display = image.copy()
points = []


def redraw():

    global display

    display = image.copy()

    for i, p in enumerate(points):

        cv2.circle(display, p, 6, (0, 0, 255), -1)

        cv2.putText(
            display,
            str(i + 1),
            (p[0] + 10, p[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    if len(points) >= 2:
        cv2.line(display, points[0], points[1], (0, 255, 255), 2)

    if len(points) >= 3:
        cv2.line(display, points[1], points[2], (0, 255, 255), 2)

    if len(points) == 4:
        cv2.line(display, points[2], points[3], (0, 255, 255), 2)
        cv2.line(display, points[3], points[0], (0, 255, 255), 2)

    cv2.imshow("Calibration", display)


def mouse(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(points) < 4:

            points.append((x, y))
            redraw()

    elif event == cv2.EVENT_RBUTTONDOWN:

        if len(points):

            points.pop()
            redraw()


cv2.namedWindow("Calibration")
cv2.setMouseCallback("Calibration", mouse)

print()
print("Click in this order:")
print("1 -> Top Left")
print("2 -> Top Right")
print("3 -> Bottom Right")
print("4 -> Bottom Left")
print()
print("Right Click = Undo")
print("ESC = Exit")
print()

redraw()

while True:

    key = cv2.waitKey(1)

    if key == 27:
        cv2.destroyAllWindows()
        exit()

    if len(points) == 4:
        break

src = np.float32(points)

width_top = np.linalg.norm(src[1] - src[0])
width_bottom = np.linalg.norm(src[2] - src[3])

height_left = np.linalg.norm(src[3] - src[0])
height_right = np.linalg.norm(src[2] - src[1])

OUTPUT_WIDTH = int(max(width_top, width_bottom))
OUTPUT_HEIGHT = int(max(height_left, height_right))

dst = np.float32([
    [0, 0],
    [OUTPUT_WIDTH, 0],
    [OUTPUT_WIDTH, OUTPUT_HEIGHT],
    [0, OUTPUT_HEIGHT]
])

H = cv2.getPerspectiveTransform(src, dst)

np.save("homography.npy", H)

bird = cv2.warpPerspective(
    image,
    H,
    (OUTPUT_WIDTH, OUTPUT_HEIGHT),
    flags=cv2.INTER_LANCZOS4,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0)
)

cv2.imshow("Bird Eye", bird)

print("\nHomography saved as homography.npy")
print(f"Output Size : {OUTPUT_WIDTH} x {OUTPUT_HEIGHT}")

cv2.waitKey(0)
cv2.destroyAllWindows()