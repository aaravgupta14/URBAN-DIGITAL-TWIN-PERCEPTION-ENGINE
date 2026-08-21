import cv2
import numpy as np


class HomographyEstimator:
    def __init__(self, frame_width=600, frame_height=800, window_size=300):
        self.width = frame_width
        self.height = frame_height
        self.window_size = window_size
        self.width_ratio = 0.60
        self.tl = []
        self.tr = []
        self.bl = []
        self.br = []
        self.valid_frames = 0

    def fit_boundary(self, points):
        if len(points) < 30:
            return None
        pts = np.array(points)
        x = pts[:, 0]
        y = pts[:, 1]
        return np.polyfit(y, x, 1)

    def evaluate_curve(self, coeff, y):
        return np.polyval(coeff, y)

    def add_frame(self, frame, left_points, right_points):
        left_curve = self.fit_boundary(left_points)
        right_curve = self.fit_boundary(right_points)

        if left_curve is None or right_curve is None:
            return None

        pts = np.array(left_points)
        image_height = frame.shape[0]

        top_boundary = int(np.min(pts[:, 1]))
        bottom_y = int(image_height * 0.95)

        bottom_width = (
            np.polyval(right_curve, bottom_y)
            - np.polyval(left_curve, bottom_y)
        )

        top_y = None

        for y in range(top_boundary, bottom_y):
            left_x = np.polyval(left_curve, y)
            right_x = np.polyval(right_curve, y)
            width = right_x - left_x

            if width >= self.width_ratio * bottom_width:
                top_y = y
                break

        if top_y is None:
            return None

        tl = (self.evaluate_curve(left_curve, top_y), top_y)
        tr = (self.evaluate_curve(right_curve, top_y), top_y)
        bl = (self.evaluate_curve(left_curve, bottom_y), bottom_y)
        br = (self.evaluate_curve(right_curve, bottom_y), bottom_y)

        top_width = tr[0] - tl[0]
        bottom_width = br[0] - bl[0]

        if tl[0] >= tr[0]:
            return None

        if bl[0] >= br[0]:
            return None

        if top_width < 60:
            return None

        if bottom_width < 200:
            return None

        if bottom_width < top_width:
            return None

        self.tl.append(tl)
        self.tr.append(tr)
        self.bl.append(bl)
        self.br.append(br)

        self.valid_frames += 1

        if self.valid_frames < self.window_size:
            return None

        H = self.compute_homography()

        self.reset()

        return H

    def compute_homography(self):
        tl = np.median(np.array(self.tl), axis=0)
        tr = np.median(np.array(self.tr), axis=0)
        bl = np.median(np.array(self.bl), axis=0)
        br = np.median(np.array(self.br), axis=0)

        print("===================================")
        print("TL:", tl)
        print("TR:", tr)
        print("BL:", bl)
        print("BR:", br)
        print("===================================")

        src = np.float32([tl, tr, br, bl])

        dst = np.float32([
            [0, 0],
            [self.width, 0],
            [self.width, self.height],
            [0, self.height]
        ])

        H = cv2.getPerspectiveTransform(src, dst)

        np.save("homography.npy", H)

        print("Homography Updated")

        return H

    def reset(self):
        self.tl.clear()
        self.tr.clear()
        self.bl.clear()
        self.br.clear()
        self.valid_frames = 0

    def warp(self, frame, H):
        return cv2.warpPerspective(
            frame,
            H,
            (self.width, self.height)
        )