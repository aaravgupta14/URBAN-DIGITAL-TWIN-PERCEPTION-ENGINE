import numpy as np

prev_left = None
prev_right = None


def extract_road_boundaries(road_mask):

    global prev_left, prev_right

    h, w = road_mask.shape

    left_points = []
    right_points = []

    SEARCH = 60

    for y in range(h - 1, 0, -3):

        xs = np.where(road_mask[y] > 0)[0]

        if len(xs) < 20:
            continue

        if prev_left is None:

            left = xs[0]
            right = xs[-1]

        else:

            left = prev_left
            right = prev_right

            left_near = xs[np.abs(xs - prev_left) < SEARCH]

            if len(left_near):

                left = left_near[0]

            right_near = xs[np.abs(xs - prev_right) < SEARCH]

            if len(right_near):

                right = right_near[-1]

        alpha = 0.35

        if prev_left is not None:

            left = int(alpha * left + (1 - alpha) * prev_left)

            right = int(alpha * right + (1 - alpha) * prev_right)

        prev_left = left
        prev_right = right

        left_points.append((left, y))
        right_points.append((right, y))

    left_points.reverse()
    right_points.reverse()

    return left_points, right_points