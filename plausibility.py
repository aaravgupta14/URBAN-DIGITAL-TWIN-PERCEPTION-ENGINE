MIN_PLAUSIBLE_DIST_M = 0.10
MIN_PLAUSIBLE_SPEED_MS = 2.0 / 3.6

def is_implausible_distance(dist_m):
    return dist_m < MIN_PLAUSIBLE_DIST_M

def is_implausible_speed(speed_ms):
    return speed_ms < MIN_PLAUSIBLE_SPEED_MS
