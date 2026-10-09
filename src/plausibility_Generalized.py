MIN_PLAUSIBLE_DIST_M = 0.10
MIN_PLAUSIBLE_SPEED_MS = 2.0 / 3.6
URBAN_SPEED_RANGE_KMH = (10.0, 60.0)
MAX_PLAUSIBLE_ACCEL_MS2 = 4.0

def is_implausible_distance(dist_m):
    return dist_m < MIN_PLAUSIBLE_DIST_M

def is_implausible_speed(speed_ms):
    return speed_ms < MIN_PLAUSIBLE_SPEED_MS

def is_urban_speed(speed_ms):
    lo, hi = URBAN_SPEED_RANGE_KMH
    return lo <= speed_ms * 3.6 <= hi

def is_implausible_accel(accel_ms2):
    return accel_ms2 > MAX_PLAUSIBLE_ACCEL_MS2
