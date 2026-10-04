import numpy as np

def constant_acceleration_kf(times, positions, q_var=1.0, r_var=0.05):
    n = len(times)
    H = np.array([[1.0, 0.0, 0.0]])
    R = np.array([[r_var]])

    s = np.array([positions[0], 0.0, 0.0])
    P = np.eye(3) * 10.0

    smoothed = []
    for i in range(n):
        if i > 0:
            dt = times[i] - times[i - 1]
            if dt <= 0:
                dt = 1e-3
            F = np.array([
                [1.0, dt, 0.5 * dt * dt],
                [0.0, 1.0, dt],
                [0.0, 0.0, 1.0],
            ])
            Q = q_var * np.array([
                [dt**4 / 4, dt**3 / 2, dt**2 / 2],
                [dt**3 / 2, dt**2,     dt],
                [dt**2 / 2, dt,        1.0],
            ])
            s = F @ s
            P = F @ P @ F.T + Q

        z = np.array([positions[i]])
        y = z - H @ s
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        s = s + (K @ y)
        P = (np.eye(3) - K @ H) @ P

        smoothed.append((float(s[0]), float(s[1]), float(s[2])))
    return smoothed
