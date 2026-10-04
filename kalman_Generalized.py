import numpy as np

def constant_acceleration_kf(times, positions, q_var=1.0, r_var=0.05, smooth=True):
    n = len(times)
    H = np.array([[1.0, 0.0, 0.0]])
    R = np.array([[r_var]])

    s = np.array([positions[0], 0.0, 0.0])
    P = np.eye(3) * 10.0

    filtered_s, filtered_P, predicted_s, predicted_P, transitions = [], [], [], [], []
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
            transitions.append(F)
        predicted_s.append(s)
        predicted_P.append(P)

        z = np.array([positions[i]])
        y = z - H @ s
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        s = s + (K @ y)
        P = (np.eye(3) - K @ H) @ P

        filtered_s.append(s)
        filtered_P.append(P)

    states = list(filtered_s)
    if smooth:
        for i in range(n - 2, -1, -1):
            F = transitions[i]
            C = filtered_P[i] @ F.T @ np.linalg.inv(predicted_P[i + 1])
            states[i] = filtered_s[i] + C @ (states[i + 1] - predicted_s[i + 1])

    return [(float(x[0]), float(x[1]), float(x[2])) for x in states]
