import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# ---------- Track class ----------

class Track:
    count = 0  # class-level counter

    def __init__(self, detection, timestamp):
        self.id = Track.count
        Track.count += 1

        self.kf = self._init_kalman(detection)
        self.time_since_update = 0
        self.last_update_time = timestamp
        self.history = [detection]

    def _init_kalman(self, detection):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0  # frame time step
        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1,  0],
                         [0, 0, 0,  1]])  # state transition

        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])    # measurement function

        kf.R *= 0.5   # measurement noise
        kf.P *= 10.0  # initial uncertainty
        kf.Q *= 0.1   # process noise

        kf.x[:2] = detection.reshape(2, 1)
        return kf

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.x[:2].reshape(-1)  # return [x, y]

    def update(self, detection, timestamp):
        self.kf.update(detection)
        self.history.append(detection)
        self.time_since_update = 0
        self.last_update_time = timestamp

    def get_state(self):
        return self.kf.x[:2].reshape(-1)

# ---------- Tracker Logic ----------

def compute_cost(tracks, detections):
    cost = np.zeros((len(tracks), len(detections)))
    for i, track in enumerate(tracks):
        pred = track.predict()
        for j, det in enumerate(detections):
            cost[i, j] = np.linalg.norm(pred - det)
    return cost

def associate(cost_matrix, max_dist=50):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, unmatched_tracks, unmatched_dets = [], set(range(cost_matrix.shape[0])), set(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < max_dist:
            matches.append((r, c))
            unmatched_tracks.remove(r)
            unmatched_dets.remove(c)

    return matches, list(unmatched_tracks), list(unmatched_dets)

# ---------- Simulated Input + Main ----------

def simulate_detections(n_frames):
    np.random.seed(0)
    pos = np.array([[100, 50], [300, 400]])
    vel = np.array([[2, 1], [-1, -1.5]])
    noise = 3

    frames = []
    for i in range(n_frames):
        detections = pos + i * vel + np.random.randn(*pos.shape) * noise
        frames.append(detections)
    return frames

# Main tracking loop
frames = simulate_detections(30)
tracks = []
max_age = 5

for t, detections in enumerate(frames):
    detections = np.array(detections)
    cost = compute_cost(tracks, detections) if tracks else np.zeros((0, len(detections)))
    matches, unmatched_tracks, unmatched_dets = associate(cost)

    # Update matched
    for i, j in matches:
        tracks[i].update(detections[j], t)

    # Create new tracks
    for j in unmatched_dets:
        tracks.append(Track(detections[j], t))

    # Delete old tracks
    tracks = [track for track in tracks if track.time_since_update <= max_age]

# ---------- Plotting ----------
plt.figure(figsize=(10, 6))
for track in tracks:
    hist = np.array(track.history)
    plt.plot(hist[:, 0], hist[:, 1], label=f'Track {track.id}')
    plt.scatter(hist[-1, 0], hist[-1, 1])

plt.title("Tracked Targets Over Time")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.legend()
plt.show()
