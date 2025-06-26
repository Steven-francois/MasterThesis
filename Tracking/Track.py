import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

class Track:
    count = 0
    
    def __init__(self, detection, ts):
        self.id = Track.count
        Track.count += 1
        self. history = [detection]
        self.time_since_update = 0
        self.last_update_time = ts
        self.predictions = []
        
    def predict(self, new_timestamp):
        self.time_since_update += 1
        
        # Radar prediction
        radar_coord = self.history[-1][2]
        dt = new_timestamp - self.last_update_time
        new_radar_coord = (radar_coord[0] + radar_coord[1] * dt, radar_coord[1])
        prediction = {
            "radar_coord": new_radar_coord,
            "timestamp": new_timestamp
        }
        return prediction
    
    def update(self, detection, ts):
        self.history.append(detection)
        self.time_since_update = 0
        self.last_update_time = ts

class TrackRadar(Track):
    def __init__(self, detection, ts):
        super().__init__(detection, ts)
        self.radar_coord = detection[2]  # Assuming detection[2] contains radar coordinates

        # Initialization of Kalman filter
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0  # frame time step
        kf.F = np.array([[1, dt, 0, 0],
                         [0, 1, 0,  0],
                         [0, 0, 1, dt],
                         [0, 0, 0,  1]])  # state transition

        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])    # measurement function

        kf.R *= 0.9   # measurement noise
        kf.P *= 10.0  # initial uncertainty
        kf.Q *= 0.1   # process noise

        kf.x[:2] = np.array(self.radar_coord).reshape(2, 1)
        self.kf = kf

    def __state_transition_dt(self, dt):
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1,  0],
                              [0, 0, 0,  1]])  # state transition

    def predict(self, new_timestamp, kalman=False):
        self.time_since_update += 1
        dt = new_timestamp - self.last_update_time
        
        # Radar prediction
        if kalman:
            self.__state_transition_dt(dt)
            self.kf.predict()
            self.time_since_update += 1
            prediction = {
                "radar_coord": self.kf.x[:2].reshape(-1),  # return [x, y]
                "timestamp": new_timestamp
            }
        else:
            new_radar_coord = (self.radar_coord[0] + self.radar_coord[1] * dt, self.radar_coord[1])
            prediction = {
                "radar_coord": new_radar_coord,
                "timestamp": new_timestamp
            }
        self.predictions.append(prediction["radar_coord"])
        return prediction
    
    def update(self, detection, ts):
        super().update(detection, ts)
        self.radar_coord = detection[2]
        self.kf.update(np.array(self.radar_coord).reshape(2, 1))  # Update Kalman filter with new measurement

class TrackLidar(Track):
    def __init__(self, detection, ts):
        super().__init__(detection, ts)
        self.lidar_coord = detection[1]

# --- Track management functions
def compute_cost(tracks, detections):
    cost = np.zeros((len(tracks), len(detections["targets_nb"])))
    for i, track in enumerate(tracks):
        pred = track.predict(detections["timestamp"], True)
        for j in range(len(detections["targets_nb"])):
            cost[i, j] = np.linalg.norm(np.array(pred["radar_coord"]) - np.array(detections["targets"][j][2]))
    return cost

def associate_tracks_and_detections(tracks, detections, max_age=5):
    cost_matrix = compute_cost(tracks, detections)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_detections = set(range(len(detections["targets_nb"])))

    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < max_age:
            matches.append((i, j))
            unmatched_tracks.discard(i)
            unmatched_detections.discard(j)

    return matches, list(unmatched_tracks), list(unmatched_detections)

if __name__ == "__main__":
    import os
    import json
    from tqdm import tqdm
    
    nb = "11_0"
    data_folder = f"Data/{nb}/"
    data_folder = f"D:/p_{nb}/"
    fusion_folder = os.path.join(data_folder, "fusion")
    with open(os.path.join(fusion_folder, "targets.npy"), "rb") as f:
        num_frames = np.load(f, allow_pickle=True)
        fusion_frames = []
        CL_frames = []
        RL_frames = []
        frames = []
        for i in range(num_frames):
            fusion_frames.append(np.load(f, allow_pickle=True))
            CL_frames.append(np.load(f, allow_pickle=True))
            RL_frames.append(np.load(f, allow_pickle=True))
            with open(os.path.join(fusion_folder, f"targets_{i}.json"), 'r') as json_file:
                frame_data = json.load(json_file)
                frames.append(frame_data)
    
    tracks = []
    old_tracks = []
    max_age = 5
    
    for detections in tqdm(frames[120:180], desc="Tracking Progress"):
        matches, unmatched_tracks, unmatched_dets = associate_tracks_and_detections(tracks, detections, max_age)
        
        # Update matched tracks
        for i, j in matches:
            tracks[i].update(detections["targets"][j],  detections["timestamp"])
            
        # Create new tracks for unmatched detections
        for j in unmatched_dets:
            new_track = TrackRadar(detections["targets"][j], detections["timestamp"])
            tracks.append(new_track)
            
        # Remove old tracks
        old_tracks.extend([track for track in tracks if track.time_since_update > max_age])
        tracks = [track for track in tracks if track.time_since_update <= max_age]
        
    # Plotting the results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for track in tracks:
        coords = np.array([detection[2] for detection in track.history])
        preds = np.array(track.predictions)
        print(f"Track {track.id} history: {coords}")
        print(f"Track {track.id} predictions: {preds}")
        plt.plot(coords[:, 0], coords[:, 1], marker='o', label=f'Track {track.id}')
        plt.plot(preds[:, 0], preds[:, 1], linestyle='-.', label=f'Predicted {track.id}')
    for track in old_tracks:
        coords = np.array([detection[2] for detection in track.history])
        preds = np.array(track.predictions)
        plt.plot(coords[:, 0], coords[:, 1], marker='o', label=f'Track {track.id}')
        plt.plot(preds[:, 0], preds[:, 1], linestyle='-.', label=f'Predicted {track.id} (old)', alpha=0.5)
    plt.xlabel('range coordinate (m)')
    plt.ylabel('Doppler coordinate (m/s)')
    plt.title('Tracked Targets Over Time')
    plt.legend()
    plt.grid()
    plt.show()