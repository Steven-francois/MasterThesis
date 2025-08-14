import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# --- Gating configuration (tunable) ---
# Base threshold and growth rate per second for time-since-update aware gating.
# Effective threshold = BASE * (1 + GROWTH_PER_SEC * track.time_since_update)
GATE_BASE_RADAR = 5.0
GATE_GROWTH_PER_SEC_RADAR = 100.0
GATE_BASE_LIDAR = 5.0
GATE_GROWTH_PER_SEC_LIDAR = 100.0

class Track:
    count = 0
    
    def __init__(self, detection, ts, nb):
        self.id = Track.count
        Track.count += 1
        self. history = [detection]
        self.history_ts = [ts]
        self.time_since_update = 0
        self.last_update_time = ts
        self.predictions = []
        self.nbs = [nb]  
        
    def predict(self, new_timestamp):
        self.time_since_update = new_timestamp - self.last_update_time
        
        # Radar prediction
        radar_coord = self.history[-1][2]
        dt = new_timestamp - self.last_update_time
        new_radar_coord = (radar_coord[0] + radar_coord[1] * dt, radar_coord[1])
        prediction = {
            "radar_coord": new_radar_coord,
            "timestamp": new_timestamp
        }
        return prediction
    
    def update(self, detection, ts, nb):
        self.history.append(detection)
        self.time_since_update = 0
        self.last_update_time = ts
        self.history_ts.append(ts)
        self.nbs.append(nb) 

class TrackRadar(Track):
    def __init__(self, detection, ts, nb):
        super().__init__(detection, ts, nb)
        self.radar_coord = detection[2]  # Assuming detection[2] contains radar coordinates

        # Initialization of Kalman filter
        kf = KalmanFilter(dim_x=2, dim_z=2)
        dt = 1.0  # frame time step
        kf.F = np.array([[1, dt],
                         [0, 1]])  # state transition

        kf.H = np.array([[1, 0],
                         [0, 1]])    # measurement function

        kf.R *= 0.9   # measurement noise
        kf.P *= 10.0  # initial uncertainty
        kf.Q *= 0.1   # process noise

        kf.x = np.array(self.radar_coord).reshape(2, 1)
        self.kf = kf

    def __state_transition_dt(self, dt):
        self.kf.F = np.array([[1, dt],
                              [0, 1]])  # state transition

    def predict(self, new_timestamp, kalman=False):
        self.time_since_update = new_timestamp - self.last_update_time
        dt = new_timestamp - self.last_update_time
        
        # Radar prediction
        if kalman:
            self.__state_transition_dt(dt)
            self.kf.predict()
            prediction = {
                "radar_coord": self.kf.x.reshape(-1),  # return [x, y]
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
    
    def update(self, detection, ts, nb):
        super().update(detection, ts, nb)
        self.radar_coord = detection[2]
        self.kf.update(np.array(self.radar_coord).reshape(2, 1))  # Update Kalman filter with new measurement

class TrackLidar(Track):
    def __init__(self, detection, ts, nb):
        super().__init__(detection, ts, nb)
        self.lidar_coord = detection[1]
        
        # Initialization of Kalman filter
        kf = KalmanFilter(dim_x=6, dim_z=3)
        dt = 1.0  # frame time step
        kf.F = np.array([[1, dt, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, dt, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, dt],
                         [0, 0, 0, 0, 0, 1]])  # state transition

        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0]])    # measurement function

        kf.R *= 0.9   # measurement noise
        kf.P *= 10.0  # initial uncertainty
        kf.Q *= 0.1   # process noise

        kf.x[::2] = np.array(self.lidar_coord).reshape(3, 1)  # Initialize state with lidar coordinates
        self.kf = kf

    def __state_transition_dt(self, dt):
        self.kf.F = np.array([[1, dt, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 1, dt, 0, 0],
                              [0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, dt],
                              [0, 0, 0, 0, 0, 1]])  # state transition
        
    def predict(self, new_timestamp, kalman=False):
        self.time_since_update = new_timestamp - self.last_update_time
        dt = new_timestamp - self.last_update_time

        #LiDar prediction
        if kalman:
            self.__state_transition_dt(dt)
            self.kf.predict()
            prediction = {
                "lidar_coord": self.kf.x[::2].reshape(-1),  # return [x, y, z]
                "timestamp": new_timestamp
            }
        else:
            if len(self.history) < 2:
                new_lidar_coord = self.lidar_coord
            else:
                speed = np.array(self.history[-1][1]) - np.array(self.history[-2][1])
                speed_dt = self.history_ts[-1] - self.history_ts[-2]
                new_lidar_coord = self.lidar_coord + speed/speed_dt * dt
            prediction = {
                "lidar_coord": new_lidar_coord,
                "timestamp": new_timestamp
            }
        self.predictions.append(prediction["lidar_coord"])
        return prediction
    
    def update(self, detection, ts, nb):
        super().update(detection, ts, nb)
        self.lidar_coord = detection[1]
        self.kf.update(np.array(self.lidar_coord).reshape(3, 1))

# --- Track management functions
def compute_cost(tracks, detections, lidar=False):
    cost = np.zeros((len(tracks), len(detections["targets_nb"])))
    for i, track in enumerate(tracks):
        pred = track.predict(detections["timestamp"], True)
        for j in range(len(detections["targets_nb"])):
            if lidar:
                cost[i, j] = np.linalg.norm(np.array(pred["lidar_coord"]) - np.array(detections["targets"][j][1]))
            else:
                cost[i, j] = np.linalg.norm(np.array(pred["radar_coord"]) - np.array(detections["targets"][j][2]))
    return cost

def associate_tracks_and_detections(tracks, detections, max_age=5, lidar=False):
    cost_matrix = compute_cost(tracks, detections, lidar=lidar)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_detections = set(range(len(detections["targets_nb"])))

    for i, j in zip(row_ind, col_ind):
        # Time-since-update aware gating: threshold grows with how long a track hasn't been updated.
        if lidar:
            gate = GATE_BASE_LIDAR * (1.0 + GATE_GROWTH_PER_SEC_LIDAR * tracks[i].time_since_update)
        else:
            gate = GATE_BASE_RADAR * (1.0 + GATE_GROWTH_PER_SEC_RADAR * tracks[i].time_since_update)

        if cost_matrix[i, j] < gate:
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
    # data_folder = f"D:/p_{nb}/"
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

    tracking_interval = slice(300, 560)
    # tracking_interval = slice(3228, 3300)
    # tracking_interval = slice(0, 450)
    
    tracks = []
    old_tracks = []
    max_age = 1000
    
    for detections in tqdm(frames[tracking_interval], desc="Tracking Progress"):
        matches, unmatched_tracks, unmatched_dets = associate_tracks_and_detections(tracks, detections, max_age)
        
        # Update matched tracks
        for i, j in matches:
            tracks[i].update(detections["targets"][j],  detections["timestamp"], j)
            
        # Create new tracks for unmatched detections
        for j in unmatched_dets:
            new_track = TrackRadar(detections["targets"][j], detections["timestamp"], j)
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
        # print(f"Track {track.id} history: {coords}")
        # print(f"Track {track.id} predictions: {preds}")
        plt.plot(coords[:, 0], coords[:, 1], marker='o', label=f'Track {track.id}')
        plt.plot(preds[:, 0], preds[:, 1], marker='o', linestyle='-', label=f'Predicted {track.id}')
    for track in old_tracks:
        coords = np.array([detection[2] for detection in track.history])
        preds = np.array(track.predictions)
        plt.plot(coords[:, 0], coords[:, 1], marker='o', label=f'Track {track.id}')
        plt.plot(preds[:, 0], preds[:, 1], marker='o', linestyle='-', label=f'Predicted {track.id} (old)', alpha=0.5)
    plt.xlabel('range coordinate (m)')
    plt.ylabel('Doppler coordinate (m/s)')
    plt.title('Tracked Targets Over Time')
    plt.legend()
    plt.grid()
    plt.show()

    tracks_lidar = []
    old_tracks_lidar = []
    max_age = 100
    Track.count = 0  # Reset track count for Lidar tracks
    with open(os.path.join(fusion_folder, "tracks_lidar.npy"), "wb") as f:
        ti = np.array(tracking_interval)
        np.save(f, ti)

        for detections in tqdm(frames[tracking_interval], desc="Tracking Progress"):
            matches, unmatched_tracks, unmatched_dets = associate_tracks_and_detections(tracks_lidar, detections, max_age, True)
            tracks_dets = np.zeros(Track.count, dtype=int)
            tracks_dets[:] = -1
            
            # Update matched tracks
            for i, j in matches:
                tracks_lidar[i].update(detections["targets"][j],  detections["timestamp"], j)
                tracks_dets[tracks_lidar[i].id] = j

            # Create new tracks for unmatched detections
            for j in unmatched_dets:
                new_track = TrackLidar(detections["targets"][j], detections["timestamp"], j)
                tracks_lidar.append(new_track)
                tracks_dets = np.append(tracks_dets, j)
                
            # Remove old tracks
            old_tracks_lidar.extend([track for track in tracks_lidar if track.time_since_update > max_age])
            tracks_lidar = [track for track in tracks_lidar if track.time_since_update <= max_age]
            print(tracks_dets)
            np.save(f, tracks_dets)
        
    # Plotting the results
    # import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for track in tracks_lidar:
        coords = np.array([detection[1] for detection in track.history])
        preds = np.array(track.predictions)
        # print(f"Track {track.id} history: {coords} ts: {track.history_ts}")
        # print(f"Track {track.id} predictions: {preds}")
        plt.plot(coords[:, 0], coords[:, 2], marker='o', label=f'Track {track.id}')
        plt.plot(preds[:, 0], preds[:, 2], marker='o', linestyle='-', label=f'Predicted {track.id}')
    for track in old_tracks_lidar:
        coords = np.array([detection[1] for detection in track.history])
        preds = np.array(track.predictions)
        plt.plot(coords[:, 0], coords[:, 2], marker='o', label=f'Track {track.id}')
        plt.plot(preds[:, 0], preds[:, 2], marker='o', linestyle='-', label=f'Predicted {track.id} (old)', alpha=0.5)
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.title('Tracked Targets Over Time')
    plt.legend()
    plt.grid()
    plt.show()

    # Plotting selected targets for each track
    for track in tracks_lidar:
        plt.plot(np.array(track.history_ts) + 0.01, track.nbs, marker='o', label=f'Track_lidar {track.id} NBs')
    for track in old_tracks_lidar:
        plt.plot(np.array(track.history_ts) + 0.01, track.nbs, marker='o', label=f'Track_lidar {track.id} NBs (old)', alpha=0.5)
    for track in tracks:
        plt.plot(track.history_ts, track.nbs, marker='o', label=f'Track {track.id} NBs')
    for track in old_tracks:
        plt.plot(track.history_ts, track.nbs, marker='o', label=f'Track {track.id} NBs (old)', alpha=0.5)
    frames_ts = [frame["timestamp"] for frame in frames[tracking_interval]]
    frames_nbs = [len(frame["targets_nb"]) for frame in frames[tracking_interval]]
    # plt.plot(frames_ts, frames_nbs, marker='x', label='Number of targets in frame')
    plt.xlabel('Timestamp')
    plt.ylabel('Target number')
    plt.title('Target Numbers Over Time')
    plt.legend()
    plt.grid()
    plt.show()