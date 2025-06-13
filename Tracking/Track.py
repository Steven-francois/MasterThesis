import numpy as np
from scipy.optimize import linear_sum_assignment

class Track:
    count = 0
    
    def __init__(self, detection):
        self.id = Track.count
        Track.count += 1
        self. history = [detection]
        self.time_since_update = 0
        self.last_update_time = detection["timestamp"]
        
    def predict(self, new_timestamp):
        self.time_since_update += 1
        
        # Radar prediction
        radar_coord = self.history[-1]["radar_coord"]
        dt = (new_timestamp - self.last_update_time).total_seconds()
        new_radar_coord = (radar_coord[0] + radar_coord[1] * dt, radar_coord[1])
        prediction = {
            "radar_coord": new_radar_coord,
            "timestamp": new_timestamp
        }
        return prediction
    
    def update(self, detection):
        self.history.append(detection)
        self.time_since_update = 0
        self.last_update_time = detection["timestamp"]


# --- Track management functions
def compute_cost(tracks, detections):
    cost = np.zeros((len(tracks), len(detections)))
    for i, track in enumerate(tracks):
        pred = track.predict(detections[0]["timestamp"])
        for j, det in enumerate(detections):
            cost[i, j] = np.linalg.norm(np.array(pred["radar_coord"]) - np.array(det["radar_coord"]))
    return cost

def associate_tracks_and_detections(tracks, detections, max_age=5):
    cost_matrix = compute_cost(tracks, detections)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_detections = set(range(len(detections)))

    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < max_age:
            matches.append((i, j))
            unmatched_tracks.discard(i)
            unmatched_detections.discard(j)

    return matches, list(unmatched_tracks), list(unmatched_detections)

if __name__ == "__main__":
    import os
    import json
    
    
    nb = "11_0"
    data_folder = f"Data/{nb}/"
    # data_folder = f"D:/processed"
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
    max_age = 5
    
    for detections in frames[120:150]:
        print(detections)
        cost = compute_cost(tracks, detections) if tracks else np.zeros((0, len(detections)))
        matches, unmatched_tracks, unmatched_dets = associate_tracks_and_detections(tracks, detections, max_age)
        
        # Update matched tracks
        for i, j in matches:
            tracks[i].update(detections[j])
            
        # Create new tracks for unmatched detections
        for j in unmatched_dets:
            new_track = Track(detections[j])
            tracks.append(new_track)
            
        # Remove old tracks
        tracks = [track for track in tracks if track.time_since_update <= max_age]
        
    # Plotting the results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for track in tracks:
        coords = np.array([detection["radar_coord"] for detection in track.history])
        plt.plot(coords[:, 0], coords[:, 1], marker='o', label=f'Track {track.id}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Tracked Targets Over Time')
    plt.legend()
    plt.grid()
    plt.show()