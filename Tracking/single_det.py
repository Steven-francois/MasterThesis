from Tracking.Track import *
import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

nb = "11_0"
data_folder = f"Data/{nb}/"
# data_folder = f"D:/p_{nb}/"
fusion_folder = os.path.join(data_folder, "fusion")
with open(os.path.join(fusion_folder, "targets.npy"), "rb") as f:
    num_frames = np.load(f, allow_pickle=True)
    frames = []
    for i in range(num_frames):
        with open(os.path.join(fusion_folder, f"targets_{i}.json"), 'r') as json_file:
            frame_data = json.load(json_file)
            frames.append(frame_data)

# tracking_intervals = [
#     slice(155, 185),
#     slice(300, 435),
#     slice(470, 540),
#     slice(980, 1058),
#     slice(1258, 1270),
#     slice(2345, 2385),
#     slice(3228, 3300),
#     slice(3587, 3624),
#     slice(3869, 3942),
#     slice(4119, 4442),
#     slice(2500, 2600)
# ]
# tracking_intervals = [
#         slice(745, 800),
#         slice(1138, 1169),
#         slice(1170, 1181),
#         slice(1181, 1191),
#         slice(1192, 1204),
#         slice(1204, 1241),
#         slice(2421, 2508),
#         slice(2509, 2535),
#         slice(2541, 2575),
#         slice(2600, 2615),
#         slice(3002, 3049),
#     ]

tracking_intervals = [
        slice(300, 540),
        slice(2500, 2600)
    ]
nb_tracks_radar = []
nb_tracks_lidar = []
nb_tracks_fused = []

for interval in tqdm(tracking_intervals, desc="Tracking intervals"):
    radar_tracks = []
    old_radar_tracks = []
    lidar_tracks = []
    old_lidar_tracks = []
    fused_tracks = []
    old_fused_tracks = []
    max_age = 1000

    for detections in frames[interval]:
        # Radar tracking
        matches, unmatched_tracks, unmatched_dets = associate_tracks_and_detections(radar_tracks, detections, max_age)
        
        # Update matched tracks
        for i, j in matches:
            radar_tracks[i].update(detections["targets"][j],  detections["timestamp"], j)
            
        # Create new tracks for unmatched detections
        for j in unmatched_dets:
            new_track = Track(detections["targets"][j], detections["timestamp"], j)
            radar_tracks.append(new_track)
            
        # Remove old tracks
        old_radar_tracks.extend([track for track in radar_tracks if track.time_since_update > max_age])
        radar_tracks = [track for track in radar_tracks if track.time_since_update <= max_age]

        # Lidar tracking
        matches, unmatched_tracks, unmatched_dets = associate_tracks_and_detections(lidar_tracks, detections, max_age, True)

        # Update matched tracks
        for i, j in matches:
            lidar_tracks[i].update(detections["targets"][j],  detections["timestamp"], j)

        # Create new tracks for unmatched detections
        for j in unmatched_dets:
            new_track = Track(detections["targets"][j], detections["timestamp"], j)
            lidar_tracks.append(new_track)

        # Remove old tracks
        old_lidar_tracks.extend([track for track in lidar_tracks if track.time_since_update > max_age])
        lidar_tracks = [track for track in lidar_tracks if track.time_since_update <= max_age]

        matches, unmatched_tracks, unmatched_dets = associate_tracks_and_detections(fused_tracks, detections, max_age, False, True)

        # Update matched tracks
        for i, j in matches:
            fused_tracks[i].update(detections["targets"][j], detections["timestamp"], j)

        # Create new tracks for unmatched detections
        for j in unmatched_dets:
            new_track = Track(detections["targets"][j], detections["timestamp"], j)
            fused_tracks.append(new_track)

        # Remove old tracks
        old_fused_tracks.extend([track for track in fused_tracks if track.time_since_update > max_age])
        fused_tracks = [track for track in fused_tracks if track.time_since_update <= max_age]

    nb_tracks_radar.append(len(radar_tracks) + len(old_radar_tracks))
    nb_tracks_lidar.append(len(lidar_tracks) + len(old_lidar_tracks))
    nb_tracks_fused.append(len(fused_tracks) + len(old_fused_tracks))

# Plotting the number of tracks
plt.figure(figsize=(10, 5))
x = np.arange(len(nb_tracks_radar))
width = 0.35

# nb_tracks_camera = [1.0, 1.0, 1.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0]
# nb_tracks_camera = [4, 3, 5, 3, 4, 5, 5, 5, 6, 7, 6]
# nb_tracks_camera = [2, 3, 3, 3, 3, 3, 2, 3, 4, 4, 3]
nb_tracks_camera = [3, 5]

plt.bar(x - width/2, nb_tracks_radar, width, label='Radar', color='b')
plt.bar(x, nb_tracks_camera, width, label='Camera', color='g')
plt.bar(x + width/2, nb_tracks_lidar, width, label='Lidar', color='r')
plt.bar(x + width, nb_tracks_fused, width, label='Fused', color='y')

plt.title('Number of Tracks Per Scene')
plt.xlabel('Scene number')
plt.ylabel('Number of Tracks')
plt.xticks(x)
plt.legend()
plt.show()