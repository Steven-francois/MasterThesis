from Tracking.Track_vote import *
import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

nb = "21_0"
# data_folder = f"Data/{nb}/"
data_folder = f"D:/p_{nb}/"


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
#     slice(4119, 4442)
# ]
tracking_intervals = [
        slice(745, 800),
        slice(1138, 1169),
        slice(1170, 1181),
        slice(1181, 1191),
        slice(1192, 1204),
        slice(1204, 1241),
        slice(2421, 2508),
        slice(2509, 2535),
        slice(2541, 2575),
        slice(2600, 2615),
        slice(3002, 3049),
    ]

tracking_intervals = [
        slice(300, 540),
        slice(2500, 2600)
    ]


fusion_folder = os.path.join(data_folder, "fusion")
with open(os.path.join(fusion_folder, "targets.npy"), "rb") as f:
    num_frames = np.load(f, allow_pickle=True)
    frames = []
    min_cam_id = 1000
    max_cam_id = 0
    cam_ids = []
    for i in range(num_frames):
        fusion_frame = np.load(f, allow_pickle=True)
        CL_frames = np.load(f, allow_pickle=True)
        RL_frames = np.load(f, allow_pickle=True)
        with open(os.path.join(fusion_folder, f"targets_{i}.json"), 'r') as json_file:
            frame_data = json.load(json_file)
            frames.append(frame_data)
            # if i >= tracking_interval.start and i < tracking_interval.stop:
            # fusion_frame = np.array(frame_data["targets_nb"])
            c_ids = np.array(frame_data["cam_info"]["ids"])
            if len(fusion_frame) > 0 and len(c_ids) > 0:
                c_ids = c_ids[fusion_frame[:, 0]]
            elif len(fusion_frame) == 0:
                c_ids = c_ids[:0]
            min_cam_id = int(min(min_cam_id, min(c_ids) if len(c_ids) > 0 else min_cam_id))
            max_cam_id = int(max(max_cam_id, max(c_ids) if len(c_ids) > 0 else max_cam_id))
            cam_ids.append(c_ids.tolist())

nb_tracks_radar = []
nb_tracks_lidar = []
nb_tracks_fused = []
nb_tracks_camera = []

for interval in tqdm(tracking_intervals, desc="Tracking intervals"):
    radar_tracks = []
    old_radar_tracks = []
    lidar_tracks = []
    old_lidar_tracks = []
    fused_tracks = []
    old_fused_tracks = []
    max_age = 100
    camera_ids = []

    for frame_idx, detections in enumerate(frames[interval]):
        # Camera
        for j in range(len(cam_ids[interval.start+frame_idx])):
            camera_ids.append(cam_ids[interval.start+frame_idx][j])

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

        matches, unmatched_tracks, unmatched_dets = associate_tracks_and_detections(fused_tracks, detections, max_age, False, True, cam_ids[interval.start+frame_idx] if len(cam_ids[interval.start+frame_idx]) > 0 else None)

        # Update matched tracks
        for i, j in matches:
            fused_tracks[i].update(detections["targets"][j], detections["timestamp"], j, cam_ids[interval.start+frame_idx][j] if len(cam_ids[interval.start+frame_idx]) > 0 else None)

        # Create new tracks for unmatched detections
        for j in unmatched_dets:
            new_track = Track(detections["targets"][j], detections["timestamp"], j, cam_ids[interval.start+frame_idx][j] if len(cam_ids[interval.start+frame_idx]) > 0 else None)
            fused_tracks.append(new_track)

        # Remove old tracks
        old_fused_tracks.extend([track for track in fused_tracks if track.time_since_update > max_age])
        fused_tracks = [track for track in fused_tracks if track.time_since_update <= max_age]
    

    nb_tracks_radar.append(len(radar_tracks) + len(old_radar_tracks))
    nb_tracks_lidar.append(len(lidar_tracks) + len(old_lidar_tracks))
    nb_tracks_fused.append(len(fused_tracks) + len(old_fused_tracks))
    nb_tracks_camera.append(len(set(camera_ids)))

# Plotting the number of tracks
plt.figure(figsize=(10, 5))
x = np.arange(len(nb_tracks_radar))
width = 0.2

# nb_tracks_camera = [1, 1, 1, 6, 1, 1, 1, 1, 1, 1]
# nb_tracks_gt = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# nb_tracks_camera = [4, 3, 5, 3, 4, 5, 5, 5, 6, 7, 6]
# nb_tracks_camera = [2, 3, 3, 3, 3, 3, 2, 3, 4, 4, 3]
# nb_tracks_gt = [2, 3, 3, 3, 3, 3, 2, 3, 4, 4, 3]
# nb_tracks_camera = [3, 6]
nb_tracks_gt = [2, 5]
s = [50 for n in range(len(x))]

plt.bar(x - 3*width/2, nb_tracks_radar, width, label='Radar', color='b', alpha = 0.7)
plt.bar(x - width/2, nb_tracks_camera, width, label='Camera', color='g', alpha = 0.7)
plt.bar(x + width/2, nb_tracks_lidar, width, label='Lidar', color='r', alpha = 0.7)
plt.bar(x + 3*width/2, nb_tracks_fused, width, label='Fused', color='y', alpha = 0.7)
first_gt_label = False
for i in range(len(x)):
    # plt.text(x[i] - 3*width/2, nb_tracks_radar[i] + 0.1, str(nb_tracks_radar[i]), ha='center', va='bottom')
    # plt.text(x[i] - width/2, nb_tracks_camera[i] + 0.1, str(nb_tracks_camera[i]), ha='center', va='bottom')
    # plt.text(x[i] + width/2, nb_tracks_lidar[i] + 0.1, str(nb_tracks_lidar[i]), ha='center', va='bottom')
    # plt.text(x[i] + 3*width/2, nb_tracks_fused[i] + 0.1, str(nb_tracks_fused[i]), ha='center', va='bottom')
    if not first_gt_label:
        plt.plot([x[i] - 2*width, x[i] + 2*width], [nb_tracks_gt[i], nb_tracks_gt[i]], color='black', linestyle='--', label='GT', linewidth=2)
        first_gt_label = True
    plt.plot([x[i] - 2*width, x[i] + 2*width], [nb_tracks_gt[i], nb_tracks_gt[i]], color='black', linestyle='--', linewidth=2)

plt.title('Number of Tracks Per Scene')
plt.xlabel('Scene number')
plt.ylabel('Number of Tracks')
plt.xticks(x)
plt.legend()
plt.show()