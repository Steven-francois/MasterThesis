import numpy as np
import os
import json
from scipy.optimize import linear_sum_assignment

def to_cam_coord(target, cam_size, cam_fov):
    """
    Convert target coordinates to camera coordinates.
    """
    coord = (target[:2] - cam_size / 2) / (cam_size / 2) * cam_fov
    return coord

def to_lidar_coord(target):
    """
    Convert target coordinates to LiDAR coordinates.
    """
    r = np.linalg.norm(target)
    phi = target[0] / target[0] * np.arccos(target[1] / np.linalg.norm(target[:2])) * 180 / np.pi - 90
    theta = np.arccos(target[2] / r) * 180 / np.pi - 90
    return np.array((phi, theta, r))

def cost_function_CL(cam_coords, lidar_coords):
    """
    Cost function for the assignment problem.
    """
    cost = np.linalg.norm(cam_coords - lidar_coords, axis=1)
    return cost

nb = "1_0"
# data_folder = f"Data/{nb}/"
data_folder = f"D:/processed"
image_folder = os.path.join(data_folder, "cam_targets")
lidar_folder = os.path.join(data_folder, "lidar")
radar_folder = os.path.join(data_folder, "radar")
with open(os.path.join(image_folder, "targets.npy"), "rb") as f:
    num_frames = np.load(f, allow_pickle=True)
    cam_frames = [np.load(f, allow_pickle=True) for _ in range(num_frames)]
with open(os.path.join(lidar_folder, "targets.npy"), "rb") as f:
    num_frames = np.load(f, allow_pickle=True)
    lidar_frames = [np.load(f, allow_pickle=True) for _ in range(num_frames)]

cam_fov = np.array((30,17))
cam_size = np.array((640, 480))

nb_frame = 201
for nb_frame in range(195, 202):
    cam_frame = cam_frames[nb_frame]
    with open(os.path.join(radar_folder, f"{nb_frame}.json")) as rt_file:
        radar_targets = json.load(rt_file)
    lidar_frame = lidar_frames[nb_frame]
    nb_lidar_targets = int(np.max(lidar_frame[:, 3]))+1
    lidar_targets = np.zeros((nb_lidar_targets, 3))
    for i in range(nb_lidar_targets):
        lidar_targets[i, :3] = np.mean(lidar_frame[lidar_frame[:, 3] == i, :3], axis=0)
    cam_coords = np.array([to_cam_coord(target, cam_size, cam_fov) for target in cam_frame])
    lidar_coords = np.array([to_lidar_coord(target[:3]) for target in lidar_targets])
    radar_coords = np.array([np.array([target["range"], target["speed"]]) for target in radar_targets])
    cost_matrix_CL = np.zeros((len(cam_coords), len(lidar_coords)))
    cost_matrix_RL = np.zeros((len(radar_coords), len(lidar_coords)))
    for i in range(len(cam_coords)):
        for j in range(len(lidar_coords)):
            cost_matrix_CL[i, j] = np.linalg.norm(cam_coords[i] - lidar_coords[j, :2])
    for i in range(len(radar_coords)):
        for j in range(len(lidar_coords)):
            cost_matrix_RL[i, j] = np.linalg.norm(radar_coords[i,0] - lidar_coords[j,2])

    row_ind, col_ind = linear_sum_assignment(cost_matrix_CL)
    matches_CL = list(zip(row_ind, col_ind))
    threshold_CL = 2
    filtered_matches_CL = [pair for pair in matches_CL if cost_matrix_CL[pair[0], pair[1]] < threshold_CL]
    print(f"Frame {nb_frame}: {len(filtered_matches_CL)} matches found")
    for i, j in filtered_matches_CL:
        print(f"Camera target {i} matched with LiDAR target {j} with cost {cost_matrix_CL[i, j]:.2f}, cam coord {cam_coords[i]}, lidar coord {lidar_coords[j]}")
    print()
