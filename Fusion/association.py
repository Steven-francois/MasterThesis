import numpy as np
import os
import json
from scipy.optimize import linear_sum_assignment
from tqdm import trange

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

def to_radar_coord(target):
    """
    Convert target coordinates to radar coordinates.
    """
    peak = target["peak"]
    idxs = np.array([np.array(t) for t in target["idxs"]])
    peak_i = np.where(idxs[:,0] == peak[0])[0]
    peak_j = np.where(idxs[:,1] == peak[1])[0]
    peak_idx = np.intersect1d(peak_i, peak_j)
    if len(peak_idx) == 0:
        return np.array([target["range"], target["speed"]])
    peak_idx = peak_idx[0]
    return np.array(target["coord"][peak_idx])

def match_dual_modalities(mod1, mod2, threshold=2):
    """
    Match two modalities using the Hungarian algorithm.
    """
    cost_matrix = np.zeros((len(mod1), len(mod2)))
    for i in range(len(mod1)):
        for j in range(len(mod2)):
            cost_matrix[i, j] = np.linalg.norm(mod1[i] - mod2[j])
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = np.column_stack((row_ind, col_ind))
    
    filtered_matches = matches[cost_matrix[row_ind, col_ind] < threshold]
    
    
    return filtered_matches, cost_matrix

def match_modalities(cam_coords, lidar_coords, radar_coords, verbose=False):
    """
    Match camera, LiDAR, and radar coordinates.
    """
    matches_CL, cost_matrix_CL = match_dual_modalities(cam_coords, lidar_coords[:, :2], 4) if len(cam_coords) > 0 and len(lidar_coords) > 0 else ([], np.zeros((0, 0)))
    matches_RL, cost_matrix_RL = match_dual_modalities(radar_coords[:, 0], lidar_coords[:, 2]) if len(radar_coords) > 0 and len(lidar_coords) > 0 else ([], np.zeros((0, 0)))
    matches_CL = np.array([match for match in matches_CL if lidar_coords[match[1], 2] < 20 or cost_matrix_CL[match[0], match[1]] < 2])

    if verbose:
        print(f"Frame {nb_frame}: {len(matches_CL)} camera-LiDAR matches, {len(matches_RL)} radar-LiDAR matches: {len(cam_frame)} camera targets, {len(lidar_targets)} LiDAR targets, {len(radar_targets)} radar targets")
        print((f"Lidar targets distance: \n{lidar_coords[:, 2]}\n"))
        print((f"Camera-LiDAR cost matrix:\n{cost_matrix_CL}\n"))
        print((f"Radar-LiDAR cost matrix:\n{cost_matrix_RL}\n"))
    
    # print(f"CL Frame {nb_frame}: {len(matches_CL)} matches found")
    # for i, j in matches_CL:
    #     print(f"Camera target {i} matched with LiDAR target {j} with cost {cost_matrix_CL[i, j]:.2f}, cam coord {cam_coords[i]}, lidar coord {lidar_coords[j]}")
    # print()
    # print(f"RL Frame {nb_frame}: {len(matches_RL)} matches found")
    # for i, j in matches_RL:
    #     print(f"Radar target {i} matched with LiDAR target {j} with cost {cost_matrix_RL[i, j]:.2f}, radar coord {radar_coords[i]}, lidar coord {lidar_coords[j]}")
    # print()
    # print(f"Frame {nb_frame}: {len(cam_frame)} camera targets, {len(lidar_targets)} LiDAR targets, {len(radar_targets)} radar targets")
    # print()
    # print()
    
    lidar_nb, idx_Cl, idx_RL = np.intersect1d(matches_CL[:, 1], matches_RL[:, 1], return_indices=True) if len(matches_CL) > 0 and len(matches_RL) > 0 else ([], [], [])
    cam_nb =  matches_CL[idx_Cl, 0] if len(matches_CL) > 0 else np.array([])
    radar_nb = matches_RL[idx_RL, 0] if len(matches_RL) > 0 else np.array([])
    
    targets_nb = np.column_stack((cam_nb, lidar_nb, radar_nb))
    if verbose:
        print(f"--- Frame {nb_frame}: {len(targets_nb)} targets found with camera, LiDAR, and radar")
    
    
    return targets_nb

nb = "1_1"
data_folder = f"Data/{nb}/"
# data_folder = f"D:/processed"
image_folder = os.path.join(data_folder, "cam_targets")
lidar_folder = os.path.join(data_folder, "lidar")
radar_folder = os.path.join(data_folder, "radar", "targets")
fusion_folder = os.path.join(data_folder, "fusion")
os.makedirs(fusion_folder, exist_ok=True)
with open(os.path.join(image_folder, "targets.npy"), "rb") as f:
    num_frames = np.load(f, allow_pickle=True)
    cam_frames = [np.load(f, allow_pickle=True) for _ in range(num_frames)]
with open(os.path.join(lidar_folder, "targets.npy"), "rb") as f:
    num_frames = np.load(f, allow_pickle=True)
    lidar_frames = [np.load(f, allow_pickle=True) for _ in range(num_frames)]

cam_fov = np.array((30,17))
cam_size = np.array((640, 480))

SAVE = False

if SAVE:
    f = open(os.path.join(fusion_folder, "targets.npy"), "wb")
    np.save(f, len(cam_frames), allow_pickle=True)
nb_frame = 201
# for nb_frame in trange(len(cam_frames)):
for nb_frame in range(4369, 4371):
    cam_frame = cam_frames[nb_frame]
    with open(os.path.join(radar_folder, f"targets_{nb_frame}.json"), "r") as rt_file:
        radar_targets = json.load(rt_file)
    lidar_frame = lidar_frames[nb_frame]
    nb_lidar_targets = int(np.max(lidar_frame[:, 3]))+1 if len(lidar_frame) > 0 else 0
    lidar_targets = np.zeros((nb_lidar_targets, 3))
    for i in range(nb_lidar_targets):
        lidar_targets[i, :3] = np.mean(lidar_frame[lidar_frame[:, 3] == i, :3], axis=0)
    cam_coords = np.array([to_cam_coord(target, cam_size, cam_fov) for target in cam_frame])
    lidar_coords = np.array([to_lidar_coord(target[:3]) for target in lidar_targets])
    radar_coords = np.array([to_radar_coord(target) for target in radar_targets])
    if SAVE:
        np.save(f,match_modalities(cam_coords, lidar_coords, radar_coords), allow_pickle=True)
    else:
        match_modalities(cam_coords, lidar_coords, radar_coords, verbose=not SAVE)
if SAVE:
    f.close()
    print(f"Fusion targets saved to {os.path.join(fusion_folder, 'targets.npy')}")
        
    
    
