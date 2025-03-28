import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import open3d as o3d
from tqdm import tqdm
import multiprocessing as mp
from time import sleep


def interpolate_velocity(velocity_timestamps, velocity_speeds, target_timestamp):
    # Interpolate velocity based on target timestamp
    idx = np.searchsorted(velocity_timestamps, target_timestamp, side="right") - 1
    if idx < 0 or idx >= len(velocity_timestamps) - 1:
        return None # No valid velocity data for interpolation

    before_timestamp, after_timestamp = velocity_timestamps[idx], velocity_timestamps[idx + 1]
    before_speed, after_speed = velocity_speeds[idx], velocity_speeds[idx + 1]

    if before_timestamp == after_timestamp:
        return before_speed
    else:
        ratio = (target_timestamp - before_timestamp)/ (after_timestamp - before_timestamp)
        ratio = ratio.astype(float)
        return before_speed + ratio * (after_speed - before_speed)
    
def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def remove_close_points(point, points, distance_threshold):
    distances = np.linalg.norm(points[:, :3] - point[:3], axis=1)  # Vectorized distance calculation
    return points[distances > distance_threshold]  # Boolean mask to filter points
    return np.array([p for p in points if euclidean_distance(point[:3], p[:3]) > distance_threshold])


# Read the data
lidar_file = 'Fusion/data/combination_20250312_123525'
lidar_data_file = f"{lidar_file}_data.npy"
lidar_ts_file = f"{lidar_file}_ts.npy"
lidar_timestamps = np.load(lidar_ts_file, allow_pickle=True)
with open(lidar_data_file, 'rb') as f:
    lidar_frames = [np.load(f, allow_pickle=True) for _ in range(len(lidar_timestamps))]

# Read speed data
speed_file = 'Fusion/captures30/speed_test.csv'
speed_df = pd.read_csv(speed_file)
speed_df['Time'] = pd.to_datetime(speed_df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
speed_df.dropna(subset=['Time'], inplace=True)
velocity_timestamps = speed_df['Time'].to_numpy()
velocity_speeds = speed_df['Speed (km/h)'].to_numpy()




if __name__ == '__main__':
    print("Removing LiDAR background of second frame...")
    print(f"Number of LiDAR frames: {len(lidar_frames)}")
    first_valid_frame = None
    for idx, lidar_frame in enumerate(lidar_frames):
        if len(lidar_frame) > 0 and interpolate_velocity(velocity_timestamps, velocity_speeds, lidar_timestamps[idx]) is not None and interpolate_velocity(velocity_timestamps, velocity_speeds, lidar_timestamps[idx]) > 10:
            first_valid_frame = idx
            break
    
    frame_data = lidar_frames[first_valid_frame+1]
    frame_ts = lidar_timestamps[first_valid_frame+1]
    previous_frame_data = lidar_frames[first_valid_frame]
    previous_frame_ts = lidar_timestamps[first_valid_frame]
    next_frame_data = lidar_frames[first_valid_frame+2]
    next_frame_ts = lidar_timestamps[first_valid_frame+2]
    # frame_data = frame_data[frame_data[:, 3] > 0]
    points = frame_data[:, :3]
    previous_points = previous_frame_data[:, :3]
    next_points = next_frame_data[:, :3]
    prev_delta_ts = (frame_ts - previous_frame_ts).astype(float) / 1e6
    next_delta_ts = (next_frame_ts - frame_ts).astype(float) / 1e6
    speed = interpolate_velocity(velocity_timestamps, velocity_speeds, frame_ts)
    speed /= 3.6
    print(speed * prev_delta_ts, 0.1)
    # exit()
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(points)
    # red color
    geometry.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]] * len(points)))
    vis.add_geometry(geometry)
    next_points[:, 0] -= speed * next_delta_ts
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(next_points)
    # blue color
    geometry.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1]] * len(next_points)))
    vis.add_geometry(geometry)
    previous_points[:, 0] += speed * prev_delta_ts
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(previous_points)
    # green color
    geometry.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0]] * len(previous_points)))
    vis.add_geometry(geometry)
    vis.run()
    # vis.run()
    
    for bg_p in tqdm(previous_points, desc="Removing background"):
        frame_data = remove_close_points(bg_p, frame_data,2 * speed * prev_delta_ts)
    for bg_p in tqdm(next_points, desc="Removing background"):
        frame_data = remove_close_points(bg_p, frame_data, 2 * speed * next_delta_ts)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(frame_data[:, :3])
    # red color
    geometry.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]] * len(frame_data)))
    vis.add_geometry(geometry)
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(previous_points)
    # green color
    geometry.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0]] * len(previous_points)))
    vis.add_geometry(geometry)
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(next_points)
    # blue color
    geometry.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1]] * len(next_points)))
    vis.add_geometry(geometry)
    vis.run()
    
