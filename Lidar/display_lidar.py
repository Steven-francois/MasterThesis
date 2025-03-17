import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import open3d as o3d
import time
from tqdm import tqdm
import multiprocessing as mp

# Read the data
df = pd.read_csv('Lidar/data/combi_20250228_120139.csv')


def filter_lidar_data(id_):
    filtered_data = df[df['ID'] == id_][['X', 'Y', 'Z', 'Intensity']].to_numpy()
    return id_, filtered_data  


if __name__ == '__main__':
    Ids = df['ID'].unique()
    # num_workers = max(1, mp.cpu_count() - 1) 
    num_workers = 4
    now = time.time()
    with mp.Pool(processes=num_workers) as pool:
        lidar_frames = dict(pool.map(filter_lidar_data, Ids))
    print(f"Precomputed {len(lidar_frames)} LiDAR frames using {num_workers} processes!")
    print(f"Precomputation took {time.time() - now:.4f} seconds")
    # lidar_frames = {id_: df[df['ID'] == id_][['X', 'Y', 'Z', 'Intensity']].to_numpy() for id_ in tqdm(Ids, desc="Reading data")}

    global_min_intensity = df['Intensity'].min()
    global_max_intensity = df['Intensity'].max()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    geometry = o3d.geometry.PointCloud()
    frame_data = lidar_frames[Ids[0]]
    points = frame_data[:, :3]
    intensity = frame_data[:, 3]
    geometry.points = o3d.utility.Vector3dVector(points)
    intensity_normalized = (intensity - global_min_intensity) / (global_max_intensity - global_min_intensity)
    colors = plt.cm.viridis(intensity_normalized)[:, :3]  # Use viridis colormap
    geometry.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(geometry)

    _ = input("Press Enter to start the animation")
    for i in range(len(Ids)):
        now = time.time()
        frame_data = lidar_frames[Ids[i]]
        points = frame_data[:, :3]
        intensity = frame_data[:, 3]
        geometry.points = o3d.utility.Vector3dVector(points)
        intensity_normalized = (intensity - global_min_intensity) / (global_max_intensity - global_min_intensity)
        colors = plt.cm.viridis(intensity_normalized)[:, :3]  # Use viridis colormap
        geometry.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()
        print(f"Frame {i+1} processed in {time.time() - now:.4f} seconds")
        

    vis.destroy_window()
    exit()
