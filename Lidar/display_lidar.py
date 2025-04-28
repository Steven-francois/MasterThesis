import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import open3d as o3d
import time
from tqdm import tqdm
from time import sleep

# Read the data
lidar_file = 'Fusion/data/1/lidar_combined'
lidar_data_file = f"{lidar_file}_data.npy"
lidar_ts_file = f"{lidar_file}_ts.npy"
lidar_timestamps = np.load(lidar_ts_file, allow_pickle=True)
with open(lidar_data_file, 'rb') as f:
    lidar_frames = [np.load(f, allow_pickle=True) for _ in range(len(lidar_timestamps))]



if __name__ == '__main__':
    print("Displaying LiDAR data...")
    print(f"Number of LiDAR frames: {len(lidar_frames)}")

    idx = 0
    for idx_, lidar_frame in enumerate(lidar_frames):
        if len(lidar_frame) >0:
            print(f"First non-empty frame index: {idx_} with {len(lidar_frame)} points")
            idx = idx_
            break
    print(f"Starting from frame index: {idx}")
    print(f"Number of points in the first frame: {len(lidar_frames[idx])}")
    lidar_frames = lidar_frames[idx:]
    lidar_timestamps = lidar_timestamps[idx:]
    global_min_intensity = min([min(lidar_frame[:, 3]) for lidar_frame in lidar_frames if len(lidar_frame) > 0])
    global_max_intensity = max([max(lidar_frame[:, 3]) for lidar_frame in lidar_frames if len(lidar_frame) > 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    geometry = o3d.geometry.PointCloud()
    frame_data = lidar_frames[0]
    points = frame_data[:, :3]
    intensity = frame_data[:, 3]
    geometry.points = o3d.utility.Vector3dVector(points)
    intensity_normalized = (intensity - global_min_intensity) / (global_max_intensity - global_min_intensity)
    colors = plt.cm.viridis(intensity_normalized)[:, :3]  # Use viridis colormap
    geometry.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(geometry)
    
    def degree_to_pixel(degree):
        return degree/(0.003 *180/np.pi)
    
    vc = vis.get_view_control()
    # vc.rotate(0, degree_to_pixel(-90))
    # vc.rotate(degree_to_pixel(100), 0)
    # vc.rotate(0, degree_to_pixel(20))
    front = [ -0.92541657839832347, 0.1631759111665346, 0.34202014332566871 ]
    lookat = [ 16.341000000000001, -5.8939999999999992, -0.38849999999999996 ]
    up = [ 0.33682408883346515, -0.059391174613884559, 0.93969262078590854 ]
    vc.set_front(front)
    vc.set_lookat(lookat)
    vc.set_up(up)
    # _ = input("Press Enter to start the animation")
    for i in range(len(lidar_frames)):
        now = time.time()
        frame_data = lidar_frames[i]
        points = frame_data[:, :3]
        print(f"Frame {i+1} with {len(frame_data)} points")
        intensity = frame_data[:, 3]
        geometry.points = o3d.utility.Vector3dVector(points)
        intensity_normalized = (intensity - global_min_intensity) / (global_max_intensity - global_min_intensity)
        colors = plt.cm.viridis(intensity_normalized)[:, :3]  # Use viridis colormap
        geometry.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()
        # print(f"Frame {i+1} processed in {time.time() - now:.4f} seconds")
        # sleep(5)
        

    vis.destroy_window()
    exit()
