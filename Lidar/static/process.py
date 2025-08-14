import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime, timedelta
import os
from Lidar.static.background import remove_background_with_voxels, remove_close_points

start_time, end_time  ="2025-05-14_14-04-33-934654", "2025-05-14_14-05-39-274261"

nb = "21_0"
data_folder = f"Data/{nb}/"
lidar_data_file = os.path.join(data_folder, "lidar_combined_data.npy")
lidar_ts_file = os.path.join(data_folder, "lidar_combined_ts.npy")
lidar_timestamps = np.load(lidar_ts_file, allow_pickle=True)
lidar_timestamps = np.array([ts + timedelta(hours=1) for ts in lidar_timestamps])
with open(lidar_data_file, 'rb') as f:
    lidar_frames = [np.load(f, allow_pickle=True) for _ in range(len(lidar_timestamps))]

lidar_foler = os.path.join(data_folder, "lidar")
os.makedirs(lidar_foler, exist_ok=True)

background_start = datetime.strptime(start_time, "%Y-%m-%d_%H-%M-%S-%f")
background_end = datetime.strptime(end_time, "%Y-%m-%d_%H-%M-%S-%f")


# background data
bg_idx_start = np.where(lidar_timestamps > background_start)[0][0]
bg_idx_end = np.where(lidar_timestamps > background_end)[0][0]
bg_lidar = lidar_frames[bg_idx_end]
for i in range(bg_idx_start, bg_idx_end):
    bg_lidar = np.concatenate((bg_lidar, lidar_frames[i]), axis=0)
print(f"Background lidar data shape: {bg_lidar.shape}")
bg_geometry = o3d.geometry.PointCloud()
bg_geometry.points = o3d.utility.Vector3dVector(bg_lidar[:, :3])
bg_geometry.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color for background
bg_geometry = o3d.geometry.VoxelGrid.create_from_point_cloud(bg_geometry, voxel_size=0.5)  # Downsample for background
# bg_lidar = np.asarray(bg_geometry.points)
o3d.visualization.draw_geometries([bg_geometry], window_name="Background Lidar Data", width=800, height=600)
o3d.io.write_voxel_grid(os.path.join(lidar_foler, "background.pcd"), bg_geometry)
    

# vis = o3d.visualization.Visualizer()
# vis.create_window()
geometry = o3d.geometry.PointCloud()
# vis.add_geometry(geometry)

# vc = vis.get_view_control()
# front = [ -0.92541657839832347, 0.1631759111665346, 0.34202014332566871 ]
# lookat = [ 16.341000000000001, -5.8939999999999992, -0.38849999999999996 ]
# up = [ 0.33682408883346515, -0.059391174613884559, 0.93969262078590854 ]
# vc.set_front(front)
# vc.set_lookat(lookat)
# vc.set_up(up)
# vis.poll_events()
# vis.update_renderer()
# parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2025-05-28-11-35-01.json")
# vc.convert_from_pinhole_camera_parameters(parameters)

with open(os.path.join(lidar_foler, "targets.npy"), 'wb') as f:
    np.save(f, len(lidar_frames))  # Save the number of results
    for lidar_frame in tqdm(lidar_frames, desc="Processing Lidar Frames"):
    # for lidar_frame in lidar_frames:
        geometry.points = o3d.utility.Vector3dVector(lidar_frame[:, :3])
        geometry.paint_uniform_color([0.1, 0.8, 0.1])  # Green color for current frame
        # Remove background
        geometry = remove_background_with_voxels(geometry, bg_geometry)
        
        labels = np.array(geometry.cluster_dbscan(eps=1.5, min_points=7))
        if len(labels) > 0:
            max_label = labels.max()
            colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0
            geometry.colors = o3d.utility.Vector3dVector(colors[:, :3])
            points = np.asarray(geometry.points)
            targets_cloud = np.hstack((points[labels>=0], labels[labels>=0, np.newaxis]))
            np.save(f, targets_cloud)
        else:
            np.save(f, [])
        
        # Create point cloud geometry
        # geometry = o3d.geometry.PointCloud()
        # geometry.points = o3d.utility.Vector3dVector(lidar_frame[:, :3])
        # geometry.paint_uniform_color([0.1, 0.8, 0.1])  # Green color for current frame
        
        # vis.add_geometry(geometry)
        # front = [ -0.92541657839832347, 0.1631759111665346, 0.34202014332566871 ]
        # lookat = [ 16.341000000000001, -5.8939999999999992, -0.38849999999999996 ]
        # up = [ 0.33682408883346515, -0.059391174613884559, 0.93969262078590854 ]
        # vc.set_front(front)
        # vc.set_lookat(lookat)
        # vc.set_up(up)
        # vis.poll_events()
        # vis.update_renderer()
        
        # if i % 10 == 0:  # Save every 10th frame
        #     o3d.io.write_point_cloud(os.path.join(data_folder, f"lidar_frame_{i}.pcd"), geometry)
