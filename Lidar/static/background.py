import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import time
from tqdm import tqdm

# Read the data
lidar_file = 'Fusion/data/2/lidar_combined'
lidar_data_file = f"{lidar_file}_data.npy"
lidar_ts_file = f"{lidar_file}_ts.npy"
lidar_timestamps = np.load(lidar_ts_file, allow_pickle=True)
with open(lidar_data_file, 'rb') as f:
    lidar_frames = [np.load(f, allow_pickle=True) for _ in range(len(lidar_timestamps))]


def remove_background_with_voxels(pcd, background_voxels):
    """
    Removes points from `pcd` that are included in `background_voxels` using voxel inclusion test.
    
    Parameters:
    - pcd: open3d.geometry.PointCloud
    - background_voxels: open3d.geometry.VoxelGrid
    
    Returns:
    - foreground_pcd: open3d.geometry.PointCloud
    """
    
    # Convert point cloud points to numpy
    queries = np.asarray(pcd.points)
    
    # Run the voxel inclusion test
    inclusion_mask = background_voxels.check_if_included(
        o3d.utility.Vector3dVector(queries)
    )
    inclusion_mask = np.array(inclusion_mask)  # Boolean array
    print(f"Number of points in background: {np.sum(inclusion_mask)}/{len(inclusion_mask)}")
    
    # Keep only points NOT in background
    foreground_points = queries[~inclusion_mask]
    print(f"Number of points in foreground: {len(foreground_points)}")
    
    # Create new point cloud
    # foreground_pcd = o3d.geometry.PointCloud()
    # foreground_pcd.points = o3d.utility.Vector3dVector(foreground_points)
    pcd.points = o3d.utility.Vector3dVector(foreground_points)
    
    # Optional: filter colors if present
    # if pcd.has_colors():
    #     print("Filtering colors...")
    #     colors = np.asarray(pcd.colors)
    #     foreground_pcd.colors = o3d.utility.Vector3dVector(colors[~inclusion_mask])
    
    colors = np.asarray(pcd.colors)
    pcd.colors = o3d.utility.Vector3dVector(colors[~inclusion_mask])
    
    return pcd



def remove_close_points(point, points, distance_threshold):
    distances = np.linalg.norm(points[:, :3] - point[:3], axis=1)  # Vectorized distance calculation
    return points[distances > distance_threshold]  # Boolean mask to filter points

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
    frame_data = lidar_frames[900]
    for i in range(600, 900):
        frame_data = np.concat((frame_data, lidar_frames[i]), axis=0)
    points = frame_data[:, :3]
    intensity = frame_data[:, 3]
    geometry.points = o3d.utility.Vector3dVector(points)
    intensity_normalized = (intensity - global_min_intensity) / (global_max_intensity - global_min_intensity)
    colors = plt.cm.viridis(intensity_normalized)[:, :3]  # Use viridis colormap
    geometry.colors = o3d.utility.Vector3dVector(colors)
    # bg_pcd = geometry.voxel_down_sample(voxel_size=1)
    geometry.paint_uniform_color([1, 0, 0])
    bg_pcd = o3d.geometry.VoxelGrid.create_from_point_cloud(geometry, voxel_size=1)
    # vis.add_geometry(bg_pcd)
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
    
    
    
    frame_data = lidar_frames[1100]
    points = frame_data[:, :3]
    intensity = frame_data[:, 3]
    geometry.points = o3d.utility.Vector3dVector(points)
    intensity_normalized = (intensity - global_min_intensity) / (global_max_intensity - global_min_intensity)
    colors = plt.cm.viridis(intensity_normalized)[:, :3]  # Use viridis colormap
    geometry.colors = o3d.utility.Vector3dVector(colors)
    geometry = remove_background_with_voxels(geometry, bg_pcd)
    o3d.visualization.draw_geometries([geometry, bg_pcd])
    
    
    
    
    
    # _ = input("Press Enter to start the animation")
    for i in range(900, 1117):
        now = time.time()
        frame_data = lidar_frames[i]
        points = frame_data[:, :3]
        print(f"Frame {i+1} with {len(frame_data)} points")
        intensity = frame_data[:, 3]
        geometry.points = o3d.utility.Vector3dVector(points)
        intensity_normalized = (intensity - global_min_intensity) / (global_max_intensity - global_min_intensity)
        colors = plt.cm.viridis(intensity_normalized)[:, :3]  # Use viridis colormap
        geometry.colors = o3d.utility.Vector3dVector(colors)
        geometry = remove_background_with_voxels(geometry, bg_pcd)
        
        labels = np.array(geometry.cluster_dbscan(eps=1, min_points=10))
        if len(labels) > 0:
            max_label = labels.max()
            print(f"point cloud has {max_label + 1} clusters")
            colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0
            geometry.colors = o3d.utility.Vector3dVector(colors[:, :3])
        vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()
        
        # print(f"Frame {i+1} processed in {time.time() - now:.4f} seconds")
        # sleep(5)
        

    vis.destroy_window()
    exit()
