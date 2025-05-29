import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import open3d as o3d
import time
from tqdm import tqdm
import os




pixel_size = 2.2e-6  # Pixel size in meters
focal_length = 5695.8 * pixel_size  # Focal length in meters
image_real_width = 640*pixel_size
image_real_height = 480*pixel_size
image_size = (480, 640)

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
    if len(queries) == 0:
        return pcd
    
    # Run the voxel inclusion test
    inclusion_mask = background_voxels.check_if_included(
        o3d.utility.Vector3dVector(queries)
    )
    inclusion_mask = np.array(inclusion_mask)  # Boolean array
    
    # Keep only points NOT in background
    foreground_points = queries[~inclusion_mask]
    
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

def isect_line_plane_v3(p1, epsilon=1e-6, ip=False):
    if ip:
        p0 = np.array([0, 0, 0])
        p_co = np.array([focal_length, 0, 0])
        p_no = np.array([1, 0, 0])
        u = p1 - p0
        dot = np.dot(p_no, u)
        if abs(dot) > epsilon:
            w = p0 - p_co
            fac = -np.dot(p_no, w) / dot
            return p0 + (u * fac)
        return None
    else:
        p = np.array([p1[0], p1[1], p1[2], 0])
        P = np.array([[1, 0, 0, 0],
                        [0, focal_length, 0, 0],
                        [0, 0, focal_length, 0]])
        
        cam_p = np.dot(P, p.T)
        cam_p = cam_p[1:] #/ cam_p[0]
        return cam_p

def X_coord_to_pixel(x):
    return x/pixel_size + image_real_width/2/pixel_size
def Y_coord_to_pixel(y):
    return y/pixel_size + image_real_height/2/pixel_size

if __name__ == '__main__':
    # Read the data
    image_folder = 'Fusion/data/31/camera/'
    lidar_file = 'Fusion/data/31/lidar_combined'
    lidar_data_file = f"{lidar_file}_data.npy"
    lidar_ts_file = f"{lidar_file}_ts.npy"
    lidar_timestamps = np.load(lidar_ts_file, allow_pickle=True)
    with open(lidar_data_file, 'rb') as f:
        lidar_frames = [np.load(f, allow_pickle=True) for _ in range(len(lidar_timestamps))]
    image_filenames = os.listdir(image_folder)
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
    # image_filenames = image_filenames[idx+900:idx+1107]
    image_filenames = image_filenames[idx:idx+10]
    global_min_intensity = min([min(lidar_frame[:, 3]) for lidar_frame in lidar_frames if len(lidar_frame) > 0])
    global_max_intensity = max([max(lidar_frame[:, 3]) for lidar_frame in lidar_frames if len(lidar_frame) > 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    geometry = o3d.geometry.PointCloud()
    frame_data = lidar_frames[280]
    for i in range(248, 280):
        frame_data = np.concat((frame_data, lidar_frames[i]), axis=0)
    points = frame_data[:, :3]
    intensity = frame_data[:, 3]
    geometry.points = o3d.utility.Vector3dVector(points)
    intensity_normalized = (intensity - global_min_intensity) / (global_max_intensity - global_min_intensity)
    colors = plt.cm.viridis(intensity_normalized)[:, :3]  # Use viridis colormap
    geometry.colors = o3d.utility.Vector3dVector(colors)
    # bg_pcd = geometry.voxel_down_sample(voxel_size=1)
    geometry.paint_uniform_color([1, 0, 0])
    bg_pcd = o3d.geometry.VoxelGrid.create_from_point_cloud(geometry, voxel_size=0.9)
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
    
    
    
    frame_data = lidar_frames[280]
    points = frame_data[:, :3]
    intensity = frame_data[:, 3]
    geometry.points = o3d.utility.Vector3dVector(points)
    intensity_normalized = (intensity - global_min_intensity) / (global_max_intensity - global_min_intensity)
    colors = plt.cm.viridis(intensity_normalized)[:, :3]  # Use viridis colormap
    geometry.colors = o3d.utility.Vector3dVector(colors)
    geometry = remove_background_with_voxels(geometry, bg_pcd)
    o3d.visualization.draw_geometries([geometry, bg_pcd])
    
    
    
    
    frame_points = []
    frame_colors = []
    # _ = input("Press Enter to start the animation")
    for i in range(280, 380):
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
        
        labels = np.array(geometry.cluster_dbscan(eps=2, min_points=10))
        if len(labels) > 0:
            max_label = labels.max()
            print(f"point cloud has {max_label + 1} clusters")
            colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0
            geometry.colors = o3d.utility.Vector3dVector(colors[:, :3])
            points = np.asarray(geometry.points)
            # Remove points that are not in the clusters
            # geometry.points = o3d.utility.Vector3dVector(points)
            # geometry.colors = o3d.utility.Vector3dVector(colors[:, :3])
            frame_points.append(points[labels >= 0])
            frame_colors.append(colors[labels >= 0])
        vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()
        
        # print(f"Frame {i+1} processed in {time.time() - now:.4f} seconds")
        # sleep(5)
        
    _ = input("Press Enter to continue")
    vis.destroy_window()
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # geometry = o3d.geometry.PointCloud()
    # geometry.points = o3d.utility.Vector3dVector(frame_points[0])
    # geometry.colors = o3d.utility.Vector3dVector(frame_colors[0][:, :3])
    # vis.add_geometry(geometry)
    # _ = input(f"Press Enter to start the animation {len(frame_points)} frames")
    # for i in range(len(frame_points)):
    #     geometry.points = o3d.utility.Vector3dVector(frame_points[i])
    #     geometry.colors = o3d.utility.Vector3dVector(frame_colors[i][:, :3])
    #     vis.add_geometry(geometry)
    #     vis.update_geometry(geometry)
    #     vis.poll_events()
    #     vis.update_renderer()
    

    # Setup Matplotlib figure
    fig = plt.figure(figsize=(10, 7))
    # fig = plt.figure(figsize=(10, 14))
    gs = gridspec.GridSpec(2,2)
    gs.update(wspace=1, hspace=1)
    ax = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[1, 0])
    ax2 = plt.subplot(gs[:, 1])
    img2 = ax2.imshow(np.zeros(image_size))
    
    factor = np.linspace(3, 6, 10)
    fr = 0
    def frame(i):
        global fr
        ax.clear()
        # ax.scatter(frame_points[i][:, 0], frame_points[i][:, 1], c=frame_colors[i][:, :3], s=2)
        ax.scatter(frame_points[i][:, 1], frame_points[i][:, 2], c=frame_colors[i][:, :3], s=2)
        
        ax.set_title(f"Frame {i+1}")
        ax.set_ylim(-5, 5)
        ax.set_xlim(-20, 20)
        new_points = np.array([isect_line_plane_v3(frame_points[i][j], ip=True) for j in range(len(frame_points[i]))])
        
        ax1.clear()
        ax2.clear()
        if len(new_points) > 0:
            ax1.scatter(new_points[:, 1], new_points[:, 2], c=frame_colors[i][:, :3], s=1)
            ax2.scatter(- X_coord_to_pixel(new_points[:, 1]), Y_coord_to_pixel(new_points[:, 2]), c=frame_colors[i][:, :3], s=1)
        ax1.set_ylim(-image_real_height*8/2, image_real_height*8/2)
        ax1.set_xlim(-image_real_width*8/2, image_real_width*8/2)
        image = np.asarray(plt.imread(image_folder + image_filenames[i]))
        # img2.set_data(image)
        ax2.imshow(image, extent=[-factor[(fr%len(frame_points))%len(factor)]*image_size[1], factor[(fr%len(frame_points))%len(factor)]*image_size[1], -factor[(fr%len(frame_points))%len(factor)]*image_size[0], factor[(fr%len(frame_points))%len(factor)]*image_size[0]], aspect='auto')
        ax2.set_ylim(-8*image_size[0], 8*image_size[0])
        ax2.set_xlim(-8*image_size[1], 8*image_size[1])
        ax2.set_title(f"Image {i+1}/{len(image_filenames)} [factor {factor[(fr%len(frame_points))%len(factor)]:.2f}]")
        fr+=1
        return ax, ax1, img2
    ani = animation.FuncAnimation(fig, frame, frames=len(frame_points), interval=100)
    plt.show()
    # vis.destroy_window()
    


