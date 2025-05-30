import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import open3d as o3d


nb = "1_0"
data_folder = f"Data/{nb}/"
# data_folder = f"D:/processed"
camera_folder = os.path.join(data_folder, "camera")
camera_target_folder = os.path.join(data_folder, "cam_targets")
lidar_folder = os.path.join(data_folder, "lidar")
radar_folder = os.path.join(data_folder, "radar", "targets")
fusion_folder = os.path.join(data_folder, "fusion")
image_files = [f for f in sorted(os.listdir(camera_folder)) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
with open(os.path.join(camera_target_folder, "targets.npy"), "rb") as f:
    num_frames = np.load(f, allow_pickle=True)
    cam_frames = [np.load(f, allow_pickle=True) for _ in range(num_frames)]
with open(os.path.join(lidar_folder, "targets.npy"), "rb") as f:
    num_frames = np.load(f, allow_pickle=True)
    lidar_frames = [np.load(f, allow_pickle=True) for _ in range(num_frames)]
with open(os.path.join(fusion_folder, "targets.npy"), "rb") as f:
    num_frames = np.load(f, allow_pickle=True)
    fusion_frames = [np.load(f, allow_pickle=True) for _ in range(num_frames)]


def display_fusion_animation(cam_filenames, cam_frames, lidar_frames, radar_folder, fusion_frames, delay=0.1):
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    ax_cam = fig.add_subplot(gs[0, :])
    ax_radar = fig.add_subplot(gs[1, :])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    geometry = o3d.geometry.PointCloud()
    points = np.concatenate([frame[:, :3] for frame in lidar_frames if len(frame>0)], axis=0)
    geometry.points = o3d.utility.Vector3dVector(points)
    vis.add_geometry(geometry)
    vc = vis.get_view_control()
    front = [ -0.92541657839832347, 0.1631759111665346, 0.34202014332566871 ]
    lookat = [ 16.341000000000001, -5.8939999999999992, -0.38849999999999996 ]
    up = [ 0.33682408883346515, -0.059391174613884559, 0.93969262078590854 ]
    vc.set_front(front)
    vc.set_lookat(lookat)
    vc.set_up(up)
    vis.poll_events()
    vis.update_renderer()
    
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
    
    def update(frame):
        ax_cam.clear()
        ax_radar.clear()
        
        # Display camera frame
        ax_cam.imshow(plt.imread(os.path.join(camera_folder, cam_filenames[frame])), origin='upper')
        ax_cam.set_title(f'Camera Frame {frame}')
        cam_frame = cam_frames[frame]
        if len(cam_frame) > 0:
            for i, target in enumerate(cam_frame):
                x, y = target[0]-target[2]/2, target[1]-target[3]/2  # Center the rectangle
                ax_cam.add_patch(plt.Rectangle((x,y), target[2], target[3], fill=False, edgecolor='blue', linewidth=2))
                ax_cam.text(x, y, f'Target {i}', color='blue', fontsize=8)
        ax_cam.set_xlim(0, 640)
        ax_cam.set_ylim(480, 0)
        ax_cam.axis('off')
        
        # Display radar frame
        radar_file = os.path.join(radar_folder, f"targets_{frame}.json")
        with open(radar_file, 'r') as f:
            radar_targets = json.load(f)
        ax_radar.clear()
        ax_radar.set_title(f'Radar Targets Frame {frame}')
        if len(radar_targets) > 0:
            for i, target in enumerate(radar_targets):
                coord = to_radar_coord(target)
                x, y = coord[1], coord[0]
                ax_radar.scatter(x, y, c='orange', label='Radar Target')
                ax_radar.text(x, y, f'Target {i}', color='orange', fontsize=8)
        ax_radar.set_xlim(-50, 50)
        ax_radar.set_ylim(0, 90)
        
        plt.tight_layout()
        
        # Update LiDAR point cloud
        points = lidar_frames[frame][:, :3] if len(lidar_frames[frame]) > 0 else np.zeros((0, 3))
        geometry.points = o3d.utility.Vector3dVector(points)
        vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()
    
    ani = animation.FuncAnimation(fig, update, frames=len(cam_frames), interval=delay * 1000)
    plt.show()
    
if __name__ == "__main__":
    display_fusion_animation(image_files, cam_frames, lidar_frames, radar_folder, fusion_frames, delay=0.1)
    
    # Optionally save the animation as a video
    # ani.save('fusion_animation.mp4', writer='ffmpeg', fps=10)
    
    # Display 3D point cloud for LiDAR data
    # pcd = o3d.geometry.PointCloud()
    # points = np.concatenate([frame[:, :3] for frame in lidar_frames if len(frame>0)], axis=0)
    # pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd])