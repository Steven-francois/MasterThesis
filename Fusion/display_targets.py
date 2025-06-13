import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import open3d as o3d
from Fusion.association import to_cam_coord, to_lidar_coord, to_radar_coord, cam_size, cam_fov


nb = "11_0"
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
    fusion_frames = []
    CL_frames = []
    RL_frames = []
    for _ in range(num_frames):
        fusion_frames.append(np.load(f, allow_pickle=True))
        CL_frames.append(np.load(f, allow_pickle=True))
        RL_frames.append(np.load(f, allow_pickle=True))


def display_fusion_animation(cam_filenames, cam_frames, lidar_frames, radar_folder, fusion_frames, CL_frames, RL_frames, delay=0.1):
    # Create a figure with subplots for camera and radar frames
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
    
    ax_cam = fig.add_subplot(gs[0, :2])
    ax_radar = fig.add_subplot(gs[1, :2])
    ax_cam_error = fig.add_subplot(gs[0, 2])
    ax_radar_error = fig.add_subplot(gs[1, 2])
    
    
    
    # Create a 3D visualizer for LiDAR data
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
    
    # Error plots variables
    max_frame = 10
    frame_nb_targets = []
    dist = []
    error_CL = []
    error_RL = []
    
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

        fusion_frame = fusion_frames[frame]
        nb_targets = fusion_frame.shape[0]
        cam_idx, lidar_idx, radar_idx = fusion_frame.T.astype(int)
        print(cam_idx, lidar_idx, radar_idx)
        colors = plt.get_cmap("tab10")(np.arange(nb_targets) / nb_targets) if nb_targets > 0 else np.array([[0, 0, 0, 0]])
        CL_frame = CL_frames[frame]
        RL_frame = RL_frames[frame]
        nb_CL_targets = CL_frame.shape[0] if len(CL_frame) > 0 else 0
        nb_RL_targets = RL_frame.shape[0] if len(RL_frame) > 0 else 0
        nb_single_targets = nb_CL_targets + nb_RL_targets
        cam_CL_idx, lidar_CL_idx = CL_frame.T.astype(int) if len(CL_frame) > 0 else ([], [])
        radar_RL_idx, lidar_RL_idx = RL_frame.T.astype(int) if len(RL_frame) > 0 else ([], [])
        single_color = plt.get_cmap("Pastel1")(np.arange(nb_single_targets) / nb_single_targets) if nb_single_targets > 0 else np.array([[0, 0, 0, 0]])
        unassociated_color = np.array([0.5, 0.5, 0.5, 1])  # Gray for unassociated targets
        
        # Update error plots
        if len(frame_nb_targets) >= max_frame:
            for _ in range(frame_nb_targets[0]):
                del dist[0]
                del error_CL[0]
                del error_RL[0]
            del frame_nb_targets[0]
        frame_nb_targets.append(nb_targets)
        
        

        # Display camera frame
        ax_cam.imshow(plt.imread(os.path.join(camera_folder, cam_filenames[frame])), origin='upper')
        ax_cam.set_title(f'Camera Frame {frame}')
        cam_frame = cam_frames[frame]
        if len(cam_frame) > 0:
            for i, target in enumerate(cam_frame):
                x, y = target[0]-target[2]/2, target[1]-target[3]/2  # Center the rectangle
                if i in cam_idx:
                    edgecolor = colors[np.where(cam_idx == i)[0][0]]
                elif i in cam_CL_idx:
                    edgecolor = single_color[np.where(cam_CL_idx == i)[0][0]]
                else:
                    edgecolor = unassociated_color
                ax_cam.add_patch(plt.Rectangle((x,y), target[2], target[3], fill=False, edgecolor=edgecolor, linewidth=2))
                ax_cam.text(x, y, f'Target {i}', color='blue', fontsize=8)
        ax_cam.vlines(320, 0, 480, color='red', linestyle='--')  # Vertical line at center
        ax_cam.hlines(240, 0, 640, color='blue', linestyle='--')  # Horizontal line at center
        ax_cam.set_xlim(0, 640)
        ax_cam.set_ylim(480, 0)
        ax_cam.axis('off')
        
        cam_coords = np.array([to_cam_coord(target, cam_size, cam_fov) for target in cam_frame])
        
        # Display radar frame
        radar_file = os.path.join(radar_folder, f"targets_{frame}.json")
        with open(radar_file, 'r') as f:
            radar_frame = json.load(f)
            radar_targets = radar_frame["targets"]
            # radar_targets = np.array(radar_targets)[radar_idx] if len(radar_targets) > 0 and len(radar_idx) > 0 else []
        ax_radar.clear()
        ax_radar.set_title(f'Radar Targets Frame {frame}')
        if len(radar_targets) > 0:
            for i, target in enumerate(radar_targets):
                coord = to_radar_coord(target)
                x, y = coord[1], coord[0]
                if i in radar_idx:
                    color = colors[np.where(radar_idx == i)[0][0]]
                elif i in radar_RL_idx:
                    color = single_color[np.where(radar_RL_idx == i)[0][0]+nb_CL_targets]
                else:
                    color = unassociated_color
                ax_radar.scatter(x, y, c=color, label='Radar Target')
                ax_radar.text(x, y, f'Target {i}', color=color, fontsize=8)
        ax_radar.set_xlim(-50, 50)
        ax_radar.set_ylim(0, 90)
    
        radar_coords = np.array([to_radar_coord(target) for target in radar_targets])
        
        plt.tight_layout()
        
        # Update LiDAR point cloud
        lidar_frame = lidar_frames[frame]
        # nb_lidar_targets = int(np.max(lidar_frame[:, 3]))+1 if len(lidar_frame) > 0 else 0
        # lidar_targets = [None] * nb_lidar_targets
        # for i in range(nb_lidar_targets):
        #     lidar_targets[i] = lidar_frame[lidar_frame[:, 3] == i, :3]
        # points = np.concatenate([lidar_targets[i] for i in lidar_idx if i < nb_lidar_targets], axis=0) if len(lidar_idx) > 0 else np.zeros((0, 3))
        points = lidar_frame[:, :3] if len(lidar_frame) > 0 else np.zeros((0, 3))
        lidar_colors = []
        if len(lidar_frame) > 0:
            for i in range(len(lidar_frame)):
                target_id = lidar_frame[i, 3]
                if target_id in lidar_idx:
                    idx = np.where(lidar_idx == target_id)[0][0]
                    color = colors[idx][:3]
                elif target_id in lidar_CL_idx:
                    idx = np.where(lidar_CL_idx == target_id)[0][0]
                    color = single_color[idx][:3]

                elif target_id in lidar_RL_idx:
                    idx = np.where(lidar_RL_idx == target_id)[0][0] + nb_CL_targets
                    color = single_color[idx][:3]
                else:
                    color = unassociated_color[:3]
                lidar_colors.append(color)
            lidar_colors = np.array(lidar_colors)
        else:
            lidar_colors = np.zeros((0, 3))
        geometry.points = o3d.utility.Vector3dVector(points)
        geometry.colors = o3d.utility.Vector3dVector(lidar_colors)
        vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()
        
        nb_lidar_targets = int(np.max(lidar_frame[:, 3]))+1 if len(lidar_frame) > 0 else 0
        lidar_targets = np.zeros((nb_lidar_targets, 3))
        for i in range(nb_lidar_targets):
            lidar_targets[i, :3] = np.mean(lidar_frame[lidar_frame[:, 3] == i, :3], axis=0)
        lidar_coords = np.array([to_lidar_coord(target[:3]) for target in lidar_targets])
        
        for i in range(nb_targets):
            dist.append(lidar_coords[lidar_idx[i], 2])
            error_CL.append(np.linalg.norm(cam_coords[cam_idx[i]] - lidar_coords[lidar_idx[i], :2]))
            error_RL.append(np.linalg.norm(radar_coords[radar_idx[i], 0] - lidar_coords[lidar_idx[i], 2]))
            
        
        # Update error plots
        ax_cam_error.clear()
        ax_cam_error.plot(dist, error_CL, 'o', color='blue', label='Camera-LiDAR Error')
        ax_cam_error.set_title('Camera-LiDAR Error')
        ax_cam_error.set_xlabel('LiDAR Distance (m)')
        ax_cam_error.set_ylabel('Camera-LiDAR Error (°)')
        ax_cam_error.legend()
        ax_cam_error.set_xlim(0, 100)
        ax_cam_error.set_ylim(0, 10)
        ax_radar_error.clear()
        ax_radar_error.plot(dist, error_RL, 'o', color='red', label='Radar-LiDAR Error')
        ax_radar_error.set_title('Radar-LiDAR Error')
        ax_radar_error.set_xlabel('LiDAR Distance (m)')
        ax_radar_error.set_ylabel('Radar-LiDAR Error (m)')
        ax_radar_error.set_xlim(0, 100)
        ax_radar_error.set_ylim(0, 10)
        ax_radar_error.legend()
        
    
    ani = animation.FuncAnimation(fig, update, frames=len(fusion_frames), interval=delay * 1000)
    # ani = animation.FuncAnimation(fig, update, frames=range(200, 300), interval=delay * 1000)
    plt.show()
    
if __name__ == "__main__":
    display_fusion_animation(image_files, cam_frames, lidar_frames, radar_folder, fusion_frames, CL_frames, RL_frames, delay=0.1)
    
    # Optionally save the animation as a video
    # ani.save('fusion_animation.mp4', writer='ffmpeg', fps=10)
    
    # Display 3D point cloud for LiDAR data
    # pcd = o3d.geometry.PointCloud()
    # points = np.concatenate([frame[:, :3] for frame in lidar_frames if len(frame>0)], axis=0)
    # pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd])