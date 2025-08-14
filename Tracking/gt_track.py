import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

def ground_truth_tracking(cam_filenames, cam_frames, fusion_frames, tracks_folder):
    # Implement your ground truth tracking logic here
    fig, ax_cam = plt.subplots()
    ax_cam.set_title("Ground Truth Tracking")
    
    target_color = np.array([0.0, 0.0, 0.0, 1])
    unassociated_color = np.array([0.5, 0.5, 0.5, 1])
    
    def update(frame):
        ax_cam.clear()
        fusion_frame = fusion_frames[frame]
        nb_targets = fusion_frame.shape[0]
        cam_idx, lidar_idx, radar_idx = fusion_frame.T.astype(int)
        
        ax_cam.imshow(plt.imread(os.path.join(camera_folder, cam_filenames[frame])), origin='upper')
        ax_cam.set_title(f'Camera Frame {frame}')
        cam_frame = cam_frames[frame]
        if len(cam_frame) > 0:
            for i, target in enumerate(cam_frame):
                x, y = target[0]-target[2]/2, target[1]-target[3]/2  # Center the rectangle
                if i in cam_idx:
                    edgecolor = target_color
                    idx = np.where(cam_idx == i)[0][0]
                    ax_cam.add_patch(plt.Rectangle((x,y), target[2], target[3], fill=False, edgecolor=edgecolor, linewidth=2))
                    ax_cam.text(x, y, f'Target {idx}', color='blue', fontsize=15)
        # ax_cam.vlines(320, 0, 480, color='red', linestyle='--')  # Vertical line at center
        # ax_cam.hlines(240, 0, 640, color='blue', linestyle='--')  # Horizontal line at center
        ax_cam.set_xlim(0, 640)
        ax_cam.set_ylim(480, 0)
        ax_cam.axis('off')
        with open(os.path.join(tracks_folder, f"frame_{frame:04d}.json"), 'w') as f:
            frame_tracks = []
            for i in range(len(cam_idx)):
                id = input(f"Enter ID for target {i} in frame {frame}: ")
                frame_tracks.append(int(id))
            json.dump({"tracks": frame_tracks}, f)

    # ani = animation.FuncAnimation(fig, update, frames=len(fusion_frames), repeat=False)
    ani = animation.FuncAnimation(fig, update, frames=range(300, 540), repeat=False)
    plt.show()

if __name__ == "__main__":
    nb = "11_0"
    data_folder = f"Data/{nb}/"
    # data_folder = f"D:/p_{nb}/"
    camera_folder = os.path.join(data_folder, "camera")
    camera_target_folder = os.path.join(data_folder, "cam_targets")
    lidar_folder = os.path.join(data_folder, "lidar")
    radar_folder = os.path.join(data_folder, "radar", "targets")
    fusion_folder = os.path.join(data_folder, "fusion")
    image_files = [f for f in sorted(os.listdir(camera_folder)) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    with open(os.path.join(camera_target_folder, "targets.npy"), "rb") as f:
        num_frames = np.load(f, allow_pickle=True)
        cam_frames = [np.load(f, allow_pickle=True) for _ in range(num_frames)]
    with open(os.path.join(fusion_folder, "targets.npy"), "rb") as f:
        num_frames = np.load(f, allow_pickle=True)
        fusion_frames = []
        CL_frames = []
        RL_frames = []
        for _ in range(num_frames):
            fusion_frames.append(np.load(f, allow_pickle=True))
            CL_frames.append(np.load(f, allow_pickle=True))
            RL_frames.append(np.load(f, allow_pickle=True))
    tracks_folder = os.path.join(data_folder, "tracks")
    os.makedirs(tracks_folder, exist_ok=True)

    ground_truth_tracking(image_files, cam_frames, fusion_frames, tracks_folder)
