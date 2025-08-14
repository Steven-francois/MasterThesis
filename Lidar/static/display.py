import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os

nb = "11_0"
data_folder = f"Data/{nb}/"
lidar_targets = os.path.join(data_folder, "lidar", "targets.npy")

with open(lidar_targets, 'rb') as f:
    nb_frames = np.load(f, allow_pickle=True)
    frames = [np.load(f, allow_pickle=True) for _ in range(nb_frames)]


vis = o3d.visualization.Visualizer()
vis.create_window()
geometry = o3d.geometry.PointCloud()
test = [frame[:, :3] for frame in frames[:400] if frame is not None and len(frame) > 0]
print(f"Number of frames with points: {len(test)}")
print(f"First frame shape: {test[0].shape if test else 'No frames with points'}")
geometry.points = o3d.utility.Vector3dVector(np.vstack([frame[:, :3] for frame in frames[:1000] if frame is not None and len(frame) > 0]))
geometry.paint_uniform_color([0.1, 0.8, 0.1])  # Set initial color
vis.add_geometry(geometry)
o3d.visualization.draw_geometries([geometry], window_name="Background Lidar Data", width=800, height=600)


vc = vis.get_view_control()
front = [ -0.92541657839832347, 0.1631759111665346, 0.34202014332566871 ]
lookat = [ 16.341000000000001, -5.8939999999999992, -0.38849999999999996 ]
up = [ 0.33682408883346515, -0.059391174613884559, 0.93969262078590854 ]
vc.set_front(front)
vc.set_lookat(lookat)
vc.set_up(up)
vis.poll_events()
vis.update_renderer()
for i in range(nb_frames):
    points = frames[i]
    if points is None or len(points) == 0:
        print(f"Frame {i+1}/{nb_frames}: No points found, skipping...")
        continue
    print(f"Frame {i+1}/{nb_frames}: {points.shape}")
    # geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(points[:, :3])
    labels = points[:, 3]
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    # print(colors)
    geometry.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # geometry.paint_uniform_color([0.1, 0, 0.5]) 

    
    vis.update_geometry(geometry)
    # front = [ -0.92541657839832347, 0.1631759111665346, 0.34202014332566871 ]
    # lookat = [ 16.341000000000001, -5.8939999999999992, -0.38849999999999996 ]
    # up = [ 0.33682408883346515, -0.059391174613884559, 0.93969262078590854 ]
    # vc.set_front(front)
    # vc.set_lookat(lookat)
    # vc.set_up(up)
    vis.poll_events()
    vis.update_renderer()