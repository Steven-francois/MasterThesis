from Radar.RadarCanReader import RadarCanReader
import open3d as o3d
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from datetime import datetime

def coordinate(target_data):
    z = np.sin(2*np.pi/360*target_data.el_angle) * target_data.t_range
    xy_plane = np.cos(2*np.pi/360*target_data.el_angle) * target_data.t_range
    x = -xy_plane * np.sin(2*np.pi/360*target_data.az_angle)
    y = xy_plane * np.cos(2*np.pi/360*target_data.az_angle)
    return np.array([x, y, z])


nb_file = "1"
can_file = f"Fusion/data/{nb_file}/radar_can_data.npy" # Replace with your output file path
image_folder = f"Fusion/data/{nb_file}/camera/"
# nb_file = "41"
# can_file = f"Fusion/data/can_data_{nb_file}.npy" # Replace with your output file path
# image_folder = "Fusion/DATA_20250417_153534/20250417_154040/camera_rgba/"

canReader = RadarCanReader()
canReader.load_npy(can_file)

image_filenames = os.listdir(image_folder)
images_timestamps = np.array([datetime.strptime(filename.split(".")[0], "%Y-%m-%d_%H-%M-%S-%f").timestamp() for filename in image_filenames])
image_size = (480, 640)


can_nb = np.where(np.array([canReader.can_targets[i].targets_header.real_time for i in range(len(canReader.can_targets))]) > images_timestamps[0])[0][0]
# print(f"First can: {first_can}")
# print(f"First can timestamp: {canReader.can_targets[first_can].targets_header.real_time}")
# print(f"First image timestamp: {images_timestamps[0]}")
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
img = ax.imshow(np.zeros(image_size))


vis = o3d.visualization.Visualizer()
vis.create_window()

# axis

line = o3d.geometry.PointCloud()
line.points = o3d.utility.Vector3dVector([[0, 0, i] for i in range(0, 21)])
line.paint_uniform_color([1, 0, 0])
vis.add_geometry(line)
line = o3d.geometry.PointCloud()
line.points = o3d.utility.Vector3dVector([[0, i, 0] for i in range(0, 21)])
line.paint_uniform_color([1, 0, 0])
vis.add_geometry(line)
line = o3d.geometry.PointCloud()
line.points = o3d.utility.Vector3dVector([[i, 0, 0] for i in range(0, 21)])
line.paint_uniform_color([1, 0, 0])
vis.add_geometry(line)



geometry = o3d.geometry.PointCloud()
points = np.array([coordinate(target_data) for target_data in canReader.can_targets[can_nb].targets_data])
geometry.points = o3d.utility.Vector3dVector(points)
vis.add_geometry(geometry)
# for i in range(1, len(canReader.can_targets)):
#     points = np.array([coordinate(target_data) for target_data in canReader.can_targets[i].targets_data if abs(target_data.speed_radial) > 0])
#     geometry.points = o3d.utility.Vector3dVector(points)
#     vis.update_geometry(geometry)
#     vis.poll_events()
#     vis.update_renderer()
#     # sleep(1)

def frame(i):
    global can_nb
    if images_timestamps[i] > canReader.can_targets[can_nb].targets_header.real_time:
        can_nb += 1
    points = np.array([coordinate(target_data) for target_data in canReader.can_targets[can_nb].targets_data if abs(target_data.speed_radial) > 1.3])
    if len(points) == 0:
        points = np.array([[0, 0, 0]])
    geometry.points = o3d.utility.Vector3dVector(points)
    vis.update_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()
    image = np.asarray(plt.imread(image_folder + image_filenames[i]))
    img.set_data(image)
    ax.set_title(f"Image {i+1}/{len(image_filenames)}")
    # sleep(1)
    
    return img, geometry

if __name__ == "__main__":
    # ani = animation.FuncAnimation(fig, frame, frames=len(image_filenames), interval=1000/30)
    # plt.show()
    # canReader.cluster_with_dbscan(5, 2, 2)
    ani = animation.FuncAnimation(fig, frame, frames=len(image_filenames), interval=1000/30)
    plt.show()
    vis.run()
    vis.destroy_window()
    
    # for i in range(len(image_filenames)):
    #     pcd = o3d.geometry.PointCloud()
    #     points = np.array([coordinate(target_data) for target_data in canReader.can_targets[can_nb+i].targets_data if abs(target_data.speed_radial) > 1.3])
    #     if len(points) == 0:
    #         points = np.array([[0, 0, 0]])
    #     pcd.points = o3d.utility.Vector3dVector(points)

    #     with o3d.utility.VerbosityContextManager(
    #             o3d.utility.VerbosityLevel.Debug) as cm:
    #         labels = np.array(
    #             pcd.cluster_dbscan(eps=2, min_points=2, print_progress=True))

    #     max_label = labels.max()
    #     print(f"point cloud has {max_label + 1} clusters")
    #     colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    #     colors[labels < 0] = 0
    #     pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    #     if max_label >= 0:
    #     #     pcd = pcd.select_by_index(np.where(labels == 0)[0])
    #     # else:
    #     #     pcd = pcd.select_by_index(np.where(labels == 1)[0])
    #         axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])
    #         o3d.visualization.draw_geometries([pcd, axis])