import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import open3d as o3d
import time

# Read the data
df = pd.read_csv('combine0715.csv')
Ids = df['ID'].unique()
x_min = df['X'].min()
x_max = df['X'].max()
y_min = df['Y'].min()
y_max = df['Y'].max()


vis = o3d.visualization.Visualizer()
vis.create_window()

geometry = o3d.geometry.PointCloud()
filtered_df = df[df['ID'] == Ids[0]]
points = filtered_df[['X', 'Y', 'Z']].values
intensity = filtered_df['Intensity'].values
geometry.points = o3d.utility.Vector3dVector(points)
intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())
colors = plt.cm.viridis(intensity_normalized)[:, :3]  # Use viridis colormap
geometry.colors = o3d.utility.Vector3dVector(colors)
vis.add_geometry(geometry)
filtered_df_list = [df[df['ID'] == Ids[i]] for i in range(len(Ids))]

for i in range(len(Ids)):
    print(Ids[i])
    now = time.time()
    # filtered_df = df[df['ID'] == Ids[i]]
    points = filtered_df_list[i][['X', 'Y', 'Z']].values
    intensity = filtered_df_list[i]['Intensity'].values
    geometry.points = o3d.utility.Vector3dVector(points)
    intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    colors = plt.cm.viridis(intensity_normalized)[:, :3]  # Use viridis colormap
    geometry.colors = o3d.utility.Vector3dVector(colors)
    vis.update_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()
    print(time.time() - now)
    # time.sleep(3)
    

vis.destroy_window()
exit()
