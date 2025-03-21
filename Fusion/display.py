import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import os
from datetime import datetime
from Radar.RadarPacketPcapngReader import RadarPacketPcapngReader as RadarPacketReader
import open3d as o3d
import pandas as pd
from scipy.interpolate import CubicSpline

nb_file = "30"
rdc_file = f"Fusion/data/radar_cube_data_{nb_file}" # Replace with your output file path
image_folder = f"Fusion/captures{nb_file}/camera_rgba/"
plot_folder = f"Fusion/plots/fusion{nb_file}"
# lidar_file = f"Fusion/data/combination_{nb_file}"
lidar_file = f"Fusion/data/combination_20250312_123525"
lidar_data_file = f"{lidar_file}_data.npy"
lidar_ts_file = f"{lidar_file}_ts.npy"
speed_file = f'Fusion/captures{nb_file}/speed_test.csv'

image_filenames = os.listdir(image_folder)
images_timestamps = np.array([datetime.strptime(filename.split(".")[0], "%Y-%m-%d_%H-%M-%S-%f").timestamp() for filename in image_filenames])
os.makedirs(plot_folder, exist_ok=True)


rdc_reader = RadarPacketReader("", rdc_file)
rdc_reader.load()
radar_filename = rdc_reader.filename
fields = rdc_reader.fields
rdc_reader.interpolate_timestamps()
radar_timestamps = rdc_reader.timestamps/1e6
radar_time = rdc_reader.time
nb_frames = rdc_reader.nb_frames
all_properties = rdc_reader.all_properties
print(all_properties)
radar_cube_data = rdc_reader.radar_cube_datas
print(f"Fields: {fields}, Number of Frames: {nb_frames}, Number RDC: {len(radar_cube_data)}, Number Properties: {len(all_properties)}")

# Read speed data
speed_df = pd.read_csv(speed_file)
speed_df['Time'] = speed_df['Time'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp())
speed_df = speed_df.groupby('Time').mean()
# speed_df = speed_df.reindex(images_timestamps, method='nearest')
speed = speed_df['Speed (km/h)'].values
speed_timestamps = speed_df.index.values



# Read LiDAR data
lidar_timestamps = np.float64(np.load(lidar_ts_file, allow_pickle=True) - np.timedelta64(1, 'h'))/1e6
with open(lidar_data_file, 'rb') as f:
    lidar_frames = [np.load(f, allow_pickle=True) for _ in range(len(lidar_timestamps))]


# Get frame timestamps
radar_timestamps = radar_timestamps - radar_timestamps[0]
# radar_timestamps += datetime.strptime(radar_filename.split(".")[0][-15:], "%Y%m%d_%H%M%S").timestamp()
radar_timestamps += radar_time[0]
first_start_time = min(images_timestamps[0], radar_timestamps[0], speed_timestamps[0], lidar_timestamps[0])
image_frame_timestamps = images_timestamps - first_start_time
radar_frame_timestamps = radar_timestamps - first_start_time
speed_frame_timestamps = speed_timestamps - first_start_time
lidar_frame_timestamps = lidar_timestamps - first_start_time
last_start_time = max(image_frame_timestamps[0], radar_frame_timestamps[0], speed_frame_timestamps[0], lidar_frame_timestamps[0]) + 150

print(f"First Start Time: {first_start_time}, Last Start Time: {last_start_time}")
print(f"1st Image Timestamp: {image_frame_timestamps[0]}, 1st Radar Timestamp: {radar_frame_timestamps[0]}, 1st Speed Timestamp: {speed_frame_timestamps[0]}, 1st LiDAR Timestamp: {lidar_frame_timestamps[0]}")

first_camera_frame = np.where(image_frame_timestamps <= last_start_time)[0][-1]
image_frame_timestamps = image_frame_timestamps[first_camera_frame:]
image_filenames = image_filenames[first_camera_frame:]

first_radar_frame = np.where(radar_frame_timestamps <= last_start_time)[0][-1]
radar_frame_timestamps = radar_frame_timestamps[first_radar_frame:]
radar_cube_data = radar_cube_data[first_radar_frame:]
all_properties = all_properties[first_radar_frame:]
nb_frames = len(radar_frame_timestamps)

first_speed_frame = np.where(speed_frame_timestamps <= last_start_time)[0][-1]
speed_frame_timestamps = speed_frame_timestamps[first_speed_frame:]
speed = speed[first_speed_frame:]
interp_speed = CubicSpline(speed_frame_timestamps, speed, bc_type='natural')

first_lidar_frame = np.where(lidar_frame_timestamps <= last_start_time)[0][-1]
lidar_frame_timestamps = lidar_frame_timestamps[first_lidar_frame:]
lidar_frames = lidar_frames[first_lidar_frame:]





# Define radar cube dimensions
N_RANGE_GATES       = int(fields[6])    # 200
N_DOPPLER_BINS      = int(fields[8])    # 128
N_RX_CHANNELS       = int(fields[9])    # 8
N_CHIRP_TYPES       = int(fields[10])   # 2
FIRST_RANGE_GATE    = int(fields[7])    # 0
print(f"Range Gates: {N_RANGE_GATES}, Doppler Bins: {N_DOPPLER_BINS}, RX Channels: {N_RX_CHANNELS}, Chirp Types: {N_CHIRP_TYPES}")

# Define range doppler plot limits
xmin = -200/3.6
xmax = 200/3.6
ymin = 0
ymax = 98
# Define image size
image_size = (480, 640)


# Setup Matplotlib figure
fig = plt.figure(figsize=(10, 7))
# fig = plt.figure(figsize=(10, 14))
gs = gridspec.GridSpec(4,4)
gs.update(wspace=1, hspace=1)
ax = plt.subplot(gs[0:2, :])
ax2 = plt.subplot(gs[2:, :-1])
ax3 = plt.subplot(gs[2:, -1])
# fig, (ax, ax2) = plt.subplots(2, figsize=(10, 14))
img = ax.imshow(np.zeros((N_RANGE_GATES, N_DOPPLER_BINS)), vmin=0, vmax=100, aspect='auto', cmap='jet', origin='lower')
ax.set_xlabel("Doppler Bins (m/s)")
ax.set_ylabel("Range Gates (m)")
ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
ax.set_title("Real-Time Range-Doppler Map")
img2 = ax2.imshow(np.zeros(image_size))
fig.colorbar(img, label="Magnitude (dB)")
timestamp_text = ax.text(xmin, ymin, "", color="white", fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
speed_bar = ax3.bar(0.5, 0, width=2)
ax3.set(xlim=(0, 1), ylim=(0, 50), xticks=np.arange(0, 1, 1), yticks=np.arange(0, 50, 5))
ax3.set_title("Speed (km/h)")


global_min_intensity = min([min(lidar_frame[:, 3]) for lidar_frame in lidar_frames if len(lidar_frame) > 0])
global_max_intensity = max([max(lidar_frame[:, 3]) for lidar_frame in lidar_frames if len(lidar_frame) > 0])

vis = o3d.visualization.Visualizer()
vis.create_window()
front = [ -0.92541657839832347, 0.1631759111665346, 0.34202014332566871 ]
lookat = [ 16.341000000000001, -5.8939999999999992, -0.38849999999999996 ]
up = [ 0.33682408883346515, -0.059391174613884559, 0.93969262078590854 ]

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
vc.set_front(front)
vc.set_lookat(lookat)
vc.set_up(up)

# def interp_speed(target_timestamp):
#     new_speed = np.interp(target_timestamp, speed_frame_timestamps, speed)
#     return new_speed


rdm_idx = 0
speed_idx = 0
lidar_idx = 0

def process_radar_cube_data(range_doppler_matrix, angle=False):
    if angle:
        range_angle_matrix = np.fft.fftshift(np.fft.fft(range_doppler_matrix[:,N_DOPPLER_BINS//2,:,0], axis=1), axes=1)
        return np.abs(range_angle_matrix)
    
    # Sum over RX channels and chirps
    # range_doppler_matrix = np.fft.fft(range_doppler_matrix, axis=1)
    # range_doppler_matrix = np.sum(range_doppler_matrix[:,:,:,0], axis=2)
    # print(range_doppler_matrix.shape)
    range_doppler_matrix = range_doppler_matrix[:, :, :, 0]


    # Apply FFT along the Doppler axis
    # range_doppler_matrix = np.fft.fftshift(np.fft.fft(range_doppler_matrix, axis=0), axes=0)
    # range_doppler_matrix = np.fft.fftshift(np.fft.fft(range_doppler_matrix, axis=1), axes=1)
    range_doppler_matrix = np.mean(range_doppler_matrix, axis=2)
    # range_doppler_matrix = np.mean(range_doppler_matrix, axis=2)
    
    return np.abs(range_doppler_matrix)
    return 20 * np.log10(np.abs(range_doppler_matrix) + 1e-6)

def update(frame):
    global rdm_idx, speed_idx, lidar_idx, global_min_intensity, global_max_intensity
    if frame == 0:
        rdm_idx = 0
        speed_idx = 0
    # print(f"Frame: {frame}")
    # print(f"RDM Index: {rdm_idx}")
    # print(f"Image Timestamp: {image_frame_timestamps[frame]}")
    # print(f"Radar Timestamp: {radar_frame_timestamps[rdm_idx]}")
    # Get image data
    image = np.asarray(plt.imread(image_folder + image_filenames[frame]))
    img2.set_data(image)
    ax2.axis('off')
    # Set speed
    if image_frame_timestamps[frame] >= speed_frame_timestamps[speed_idx]:
        speed_bar[0].set_height(speed[speed_idx])
        # ax3.bar(0.5, speed[frame], width=1)
        if speed_idx < len(speed_frame_timestamps)-1:
            speed_idx += 1
    # Set LiDAR data
    if image_frame_timestamps[frame] >= lidar_frame_timestamps[lidar_idx]:
        # vis.clear_geometries()
        frame_data = lidar_frames[lidar_idx]
        points = frame_data[:, :3]
        intensity = frame_data[:, 3]
        geometry.points = o3d.utility.Vector3dVector(points)
        intensity_normalized = (intensity - global_min_intensity) / (global_max_intensity - global_min_intensity)
        colors = plt.cm.viridis(intensity_normalized)[:, :3]
        geometry.colors = o3d.utility.Vector3dVector(colors)
        vc.set_front(front)
        vc.set_lookat(lookat)
        vc.set_up(up)
        vis.add_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()
        if lidar_idx < len(lidar_frame_timestamps)-1:
            lidar_idx += 1
    if image_frame_timestamps[frame] < radar_frame_timestamps[rdm_idx]:
        return img, timestamp_text
    
    # Get radar cube data
    range_doppler_matrix = radar_cube_data[rdm_idx]
    # if range_doppler_matrix == 0:
    #     return img,
    # Define Bin Properties
    properties = all_properties[rdm_idx]
    DOPPLER_RESOLUTION  = properties[0]
    RANGE_RESOLUTION    = properties[1]
    BIN_PER_SPEED       = properties[2]
    xt_left = -N_DOPPLER_BINS//2*DOPPLER_RESOLUTION
    xt_right = (N_DOPPLER_BINS//2 - 1)*DOPPLER_RESOLUTION
    print(f"xt_left: {xt_left}, xt_right: {xt_right}")
    # Correction of speed
    radar_speed = interp_speed(radar_frame_timestamps[rdm_idx])
    xt_left += radar_speed/3.6
    xt_right += radar_speed/3.6
    yt_bottom = 0
    yt_top = N_RANGE_GATES*RANGE_RESOLUTION
    # fill_neg_bins = np.zeros((N_RANGE_GATES, int((xmin-xt_left)//DOPPLER_RESOLUTION)), dtype=np.float64)
    # fill_pos_bins = np.zeros((N_RANGE_GATES, int((xt_right-xmax)//DOPPLER_RESOLUTION)), dtype=np.float64)
    range_doppler_matrix = process_radar_cube_data(range_doppler_matrix)
    # range_doppler_matrix = np.concatenate((fill_pos_bins, range_doppler_matrix, fill_neg_bins), axis=1)
    img.set_data(range_doppler_matrix)
    # img.set_extent([xmax, xmin, yt_bottom, yt_top])
    img.set_extent([xt_left, xt_right, yt_bottom, yt_top])
    timestamp_text.set_text(f"Timestamp: {rdm_idx}")
    if rdm_idx < len(radar_cube_data)-1:
        rdm_idx += 1
        
    # if image_filenames[frame] in Ids:
    #     # idx = np.where(Ids == image_filenames[frame])[0][0]
    #     vis.clear_geometries()
    #     # filtered_df = filtered_df_list[idx]
    #     filtered_df = df[df['ID'] == image_filenames[frame]]
    #     points = filtered_df[['X', 'Y', 'Z']].values
    #     intensity = filtered_df['Intensity'].values
    #     geometry.points = o3d.utility.Vector3dVector(points)
    #     intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    #     colors = plt.cm.viridis(intensity_normalized)[:, :3]
    #     geometry.colors = o3d.utility.Vector3dVector(colors)
    #     vis.add_geometry(geometry)
    #     vis.poll_events()
    #     vis.update_renderer()
    return img, timestamp_text

def animate():
    ani = animation.FuncAnimation(fig, update, frames=len(image_filenames), interval=10)
    plt.show()

def save():
    ani = animation.FuncAnimation(fig, update, frames=len(image_filenames), interval=30)
    # plt.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\openadm\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-7.1-full_build\\bin\\ffmpeg.exe'
    # writervideo = animation.FFMpegWriter(fps=60) 
    # ani.save(f"rdms/rdm{nb_file}.mp4", writer=writervideo)
    ani.save(f"{plot_folder}/fusion{nb_file}.mp4", writer="ffmpeg", fps=60)
    
def save_png():
    for frame in range(len(image_filenames)):
        global rdm_idx
        if frame == 0:
            rdm_idx = 0
        print(f"Frame: {frame}")
        print(f"RDM Index: {rdm_idx}")
        print(f"Image Timestamp: {image_frame_timestamps[frame]}")
        print(f"Radar Timestamp: {radar_frame_timestamps[rdm_idx]}")
        # Get image data
        image = np.asarray(plt.imread(image_folder + image_filenames[frame]))
        img2.set_data(image)
        ax2.axis('off')
        if image_frame_timestamps[frame] < radar_frame_timestamps[rdm_idx]:
            return img, timestamp_text
        
        # Get radar cube data
        range_doppler_matrix = radar_cube_data[rdm_idx]
        # if range_doppler_matrix == 0:
        #     return img,
        range_doppler_matrix = process_radar_cube_data(range_doppler_matrix)
        img.set_data(range_doppler_matrix)
        # Define Bin Properties
        properties = all_properties[rdm_idx]
        DOPPLER_RESOLUTION  = properties[0]
        RANGE_RESOLUTION    = properties[1]
        BIN_PER_SPEED       = properties[2]
        xt_left = -N_DOPPLER_BINS//2*DOPPLER_RESOLUTION
        xt_right = (N_DOPPLER_BINS//2 - 1)*DOPPLER_RESOLUTION
        yt_bottom = 0
        yt_top = N_RANGE_GATES*RANGE_RESOLUTION
        img.set_extent([xt_left, xt_right, yt_bottom, yt_top])
        timestamp_text.set_text(f"Timestamp: {rdm_idx}")
        if rdm_idx < len(radar_cube_data)-1:
            rdm_idx += 1
        plt.savefig(f"{plot_folder}/f{frame}.png")


if __name__ == "__main__":
    # save()
    # save_png()
    # rdm_idx = 0
    animate()