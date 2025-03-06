import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from datetime import datetime
from Radar.RadarPacketPcapngReader import RadarPacketPcapngReader as RadarPacketReader

nb_file = "20"
rdc_file = f"Fusion/data/radar_cube_data_{nb_file}" # Replace with your output file path
image_folder = "Fusion/captures/camera_rgba/"
plot_folder = f"Fusion/plots/fusion{nb_file}"

image_filenames = os.listdir(image_folder)
images_timestamps = np.array([datetime.strptime(filename.split(".")[0], "%Y-%m-%d_%H-%M-%S-%f").timestamp() for filename in image_filenames])
# temp_timestamps = np.array(images_timestamps)
# temp_timestamps = np.append(temp_timestamps[0], temp_timestamps[:-1])
# image_frame_timestamps = images_timestamps - temp_timestamps
image_frame_timestamps = images_timestamps - images_timestamps[0]
os.makedirs(plot_folder, exist_ok=True)


rdc_reader = RadarPacketReader("", rdc_file)
rdc_reader.load()
fields = rdc_reader.fields
radar_timestamps = rdc_reader.timestamps
nb_frames = rdc_reader.nb_frames
all_properties = rdc_reader.all_properties
radar_cube_data = rdc_reader.radar_cube_datas
print(f"Fields: {fields}, Number of Frames: {nb_frames}, Number RDC: {len(radar_cube_data)}, Number Properties: {len(all_properties)}")


# with open(rdc_file, 'rb') as f:
#     fields = np.load(f, allow_pickle=True)
#     radar_timestamps = np.load(f, allow_pickle=True)
#     nb_frames = np.load(f, allow_pickle=True)
#     all_properties = np.array([np.load(f, allow_pickle=True)])
#     for _ in range(nb_frames-1):
#         all_properties = np.append(all_properties, [np.load(f, allow_pickle=True)], axis=0)
#     radar_cube_data = np.array([np.load(f, allow_pickle=True)])
#     for _ in range(nb_frames-1):
#     # for _ in range(200-1):
#         radar_cube_data = np.append(radar_cube_data, [np.load(f, allow_pickle=True)], axis=0)

# temp_timestamps = np.array(radar_timestamps)
# temp_timestamps = np.append(temp_timestamps[0], temp_timestamps[:-1])
# radar_frame_timestamps = radar_timestamps - temp_timestamps
non_zero_indices = np.where(radar_timestamps != 0)[0]
zero_indices = np.where(radar_timestamps == 0)[0]
radar_timestamps[zero_indices] = np.interp(zero_indices, non_zero_indices, radar_timestamps[non_zero_indices])
radar_frame_timestamps = (radar_timestamps - radar_timestamps[0])/1e6




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
fig, (ax, ax2) = plt.subplots(2, figsize=(10, 14))
img = ax.imshow(np.zeros((N_RANGE_GATES, N_DOPPLER_BINS)), vmin=0, vmax=100, aspect='auto', cmap='jet', origin='lower')
ax.set_xlabel("Doppler Bins")
ax.set_ylabel("Range Gates")
ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
ax.set_title("Real-Time Range-Doppler Map")
img2 = ax2.imshow(np.zeros(image_size))
fig.colorbar(img, label="Magnitude (dB)")
timestamp_text = ax.text(xmin, ymin, "", color="white", fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

rdm_idx = 0

def process_radar_cube_data(range_doppler_matrix, angle=False):
    if angle:
        range_angle_matrix = np.fft.fftshift(np.fft.fft(range_doppler_matrix[:,N_DOPPLER_BINS//2,:,0], axis=1), axes=1)
        return np.abs(range_angle_matrix)
    
    # Sum over RX channels and chirps
    # range_doppler_matrix = np.fft.fft(range_doppler_matrix, axis=1)
    # range_doppler_matrix = np.sum(range_doppler_matrix[:,:,:,0], axis=2)
    # print(range_doppler_matrix.shape)
    range_doppler_matrix = range_doppler_matrix[:, :, 0, 0]


    # Apply FFT along the Doppler axis
    # range_doppler_matrix = np.fft.fftshift(np.fft.fft(range_doppler_matrix, axis=0), axes=0)
    # range_doppler_matrix = np.fft.fftshift(np.fft.fft(range_doppler_matrix, axis=1), axes=1)
    # range_doppler_matrix = np.mean(range_doppler_matrix, axis=2)
    # range_doppler_matrix = np.mean(range_doppler_matrix, axis=2)
    
    return np.abs(range_doppler_matrix)
    return 20 * np.log10(np.abs(range_doppler_matrix) + 1e-6)

def update(frame):
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
    # Define Bin Properties
    properties = all_properties[rdm_idx]
    DOPPLER_RESOLUTION  = properties[0]
    RANGE_RESOLUTION    = properties[1]
    BIN_PER_SPEED       = properties[2]
    xt_left = -N_DOPPLER_BINS//2*DOPPLER_RESOLUTION
    xt_right = (N_DOPPLER_BINS//2 - 1)*DOPPLER_RESOLUTION
    yt_bottom = 0
    yt_top = N_RANGE_GATES*RANGE_RESOLUTION
    fill_neg_bins = np.zeros((N_RANGE_GATES, int((xmin-xt_left)//DOPPLER_RESOLUTION)), dtype=np.float64)
    fill_pos_bins = np.zeros((N_RANGE_GATES, int((xt_right-xmax)//DOPPLER_RESOLUTION)), dtype=np.float64)
    range_doppler_matrix = process_radar_cube_data(range_doppler_matrix)
    range_doppler_matrix = np.concatenate((fill_pos_bins, range_doppler_matrix, fill_neg_bins), axis=1)
    img.set_data(range_doppler_matrix)
    img.set_extent([xmax, xmin, yt_bottom, yt_top])
    timestamp_text.set_text(f"Timestamp: {rdm_idx}")
    if rdm_idx < len(radar_cube_data)-1:
        rdm_idx += 1
    return img, timestamp_text

def animate():
    ani = animation.FuncAnimation(fig, update, frames=len(image_filenames), interval=30)
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