import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Radar.RadarPacketPcapngReader import RadarPacketPcapngReader as RadarPacketReader
from Radar.RadarCanReader import RadarCanReader
from Radar.cfar import cfar

nb_file = "41"
rdc_file = f"Fusion/data/radar_cube_data_{nb_file}" # Replace with your output file path
can_file = f"Fusion/data/can_data_{nb_file}.npy" # Replace with your output file path

packetReader = RadarPacketReader("Radar/captures/radar_log_21.pcapng", rdc_file)
packetReader.load()
fields = packetReader.fields
timestamps = packetReader.timestamps
time = packetReader.time
nb_frames = packetReader.nb_frames
all_properties = packetReader.all_properties
radar_cube_data = packetReader.radar_cube_datas
print(f"Fields: {fields}, Number of Frames: {nb_frames}, Number RDC: {len(radar_cube_data)}, Number Properties: {len(all_properties)}")

canReader = RadarCanReader()
canReader.load_npy(can_file)
print(f"rdc ts start: {timestamps[0]} can ts start: {canReader.can_targets[0].targets_header.timestamp}")
print(f"rdc ts end: {timestamps[-1]} can ts end: {canReader.can_targets[-1].targets_header.timestamp}")
for i in range(25):
    print(f"rdc ts: {timestamps[i+1]} can ts: {canReader.can_targets[i+3].targets_header.timestamp}, diff: {timestamps[i+1]/1e6-canReader.can_targets[i+3].targets_header.timestamp}")
for i in range(25):
    print(f"rdc time: {time[i+1]} can time: {canReader.can_targets[i+3].targets_header.real_time}, diff: {time[i+1]-canReader.can_targets[i+3].targets_header.real_time}")
for i in range(20):
    print(timestamps[i+1]-timestamps[i], canReader.can_targets[i+1].targets_header.timestamp-canReader.can_targets[i].targets_header.timestamp)


# Define radar cube dimensions
N_RANGE_GATES       = int(fields[6])    # 200
N_DOPPLER_BINS      = int(fields[8])    # 128
N_RX_CHANNELS       = int(fields[9])    # 8
N_CHIRP_TYPES       = int(fields[10])   # 2
FIRST_RANGE_GATE    = int(fields[7])    # 0
print(f"Range Gates: {N_RANGE_GATES}, Doppler Bins: {N_DOPPLER_BINS}, RX Channels: {N_RX_CHANNELS}, Chirp Types: {N_CHIRP_TYPES}")

# Define plot limits
xmin = -70/3.6
xmax = 70/3.6
# xmin = -200/3.6
# xmax = 200/3.6
ymin = 0
ymax = 96

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
img = ax1.imshow(np.zeros((N_RANGE_GATES, N_DOPPLER_BINS)), vmin=0, vmax=100, aspect='auto', cmap='jet', origin='lower')
ax1.set_title("Range-Doppler Map")
ax1.set_xlabel("Doppler (m/s)")
ax1.set_ylabel("Range (m)")
ax1.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
img2 = ax2.imshow(np.zeros((N_RANGE_GATES, N_DOPPLER_BINS)), vmin=0, vmax=1, aspect='auto', cmap='gray', origin='lower')
ax2.set_title("Range-Doppler Map")
ax2.set_xlabel("Doppler (m/s)")
ax2.set_ylabel("Range (m)")
ax2.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
scat = ax3.scatter([], [], marker='o', color='b')
img31 = ax3.imshow(np.zeros((N_RANGE_GATES, N_DOPPLER_BINS)), vmin=0, vmax=1, aspect='auto', cmap='gray', origin='lower')
img32 = ax3.imshow(np.zeros((N_RANGE_GATES, N_DOPPLER_BINS)), vmin=0, vmax=1, aspect='auto', cmap='gray', origin='lower')
img33 = ax3.imshow(np.zeros((N_RANGE_GATES, N_DOPPLER_BINS)), vmin=0, vmax=1, aspect='auto', cmap='gray', origin='lower')
ax3.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
ax3.set_title("Detected Targets")
ax3.set_xlabel("Speed (m/s)")
ax3.set_ylabel("Range (m)")


def update(frame):
    if frame < len(radar_cube_data)-1:
        # Update radar cube data
        radar_cube = radar_cube_data[frame+1]
        # range_doppler_matrix = np.sum(radar_cube[:,:,:,0], axis=2)
        range_doppler_matrix = radar_cube[:,:,0,0]
        # range_doppler_matrix = 10 * np.log10(np.abs(range_doppler_matrix) + 1e-6)  # Convert to dB scale
        range_doppler_matrix = np.abs(range_doppler_matrix)  # Convert to dB scale
        img.set_data(range_doppler_matrix)
        
        properties = all_properties[frame+1]
        DOPPLER_RESOLUTION  = properties[0]
        RANGE_RESOLUTION    = properties[1]
        BIN_PER_SPEED       = properties[2]
        xt_left = -N_DOPPLER_BINS//2*DOPPLER_RESOLUTION
        xt_right = (N_DOPPLER_BINS//2 - 1)*DOPPLER_RESOLUTION
        yt_bottom = 0
        yt_top = N_RANGE_GATES*RANGE_RESOLUTION
        xt_diff = abs(xt_right - xt_left)
        img.set_extent([xt_left, xt_right, yt_bottom, yt_top])
        
        mask, _, _ = cfar(range_doppler_matrix, n_guard=(1,1), n_ref=(2,3), bias=3, method='CA')
        img2.set_data(mask)
        # img2 = ax2.imshow(mask, cmap='gray', origin='lower')
        img2.set_extent([xt_left, xt_right, yt_bottom, yt_top])
        img31.set_data(mask)
        img31.set_extent([xt_left-xt_diff, xt_right-xt_diff, yt_bottom, yt_top])
        img32.set_data(mask)
        img32.set_extent([xt_left, xt_right, yt_bottom, yt_top])
        img33.set_data(mask)
        img33.set_extent([xt_left+xt_diff, xt_right+xt_diff, yt_bottom, yt_top])
        
    # Update scatter plot with CAN data
    can_target = canReader.can_targets[frame+3]
    targets_data = can_target.targets_data
    x = []
    y = []
    values = []
    for target in targets_data:
        x.append(target.t_range)
        y.append(target.speed_radial)
        values.append(target.rcs)
    data = np.stack([y,x]).T
    scat.set_offsets(data)
    scat.set_array(np.array(values))

    plt.title(f"Frame {frame}")
    
    return img, img2, scat

ani = animation.FuncAnimation(fig, update, frames=range(len(radar_cube_data)-1), interval=100)
# ani = animation.FuncAnimation(fig, update, frames=range(2530, 4000), interval=100)
plt.show()
# ani.save('animation.mp4', writer='ffmpeg', fps=30)