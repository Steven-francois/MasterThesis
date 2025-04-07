import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Radar.RadarPacketPcapngReader import RadarPacketPcapngReader as RadarPacketReader
from Radar.RadarCanReader import RadarCanReader

nb_file = "50"
rdc_file = f"Radar/data/radar_cube_data_{nb_file}" # Replace with your output file path
can_file = f"Radar/data/can_data_{nb_file}.npy" # Replace with your output file path

packetReader = RadarPacketReader("Radar/captures/radar_log_21.pcapng", rdc_file)
packetReader.load()
fields = packetReader.fields
timestamps = packetReader.timestamps
nb_frames = packetReader.nb_frames
all_properties = packetReader.all_properties
radar_cube_data = packetReader.radar_cube_datas
print(f"Fields: {fields}, Number of Frames: {nb_frames}, Number RDC: {len(radar_cube_data)}, Number Properties: {len(all_properties)}")

canReader = RadarCanReader()
canReader.load_npy(can_file)
print(f"rdc ts start: {timestamps[0]} can ts start: {canReader.can_targets[0].targets_header.timestamp}")
print(f"rdc ts end: {timestamps[-1]} can ts end: {canReader.can_targets[-1].targets_header.timestamp}")
for i in range(25):
    print(f"rdc ts: {timestamps[i+1]} can ts: {canReader.can_targets[i+19].targets_header.timestamp}")
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
xmin = -200/3.6
xmax = 200/3.6
ymin = 0
ymax = 98

fig, (ax1, ax2) = plt.subplots(1, 2)
img = ax1.imshow(np.zeros((N_RANGE_GATES, N_DOPPLER_BINS)), vmin=0, vmax=100, aspect='auto', cmap='jet', origin='lower')
scat = ax2.scatter([], [], marker='o', color='b')
plt.xlim(-200,200)
plt.ylim(0,96)
plt.xlabel("Range (m)")
plt.ylabel("Speed (m/s)")


def update(frame):
    if frame < len(radar_cube_data)-1:
        # Update radar cube data
        radar_cube = radar_cube_data[frame+1]
        range_doppler_matrix = np.sum(radar_cube[:,:,:,0], axis=2)
        range_doppler_matrix = 10 * np.log10(np.abs(range_doppler_matrix) + 1e-6)  # Convert to dB scale
        img.set_array(range_doppler_matrix)
        
    # Update scatter plot with CAN data
    can_target = canReader.can_targets[frame+19]
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
    
    return img, scat

# ani = animation.FuncAnimation(fig, update, frames=range(len(radar_cube_data)-1), interval=100)
ani = animation.FuncAnimation(fig, update, frames=range(len(canReader.can_targets)), interval=100)
plt.show()
# ani.save('animation.mp4', writer='ffmpeg', fps=30)