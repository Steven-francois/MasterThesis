import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

rdc_file = "radar_cube_data_11.npy" # Replace with your output file path
rdc_file_bg = "radar_cube_data_bg11.npy" # Replace with your output file path

with open(rdc_file, 'rb') as f:
    fields = np.load(f, allow_pickle=True)
    properties = np.load(f, allow_pickle=True)
    timestamps = np.load(f, allow_pickle=True)
    nb_frames = np.load(f, allow_pickle=True)
    # print(f"Fields: {fields}, Properties: {properties}, Timestamps: {timestamps}, Number of Frames: {nb_frames}")
    radar_cube_data = np.array([np.load(f, allow_pickle=True)])
    for _ in range(nb_frames-1):
        radar_cube_data = np.append(radar_cube_data, [np.load(f, allow_pickle=True)], axis=0)
    # radar_cube_data = np.load(f, allow_pickle=True)


with open(rdc_file_bg, 'rb') as f:
    fields_bg = np.load(f, allow_pickle=True)
    properties_bg = np.load(f, allow_pickle=True)
    timestamps_bg = np.load(f, allow_pickle=True)
    nb_frames_bg = np.load(f, allow_pickle=True)
    radar_cube_data_bg = np.array([np.load(f, allow_pickle=True)])
    for _ in range(nb_frames_bg-1):
        radar_cube_data_bg = np.append(radar_cube_data_bg, [np.load(f, allow_pickle=True)], axis=0)
    print(radar_cube_data_bg.shape)



# Define radar cube dimensions
N_RANGE_GATES       = int(fields[6])    # 200
N_DOPPLER_BINS      = int(fields[8])    # 128
N_RX_CHANNELS       = int(fields[9])    # 8
N_CHIRP_TYPES       = int(fields[10])   # 2
FIRST_RANGE_GATE    = int(fields[7])    # 0
print(f"Range Gates: {N_RANGE_GATES}, Doppler Bins: {N_DOPPLER_BINS}, RX Channels: {N_RX_CHANNELS}, Chirp Types: {N_CHIRP_TYPES}")

# Define Bin Properties
DOPPLER_RESOLUTION  = properties[0]
RANGE_RESOLUTION    = properties[1]
BIN_PER_SPEED       = properties[2]
print(f"Doppler Resolution: {DOPPLER_RESOLUTION}, Range Resolution: {RANGE_RESOLUTION}, Bins per m/s: {BIN_PER_SPEED}")

mean_bg = np.mean(radar_cube_data_bg[:34], axis=0)
print(mean_bg.shape)
range_doppler_matrix = np.abs(mean_bg[:,:,0,0])
print(range_doppler_matrix.shape)
plt.figure(figsize=(10, 6))
plt.imshow(range_doppler_matrix, aspect='auto', cmap='jet', extent=[-N_DOPPLER_BINS/2*DOPPLER_RESOLUTION, N_DOPPLER_BINS/2*DOPPLER_RESOLUTION, 0, N_RANGE_GATES*RANGE_RESOLUTION])
plt.xlabel('Doppler Bins')
plt.ylabel('Range Gates')
plt.title(f'Range-Doppler Map at BG')
plt.colorbar(label='Magnitude (dB)')
plt.show()
radar_cube_data = radar_cube_data - mean_bg


# Setup Matplotlib figure
fig, ax = plt.subplots(figsize=(10, 6))
img = ax.imshow(np.zeros((N_RANGE_GATES, N_DOPPLER_BINS)), vmin=0, vmax=100, aspect='auto', cmap='jet', extent=[-N_DOPPLER_BINS/2*DOPPLER_RESOLUTION, N_DOPPLER_BINS/2*DOPPLER_RESOLUTION, 0, N_RANGE_GATES*RANGE_RESOLUTION])
ax.set_xlabel("Doppler Bins")
ax.set_ylabel("Range Gates")
ax.set_title("Real-Time Range-Doppler Map")
fig.colorbar(img, label="Magnitude (dB)")
timestamp_text = ax.text(5, 5, "", color="white", fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

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
    range_doppler_matrix = radar_cube_data[frame]
    # if range_doppler_matrix == 0:
    #     return img,
    range_doppler_matrix = process_radar_cube_data(range_doppler_matrix)
    img.set_data(range_doppler_matrix)
    timestamp_text.set_text(f"Timestamp: {frame}")
    return img, timestamp_text

def animate():
    ani = animation.FuncAnimation(fig, update, frames=len(radar_cube_data), interval=100)
    plt.show()

def save():
    ani = animation.FuncAnimation(fig, update, frames=len(radar_cube_data), interval=100)
    ani.save('radar_rdm_11.gif', writer='pillow')

def plot_angles_for_0_speed():
    # Plot a Range-Doppler Map for a specific frame
    frame_idx = 3 
    range_doppler_matrix = radar_cube_data[frame_idx]
    range_doppler_matrix = process_radar_cube_data(range_doppler_matrix)

    plt.figure(figsize=(10, 6))
    plt.imshow(range_doppler_matrix, aspect='auto', cmap='jet', extent=[-N_DOPPLER_BINS/2*DOPPLER_RESOLUTION, N_DOPPLER_BINS/2*DOPPLER_RESOLUTION, 0, N_RANGE_GATES*RANGE_RESOLUTION])
    plt.xlabel('Doppler Bins')
    plt.ylabel('Range Gates')
    plt.title(f'Range-Doppler Map at Frame {frame_idx}')
    plt.colorbar(label='Magnitude (dB)')

    # Plot Range-Angle for a specific doppler bin
    doppler_bin = N_DOPPLER_BINS // 2
    d = 0.006  # Antenna spacing (meters)
    fc = 77e9  # Radar carrier frequency (Hz)
    c = 3e8  # Speed of light (m/s)
    wavelength = c / fc  # Wavelength (meters)
    range_angle_matrix = process_radar_cube_data(radar_cube_data[frame_idx], angle=True)
    plt.figure(figsize=(10, 6))
    plt.imshow(range_angle_matrix, aspect='auto', cmap='jet', extent=[np.rad2deg(np.arcsin(-0.5*wavelength/d)), np.rad2deg(np.arcsin(0.5*wavelength/d)), 0, N_RANGE_GATES*RANGE_RESOLUTION])
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Range Gates')
    plt.title(f'Range-Angle Spectrum at Doppler Bin {doppler_bin}')
    plt.colorbar(label='Magnitude')

    plt.show()


if __name__ == "__main__":
    save()
    animate()
    # plot_angles_for_0_speed()