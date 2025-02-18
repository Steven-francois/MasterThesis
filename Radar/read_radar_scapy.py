from scapy.all import rdpcap, UDP
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load PCAP file
pcap_file = "radar_log_12.pcapng"  # Replace with your PCAP file path
packets = rdpcap(pcap_file)

# Define parameters
RADAR_CUBE_UDP_PORT = 50005  # Radar Cube Data port
BIN_PROPERTIES_UDP_PORT = 50063  # Bin Properties port
header_size = 22 + 64  # Header size from document

# Data fields greater than 1 Byte are transmitted in Big Endian.
dt_uint64 = np.dtype(np.uint64).newbyteorder('>')
dt_uint32 = np.dtype(np.uint32).newbyteorder('>')
dt_uint16 = np.dtype(np.uint16).newbyteorder('>')
dt_int16 = np.dtype(np.int16).newbyteorder('>')
dt_float32 = np.dtype(np.float32).newbyteorder('>')

old_ts = 0
p_old_ts = 0


def extract_header(data):
    global old_ts
    data_ = data[22:] # Skip the first 22 bytes
    header_size = 64
    frame = np.frombuffer(data[14:18], dtype=dt_uint32)
    ts = np.frombuffer(data[30:38], dtype=dt_uint64)
    # print(f"Frame: {frame}, Timestamp: {ts}, Delta: {(ts-old_ts)/10**6}")
    old_ts = ts
    header = data_[24:header_size] # Extract the header
    offsets = header[:24]
    imag_offset, real_offset, range_gate_offset, doppler_bin_offset, rx_channel_offset, chirp_type_offset = np.frombuffer(offsets, dtype=dt_uint32)
    indexes = header[24:30]
    indexes = np.frombuffer(indexes, dtype=dt_uint16)
    range_gates, first_range_gate, doppler_bins = indexes
    rx_channels, chirp_types, element_size, element_type = header[30:34]
    
    return {"imag_offset": imag_offset, "real_offset": real_offset, "range_gate_offset": range_gate_offset, "doppler_bin_offset": doppler_bin_offset, "rx_channel_offset": rx_channel_offset, "chirp_type_offset": chirp_type_offset, "range_gates": range_gates, "first_range_gate": first_range_gate, "doppler_bins": doppler_bins, "rx_channels": rx_channels, "chirp_types": chirp_types, "element_size": element_size, "element_type": element_type}

def read_element_by_offset(radar_cube_data, offset, real_offset, imag_offset):
    real = radar_cube_data[offset+real_offset:offset+real_offset+2]
    imag = radar_cube_data[offset+imag_offset:offset+imag_offset+2]
    real = np.frombuffer(real, dtype=dt_int16)
    imag = np.frombuffer(imag, dtype=dt_int16)
    return complex(real, imag)

def radar_cube(radar_cube_data, fields):
    radar_cube = np.zeros((fields["range_gates"], fields["doppler_bins"], fields["rx_channels"], fields["chirp_types"]), dtype=np.complex64)
    for rg in range(fields["range_gates"]):
        for db in range(fields["doppler_bins"]):
            for rx in range(fields["rx_channels"]):
                for seq in range(fields["chirp_types"]):
                    offset = (fields["range_gates"]-1-rg) * fields["range_gate_offset"] + ((fields["doppler_bins"]//2+db)%fields["doppler_bins"]) * fields["doppler_bin_offset"] + rx * fields["rx_channel_offset"] + seq * fields["chirp_type_offset"]
                    radar_cube[rg, db, rx, seq] = read_element_by_offset(radar_cube_data, offset, fields["real_offset"], fields["imag_offset"])
    return radar_cube

def extract_properties(data):
    global p_old_ts
    frame = np.frombuffer(data[14:18], dtype=dt_uint32)
    ts = np.frombuffer(data[30:38], dtype=dt_uint64)
    # print(f"BIN Frame: {frame}, Timestamp: {ts}, Delta: {(ts-p_old_ts)/10**6}")
    p_old_ts = ts
    properties = data[22+24:]
    properties = np.frombuffer(properties, dtype=dt_float32)
    return properties

# Extract radar cube packets from PCAP
rc_udp_packets = [pkt for pkt in packets if UDP in pkt and pkt[UDP].dport == RADAR_CUBE_UDP_PORT]
# Extract bin properties packets from PCAP
bp_udp_packets = [pkt for pkt in packets if UDP in pkt and pkt[UDP].dport == BIN_PROPERTIES_UDP_PORT]


if not rc_udp_packets or not bp_udp_packets:
    raise ValueError("No UDP packets found on port 50005|50063")
print(f"Radar Cube Packets: {len(rc_udp_packets)}, Bin Properties Packets: {len(bp_udp_packets)}")

while rc_udp_packets[0][UDP].payload.load[18] != 0x01:
    rc_udp_packets.pop(0)
first_payload = bytes(rc_udp_packets[0][UDP].payload)
fields = extract_header(first_payload)
print(fields)
all_properties = []
for i in range(len(bp_udp_packets)):
    property_payload = bytes(bp_udp_packets[i][UDP].payload)
    all_properties.append(extract_properties(property_payload))
properties = np.mean(all_properties, axis=0)
print(properties)

# Define Bin Properties
DOPPLER_RESOLUTION  = properties[0]
RANGE_RESOLUTION    = properties[1]
BIN_PER_SPEED       = properties[2]

# Define radar cube dimensions
N_RANGE_GATES       = int(fields["range_gates"])         # 200
N_DOPPLER_BINS      = int(fields["doppler_bins"])        # 128
N_RX_CHANNELS       = int(fields["rx_channels"])         # 8
N_CHIRP_TYPES       = int(fields["chirp_types"])         # 2
FIRST_RANGE_GATE    = int(fields["first_range_gate"])    # 0
print(f"Range Gates: {N_RANGE_GATES}, Doppler Bins: {N_DOPPLER_BINS}, RX Channels: {N_RX_CHANNELS}, Chirp Types: {N_CHIRP_TYPES}")


# Setup Matplotlib figure
fig, ax = plt.subplots(figsize=(10, 6))
img = ax.imshow(np.zeros((N_RANGE_GATES, N_DOPPLER_BINS)), aspect='auto', cmap='jet', vmin=0, vmax=100, extent=[-N_DOPPLER_BINS/2*DOPPLER_RESOLUTION, N_DOPPLER_BINS/2*DOPPLER_RESOLUTION, 0, N_RANGE_GATES*RANGE_RESOLUTION])
ax.set_xlabel("Doppler Bins")
ax.set_ylabel("Range Gates")
ax.set_title("Real-Time Range-Doppler Map")
fig.colorbar(img, label="Magnitude (dB)")





def process_radar_cube_data(radar_data):
    # Check if the data size is correct
    expected_size = N_RANGE_GATES * N_DOPPLER_BINS * N_RX_CHANNELS * N_CHIRP_TYPES * 2 * 2
    if len(radar_data) != expected_size: #Skip incomplete frames
        # print(f"Expected size: {expected_size}, Received size: {len(radar_data)}")
        print("!", end="")
        return None
    print(".", end="")
    # Convert to NumPy array and reshape
    radar_cube = np.frombuffer(radar_data, dtype=dt_int16)
    radar_cube = radar_cube.reshape((N_CHIRP_TYPES, N_RANGE_GATES, N_RX_CHANNELS, N_DOPPLER_BINS, 2))
    # Transpose to (range gate, doppler, rx, chirp)
    radar_cube = np.transpose(radar_cube, (1, 3, 2, 0, 4))
    radar_cube = np.flip(radar_cube, axis=0)

    # Extract real and imaginary parts
    real_part = radar_cube[..., 0]
    imag_part = radar_cube[..., 1]

    # Convert to complex values
    complex_data = real_part + 1j * imag_part

    range_doppler_matrix = np.fft.fftshift(complex_data, axes=1)

    # Sum over RX channels and chirps
    # range_doppler_matrix = np.fft.fft(range_doppler_matrix, axis=1)
    # range_doppler_matrix = np.mean(range_doppler_matrix[:,:,:,0], axis=2)
    range_doppler_matrix = range_doppler_matrix[:, :, 0, 0]

    # Apply FFT along the Doppler axis
    # range_doppler_matrix = np.fft.fftshift(np.fft.fft(range_doppler_matrix, axis=0), axes=0)
    # range_doppler_matrix = np.fft.fftshift(np.fft.fft(range_doppler_matrix, axis=1), axes=1)
    # range_doppler_matrix = np.mean(range_doppler_matrix, axis=2)
    # range_doppler_matrix = np.mean(range_doppler_matrix, axis=2)
    return np.abs(range_doppler_matrix)
    return 20 * np.log10(np.abs(range_doppler_matrix) + 1e-6)

def update_plot(frame):
    global rc_udp_packets
    if not rc_udp_packets:
        return img,
    while rc_udp_packets[0][UDP].payload.load[18] != 0x01:
        rc_udp_packets.pop(0)
    first_payload = bytes(rc_udp_packets[0][UDP].payload)
    prev_message_counter = np.frombuffer(first_payload[10:12], dtype=dt_uint16)
    ts=rc_udp_packets[0].time
    timestamp = np.frombuffer(first_payload[30:38], dtype=dt_uint64)
    # print(f"Time: {ts}, TS: {timestamp}")
    radar_data = bytearray(first_payload[header_size:])
    rc_udp_packets.pop(0)
    while rc_udp_packets and rc_udp_packets[0][UDP].payload.load[18] != 0x01:
        payload = bytes(rc_udp_packets[0][UDP].payload)  # Extract UDP payload
        current_message_counter = np.frombuffer(payload[10:12], dtype=dt_uint16)
        ts=rc_udp_packets[0].time
        timestamp = np.frombuffer(payload[30:38], dtype=dt_uint64)
        # print(f"Time: {ts}, TS: {timestamp}")
        if current_message_counter != prev_message_counter + 1:
            # print(f"Message counter skipped: {prev_message_counter} -> {current_message_counter}")
            print("x", end="")
            return img,
        prev_message_counter = current_message_counter
        # print(f"{payload[18]}.", end="")
        radar_data.extend(payload[22:])
        rc_udp_packets.pop(0)
    # print("/")

    range_doppler_map = process_radar_cube_data(radar_data)
    if range_doppler_map is not None:
        img.set_data(range_doppler_map)
    return img,

# # Plot Range-Doppler Map
# plt.figure(figsize=(10, 6))
# plt.imshow(20 * np.log10(np.abs(range_doppler_matrix) + 1e-6), aspect='auto', cmap='jet')
# plt.xlabel("Doppler Bins")
# plt.ylabel("Range Gates")
# plt.title("Range-Doppler Map from PCAP")
# plt.colorbar(label="Magnitude (dB)")
# plt.show()
def animate():
    # Animate Range-Doppler Map
    ani = animation.FuncAnimation(fig, update_plot, interval=50, cache_frame_data=False, repeat=True)
    plt.show()

def single_frame():
    # Update plot once
    update_plot(0)
    plt.show()

if __name__ == "__main__":
    animate()
    # single_frame()