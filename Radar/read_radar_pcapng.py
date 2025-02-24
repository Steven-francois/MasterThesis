from scapy.all import rdpcap, UDP, IP
import numpy as np
import pyshark

# Load PCAP file
nb_file = "21"
pcap_file = f"captures/radar_log_{nb_file}.pcapng"  # Replace with your PCAP file path
# pcap_file = f"../Fusion/captures/radar_20250214_120701.pcapng"  # Replace with your PCAP file path
packets = rdpcap(pcap_file)
# packets = pyshark.FileCapture(pcap_file)
rdc_file = f"../data/radar_cube_data_{nb_file}.npy" # Replace with your output file path
# rdc_file = f"../Fusion/data/radar_cube_data_{nb_file}.npy" # Replace with your output file path

# Define parameters
IP_SOURCE = "192.168.11.11"
IP_DEST = "192.168.11.17"
RADAR_CUBE_UDP_PORT = 50005  # Radar Cube Data port
BIN_PROPERTIES_UDP_PORT = 50063  # Bin Properties port
header_size = 22 + 64  # Header size from document

# Data fields greater than 1 Byte are transmitted in Big Endian.
dt_uint64 = np.dtype(np.uint64).newbyteorder('>')
dt_uint32 = np.dtype(np.uint32).newbyteorder('>')
dt_uint16 = np.dtype(np.uint16).newbyteorder('>')
dt_int16 = np.dtype(np.int16).newbyteorder('>')
dt_float32 = np.dtype(np.float32).newbyteorder('>')

def extract_header(data):
    data_ = data[22:] # Skip the first 22 bytes
    header_size = 64
    header = data_[24:header_size] # Extract the header
    offsets = header[:24]
    imag_offset, real_offset, range_gate_offset, doppler_bin_offset, rx_channel_offset, chirp_type_offset = np.frombuffer(offsets, dtype=dt_uint32)
    indexes = header[24:30]
    indexes = np.frombuffer(indexes, dtype=dt_uint16)
    range_gates, first_range_gate, doppler_bins = indexes
    rx_channels, chirp_types, element_size, element_type = header[30:34]
    
    # Return np array to be saved
    return np.array([imag_offset, real_offset, range_gate_offset, doppler_bin_offset, rx_channel_offset, chirp_type_offset, range_gates, first_range_gate, doppler_bins, rx_channels, chirp_types, element_size, element_type])
    
def extract_properties(data):
    frame_counter  = np.frombuffer(data[14:18], dtype=dt_uint32)[0]
    properties = data[22+24:]
    properties = np.frombuffer(properties, dtype=dt_float32)

    # Return np array to be saved
    return (frame_counter, properties)


# Extract radar cube packets from PCAP
rc_udp_packets = [pkt for pkt in packets if UDP in pkt and pkt[UDP].dport == RADAR_CUBE_UDP_PORT and pkt[IP].src == IP_SOURCE and pkt[IP].dst == IP_DEST]
# Extract bin properties packets from PCAP
bp_udp_packets = [pkt for pkt in packets if UDP in pkt and pkt[UDP].dport == BIN_PROPERTIES_UDP_PORT and pkt[IP].src == IP_SOURCE and pkt[IP].dst == IP_DEST]
print(len(rc_udp_packets), len(bp_udp_packets))
if not rc_udp_packets or not bp_udp_packets:
    raise ValueError("No UDP packets found on port 50005|50063")
print(f"Radar Cube Packets: {len(rc_udp_packets)}, Bin Properties Packets: {len(bp_udp_packets)}")

rc_udp_packets.sort(key=lambda x: (np.frombuffer(x[UDP].payload.load[14:18], dtype=dt_uint32), np.frombuffer(x[UDP].payload.load[10:12], dtype=dt_uint16)))
bp_udp_packets.sort(key=lambda x: (np.frombuffer(x[UDP].payload.load[14:18], dtype=dt_uint32), np.frombuffer(x[UDP].payload.load[10:12], dtype=dt_uint16)))
while rc_udp_packets[0][UDP].payload.load[18] != 0x01:
    rc_udp_packets.pop(0)
first_payload = bytes(rc_udp_packets[0][UDP].payload)
fields = extract_header(first_payload)
print(fields)

all_properties = []
all_frames_counter = []
for i in range(len(bp_udp_packets)):
    property_payload = bytes(bp_udp_packets[i][UDP].payload)
    frame_counter, properties = extract_properties(property_payload)
    all_frames_counter.append(frame_counter)
    all_properties.append(properties)

# Fill missing properties
for i in range(len(all_properties)-1):
    current_frame, current_properties = all_frames_counter[i], all_properties[i]
    next_frame, next_properties = all_frames_counter[i+1], all_properties[i+1]
    # Check for missing frames
    if next_frame - current_frame > 1:
        nb_missing_frames = next_frame - current_frame - 1
        delta_properties = (next_properties - current_properties) / nb_missing_frames
        for j in range(current_frame+1, next_frame):
            all_properties.insert(i+1, current_properties + delta_properties * (j - current_frame))
            print(f"Missing frame: {j}")
first_property_frame = all_frames_counter[0]
first_properties = all_properties[0]
print(f"First property frame :{first_property_frame}")


# Define radar cube dimensions
N_RANGE_GATES       = int(fields[6])    # 200
N_DOPPLER_BINS      = int(fields[8])    # 128
N_RX_CHANNELS       = int(fields[9])    # 8
N_CHIRP_TYPES       = int(fields[10])   # 2
FIRST_RANGE_GATE    = int(fields[7])    # 0
print(f"Range Gates: {N_RANGE_GATES}, Doppler Bins: {N_DOPPLER_BINS}, RX Channels: {N_RX_CHANNELS}, Chirp Types: {N_CHIRP_TYPES}")


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

    # Extract real and imaginary parts
    real_part = radar_cube[..., 0]
    imag_part = radar_cube[..., 1]

    # Convert to complex values
    complex_data = real_part + 1j * imag_part

    range_doppler_matrix = np.fft.fftshift(complex_data, axes=1)

    # Return np array to be saved
    return range_doppler_matrix


# Process radar cube data
radar_cube_datas = []
timestamps = []
first_radar_cube_frame = np.frombuffer(rc_udp_packets[0][UDP].payload.load[14:18], dtype=dt_uint32)[0]
frame_diff = first_radar_cube_frame - first_property_frame
print(f"First Radar Cube Frame: {first_radar_cube_frame}, First Property Frame: {first_property_frame}, Frame Diff: {frame_diff}")
if first_property_frame < first_radar_cube_frame:
    all_properties = all_properties[frame_diff:]
else:
    for i in range(frame_diff):
        all_properties.insert(0, first_properties)
        print(f"Missing property frame: {first_radar_cube_frame-i}")
i = 0
nb_packets = len(rc_udp_packets)
while i<nb_packets:
    radar_cube_data = bytearray(rc_udp_packets[i][UDP].payload.load[22+64:])
    current_frame = rc_udp_packets[i][UDP].payload.load[14:18]
    timestamp = np.frombuffer(rc_udp_packets[i][UDP].payload.load[30:38], dtype=dt_uint64)[0]
    i+=1
    while i<nb_packets and rc_udp_packets[i][UDP].payload.load[18] != 0x01 and rc_udp_packets[i][UDP].payload.load[14:18] == current_frame:
        radar_cube_data.extend(rc_udp_packets[i][UDP].payload.load[22:])
        i+=1
    radar_cube_data = process_radar_cube_data(radar_cube_data)
    if radar_cube_data is not None:
        timestamps.append(timestamp)
        radar_cube_datas.append(radar_cube_data)
    else:
        timestamps.append(0)
        radar_cube_datas.append(np.zeros((N_RANGE_GATES, N_DOPPLER_BINS, N_RX_CHANNELS, N_CHIRP_TYPES), dtype=np.complex64))
print(f"\nProcessed Radar Cube Data: {len(radar_cube_datas)} frames")

if (len(radar_cube_datas) != len(all_properties)):
    print(f"Radar Cube Data: {len(radar_cube_datas)}, Properties: {len(all_properties)}")
    for i in range(len(radar_cube_datas)-len(all_properties)):
        all_properties.insert(-1, all_properties[-1])
    print(f"Filled Properties: {len(all_properties)}")

# Save all datas
with open(rdc_file, "wb") as f:
    np.save(f, fields)
    np.save(f, timestamps)
    np.save(f, len(radar_cube_datas))
    for i in range(len(all_properties)):
        np.save(f, all_properties[i])
    for i in range(len(radar_cube_datas)):
        np.save(f, radar_cube_datas[i])
