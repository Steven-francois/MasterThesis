from scapy.all import rdpcap, UDP, IP, PcapReader
import struct
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import numpy as np

def create_output_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def parse_lidar_payload(payload):
    """
    Parse the payload data of the LiDAR point cloud protocol according to the provided format.
    """
    try:
        timestamp = struct.unpack('Q', payload[28:36])[0]  # >Q unsigned long long
        data = payload[36:]  # Remaining data

        return timestamp, data
    except Exception as e:
        print(f"Error parsing payload: {e}")
        return None, None

def parse_lidar_data(data):
    """
    Parse point cloud data, assuming each point contains x, y, z coordinates and intensity value,
    each coordinate and intensity value is of type int and 2 uint8.
    """
    point_size = 14  # Each point contains 4 floats (x, y, z, intensity)
    num_points = len(data) // point_size
    point_format = 'iiiBB'  # struct format: 3 ints, 2 uint8 = 14 bytes

    points = [
        struct.unpack(point_format, data[i*point_size:(i+1)*point_size])
        for i in range(num_points)
    ]

    return [
        {
            'x': p[0] / 1000,
            'y': p[1] / 1000,
            'z': p[2] / 1000,
            'intensity': p[3],
            'tag_information': p[4]
        }
        for p in points
    ]

def Formatted_Realtime(value):
    return datetime(1970, 1, 1) + timedelta(seconds=value/1e9)

#output_folder = 'Lidar'
#create_output_folder(output_folder)
#output_file = os.path.join(output_folder, 'output.csv')
def process_lidar_packets(file_path, output_file):
    """
    Process LiDAR packets from a PCAP file and save the data to a CSV file.
    """
    output_timestamps = f"{output_file}_ts.npy"
    output_data = f"{output_file}_data.npy"
    timestamps = []
    # packets = rdpcap(file_path)
    packets = PcapReader(file_path)
        
    with open(output_data, 'wb') as f:
        # Iterate over packets and display information
        for packet in tqdm(packets, desc="Processing LiDAR packets"):
            try:
                # Check if there is payload information
                if UDP in packet and hasattr(packet, 'load'):
                    timestamp, data = parse_lidar_payload(packet.load)
                    if timestamp:
                        UTC_Timestamp = Formatted_Realtime(timestamp)
                        Local_Timestamp = UTC_Timestamp + timedelta(hours=1)
                        timestamps.append(Local_Timestamp)
                        points = parse_lidar_data(data)
                        data =np.array([
                            [point['x'], point['y'], point['z'], point['intensity'], point['tag_information']]
                            for point in points
                        ])
                        np.save(f, data)
            except AttributeError as e:
                continue
    np.save(output_timestamps, timestamps)
    print(f"Data has been saved to {output_file}.")

if __name__ == "__main__":
    # Process LiDAR packets from a PCAP file
    process_lidar_packets("Fusion/captures30/lidar_20250312_123525.pcapng", "Fusion/data/lidar_20250312_123525_2")