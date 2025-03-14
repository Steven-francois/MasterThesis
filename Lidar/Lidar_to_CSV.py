import pyshark
import struct
from datetime import datetime, timedelta
import csv
import os
from tqdm import tqdm

def create_output_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def parse_lidar_payload(payload):
    """
    Parse the payload data of the LiDAR point cloud protocol according to the provided format.
    """
    try:
        # Parse each field
        version = struct.unpack('B', payload[0:1])[0]
        length = struct.unpack('>H', payload[1:3])[0]
        time_interval = struct.unpack('>H', payload[3:5])[0]
        dot_num = struct.unpack('>H', payload[5:7])[0]
        udp_cnt = struct.unpack('>H', payload[7:9])[0]
        frame_cnt = struct.unpack('B', payload[9:10])[0]
        data_type = struct.unpack('B', payload[10:11])[0]
        time_type = struct.unpack('B', payload[11:12])[0]
        pack_info = struct.unpack('B', payload[12:13])[0]
        resv = payload[13:24]  # Reserved field, length is 11 bytes
        crc32 = struct.unpack('>I', payload[24:28])[0]
        timestamp = struct.unpack('Q', payload[28:36])[0]  # >Q unsigned long long
        data = payload[36:]  # Remaining data

        return {
            "version": version,
            "length": length,
            "time_interval": time_interval,
            "dot_num": dot_num,
            "udp_cnt": udp_cnt,
            "frame_cnt": frame_cnt,
            "data_type": data_type,
            "time_type": time_type,
            "pack_info": pack_info,
            "resv": resv,
            "crc32": crc32,
            "timestamp": timestamp,
            "data": data
        }
    except Exception as e:
        print(f"Error parsing payload: {e}")
        return None

def parse_lidar_data(data):
    """
    Parse point cloud data, assuming each point contains x, y, z coordinates and intensity value,
    each coordinate and intensity value is of type int and 2 uint8.
    """
    point_size = 14  # Each point contains 4 floats (x, y, z, intensity)
    num_points = len(data) // point_size
    point_format = 'iiiBB'  # struct format: 3 ints, 2 uint8 = 14 bytes

    points = []

    for i in range(num_points):
        offset = i * point_size
        point = struct.unpack(point_format, data[offset:offset + point_size])
        points.append({
            'x': point[0] / 1000,
            'y': point[1] / 1000,
            'z': point[2] / 1000,
            'intensity': point[3],
            'tag_information': point[4]
        })
    return points

def Formatted_Realtime(value):
    value = value / 1e9
    epoch = datetime(1970, 1, 1)
    timestamp_datetime = epoch + timedelta(seconds=value)
    return timestamp_datetime

#output_folder = 'Lidar'
#create_output_folder(output_folder)
#output_file = os.path.join(output_folder, 'output.csv')

with open('Fusion/data/lidar_20.csv', 'w', newline='') as csvfile:
    fieldnames = ['UTC Timestamp', 'Local Timestamp', 'X', 'Y', 'Z', 'Intensity', 'Tag Information']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Read PCAP file
    cap = pyshark.FileCapture('Fusion/captures/lidar_20250307_123903.pcapng', display_filter='udp')

    # Iterate over packets and display information
    for packet in tqdm(cap):
        try:
            # Check if there is payload information
            if hasattr(packet.udp, 'payload'):
                payload = packet.udp.payload.binary_value
                parsed_payload = parse_lidar_payload(payload)
                if parsed_payload:
                    UTC_Timestamp = Formatted_Realtime(parsed_payload['timestamp'])
                    Local_Timestamp = UTC_Timestamp + timedelta(hours=1)
                    # print(Local_Timestamp)
                    # print("-"*50)
                    points = parse_lidar_data(parsed_payload['data'])
                    for point in points:
                        row = {
                            'UTC Timestamp': UTC_Timestamp,
                            'Local Timestamp': Local_Timestamp,
                            'X': point['x'],
                            'Y': point['y'],
                            'Z': point['z'],
                            'Intensity': point['intensity'],
                            'Tag Information': point['tag_information']
                        }
                        writer.writerow(row)
            else:
                # Placeholder handling when there is no payload (optional)
                row = {
                    'UTC Timestamp': 'No payload available',
                    'Local Timestamp': 'No payload available',
                    'X': 'No payload available',
                    'Y': 'No payload available',
                    'Z': 'No payload available',
                    'Intensity': 'No payload available',
                    'Tag Information': 'No payload available',
                }
                writer.writerow(row)
                # print("end")
        except AttributeError as e:
            continue

print("Data has been saved to output.csv.")
