from Radar.RadarPacketPcapngReader import RadarPacketPcapngReader as Rpr
from Radar.RadarCanReader import RadarCanReader
from Fusion.filter_camera_radar import filter
from Lidar.Lidar_to_npy import process_lidar_packets
from Lidar.combination_to_ply import combine_lidar_speed
import os
import numpy as np


def read_radar_data(eth_file, rdc_file, can_file, can_output_file):
    rdc_reader = Rpr(eth_file, rdc_file)
    canReader = RadarCanReader()
    rdc_reader.read()
    canReader.read(can_file)
    canReader.save_npy(can_output_file)
    del rdc_reader, canReader

if __name__ == "__main__":
    name = "1"
    main_folder = "Fusion/static/DATA_20250514_140309/"
    print(f"Main folder: {main_folder}")
    for i, folder in enumerate(os.listdir(main_folder)):
        data_folder = os.path.join(main_folder, folder)
        if os.path.isdir(data_folder):
            print(f"Data folder: {data_folder}")
            files = os.listdir(data_folder)
            files = sorted([f for f in files if f.endswith('.pcapng')])
            image_folder = os.path.join(data_folder, "camera_rgba/")
            nb_file = f"{name}_{i}"
            result_folder = f"Data/{nb_file}/"
            new_image_folder = f"{result_folder}camera/"
            os.makedirs(result_folder, exist_ok=True)
            os.makedirs(new_image_folder, exist_ok=True)
            rdc_file = f"{result_folder}radar_cube_data"
            eth_file = os.path.join(data_folder, files[2])
            can_file = os.path.join(data_folder, files[1])
            can_output_file = f"{result_folder}radar_can_data.npy"
            lidar_file = os.path.join(data_folder, files[0])
            lidar_npy_file = f"{result_folder}lidar"
            combined_lidar_npy_file = f"{result_folder}lidar_combined"
            # speed_file = f"{data_folder}speed_test.csv"
            speed_file = None
            
            # Read radar data
            read_radar_data(eth_file, rdc_file, can_file, can_output_file)
            
            # Load radar data
            rdc_reader = Rpr(eth_file, rdc_file)
            canReader = RadarCanReader()
            rdc_reader.load()
            canReader.load_npy(can_output_file)
            
            # Filter images based on radar timestamps
            filter(rdc_reader, image_folder, new_image_folder)
            can_ts = np.array([canReader.can_targets[i].targets_header.real_time for i in range(len(canReader.can_targets))])
            can_start = np.where(can_ts > rdc_reader.timestamps[0])[0][0]
            can_end = np.where(can_ts > rdc_reader.timestamps[-1])[0][0]
            can_end = min(can_end, len(canReader.can_targets) - 1 + can_start)
            canReader.can_targets = canReader.can_targets[can_start:can_end]
            print(f"Filtered CAN data from {can_start} to {can_end}")
            # Save filtered CAN data
            canReader.save_npy(can_output_file)
            
            
            # Extract LiDAR data
            process_lidar_packets(lidar_file, lidar_npy_file)
            
            # Combine LiDAR and speed data based on image timestamps
            combine_lidar_speed(lidar_npy_file, new_image_folder, combined_lidar_npy_file, speed_file)