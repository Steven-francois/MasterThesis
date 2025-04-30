from Radar.RadarPacketPcapngReader import RadarPacketPcapngReader as Rpr
from Radar.RadarCanReader import RadarCanReader
from Fusion.filter_camera_radar import filter
from Lidar.Lidar_to_npy import process_lidar_packets
from Lidar.combination_to_ply import combine_lidar_speed
import os


if __name__ == "__main__":
    data_folder = "Fusion/20250429_072437/"
    image_folder = f"{data_folder}camera_rgba/"
    nb_file = "2"
    result_folder = f"Fusion/data/{nb_file}/"
    new_image_folder = f"{result_folder}camera/"
    # os.makedirs(result_folder, exist_ok=True)
    # os.makedirs(new_image_folder, exist_ok=True)
    rdc_file = f"{result_folder}radar_cube_data"
    rdc_reader = Rpr(f"{data_folder}radar_eth_20250429_072437.pcapng", rdc_file)
    can_file = f"{data_folder}radar_can_20250429_072437.pcapng"
    can_output_file = f"{result_folder}radar_can_data.npy"
    canReader = RadarCanReader()
    lidar_file = f"{data_folder}lidar_20250429_072437.pcapng"
    lidar_npy_file = f"{result_folder}lidar"
    combined_lidar_npy_file = f"{result_folder}lidar_combined"
    speed_file = f"{data_folder}speed_test.csv"
    
    # Read radar data
    # rdc_reader.read()
    # canReader.read(can_file)
    # canReader.save_npy(can_output_file)
    
    # Load radar data
    # rdc_reader.load()
    # canReader.load_npy(f"{result_folder}radar_can_data.npy")
    
    # Filter images based on radar timestamps
    # filter(rdc_reader, image_folder, new_image_folder)
    
    # Extract LiDAR data
    # process_lidar_packets(lidar_file, lidar_npy_file)
    
    # Combine LiDAR and speed data based on image timestamps
    combine_lidar_speed(lidar_npy_file, new_image_folder, combined_lidar_npy_file, speed_file)