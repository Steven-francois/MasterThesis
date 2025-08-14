from Radar.RadarPacketPcapngReader import RadarPacketPcapngReader as Rpr
from Radar.RadarCanReader import RadarCanReader
import os


def read_radar_data(eth_file, rdc_file, can_file, can_output_file):
    rdc_reader = Rpr(eth_file, rdc_file)
    canReader = RadarCanReader()
    rdc_reader.read()
    canReader.read(can_file)
    canReader.save_npy(can_output_file)
    del rdc_reader, canReader

if __name__ == "__main__":
    name = "21"
    main_folder = "Fusion/static/DATA_20250514_140309/"
    print(f"Main folder: {main_folder}")
    for i, folder in enumerate(os.listdir(main_folder)):
        data_folder = os.path.join(main_folder, folder)
        if os.path.isdir(data_folder):
            print(f"Data folder: {data_folder}")
            files = os.listdir(data_folder)
            files = sorted([f for f in files if f.endswith('.pcapng')])
            nb_file = f"{name}_{i}"
            result_folder = f"Data/{nb_file}/"
            os.makedirs(result_folder, exist_ok=True)
            rdc_file = f"{result_folder}radar_cube_data"
            eth_file = os.path.join(data_folder, files[2])
            can_file = os.path.join(data_folder, files[1])
            can_output_file = f"{result_folder}radar_can_data.npy"
            
            # Read radar data
            read_radar_data(eth_file, rdc_file, can_file, can_output_file)
        exit()