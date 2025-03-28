import os
from Radar.RadarPacketPcapngReader import RadarPacketPcapngReader as RadarPacketReader
from tqdm import tqdm
import numpy as np
import shutil

nb_file = "31"
rdc_file = f"Fusion/data/radar_cube_data_{nb_file}" # Replace with your output file path
image_folder = f"Fusion/data/camera_{nb_file}/"
lidar_file = 'Fusion/data/combination_20250312_123525_2'
lidar_data_file = f"{lidar_file}_data.npy"
lidar_ts_file = f"{lidar_file}_ts.npy"
lidar_timestamps = np.load(lidar_ts_file, allow_pickle=True)
with open(lidar_data_file, 'rb') as f:
    lidar_frames = [np.load(f, allow_pickle=True) for _ in range(len(lidar_timestamps))]
    
rdc_reader = RadarPacketReader("Radar/captures/radar_log_30.pcapng", rdc_file)
rdc_reader.load()
fields = rdc_reader.fields
timestamps = rdc_reader.timestamps
nb_frames = rdc_reader.nb_frames
all_properties = rdc_reader.all_properties
radar_cube_data = rdc_reader.radar_cube_datas

for idx, lidar_frame in enumerate(lidar_frames):
    if len(lidar_frame) > 0 :
        first_valid_frame = idx
        break
for i in range(len(lidar_frames)-1, -1, -1):
    if len(lidar_frames[i]) > 0:
        last_valid_frame = i
        break

lidar_timestamps = lidar_timestamps[first_valid_frame:last_valid_frame+1]
lidar_frames = lidar_frames[first_valid_frame:last_valid_frame+1]
np.save(lidar_ts_file, lidar_timestamps)
with open(lidar_data_file, 'wb') as f:
    for frame in tqdm(lidar_frames, desc="Saving LiDAR data"):
        np.save(f, frame)
rdc_reader.timestamps = rdc_reader.timestamps[first_valid_frame:last_valid_frame+1]
rdc_reader.time = rdc_reader.time[first_valid_frame:last_valid_frame+1]
rdc_reader.radar_cube_datas = rdc_reader.radar_cube_datas[first_valid_frame:last_valid_frame+1]
rdc_reader.nb_frames = len(rdc_reader.timestamps)
rdc_reader.all_properties = rdc_reader.all_properties[first_valid_frame:last_valid_frame+1]
rdc_reader.update()
print(f"Number of radar frames: {rdc_reader.nb_frames}")

image_filenames = sorted(os.listdir(image_folder))
# Remove images that are not in the valid range
for filename in tqdm(image_filenames[:first_valid_frame],desc="Removing images"):
    os.remove(image_folder+filename)
for filename in tqdm(image_filenames[last_valid_frame+1:], desc="Removing images"):
    os.remove(image_folder+filename)


