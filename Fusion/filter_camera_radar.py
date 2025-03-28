import os
from datetime import datetime
from Radar.RadarPacketPcapngReader import RadarPacketPcapngReader as RadarPacketReader
from tqdm import tqdm
import numpy as np
import shutil

def extract_timestamp_from_filename(filename):
    # Extract timestamp from filename
    timestamp_str = filename.replace('.png', '')
    return np.datetime64(datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S-%f')).astype(float)/1.0e6

def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def closest_idx(lst, K):
    return min(range(len(lst)), key = lambda i: abs(lst[i]-K))


nb_file = "31"
rdc_file = f"Fusion/data/radar_cube_data_{nb_file}" # Replace with your output file path
# image_folder = f"Fusion/captures{nb_file}/camera_rgba/"
image_folder = f"Fusion/captures30/camera_rgba/"
new_image_folder = f"Fusion/data/camera_{nb_file}/"
os.makedirs(new_image_folder, exist_ok=True)

image_filenames = os.listdir(image_folder)
image_filenames = np.array(sorted([f for f in image_filenames if f.endswith('.png')]))
image_timestamps = np.array([datetime.strptime(filename.split(".")[0], "%Y-%m-%d_%H-%M-%S-%f").timestamp() for filename in image_filenames])

rdc_reader = RadarPacketReader("Radar/captures/radar_log_21.pcapng", rdc_file)
rdc_reader.load()
rdc_reader.interpolate_timestamps()
radar_timestamps = rdc_reader.timestamps/1e6
radar_times = rdc_reader.time
radar_timestamps = radar_timestamps - radar_timestamps[0]
radar_timestamps += radar_times[0] - 55/1.0e3
valid_images_idx = np.where((radar_timestamps[0]-(1/60) <= image_timestamps) & (image_timestamps <= radar_timestamps[-1]+(1/60)))[0]
image_filenames = image_filenames[valid_images_idx]
image_timestamps = image_timestamps[valid_images_idx]
valid_radar_idx = np.where((image_timestamps[0] <= radar_timestamps) & (radar_timestamps <= image_timestamps[-1]))[0]
rdc_reader.timestamps = radar_timestamps[valid_radar_idx]
rdc_reader.time = radar_times[valid_radar_idx]
rdc_reader.radar_cube_datas = rdc_reader.radar_cube_datas[valid_radar_idx]
rdc_reader.nb_frames = len(rdc_reader.timestamps)
rdc_reader.all_properties = rdc_reader.all_properties[valid_radar_idx]

rdc_reader.update()
print(f"Number of radar frames: {rdc_reader.nb_frames}")


images_filtered_idx = [closest_idx(image_timestamps, radar_timestamp) for radar_timestamp in tqdm(rdc_reader.timestamps, desc="Filtering images")]
images_filtered = image_filenames[images_filtered_idx]


# Copy images to new folder
for image in tqdm(images_filtered, desc="Copying images"):
    shutil.copyfile(image_folder+image, new_image_folder+image)


