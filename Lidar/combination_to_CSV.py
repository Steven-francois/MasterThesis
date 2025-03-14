import os
import pandas as pd
from datetime import datetime, timedelta
import base64
from mpl_toolkits.mplot3d import Axes3D
import csv
from tqdm import tqdm

import matplotlib.pyplot as plt

def extract_timestamp_from_filename(filename):
    # Extract timestamp from filename
    timestamp_str = filename.replace('.png', '')
    return datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S-%f')

def encode_image_to_base64(filepath):
    # Encode image to base64
    with open(filepath, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def interpolate_velocity(velocity_df, target_timestamp):
    # Interpolate velocity based on target timestamp
    before = velocity_df[velocity_df['Time'] <= target_timestamp]
    after = velocity_df[velocity_df['Time'] >= target_timestamp]

    if before.empty or after.empty:
        return None  # No valid velocity data for interpolation

    closest_before = before.iloc[-1]
    closest_after = after.iloc[0]

    if closest_before['Time'] == closest_after['Time']:
        return closest_before['Speed (km/h)']
    else:
        time_diff = (closest_after['Time'] - closest_before['Time']).total_seconds()
        ratio = (target_timestamp - closest_before['Time']).total_seconds() / time_diff
        vx = closest_before['Speed (km/h)'] + ratio * (closest_after['Speed (km/h)'] - closest_before['Speed (km/h)'])
        return vx
    

# Read LiDAR data
csv_file = 'Fusion/data/lidar_20.csv'
df = pd.read_csv(csv_file, header=None, dtype=str, low_memory=False)
df.columns = ['UTC Timestamp', 'Local Timestamp', 'X', 'Y', 'Z', 'Intensity', 'Tag Information']
#df = df.dropna(subset=['timestamp'])
df['Local Timestamp'] = pd.to_datetime(df['Local Timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
df = df.dropna(subset=['Local Timestamp'])

# Read speed data
speed_file = 'Fusion/captures/speed_test.csv'
speed_df = pd.read_csv(speed_file)
speed_df['Time'] = pd.to_datetime(speed_df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
speed_df = speed_df.dropna(subset=['Time'])

# Read PNG files
png_folder = 'Fusion/captures/camera_rgba'
results = {}
file_list = sorted(os.listdir(png_folder))

# Match Lidar timestamp with PNG timestamp
with open('Fusion/data/combi_20.csv', 'w', newline='') as csvfile:
    fieldnames = ['ID','Local Timestamp', 'X', 'Y', 'Z', 'Intensity', 'Tag Information','Speed']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    previous_filename = None
    for filename in tqdm(file_list):
        if filename.endswith('.png'):
            png_filepath = os.path.join(png_folder, filename)
            png_timestamp = extract_timestamp_from_filename(filename)
            if previous_filename is None:
                previous_timestamp = png_timestamp - timedelta(milliseconds=33.3333)
            filtered_df = df[(df['Local Timestamp'] > previous_timestamp) &  
                            (df['Local Timestamp'] <= png_timestamp)]  
            
            updated_records = [] 
            # Start to combine LiDAR data with speed data
            for _, row in filtered_df.iterrows():
                target_timestamp = row['Local Timestamp']
                # vx = interpolate_velocity(speed_df, target_timestamp)
                # if vx is None:
                #     print("It's none")
                #     continue
                time_diff = (target_timestamp - png_timestamp).total_seconds()
                # new_x = str(float(row['X']) + float(vx * time_diff))  
                updated_record = {
                'ID': filename,
                'Local Timestamp': target_timestamp,
                # 'X': new_x,
                'X': row['X'],
                'Y': row['Y'],
                'Z': row['Z'],
                'Intensity': row['Intensity'],
                'Tag Information': row['Tag Information'],
                # 'Speed': vx 
                'Speed': 0
                }
                writer.writerow(updated_record)

            previous_filename = filename
            previous_timestamp = png_timestamp

print(f'The results are saved in output_test.csv')
