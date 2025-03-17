import os
import pandas as pd
from datetime import datetime
import csv
from tqdm import tqdm
import numpy as np
import multiprocessing as mp

import matplotlib.pyplot as plt

def extract_timestamp_from_filename(filename):
    # Extract timestamp from filename
    timestamp_str = filename.replace('.png', '')
    return np.datetime64(datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S-%f'))

def interpolate_velocity(velocity_timestamps, velocity_speeds, target_timestamp):
    # Interpolate velocity based on target timestamp
    idx = np.searchsorted(velocity_timestamps, target_timestamp, side="right") - 1
    if idx < 0 or idx >= len(velocity_timestamps) - 1:
        return None # No valid velocity data for interpolation

    before_timestamp, after_timestamp = velocity_timestamps[idx], velocity_timestamps[idx + 1]
    before_speed, after_speed = velocity_speeds[idx], velocity_speeds[idx + 1]

    if before_timestamp == after_timestamp:
        return before_speed
    else:
        ratio = (target_timestamp - before_timestamp).total_seconds() / (after_timestamp - before_timestamp).total_seconds()
        return before_speed + ratio * (after_speed - before_speed)
    
def process_png_file(args):
    filename, lidar_timestamps, df, velocity_timestamps, velocity_speeds, previous_timestamp, png_timestamp, output_queue = args

    start_idx = np.searchsorted(lidar_timestamps, previous_timestamp, side="right")
    end_idx = np.searchsorted(lidar_timestamps, png_timestamp, side="right")
    filtered_df = df.iloc[start_idx:end_idx]

    updated_records = [] 
    # Start to combine LiDAR data with speed data
    for _, row in filtered_df.iterrows():
        target_timestamp = row['Local Timestamp']
        # vx = interpolate_velocity(velocity_timestamps, velocity_speeds, target_timestamp)
        vx = 0
        if vx is None:
            print("It's none")
            continue
        # time_diff = (target_timestamp - png_timestamp).total_seconds()
        # new_x = float(row['X']) + float(vx * time_diff)
        new_x = row['X']
        updated_records.append([filename, target_timestamp, new_x, row['Y'], row['Z'], row['Intensity'], row['Tag Information'], vx])
        
    output_queue.put(updated_records)
    return len(updated_records)

def csv_writer(output_file, queue, total_files):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'Local Timestamp', 'X', 'Y', 'Z', 'Intensity', 'Tag Information', 'Speed'])
        with tqdm(total=total_files, desc="Writing to csv") as pbar:
            while True:
                records = queue.get()
                if records is None:
                    break
                writer.writerows(records)
                pbar.update(1)

def main():
    # Read LiDAR data
    csv_file = 'Lidar/data/lidar_20250228_120139.csv'
    df = pd.read_csv(csv_file, dtype={'X': float, 'Y': float, 'Z': float, 'Intensity': int, 'Tag Information': int}, low_memory=True)
    df.columns = ['UTC Timestamp', 'Local Timestamp', 'X', 'Y', 'Z', 'Intensity', 'Tag Information']
    df['Local Timestamp'] = pd.to_datetime(df['Local Timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    df.dropna(subset=['Local Timestamp'], inplace=True)
    lidar_timestamps = df['Local Timestamp'].to_numpy()

    # Read speed data
    # speed_file = 'Fusion/captures/speed_test.csv'
    # speed_df = pd.read_csv(speed_file)
    # speed_df['Time'] = pd.to_datetime(speed_df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    # speed_df.dropna(subset=['Time'], inplace=True)
    # velocity_timestamps = speed_df['Time'].to_numpy()
    # velocity_speeds = speed_df['Speed (km/h)'].to_numpy()

    # Read PNG files
    png_folder = 'Fusion/captures/DATA/camera_rgba'
    results = {}
    file_list = sorted(os.listdir(png_folder))

    num_workers = max(1, mp.cpu_count()-1)
    output_queue = mp.Manager().Queue()
    output_file = 'Lidar/data/combi_20250228_120139_2.csv'

    writer_process = mp.Process(target=csv_writer, args=(output_file, output_queue, len(file_list)))
    writer_process.start()

    with mp.Pool(num_workers) as pool:
        tasks = []
        previous_timestamp = None
        for filename in file_list:
            if filename.endswith('.png'):
                png_timestamp = extract_timestamp_from_filename(filename)
                if previous_timestamp is None:
                    previous_timestamp = png_timestamp - np.timedelta64(33333, 'us')

                # tasks.append((filename, lidar_timestamps, df, velocity_timestamps, velocity_speeds, previous_timestamp, png_timestamp, output_queue))
                tasks.append((filename, lidar_timestamps, df, None, None, previous_timestamp, png_timestamp, output_queue))
                previous_timestamp = png_timestamp

        results = list(tqdm(pool.imap_unordered(process_png_file, tasks), total=len(tasks), desc="Processing PNG files"))

    output_queue.put(None)
    writer_process.join()
    print(f'The results are saved in {output_file}')

# if __name__ == '__main__':
#     main()
#     exit()
    
# Read LiDAR data
csv_file = 'Lidar/data/lidar_20250228_120139.csv'
df = pd.read_csv(csv_file, dtype={'X': float, 'Y': float, 'Z': float, 'Intensity': int, 'Tag Information': int}, low_memory=True)
df.columns = ['UTC Timestamp', 'Local Timestamp', 'X', 'Y', 'Z', 'Intensity', 'Tag Information']
df['Local Timestamp'] = pd.to_datetime(df['Local Timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
df.dropna(subset=['Local Timestamp'], inplace=True)
lidar_timestamps = df['Local Timestamp'].to_numpy()

# # Read speed data
# speed_file = 'Fusion/captures/speed_test.csv'
# speed_df = pd.read_csv(speed_file)
# speed_df['Time'] = pd.to_datetime(speed_df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
# speed_df.dropna(subset=['Time'], inplace=True)
# # speed_df = pd.read_csv(speed_file, parse_dates=['Time'])
# # speed_df.dropna(subset=['Time'], inplace=True)
# velocity_timestamps = speed_df['Time'].to_numpy()
# velocity_speeds = speed_df['Speed (km/h)'].to_numpy()

# Read PNG files
png_folder = 'Fusion/captures/DATA/camera_rgba'
results = {}
file_list = sorted(os.listdir(png_folder))

output_file = 'Lidar/data/combi_20250228_120139_3.csv'
# Match Lidar timestamp with PNG timestamp
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'Local Timestamp', 'X', 'Y', 'Z', 'Intensity', 'Tag Information', 'Speed'])

    previous_timestamp = None
    for filename in tqdm(file_list):
        if filename.endswith('.png'):
            png_timestamp = extract_timestamp_from_filename(filename)
            if previous_timestamp is None:
                previous_timestamp = png_timestamp - np.timedelta64(33333, 'us')
            
            start_idx = np.searchsorted(lidar_timestamps, previous_timestamp, side="right")
            end_idx = np.searchsorted(lidar_timestamps, png_timestamp, side="right")
            filtered_df = df.iloc[start_idx:end_idx]
            
            updated_records = [] 
            # Start to combine LiDAR data with speed data
            for _, row in filtered_df.iterrows():
                target_timestamp = row['Local Timestamp']
                # vx = interpolate_velocity(velocity_timestamps, velocity_speeds, target_timestamp)
                vx = 0
                if vx is None:
                    print("It's none")
                    continue
                # time_diff = (target_timestamp - png_timestamp).total_seconds()
                # new_x = float(row['X']) + float(vx * time_diff)
                new_x = row['X']
                updated_records.append([filename, target_timestamp, new_x, row['Y'], row['Z'], row['Intensity'], row['Tag Information'], vx])
                
            writer.writerows(updated_records)
            previous_timestamp = png_timestamp

print(f'The results are saved in {output_file}')
