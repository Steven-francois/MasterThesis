import os
import pandas as pd
from datetime import datetime
import csv
from tqdm import tqdm
import numpy as np
import multiprocessing as mp

import matplotlib.pyplot as plt
import open3d as o3d

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
        ratio = (target_timestamp - before_timestamp)/ (after_timestamp - before_timestamp)
        ratio = ratio.astype(float)
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
        vx = interpolate_velocity(velocity_timestamps, velocity_speeds, target_timestamp)
        # vx = 0
        if vx is None:
            print("It's none")
            continue
        time_diff = (target_timestamp - png_timestamp).total_seconds()
        new_x = float(row['X']) + float(vx * time_diff)
        # new_x = row['X']
        updated_records.append([filename, target_timestamp, new_x, row['Y'], row['Z'], row['Intensity'], row['Tag Information'], vx])
        
    output_queue.put(updated_records)
    return len(updated_records)

def csv_writer(output_file, queue, total_files):
    print(f"Writing to {output_file}")
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
    print("Done writing to csv")

def main():
# if __name__ == '__main__':
    # Read LiDAR data
    print("Reading LiDAR data...")
    lidar_file = 'Fusion/data/lidar_20250312_123525'
    lidar_timestamps_file = f"{lidar_file}_ts.npy"
    lidar_data_file = f"{lidar_file}_data.npy"
    lidar_timestamps = np.load(lidar_timestamps_file)
    with open(lidar_data_file, 'rb') as f:
        lidar_data = [np.load(f) for _ in range(len(lidar_timestamps))]
    print("LiDAR data has been read")

    # Read speed data
    print("Reading speed data...")
    speed_file = 'Fusion/captures30/speed_test.csv'
    speed_df = pd.read_csv(speed_file)
    speed_df['Time'] = pd.to_datetime(speed_df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    speed_df.dropna(subset=['Time'], inplace=True)
    velocity_timestamps = speed_df['Time'].to_numpy()
    velocity_speeds = speed_df['Speed (km/h)'].to_numpy()
    print("Speed data has been read")

    # Read PNG files
    print("Reading PNG files...")
    png_folder = 'Fusion/captures30/camera_rgba'
    file_list = sorted(os.listdir(png_folder))
    print("PNG files have been read")

    # num_workers = max(1, mp.cpu_count()-1)
    num_workers = 4
    print(f"Number of workers: {num_workers}")
    output_queue = mp.Manager().Queue()
    output_file = 'Fusion/data/combination_20250312_123525.csv'

    writer_process = mp.Process(target=csv_writer, args=(output_file, output_queue, len(file_list)))
    writer_process.start()
    
    print("Creating pool...")
    with mp.Pool(num_workers) as pool:
        tasks = []
        previous_timestamp = None
        for filename in tqdm(file_list, desc="Creating tasks"):
            if filename.endswith('.png'):
                png_timestamp = extract_timestamp_from_filename(filename)
                if previous_timestamp is None:
                    previous_timestamp = png_timestamp - np.timedelta64(33333, 'us')

                tasks.append((filename, lidar_timestamps, df, velocity_timestamps, velocity_speeds, previous_timestamp, png_timestamp, output_queue))
                # tasks.append((filename, lidar_timestamps, df, None, None, previous_timestamp, png_timestamp, output_queue))
                previous_timestamp = png_timestamp
        print("Processing PNG files...")
        # pool.imap_unordered(process_png_file, tasks)
        results = list(tqdm(pool.imap_unordered(process_png_file, tasks), total=len(tasks), desc="Processing PNG files"))

    output_queue.put(None)
    writer_process.join()
    print(f'The results are saved in {output_file}')

# if __name__ == '__main__':
#     main()  
#     print("Done")
#     exit()
# print("Done2")
# exit()    


# Read LiDAR data
# csv_file = 'Fusion/data/lidar_20250312_123525.csv'
# df = pd.read_csv(csv_file, dtype={'X': float, 'Y': float, 'Z': float, 'Intensity': int, 'Tag Information': int}, low_memory=True)
# df.columns = ['UTC Timestamp', 'Local Timestamp', 'X', 'Y', 'Z', 'Intensity', 'Tag Information']
# df['Local Timestamp'] = pd.to_datetime(df['Local Timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
# df.dropna(subset=['Local Timestamp'], inplace=True)
# lidar_timestamps = df['Local Timestamp'].to_numpy()
lidar_file = 'Fusion/data/lidar_20250312_123525'
lidar_timestamps_file = f"{lidar_file}_ts.npy"
lidar_data_file = f"{lidar_file}_data.npy"
lidar_timestamps = np.load(lidar_timestamps_file, allow_pickle=True)
lidar_timestamps = np.array([np.datetime64(ts) for ts in lidar_timestamps])
with open(lidar_data_file, 'rb') as f:
    lidar_data = [np.load(f, allow_pickle=True) for _ in range(len(lidar_timestamps))]

# Read speed data
speed_file = 'Fusion/captures30/speed_test.csv'
speed_df = pd.read_csv(speed_file)
speed_df['Time'] = pd.to_datetime(speed_df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
speed_df.dropna(subset=['Time'], inplace=True)
# speed_df = pd.read_csv(speed_file, parse_dates=['Time'])
# speed_df.dropna(subset=['Time'], inplace=True)
velocity_timestamps = speed_df['Time'].to_numpy()
velocity_speeds = speed_df['Speed (km/h)'].to_numpy()

# Read PNG files
png_folder = 'Fusion/captures30/camera_rgba'
results = {}
file_list = sorted(os.listdir(png_folder))

output_file = 'Fusion/data/combination_20250312_123525'
output_data = f"{output_file}_data.npy"
output_timestamps = f"{output_file}_ts.npy"
timestamps = []
# Match Lidar timestamp with PNG timestamp
with open(output_data, 'wb') as f:
    previous_timestamp = None
    for filename in tqdm(file_list):
        if filename.endswith('.png'):
            png_timestamp = extract_timestamp_from_filename(filename)
            if previous_timestamp is None:
                previous_timestamp = png_timestamp - np.timedelta64(33333, 'us')
            
            start_idx = np.searchsorted(lidar_timestamps, previous_timestamp, side="right")
            end_idx = np.searchsorted(lidar_timestamps, png_timestamp, side="right")
            filtered_df = lidar_data[start_idx:end_idx]
            updated_records = np.empty((0, 5))
            # Start to combine LiDAR data with speed data
            for idx, points in enumerate(filtered_df):
                target_timestamp = lidar_timestamps[start_idx+idx]
                print(target_timestamp)
                # target_timestamp = row['Local Timestamp']
                # print(target_timestamp, type(target_timestamp))
                # print(velocity_timestamps[0], type(velocity_timestamps[0]))
                vx = interpolate_velocity(velocity_timestamps, velocity_speeds, target_timestamp)
                # vx = 0
                if vx is None:
                    print("It's none")
                    continue
                time_diff = (target_timestamp - png_timestamp)
                time_diff = time_diff.astype(float) / 1e9
                points[:, 0] = points[:, 0] + vx * time_diff
                updated_records = np.vstack((updated_records, points))
            # writer.writerows(updated_records)
            np.save(f, updated_records)
            timestamps.append(png_timestamp)
            previous_timestamp = png_timestamp
    print(f"Data has been saved to {output_data}.")
np.save(output_timestamps, timestamps)
print(f"Timestamps have been saved to {output_timestamps}.")
print(f'The results are saved in {output_file}')
