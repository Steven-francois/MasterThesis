import os
import json
import numpy as np

nb = "11_0"
data_folder = f"Data/{nb}/"
# data_folder = f"D:/p_{nb}/"
fusion_folder = os.path.join(data_folder, "fusion")
with open(os.path.join(fusion_folder, "targets.npy"), "rb") as f:
    num_frames = np.load(f, allow_pickle=True)
    frames = []
    for i in range(num_frames):
        with open(os.path.join(fusion_folder, f"targets_{i}.json"), 'r') as json_file:
            frame_data = json.load(json_file)
            frames.append(frame_data)
    number_of_targets = []
    for frame in frames:
        # number_of_targets.append(len(frame["cam_coords"]))
        number_of_targets.append(len(frame["targets_nb"]))
print(f"Number of frames: {num_frames}")
print(f"Number of targets in each frame: {number_of_targets}")

import matplotlib.pyplot as plt
def plot_targets_per_frame(number_of_targets):
    plt.figure(figsize=(10, 5))
    plt.plot(number_of_targets, marker='o', linestyle='-', color='b')
    plt.title('Number of Targets per Frame')
    plt.xlabel('Frame Index')
    plt.ylabel('Number of Targets')
    plt.grid(True)
    plt.show()

plot_targets_per_frame(number_of_targets)
