import os
import time
from PIL import Image

import matplotlib.pyplot as plt

def display_images_as_animation(folder_path, delay=1):
    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    image_files.sort()  # Sort files alphabetically

    if not image_files:
        print("No images found in the folder.")
        return

    # Initialize the plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()

    # for image_file in image_files[300: 560]:
    # for image_file in image_files[300: 450]:
    # for image_file in image_files[155: 185]:
    # for image_file in image_files[300: 435]:
    # for image_file in image_files[470: 540]:
    # for image_file in image_files[818: 1058]:
    # for image_file in image_files[1258: 1285]:
    # for image_file in image_files[2345: 2385]:
    # for image_file in image_files[3227: 3318]:
    # for image_file in image_files[3587: 3624]:
    # for image_file in image_files[3869: 3942]:
    # for image_file in image_files[4119: 4442]:
    # for image_file in image_files[2500: 2800]:

    tracking_intervals = [
        # slice(155, 185),
        # slice(300, 435),
        # slice(470, 540),
        # slice(980, 1058),
        # slice(1258, 1285),
        # slice(2345, 2385),
        # slice(3228, 3300),
        # slice(3587, 3624),
        # slice(3869, 3942),
        # slice(4119, 4442),
        # slice(2500, 2800)
        slice(2500, 3000)
    ]
    # tracking_intervals = [
    #     slice(745, 816),
    #     slice(1138, 1169),
    #     slice(1170, 1181),
    #     slice(1181, 1191),
    #     slice(1192, 1204),
    #     slice(1204, 1241),
    #     slice(2421, 2508),
    #     slice(2509, 2535),
    #     slice(2541, 2580),
    #     slice(2600, 2619),
    #     slice(2969, 3053),
    # ]
    for interval in tracking_intervals:
        for image_file in image_files[interval]:
            image_path = os.path.join(folder_path, image_file)
            img = Image.open(image_path)

            # Display the image
            ax.clear()
            ax.imshow(img)
            ax.axis('off')  # Hide axes
            plt.title(image_file)
            plt.pause(delay)  # Pause for the specified delay

    plt.ioff()  # Turn off interactive mode
    plt.show()

# Replace 'your_folder_path' with the path to your folder containing images
folder_path = 'Data/11_0/cam_targets/targets'
# folder_path = 'D:/p_11_0/cam_targets/targets'
display_images_as_animation(folder_path, delay=0.06)