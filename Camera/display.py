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

    for image_file in image_files:
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
folder_path = 'runs/detect/predict4'
display_images_as_animation(folder_path, delay=0.06)