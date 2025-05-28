from ultralytics import YOLO
import os
import numpy as np
import json
from tqdm import trange

# Load a model
model = YOLO("yolov9e.pt")

# Train the model
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

nb = "1_0"
data_folder = f"Data/{nb}/"
image_folder = os.path.join(data_folder, "camera")
image_filenames = sorted(os.listdir(image_folder))
images = [os.path.join(image_folder, filename) for filename in image_filenames]
targets_folder = os.path.join(data_folder, "camera", "targets")
os.makedirs(targets_folder, exist_ok=True)


data = []
with open(os.path.join(targets_folder, "targets.npy"), "wb") as f:
    np.save(f, len(images))  # Save the number of results
    for i in trange(len(images), desc="Processing images"):
        result = model.predict(images[i], save=True, show=False, conf=0.1, project=image_folder, name='targets', exist_ok=True, classes=[0,1,2,3,5,7], verbose=False)[0]
        data.append({
            "conf": result.boxes.conf.cpu().tolist(),
            "cls": result.boxes.cls.cpu().tolist()
        })
        np.save(f, result.boxes.xywh.cpu().numpy())
with open(os.path.join(targets_folder, "targets.json"), "w") as f:
    json.dump(data, f)

