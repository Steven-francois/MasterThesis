from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import json
from tqdm import trange

# Load a model
model = YOLO("yolov9e.pt")

# Train the model
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

nb = "21_0"
data_folder = f"Data/{nb}/"
# data_folder = f"D:/p_{nb}/"
image_folder = os.path.join(data_folder, "camera")
image_filenames = sorted(os.listdir(image_folder))
images = [os.path.join(image_folder, filename) for filename in image_filenames]
targets_folder = os.path.join(data_folder, "cam_targets")
os.makedirs(targets_folder, exist_ok=True)


data = []
with open(os.path.join(targets_folder, "targets.npy"), "wb") as f:
    np.save(f, len(images))  # Save the number of results
    for i in trange(len(images), desc="Processing images"):
    # for i in range(570, 590):
    # for i in range(2500, 2800):
        result = model.track(images[i], persist=True, classes=[0,1,2,3,5,7], show=False, tracker="botsort_reid.yaml", save=True, conf=0.01, project=targets_folder, name='targets', exist_ok=True, verbose=False)[0]
        data.append({
            "conf": result.boxes.conf.cpu().tolist(),
            "cls": result.boxes.cls.cpu().tolist(),
            "ids": result.boxes.id.cpu().tolist() if result.boxes.id is not None else []
        })
        np.save(f, result.boxes.xywh.cpu().numpy())
    with open(os.path.join(targets_folder, "targets.json"), "w") as f:
        json.dump(data, f)
