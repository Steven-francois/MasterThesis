from ultralytics import YOLO
import os
import numpy as np
import json
from tqdm import trange

# Load a model
model = YOLO("yolov9e.pt")

# Train the model
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

nb = "11_0"
data_folder = f"Data/{nb}/"
# data_folder = f"D:/p_{nb}/"
image_folder = os.path.join(data_folder, "camera")
image_filenames = sorted(os.listdir(image_folder))
images = [os.path.join(image_folder, filename) for filename in image_filenames]
# targets_folder = os.path.join(data_folder, "cam_targets")
# os.makedirs(targets_folder, exist_ok=True)


data = []
# with open(os.path.join(targets_folder, "targets.npy"), "wb") as f:
    # np.save(f, len(images))  # Save the number of results
# for i in trange(len(images), desc="Processing images"):
# for i in range(570, 590):
# for i in range(2500, 2800):
#     result = model.track(images[i], persist=True, classes=[0,1,2,3,5,7], show=True, tracker="botsort_reid.yaml")[0]
#     if result.boxes is not None:
#         print(f"Image {i}:")
#         print(result.boxes.xywh.cpu().numpy())
#     if result.boxes.id is not None:
#         print(result.boxes.id.cpu().tolist())
    # result = model.predict(images[i], save=True, show=False, conf=0.1, project=targets_folder, name='targets', exist_ok=True, classes=[0,1,2,3,5,7], verbose=False)[0]
    # data.append({
    #     "conf": result.boxes.conf.cpu().tolist(),
    #     "cls": result.boxes.cls.cpu().tolist()
    # })
        # np.save(f, result.boxes.xywh.cpu().numpy())
# with open(os.path.join(targets_folder, "targets.json"), "w") as f:
#     json.dump(data, f)
tracking_intervals = [
        # slice(155, 185),
        # slice(300, 435),
        # slice(470, 540),
        # slice(980, 1058),
        # slice(1258, 1270)
        # slice(2345, 2385),
        # slice(3228, 3300),
        # slice(3587, 3624),
        # slice(3869, 3942),
        # slice(4119, 4442)
        # slice(2500, 2600)
        slice(300, 540)
    ]
# tracking_intervals = [
#         slice(745, 816),
#         slice(1138, 1169),
#         slice(1170, 1181),
#         slice(1181, 1191),
#         slice(1192, 1204),
#         slice(1204, 1241),
#         slice(2421, 2508),
#         slice(2509, 2535),
#         slice(2541, 2575),
#         slice(2600, 2615),
#         slice(3002, 3049),
#     ]
nb_tracks = []
for interval in tracking_intervals:
    nb_t = 0
    model = YOLO("yolov9e.pt")
    ids = []
    for i in range(interval.start, interval.stop):
        result = model.track(images[i], persist=True, classes=[0,1,2,3,5,7], show=True, tracker="botsort_reid.yaml", save=False)[0]
        # result = model.predict(images[i], classes=[0,1,2,3,5,7], show=False, save=True)[0]
        if result.boxes is not None:
            if result.boxes.id is not None:
                ids.extend(result.boxes.id.cpu().tolist())
                print(result.boxes.id.cpu().tolist(), len(result.boxes.xywh.cpu().numpy()))
    nb_t = len(set(ids))
            
    nb_tracks.append(nb_t)
print(f"Number of tracks in each interval: {nb_tracks}")
