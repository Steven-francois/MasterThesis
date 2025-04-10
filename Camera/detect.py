from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Predict with the model
results = model.predict(source="Fusion/captures30/camera_rgba/2025-03-12_12-39-52-200401.png", save=True, conf=0.5, show=True)
print(results)
# results = model.predict(source="Fusion/captures30/camera_rgba/2025-03-12_12-37-38-739616.png", save=True, conf=0.5, show=True)
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box
    
    print("xywh:", xywh)
    print("xywhn:", xywhn)
    print("xyxy:", xyxy)
    print("xyxyn:", xyxyn)
    print("names:", names)
    print("confs:", confs)
    print("--------------------------------------------------")
