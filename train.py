from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(data="dataset.yaml", epochs=10, device="cuda", batch=24)
