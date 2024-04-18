from ultralytics import YOLO

model = YOLO("best.pt")
model.val(data="dataset.yaml", batch=24, imgsz=640, conf=0.001, iou=0.6)
