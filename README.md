# yolov8-face-carPlate

### dataset
[aihub]https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=172

[roboflow-face]https://universe.roboflow.com/mohamed-traore-2ekkp/face-detection-mik1i

[roboflow-carPlate]https://universe.roboflow.com/university-hassiba-ben-bouli/model-plate

### 학습
```
# train

from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(data="dataset.yaml", epochs=10, device="cuda", batch=24)
```


### 검증
```
# validate.py

from ultralytics import YOLO

model = YOLO("best.pt")
model.val(data="dataset.yaml", batch=24, imgsz=640, conf=0.001, iou=0.6)
```

### 이미지 예측
```
# image_predict.py

from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

results = model.predict("image_test.jpg")

for result in results:
    result.save(filename="image_result.jpg")  # save to disk


cv2.namedWindow("result", flags=cv2.WINDOW_NORMAL)
cv2.imshow("result", cv2.imread("image_result.jpg"))

cv2.waitKey(0)
cv2.destroyAllWindows()

```

### 비디오 예측
```
# video_predict.py

from time import time

start_time = time()

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2

model = YOLO("best.pt")
names = model.names

cap = cv2.VideoCapture("test_video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

# Blur ratio
blur_ratio = 50

# Video writer
video_writer = cv2.VideoWriter(
    "test_video_result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print(
            "Video frame is empty or video processing has been successfully completed."
        )
        break

    results = model.predict(im0, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    confs = results[0].boxes.conf.cpu().tolist()

    annotator = Annotator(im0, line_width=2, example=names)

    if boxes is not None:
        for box, cls, conf in zip(boxes, clss, confs):
            # annotaion box and label on video frame
            annotator.box_label(
                box, color=colors(int(cls), True), label=f"{names[int(cls)]} {conf:.2f}"
            )

            obj = im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
            blur_obj = cv2.blur(obj, (blur_ratio, blur_ratio))

            im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = blur_obj

    cv2.imshow("ultralytics", im0)
    video_writer.write(im0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Total time taken: {time() - start_time:.2f} seconds")

```