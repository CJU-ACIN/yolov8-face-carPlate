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
