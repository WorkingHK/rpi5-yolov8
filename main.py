import cv2
from ultralytics import YOLO
import numpy as np

import torch 
#print(torch.backends.mps.is_available())

cap = cv2.VideoCapture(0)


model = YOLO("yolomodel/best.pt")



while True:
    ret , frame = cap.read()
    if not ret:
        break
    #Change according to processors(cpu,gpu,macbook)
    height, width = frame.shape[:2]
    center_x = width // 2
    center_y = height // 2
    results = model(frame, device="mps") 
    result = results[0]

    bboxes = np.array(result.boxes.xyxy.cpu() , dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    cv2.line(frame, (center_x, 0), (center_x, height), (0, 255, 0), 2)
    cv2.line(frame, (0, center_y), (width, center_y), (0, 255, 0), 2)

    for cls , bbox in zip(classes,bboxes):
        (x , y , x2 , y2) = bbox
        object_center_x = (x + x2) // 2
        object_center_y = (y + y2) // 2

        cv2.rectangle(frame, (x, y),(x2, y2), (0, 0, 225), 2)
        cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,225),2)
        cv2.putText(frame, f"({object_center_x}, {object_center_y})", (object_center_x, object_center_y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Detection" , frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()