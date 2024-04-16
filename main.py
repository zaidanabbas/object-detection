import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO



model=YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

input("Password  ")
abbas = input("Enter file name  ")
cap=cv2.VideoCapture(abbas)
# image = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
print(class_list)
count=0
while True:
    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame, show=True)



    # cv2.imshow("RGB", frame)
    if cv2.waitKey(30)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()