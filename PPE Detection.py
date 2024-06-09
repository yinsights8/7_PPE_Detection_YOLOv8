from ultralytics import YOLO
import cv2
import cvzone
import math
from utils.ImgResizer import ImageResizer

# create an instance for yolo model
model = YOLO("ppe.pt")

# create,set and start the dimensions for video rectangle

# cam = cv2.VideoCapture(0)   # for live webcam
# cam.set(3,1280)
# cam.set(4, 480)

videoPath = "videos/ppe-1-1.mp4"

cam = cv2.VideoCapture(videoPath)

class_names = [
    'Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone',
    'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck',
    'van', 'vehicle', 'wheel loader'
]

myColor = (0, 0, 255)

# when condition is true
while True:
    rate, image = cam.read()
    img = ImageResizer(scale_percentage=70).resize(image)
    results = model(img, stream=True)
    # traverse through results
    for r in results:
        # get the boxes
        boxes = r.boxes

        # now get the coordinates
        for box in boxes:
            # with CV2
            # x1, y1, x2, y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # # pass the coordinates to the cv2 rectangle
            # cv2.rectangle(img=img, pt1=(x1,y1), pt2=(x2,y2), color=(255,0,255), thickness=2)
            # print(x1, y1, x2, y2)

            # with cvzone
            # Bounding boxes
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # pass the coordinates to the cvzone rectangle
            # cvzone.cornerRect(img=img, bbox=(x1, y1, w, h), l=15)

            # confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # classes
            cls = int(box.cls[0])

            CurrentClass = class_names[cls]
            if CurrentClass == 'NO-Hardhat' or CurrentClass == 'NO-Mask' or CurrentClass == 'NO-Safety Vest':
                myColor = (0, 0, 255)

            elif CurrentClass == "bus" or CurrentClass == 'truck and trailer' or CurrentClass == 'truck' or CurrentClass == "Excavator" \
                    or CurrentClass == 'dump truck' or CurrentClass == 'van' or CurrentClass == 'vehicle' or CurrentClass == 'wheel loader':
                myColor = (255, 0, 0)

            else:
                myColor = (0, 255, 0)

            cvzone.putTextRect(img, f"{CurrentClass}, {conf}", pos=(max(0, x1), max(35, y1 - 15)), scale=1, thickness=2, colorB=myColor,
                               colorR=myColor, colorT=(255, 255, 255))
            cvzone.cornerRect(img=img, bbox=(x1, y1, w, h), l=20, t=3, colorR=myColor)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
