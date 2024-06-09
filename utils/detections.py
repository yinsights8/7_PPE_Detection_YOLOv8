import cv2
import cvzone
import math
from utils.sort import *
from ultralytics import YOLO

class Detector:
    def __init__(self, videoPath, modelPath, target_video_width = 640):
        self.videoPath = videoPath
        self.modelPath = modelPath

        self.model = YOLO(self.modelPath)
        self.cam = cv2.VideoCapture(self.videoPath)

        self.classNames = "utils/coco.names"
        self.class_names = self.readClass(self.classNames)


        # Get original video width and height
        self.original_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate the aspect ratio
        aspect_ratio = self.original_width / self.original_height

        # Set the target width and height
        self.target_width = target_video_width
        self.target_height = int(self.target_width / aspect_ratio)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (self.target_width, self.target_height))

    # video will automatically turn off after closing website
    def __del__(self):
        return self.cam.release()
    
    # Method for reading classes
    def readClass(self, classFile):
        """
        This function will read the class names
        """
        with open(classFile, 'r') as f:
            self.classList = f.read().split()
        return self.classList

    def outputFrames(self):
        """
        This function is use to display the predictions on live video
        This function can be use when predicting on live fotage
        """
        # while True:
        success, image = self.cam.read()
        resized_frame = cv2.resize(image, (self.target_width, self.target_height))
        results = self.model(resized_frame, stream=True)
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
                clss = int(box.cls[0])

                CurrentClass = self.class_names[clss]
                if CurrentClass == 'NO-Hardhat' or CurrentClass == 'NO-Mask' or CurrentClass == 'NO-Safety Vest':
                    myColor = (0, 0, 255)

                elif CurrentClass == "bus" or CurrentClass == 'truck and trailer' or CurrentClass == 'truck' or CurrentClass == "Excavator" \
                        or CurrentClass == 'dump truck' or CurrentClass == 'van' or CurrentClass == 'vehicle' or CurrentClass == 'wheel loader':
                    myColor = (255, 0, 0)

                else:
                    myColor = (0, 255, 0)

                cvzone.putTextRect(resized_frame, f"{CurrentClass}, {conf}", pos=(max(0, x1), max(35, y1 - 15)), scale=1, thickness=2, colorB=myColor,
                                colorR=myColor, colorT=(255, 255, 255))
                cvzone.cornerRect(img=resized_frame, bbox=(x1, y1, w, h), l=20, t=3, colorR=myColor)

        # cv2.imshow("Image", img)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


    ####################################################################################################################
        self.out.write(resized_frame)
        ret, buffer = cv2.imencode(".jpg", resized_frame)
        frame = buffer.tobytes()

        return frame