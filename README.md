# 7_PPE_Detection_YOLOv8


The Personal Protective Equipment (PPE) Detection Project is designed to enhance workplace safety by utilizing computer vision and deep learning technologies. The primary goal is to detect and ensure the proper use of PPE such as helmets, gloves, and safety vests in real-time through live video streams or uploaded video files.

# Key Features:
## 1. Real-Time Detection:

#### Webcam Integration: Users can start and stop the webcam to detect PPE usage in real-time. The system automatically adjusts the video stream dimensions for optimal display. Live Streaming: Real-time video processing is facilitated to ensure immediate detection and feedback.


## 2. Video File Processing:

#### File Upload: Users can upload video files for PPE detection. The system supports multiple file types, including .jpg, .jpeg, .png, and .mp4. Stream Playback: Once a video is uploaded, users can choose to start streaming the video, allowing the system to process and detect PPE usage throughout the video.

## 3. User-Friendly Interface:
#### Dynamic Display Adjustment: The display size of the video stream adjusts automatically based on the target height and width specified during the initialization of the webcam or video file processing. Interactive Controls: Simple buttons and prompts guide users through starting and stopping the webcam, uploading files, and initiating video streams.

## 4. Efficient Detection System:
#### YOLO Model: Utilizes the YOLO (You Only Look Once) deep learning model for fast and accurate object detection. Customized Detector: The Detector class processes the video input, applies the YOLO model, and outputs frames with detected PPE, ensuring efficient and accurate detection.

## 5. Responsive Design:
#### Progress Tracking: A progress bar displays the upload status of video files, providing real-time feedback on the upload process. Alerts and Prompts: Users are prompted to start streaming the uploaded video, ensuring they are always informed of the next steps. The PPE Detection Project leverages advanced technologies to promote safety and compliance in various environments. It is particularly useful in construction sites, factories, and other industrial settings where PPE is mandatory. By providing real-time feedback and ensuring the proper use of protective equipment, this project contributes significantly to reducing workplace accidents and injuries.


## Here are the step-by-step instructions for running the code:

### 1- Install requirements.txt:
To run this code type the given command in the terminal: pip install -r requirements.txt

## 2- Navigate to the Code Directory:
Use the cd command to navigate to the directory where your code is located. For example: cd path/to/your/code/directory

## 3- Download yolov8 weights:
Download yolov8 weigths and place in your Flask app code directory.

## 5- Run the Python Script:
Once you're in the directory containing your ClientApp.py file, run the following command: python ClientApp.py

Following these steps should launch your Flask web application, allowing you to access it through a web browser or any other HTTP client.

https://github.com/yinsights8/7_PPE_Detection_YOLOv8/assets/108249945/f1031d3b-d4bb-4621-b6e5-f388357f9075

