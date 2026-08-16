import streamlit as st
import cv2
import numpy as np
import pyttsx3  # Import the pyttsx3 library for text-to-speech (TTS) functionality

# Opencv DNN
net = cv2.dnn.readNet("../model/dnn_model/yolo-tiny.weights", "../model/dnn_model/yolo-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Cargar class list
classes = []
with open("../model/dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

# Initialize pyttsx3 for voice output
engine = pyttsx3.init()

# Function to estimate distance based on object size
def estimate_distance(w):
    # Assuming a constant relationship between object size and distance
    # Adjust this ratio based on your specific setup and calibration
    constant_ratio = 20  # Experimentally determined constant ratio
    
    # Calculate approximate distance
    distance = constant_ratio / w
    
    return distance * 10

if st.button("Obstacle Detection"):
    # Initialize camera
    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        (class_ids, _, bboxes) = model.detect(frame)

        if len(class_ids) == 0:
            print("No object detected.")

        for class_id, bbox in zip(class_ids, bboxes):
            (x, y, w, h) = bbox

            # Perform distance estimation
            distance = estimate_distance(w)

            # Draw distance on the frame
            class_name = classes[class_id]
            cv2.putText(frame, f"{class_name} - Distance: {distance:.2f} meters", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)

            engine.say(f"Detected {class_name} at {distance:.2f} meters")
            engine.runAndWait()

        if frame.shape[0] > 0 and frame.shape[1] > 0:
            FRAME_WINDOW.image(frame)
