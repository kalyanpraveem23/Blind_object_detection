import streamlit as st
import cv2
import numpy as np
import tempfile
import pyttsx3

# Load YOLO model
net = cv2.dnn.readNet("../model/dnn_model/yolo-tiny.weights", "../model/dnn_model/yolo-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Load class list
classes = []
with open("../model/dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

# Initialize pyttsx3 for voice output
engine = pyttsx3.init()

# Streamlit UI
st.markdown("<h1 style='text-decoration: underline;'>Sidewalk Obstacle Detection and Navigation Assistance for the Visually Impaired</h1>", unsafe_allow_html=True)
st.title("Obstacle Detection")
st.write("Press the button to start obstacle detection using your camera.")

# Button to start detection
if st.button("Start Detection"):
    # Initialize camera
    cap = cv2.VideoCapture(0)

    # Function to convert text to speech
    def speak(text):
        engine.say(text)
        engine.runAndWait()

    # Main detection loop
    while True:
        ret, frame = cap.read()

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        classes, scores, boxes = model.detect(frame, confThreshold=0.3, nmsThreshold=0.4)

        # Process detected objects
        if len(classes) > 0:
            for class_id, score, box in zip(classes.flatten(), scores.flatten(), boxes):
                class_name = classes[class_id]
                score = scores[class_id]
                bbox = boxes[class_id]
                
                # Draw bounding box
                cv2.rectangle(frame, bbox, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, f"{class_name}: {score:.2f}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Speak out the detected object
                speak(f"Detected {class_name}")

        # Display frame with detections
        st.image(frame_rgb, channels="RGB")

    # Release camera and close
    cap.release()
    cv2.destroyAllWindows()
