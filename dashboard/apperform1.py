import streamlit as st
import cv2
import numpy as np
import tempfile
import pyttsx3

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

# Evaluation variables
true_positives = 0
false_positives = 0
false_negatives = 0

st.markdown("<h1 style='text-decoration: underline;'>Sidewalk Obstacle Detection and Navigation Assistance for the Visually Impaired</h1>", unsafe_allow_html=True)

st.title("Obstacle Detection")
st.write("Press and activate your camera to start")

if st.button("Obstacle Detection"):
    # Iniciar camara
    cap = cv2.VideoCapture(0)

    FRAME_WINDOW = st.image([])

    while True:
        ret, frame = cap.read()

        # Convertir la imagen a RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Object detection
        (class_ids, score, bboxes) = model.detect(frame)

        if len(class_ids) == 0:
            print("Object Detected.")

        for class_id, score, bbox in zip(class_ids, score, bboxes):
            if class_id < len(classes):
                class_name = classes[class_id]
            else:
                class_name = "Unknown Class"
                print(f"Invalid class_id: {class_id}")
            (x, y, w, h) = bbox

            # Colocar el nombre de las clases
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
            # Colocar el rectangulo
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)

            # Evaluate against ground truth
            # Here, you would compare the predicted bounding box (bbox) with ground truth annotations
            # Update true_positives, false_positives, false_negatives accordingly

            # Speak out the detected object
            engine.say(f"Detected {class_name}")
            engine.runAndWait()

        if frame.shape[0] > 0 and frame.shape[1] > 0:
            FRAME_WINDOW.image(frame)

# Compute precision and recall
if true_positives + false_positives > 0:
    precision = true_positives / (true_positives + false_positives)
else:
    precision = 0.0

if true_positives + false_negatives > 0:
    recall = true_positives / (true_positives + false_negatives)
else:
    recall = 0.0

print("Precision:", precision)
print("Recall:", recall)
