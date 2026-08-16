import streamlit as st
import cv2
import numpy as np
import tempfile
import pyttsx3

# Import the performance evaluation function
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, average_precision_score, precision_recall_curve

# Function to calculate the metric IoU (Intersection over Union) for a couple of bounding boxes
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou

# Function to calculate evaluation metrics based on detected bounding boxes
def evaluate_performance(ground_truth_boxes, detected_boxes):
    iou_scores = []
    binary_iou_scores = []

    for true_box in ground_truth_boxes:
        iou_scores_for_true_box = [calculate_iou(true_box, pred_box) for pred_box in detected_boxes]
        iou_scores.append(max(iou_scores_for_true_box))
        binary_iou_scores.append(max(iou_scores_for_true_box) >= 0.5)  # IoU threshold for correct detection

    precision = precision_score([1] * len(ground_truth_boxes), binary_iou_scores)
    recall = recall_score([1] * len(ground_truth_boxes), binary_iou_scores)
    f1 = f1_score([1] * len(ground_truth_boxes), binary_iou_scores)
    accuracy = accuracy_score([1] * len(ground_truth_boxes), binary_iou_scores)

    return precision, recall, f1, accuracy

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

st.markdown("<h1 style='text-decoration: underline;'>Sidewalk Obstacle Detection and Navigation Assistance for the Visually Impaired</h1>", unsafe_allow_html=True)

st.title("Obstacle Detection")
st.write("Press and activate your camera to start")

if st.button("Obstacle Detection"):
    # Iniciar camara
    cap = cv2.VideoCapture(0)

    FRAME_WINDOW = st.image([])

    ground_truth_boxes = np.array([[50, 50, 200, 200], [300, 300, 450, 450], [100, 100, 250, 250]])

    while True:
        ret, frame = cap.read()

        # Convertir la imagen a RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Object detection
        (class_ids, scores, bboxes) = model.detect(frame)

        detected_boxes = []
        if len(class_ids) > 0:
            for bbox in bboxes:
                (x, y, w, h) = bbox
                detected_boxes.append([x, y, x + w, y + h])  # Format bounding boxes to [x1, y1, x2, y2]

                # Colocar el rectangulo
                cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)

        # Calculate performance metrics
        precision, recall, f1, accuracy = evaluate_performance(ground_truth_boxes, detected_boxes)

        if frame.shape[0] > 0 and frame.shape[1] > 0:
            FRAME_WINDOW.image(frame)

        # Display performance metrics
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")
        st.write(f"Accuracy: {accuracy:.2f}")
