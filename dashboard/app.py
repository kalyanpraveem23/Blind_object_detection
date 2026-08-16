import streamlit as st
import cv2
import numpy as np
import math
import time
import pyttsx3
import threading

# Load YOLO model
net = cv2.dnn.readNet("../model/dnn_model/yolo-tiny.weights", "../model/dnn_model/yolo-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)

# Load class list
classes = []
with open("../model/dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        classes.append(class_name.strip())

# Initialize pyttsx3 (Offline TTS)
engine = None


def initialize_tts():
    global engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
    return engine


# Initialize TTS in a separate thread to avoid blocking the main thread
tts_thread = threading.Thread(target=initialize_tts)
tts_thread.daemon = True
tts_thread.start()

# Parameters for distance estimation
FOCAL_LENGTH = 615  # Example value, needs calibration
KNOWN_WIDTH = 50  # Average width of a person in cm
PROXIMITY_THRESHOLD = 2.0  # meters - threshold to consider object "near" person
ALERT_COOLDOWN = 5  # seconds between alerts for the same object

# Dictionary to store last alert times for objects
last_alerts = {}


def speak_text(text):
    """Speak text using pyttsx3 in a separate thread"""

    def speak():
        try:
            global engine
            if engine is not None:
                engine.say(text)
                engine.runAndWait()
        except Exception as e:
            st.error(f"Error generating speech: {e}")

    # Start speech in a separate thread
    threading.Thread(target=speak).start()


def calculate_distance(bbox_width, real_width, focal_length):
    """Calculate distance using the formula: distance = (real_width * focal_length) / bbox_width"""
    if bbox_width == 0:
        return float('inf')
    distance = (real_width * focal_length) / bbox_width
    return distance


def calculate_distance_between_objects(obj1, obj2):
    """Calculate the Euclidean distance between two objects in meters"""
    center_distance_px = math.sqrt(
        (obj1["center"][0] - obj2["center"][0]) ** 2 +
        (obj1["center"][1] - obj2["center"][1]) ** 2
    )

    # Average of the two depth distances
    avg_distance = (obj1["distance"] + obj2["distance"]) / 2

    # Convert pixel distance to real-world distance
    scale_factor = center_distance_px / KNOWN_WIDTH  # pixels per cm
    real_distance = scale_factor * avg_distance / 100  # convert to meters

    return real_distance


st.title("Assistive Object Detection")
st.write("Detection system that alerts when items are near a person but not visible to others")

# Add calibration options
st.sidebar.header("Calibration Settings")
focal_length = st.sidebar.slider("Focal Length", 400, 1000, FOCAL_LENGTH)
known_width = st.sidebar.slider("Known Width (cm)", 10, 200, KNOWN_WIDTH)
proximity_threshold = st.sidebar.slider("Proximity Threshold (m)", 0.5, 5.0, PROXIMITY_THRESHOLD)

# Wait for TTS engine to initialize
tts_thread.join()

if st.button("Start Detection"):
    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])
    status_text = st.empty()

    if not cap.isOpened():
        st.error("Failed to access the camera.")
    else:
        st.write("Press *ESC* to stop.")

    current_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        (class_ids, scores, bboxes) = model.detect(frame)

        # Lists to store detected objects by category
        persons = []
        items = []
        other_people = []  # Will store people who might be able to see items

        # First loop: categorize detected objects
        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            class_name = classes[class_id] if class_id < len(classes) else "Unknown Class"
            x, y, w, h = bbox

            # Estimate distance from camera
            distance = calculate_distance(w, known_width, focal_length)

            # Create object info dictionary
            obj_info = {
                "class_name": class_name,
                "bbox": bbox,
                "center": (x + w // 2, y + h // 2),
                "distance": distance,
                "id": f"{class_name}{x}{y}"  # Create a simple ID for tracking
            }

            # Categorize objects
            if class_name.lower() == "person":
                # The first detected person is considered the primary user
                if not persons:
                    obj_info["type"] = "primary_user"
                    persons.append(obj_info)
                    # Draw box with special color for primary user
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(frame, f"User ({distance:.2f}m)", (x, y - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                else:
                    obj_info["type"] = "other_person"
                    other_people.append(obj_info)
                    # Draw box with different color for other people
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    cv2.putText(frame, f"Person ({distance:.2f}m)", (x, y - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
            else:
                obj_info["type"] = "item"
                items.append(obj_info)
                # Draw box for items
                cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)
                cv2.putText(frame, f"{class_name} ({distance:.2f}m)", (x, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 0, 50), 2)

        # Process detections only if a primary user is detected
        if persons:
            primary_user = persons[0]
            current_time = time.time()

            # Check each item's proximity to the primary user
            for item in items:
                item_distance = calculate_distance_between_objects(primary_user, item)

                # If item is close to primary user
                if item_distance < proximity_threshold:
                    # Draw line between primary user and nearby item
                    cv2.line(frame, primary_user["center"], item["center"], (0, 255, 255), 2)

                    # Check if other people can see this item
                    item_visible_to_others = False
                    for other_person in other_people:
                        # Check if other person is facing the item (simplified check)
                        # In a more advanced implementation, you could check facing direction
                        if calculate_distance_between_objects(other_person, item) < proximity_threshold * 1.5:
                            item_visible_to_others = True
                            break

                    # Generate alert if item is not visible to others and cooldown period has passed
                    if not item_visible_to_others:
                        item_id = item["id"]
                        if item_id not in last_alerts or (current_time - last_alerts[item_id]) > ALERT_COOLDOWN:
                            alert_message = f"Alert: {item['class_name']} is {item_distance:.1f} meters from you"

                            # Update last alert time
                            last_alerts[item_id] = current_time

                            # Display alert on screen
                            alert_pos = (primary_user["center"][0], primary_user["center"][1] - 30)
                            cv2.putText(frame, alert_message, alert_pos,
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                            # Generate and play voice alert
                            speak_text(alert_message)

                            # Update status
                            status_text.text(f"Alert: {item['class_name']} detected near you")

        FRAME_WINDOW.image(frame)

        # Press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            st.write("Stopping detection...")
            break

    cap.release()
    cv2.destroyAllWindows()

