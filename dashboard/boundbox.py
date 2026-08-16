import cv2
import numpy as np
import pyttsx3

# Opencv DNN
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

# Evaluation variables
true_positives = 0
false_positives = 0
false_negatives = 0

# Function to update evaluation metrics
def update_metrics(class_ids, bboxes, ground_truth):
    # Implement your logic to update true_positives, false_positives, false_negatives
    pass

# Main program
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Convert the image to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Object detection
        (class_ids, _, bboxes) = model.detect(frame)

        # Update evaluation metrics
        update_metrics(class_ids, bboxes, ground_truth)  # You need to define ground_truth

        for class_id, bbox in zip(class_ids, bboxes):
            if class_id < len(classes):
                class_name = classes[class_id]
            else:
                class_name = "Unknown Class"
                print(f"Invalid class_id: {class_id}")
            (x, y, w, h) = bbox

            # Draw class name and rectangle
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)

            # Speak out the detected object
            engine.say(f"Detected {class_name}")
            engine.runAndWait()

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Compute precision and recall
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0

    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0.0

    # Save performance metrics to a file
    with open("performance_metrics.txt", "w") as file:
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
