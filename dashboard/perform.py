import numpy as np
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
    dice_coefficient = 2 * np.sum(binary_iou_scores) / (len(binary_iou_scores) + 1e-8)

    precision_values, recall_values, _ = precision_recall_curve([1] * len(ground_truth_boxes),
                                                                [max(iou_scores_for_true_box) for _ in range(len(detected_boxes))])
    average_precision = average_precision_score([1] * len(ground_truth_boxes),
                                                 [max(iou_scores_for_true_box) for _ in range(len(detected_boxes))])

    mAP = average_precision
    mIoU = np.mean(iou_scores)

    return precision, recall, f1, accuracy, dice_coefficient, average_precision, mAP, mIoU

# Ground truth bounding boxes (you need to provide these)
ground_truth_boxes = np.array([[50, 50, 200, 200], [300, 300, 450, 450], [100, 100, 250, 250]])

# Detected bounding boxes from your object detection program (replace this with actual detected boxes)
# Here, you need to extract the bounding boxes from the 'bboxes' variable in your object detection program
detected_boxes = np.array([[60, 60, 190, 190], [310, 310, 440, 440], [90, 90, 240, 240]])

# Call the evaluate_performance function with ground truth and detected boxes
precision, recall, f1, accuracy, dice_coefficient, average_precision, mAP, mIoU = evaluate_performance(ground_truth_boxes, detected_boxes)

# Print or use the evaluation metrics as needed
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Dice Coefficient: {dice_coefficient:.2f}")
print(f"Average Precision (AP): {average_precision:.2f}")
print(f"Mean Average Precision (mAP): {mAP:.2f}")
print(f"Mean Intersection over Union (mIoU): {mIoU:.2f}")
