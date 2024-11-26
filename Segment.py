import cv2
import numpy as np
import argparse
import time
from ultralytics import YOLO

# Load the YOLOv8 model with segmentation
model = YOLO("yolov8s-seg.pt")

# Define the target classes and their colors
TARGET_CLASSES = {
    'cup': (0, 0, 255),'person': (0, 0, 255), 'bottle': (0, 0, 255),  # Red
    'keyboard': (0, 255, 0), 'mouse': (0, 255, 0), 'cell phone': (0, 255, 0), 
    'laptop': (0, 255, 0), 'book': (0, 255, 0), 'remote': (0, 255, 0)   # Green
}

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='0', help='Input source. Use "0" for webcam or specify an image file path.')
parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on: "cpu" or "mps" for Apple Silicon GPU')
args = parser.parse_args()

# Load image or webcam input
def get_input_source(input_source):
    if input_source.isdigit():
        # Webcam input
        return cv2.VideoCapture(int(input_source))
    else:
        # Image input
        image = cv2.imread(input_source)
        if image is None:
            print(f"Error: Failed to read image from {input_source}")
            exit(1)
        return image

# Apply semi-transparent mask
def apply_mask_with_opacity(annotated_frame, mask, color, opacity=0.5):
    overlay = annotated_frame.copy()
    points = np.int32([mask])
    cv2.fillPoly(overlay, points, color)
    cv2.addWeighted(overlay, opacity, annotated_frame, 1 - opacity, 0, annotated_frame)

# Filter detections and draw semi-transparent masks without class names
def filter_and_annotate(results, frame):
    annotated_frame = frame.copy()

    # Retrieve all class names from the YOLO model
    yolo_classes = list(model.names.values())
    
    for result in results:
        # Check if masks exist
        if result.masks is not None:
            for mask, box in zip(result.masks.xy, result.boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # Check if the detected class is in the target list
                if class_name in TARGET_CLASSES:
                    # Use predefined color for the class
                    color = TARGET_CLASSES[class_name]

                    # Apply the mask with reduced opacity (semi-transparent)
                    apply_mask_with_opacity(annotated_frame, mask, color)

    return annotated_frame

# Function to calculate and display FPS
def display_fps(annotated_frame, fps):
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Main function
if __name__ == '__main__':
    # Check if input is webcam or image
    input_source = args.input
    device = args.device  # 'cpu' or 'mps'

    # Set device (cpu or mps) for YOLO model
    if input_source.isdigit():
        # Webcam
        cap = get_input_source(input_source)
        prev_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break

            # Perform inference with specified device
            results = model(frame, conf=0.5, device=device)

            # Filter and annotate results for target classes
            if results:
                annotated_frame = filter_and_annotate(results, frame)
            else:
                annotated_frame = frame  # If no results, show original frame

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Display FPS on the frame
            display_fps(annotated_frame, fps)

            # Display the image with detections
            cv2.imshow("YOLOv8 Detection", annotated_frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release webcam and close windows
        cap.release()
        cv2.destroyAllWindows()

    else:
        # Image input
        frame = get_input_source(input_source)

        if frame is not None:
            # Perform inference with specified device
            results = model(frame, conf=0.5, device=device)

            # Filter and annotate results for target classes
            annotated_frame = filter_and_annotate(results, frame)

            # Display the image with detections
            cv2.imshow("YOLOv8 Detection", annotated_frame)

            # Wait for key press to close window
            cv2.waitKey(0)
            cv2.destroyAllWindows()
