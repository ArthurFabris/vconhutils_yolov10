from ultralytics import YOLO
import cv2
import numpy as np
import time

# Load the YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')

# Set confidence threshold (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.5

# Define the color to use for all outlines and text (green)
GREEN = (0, 255, 0)

# Open webcam
cap = cv2.VideoCapture(0)


# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set frame dimensions (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 380)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Process frames in a loop
while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame, conf=CONFIDENCE_THRESHOLD)
    
    # Create a copy of the frame to draw on
    annotated_frame = frame.copy()
    
    # Process the results
    for result in results:
        # Get detected objects' masks
        if result.masks is not None:
            # Process each mask
            for i, mask in enumerate(result.masks):
                # Get contour points of the mask
                contour_points = mask.xy[0]
                contour = np.array(contour_points, dtype=np.int32)
                
                # Get the class of this detection
                if result.boxes and i < len(result.boxes):
                    cls_id = int(result.boxes[i].cls.item())
                    conf = float(result.boxes[i].conf.item())
                    
                    # Only process if confidence is above threshold
                    if conf >= CONFIDENCE_THRESHOLD:
                        # Draw the outline of the object
                        cv2.polylines(annotated_frame, [contour], True, GREEN, 2)
                        
                        # Get position for label
                        x, y = contour.min(0)
                        
                        # Get the class name
                        class_name = result.names[cls_id]
                        
                        # Create label with class name and confidence
                        label = f"{class_name} {conf:.2f}"
                        
                        # Add text with class name and confidence
                        cv2.putText(annotated_frame, label, (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
    
    # Add confidence threshold info to the display
    cv2.putText(annotated_frame, f"Conf Threshold: {CONFIDENCE_THRESHOLD}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
    
    # Show the frame with detections
    annotated_frame = cv2.resize(annotated_frame, (1920,1080))
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()