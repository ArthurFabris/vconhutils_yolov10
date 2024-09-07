from ultralytics import YOLOv10
import os
import cv2
import time

# List of all model names from jameslahm and onnx-community
models = [
    'jameslahm/yolov10n',  # 0 
    'jameslahm/yolov10s',  # 1
    'jameslahm/yolov10m',  # 2
    'jameslahm/yolov10b',  # 3
    'jameslahm/yolov10l',  # 4
    'jameslahm/yolov10x',  # 5
]

# Choose the model based on an integer variable
model_choice = 0  # Change this value to select different models (0 for yolov10n, 1 for yolov10s, etc.)

# Load the chosen model
model_name = models[model_choice]
try:
    model = YOLOv10.from_pretrained(model_name)
    print(f'Using model: {model_name}')
except Exception as e:
    print(f"Error loading model {model_name}: {e}")
    exit()

# Open the webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Customize the display options
confidence_threshold = 0.5  # Minimum confidence for predictions
font_scale = 1  # Scale of the font for displaying text
font_color = (0, 255, 0)  # Color of the font (BGR format)
box_color = (0, 255, 0)  # Color of the bounding box (BGR format)
thickness = 1  # Thickness of the bounding box lines

# Process frames from the webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    start_time = time.time()  # Start time for frame rendering
    
    # Predict on the current frame, using the confidence threshold
    results = model.predict(source=frame, conf=confidence_threshold)
    
    # Extract detection results from the prediction
    for pred in results[0].boxes:
        x1, y1, x2, y2 = map(int, pred.xyxy[0])  # Bounding box coordinates
        confidence = pred.conf[0]  # Confidence score
        class_id = int(pred.cls[0])  # Class label index
        
        if confidence < confidence_threshold:
            continue  # Skip if below confidence threshold
        
        label = f"Class {class_id}: {confidence:.2f}"  # Label with class and confidence
        
        # Draw the bounding box manually
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
        
        # Put the label above the bounding box
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)
    
    # Calculate render time and FPS
    end_time = time.time()
    render_time_ms = (end_time - start_time) * 1000  # Render time in milliseconds
    fps = 1 / (end_time - start_time)  # FPS
    
    # Display render time and FPS on the frame
    cv2.putText(frame, f"Render time: {render_time_ms:.2f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)
    
    # Display the frame
    cv2.imshow('YOLOv10 Prediction', frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
