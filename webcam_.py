from ultralytics import YOLOv10
import os
import cv2
import time

models = [
    'jameslahm/yolov10n',  # 0 
    'jameslahm/yolov10s',  # 1
    'jameslahm/yolov10m',  # 2
    'jameslahm/yolov10b',  # 3
    'jameslahm/yolov10l',  # 4
    'jameslahm/yolov10x',  # 5
]

model_choice = 0

model_name = models[model_choice]

try:
    model = YOLOv10.from_pretrained(model_name)
    print(f'Using model: {model_name}')
except Exception as e:
    print(f"Error loading model {model_name}: {e}")
    exit()


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

                # opcoes de display

confidence_threshold = 0.5  # Confianca minima para as predicoes
font_scale = 1  # Escala da fonte para exibir o texto
font_color = (0, 255, 0)  # Cor da fonte             (formato blue,gree,red) ; 0 a 255
box_color = (0, 255, 0)  # Cor da caixa delimitadora (formato blue,gree,red) ; 0 a 255
thickness = 1  # Espessura da linha da caixa delimitadora   



while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    start_time = time.time()  
    
    results = model.predict(source=frame, conf=confidence_threshold)
    
    
    for pred in results[0].boxes:
        x1, y1, x2, y2 = map(int, pred.xyxy[0])  
        confidence = pred.conf[0]  
        class_id = int(pred.cls[0])
        
        if confidence < confidence_threshold:
            continue  
        
        label = f"Class {class_id}: {confidence:.2f}"  
        
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
        
        
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)
    
    
    end_time = time.time()
    render_time_ms = (end_time - start_time) * 1000 # ms
    fps = 1 / (end_time - start_time)  
    
    
    cv2.putText(frame, f"Render time: {render_time_ms:.2f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)
    
    cv2.imshow('YOLOv10 Prediction', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
