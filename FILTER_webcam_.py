import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLOv10
import os
import cv2
import time

class YOLOv10ObjectDetector:
    def __init__(self, master):
        self.master = master
        master.title("YOLOv10 Object Detection")
        master.geometry("400x600")

        # Model selection
        self.models = [
            'jameslahm/yolov10n',  # 0
            'jameslahm/yolov10s',  # 1
            'jameslahm/yolov10m',  # 2
            'jameslahm/yolov10b',  # 3
            'jameslahm/yolov10l',  # 4
            'jameslahm/yolov10x',  # 5
        ]

        # Model selection dropdown
        tk.Label(master, text="Select Model:").pack(pady=5)
        self.model_var = tk.StringVar(value=self.models[0])
        self.model_dropdown = tk.OptionMenu(master, self.model_var, *self.models)
        self.model_dropdown.pack(pady=5)

        # Model load button
        self.load_model_btn = tk.Button(master, text="Load Model", command=self.load_model)
        self.load_model_btn.pack(pady=5)

        # Confidence threshold
        tk.Label(master, text="Confidence Threshold:").pack(pady=5)
        self.confidence_var = tk.DoubleVar(value=0.5)
        self.confidence_slider = tk.Scale(
            master, 
            from_=0.0, 
            to=1.0, 
            resolution=0.1, 
            orient=tk.HORIZONTAL, 
            variable=self.confidence_var
        )
        self.confidence_slider.pack(pady=5)

        # Input source buttons
        tk.Label(master, text="Select Input Source:").pack(pady=5)
        
        webcam_btn = tk.Button(master, text="Open Webcam", command=self.open_webcam)
        webcam_btn.pack(pady=5)
        
        video_btn = tk.Button(master, text="Open Video File", command=self.open_video)
        video_btn.pack(pady=5)
        
        image_btn = tk.Button(master, text="Open Image", command=self.open_image)
        image_btn.pack(pady=5)

        # Status label
        self.status_var = tk.StringVar(value="Ready to detect")
        status_label = tk.Label(master, textvariable=self.status_var)
        status_label.pack(pady=10)

        # Initialize model
        self.model = None

    def load_model(self):
        try:
            model_name = self.model_var.get()
            self.model = YOLOv10.from_pretrained(model_name)
            self.status_var.set(f"Model loaded: {model_name}")
        except Exception as e:
            messagebox.showerror("Model Loading Error", str(e))
            self.status_var.set("Failed to load model")

    def detect_and_display(self, source):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded")
            return

        confidence_threshold = self.confidence_var.get()

        # Display settings
        font_scale = 1
        font_color = (0, 255, 0)
        box_color = (0, 255, 0)
        thickness = 1

        # Perform detection
        try:
            start_time = time.time()
            results = self.model.predict(source=source, conf=confidence_threshold)

            # Create a copy of the source for drawing
            frame = source.copy() if isinstance(source, type(source)) else source

            for pred in results[0].boxes:
                x1, y1, x2, y2 = map(int, pred.xyxy[0])
                confidence = pred.conf[0]
                class_id = int(pred.cls[0])

                if confidence < confidence_threshold:
                    continue

                label = f"Class {class_id}: {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)

            end_time = time.time()
            render_time_ms = (end_time - start_time) * 1000

            # Add performance metrics
            cv2.putText(frame, f"Render time: {render_time_ms:.2f} ms", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, font_color, thickness)

            return frame

        except Exception as e:
            messagebox.showerror("Detection Error", str(e))
            return None

    def open_webcam(self):
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection
            detected_frame = self.detect_and_display(frame)
            
            if detected_frame is not None:
                cv2.imshow('YOLOv10 Webcam Detection', detected_frame)

            # Break loop if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 27:  # ESC key
                break

        cap.release()
        cv2.destroyAllWindows()

    def open_video(self):
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first")
            return

        file_path = filedialog.askopenfilename(
            title="Select Video File", 
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if not file_path:
            return

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video file")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                # Restart the video when it ends
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Perform detection
            detected_frame = self.detect_and_display(frame)
            
            if detected_frame is not None:
                cv2.imshow('YOLOv10 Video Detection', detected_frame)

            # Break loop if 'q' or ESC is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 27:  # ESC key
                break

        cap.release()
        cv2.destroyAllWindows()

    def open_image(self):
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first")
            return

        file_path = filedialog.askopenfilename(
            title="Select Image File", 
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if not file_path:
            return

        while True:
            # Read the image
            frame = cv2.imread(file_path)
            if frame is None:
                messagebox.showerror("Error", "Could not read image file")
                break

            # Perform detection
            detected_frame = self.detect_and_display(frame)
            
            if detected_frame is not None:
                cv2.imshow('YOLOv10 Image Detection', detected_frame)
            
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('n'):  # 'n' for next image
                break

        cv2.destroyAllWindows()

def main():
    root = tk.Tk()
    app = YOLOv10ObjectDetector(root)
    root.mainloop()

if __name__ == "__main__":
    main()