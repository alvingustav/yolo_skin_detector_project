import os
import sys
import time
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import base64
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ultrayolo123'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
socketio = SocketIO(app)

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set default model path - make sure to place your model file in this directory
DEFAULT_MODEL_PATH = os.environ.get('MODEL_PATH', 'models/yolov8n.pt')

# Global variables
model = None
camera = None
is_running = False
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

def load_model(model_path=DEFAULT_MODEL_PATH):
    """Load the YOLO model"""
    global model
    try:
        # Make sure the models directory exists
        os.makedirs('models', exist_ok=True)
        
        # If model doesn't exist, download the default YOLOv8n model
        if not os.path.exists(model_path):
            print(f"Model {model_path} not found. Downloading default YOLOv8n model...")
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            model.save('models/yolov8n.pt')
        else:
            model = YOLO(model_path, task='detect')
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def detect_objects(frame, min_confidence=0.5):
    """Run object detection on a single frame"""
    global model
    
    if model is None:
        load_model()
    
    # Run inference
    results = model(frame, verbose=False)
    detections = results[0].boxes
    
    # Create a copy of the frame to draw on
    output_frame = frame.copy()
    
    # Object count for statistics
    object_counts = {}
    
    # Process detections
    for i in range(len(detections)):
        # Get bounding box coordinates
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        
        # Get class ID and name
        classidx = int(detections[i].cls.item())
        classname = model.names[classidx]
        
        # Get confidence
        conf = detections[i].conf.item()
        
        # Draw box if confidence is high enough
        if conf > min_confidence:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(output_frame, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Add label
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(output_frame, (xmin, label_ymin-labelSize[1]-10), 
                         (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(output_frame, label, (xmin, label_ymin-7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Update object count
            if classname in object_counts:
                object_counts[classname] += 1
            else:
                object_counts[classname] = 1
    
    return output_frame, object_counts

def process_webcam():
    """Generator function for webcam frames"""
    global camera, is_running
    
    if camera is None:
        try:
            camera = cv2.VideoCapture(0)
        except Exception as e:
            print(f"Error opening camera: {e}")
            yield None
            return
    
    fps_buffer = []
    start_time = time.time()
    
    while is_running:
        success, frame = camera.read()
