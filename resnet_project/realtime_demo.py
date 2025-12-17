import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import numpy as np
import os
import sys
import time

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLS_CLASSES = 4 # Classification: 4
NUM_DET_CLASSES = 5 # Detection: 4 + 1 background

# Paths
CLS_MODEL_PATH = "best_resnet.pth"
DET_MODEL_PATH = "fasterrcnn_resnet50.pth"

# Class Names
# Classification: 0-3
CLS_NAMES = ['backward', 'book', 'flat', 'reverse']
# Detection: 0 is background, 1-4 are classes
DET_NAMES = {0: 'background', 1: 'backward', 2: 'book', 3: 'flat', 4: 'reverse'}

# --- Model Loaders ---
def get_classification_model():
    print("Initializing ResNet18 (Classification)...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLS_CLASSES)
    return model

def get_detection_model():
    print("Initializing Faster R-CNN (Detection)...")
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_DET_CLASSES)
    return model

def load_best_model():
    """
    Checks for available models. Prioritizes Detection.
    Returns: (model, mode)
    mode: 'detection' or 'classification'
    """
    # Check for Detection Model first (Improvement)
    if os.path.exists(DET_MODEL_PATH):
        print(f"Found Detection Model: {DET_MODEL_PATH}")
        try:
            model = get_detection_model()
            model.load_state_dict(torch.load(DET_MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            return model, 'detection'
        except Exception as e:
            print(f"Failed to load detection model: {e}")
            print("Falling back to classification...")

    # Check for Classification Model
    if os.path.exists(CLS_MODEL_PATH):
        print(f"Found Classification Model: {CLS_MODEL_PATH}")
        try:
            model = get_classification_model()
            model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            return model, 'classification'
        except Exception as e:
            print(f"Failed to load classification model: {e}")
    
    print("Error: No trained model files found (best_resnet.pth or fasterrcnn_resnet50.pth).")
    return None, None

# --- Transforms ---
def get_cls_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_det_transform():
    return transforms.Compose([
        transforms.ToTensor() # Faster R-CNN expects 0-1 tensor
    ])

# --- Main Logic ---
def main():
    # 1. Load Model
    model, mode = load_best_model()
    if model is None:
        return

    print(f"Starting Real-time Demo in [{mode.upper()}] mode.")
    print("Press 'q' to quit.")

    # 2. Setup Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Transforms
    cls_transform = get_cls_transform()
    det_transform = get_det_transform()

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Preprocess
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # 3. Inference & Draw
        if mode == 'detection':
            input_tensor = det_transform(pil_image).to(DEVICE)
            with torch.no_grad():
                # Input must be a list of tensors
                predictions = model([input_tensor])[0]

            # Draw Boxes
            boxes = predictions['boxes'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()

            for box, label, score in zip(boxes, labels, scores):
                if score > 0.7: # Threshold
                    x1, y1, x2, y2 = box.astype(int)
                    label_name = DET_NAMES.get(label, 'Unknown')
                    
                    # Draw
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{label_name} {score:.2f}"
                    cv2.putText(frame, text, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(frame, "Mode: Object Detection (Faster R-CNN)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        elif mode == 'classification':
            # Classification Logic (Crop-less, full frame - naive)
            input_tensor = cls_transform(pil_image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, preds = torch.max(probs, 1)
                
            class_idx = preds.item()
            conf_score = conf.item()
            class_name = CLS_NAMES[class_idx]
            
            # Draw
            text = f"{class_name} ({conf_score:.1%})"
            color = (0, 255, 0) if conf_score > 0.8 else (0, 0, 255)
            
            cv2.putText(frame, "Mode: Classification (ResNet18)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, text, (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Show
        cv2.imshow('ResNet AI Demo', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
