import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import sys

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
MODEL_PATH = "best_resnet.pth"  # Assumes script is run from resnet_project/ or file is in same dir

# Class Names (Must match training mapping)
# {0: 'backward', 1: 'book', 2: 'flat', 3: 'reverse'}
CLASS_NAMES = ['backward', 'book', 'flat', 'reverse']

def get_resnet_model(num_classes):
    """
    Re-create the ResNet18 model architecture to match training.
    """
    # Initialize model without pretrained weights first (weights will be loaded)
    # We can use weights=None or weights='DEFAULT' but we overwrite them anyway
    model = models.resnet18(weights=None) 
    
    # Modify FC layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        print("Please ensure you have trained the model and 'best_resnet.pth' exists.")
        sys.exit(1)
        
    model = get_resnet_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")
    return model

def get_transforms():
    """
    Standard ResNet validation transforms.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def main():
    # 1. Load Model
    model = load_model()
    transform = get_transforms()
    
    # 2. Open Webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Webcam Inference...")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # 3. Preprocess Frame
        # Convert BGR (OpenCV) to RGB (PIL/PyTorch)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Prepare input tensor
        input_tensor = transform(pil_image).unsqueeze(0) # Add batch dimension -> [1, 3, 224, 224]
        input_tensor = input_tensor.to(DEVICE)
        
        # 4. Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            # Apply Softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, preds = torch.max(probs, 1)
            
            class_idx = preds.item()
            conf_score = confidence.item()
            class_name = CLASS_NAMES[class_idx]

        # 5. Visualize
        # Draw text on the original BGR frame
        label_text = f"{class_name} ({conf_score:.2%})"
        
        # Color logic: Green if high confidence, Yellow if mid, Red if low
        if conf_score > 0.8:
            color = (0, 255, 0) # Green
        elif conf_score > 0.5:
            color = (0, 255, 255) # Cyan/Yellowish
        else:
            color = (0, 0, 255) # Red

        cv2.putText(frame, "ResNet Live Inference", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, label_text, (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Display Bar for confidence
        bar_width = int(conf_score * 200)
        cv2.rectangle(frame, (10, 85), (10 + bar_width, 105), color, -1)
        cv2.rectangle(frame, (10, 85), (210, 105), (255, 255, 255), 1)

        cv2.imshow('Book Pose Classification', frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

