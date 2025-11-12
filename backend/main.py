from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import json
import base64
from io import BytesIO
from grad_cam import GradCAM, apply_heatmap
import cv2
import os

# Global variables
model = None
class_labels = None
lesion_type_dict = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and labels
def load_model():
    global model, class_labels, lesion_type_dict
    
    try:
        # Load class labels
        with open('models/class_labels.json', 'r') as f:
            label_data = json.load(f)
            class_labels = label_data
            lesion_type_dict = label_data['label_names']
        
        # Load model
        num_classes = len(class_labels['idx_to_label'])
        
        # Use the same model architecture as in training
        model = models.mobilenet_v2(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        model.load_state_dict(torch.load('models/skin_lesion_model.pth', map_location=device))
        model.to(device)
        model.eval()
        
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

# Modern lifespan handler (replaces @app.on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Dermatology AI Lesion Analyzer API...")
    try:
        load_model()
        print("✅ Startup completed successfully!")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise
    yield
    # Shutdown would go here (if needed)

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Dermatology AI Lesion Analyzer",
    description="AI-powered skin lesion classification with Grad-CAM visualization",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Image preprocessing (should match training)
def preprocess_image(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

# Routes
@app.get("/")
async def root():
    return {
        "message": "Dermatology AI Lesion Analyzer API",
        "status": "operational",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.post("/predict")
async def predict_lesion(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Preprocess image
        input_tensor = preprocess_image(image_bytes).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_label = class_labels['idx_to_label'][str(predicted_idx.item())]
        diagnosis = lesion_type_dict[predicted_label]
        
        # Get confidence scores for all classes
        confidence_scores = {}
        for idx, score in enumerate(probabilities[0]):
            label = class_labels['idx_to_label'][str(idx)]
            confidence_scores[lesion_type_dict[label]] = round(score.item() * 100, 2)
        
        # Generate Grad-CAM heatmap
        grad_cam = GradCAM(model, 'features.18')
        heatmap = grad_cam.generate_heatmap(input_tensor)
        
        # Create overlay image
        original_image = Image.open(BytesIO(image_bytes)).convert('RGB')
        original_array = np.array(original_image)
        
        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap, (original_array.shape[1], original_array.shape[0]))
        overlay_image = apply_heatmap(original_array, heatmap_resized)
        
        # Convert to base64
        overlay_pil = Image.fromarray(overlay_image)
        buffered = BytesIO()
        overlay_pil.save(buffered, format="JPEG")
        heatmap_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "diagnosis": diagnosis,
            "confidence": round(confidence.item() * 100, 2),
            "confidence_scores": confidence_scores,
            "heatmap_image": heatmap_b64,
            "prediction_id": f"pred_{hash(image_bytes) % 10000:04d}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/classes")
async def get_classes():
    if class_labels is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": lesion_type_dict,
        "total_classes": len(lesion_type_dict)
    }

if __name__ == "__main__":
    import uvicorn
    # Remove reload=True for standalone execution
    uvicorn.run(app, host="0.0.0.0", port=8000)