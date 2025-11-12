# Capstone-Project: AI-Powered Dermatology Assistant

This project explores how automation can enhance medical diagnostics by focusing on **early detection** and **severity assessment** of common skin conditions.  
Using image processing and machine learning, it serves as a **decision-support tool**, improving accuracy, accessibility, and reducing the burden on specialists — bridging healthcare gaps with **AI-driven dermatology support**.

---

## AI-Powered Dermatology Assistant

**Early detection made smarter.**  
This project offers a new perspective on how automation and AI can transform medical diagnostics.  
By combining advanced image processing and machine learning, the system identifies common skin conditions and evaluates their severity, empowering healthcare providers to deliver faster and more reliable decisions.

---

## Why It Matters

- **Global Challenge:** Millions suffer from skin diseases, but access to dermatology specialists is limited.  
- **Our Solution:** This assistant accelerates early detection, reducing delays in treatment.  
- **Supportive, Not Substitutive:** Designed as a **decision-support tool**, augmenting dermatologists rather than replacing them.  
- **Transparent AI:** Provides **confidence scores** and **Grad-CAM visual explanations** for better trust and interpretability.  
- **Accessible Anywhere:** Works via web and mobile, extending reach into underserved communities.

---

## How It Works

1. **Frontend (HTML, CSS, JS)** – Lets users upload or capture lesion images directly from their camera.  
2. **Backend (FastAPI)** – Receives the image, preprocesses it, runs the trained model, and returns predictions.  
3. **Model (PyTorch – MobileNetV2)** – Classifies the image into seven common lesion types (e.g., melanoma, basal cell carcinoma, etc.).  
4. **Grad-CAM Module** – Generates a heatmap highlighting key regions influencing the model’s decision.  
5. **Results Display** – Shows predicted class, confidence scores, and visual explanations within the browser.

---

##  Features

- Upload or capture lesion images directly from the browser  
- Real-time prediction using fine-tuned **MobileNetV2** model  
- **Grad-CAM heatmap generation** for visual explainability  
- Confidence scores for each predicted disease class  
- Secure **FastAPI backend** with SQLite logging  
- Responsive **frontend interface** with built-in medical disclaimer  
- Lightweight, GPU-compatible, and deployable to any cloud platform  

---

##  Project Structure



```
dermatology-ai-analyzer/
│
├── backend/
│   ├── main.py              # FastAPI application
│   ├── train_model.py       # Model training script
│   ├── grad_cam.py          # Grad-CAM implementation
│   ├── requirements.txt     # Python dependencies
│   └── models/              # Saved models and labels
│       ├── skin_lesion_model.pth
│       ├── class_labels.json
│       └── training_history.png
│
├── frontend/
│   ├── index.html           # Main web interface
│   ├── style.css            # Styling
│   └── script.js            # Frontend logic
│
└── data/                    # HAM10000 dataset (not included in repo)
    ├── HAM10000_metadata.csv
    └── HAM10000_images/
```


---

##  How to Run Locally

1. **Clone this repository**
   ```bash
   git clone https://github.com/Stu-Affan/Capstone-Project.git
   cd Capstone-Project/backend
2. **Create a virtual environment**
    python -m venv venv
    venv\Scripts\activate       # On Windows
    source venv/bin/activate    # On macOS/Linux
4. **Install dependencies**
     pip install -r requirements.txt
5. **Run the backend**
     uvicorn main:app --host 0.0.0.0 --port 8000 --reload
6. **Launch the frontend**
     Open frontend/index.html in your browser.
     Upload or capture a skin lesion image to see predictions and Grad-CAM visualization.
---
 **Model Performance**

The model was trained using transfer learning (MobileNetV2 pretrained on ImageNet) and fine-tuned on the HAM10000 dataset for 7-class skin lesion classification.

| Metric                  | Result |
| ----------------------- | ------ |
| **Validation Accuracy** | ~97.8% |
| **Macro F1-Score**      | ~0.96  |
| **Epochs Trained**      | 15     |
---

Training and Validation Curves:
<img width="1200" height="500" alt="training data" src="https://github.com/user-attachments/assets/1ee2a07e-2e87-485f-a95e-a9d35ba450fc" />
---

 **Tech Stack Highlights**

AI/ML: PyTorch, TensorFlow, OpenCV
Backend: FastAPI (Python)
Frontend: HTML, CSS, JavaScript
Visualization: Grad-CAM (Explainable AI)
Cloud Deployment: AWS / Google Cloud (optional)
---
 **Disclaimer**

This tool is intended for educational and research purposes only.
It is not a substitute for professional medical diagnosis or treatment.
Always consult a certified dermatologist for medical evaluation.
---

 **Acknowledgments**

Dataset: HAM10000, Skin Lesion Dataset

Frameworks: PyTorch, FastAPI, OpenCV

Inspiration: Explainable AI in healthcare and accessible medical diagnostics research
---

 **The Vision**

Our vision is to make dermatological care more inclusive, transparent, and accessible — bridging the gap between urban hospitals and rural communities to ensure better outcomes everywhere.
By merging AI, automation, and ethical design, this project takes a meaningful step toward equitable, technology-driven healthcare.
---

**Release History**

v1.0 (Current) – First stable release

Added trained model weights and label mappings

Integrated FastAPI backend with Grad-CAM explainability

Developed full frontend for image upload and visualization
Logged prediction data for audit and retraining


---

 ## **Contact**
    Developer: Stu-Affan
---

 Email: (optional – you can add your academic or professional mail)
 Project Type: Academic Capstone / AI Research Project
---
“Bridging healthcare gaps with AI-driven dermatology support.”
---


---




