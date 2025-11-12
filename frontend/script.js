const BACKEND_URL = 'http://localhost:8000';
let selectedFile = null;
let cameraStream = null;

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const cameraBtn = document.getElementById('cameraBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const cameraModal = document.getElementById('cameraModal');
const cameraStreamEl = document.getElementById('cameraStream');
const captureBtn = document.getElementById('captureBtn');
const closeCameraBtn = document.getElementById('closeCameraBtn');

// 📂 File Upload Logic
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) handleFileSelect(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
});

// 📸 Camera Logic
cameraBtn.addEventListener('click', async () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
            cameraModal.style.display = 'flex';
            cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
            cameraStreamEl.srcObject = cameraStream;
        } catch (error) {
            alert('Unable to access camera. Please allow permission.');
            console.error(error);
        }
    } else {
        // fallback for mobile browsers
        fileInput.setAttribute('capture', 'environment');
        fileInput.click();
    }
});

captureBtn.addEventListener('click', () => {
    const canvas = document.createElement('canvas');
    canvas.width = cameraStreamEl.videoWidth;
    canvas.height = cameraStreamEl.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(cameraStreamEl, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
        selectedFile = new File([blob], 'captured-image.jpg', { type: 'image/jpeg' });
        handleFileSelect(selectedFile);
        closeCamera();
    }, 'image/jpeg');
});

closeCameraBtn.addEventListener('click', closeCamera);

function closeCamera() {
    cameraModal.style.display = 'none';
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
}

function handleFileSelect(file) {
    if (!file.type.match('image.*')) {
        alert('Please select an image file (JPG, PNG, etc.)');
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB');
        return;
    }

    selectedFile = file;
    analyzeBtn.disabled = false;

    const reader = new FileReader();
    reader.onload = (e) => {
        uploadArea.innerHTML = `
            <img src="${e.target.result}" style="max-width: 200px; max-height: 200px; border-radius: 10px;">
            <p>${file.name}</p>
            <p>Click "Analyze Image" to proceed</p>
        `;
    };
    reader.readAsDataURL(file);
}

// 🧠 AI Analysis
analyzeBtn.addEventListener('click', analyzeImage);

async function analyzeImage() {
    if (!selectedFile) return;

    loadingSection.style.display = 'block';
    resultsSection.style.display = 'none';
    analyzeBtn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch(`${BACKEND_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error(await response.text());
        const result = await response.json();
        displayResults(result);
    } catch (error) {
        console.error(error);
        alert('Analysis failed. Please try again.');
    } finally {
        loadingSection.style.display = 'none';
    }
}

function displayResults(result) {
    document.getElementById('diagnosisText').textContent = result.diagnosis;

    const confidenceElem = document.getElementById('confidenceText');
    confidenceElem.textContent = `${result.confidence}% confidence`;
    confidenceElem.className = 'confidence';
    confidenceElem.classList.add(
        result.confidence < 50 ? 'confidence-low' :
        result.confidence < 75 ? 'confidence-medium' : 'confidence-high'
    );

    const confidenceBars = document.getElementById('confidenceBars');
    confidenceBars.innerHTML = '';

    Object.entries(result.confidence_scores)
        .sort((a, b) => b[1] - a[1])
        .forEach(([disease, score]) => {
            const barContainer = document.createElement('div');
            barContainer.className = 'confidence-bar';
            barContainer.innerHTML = `
                <div class="confidence-label">
                    <span>${disease}</span><span>${score}%</span>
                </div>
                <div class="bar-container">
                    <div class="bar" style="width: ${score}%"></div>
                </div>
            `;
            confidenceBars.appendChild(barContainer);
        });

    const resultImage = document.getElementById('resultImage');
    resultImage.src = 'data:image/jpeg;base64,' + result.heatmap_image;
    resultImage.style.display = 'block';
    resultsSection.style.display = 'block';
}

function resetAnalysis() {
    selectedFile = null;
    analyzeBtn.disabled = true;
    resultsSection.style.display = 'none';
    loadingSection.style.display = 'none';
    uploadArea.innerHTML = `
        <div class="upload-icon">📁</div>
        <h3>Drag & Drop your image here</h3>
        <p>or click to browse files</p>
    `;
    fileInput.value = '';
    closeCamera();
}

// Health check
window.addEventListener('load', async () => {
    try {
        const res = await fetch(`${BACKEND_URL}/health`);
        console.log('Backend health:', await res.json());
    } catch (e) {
        console.warn('Backend not reachable:', e);
    }
});
