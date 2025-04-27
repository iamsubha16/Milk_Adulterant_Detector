const video = document.getElementById('camera');
const canvas = document.getElementById('canvas');
const predictionText = document.getElementById('predictionText');
const captureBtn = document.querySelector('button[onclick="capture()"]');
const uploadInput = document.getElementById('uploadInput');
const imagePreview = document.getElementById('imagePreview');

let currentFacingMode = 'environment'; // Default to back camera
let currentStream = null;

// Start the camera with given facing mode
async function startCamera(facingMode) {
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
  }
  
  try {
    const constraints = {
      video: { 
        facingMode: { ideal: facingMode },
        width: { ideal: 640 },
        height: { ideal: 480 }
      }
    };
    
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    currentStream = stream;
    currentFacingMode = facingMode;
    showToast(`Camera started: ${facingMode === 'environment' ? 'back' : 'front'} camera`);
  } catch (err) {
    showToast('Camera access error: ' + err.message);
  }
}

// Initial camera load
document.addEventListener('DOMContentLoaded', function() {
  startCamera(currentFacingMode);

  if (uploadInput) {
    uploadInput.addEventListener('change', handleUpload);
  }
});

// Switch camera between front and back
function switchCamera() {
  const newFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
  startCamera(newFacingMode);
}

// Common function to send image data to Flask and handle UI
async function sendPrediction(base64Image) {
  // Update image preview
  imagePreview.src = base64Image;
  imagePreview.style.display = 'block';
  
  // Disable capture button and show loading
  if (captureBtn) captureBtn.disabled = true;
  predictionText.textContent = 'Loading...';
  
  // Reset result styling
  const resultDiv = document.getElementById('result');
  resultDiv.classList.remove('border-green-500', 'border-red-500');
  
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64Image })
    });
    
    const data = await res.json();
    
    if (data.error) {
      showToast('Prediction error');
    } else {
      predictionText.innerHTML = `Predicted Class: <strong>${data.predicted_class}</strong><br>Confidence: <strong>${data.confidence}%</strong>`;
      
      // Update styling based on result
      if (data.predicted_class.toLowerCase().includes('pure')) {
        resultDiv.classList.add('border-green-500');
        predictionText.className = 'text-lg font-semibold text-center text-green-600 dark:text-green-400';
      } else {
        resultDiv.classList.add('border-red-500');
        predictionText.className = 'text-lg font-semibold text-center text-red-600 dark:text-red-400';
      }
      
      showToast('Prediction complete');
    }
  } catch (err) {
    predictionText.textContent = `Error: ${err.message}`;
    showToast('Connection error');
  } finally {
    if (captureBtn) captureBtn.disabled = false;
  }
}

// Capture current frame and predict
function capture() {
  if (!currentStream) {
    showToast('Camera not available');
    return;
  }
  
  const context = canvas.getContext('2d');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const base64Image = canvas.toDataURL('image/jpeg');
  
  sendPrediction(base64Image);
}

// Handle file uploads and predict
function handleUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  
  if (!file.type.match('image.*')) {
    showToast('Please select an image file');
    return;
  }
  
  const reader = new FileReader();
  reader.onload = function(e) {
    const img = new Image();
    img.onload = function() {
      const context = canvas.getContext('2d');
      canvas.width = img.width;
      canvas.height = img.height;
      context.drawImage(img, 0, 0);
      const base64Image = canvas.toDataURL('image/jpeg');
      
      sendPrediction(base64Image);
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// Show toast notification
function showToast(message) {
  const toast = document.getElementById('toast');
  toast.textContent = message;
  toast.classList.remove('hidden');
  toast.classList.add('visible');
  
  setTimeout(() => {
    toast.classList.remove('visible');
    toast.classList.add('hidden');
  }, 3000);
}