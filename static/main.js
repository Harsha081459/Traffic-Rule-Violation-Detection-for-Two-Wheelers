const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');
const canvas = document.getElementById('result-canvas');
const ctx = canvas.getContext('2d');
const placeholder = document.getElementById('placeholder-text');
const loader = document.getElementById('loader');
const statsPanel = document.getElementById('stats-panel');
const statGrid = document.getElementById('stat-grid');

let currentController = null;

// Drag and drop event listeners
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

dropZone.addEventListener('drop', handleDrop, false);
browseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFiles, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles({ target: { files } });
}

function handleFiles(e) {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        processImage(file);
    } else {
        alert("Please upload a valid image file.");
    }
}

async function processImage(file) {
    // Show image on canvas immediately
    const img = new Image();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        canvas.style.display = 'block';
        placeholder.style.display = 'none';
        statsPanel.style.display = 'none';
        
        // Call API
        uploadToAPI(file, img);
    }
    img.src = URL.createObjectURL(file);
}

async function uploadToAPI(file, imgData) {
    if (currentController) {
        currentController.abort();
    }
    currentController = new AbortController();
    const signal = currentController.signal;

    loader.style.display = 'flex';
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
            signal: signal
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const data = await response.json();
        drawResults(data, imgData);
        updateStats(data);
    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('Previous request aborted for new upload.');
            return;
        }
        console.error("Error calling API:", error);
        alert("Failed to analyze image. Ensure the FastAPI server is running.");
    } finally {


#     currentController = new AbortController();
#     const signal = currentController.signal;
# 
#     loader.style.display = 'flex';
#     
#     const formData = new FormData();
#     formData.append('file', file);
#     
#     try {
#         const response = await fetch('/predict', {
#             method: 'POST',
#             body: formData,
#             signal: signal
#         });
#         
#         if (!response.ok) {
#             throw new Error(`Server error: ${response.status}`);
#         }
#         
#         const data = await response.json();
#         drawResults(data, imgData);
#         updateStats(data);
#     } catch (error) {
#         if (error.name === 'AbortError') {
#             console.log('Previous request aborted for new upload.');
#             return;
#         }
#         console.error("Error calling API:", error);
#         alert("Failed to analyze image. Ensure the FastAPI server is running.");
#     } finally {
