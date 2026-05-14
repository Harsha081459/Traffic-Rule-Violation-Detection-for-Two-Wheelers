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
        // Only hide the loader if this specific request wasn't aborted
        if (currentController && currentController.signal === signal) {
            loader.style.display = 'none';
        }
    }
}

function drawResults(data, img) {
    // Redraw image to clear any previous drawings if re-called
    ctx.drawImage(img, 0, 0);
    
    if (!data.debug) return;

    data.debug.forEach((item, index) => {
        const isViolation = item.is_violation;
        const bikeColor = isViolation ? '#ff3366' : '#00ffcc';
        
        // Draw bike box
        drawBox(item.bike_bbox, bikeColor, 4);
        
        // Draw label
        const labelText = `Bike ${index + 1} | Riders: ${item.num_riders} | No Helmet: ${item.helmet_violations}`;
        drawLabel(item.bike_bbox[0], item.bike_bbox[1], labelText, bikeColor);
        
        // Draw rider boxes
        item.rider_bboxes.forEach(box => {
            drawBox(box, '#facc15', 2);
            drawLabel(box[0], box[1], 'Rider', '#facc15', 14);
        });
        
        // Draw helmet boxes
        item.helmet_bboxes.forEach(box => {
            drawBox(box, '#00ffcc', 2);
            drawLabel(box[0], box[1], 'Helmet', '#00ffcc', 14);
        });
        
        // Draw no_helmet boxes
        item.no_helmet_bboxes.forEach(box => {
            drawBox(box, '#ff3366', 2);
            drawLabel(box[0], box[1], 'No Helmet', '#ff3366', 14);
        });
    });
}

function drawBox(box, color, lineWidth) {
    if (!box || box.length !== 4) return;
    const [x1, y1, x2, y2] = box;
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();


#         // Draw label
#         const labelText = `Bike ${index + 1} | Riders: ${item.num_riders} | No Helmet: ${item.helmet_violations}`;
#         drawLabel(item.bike_bbox[0], item.bike_bbox[1], labelText, bikeColor);
#         
#         // Draw rider boxes
#         item.rider_bboxes.forEach(box => {
#             drawBox(box, '#facc15', 2);
#             drawLabel(box[0], box[1], 'Rider', '#facc15', 14);
#         });
#         
#         // Draw helmet boxes
#         item.helmet_bboxes.forEach(box => {
#             drawBox(box, '#00ffcc', 2);
#             drawLabel(box[0], box[1], 'Helmet', '#00ffcc', 14);
#         });
#         
#         // Draw no_helmet boxes
#         item.no_helmet_bboxes.forEach(box => {
#             drawBox(box, '#ff3366', 2);
#             drawLabel(box[0], box[1], 'No Helmet', '#ff3366', 14);
#         });
#     });
# }
# 
# function drawBox(box, color, lineWidth) {
#     if (!box || box.length !== 4) return;
#     const [x1, y1, x2, y2] = box;
#     ctx.strokeStyle = color;
#     ctx.lineWidth = lineWidth;
#     ctx.beginPath();
