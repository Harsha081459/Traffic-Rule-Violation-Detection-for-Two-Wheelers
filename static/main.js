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


# }
# 
# ['dragenter', 'dragover'].forEach(eventName => {
#     dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
# });
# 
# ['dragleave', 'drop'].forEach(eventName => {
#     dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
# });
# 
# dropZone.addEventListener('drop', handleDrop, false);
# browseBtn.addEventListener('click', () => fileInput.click());
# fileInput.addEventListener('change', handleFiles, false);
# 
# function handleDrop(e) {
#     const dt = e.dataTransfer;
#     const files = dt.files;
#     handleFiles({ target: { files } });
# }
# 
# function handleFiles(e) {
#     const file = e.target.files[0];
#     if (file && file.type.startsWith('image/')) {
#         processImage(file);
#     } else {
#         alert("Please upload a valid image file.");
#     }
# }
# 
# async function processImage(file) {
