/* =========================================================================
   Traffic Sentinel AI — main.js
   Phase 3: animated bounding boxes, toasts, inference badge, step tracker
   ========================================================================= */

'use strict';

// ── DOM refs ──────────────────────────────────────────────────────────────
const dropZone       = document.getElementById('drop-zone');
const fileInput      = document.getElementById('file-input');
const browseBtn      = document.getElementById('browse-btn');
const changeBtn      = document.getElementById('change-btn');
const analyseBtn     = document.getElementById('analyse-btn');
const dzIdle         = document.getElementById('dz-idle');
const dzPreview      = document.getElementById('dz-preview');
const previewThumb   = document.getElementById('preview-thumb');
const previewName    = document.getElementById('preview-name');
const previewSize    = document.getElementById('preview-size');
const stepsPanel     = document.getElementById('steps-panel');
const canvasWrap     = document.getElementById('canvas-wrap');
const placeholder    = document.getElementById('canvas-placeholder');
const skeleton       = document.getElementById('canvas-skeleton');
const canvas         = document.getElementById('result-canvas');
const ctx            = canvas.getContext('2d');
const inferenceBadge = document.getElementById('inference-badge');
const inferenceMs    = document.getElementById('inference-ms');
const resultsArea    = document.getElementById('results-area');
const cardsGrid      = document.getElementById('cards-grid');
const summaryChips   = document.getElementById('summary-chips');
const noDetections   = document.getElementById('no-detections');
const toastShelf     = document.getElementById('toast-shelf');

// ── State ─────────────────────────────────────────────────────────────────
let selectedFile = null;
let currentAbort = null;
let loadedImage  = null;   // HTMLImageElement currently on canvas

// ── Allowed MIME types (client-side pre-filter) ───────────────────────────
const ALLOWED_TYPES = new Set(['image/jpeg','image/png','image/webp','image/bmp']);
const MAX_MB        = 15;

// ── Colour palette for detection types ───────────────────────────────────
const COLOURS = {
  violation: '#ff3366',
  safe:      '#00ffcc',
  rider:     '#facc15',
  helmet:    '#00ffcc',
  no_helmet: '#ff3366',
  plate:     '#a78bfa',
};


// =========================================================================
// Toast system
// =========================================================================

/**
 * Show a toast notification.
 * @param {string} message  Text to display.
 * @param {'info'|'success'|'error'|'warn'} type  Visual variant.
 * @param {number} [duration=4000]  Auto-dismiss delay in ms.
 */
function toast(message, type = 'info', duration = 4000) {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.setAttribute('role', 'alert');

  const icons = { info: 'ℹ', success: '✓', error: '✕', warn: '⚠' };
  el.innerHTML = `
    <span class="toast-icon">${icons[type] || icons.info}</span>
    <span class="toast-msg">${message}</span>
    <button class="toast-close" aria-label="Dismiss">✕</button>
  `;

  el.querySelector('.toast-close').addEventListener('click', () => dismissToast(el));
  toastShelf.appendChild(el);

  // Trigger enter animation on next frame
  requestAnimationFrame(() => el.classList.add('toast-enter'));

  if (duration > 0) {
    setTimeout(() => dismissToast(el), duration);
  }
  return el;
}

function dismissToast(el) {
  el.classList.remove('toast-enter');
  el.classList.add('toast-leave');
  el.addEventListener('transitionend', () => el.remove(), { once: true });
}


// =========================================================================
// Step progress tracker
// =========================================================================

const STEPS = ['step-upload', 'step-detect', 'step-helmet', 'step-plate'];
let _stepIndex = -1;

function resetSteps() {
  _stepIndex = -1;
  STEPS.forEach(id => {
    const el = document.getElementById(id);
    el.className = 'step';
  });
  stepsPanel.classList.remove('hidden');
}

function advanceStep() {
  if (_stepIndex >= 0 && _stepIndex < STEPS.length) {
    document.getElementById(STEPS[_stepIndex]).className = 'step step-done';
  }
  _stepIndex++;
  if (_stepIndex < STEPS.length) {
    document.getElementById(STEPS[_stepIndex]).className = 'step step-active';
  }
}

function completeAllSteps() {
  STEPS.forEach(id => {
    document.getElementById(id).className = 'step step-done';
  });
}

function hideSteps() {
  stepsPanel.classList.add('hidden');
}


// =========================================================================
// File selection & validation
// =========================================================================

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(2)} MB`;
}

function selectFile(file) {
  if (!file) return;

  // Client-side MIME check
  if (!ALLOWED_TYPES.has(file.type)) {
    toast(`Invalid file type: <strong>${file.type || 'unknown'}</strong>. Please upload a JPEG, PNG, WebP, or BMP image.`, 'error');
    return;
  }

  // Client-side size check
  if (file.size > MAX_MB * 1024 * 1024) {
    toast(`File too large (${formatBytes(file.size)}). Maximum is ${MAX_MB} MB.`, 'error');
    return;
  }

  selectedFile = file;

  // Show preview
  const url = URL.createObjectURL(file);
  previewThumb.src   = url;
  previewName.textContent = file.name;
  previewSize.textContent = formatBytes(file.size);
  dzIdle.classList.add('hidden');
  dzPreview.classList.remove('hidden');

  analyseBtn.disabled = false;

  // Reset result area
  resetResultArea();
}

function resetResultArea() {
  [resultsArea, noDetections, inferenceBadge].forEach(el => el.classList.add('hidden'));
  canvas.style.display = 'none';
  placeholder.classList.remove('hidden');
  skeleton.classList.add('hidden');
  cardsGrid.innerHTML    = '';
  summaryChips.innerHTML = '';
  loadedImage = null;
}


// =========================================================================
// Event wiring
// =========================================================================

// Drag & drop
['dragenter','dragover','dragleave','drop'].forEach(ev => {
  dropZone.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); });
});
dropZone.addEventListener('dragenter', () => dropZone.classList.add('dz-hover'));
dropZone.addEventListener('dragover',  () => dropZone.classList.add('dz-hover'));
['dragleave','drop'].forEach(ev => {
  dropZone.addEventListener(ev, () => dropZone.classList.remove('dz-hover'));
});
dropZone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (file) selectFile(file);
});

// Click to open picker
dropZone.addEventListener('click', e => {
  if (e.target === dropZone || e.target.closest('#dz-idle')) fileInput.click();
});
browseBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
changeBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) selectFile(fileInput.files[0]);
  fileInput.value = '';   // allow re-selecting same file
});

// Keyboard accessibility for drop zone
dropZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});

// Analyse button
analyseBtn.addEventListener('click', () => {
  if (selectedFile) runAnalysis(selectedFile);
});


// =========================================================================
// Main analysis pipeline
// =========================================================================

async function runAnalysis(file) {
  // Abort any in-flight request
  if (currentAbort) { currentAbort.abort(); currentAbort = null; }
  currentAbort = new AbortController();
  const signal = currentAbort.signal;

  analyseBtn.disabled = true;

  // ── Show skeleton + steps ──────────────────────────────────────
  placeholder.classList.add('hidden');
  canvas.style.display = 'none';
  skeleton.classList.remove('hidden');
  inferenceBadge.classList.add('hidden');
  resultsArea.classList.add('hidden');
  noDetections.classList.add('hidden');

  resetSteps();
  advanceStep();   // step 0: uploading

  // ── Draw the image immediately for visual feedback ─────────────
  const img = await loadImage(URL.createObjectURL(file));
  loadedImage = img;

  // ── Build form data and upload ─────────────────────────────────
  const formData = new FormData();
  formData.append('file', file);

  let data;
  try {
    advanceStep();   // step 1: detecting vehicles
    const t0 = performance.now();

    const response = await fetch('/predict', {
      method: 'POST',
      body:   formData,
      signal,
    });

    const fetchMs = performance.now() - t0;

    if (!response.ok) {
      const body = await response.json().catch(() => ({ detail: response.statusText }));
      const status = response.status;
      if (status === 429) {
        toast('Rate limit reached. Please wait a moment before trying again.', 'warn', 6000);
      } else if (status === 413) {
        toast(body.detail || 'File too large.', 'error');
      } else if (status === 415) {
        toast(body.detail || 'Unsupported file type.', 'error');
      } else {
        toast(`Server error (${status}): ${body.detail || 'Unknown error.'}`, 'error');
      }
      return;
    }

    advanceStep();   // step 2: analysing helmets
    data = await response.json();
    advanceStep();   // step 3: reading plates
    completeAllSteps();

    // Use server-reported inference time if available, else fetch RTT
    const reportedMs = data.inference_time_sec != null
      ? Math.round(data.inference_time_sec * 1000)
      : Math.round(fetchMs);

    // ── Paint image then animate bboxes ───────────────────────────
    skeleton.classList.add('hidden');
    paintCanvas(img);

    await animateDetections(data, img);

    // ── Show inference badge ──────────────────────────────────────
    inferenceMs.textContent = `${reportedMs} ms`;
    inferenceBadge.classList.remove('hidden');

    // ── Populate result cards ─────────────────────────────────────
    renderResults(data);

    toast(`Analysis complete — ${reportedMs} ms`, 'success', 3000);

  } catch (err) {
    if (err.name === 'AbortError') return;
    console.error(err);
    toast('Could not reach the API. Is the FastAPI server running?', 'error', 6000);
  } finally {
    skeleton.classList.add('hidden');
    hideSteps();
    analyseBtn.disabled = false;
  }
}


// =========================================================================
// Canvas helpers
// =========================================================================

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload  = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function paintCanvas(img) {
  // Size canvas to match the natural image dimensions
  canvas.width  = img.naturalWidth  || img.width;
  canvas.height = img.naturalHeight || img.height;
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  canvas.style.display = 'block';
}


// =========================================================================
// Animated bounding box drawing
// =========================================================================

/**
 * Animate a rectangle being drawn by progressively un-dashing its stroke.
 * @param {number[]} box        [x1, y1, x2, y2]
 * @param {string}   colour     CSS colour string
 * @param {number}   lineWidth
 * @param {number}   duration   Animation duration in ms
 * @returns {Promise<void>}     Resolves when animation finishes
 */
function animateBox(box, colour, lineWidth, duration = 300) {
  return new Promise(resolve => {
    const [x1, y1, x2, y2] = box;
    const w = x2 - x1, h = y2 - y1;
    const perimeter = 2 * (w + h);
    const start = performance.now();

    function frame(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased    = 1 - Math.pow(1 - progress, 3);   // ease-out cubic

      // Glow shadow
      ctx.save();
      ctx.shadowColor = colour;
      ctx.shadowBlur  = 10;
      ctx.strokeStyle = colour;
      ctx.lineWidth   = lineWidth;

      // Dash trick: visible segment grows from 0 → perimeter
      const visible = perimeter * eased;
      ctx.setLineDash([visible, perimeter]);
      ctx.lineDashOffset = 0;

      ctx.beginPath();
      ctx.rect(x1, y1, w, h);
      ctx.stroke();
      ctx.restore();

      if (progress < 1) {
        requestAnimationFrame(frame);
      } else {
        // Final clean stroke (no dash)
        ctx.save();
        ctx.shadowColor = colour;
        ctx.shadowBlur  = 8;
        ctx.strokeStyle = colour;
        ctx.lineWidth   = lineWidth;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.rect(x1, y1, w, h);
        ctx.stroke();
        ctx.restore();
        resolve();
      }
    }
    requestAnimationFrame(frame);
  });
}

/**
 * Draw a label chip above the box.
 */
function drawLabel(x, y, text, colour, fontSize = 13) {
  ctx.save();
  ctx.font = `600 ${fontSize}px Inter, sans-serif`;
  const tw = ctx.measureText(text).width;
  const ph = fontSize + 8, pw = tw + 12;

  // Pill background
  const rx = 4;
  ctx.fillStyle = colour;
  ctx.shadowColor = colour;
  ctx.shadowBlur  = 8;
  ctx.beginPath();
  ctx.roundRect(x, y - ph, pw, ph, rx);
  ctx.fill();

  // Text
  ctx.shadowBlur  = 0;
  ctx.fillStyle   = '#000';
  ctx.fillText(text, x + 6, y - 6);
  ctx.restore();
}

/**
 * Animate all detections in sequence: bike boxes → rider → helmet → plate.
 */
async function animateDetections(data, img) {
  if (!data.debug) return;

  // Ensure canvas has the image
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  for (let i = 0; i < data.debug.length; i++) {
    const item = data.debug[i];
    const bikeColour = item.is_violation ? COLOURS.violation : COLOURS.safe;

    // Bike bounding box
    await animateBox(item.bike_bbox, bikeColour, 3, 350);
    const label = `Bike ${i + 1} · ${item.num_riders} rider${item.num_riders !== 1 ? 's' : ''} · ${item.helmet_violations} violation${item.helmet_violations !== 1 ? 's' : ''}`;
    drawLabel(item.bike_bbox[0], item.bike_bbox[1], label, bikeColour, 12);

    // Rider boxes (staggered, shorter animation)
    for (const box of item.rider_bboxes) {
      await animateBox(box, COLOURS.rider, 2, 200);
      drawLabel(box[0], box[1], 'Rider', COLOURS.rider, 11);
    }

    // Helmet boxes
    for (const box of item.helmet_bboxes) {
      await animateBox(box, COLOURS.helmet, 2, 200);
      drawLabel(box[0], box[1], 'Helmet ✓', COLOURS.helmet, 11);
    }

    // No-helmet boxes
    for (const box of item.no_helmet_bboxes) {
      await animateBox(box, COLOURS.no_helmet, 2, 200);
      drawLabel(box[0], box[1], 'No Helmet !', COLOURS.no_helmet, 11);
    }
  }
}


// =========================================================================
// Result cards
// =========================================================================

function renderResults(data) {
  cardsGrid.innerHTML    = '';
  summaryChips.innerHTML = '';

  const debug = data.debug || [];

  if (debug.length === 0) {
    noDetections.classList.remove('hidden');
    return;
  }

  // Summary chips
  const violations = debug.filter(d => d.is_violation).length;
  const safe       = debug.length - violations;

  if (violations > 0) {
    summaryChips.innerHTML += `<span class="chip chip-violation">${violations} violation${violations > 1 ? 's' : ''}</span>`;
  }
  if (safe > 0) {
    summaryChips.innerHTML += `<span class="chip chip-safe">${safe} compliant</span>`;
  }
  summaryChips.innerHTML += `<span class="chip chip-neutral">${debug.length} bike${debug.length > 1 ? 's' : ''} detected</span>`;

  // Individual cards (animate in)
  debug.forEach((item, idx) => {
    const card = buildCard(item, idx);
    cardsGrid.appendChild(card);
    setTimeout(() => card.classList.add('card-visible'), idx * 80);
  });

  resultsArea.classList.remove('hidden');
}

function buildCard(item, idx) {
  const isV    = item.is_violation;
  const card   = document.createElement('div');
  card.className = `det-card ${isV ? 'det-violation' : 'det-safe'}`;

  const statusIcon  = isV ? '⚠' : '✓';
  const statusLabel = isV ? 'Violation' : 'Compliant';

  let plateHtml = '';
  if (item.license_plate) {
    plateHtml = `
      <div class="plate-wrap">
        <span class="plate-label">Plate</span>
        <span class="plate-value">${item.license_plate}</span>
      </div>`;
  } else if (isV) {
    plateHtml = `<div class="plate-unreadable">Plate not readable</div>`;
  }

  card.innerHTML = `
    <div class="card-header">
      <span class="card-title">Bike ${idx + 1}</span>
      <span class="card-status ${isV ? 'status-bad' : 'status-ok'}">
        ${statusIcon} ${statusLabel}
      </span>
    </div>
    <div class="card-stats">
      <div class="card-stat">
        <span class="cs-value">${item.num_riders}</span>
        <span class="cs-label">Riders</span>
      </div>
      <div class="card-stat">
        <span class="cs-value ${item.helmet_violations > 0 ? 'stat-bad' : ''}">${item.helmet_violations}</span>
        <span class="cs-label">No Helmet</span>
      </div>
      <div class="card-stat">
        <span class="cs-value">${item.helmet_bboxes.length}</span>
        <span class="cs-label">Helmeted</span>
      </div>
    </div>
    ${plateHtml}
  `;
  return card;
}
