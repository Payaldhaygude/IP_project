/* ══════════════════════════════════════
   STATE
══════════════════════════════════════ */
var selectedImageFile = null;
var selectedVideoFile = null;

/* ══════════════════════════════════════
   SERVICE CARD SELECTION
══════════════════════════════════════ */
function selectService(name) {
  document.querySelectorAll('.service-card').forEach(function(c) {
    c.classList.remove('service-card--active');
  });
  document.getElementById('card-' + name).classList.add('service-card--active');

  document.querySelectorAll('.tab-panel').forEach(function(p) {
    p.classList.remove('tab-panel--active');
  });
  document.getElementById('tab-' + name).classList.add('tab-panel--active');

  var titles = {
    image:  '📷 Image Detection',
    video:  '🎬 Video Detection',
    manual: '⌨️ Manual Plate Lookup'
  };
  document.getElementById('panelTitle').textContent = titles[name];
  document.getElementById('mainPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ══════════════════════════════════════
   IMAGE TAB
══════════════════════════════════════ */
function onImageSelect(e) {
  var file = e.target.files[0];
  if (!file) return;
  selectedImageFile = file;
  document.getElementById('imagePreview').src = URL.createObjectURL(file);
  document.getElementById('imagePreviewWrap').classList.add('preview-wrap--visible');
  document.getElementById('btnDetectImage').disabled = false;
  document.getElementById('imageResult').innerHTML = '';
}

function clearImage() {
  selectedImageFile = null;
  document.getElementById('imageFile').value = '';
  document.getElementById('imagePreviewWrap').classList.remove('preview-wrap--visible');
  document.getElementById('imageResult').innerHTML = '';
  document.getElementById('btnDetectImage').disabled = true;
}

function detectImage() {
  if (!selectedImageFile) return;
  var btn = document.getElementById('btnDetectImage');
  btn.disabled = true;
  document.getElementById('imageLoader').classList.add('loader--visible');
  document.getElementById('imageResult').innerHTML = '';

  var fd = new FormData();
  fd.append('file', selectedImageFile);

  fetch('/detect/image', { method: 'POST', body: fd })
    .then(function(res) { return res.json(); })
    .then(function(data) {
      document.getElementById('imageResult').innerHTML = renderImageResult(data);
    })
    .catch(function(err) {
      document.getElementById('imageResult').innerHTML = makeAlert('error', '❌ ' + err.message);
    })
    .finally(function() {
      document.getElementById('imageLoader').classList.remove('loader--visible');
      btn.disabled = false;
    });
}

function renderImageResult(data) {
  if (!data.success && data.error)
    return makeAlert('error', '❌ ' + data.error);

  var html = '';

  /* ── Result Tabs ── */
  html += '<div class="result-tabs">';
  html += '<button class="result-tab result-tab--active" onclick="switchResultTab(\'detection\')">🚗 Detection Result</button>';
  html += '<button class="result-tab" onclick="switchResultTab(\'pipeline\')">🔬 Processing Pipeline</button>';
  html += '<button class="result-tab" onclick="switchResultTab(\'algorithms\')">📊 Algorithm Info</button>';
  html += '</div>';

  /* ── TAB 1: Detection Result ── */
  html += '<div class="result-tab-panel result-tab-panel--active" id="rtab-detection">';

  if (!data.plates_found || data.plates_found.length === 0) {
    html += makeAlert('warning', '⚠️ No number plates detected. Try a clearer photo with the plate fully visible.');
  } else {
    var best = data.plates_found.reduce(function(a, b) {
      return scorePlate(b.ocr_text) > scorePlate(a.ocr_text) ? b : a;
    });
    html += makeAlert('success', '✅ Detected ' + data.plates_found.length + ' plate(s) in ' + data.image_size + ' image');
    html += '<div class="result-section"><div class="result-grid">';
    if (data.annotated_image_url) {
      html += '<div class="result-card">'
            + '<div class="result-card__header">📸 Annotated Image</div>'
            + '<div class="result-card__body">'
            + '<img class="annotated-img" src="' + data.annotated_image_url + '" />'
            + '</div></div>';
    }
    html += renderPlateCard(best.ocr_text, best.plate_info, best.detection_method);
    html += '</div></div>';
  }
  html += '</div>';

  /* ── TAB 2: Processing Pipeline ── */
  html += '<div class="result-tab-panel" id="rtab-pipeline">';
  html += '<div class="pipeline-header">';
  html += '<h3 class="pipeline-title">🔬 Image Processing Pipeline — All 17 Algorithms</h3>';
  html += '<p class="pipeline-subtitle">Each step shows how the image is transformed through different algorithms before OCR.</p>';
  html += '</div>';

  if (data.pipeline_steps && data.pipeline_steps.length > 0) {
    html += '<div class="pipeline-grid">';
    data.pipeline_steps.forEach(function(step, index) {
      html += '<div class="pipeline-card">';
      html += '<div class="pipeline-card__header">';
      html += '<span class="pipeline-step-num">' + (index + 1) + '</span>';
      html += '<span class="pipeline-step-title">' + step.title + '</span>';
      html += '</div>';
      html += '<div class="pipeline-card__img-wrap">';
      html += '<img class="pipeline-img" src="' + step.url + '" alt="' + step.title + '" />';
      html += '</div>';
      html += '<div class="pipeline-card__desc">' + step.description + '</div>';
      html += '</div>';
    });
    html += '</div>';
  } else {
    html += makeAlert('warning', '⚠️ Pipeline images not available.');
  }
  html += '</div>';

  /* ── TAB 3: Algorithm Info ── */
  html += '<div class="result-tab-panel" id="rtab-algorithms">';
  html += '<div class="pipeline-header">';
  html += '<h3 class="pipeline-title">📊 Algorithms Used in This Project</h3>';
  html += '<p class="pipeline-subtitle">Complete list of image processing algorithms implemented.</p>';
  html += '</div>';
  html += renderAlgorithmTable();
  html += '</div>';

  return html;
}

/* ══════════════════════════════════════
   RESULT TAB SWITCHING
══════════════════════════════════════ */
function switchResultTab(name) {
  document.querySelectorAll('.result-tab').forEach(function(t) {
    t.classList.remove('result-tab--active');
  });
  document.querySelectorAll('.result-tab-panel').forEach(function(p) {
    p.classList.remove('result-tab-panel--active');
  });

  event.target.classList.add('result-tab--active');
  var panel = document.getElementById('rtab-' + name);
  if (panel) panel.classList.add('result-tab-panel--active');
}

/* ══════════════════════════════════════
   ALGORITHM TABLE
══════════════════════════════════════ */
function renderAlgorithmTable() {
  var algorithms = [
    { no: 1,  name: 'Gaussian Blur',               category: 'Noise Removal',        purpose: 'Smooths image using Gaussian function to remove noise',                    func: 'cv2.GaussianBlur()',              params: 'Kernel: 5x5' },
    { no: 2,  name: 'Bilateral Filter',             category: 'Smoothing',            purpose: 'Edge-preserving smoothing — keeps sharp edges while removing noise',       func: 'cv2.bilateralFilter()',           params: 'd=11, σ=17' },
    { no: 3,  name: 'Histogram Equalization',       category: 'Contrast Enhancement', purpose: 'Spreads pixel intensities evenly across full range',                       func: 'cv2.equalizeHist()',              params: 'Global' },
    { no: 4,  name: 'CLAHE',                        category: 'Contrast Enhancement', purpose: 'Adaptive histogram equalization for non-uniform lighting',                 func: 'cv2.createCLAHE()',               params: 'clipLimit=3.0, tile=8x8' },
    { no: 5,  name: 'Sobel Edge Detection',         category: 'Edge Detection',       purpose: 'Gradient-based detection of horizontal and vertical edges',                func: 'cv2.Sobel()',                     params: 'ksize=3, X+Y' },
    { no: 6,  name: 'Laplacian Edge Detection',     category: 'Edge Detection',       purpose: 'Second derivative edge detection in all directions',                       func: 'cv2.Laplacian()',                 params: 'CV_64F' },
    { no: 7,  name: 'Canny Edge Detection',         category: 'Edge Detection',       purpose: 'Multi-stage edge detector — best for plate boundary detection',            func: 'cv2.Canny()',                     params: 'thresh=20,180' },
    { no: 8,  name: 'Otsu Thresholding',            category: 'Thresholding',         purpose: 'Automatic global threshold using intra-class variance minimization',       func: 'cv2.threshold(THRESH_OTSU)',      params: 'Auto threshold' },
    { no: 9,  name: 'Adaptive Thresholding',        category: 'Thresholding',         purpose: 'Local area threshold for varying lighting conditions',                     func: 'cv2.adaptiveThreshold()',         params: 'Gaussian, block=11' },
    { no: 10, name: 'Erosion',                      category: 'Morphological',        purpose: 'Shrinks white regions, removes small noise dots',                          func: 'cv2.erode()',                     params: 'Rect kernel 3x3' },
    { no: 11, name: 'Dilation',                     category: 'Morphological',        purpose: 'Expands white regions, fills gaps in edges',                               func: 'cv2.dilate()',                    params: 'Rect kernel 3x3' },
    { no: 12, name: 'Morphological Opening',        category: 'Morphological',        purpose: 'Erosion then Dilation — removes small objects',                            func: 'cv2.morphologyEx(MORPH_OPEN)',    params: 'Rect kernel 3x3' },
    { no: 13, name: 'Morphological Closing',        category: 'Morphological',        purpose: 'Dilation then Erosion — fills holes, connects nearby regions',             func: 'cv2.morphologyEx(MORPH_CLOSE)',   params: 'Rect kernel 3x3' },
    { no: 14, name: 'Contour Detection',            category: 'Shape Detection',      purpose: 'Finds plate-shaped rectangles using aspect ratio filtering',               func: 'cv2.findContours()',              params: 'Aspect 1.5-6.0' },
    { no: 15, name: 'Haar Cascade',                 category: 'ML Detection',         purpose: 'Machine learning plate detector using Viola-Jones algorithm',              func: 'cv2.CascadeClassifier()',         params: 'scaleFactor=1.05' },
    { no: 16, name: 'Affine Transform',             category: 'Geometric',            purpose: 'Corrects rotation and tilt of number plates',                              func: 'cv2.warpAffine()',                params: 'Rotation matrix' },
    { no: 17, name: 'Connected Components',         category: 'Segmentation',         purpose: 'Groups connected pixels — identifies character regions',                   func: 'cv2.connectedComponentsWithStats()', params: 'connectivity=8' },
  ];

  var categoryColors = {
    'Noise Removal':        'tag--blue',
    'Smoothing':            'tag--blue',
    'Contrast Enhancement': 'tag--purple',
    'Edge Detection':       'tag--orange',
    'Thresholding':         'tag--green',
    'Morphological':        'tag--red',
    'Shape Detection':      'tag--green',
    'ML Detection':         'tag--purple',
    'Geometric':            'tag--blue',
    'Segmentation':         'tag--orange',
  };

  var html = '<div class="algo-table-wrap"><table class="algo-table">';
  html += '<thead><tr>'
        + '<th>#</th>'
        + '<th>Algorithm</th>'
        + '<th>Category</th>'
        + '<th>Purpose</th>'
        + '<th>OpenCV Function</th>'
        + '<th>Parameters</th>'
        + '</tr></thead><tbody>';

  algorithms.forEach(function(a) {
    var tagClass = categoryColors[a.category] || 'tag--blue';
    html += '<tr>'
          + '<td><strong>' + a.no + '</strong></td>'
          + '<td><strong>' + a.name + '</strong></td>'
          + '<td><span class="tag ' + tagClass + '">' + a.category + '</span></td>'
          + '<td>' + a.purpose + '</td>'
          + '<td><code>' + a.func + '</code></td>'
          + '<td><code>' + a.params + '</code></td>'
          + '</tr>';
  });

  html += '</tbody></table></div>';
  return html;
}

/* ══════════════════════════════════════
   VIDEO TAB
══════════════════════════════════════ */
function onVideoSelect(e) {
  var file = e.target.files[0];
  if (!file) return;
  selectedVideoFile = file;
  document.getElementById('videoPreview').src = URL.createObjectURL(file);
  document.getElementById('videoPreviewWrap').classList.add('preview-wrap--visible');
  document.getElementById('btnDetectVideo').disabled = false;
  document.getElementById('videoResult').innerHTML = '';
}

function clearVideo() {
  selectedVideoFile = null;
  document.getElementById('videoFile').value = '';
  document.getElementById('videoPreviewWrap').classList.remove('preview-wrap--visible');
  document.getElementById('videoResult').innerHTML = '';
  document.getElementById('btnDetectVideo').disabled = true;
}

function detectVideo() {
  if (!selectedVideoFile) return;
  var btn = document.getElementById('btnDetectVideo');
  btn.disabled = true;
  document.getElementById('videoLoader').classList.add('loader--visible');
  document.getElementById('videoResult').innerHTML = '';

  var fd = new FormData();
  fd.append('file', selectedVideoFile);

  fetch('/detect/video', { method: 'POST', body: fd })
    .then(function(res) { return res.json(); })
    .then(function(data) {
      document.getElementById('videoResult').innerHTML = renderVideoResult(data);
    })
    .catch(function(err) {
      document.getElementById('videoResult').innerHTML = makeAlert('error', '❌ ' + err.message);
    })
    .finally(function() {
      document.getElementById('videoLoader').classList.remove('loader--visible');
      btn.disabled = false;
    });
}

function renderVideoResult(data) {
  if (!data.success && data.error)
    return makeAlert('error', '❌ ' + data.error);

  var vi = data.video_info || {};
  var html = makeAlert('success',
    '✅ Analyzed ' + (vi.frames_analyzed || 0) + ' frames from ' + (vi.duration_seconds || 0) + 's video');

  html += '<div class="result-section"><div class="result-grid">';
  html += '<div class="result-card">'
        + '<div class="result-card__header">🎬 Video Information</div>'
        + '<div class="result-card__body"><div class="video-stats">'
        + statBox('Duration',        (vi.duration_seconds || 0) + 's')
        + statBox('FPS',             vi.fps || 0)
        + statBox('Frames Analyzed', vi.frames_analyzed || 0)
        + statBox('Detections',      (data.all_detections || []).length)
        + '</div></div></div>';

  if (data.best_plate_text) {
    html += renderPlateCard(data.best_plate_text, data.plate_info, 'video');
  } else {
    html += '<div class="result-card"><div class="result-card__header">Result</div>'
          + '<div class="result-card__body"><p class="text-muted">No clear plate detected.</p></div></div>';
  }

  if (data.all_detections && data.all_detections.length > 0) {
    html += '<div class="result-card result-grid--full">'
          + '<div class="result-card__header">📋 All Frame Detections</div>'
          + '<div class="result-card__body"><div class="detection-list">';
    data.all_detections.forEach(function(d) {
      html += '<div class="detection-item">'
            + '<span class="detection-item__time">🕐 ' + d.timestamp + 's &nbsp;|&nbsp; Frame ' + d.frame + '</span>'
            + '<span class="detection-item__plate">' + d.text + '</span>'
            + '</div>';
    });
    html += '</div></div></div>';
  }

  html += '</div></div>';
  return html;
}

/* ══════════════════════════════════════
   MANUAL TAB
══════════════════════════════════════ */
function parseManual() {
  var text = document.getElementById('manualPlate').value.trim();
  if (!text) return;

  fetch('/parse', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ plate_text: text })
  })
    .then(function(res) { return res.json(); })
    .then(function(data) {
      document.getElementById('manualResult').innerHTML =
        '<div class="result-grid">' + renderPlateCard(text, data, 'manual') + '</div>';
    })
    .catch(function(err) {
      document.getElementById('manualResult').innerHTML = makeAlert('error', err.message);
    });
}

/* ══════════════════════════════════════
   HELPERS
══════════════════════════════════════ */
function renderPlateCard(ocrText, info, method) {
  var conf      = info.confidence || 'Low';
  var confClass = conf === 'High' ? 'tag--green' : conf === 'Medium' ? 'tag--orange' : 'tag--red';
  var validTag  = info.valid
    ? '<span class="tag tag--green">✓ Valid Format</span>'
    : '<span class="tag tag--red">⚠ Unknown Format</span>';
  var evTag  = info.is_electric  ? '<span class="tag tag--purple">⚡ Electric Vehicle</span>' : '';
  var bhTag  = info.is_bh_series ? '<span class="tag tag--blue">🇮🇳 BH Series</span>' : '';
  var pClass = ocrText ? '' : 'plate--empty';
  var pText  = ocrText || '— — —';
  var regYear = (info.registration_year && info.registration_year !== 'Unknown')
    ? '<tr><td>Reg. Year</td><td>' + info.registration_year + '</td></tr>' : '';

  return '<div class="result-card">'
    + '<div class="result-card__header">🚗 Vehicle Details</div>'
    + '<div class="result-card__body">'
    + '<div class="plate ' + pClass + '">'
    + '<div class="plate__stripe"></div>'
    + pText
    + '</div>'
    + '<div class="tags">' + validTag + evTag + bhTag
    + '<span class="tag ' + confClass + '">Confidence: ' + conf + '</span></div>'
    + '<table class="info-table">'
    + '<tr><td>Plate Type</td><td>'    + (info.plate_type     || '—') + '</td></tr>'
    + '<tr><td>State</td><td>'         + (info.state          || '—') + '</td></tr>'
    + '<tr><td>RTO Office</td><td>'    + (info.rto_office     || '—') + '</td></tr>'
    + '<tr><td>Series</td><td>'        + (info.series         || '—') + '</td></tr>'
    + '<tr><td>Vehicle No.</td><td>'   + (info.vehicle_number || '—') + '</td></tr>'
    + '<tr><td>Vehicle Class</td><td>' + (info.vehicle_class  || '—') + '</td></tr>'
    + '<tr><td>Fuel Type</td><td>'     + (info.fuel_type      || '—') + '</td></tr>'
    + regYear
    + '</table>'
    + '<p class="text-muted mt-8">Detection method: ' + method + '</p>'
    + '</div></div>';
}

function statBox(label, value) {
  return '<div class="stat-box">'
    + '<div class="stat-box__label">' + label + '</div>'
    + '<div class="stat-box__value">' + value + '</div>'
    + '</div>';
}

function makeAlert(type, msg) {
  var cls = type === 'success' ? 'alert--success'
          : type === 'warning' ? 'alert--warning'
          : 'alert--error';
  return '<div class="alert ' + cls + '">' + msg + '</div>';
}

function scorePlate(t) {
  if (!t) return 0;
  var s = t.length;
  if (/^[A-Z]{2}\d{2}/.test(t)) s += 5;
  if (/^[A-Z]{2}\d{2}[A-Z]{1,3}\d{1,4}$/.test(t)) s += 10;
  return s;
}

/* ══════════════════════════════════════
   DRAG AND DROP
══════════════════════════════════════ */
['imageZone', 'videoZone'].forEach(function(id) {
  var zone = document.getElementById(id);
  if (!zone) return;
  zone.addEventListener('dragover', function(e) {
    e.preventDefault();
    zone.classList.add('upload-zone--drag');
  });
  zone.addEventListener('dragleave', function() {
    zone.classList.remove('upload-zone--drag');
  });
  zone.addEventListener('drop', function(e) {
    e.preventDefault();
    zone.classList.remove('upload-zone--drag');
    var files = e.dataTransfer.files;
    if (files.length > 0) {
      var input = zone.querySelector('.upload-zone__input');
      var dt = new DataTransfer();
      dt.items.add(files[0]);
      input.files = dt.files;
      input.dispatchEvent(new Event('change'));
    }
  });
});