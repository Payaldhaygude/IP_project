"""
Number Plate Detection Engine
All 17 Image Processing Algorithms Implementation
"""

import cv2
import numpy as np
import pytesseract
import re
import os
import shutil
from PIL import Image

# ─────────────────────────────────────────────
#  TESSERACT PATH — Auto-detect (cross-platform)
# ─────────────────────────────────────────────

def _find_tesseract():
    # 1. Check environment variable
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path and os.path.isfile(env_path):
        return env_path
    # 2. Windows default install path
    win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.isfile(win_path):
        return win_path
    win_path2 = r"C:\Users\Admin\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    if os.path.isfile(win_path2):
        return win_path2
    # 3. Linux/Mac — use which
    found = shutil.which("tesseract")
    if found:
        return found
    # 4. Common Linux paths
    for p in ["/usr/bin/tesseract", "/usr/local/bin/tesseract"]:
        if os.path.isfile(p):
            return p
    return "tesseract"  # fallback: hope it's on PATH

pytesseract.pytesseract.tesseract_cmd = _find_tesseract()


# ─────────────────────────────────────────────
#  HAAR CASCADE PATHS
# ─────────────────────────────────────────────

CASCADE_PATHS = [
    cv2.data.haarcascades + "haarcascade_russian_plate_number.xml",
    cv2.data.haarcascades + "haarcascade_license_plate_rus_16stages.xml",
]


def load_cascades():
    cascades = []
    for path in CASCADE_PATHS:
        if os.path.exists(path):
            cascades.append(cv2.CascadeClassifier(path))
    return cascades


CASCADES = load_cascades()


# ─────────────────────────────────────────────
#  RESULTS FOLDER
# ─────────────────────────────────────────────

def get_results_folder():
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "results")
    os.makedirs(folder, exist_ok=True)
    return folder


def save_step_image(img, name, base_id):
    """Save a pipeline step image and return its URL."""
    folder = get_results_folder()
    filename = f"pipeline_{base_id}_{name}.jpg"
    path = os.path.join(folder, filename)
    if len(img.shape) == 2:
        save_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        save_img = img.copy()
    cv2.imwrite(path, save_img)
    return f"/static/results/{filename}"


# ─────────────────────────────────────────────
#  ALGORITHM 1 — GAUSSIAN BLUR
# ─────────────────────────────────────────────

def apply_gaussian_blur(img, kernel_size=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    return blurred


# ─────────────────────────────────────────────
#  ALGORITHM 2 — BILATERAL FILTER
# ─────────────────────────────────────────────

def apply_bilateral_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    return bilateral


# ─────────────────────────────────────────────
#  ALGORITHM 3 — HISTOGRAM EQUALIZATION
# ─────────────────────────────────────────────

def apply_histogram_equalization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    equalized = cv2.equalizeHist(gray)
    return equalized


# ─────────────────────────────────────────────
#  ALGORITHM 4 — CLAHE
# ─────────────────────────────────────────────

def apply_clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    return clahe_img


# ─────────────────────────────────────────────
#  ALGORITHM 5 — SOBEL EDGE DETECTION
# ─────────────────────────────────────────────

def apply_sobel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))
    return sobel_combined


# ─────────────────────────────────────────────
#  ALGORITHM 6 — LAPLACIAN EDGE DETECTION
# ─────────────────────────────────────────────

def apply_laplacian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))
    return laplacian


# ─────────────────────────────────────────────
#  ALGORITHM 7 — CANNY EDGE DETECTION
# ─────────────────────────────────────────────

def apply_canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 20, 180)
    return canny


# ─────────────────────────────────────────────
#  ALGORITHM 8 — OTSU THRESHOLDING
# ─────────────────────────────────────────────

def apply_otsu_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu


# ─────────────────────────────────────────────
#  ALGORITHM 9 — ADAPTIVE THRESHOLDING
# ─────────────────────────────────────────────

def apply_adaptive_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    adaptive = cv2.adaptiveThreshold(
        bilateral, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return adaptive


# ─────────────────────────────────────────────
#  ALGORITHM 10 — EROSION
# ─────────────────────────────────────────────

def apply_erosion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(binary, kernel, iterations=1)
    return eroded


# ─────────────────────────────────────────────
#  ALGORITHM 11 — DILATION
# ─────────────────────────────────────────────

def apply_dilation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return dilated


# ─────────────────────────────────────────────
#  ALGORITHM 12 — MORPHOLOGICAL OPENING
# ─────────────────────────────────────────────

def apply_morphological_opening(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return opened


# ─────────────────────────────────────────────
#  ALGORITHM 13 — MORPHOLOGICAL CLOSING
# ─────────────────────────────────────────────

def apply_morphological_closing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return closed


# ─────────────────────────────────────────────
#  ALGORITHM 14 — CONTOUR DETECTION
# ─────────────────────────────────────────────

def apply_contour_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 20, 180)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(img.shape) == 2:
        contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        contour_img = img.copy()

    cv2.drawContours(contour_img, cnts, -1, (0, 255, 0), 1)

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h if h > 0 else 0
        if 1.5 < aspect < 6.0 and w > 50 and h > 15:
            cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return contour_img


# ─────────────────────────────────────────────
#  ALGORITHM 15 — HAAR CASCADE DETECTION
# ─────────────────────────────────────────────

def apply_haar_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    result_img = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for cascade in CASCADES:
        plates = cascade.detectMultiScale(
            gray, scaleFactor=1.05,
            minNeighbors=3,
            minSize=(60, 15)
        )
        if len(plates):
            for (x, y, w, h) in plates:
                cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(result_img, "Plate", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return result_img


# ─────────────────────────────────────────────
#  ALGORITHM 16 — AFFINE TRANSFORM
# ─────────────────────────────────────────────

def apply_affine_transform(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle=0, scale=1.0)
    affine = cv2.warpAffine(gray, rotation_matrix, (w, h))
    return affine


# ─────────────────────────────────────────────
#  ALGORITHM 17 — CONNECTED COMPONENTS
# ─────────────────────────────────────────────

def apply_connected_components(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    output = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 50:
            color = (
                np.random.randint(100, 255),
                np.random.randint(100, 255),
                np.random.randint(100, 255)
            )
            output[labels == i] = color

    return output


# ─────────────────────────────────────────────
#  FULL PIPELINE — ALL ALGORITHMS
# ─────────────────────────────────────────────

def run_full_pipeline(img, base_id):
    pipeline_steps = []

    def add_step(name, title, description, result_img):
        url = save_step_image(result_img, name, base_id)
        pipeline_steps.append({
            "name": name,
            "title": title,
            "description": description,
            "url": url,
        })

    add_step("01_original", "Original Image",
             "Raw input image before any processing.", img)
    gaussian = apply_gaussian_blur(img)
    add_step("02_gaussian", "Gaussian Blur",
             "Noise removal using Gaussian function. Kernel size: 5x5.", gaussian)
    bilateral = apply_bilateral_filter(img)
    add_step("03_bilateral", "Bilateral Filter",
             "Edge-preserving smoothing. Keeps sharp edges while removing noise.", bilateral)
    hist_eq = apply_histogram_equalization(img)
    add_step("04_histogram", "Histogram Equalization",
             "Basic contrast enhancement. Spreads pixel intensities evenly.", hist_eq)
    clahe = apply_clahe(img)
    add_step("05_clahe", "CLAHE",
             "Contrast Limited Adaptive Histogram Equalization. Better for non-uniform lighting.", clahe)
    sobel = apply_sobel(img)
    add_step("06_sobel", "Sobel Edge Detection",
             "Gradient-based edge detection in X and Y directions.", sobel)
    laplacian = apply_laplacian(img)
    add_step("07_laplacian", "Laplacian Edge Detection",
             "Second derivative edge detection. Detects edges in all directions.", laplacian)
    canny = apply_canny(img)
    add_step("08_canny", "Canny Edge Detection",
             "Multi-stage edge detection. Best algorithm for plate boundary detection.", canny)
    otsu = apply_otsu_threshold(img)
    add_step("09_otsu", "Otsu Thresholding",
             "Automatic global threshold. Minimizes intra-class variance.", otsu)
    adaptive = apply_adaptive_threshold(img)
    add_step("10_adaptive", "Adaptive Thresholding",
             "Local area threshold. Works better for varying lighting conditions.", adaptive)
    erosion = apply_erosion(img)
    add_step("11_erosion", "Erosion",
             "Shrinks white regions. Removes small noise dots and thin lines.", erosion)
    dilation = apply_dilation(img)
    add_step("12_dilation", "Dilation",
             "Expands white regions. Fills gaps and connects broken edges.", dilation)
    opening = apply_morphological_opening(img)
    add_step("13_opening", "Morphological Opening",
             "Erosion then Dilation. Removes small objects while keeping large structures.", opening)
    closing = apply_morphological_closing(img)
    add_step("14_closing", "Morphological Closing",
             "Dilation then Erosion. Fills holes and connects nearby regions.", closing)
    contours = apply_contour_detection(img)
    add_step("15_contours", "Contour Detection",
             "Finds plate-shaped rectangles. Green=all contours, Red=plate candidates.", contours)
    haar = apply_haar_detection(img)
    add_step("16_haar", "Haar Cascade Detection",
             "ML-based plate detector using Viola-Jones algorithm. Blue boxes=detected plates.", haar)
    affine = apply_affine_transform(img)
    add_step("17_affine", "Affine Transform",
             "Geometric correction for tilted/rotated plates.", affine)
    components = apply_connected_components(img)
    add_step("18_components", "Connected Components",
             "Groups connected pixels. Each color = one component (character region).", components)

    return pipeline_steps


# ─────────────────────────────────────────────
#  PREPROCESSING PIPELINE (for OCR)
# ─────────────────────────────────────────────

def preprocess_image(img: np.ndarray) -> dict:
    h, w = img.shape[:2]

    if w < 300:
        scale = 300 / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if w > 1200:
        scale = 1200 / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(bilateral)
    _, otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    return {
        "original": img,
        "gray": gray,
        "bilateral": bilateral,
        "clahe": clahe_img,
        "otsu": otsu,
        "adaptive": adaptive,
        "morph": morph,
    }


# ─────────────────────────────────────────────
#  PLATE REGION DETECTION
# ─────────────────────────────────────────────

def detect_plates_haar(img: np.ndarray) -> list:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    regions = []
    for cascade in CASCADES:
        plates = cascade.detectMultiScale(
            gray, scaleFactor=1.05,
            minNeighbors=3,
            minSize=(60, 15),
            maxSize=(500, 200),
        )
        if len(plates):
            for (x, y, w, h) in plates:
                regions.append((x, y, w, h, "haar"))
    return regions


def detect_plates_contour(img: np.ndarray) -> list:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 20, 180)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:50]
    regions = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) >= 4:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / h if h > 0 else 0
            if 1.5 < aspect < 6.0 and w > 50 and h > 15 and h < 200:
                regions.append((x, y, w, h, "contour"))
    return regions


def deduplicate_regions(regions: list, overlap_thresh=0.5) -> list:
    if not regions:
        return []
    final = []
    used = [False] * len(regions)
    for i, r1 in enumerate(regions):
        if used[i]:
            continue
        x1, y1, w1, h1 = r1[:4]
        keep = True
        for j, r2 in enumerate(regions):
            if i == j or used[j]:
                continue
            x2, y2, w2, h2 = r2[:4]
            ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            inter = ix * iy
            union = w1 * h1 + w2 * h2 - inter
            iou = inter / union if union > 0 else 0
            if iou > overlap_thresh and w1 * h1 < w2 * h2:
                keep = False
                break
        if keep:
            final.append(r1)
            used[i] = True
    return final


# ─────────────────────────────────────────────
#  OCR ENGINE
# ─────────────────────────────────────────────

def run_ocr_on_region(region_img: np.ndarray) -> str:
    processed = preprocess_image(region_img)
    configs = [
        ("--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", "otsu"),
        ("--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", "morph"),
        ("--psm 13 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", "adaptive"),
        ("--psm 6 --oem 3", "clahe"),
    ]
    results = []
    for config, img_key in configs:
        try:
            text = pytesseract.image_to_string(processed[img_key], config=config)
            text = clean_plate_text(text)
            if len(text) >= 4:
                results.append(text)
        except Exception:
            pass
    if not results:
        return ""
    results.sort(key=lambda x: score_plate_text(x), reverse=True)
    return results[0]


def clean_plate_text(text: str) -> str:
    text = text.upper().strip()
    text = re.sub(r'[^A-Z0-9]', '', text)

    # ── Step 1: Strip leading garbage until we find a known state code ──
    # OCR often prepends 1–3 junk chars before the actual plate (e.g. "IDMH42..." → "MH42...")
    KNOWN_STATES = {
        'AN','AP','AR','AS','BR','CG','CH','DD','DL','DN','GA','GJ',
        'HP','HR','JH','JK','KA','KL','LA','LD','MH','ML','MN','MP',
        'MZ','NL','OD','OR','PB','PY','RJ','SK','TG','TN','TR','TS',
        'UK','UP','WB',
    }
    # Try to find a valid 2-letter state code starting from pos 0 to 3
    found_at = -1
    for start in range(min(4, len(text) - 1)):
        candidate = text[start:start+2]
        if candidate in KNOWN_STATES:
            found_at = start
            break
    if found_at > 0:
        text = text[found_at:]

    # ── Step 2: Common OCR misread corrections on first two chars ──
    if len(text) >= 2:
        common_fixes = {
            'WH': 'MH', 'VH': 'MH', 'NH': 'MH', 'RH': 'MH', 'IH': 'MH',
            'WA': 'MA', 'WB': 'WB',  # WB is West Bengal — keep it
            'WL': 'ML', 'WN': 'MN', 'WP': 'MP',
            'OL': 'DL', 'OA': 'DA', 'QL': 'DL', 'IL': 'DL',
            'KN': 'KA', 'TW': 'TN',
            'UN': 'UP', 'UF': 'UP', 'HB': 'HR',
            'G0': 'GO', '0D': 'OD',
        }
        first_two = text[:2]
        if first_two in common_fixes:
            text = common_fixes[first_two] + text[2:]

    # ── Step 3: Fix digit/letter confusions inside the plate ──
    # Skip for BH series (starts with 2 digits like "22BH...")
    is_bh = bool(re.match(r'^\d{2}BH', text))
    if len(text) >= 4 and not is_bh:
        # pos 2 and 3 must be digits — fix common letter→digit misreads
        letter_to_digit = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'G': '6', 'Z': '2'}
        digit_to_letter = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G', '2': 'Z'}
        t = list(text)
        # positions 2, 3 should be digits
        for pos in [2, 3]:
            if pos < len(t) and t[pos].isalpha() and t[pos] in letter_to_digit:
                t[pos] = letter_to_digit[t[pos]]
        # positions 0, 1 should be letters
        for pos in [0, 1]:
            if pos < len(t) and t[pos].isdigit() and t[pos] in digit_to_letter:
                t[pos] = digit_to_letter[t[pos]]
        text = ''.join(t)

    # ── Step 4: Enforce max length (Indian plates are max 10 chars) ──
    if len(text) > 10:
        # Try to extract a valid sub-pattern
        m = re.search(r'[A-Z]{2}\d{2}[A-Z]{1,3}\d{1,4}', text)
        if m:
            text = m.group(0)

    return text


def score_plate_text(text: str) -> int:
    score = 0

    # Ideal Indian plate length is 8–10 chars; penalise outside that range
    if 8 <= len(text) <= 10:
        score += 4
    elif 6 <= len(text) <= 7:
        score += 2
    elif len(text) > 10:
        score -= 5   # punish garbage-padded strings hard
    elif len(text) < 6:
        score -= 2

    # Must start with two letters (state code)
    if re.match(r'^[A-Z]{2}', text):
        score += 3

    # Two letters + two digits (state + RTO number)
    if re.match(r'^[A-Z]{2}\d{2}', text):
        score += 3

    # Full standard Indian plate pattern — highest reward
    if re.match(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{1,4}$', text):
        score += 8

    # BH series pattern
    if re.match(r'^\d{2}BH\d{4}[A-Z]{1,2}$', text):
        score += 8

    return score


# ─────────────────────────────────────────────
#  MAIN DETECTION FUNCTION
# ─────────────────────────────────────────────

def detect_number_plate(image_path: str) -> dict:
    from rto_database import parse_indian_number_plate

    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Could not load image", "success": False}

    h, w = img.shape[:2]
    base_id = os.path.splitext(os.path.basename(image_path))[0]

    results = {
        "success": False,
        "image_size": f"{w}x{h}",
        "plates_found": [],
        "annotated_image": None,
        "pipeline_steps": [],
    }

    pipeline_steps = run_full_pipeline(img, base_id)
    results["pipeline_steps"] = pipeline_steps

    attempts = [
        img,
        cv2.convertScaleAbs(img, alpha=1.3, beta=30),
        cv2.convertScaleAbs(img, alpha=1.5, beta=50),
        cv2.convertScaleAbs(img, alpha=1.8, beta=60),
    ]

    all_regions = []
    best_img = img
    for attempt_img in attempts:
        haar = detect_plates_haar(attempt_img)
        contour = detect_plates_contour(attempt_img)
        regions = deduplicate_regions(haar + contour)
        if regions:
            all_regions = regions
            best_img = attempt_img
            break

    annotated = cv2.imread(image_path)
    valid_plates = []

    for region in all_regions:
        x, y, w_r, h_r, method = region
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(best_img.shape[1], x + w_r + pad)
        y2 = min(best_img.shape[0], y + h_r + pad)

        plate_crop = best_img[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue

        scale = max(1, 150 // plate_crop.shape[0])
        if scale > 1:
            plate_crop = cv2.resize(
                plate_crop, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        ocr_text = run_ocr_on_region(plate_crop)

        if not ocr_text or len(ocr_text) < 4:
            continue
        has_letters = any(c.isalpha() for c in ocr_text)
        has_numbers = any(c.isdigit() for c in ocr_text)
        if not has_letters or not has_numbers:
            continue
        if len(ocr_text) > 10:
            continue

        plate_info = parse_indian_number_plate(ocr_text)
        valid_plates.append({
            "bbox": [x1, y1, x2, y2],
            "detection_method": method,
            "ocr_text": ocr_text,
            "plate_info": plate_info,
        })

        color = (0, 180, 0) if plate_info["valid"] else (0, 140, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label = ocr_text
        font_scale = 0.7
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "static", "results", f"annotated_{base}.jpg"
    )
    cv2.imwrite(out_path, annotated)

    results["annotated_image"] = out_path
    results["annotated_image_url"] = f"/static/results/annotated_{base}.jpg"
    results["plates_found"] = valid_plates
    results["total_plates"] = len(valid_plates)
    results["success"] = True

    return results


# ─────────────────────────────────────────────
#  VIDEO PROCESSING
# ─────────────────────────────────────────────

def process_video(video_path: str, max_frames: int = 150) -> dict:
    from rto_database import parse_indian_number_plate

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video", "success": False}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0
    sample_every = max(1, total_frames // max_frames)

    all_texts = []
    frame_results = []
    frame_idx = 0
    processed = 0

    while cap.isOpened() and processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            haar = detect_plates_haar(frame)
            contour = detect_plates_contour(frame)
            regions = deduplicate_regions(haar + contour)

            for region in regions:
                x, y, w_r, h_r, method = region
                crop = frame[y:y+h_r, x:x+w_r]
                if crop.size == 0:
                    continue
                scale = max(1, 120 // crop.shape[0])
                if scale > 1:
                    crop = cv2.resize(crop, None, fx=scale, fy=scale,
                                      interpolation=cv2.INTER_CUBIC)
                text = run_ocr_on_region(crop)

                if not text or len(text) < 5:
                    continue
                has_letters = any(c.isalpha() for c in text)
                has_numbers = any(c.isdigit() for c in text)
                if not has_letters or not has_numbers:
                    continue
                if len(text) > 10:
                    continue

                all_texts.append(text)
                frame_results.append({
                    "frame": frame_idx,
                    "text": text,
                    "timestamp": round(frame_idx / fps, 2),
                })
            processed += 1
        frame_idx += 1

    cap.release()

    best_text = ""
    if all_texts:
        from collections import Counter
        counts = Counter(all_texts)
        best_text = counts.most_common(1)[0][0]

    plate_info = parse_indian_number_plate(best_text) if best_text else {}

    return {
        "success": True,
        "video_info": {
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "duration_seconds": round(duration, 2),
            "frames_analyzed": processed,
        },
        "best_plate_text": best_text,
        "plate_info": plate_info,
        "all_detections": frame_results[:20],
        "unique_plates": list(set(all_texts)),
    }