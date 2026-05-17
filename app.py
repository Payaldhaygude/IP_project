"""
Number Plate Detection Web App
Flask backend serving the UI and processing endpoints
"""
import os
import json
import uuid
import numpy as np
import xml.etree.ElementTree as ET
from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__,
            static_folder='static',
            static_url_path='/static')

app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
RESULT_FOLDER = os.path.join(os.path.dirname(__file__), "static", "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

ALLOWED_IMAGES = {"png", "jpg", "jpeg", "webp", "bmp"}
ALLOWED_VIDEOS  = {"mp4", "avi", "mov", "mkv", "webm"}

# ── dataset folder (images + XML annotations) ─────────────────────────────────
GOOGLE_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "google_images")
# ──────────────────────────────────────────────────────────────────────────────


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        return super().default(obj)


def json_response(data, status=200):
    return app.response_class(
        response=json.dumps(data, cls=NumpyEncoder),
        status=status,
        mimetype='application/json'
    )


def allowed_file(filename, types):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in types


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect/image", methods=["POST"])
def detect_image():
    if "file" not in request.files:
        return json_response({"error": "No file provided"}, 400)

    file = request.files["file"]
    if not file.filename:
        return json_response({"error": "No file selected"}, 400)

    if not allowed_file(file.filename, ALLOWED_IMAGES):
        return json_response({"error": "Invalid image format. Use JPG, PNG, WEBP, BMP."}, 400)

    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(save_path)

    try:
        from plate_detector import detect_number_plate
        result = detect_number_plate(save_path)

        if result.get("annotated_image"):
            fname = os.path.basename(result["annotated_image"])
            result["annotated_image_url"] = f"/static/results/{fname}"

        result["uploaded_image_url"] = f"/static/uploads/{unique_name}"
        return json_response(result)

    except Exception as e:
        return json_response({"error": str(e), "success": False}, 500)


@app.route("/detect/video", methods=["POST"])
def detect_video():
    if "file" not in request.files:
        return json_response({"error": "No file provided"}, 400)

    file = request.files["file"]
    if not file.filename:
        return json_response({"error": "No file selected"}, 400)

    if not allowed_file(file.filename, ALLOWED_VIDEOS):
        return json_response({"error": "Invalid video format. Use MP4, AVI, MOV, MKV."}, 400)

    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(save_path)

    try:
        from plate_detector import process_video
        result = process_video(save_path)
        return json_response(result)

    except Exception as e:
        return json_response({"error": str(e), "success": False}, 500)


@app.route("/parse", methods=["POST"])
def parse_plate():
    data = request.get_json()
    plate_text = data.get("plate_text", "")
    if not plate_text:
        return json_response({"error": "No plate text"}, 400)
    from rto_database import parse_indian_number_plate
    return json_response(parse_indian_number_plate(plate_text))


# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY MATRIX BLOCK
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth(images_dir: str) -> dict:
    """
    Scans images_dir for .xml annotation files and returns:
        { image_filename: plate_text }
    Plate text comes from <object><name>; empty string for negative samples.
    """
    ground_truth = {}
    for fname in os.listdir(images_dir):
        if not fname.lower().endswith(".xml"):
            continue
        xml_path = os.path.join(images_dir, fname)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            img_filename = root.findtext("filename", default="").strip()
            if not img_filename:
                continue

            obj = root.find("object")
            plate_text = obj.findtext("name", default="").strip() if obj is not None else ""
            ground_truth[img_filename] = plate_text.upper()

        except Exception as e:
            print(f"  [WARN] Could not parse {fname}: {e}")
    return ground_truth


def evaluate_accuracy(images_dir: str, ground_truth: dict) -> dict:
    """
    Runs detector on every image, compares against ground truth,
    and returns accuracy, precision, recall, f1 + confusion counts.
    """
    from plate_detector import detect_number_plate

    TP = TN = FP = FN = 0
    results = []
    total = len(ground_truth)

    for idx, (img_filename, expected) in enumerate(ground_truth.items(), 1):
        img_path = os.path.join(images_dir, img_filename)

        if not os.path.exists(img_path):
            print(f"  [SKIP] {img_filename}  <- image file not found")
            continue

        try:
            result   = detect_number_plate(img_path)
            detected = result.get("plate_text", "").strip().upper()

            has_plate_gt   = bool(expected)
            has_plate_pred = bool(detected)
            exact_match    = (detected == expected)

            if   has_plate_gt  and exact_match:        TP += 1
            elif has_plate_gt  and not exact_match:    FN += 1
            elif not has_plate_gt and has_plate_pred:  FP += 1
            else:                                      TN += 1

            status = "OK" if exact_match else "XX"
            print(f"  [{status}] ({idx:>3}/{total}) {img_filename[:45]:<45}"
                  f"  expected='{expected}'  got='{detected}'")

            results.append({
                "file"    : img_filename,
                "expected": expected,
                "detected": detected,
                "match"   : exact_match,
            })

        except Exception as e:
            print(f"  [ERR] {img_filename}: {e}")
            FN += 1

    total_counted = TP + TN + FP + FN
    accuracy  = (TP + TN) / total_counted                         if total_counted        > 0 else 0.0
    precision = TP        / (TP + FP)                             if (TP + FP)            > 0 else 0.0
    recall    = TP        / (TP + FN)                             if (TP + FN)            > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall)     if (precision + recall) > 0 else 0.0

    return {
        "accuracy" : round(accuracy,  4),
        "precision": round(precision, 4),
        "recall"   : round(recall,    4),
        "f1_score" : round(f1,        4),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "total": total_counted,
        "per_image_results": results,
    }


def run_accuracy_check():
    """Called when you run:  python app.py --accuracy"""
    import sys

    print("\n" + "=" * 60)
    print("   NUMBER PLATE DETECTION - ACCURACY MATRIX")
    print("=" * 60)
    print(f"  Dataset folder : {GOOGLE_IMAGES_DIR}")

    if not os.path.isdir(GOOGLE_IMAGES_DIR):
        print(f"\n  [ERROR] Folder not found: {GOOGLE_IMAGES_DIR}")
        sys.exit(1)

    print("  Loading ground truth from XML files ...")
    ground_truth = load_ground_truth(GOOGLE_IMAGES_DIR)

    if not ground_truth:
        print("  [ERROR] No XML annotation files found.")
        sys.exit(1)

    has_plate = sum(1 for v in ground_truth.values() if v)
    no_plate  = len(ground_truth) - has_plate
    print(f"  Annotations    : {len(ground_truth)}  "
          f"(with plate: {has_plate},  negatives: {no_plate})")
    print("-" * 60)

    metrics = evaluate_accuracy(GOOGLE_IMAGES_DIR, ground_truth)

    print("-" * 60)
    print(f"  Accuracy   : {metrics['accuracy']}")
    print(f"  Precision  : {metrics['precision']}")
    print(f"  Recall     : {metrics['recall']}")
    print(f"  F1 Score   : {metrics['f1_score']}")
    print("-" * 60)
    print(f"  TP={metrics['TP']}  TN={metrics['TN']}  "
          f"FP={metrics['FP']}  FN={metrics['FN']}  "
          f"Total={metrics['total']}")
    print("=" * 60 + "\n")


# ── optional API endpoint ─────────────────────────────────────────────────────
@app.route("/accuracy", methods=["GET"])
def accuracy_endpoint():
    """GET /accuracy  — runs evaluation and returns JSON metrics."""
    try:
        ground_truth = load_ground_truth(GOOGLE_IMAGES_DIR)
        if not ground_truth:
            return json_response({"error": "No XML annotations found in google_images/"}, 400)
        metrics = evaluate_accuracy(GOOGLE_IMAGES_DIR, ground_truth)
        return json_response({"success": True, "metrics": metrics})
    except Exception as e:
        return json_response({"error": str(e), "success": False}, 500)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--accuracy":
        run_accuracy_check()
    else:
        app.run(debug=True, host="0.0.0.0", port=5000)