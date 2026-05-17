# 🚗 Number Plate Detection System

> A classical computer-vision pipeline for detecting and parsing Indian vehicle number plates — built with OpenCV, Tesseract OCR, and Flask. No deep learning, no GPU required.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-orange)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [17 Image Processing Algorithms](#17-image-processing-algorithms)
- [Indian Plate Format Support](#indian-plate-format-support)
- [API Endpoints](#api-endpoints)
- [Accuracy Evaluation](#accuracy-evaluation)

---

## Overview

The **Number Plate Detection System** is a web-based application that automatically detects and reads Indian vehicle registration plates from uploaded images or videos. It uses a 17-step classical image processing pipeline to locate plate regions, extract text via Tesseract OCR, and then parses the plate against a comprehensive Indian RTO database to return state, district, RTO office, vehicle class, and fuel type information.

**Key design decisions:**
- Pure classical image processing — no neural networks, no GPU dependency
- 17 distinct OpenCV algorithms applied in sequence for robust detection
- Full Indian number plate format support (Standard + BH Series + Electric)
- Covers all 28 States and 8 Union Territories with 80+ RTO office codes

---

## Architecture

![Architecture Diagram](architecture_diagram.svg)

The system is organised into four layers:

| Layer | Components |
|---|---|
| **Input** | Browser UI, Image Upload, Manual Lookup, XML Dataset |
| **Backend** | Flask Web Server with 5 route handlers (`app.py`) |
| **Core Modules** | `plate_detector.py` · `rto_database.py` · Tesseract OCR |
| **Output** | Annotated Image, Vehicle Details JSON, Pipeline Viewer, Accuracy Metrics |

---

## Features

### Image Detection
- Upload any JPG / PNG / WEBP / BMP vehicle photo
- Automatic plate region detection using Haar Cascade + Contour methods
- Full 17-step processing pipeline with intermediate image previews
- OCR extraction with multi-config Tesseract passes
- Annotated result image with bounding boxes overlaid

### Manual Lookup
- Type any Indian number plate (e.g. `MH12AB1234`) to instantly parse it
- Sample plates provided for quick testing
- Returns complete state, RTO, series, vehicle class, and fuel type

### Accuracy Evaluation
- Batch evaluation against annotated XML datasets
- Reports Accuracy, Precision, Recall, F1-Score
- Full confusion matrix (TP / TN / FP / FN) per image

---

## Tech Stack

| Component | Technology |
|---|---|
| Web Framework | Flask (Python) |
| Image Processing | OpenCV 4.x |
| Preprocessing | Gaussian Blur, Bilateral Filter, CLAHE, Histogram Equalization |
| Edge Detection | Sobel, Laplacian, Canny |
| Thresholding | Otsu, Adaptive Thresholding |
| Morphological Ops | Erosion, Dilation, Opening, Closing |
| Region Detection | Contour Analysis, Haar Cascade, Connected Components |
| Geometric Correction | Affine Transform |
| OCR Engine | Tesseract + pytesseract |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Data Serialization | JSON (custom NumPy encoder) |

---

## Project Structure

```
number_plate_project/
│
├── app.py                  # Flask backend — routes & accuracy evaluation
├── plate_detector.py       # 17-algorithm CV pipeline + OCR engine
├── rto_database.py         # Indian RTO state/district/vehicle data
├── test_system.py          # Synthetic plate test runner
│
├── templates/
│   └── index.html          # Single-page frontend UI
│
├── static/
│   ├── uploads/            # Saved input images & videos
│   └── results/            # Annotated outputs & pipeline step images
│
└── google_images/          # Labelled dataset (images + XML annotations)
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR installed on your system

#### Install Tesseract

| OS | Command |
|---|---|
| Ubuntu / Debian | `sudo apt install tesseract-ocr` |
| macOS | `brew install tesseract` |
| Windows | Download installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) |

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Payaldhaygude/IP_project.git
cd IP_project

# 2. Create a virtual environment (recommended)
python -m venv plate_env
source plate_env/bin/activate        # Linux / macOS
plate_env\Scripts\activate           # Windows

# 3. Install Python dependencies
pip install flask opencv-python pytesseract pillow numpy

# 4. Run the application
python app.py
```

Open your browser at **http://localhost:5000**

> **Windows note:** If Tesseract is not on your PATH, set the environment variable:
> `set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`

---

## Usage

### Web Interface

1. Open `http://localhost:5000`
2. Choose **Image Detection** or **Manual Lookup**
3. For image detection — upload a photo and click **DETECT NUMBER PLATE**
4. View the annotated result and the 17-step processing pipeline

### Command Line — Quick Test

```bash
python test_system.py
```

This creates a synthetic plate image and runs the full detection pipeline, printing results to the terminal.

### Command Line — Accuracy Evaluation

```bash
python app.py --accuracy
```

Place your labelled images and Pascal VOC XML annotations in the `google_images/` folder first.

---

## 17 Image Processing Algorithms

The detection pipeline applies these algorithms in sequence:

| Step | Algorithm | Purpose |
|---|---|---|
| 1 | Original Image | Raw input reference |
| 2 | Gaussian Blur | Noise removal (5×5 kernel) |
| 3 | Bilateral Filter | Edge-preserving smoothing |
| 4 | Histogram Equalization | Basic contrast enhancement |
| 5 | CLAHE | Adaptive contrast (handles uneven lighting) |
| 6 | Sobel Edge Detection | Gradient-based X/Y edge finding |
| 7 | Laplacian Edge Detection | Second-derivative, all-direction edges |
| 8 | Canny Edge Detection | Multi-stage, best for plate boundaries |
| 9 | Otsu Thresholding | Automatic global binarization |
| 10 | Adaptive Thresholding | Local-area threshold for varying light |
| 11 | Erosion | Shrinks white regions, removes noise |
| 12 | Dilation | Expands white regions, fills gaps |
| 13 | Morphological Opening | Erosion → Dilation; removes small objects |
| 14 | Morphological Closing | Dilation → Erosion; fills holes |
| 15 | Contour Detection | Finds plate-shaped rectangular contours |
| 16 | Haar Cascade | Viola-Jones detector for plate regions |
| 17 | Affine Transform | Corrects tilt / rotation |
| 18 | Connected Components | Groups character-region pixels |

---

## Indian Plate Format Support

### Standard Format
```
MH  12  AB  1234
└─┘ └─┘ └─┘ └──┘
 │   │   │    └─ Vehicle number (1–4 digits)
 │   │   └────── Series letters (1–3 chars)
 │   └────────── RTO district number
 └────────────── State code
```

### BH Series (Bharat)
```
22  BH  1234  E
└─┘     └──┘  └─ Fuel type (E=Electric, D=Diesel, P=Petrol)
 └──────────── Year of registration
```

### Supported States / UTs
All 28 States and 8 Union Territories — including Ladakh (LA), Telangana (TG/TS), and the latest BH-series registrations.

### Electric Vehicle Detection
Plates with series starting with `E` in states like MH, DL, KA, TN, GJ are automatically flagged as electric vehicles.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the main UI |
| `POST` | `/detect/image` | Detects plate in uploaded image. Body: `multipart/form-data` with `file` field |
| `POST` | `/parse` | Parses a plate string. Body: `{ "plate_text": "MH12AB1234" }` |
| `GET` | `/accuracy` | Runs batch evaluation on `google_images/` dataset |

### Sample Response — `/detect/image`

```json
{
  "success": true,
  "image_size": "1280x720",
  "total_plates": 1,
  "annotated_image_url": "/static/results/annotated_abc123.jpg",
  "plates_found": [
    {
      "ocr_text": "MH12AB1234",
      "detection_method": "haar",
      "plate_info": {
        "state": "Maharashtra",
        "rto_office": "Pune",
        "series": "AB",
        "vehicle_number": "1234",
        "vehicle_class": "Private / General",
        "is_electric": false,
        "valid": true,
        "confidence": "High"
      }
    }
  ],
  "pipeline_steps": [ "..." ]
}
```

---

## Accuracy Evaluation

To evaluate the system against a labelled dataset:

1. Place images and their Pascal VOC XML annotation files in `google_images/`
2. Each XML must follow the standard `<filename>` / `<object><name>` structure
3. Run `python app.py --accuracy` or call `GET /accuracy`

**Output metrics:**

```
Accuracy   : 0.87
Precision  : 0.91
Recall     : 0.84
F1 Score   : 0.87
TP=46  TN=12  FP=5  FN=9  Total=72
```

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## Author

**Payal Dhaygude**  
GitHub: [@Payaldhaygude](https://github.com/Payaldhaygude)

---

*Built with OpenCV, Tesseract OCR, and Flask — Classical image processing, no deep learning required.*
