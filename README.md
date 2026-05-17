# 🚗 Number Plate Detection System
### Indian Vehicle Number Plate Recognition using Image Processing

---

## Project Overview

This project detects Indian vehicle number plates from **images** and **videos** using classical image processing techniques (OpenCV) and OCR (Tesseract). It then provides complete vehicle information from the plate using an Indian RTO database.

---

## Dataset Structure (Your Files)

| Folder | Purpose |
|--------|---------|
| `google_images/` | Real-world plate images scraped from Google |
| `State-wise_OLX/` | OLX vehicle listing images, state-wise |
| `video_images/` | Frames or video clips of vehicles |

Place your dataset images in `static/uploads/` for batch testing, or use the web UI to upload one at a time.

---

## Technology Stack

| Component | Library |
|-----------|---------|
| Image Processing | OpenCV 4.x |
| Number Plate OCR | Tesseract 5 + pytesseract |
| Plate Detection | Haar Cascades + Contour Detection |
| Web Backend | Flask (Python) |
| Frontend | HTML5 + CSS3 + Vanilla JS |
| Database | Python dictionary (RTO codes) |

---

## Installation

```bash
# Install dependencies
pip install flask opencv-python pytesseract pillow numpy

# Install Tesseract OCR engine (Ubuntu/Debian)
sudo apt install tesseract-ocr

# Run the application
python3 app.py
# Open http://localhost:5000
```

---

## Image Processing Pipeline

```
Input Image/Video
      │
      ▼
  Preprocessing
  ┌─────────────────────────────────────┐
  │  • Resize to optimal dimensions     │
  │  • Bilateral Filter (noise removal) │
  │  • CLAHE (contrast enhancement)     │
  │  • Otsu / Adaptive Thresholding     │
  │  • Morphological operations         │
  └─────────────────────────────────────┘
      │
      ▼
  Plate Region Detection
  ┌─────────────────────────────────────┐
  │  Method 1: Haar Cascade             │
  │  • haarcascade_russian_plate        │
  │  • haarcascade_license_plate_rus    │
  │                                     │
  │  Method 2: Contour Detection        │
  │  • Canny edge detection             │
  │  • Contour approximation (4-sided)  │
  │  • Aspect ratio filtering (1.5–6.0) │
  └─────────────────────────────────────┘
      │
      ▼
  OCR (Tesseract)
  ┌─────────────────────────────────────┐
  │  • Multiple PSM configs (6,7,8,13)  │
  │  • Whitelist: A-Z, 0-9 only         │
  │  • Best result by scoring           │
  └─────────────────────────────────────┘
      │
      ▼
  RTO Database Lookup
  ┌─────────────────────────────────────┐
  │  • State code → State name          │
  │  • RTO code → Office location       │
  │  • BH Series detection              │
  │  • Electric vehicle detection       │
  │  • Vehicle class estimation         │
  └─────────────────────────────────────┘
      │
      ▼
  Results + Annotated Image
```

---

## Supported Plate Formats

| Format | Example | Description |
|--------|---------|-------------|
| Standard | MH 12 AB 1234 | State + RTO + Series + Number |
| Electric | MH 12 EA 1234 | Series starts with E |
| BH Series | 22 BH 1234 E | Year + BH + Number + Fuel |
| Partial | MH 12 __ ____ | Partial detection |

---

## State Codes Supported

All 28 States + 8 Union Territories: MH, DL, KA, TN, GJ, UP, HR, PB, RJ, MP, WB, BR, AP, TG, KL, OR, AS, JH, CG, HP, and more.

---

## File Structure

```
number_plate_project/
│
├── app.py                  # Flask web application
├── plate_detector.py       # Core detection engine (OpenCV + OCR)
├── rto_database.py         # Indian RTO / state database
├── test_system.py          # Test script
├── README.md               # This file
│
├── templates/
│   └── index.html          # Web UI
│
└── static/
    ├── uploads/            # Uploaded images/videos
    └── results/            # Annotated output images
```

---

## Future Improvements

- [ ] Train custom YOLO model on Indian plates for better detection
- [ ] Add LPR (License Plate Recognition) with deep learning
- [ ] Connect to VAHAN government database API for real ownership data
- [ ] Add batch processing for entire dataset folders
- [ ] Export results to CSV/Excel
- [ ] Add vehicle color detection using color histograms
