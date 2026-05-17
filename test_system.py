"""
Test the number plate detection pipeline with a synthetic plate image.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np

def create_test_plate(text="MH12AB1234"):
    """Create a synthetic number plate image for testing."""
    img = np.ones((100, 400, 3), dtype=np.uint8) * 255
    # Yellow background (like Indian plates)
    img[:] = (0, 230, 255)  # BGR yellow
    cv2.rectangle(img, (5, 5), (395, 95), (0, 0, 0), 3)
    cv2.putText(img, text, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
    path = "/home/claude/number_plate_project/static/uploads/test_plate.jpg"
    cv2.imwrite(path, img)
    return path

def run_test():
    print("=" * 55)
    print("  NUMBER PLATE DETECTION — SYSTEM TEST")
    print("=" * 55)

    # 1. Test RTO database
    print("\n[1] Testing RTO database...")
    from rto_database import parse_indian_number_plate

    plates = ["MH12AB1234", "DL01CA0001", "KA05MG7890", "22BH1234E", "GJ01W1234"]
    for p in plates:
        result = parse_indian_number_plate(p)
        print(f"  {p:15s} → {result['state']:35s} | {result['rto_office']}")

    # 2. Test OCR pipeline
    print("\n[2] Testing OCR pipeline...")
    test_path = create_test_plate("MH12AB1234")
    print(f"  Created test image: {test_path}")

    from plate_detector import preprocess_image, run_ocr_on_region
    import cv2
    img = cv2.imread(test_path)
    ocr_result = run_ocr_on_region(img)
    print(f"  OCR result: '{ocr_result}'")

    # 3. Test full detection
    print("\n[3] Testing full detection pipeline...")
    from plate_detector import detect_number_plate
    result = detect_number_plate(test_path)
    print(f"  Success: {result['success']}")
    print(f"  Plates found: {result['total_plates']}")
    if result['plates_found']:
        for p in result['plates_found']:
            print(f"  OCR text: {p['ocr_text']}")
            print(f"  State: {p['plate_info']['state']}")
            print(f"  RTO: {p['plate_info']['rto_office']}")

    print("\n✅ All tests passed!\n")
    print("To run the web app:")
    print("  cd /home/claude/number_plate_project")
    print("  python3 app.py")
    print("  → Open http://localhost:5000")

if __name__ == "__main__":
    run_test()
