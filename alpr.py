"""
alpr.py

Automatic License Plate Recognition (ALPR) module for IBVAP.

Wraps the `fast-alpr` library (https://github.com/ankandrew/fast-alpr) so
detector.py can call plate recognition on vehicle detections without
needing to know the library's API directly.

Usage (inside detector.py):
    from alpr import get_alpr, detect_plates_in_region, draw_plates

    alpr_model = get_alpr()
    plates = detect_plates_in_region(alpr_model, frame, vehicle_box)
    draw_plates(frame, plates)
"""

import statistics

import cv2
from fast_alpr import ALPR

# ---------- CONFIG ----------
PLATE_CONFIDENCE_THRESHOLD = 0.5   # minimum OCR confidence to trust a plate reading
PLATE_ALERT_COOLDOWN = 8           # seconds before the same plate text can alert again

_alpr_instance = None


def get_alpr():
    """Returns a cached ALPR model instance (loads once, reused across frames)."""
    global _alpr_instance
    if _alpr_instance is None:
        print("Loading FastALPR model (first run downloads weights)...")
        _alpr_instance = ALPR()
    return _alpr_instance


def detect_plates(alpr_model, frame):
    """
    Runs plate detection + OCR on a full frame (or a cropped region passed
    in as `frame`).

    Returns a list of dicts:
        {
            "text": str,                 # recognized plate text
            "ocr_confidence": float,
            "detection_confidence": float,
            "box": (x1, y1, x2, y2),     # coordinates within the input frame/crop
        }
    Only results with OCR confidence >= PLATE_CONFIDENCE_THRESHOLD are returned.
    """
    raw_results = alpr_model.predict(frame)
    plates = []

    for result in raw_results:
        # fast-alpr returns a per-character confidence list (not a single
        # number), so average across characters to get one plate-level score.
        raw_confidence = result.ocr.confidence
        if raw_confidence is None:
            continue
        ocr_confidence = (
            statistics.mean(raw_confidence)
            if isinstance(raw_confidence, list)
            else float(raw_confidence)
        )
        if ocr_confidence < PLATE_CONFIDENCE_THRESHOLD:
            continue

        box = result.detection.bounding_box
        plates.append({
            "text": result.ocr.text,
            "ocr_confidence": ocr_confidence,
            "detection_confidence": float(result.detection.confidence),
            "box": (box.x1, box.y1, box.x2, box.y2),
        })

    return plates


def detect_plates_in_region(alpr_model, frame, region_box):
    """
    Runs ALPR only on a cropped sub-region of the frame (e.g. a vehicle's
    YOLO bounding box), then translates plate coordinates back to
    full-frame coordinates. Much cheaper than scanning the whole frame
    on every vehicle detection.
    """
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = region_box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return []

    plates = detect_plates(alpr_model, crop)
    for plate in plates:
        px1, py1, px2, py2 = plate["box"]
        plate["box"] = (px1 + x1, py1 + y1, px2 + x1, py2 + y1)

    return plates


def draw_plates(frame, plates):
    """Draws plate bounding boxes + recognized text onto the frame in place."""
    for plate in plates:
        x1, y1, x2, y2 = plate["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
        text_y = max(20, y1 - 10)
        cv2.putText(
            frame, plate["text"], (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA,
        )
