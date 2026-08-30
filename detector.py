"""
detector.py
Core detection logic for IBVAP prototype:
 - Reads video frames
 - Runs YOLOv8 for human/vehicle detection
 - Checks a "virtual fence" zone for intrusion
 - Logs alerts to a CSV file with timestamp + snapshot

Run standalone for testing:
    python detector.py
"""

import cv2
import csv
import numpy as np
import os
import time
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

# ---------- CONFIG ----------
VIDEO_SOURCE = "test.mp4"          # change to 0 for webcam, or an RTSP url for real CCTV
MODEL_NAME = "yolov8n.pt"          # smallest/fastest YOLOv8 model, auto-downloads first run
FACE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx")
FACE_CONFIDENCE_THRESHOLD = 0.6
FACE_RECOGNITION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_recognition_sface_2021dec.onnx")
FACE_MATCH_THRESHOLD = 0.40
AUTHORIZED_PEOPLE_FILE = os.path.join(os.path.dirname(__file__), "authorized_people.json")
CONFIDENCE_THRESHOLD = 0.4
NIGHT_LUMA_THRESHOLD = 70       # mean grayscale brightness below this = low light
MOTION_PIXEL_THRESHOLD = 0.01   # fraction of changed pixels needed for movement
NIGHT_ALERT_COOLDOWN = 10       # seconds between night movement alerts
FACE_ALERT_COOLDOWN = 15
LOITERING_FRAME_THRESHOLD = 30  # ~15 seconds for the bundled 2 FPS video
ALERT_LOG_FILE = "alerts.csv"
SNAPSHOT_DIR = "snapshots"

# Virtual fence zone: a polygon of (x, y) points in frame pixel coordinates.
# NOTE: these are placeholder coordinates — adjust them to match your video's resolution.
# Easiest way to find good points: print frame.shape once and eyeball a rectangle
# over the area you want to treat as "restricted".
VIRTUAL_FENCE_ZONE = [(400, 200), (900, 200), (900, 600), (400, 600)]

# Classes we care about (COCO class ids): 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
TARGET_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
PLATE_ALERT_COOLDOWN = 8
FACE_DETECTOR = None
FACE_RECOGNIZER = None


def ensure_dirs():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    if not os.path.exists(ALERT_LOG_FILE):
        with open(ALERT_LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event_type", "object_class", "confidence", "snapshot_path"])


def log_alert(event_type, object_class, confidence, frame):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    snapshot_path = os.path.join(SNAPSHOT_DIR, f"{event_type}_{timestamp}.jpg")
    cv2.imwrite(snapshot_path, frame)
    with open(ALERT_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, event_type, object_class, f"{confidence:.2f}", snapshot_path])
    print(f"[ALERT] {event_type} | {object_class} | conf={confidence:.2f} | saved={snapshot_path}")


def point_in_zone(point, zone):
    """Check if a point lies inside the virtual fence polygon."""
    import numpy as np
    contour = np.array(zone, dtype=np.int32)
    result = cv2.pointPolygonTest(contour, point, False)
    return result >= 0


def draw_fence(frame, zone):
    import numpy as np
    pts = np.array(zone, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.putText(frame, "RESTRICTED ZONE", (zone[0][0], zone[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def detect_night_motion(frame, motion_state):
    """Return (is_low_light, motion_ratio) using brightness and frame change."""
    import numpy as np

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    is_low_light = float(np.mean(gray)) < NIGHT_LUMA_THRESHOLD

    previous = motion_state.get("previous_gray")
    motion_state["previous_gray"] = gray
    if previous is None:
        return is_low_light, 0.0

    difference = cv2.absdiff(previous, gray)
    _, changed = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    changed = cv2.dilate(changed, np.ones((5, 5), np.uint8), iterations=1)
    return is_low_light, float(np.count_nonzero(changed)) / changed.size


def detect_faces(frame):
    """Detect visible faces without identifying people, using OpenCV YuNet."""
    global FACE_DETECTOR
    if not os.path.exists(FACE_MODEL_PATH) or not hasattr(cv2, "FaceDetectorYN"):
        return []
    if FACE_DETECTOR is None:
        FACE_DETECTOR = cv2.FaceDetectorYN.create(
            FACE_MODEL_PATH, "", (320, 320), FACE_CONFIDENCE_THRESHOLD, 0.3, 5000
        )
    height, width = frame.shape[:2]
    FACE_DETECTOR.setInputSize((width, height))
    _, detections = FACE_DETECTOR.detect(frame)
    return [] if detections is None else detections


def load_authorized_people():
    """Load enrolled people; face embeddings are kept locally in this file."""
    import json
    if not os.path.exists(AUTHORIZED_PEOPLE_FILE):
        return []
    with open(AUTHORIZED_PEOPLE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_authorized_person(person):
    import json
    people = load_authorized_people()
    people = [p for p in people if p.get("person_id") != person["person_id"]]
    people.append(person)
    with open(AUTHORIZED_PEOPLE_FILE, "w", encoding="utf-8") as f:
        json.dump(people, f, indent=2)


def face_embedding(frame, detection):
    """Create an SFace embedding from a YuNet detection."""
    global FACE_RECOGNIZER
    if not os.path.exists(FACE_RECOGNITION_MODEL_PATH) or not hasattr(cv2, "FaceRecognizerSF"):
        return None
    if FACE_RECOGNIZER is None:
        FACE_RECOGNIZER = cv2.FaceRecognizerSF.create(FACE_RECOGNITION_MODEL_PATH, "")
    aligned = FACE_RECOGNIZER.alignCrop(frame, detection)
    feature = FACE_RECOGNIZER.feature(aligned)
    return feature.flatten().astype(float).tolist()


def recognize_face(frame, detection, people):
    """Return the best authorized person match, or None."""
    if not people:
        return None
    embedding = face_embedding(frame, detection)
    if embedding is None:
        return None
    best_person, best_score = None, -1.0
    for person in people:
        score = float(np.dot(embedding, person["embedding"]) /
                     ((np.linalg.norm(embedding) * np.linalg.norm(person["embedding"])) or 1.0))
        if score > best_score:
            best_person, best_score = person, score
    return best_person if best_score >= FACE_MATCH_THRESHOLD else None


def process_frame(model, frame, frame_count, alert_cooldown, motion_state=None,
                  authorized_people=None, alpr_model=None, plate_cooldown=None):
    """Runs detection on a single frame, draws boxes, checks fence, returns annotated frame."""
    if motion_state is None:
        motion_state = {}
    if plate_cooldown is None:
        plate_cooldown = {}

    is_low_light, motion_ratio = detect_night_motion(frame, motion_state)
    if is_low_light and motion_ratio >= MOTION_PIXEL_THRESHOLD:
        last_alert_time = motion_state.get("last_night_alert", 0)
        if time.time() - last_alert_time > NIGHT_ALERT_COOLDOWN:
            log_alert("NIGHT_MOVEMENT", "unknown", motion_ratio, frame)
            motion_state["last_night_alert"] = time.time()

    # Tracking gives people and vehicles stable IDs across adjacent frames.
    results = model.track(
        frame, persist=True, verbose=False, classes=list(TARGET_CLASSES)
    )[0]
    draw_fence(frame, VIRTUAL_FENCE_ZONE)

    light_label = "NIGHT / MOVEMENT" if is_low_light and motion_ratio >= MOTION_PIXEL_THRESHOLD else \
        ("NIGHT / CLEAR" if is_low_light else "DAY")
    light_color = (0, 165, 255) if is_low_light else (255, 255, 255)
    cv2.putText(frame, f"MODE: {light_label}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, light_color, 2)

    faces = detect_faces(frame)
    face_records = []
    for detection in faces:
        x, y, w, h, score = detection[:5]
        x, y, w, h = map(int, (x, y, w, h))
        person = recognize_face(frame, detection, authorized_people or [])
        face_label = f"AUTHORIZED: {person['name']}" if person else "UNKNOWN FACE"
        face_color = (0, 200, 0) if person else (0, 0, 255)
        face_records.append((x, y, w, h, person))
        cv2.rectangle(frame, (x, y), (x + w, y + h), face_color, 2)
        cv2.putText(frame, f"{face_label} {score:.2f}", (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2)
    unknown_faces = [record for record in face_records if record[4] is None]
    if unknown_faces and time.time() - motion_state.get("last_face_alert", 0) > FACE_ALERT_COOLDOWN:
        log_alert("UNAUTHORIZED_FACE" if authorized_people else "FACE_DETECTED", "face", 1.0, frame)
        motion_state["last_face_alert"] = time.time()

    tracks = motion_state.setdefault("tracks", defaultdict(dict))
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id not in TARGET_CLASSES or conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = TARGET_CLASSES[cls_id]
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        track_id = int(box.id[0]) if box.id is not None else None
        display_label = f"{label} #{track_id}" if track_id is not None else label

        # Default box color: green
        color = (0, 255, 0)
        for fx, fy, fw, fh, matched_person in face_records:
            if fx <= center[0] <= fx + fw and fy <= center[1] <= fy + fh:
                color = (0, 200, 0) if matched_person else (0, 0, 255)
                break

        # Check virtual fence intrusion
        if point_in_zone(center, VIRTUAL_FENCE_ZONE):
            color = (0, 0, 255)  # red = intrusion
            cooldown_key = f"{label}_{track_id or cls_id}"
            last_alert_time = alert_cooldown.get(cooldown_key, 0)
            if time.time() - last_alert_time > 5:  # avoid spamming alerts every frame
                log_alert("INTRUSION", label, conf, frame)
                alert_cooldown[cooldown_key] = time.time()

            if label == "person" and track_id is not None:
                track = tracks[track_id]
                track.setdefault("first_zone_frame", frame_count)
                if frame_count - track["first_zone_frame"] >= LOITERING_FRAME_THRESHOLD:
                    last_suspicious = track.get("last_suspicious", 0)
                    if time.time() - last_suspicious > NIGHT_ALERT_COOLDOWN:
                        log_alert("SUSPICIOUS_ACTIVITY", "person", conf, frame)
                        track["last_suspicious"] = time.time()
        elif track_id is not None:
            tracks.pop(track_id, None)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{display_label} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, center, 4, color, -1)

        if alpr_model is not None and label in VEHICLE_CLASSES:
            from alpr import detect_plates_in_region, draw_plates
            plates = detect_plates_in_region(alpr_model, frame, (x1, y1, x2, y2))
            draw_plates(frame, plates)
            for plate in plates:
                last_plate_time = plate_cooldown.get(plate["text"], 0)
                if time.time() - last_plate_time > PLATE_ALERT_COOLDOWN:
                    log_alert("PLATE_DETECTED", plate["text"], plate["ocr_confidence"], frame)
                    plate_cooldown[plate["text"]] = time.time()

    return frame


def run():
    ensure_dirs()
    print("Loading YOLOv8 model (first run downloads weights)...")
    model = YOLO(MODEL_NAME)

    print("Loading ALPR model (first run downloads weights)...")
    from alpr import get_alpr
    alpr_model = get_alpr()

    authorized_people = load_authorized_people()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"ERROR: could not open video source '{VIDEO_SOURCE}'")
        return

    frame_count = 0
    alert_cooldown = {}
    motion_state = {}
    plate_cooldown = {}
    print("Press 'q' to quit the preview window.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        frame_count += 1
        if frame_count % 3 != 0:   # skip frames to keep things fast (process ~1 of every 3)
            continue

        annotated = process_frame(model, frame, frame_count, alert_cooldown, motion_state,
                                  authorized_people, alpr_model, plate_cooldown)
        cv2.imshow("IBVAP Prototype - Border Video Analytics", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
