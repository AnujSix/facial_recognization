from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from liveness import Challenge, ChallengeDetector, average_ear, nose_horizontal_ratio
from storage import DATA_DIR, load_profile, reset_profile, save_profile


WINDOW = "Facial Liveness Authentication"
LOG_PATH = DATA_DIR / "authentication_log.csv"
FACE_SIZE = 160


def text(frame, value, y, color=(255, 255, 255), scale=0.65):
    cv2.putText(frame, value, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, value, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def bbox_from_landmarks(points: np.ndarray, width: int, height: int):
    x1, y1 = np.maximum(points.min(axis=0) - 0.08, 0)
    x2, y2 = np.minimum(points.max(axis=0) + 0.08, 1)
    return int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)


def crop_gray(frame, box):
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (FACE_SIZE, FACE_SIZE)) if gray.size else None


def append_log(name: str, result: str, score: float | None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["timestamp_utc", "name", "result", "lbph_distance"])
        writer.writerow([datetime.now(timezone.utc).isoformat(), name, result, "" if score is None else f"{score:.2f}"])


def open_camera(camera_idx: int = 0) -> cv2.VideoCapture:
    """Robust camera opener: tries DSHOW/MSMF and scans fallback indices."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
    indices_to_try = [camera_idx]
    if camera_idx == 0:
        indices_to_try.extend([1, 2])

    for idx in indices_to_try:
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap
                cap.release()
            else:
                cap.release()

    raise SystemExit(
        f"\n[CAMERA ERROR] Could not open camera (tested index {camera_idx} and backends).\n\n"
        "If you are using a built-in laptop camera (e.g. Lenovo IdeaPad/ThinkPad, HP, Dell, ASUS):\n"
        "  1. Physical Privacy Shutter (Most Common on Lenovo):\n"
        "     Look directly above the laptop screen at the webcam lens.\n"
        "     Slide the tiny physical shutter open (make sure no orange/red dot or shutter cover is over the lens).\n"
        "     On Lenovo laptops, closing this shutter electrically disconnects the webcam!\n\n"
        "  2. Lenovo Vantage 'Camera Privacy Mode':\n"
        "     Open the 'Lenovo Vantage' app -> Device -> Display & Camera.\n"
        "     Turn OFF 'Camera Privacy Mode'.\n\n"
        "  3. Keyboard Function Key:\n"
        "     Press F8 or F10 (or Fn + F8 / Fn + F10) on your keyboard (the key with the camera icon).\n\n"
        "  4. Windows Camera Privacy Settings:\n"
        "     Open Windows Settings -> Privacy & security -> Camera.\n"
        "     Ensure 'Camera access' and 'Let desktop apps access your camera' are turned ON.\n"
    )


def enroll(camera: int, name: str):
    camera_stream = open_camera(camera)
    samples = []
    last_capture = 0.0
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    while camera_stream.isOpened() and len(samples) < 35:
        ok, frame = camera_stream.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        text(frame, f"ENROLLING: {name}", 30, (0, 220, 255))
        text(frame, "Look at the camera and slowly move your head", 58)
        text(frame, f"Samples: {len(samples)}/35   Q: quit", 86)
        if result.multi_face_landmarks:
            pts = np.array([(p.x, p.y) for p in result.multi_face_landmarks[0].landmark])
            box = bbox_from_landmarks(pts, frame.shape[1], frame.shape[0])
            crop = crop_gray(frame, box)
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
            if crop is not None and time.monotonic() - last_capture > 0.12:
                # Skip very blurry samples.
                if cv2.Laplacian(crop, cv2.CV_64F).var() > 45:
                    samples.append(crop)
                    last_capture = time.monotonic()
        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    camera_stream.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    if len(samples) >= 15:
        save_profile(name, samples)
        print(f"Enrolled {name} with {len(samples)} samples.")
    else:
        raise SystemExit("Enrollment cancelled or too few clear samples.")


def authenticate(camera: int, threshold: float):
    name, recognizer = load_profile()
    if recognizer is None:
        raise SystemExit("No profile found. Run: python app.py enroll --name 'Your Name'")

    stream = open_camera(camera)
    mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    challenges = random.sample(list(Challenge), 2)
    detector = ChallengeDetector()
    baseline_values = deque(maxlen=25)
    challenge_index = 0
    start = time.monotonic()
    distances = deque(maxlen=12)
    final_result = None
    final_since = None

    while stream.isOpened():
        ok, frame = stream.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        text(frame, "FACIAL LIVENESS AUTHENTICATION", 30, (0, 220, 255), 0.72)
        if result.multi_face_landmarks and final_result is None:
            pts = np.array([(p.x, p.y) for p in result.multi_face_landmarks[0].landmark])
            box = bbox_from_landmarks(pts, frame.shape[1], frame.shape[0])
            crop = crop_gray(frame, box)
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
            ear, nose = average_ear(pts), nose_horizontal_ratio(pts)
            if time.monotonic() - start < 1.5:
                baseline_values.append(nose)
                text(frame, "Hold still - calibrating...", 65)
            else:
                if baseline_values:
                    detector.baseline_nose_ratio = float(np.median(baseline_values))
                current = challenges[challenge_index]
                text(frame, f"Challenge {challenge_index + 1}/2: {current.value.upper()}", 65, (0, 220, 255))
                if detector.update(current, ear, nose):
                    challenge_index += 1
                if crop is not None:
                    _, distance = recognizer.predict(crop)
                    distances.append(float(distance))
                    text(frame, f"Identity distance: {distance:.1f} (lower is better)", 94)
                if challenge_index == len(challenges):
                    score = float(np.median(distances)) if distances else 999.0
                    final_result = "ACCESS GRANTED" if score <= threshold else "ACCESS DENIED - FACE MISMATCH"
                    append_log(name, final_result, score)
                    final_since = time.monotonic()
        elif final_result is None:
            text(frame, "No face detected", 65, (0, 0, 255))

        if final_result:
            granted = final_result == "ACCESS GRANTED"
            text(frame, final_result, 76, (0, 220, 0) if granted else (0, 0, 255), 0.9)
            if granted:
                text(frame, f"Identity: {name}", 110, (0, 220, 0))
            if time.monotonic() - final_since > 4:
                break
        elif time.monotonic() - start > 25:
            final_result = "ACCESS DENIED - LIVENESS TIMEOUT"
            append_log(name, final_result, None)
            final_since = time.monotonic()

        text(frame, "Q: quit", frame.shape[0] - 18, (180, 180, 180), 0.5)
        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    stream.release()
    mesh.close()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Facial liveness authentication student prototype")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    sub = parser.add_subparsers(dest="command", required=True)
    enroll_parser = sub.add_parser("enroll", help="Create or replace the local face profile")
    enroll_parser.add_argument("--name", required=True)
    auth_parser = sub.add_parser("authenticate", help="Run liveness and face authentication")
    auth_parser.add_argument("--threshold", type=float, default=70.0, help="LBPH match threshold; lower is stricter")
    sub.add_parser("reset", help="Delete the enrolled local profile")
    args = parser.parse_args()
    if args.command == "enroll":
        enroll(args.camera, args.name)
    elif args.command == "authenticate":
        authenticate(args.camera, args.threshold)
    else:
        reset_profile()
        print("Local profile removed.")


if __name__ == "__main__":
    main()

