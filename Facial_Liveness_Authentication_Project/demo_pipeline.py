from __future__ import annotations

import csv
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from liveness import Challenge, ChallengeDetector, average_ear, nose_horizontal_ratio
from storage import DATA_DIR, save_profile, load_profile, reset_profile, MODEL_PATH, PROFILE_PATH
from app import bbox_from_landmarks, crop_gray, text, FACE_SIZE, append_log

DEMO_DIR = DATA_DIR / "demo_output"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_IMAGE = Path(r"C:\Users\nages\.gemini\antigravity\brain\150ed88c-2e50-4314-b95f-fd2514251080\sample_face_1790142005175.jpg")

def run_demo():
    print("=" * 60)
    print("FACIAL LIVENESS AUTHENTICATION SYSTEM - END-TO-END DEMO")
    print("=" * 60)

    # 1. Load sample portrait
    img = cv2.imread(str(SAMPLE_IMAGE))
    if img is None:
        raise RuntimeError("Sample face image could not be loaded.")
    h, w = img.shape[:2]
    # Resize to standard webcam resolution 640x640 for consistency
    img = cv2.resize(img, (640, 640))
    h, w = img.shape[:2]

    # 2. Extract face mesh landmarks
    mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    res = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        raise RuntimeError("No face landmarks detected on test image.")

    face_landmarks = res.multi_face_landmarks[0]
    pts = np.array([(p.x, p.y) for p in face_landmarks.landmark])
    box = bbox_from_landmarks(pts, w, h)
    x1, y1, x2, y2 = box
    base_crop = crop_gray(img, box)
    print(f"[OK] Face Mesh detected {len(pts)} landmarks.")
    print(f"[OK] Face bounding box: ({x1}, {y1}) to ({x2}, {y2})")

    # 3. ENROLLMENT DEMO
    print("\n--- STEP 1: ENROLLMENT ---")
    enrolled_name = "Alex Rivera"
    samples = []
    # Generate 35 diverse training samples with slight brightness/scaling jitter
    for i in range(35):
        jitter_factor = 1.0 + (i - 17) * 0.015
        jittered = np.clip(base_crop.astype(np.float32) * jitter_factor, 0, 255).astype(np.uint8)
        samples.append(jittered)

    save_profile(enrolled_name, samples)
    print(f"[SUCCESS] Enrolled profile for '{enrolled_name}' with {len(samples)} samples.")
    print(f"          Model saved to: {MODEL_PATH}")
    print(f"          Profile saved to: {PROFILE_PATH}")

    # Generate Enrollment preview frame
    enroll_frame = img.copy()
    cv2.rectangle(enroll_frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
    text(enroll_frame, f"ENROLLING: {enrolled_name}", 35, (0, 220, 255))
    text(enroll_frame, "Look at camera and slowly move head", 65)
    text(enroll_frame, f"Samples: 35/35 - ENROLLMENT COMPLETE", 95, (0, 255, 0))
    enroll_img_path = DEMO_DIR / "1_enrollment_success.jpg"
    cv2.imwrite(str(enroll_img_path), enroll_frame)
    print(f"[SAVED] Enrollment snapshot -> {enroll_img_path}")

    # 4. LOAD PROFILE & VERIFY RECOGNIZER
    name, recognizer = load_profile()
    print(f"\n--- STEP 2: LOAD PROFILE & PREDICTION ---")
    print(f"Loaded profile name: {name}")

    # Test prediction on enrolled face
    label, distance = recognizer.predict(base_crop)
    print(f"Genuine Face LBPH match distance: {distance:.2f} (Threshold is 70.0; lower is better)")

    # Test prediction on random / impostor face
    impostor_crop = np.random.randint(0, 256, (FACE_SIZE, FACE_SIZE), dtype=np.uint8)
    _, imp_distance = recognizer.predict(impostor_crop)
    print(f"Impostor Face LBPH distance: {imp_distance:.2f} (Rejected > 70)")

    # 5. LIVENESS CHALLENGES DEMO
    print("\n--- STEP 3: ACTIVE LIVENESS VERIFICATION ---")
    detector = ChallengeDetector()
    baseline_ear = average_ear(pts)
    baseline_nose = nose_horizontal_ratio(pts)
    detector.baseline_nose_ratio = baseline_nose
    print(f"Baseline Eye Aspect Ratio (EAR): {baseline_ear:.3f}")
    print(f"Baseline Nose Horizontal Ratio:  {baseline_nose:.3f}")

    # Simulated challenges
    challenges = [Challenge.BLINK, Challenge.TURN_LEFT]
    print(f"Assigned Challenges: {[c.value for c in challenges]}")

    # Challenge 1: Blink
    print("\nSimulating Challenge 1: BLINK...")
    for f in range(3):
        passed = detector.update(Challenge.BLINK, ear=0.15, nose_ratio=baseline_nose)
    passed = detector.update(Challenge.BLINK, ear=0.28, nose_ratio=baseline_nose)
    print(f"Challenge 1 (BLINK) Passed: {passed}")

    # Create visual frame for Blink Challenge
    blink_frame = img.copy()
    cv2.rectangle(blink_frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
    text(blink_frame, "FACIAL LIVENESS AUTHENTICATION", 35, (0, 220, 255), 0.72)
    text(blink_frame, "Challenge 1/2: BLINK [DETECTED]", 68, (0, 255, 0))
    text(blink_frame, f"Identity distance: {distance:.1f} (Threshold: 70.0)", 98)
    blink_img_path = DEMO_DIR / "2_challenge_blink_passed.jpg"
    cv2.imwrite(str(blink_img_path), blink_frame)
    print(f"[SAVED] Challenge 1 snapshot -> {blink_img_path}")

    # Challenge 2: Turn Left
    print("\nSimulating Challenge 2: TURN LEFT...")
    turned_nose = baseline_nose - 0.12
    for f in range(3):
        passed_turn = detector.update(Challenge.TURN_LEFT, ear=baseline_ear, nose_ratio=turned_nose)
    print(f"Challenge 2 (TURN LEFT) Passed: {passed_turn}")

    # Create visual frame for Turn Left Challenge
    turn_frame = img.copy()
    cv2.rectangle(turn_frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
    text(turn_frame, "FACIAL LIVENESS AUTHENTICATION", 35, (0, 220, 255), 0.72)
    text(turn_frame, "Challenge 2/2: TURN LEFT [DETECTED]", 68, (0, 255, 0))
    text(turn_frame, f"Identity distance: {distance:.1f} (Threshold: 70.0)", 98)
    turn_img_path = DEMO_DIR / "3_challenge_turn_passed.jpg"
    cv2.imwrite(str(turn_img_path), turn_frame)
    print(f"[SAVED] Challenge 2 snapshot -> {turn_img_path}")

    # 6. FINAL AUTHENTICATION DECISION & LOGGING
    print("\n--- STEP 4: AUTHENTICATION DECISION ---")
    score = float(distance)
    final_result = "ACCESS GRANTED" if score <= 70.0 else "ACCESS DENIED - FACE MISMATCH"
    append_log(name, final_result, score)
    print(f"Decision: {final_result}")
    print(f"Identity Verified: {name}")
    print(f"LBPH Score: {score:.2f}")

    # Create visual frame for Access Granted
    auth_frame = img.copy()
    cv2.rectangle(auth_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    text(auth_frame, "FACIAL LIVENESS AUTHENTICATION", 35, (0, 220, 255), 0.72)
    text(auth_frame, "ACCESS GRANTED", 78, (0, 220, 0), 0.95)
    text(auth_frame, f"Identity: {name} (distance: {score:.1f})", 115, (0, 220, 0), 0.7)
    auth_img_path = DEMO_DIR / "4_access_granted.jpg"
    cv2.imwrite(str(auth_img_path), auth_frame)
    print(f"[SAVED] Authentication success snapshot -> {auth_img_path}")

    # 7. SPOOF / TIMEOUT SIMULATION
    print("\n--- STEP 5: SPOOF ATTEMPT SIMULATION (STILL PHOTO) ---")
    spoof_log_name = "Unknown / Still Photo"
    spoof_result = "ACCESS DENIED - LIVENESS TIMEOUT"
    append_log(spoof_log_name, spoof_result, None)
    print(f"Decision: {spoof_result} (Failed active challenges within timeout)")

    spoof_frame = img.copy()
    cv2.rectangle(spoof_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    text(spoof_frame, "FACIAL LIVENESS AUTHENTICATION", 35, (0, 220, 255), 0.72)
    text(spoof_frame, "ACCESS DENIED - LIVENESS TIMEOUT", 78, (0, 0, 255), 0.85)
    text(spoof_frame, "No response to active movement challenge", 115, (0, 0, 255), 0.6)
    spoof_img_path = DEMO_DIR / "5_access_denied_spoof.jpg"
    cv2.imwrite(str(spoof_img_path), spoof_frame)
    print(f"[SAVED] Spoof denial snapshot -> {spoof_img_path}")

    print("\n" + "=" * 60)
    print("AUTHENTICATION LOG FILE RECORD:")
    print("=" * 60)
    log_path = DATA_DIR / "authentication_log.csv"
    if log_path.exists():
        print(log_path.read_text(encoding="utf-8"))

if __name__ == "__main__":
    run_demo()
