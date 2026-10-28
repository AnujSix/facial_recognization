from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_PATH = DATA_DIR / "face_model.yml"
PROFILE_PATH = DATA_DIR / "profile.json"


def save_profile(name: str, face_crops: list[np.ndarray]) -> None:
    if len(face_crops) < 15:
        raise ValueError("At least 15 clear face samples are required.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    normalized = [cv2.resize(crop, (160, 160)) for crop in face_crops]
    recognizer.train(normalized, np.zeros(len(normalized), dtype=np.int32))
    recognizer.write(str(MODEL_PATH))
    PROFILE_PATH.write_text(json.dumps({"name": name.strip() or "User"}, indent=2), encoding="utf-8")


def load_profile():
    if not MODEL_PATH.exists() or not PROFILE_PATH.exists():
        return None, None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    name = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))["name"]
    return name, recognizer


def reset_profile() -> None:
    MODEL_PATH.unlink(missing_ok=True)
    PROFILE_PATH.unlink(missing_ok=True)

