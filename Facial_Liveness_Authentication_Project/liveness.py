from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)


class Challenge(str, Enum):
    BLINK = "blink"
    TURN_LEFT = "turn left"
    TURN_RIGHT = "turn right"


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def eye_aspect_ratio(points: np.ndarray, indexes: Sequence[int]) -> float:
    p1, p2, p3, p4, p5, p6 = (points[i] for i in indexes)
    horizontal = 2.0 * _distance(p1, p4)
    if horizontal < 1e-6:
        return 0.0
    return (_distance(p2, p6) + _distance(p3, p5)) / horizontal


def average_ear(points: np.ndarray) -> float:
    return (eye_aspect_ratio(points, LEFT_EYE) + eye_aspect_ratio(points, RIGHT_EYE)) / 2.0


def nose_horizontal_ratio(points: np.ndarray) -> float:
    """Nose x-position normalized between the visible face sides.

    A frontal value is near 0.5. The direction mapping is deliberately calibrated
    by the UI because webcam mirroring differs between systems.
    """
    left_x = float(points[234][0])
    right_x = float(points[454][0])
    width = right_x - left_x
    if abs(width) < 1e-6:
        return 0.5
    return (float(points[1][0]) - left_x) / width


@dataclass
class ChallengeDetector:
    blink_threshold: float = 0.20
    turn_delta: float = 0.085
    required_frames: int = 3
    baseline_nose_ratio: float = 0.5
    closed_frames: int = 0
    direction_frames: int = 0
    blink_was_closed: bool = False

    def reset(self) -> None:
        self.closed_frames = 0
        self.direction_frames = 0
        self.blink_was_closed = False

    def update(self, challenge: Challenge, ear: float, nose_ratio: float) -> bool:
        if challenge == Challenge.BLINK:
            if ear < self.blink_threshold:
                self.closed_frames += 1
                self.blink_was_closed |= self.closed_frames >= self.required_frames
                return False
            passed = self.blink_was_closed and ear > self.blink_threshold + 0.025
            if passed:
                self.reset()
            else:
                self.closed_frames = 0
            return passed

        # The preview is mirrored. Moving the user's face left normally moves the
        # detected nose ratio toward the corresponding side in the mirrored image.
        delta = nose_ratio - self.baseline_nose_ratio
        correct = delta < -self.turn_delta if challenge == Challenge.TURN_LEFT else delta > self.turn_delta
        self.direction_frames = self.direction_frames + 1 if correct else 0
        passed = self.direction_frames >= self.required_frames
        if passed:
            self.reset()
        return passed

