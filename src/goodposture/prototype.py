"""Local webcam feasibility viewer for pose landmarks and initial metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from goodposture.adapters.landmark_mapping import DetectedPose
from goodposture.adapters.mediapipe_pose import MediaPipePoseLandmarker
from goodposture.core.metrics import extract_posture_metrics
from goodposture.core.models import Landmark, MetricReading, PostureMetrics

_WINDOW_TITLE = "GoodPosture - Local Feasibility Viewer"
_LANDMARK_CONFIDENCE = 0.5

# MediaPipe's canonical 33-landmark pose connections, represented as indices so
# the viewer remains independent of MediaPipe result classes.
_POSE_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)


@dataclass(frozen=True, slots=True)
class PrototypeOptions:
    model_path: Path
    camera_index: int = 0
    mirror: bool = True


def _pixel(landmark: Landmark, width: int, height: int) -> tuple[int, int]:
    return (
        max(0, min(width - 1, round(landmark.x * width))),
        max(0, min(height - 1, round(landmark.y * height))),
    )


def _draw_pose(frame: Any, detected: DetectedPose) -> None:
    height, width = frame.shape[:2]
    landmarks = detected.all_landmarks

    for start_index, end_index in _POSE_CONNECTIONS:
        if start_index >= len(landmarks) or end_index >= len(landmarks):
            continue
        start = landmarks[start_index]
        end = landmarks[end_index]
        if min(start.confidence, end.confidence) < _LANDMARK_CONFIDENCE:
            continue
        cv2.line(
            frame,
            _pixel(start, width, height),
            _pixel(end, width, height),
            (80, 210, 120),
            2,
            cv2.LINE_AA,
        )

    for landmark in landmarks:
        if landmark.confidence >= _LANDMARK_CONFIDENCE:
            cv2.circle(
                frame,
                _pixel(landmark, width, height),
                3,
                (40, 235, 255),
                -1,
                cv2.LINE_AA,
            )


def _metric_text(label: str, metric: MetricReading, *, suffix: str = "") -> str:
    if metric.value is None:
        return f"{label}: unknown"
    return f"{label}: {metric.value:+.2f}{suffix}  conf {metric.confidence:.2f}"


def _draw_status(frame: Any, metrics: PostureMetrics | None) -> None:
    lines = [
        "LOCAL ONLY | Frames are processed in memory and not saved",
        "Posture awareness prototype - not medical guidance | Q or Esc exits",
    ]
    if metrics is None:
        lines.append("Pose metrics: unknown (move upper body into view)")
    else:
        lines.extend(
            [
                _metric_text("Shoulder tilt", metrics.shoulder_tilt_degrees, suffix=" deg"),
                _metric_text("Torso lean", metrics.torso_lean_degrees, suffix=" deg"),
                _metric_text("Head lateral", metrics.head_lateral_offset_ratio),
                _metric_text("Head vertical", metrics.head_vertical_offset_ratio),
                _metric_text("Head depth", metrics.head_depth_ratio),
            ]
        )

    overlay = frame.copy()
    box_height = 18 + (len(lines) * 25)
    cv2.rectangle(overlay, (8, 8), (min(frame.shape[1] - 8, 720), box_height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    for index, line in enumerate(lines):
        color = (210, 240, 210) if index == 0 else (235, 235, 235)
        cv2.putText(
            frame,
            line,
            (18, 32 + (index * 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )


def run_prototype(options: PrototypeOptions) -> None:
    """Run the blocking local viewer until the user exits."""

    capture = cv2.VideoCapture(options.camera_index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open camera {options.camera_index}. Check Windows camera permissions."
        )

    failed_reads = 0
    try:
        with MediaPipePoseLandmarker(options.model_path) as landmarker:
            while True:
                ok, frame = capture.read()
                if not ok:
                    failed_reads += 1
                    if failed_reads >= 30:
                        raise RuntimeError("Camera stopped returning frames.")
                    continue
                failed_reads = 0

                if options.mirror:
                    frame = cv2.flip(frame, 1)

                timestamp_ms = time.monotonic_ns() // 1_000_000
                detected = landmarker.detect(frame, timestamp_ms=timestamp_ms)
                metrics = None
                if detected is not None:
                    _draw_pose(frame, detected)
                    metrics = extract_posture_metrics(detected.observation)
                _draw_status(frame, metrics)

                cv2.imshow(_WINDOW_TITLE, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
