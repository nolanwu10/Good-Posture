"""On-device MediaPipe Pose Landmarker adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from goodposture.adapters.landmark_mapping import DetectedPose, map_pose_landmarks


class MediaPipePoseLandmarker:
    """Synchronous video-mode landmarker for a bounded webcam loop."""

    def __init__(
        self,
        model_path: Path,
        *,
        minimum_detection_confidence: float = 0.5,
        minimum_presence_confidence: float = 0.5,
        minimum_tracking_confidence: float = 0.5,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Pose model not found: {model_path}. Run scripts/download_model.ps1 first."
            )

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=minimum_detection_confidence,
            min_pose_presence_confidence=minimum_presence_confidence,
            min_tracking_confidence=minimum_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker: Any = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame_bgr: Any, *, timestamp_ms: int) -> DetectedPose | None:
        """Process one in-memory BGR frame without retaining or writing it."""

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return None
        return map_pose_landmarks(result.pose_landmarks[0], timestamp_ms=timestamp_ms)

    def close(self) -> None:
        """Release native model resources."""

        self._landmarker.close()

    def __enter__(self) -> MediaPipePoseLandmarker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
