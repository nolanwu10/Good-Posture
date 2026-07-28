"""Camera-free validation for a frozen Windows package."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def run_package_smoke(model_path: Path) -> int:
    """Load Qt and the pinned native model without opening a camera."""

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication

        application = QApplication.instance() or QApplication([])
        # Load Qt before OpenCV/MediaPipe so their native dependency search does
        # not preempt the bundled PySide6 runtime in a frozen process.
        from goodposture.adapters.mediapipe_pose import MediaPipePoseLandmarker

        with MediaPipePoseLandmarker(model_path):
            application.processEvents()
    except (FileNotFoundError, RuntimeError, OSError):
        print("GoodPosture package smoke test failed.", file=sys.stderr)
        return 2
    return 0
