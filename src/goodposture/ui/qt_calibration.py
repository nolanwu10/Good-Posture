"""Launch the local PySide6 calibration experience."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from goodposture.adapters.runtime import PoseAdapter, PoseModelSpec
from goodposture.app import AnalysisSession
from goodposture.ui.calibration_flow import CalibrationFlow
from goodposture.ui.calibration_window import CalibrationWindow
from goodposture.ui.camera_worker import CameraBackend

_MODEL_SHA256 = "59929E1D1EE95287735DDD833B19CF4AC46D29BC7AFDDBBF6753C459690D574A"
_MODEL_ID = "pose_landmarker_lite@sha256:59929e1d"


def run_calibration_ui(model_path: Path) -> int:
    """Run first-use setup until the user finishes or leaves."""

    application = QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QApplication(sys.argv)
    assert isinstance(application, QApplication)
    application.setApplicationName("GoodPosture")
    application.setOrganizationName("GoodPosture")

    model = PoseModelSpec(path=model_path, model_id=_MODEL_ID, sha256=_MODEL_SHA256)
    backend = CameraBackend(PoseAdapter(model=model))
    flow = CalibrationFlow(session=AnalysisSession(model_id=model.model_id))
    window = CalibrationWindow(flow=flow, backend=backend)
    window.show()
    if owns_application:
        return application.exec()
    return 0
