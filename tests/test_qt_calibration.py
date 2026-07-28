from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from goodposture.adapters.runtime import FramePreview
from goodposture.app import AnalysisSession
from goodposture.core.calibration import (
    CALIBRATION_CONFIG_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    CalibrationBaseline,
    CalibrationFeature,
)
from goodposture.ui.calibration_flow import CalibrationFlow, CalibrationUiState
from goodposture.ui.calibration_window import CalibrationWindow


class FakeBackend(QObject):
    cameras_ready = Signal(tuple)
    result_ready = Signal(object)
    preview_ready = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.discover_count = 0
        self.started_camera: int | None = None
        self.pause_count = 0
        self.shutdown_count = 0
        self.preview_enabled: list[bool] = []
        self.preview_consumed_count = 0

    def discover_cameras(self) -> None:
        self.discover_count += 1

    def start_camera(self, camera_index: int) -> None:
        self.started_camera = camera_index

    def pause(self) -> None:
        self.pause_count += 1

    def shutdown(self) -> None:
        self.shutdown_count += 1

    def set_preview_enabled(self, enabled: bool) -> None:
        self.preview_enabled.append(enabled)

    def preview_consumed(self) -> None:
        self.preview_consumed_count += 1


def application() -> QApplication:
    instance = QApplication.instance()
    if isinstance(instance, QApplication):
        return instance
    return QApplication([])


def window() -> tuple[CalibrationWindow, FakeBackend]:
    application()
    flow = CalibrationFlow(session=AnalysisSession(model_id="pose-test-v1"))
    backend = FakeBackend()
    return CalibrationWindow(flow=flow, backend=backend), backend


def saved_baseline() -> CalibrationBaseline:
    return CalibrationBaseline(
        calibration_schema_version=CALIBRATION_SCHEMA_VERSION,
        calibration_config_version=CALIBRATION_CONFIG_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_id="pose-test-v1",
        started_at_ms=100,
        completed_at_ms=200,
        accepted_sample_count=60,
        rejected_sample_count=0,
        features=(
            CalibrationFeature("shoulder_tilt_degrees", 0.0, 1.0),
            CalibrationFeature("head_lateral_offset_ratio", 0.0, 0.02),
            CalibrationFeature("head_vertical_offset_ratio", 0.73, 0.02),
            CalibrationFeature("head_depth_ratio", 0.60, 0.02),
        ),
    )


def test_window_renders_accessible_privacy_and_leave_actions() -> None:
    calibration_window, backend = window()

    assert calibration_window.flow.view.state is CalibrationUiState.PRIVACY
    assert calibration_window.windowTitle() == "GoodPosture setup"
    assert calibration_window.primary_button.text() == "Continue"
    assert calibration_window.primary_button.accessibleName() == "Continue"
    assert calibration_window.secondary_buttons[0].text() == "Leave"

    calibration_window.close()
    assert backend.shutdown_count == 1


def test_consent_discovers_cameras_and_selection_starts_chosen_camera() -> None:
    calibration_window, backend = window()

    calibration_window.primary_button.click()

    assert calibration_window.camera_combo.count() == 1
    assert calibration_window.camera_combo.currentText() == "Searching for local cameras…"
    assert calibration_window.primary_button.isEnabled() is False

    backend.cameras_ready.emit((0, 2))

    assert calibration_window.flow.view.state is CalibrationUiState.CAMERA_SELECTION
    assert calibration_window.camera_combo.count() == 2
    assert calibration_window.camera_combo.itemText(0) == "Camera source 1 (Windows index 0)"
    assert calibration_window.primary_button.isEnabled() is True
    calibration_window.camera_combo.setCurrentIndex(1)
    calibration_window.primary_button.click()

    assert backend.started_camera == 2
    calibration_window.close()


def test_framing_can_release_camera_and_return_to_selection() -> None:
    calibration_window, backend = window()
    calibration_window.primary_button.click()
    backend.cameras_ready.emit((0, 2))
    calibration_window.primary_button.click()
    calibration_window.flow.camera_ready()
    calibration_window._render()

    assert calibration_window.secondary_buttons[0].text() == "Choose another camera"

    calibration_window.secondary_buttons[0].click()

    assert backend.pause_count == 1
    assert backend.discover_count == 2
    assert calibration_window.flow.view.state is CalibrationUiState.CAMERA_SELECTION
    calibration_window.close()


def test_setup_shows_only_latest_local_preview_and_disables_it_on_handoff() -> None:
    calibration_window, backend = window()
    calibration_window.primary_button.click()
    backend.cameras_ready.emit((0,))
    calibration_window.primary_button.click()
    calibration_window.flow.camera_ready()
    calibration_window._render()
    preview = FramePreview(
        width=2,
        height=1,
        bytes_per_line=6,
        rgb_bytes=b"\xff\x00\x00" * 2,
    )

    backend.preview_ready.emit(preview)

    assert backend.preview_enabled == [True]
    assert backend.preview_consumed_count == 1
    assert calibration_window.preview_label.isVisibleTo(calibration_window)
    assert calibration_window.preview_label.pixmap().isNull() is False
    assert calibration_window.preview_label.accessibleName() == "Live local camera preview"

    calibration_window.detach_for_handoff()

    assert backend.preview_enabled == [True, False]


def test_closing_setup_clears_current_preview_pixels() -> None:
    calibration_window, backend = window()
    calibration_window.primary_button.click()
    backend.cameras_ready.emit((0,))
    calibration_window.primary_button.click()
    calibration_window.flow.camera_ready()
    calibration_window._render()
    backend.preview_ready.emit(
        FramePreview(
            width=2,
            height=1,
            bytes_per_line=6,
            rgb_bytes=b"\x00\xff\x00" * 2,
        )
    )
    assert calibration_window.preview_label.pixmap().isNull() is False

    calibration_window.close()

    assert calibration_window.preview_label.pixmap().isNull() is True
    assert backend.shutdown_count == 1


def test_saved_baseline_can_be_deleted_from_setup_and_notifies_persistence() -> None:
    application()
    deleted: list[bool] = []
    flow = CalibrationFlow(
        session=AnalysisSession(model_id="pose-test-v1", baseline=saved_baseline())
    )
    backend = FakeBackend()
    calibration_window = CalibrationWindow(
        flow=flow,
        backend=backend,
        on_baseline_deleted=lambda: deleted.append(True),
    )
    calibration_window.primary_button.click()
    backend.cameras_ready.emit((0,))
    calibration_window.primary_button.click()
    flow.camera_ready()
    calibration_window._render()

    assert calibration_window.secondary_buttons[1].text() == "Delete baseline"
    calibration_window.secondary_buttons[1].click()

    assert deleted == [True]
    assert flow.session.baseline is None
    calibration_window.close()
