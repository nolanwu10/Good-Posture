"""Accessible PySide6 window for consent and personalized calibration."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from PySide6.QtCore import Qt, SignalInstance
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import QMainWindow, QPushButton

from goodposture.adapters.runtime import AdapterResult, AdapterState, FramePreview
from goodposture.ui.calibration_flow import CalibrationFlow, CalibrationUiState
from goodposture.ui.calibration_panel import CalibrationPanel


class CalibrationBackendProtocol(Protocol):
    cameras_ready: SignalInstance
    result_ready: SignalInstance
    preview_ready: SignalInstance

    def discover_cameras(self) -> None: ...

    def start_camera(self, camera_index: int) -> None: ...

    def pause(self) -> None: ...

    def set_preview_enabled(self, enabled: bool) -> None: ...

    def preview_consumed(self) -> None: ...

    def shutdown(self) -> None: ...


class CalibrationWindow(QMainWindow):
    """Render one focused setup flow with keyboard-accessible native controls."""

    def __init__(
        self,
        *,
        flow: CalibrationFlow,
        backend: CalibrationBackendProtocol,
        on_complete: Callable[[CalibrationWindow], None] | None = None,
        on_baseline_deleted: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.flow = flow
        self._backend = backend
        self._on_complete = on_complete
        self._on_baseline_deleted = on_baseline_deleted
        self._handed_off = False
        self._last_timestamp_ms = 0
        self.setWindowTitle("GoodPosture setup")
        self.setAccessibleName("GoodPosture calibration setup")
        self.setMinimumSize(720, 620)
        self.resize(900, 720)

        panel = CalibrationPanel()
        self.title_label = panel.title_label
        self.body_label = panel.body_label
        self.preview_label = panel.preview_label
        self.countdown_label = panel.countdown_label
        self.camera_combo = panel.camera_combo
        self.progress_bar = panel.progress_bar
        self.secondary_buttons = panel.secondary_buttons
        self.primary_button = panel.primary_button
        self._panel = panel
        for button in self.secondary_buttons:
            button.clicked.connect(self._secondary_clicked)
        self.primary_button.clicked.connect(self._primary_clicked)
        self.setCentralWidget(panel)

        self._backend.cameras_ready.connect(self._cameras_discovered)
        self._backend.result_ready.connect(self._adapter_result)
        self._backend.preview_ready.connect(self._preview_ready)
        self._backend.set_preview_enabled(True)
        self._render()

    def _timestamp_ms(self) -> int:
        timestamp_ms = time.monotonic_ns() // 1_000_000
        self._last_timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        return self._last_timestamp_ms

    def _render(self) -> None:
        self._panel.apply_view(self.flow.view)

    def _primary_clicked(self) -> None:
        state = self.flow.view.state
        if state is CalibrationUiState.PRIVACY:
            self.flow.accept_privacy(timestamp_ms=self._timestamp_ms())
            self._begin_camera_discovery()
            return
        if state is CalibrationUiState.CAMERA_SELECTION:
            camera_index = self.camera_combo.currentData()
            if isinstance(camera_index, int):
                self.flow.select_camera(camera_index)
                self._render()
                self._backend.start_camera(camera_index)
            return
        if state is CalibrationUiState.CAMERA_ERROR:
            self.flow.retry_camera_selection()
            self._backend.pause()
            self._begin_camera_discovery()
            return
        if state is CalibrationUiState.FAILURE:
            self.flow.retry(timestamp_ms=self._timestamp_ms())
            self._render()
            return
        if state is CalibrationUiState.SUCCESS:
            if self._on_complete is None:
                self.close()
            else:
                self.detach_for_handoff()
                self._on_complete(self)

    def _secondary_clicked(self) -> None:
        sender = self.sender()
        if not isinstance(sender, QPushButton):
            return
        action = sender.text()
        if action == "Leave":
            self.close()
        elif action == "Choose another camera":
            self._backend.pause()
            self.flow.choose_another_camera(timestamp_ms=self._timestamp_ms())
            self._begin_camera_discovery()
        elif action == "Recalibrate":
            self.flow.recalibrate(timestamp_ms=self._timestamp_ms())
            selected = self.flow.selected_camera_index
            self._render()
            if selected is not None:
                self._backend.start_camera(selected)
        elif action == "Delete baseline":
            self._backend.pause()
            self.flow.delete_baseline(timestamp_ms=self._timestamp_ms())
            if self._on_baseline_deleted is not None:
                self._on_baseline_deleted()
            self._begin_camera_discovery()

    def _begin_camera_discovery(self) -> None:
        self.camera_combo.clear()
        self.camera_combo.addItem("Searching for local cameras…", None)
        self._render()
        self._backend.discover_cameras()

    def _cameras_discovered(self, camera_indices: tuple[int, ...]) -> None:
        if self.flow.view.state is CalibrationUiState.CLOSED:
            return
        self.flow.set_cameras(camera_indices)
        self.camera_combo.clear()
        for ordinal, camera_index in enumerate(camera_indices, start=1):
            self.camera_combo.addItem(
                f"Camera source {ordinal} (Windows index {camera_index})",
                camera_index,
            )
        self._render()

    def _adapter_result(self, value: object) -> None:
        if not isinstance(value, AdapterResult):
            return
        if value.observation is not None:
            self._last_timestamp_ms = max(
                self._last_timestamp_ms,
                value.observation.timestamp_ms,
            )
            self.flow.process_observation(value.observation)
        elif value.failure is not None:
            self.flow.process_failure(
                timestamp_ms=self._timestamp_ms(),
                failure=value.failure,
                message=value.message,
            )
        elif value.state in {AdapterState.READY, AdapterState.RECOVERED}:
            if self.flow.view.state is CalibrationUiState.CAMERA_SELECTION:
                self.flow.camera_ready()
        else:
            self.flow.camera_error(value.message)
        self._render()
        if self.flow.view.state is CalibrationUiState.SUCCESS:
            self._backend.pause()

    def _preview_ready(self, value: object) -> None:
        try:
            if not isinstance(value, FramePreview):
                return
            if self.flow.view.state not in {
                CalibrationUiState.FRAMING,
                CalibrationUiState.COUNTDOWN,
                CalibrationUiState.CALIBRATING,
            }:
                return
            image = QImage(
                value.rgb_bytes,
                value.width,
                value.height,
                value.bytes_per_line,
                QImage.Format.Format_RGB888,
            ).copy()
            pixmap = QPixmap.fromImage(image).scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.preview_label.setText("")
            self.preview_label.setPixmap(pixmap)
        finally:
            self._backend.preview_consumed()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._handed_off:
            event.accept()
            return
        if self.flow.view.state is not CalibrationUiState.CLOSED:
            self.flow.leave(timestamp_ms=self._timestamp_ms())
        self.preview_label.clear()
        self._backend.set_preview_enabled(False)
        self._backend.shutdown()
        event.accept()

    def detach_for_handoff(self) -> None:
        """Transfer backend/session ownership to the long-running desktop app."""

        if self._handed_off:
            return
        self._handed_off = True
        self._backend.cameras_ready.disconnect(self._cameras_discovered)
        self._backend.result_ready.disconnect(self._adapter_result)
        self._backend.preview_ready.disconnect(self._preview_ready)
        self._backend.set_preview_enabled(False)
        self.preview_label.clear()
        self.hide()
