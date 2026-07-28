"""Qt worker boundary for local capture and inference."""

from __future__ import annotations

import time
from dataclasses import replace

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot

from goodposture.adapters.runtime import AdapterState, PoseAdapter


class CameraWorker(QObject):
    """Run the synchronous adapter off the UI thread."""

    cameras_ready = Signal(tuple)
    result_ready = Signal(object)
    preview_ready = Signal(object)

    def __init__(self, adapter: PoseAdapter, *, poll_interval_ms: int = 33) -> None:
        super().__init__()
        self._adapter = adapter
        self._timer = QTimer(self)
        self._timer.setInterval(poll_interval_ms)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._poll)
        self._preview_pending = False

    @Slot()
    def discover_cameras(self) -> None:
        self.cameras_ready.emit(self._adapter.enumerate_cameras())

    @Slot(int)
    def start_camera(self, camera_index: int) -> None:
        self._timer.stop()
        result = self._adapter.start(camera_index=camera_index)
        self.result_ready.emit(result)
        if result.state is AdapterState.READY:
            self._timer.start()

    @Slot()
    def pause(self) -> None:
        self._timer.stop()
        self._preview_pending = False
        self._adapter.pause()

    @Slot()
    def shutdown(self) -> None:
        self._timer.stop()
        self._preview_pending = False
        self._adapter.set_preview_enabled(False)
        self._adapter.close()

    @Slot(bool)
    def set_preview_enabled(self, enabled: bool) -> None:
        self._preview_pending = False
        self._adapter.set_preview_enabled(enabled)

    @Slot()
    def preview_consumed(self) -> None:
        self._preview_pending = False

    @Slot()
    def _poll(self) -> None:
        result = self._adapter.poll(timestamp_ms=time.monotonic_ns() // 1_000_000)
        preview = result.preview
        self.result_ready.emit(replace(result, preview=None))
        if preview is not None and not self._preview_pending:
            self._preview_pending = True
            self.preview_ready.emit(preview)
        if result.state not in {AdapterState.READY, AdapterState.RECOVERED}:
            self._timer.stop()


class CameraBackend(QObject):
    """Own a worker thread and expose a small window-facing API."""

    cameras_ready = Signal(tuple)
    result_ready = Signal(object)
    preview_ready = Signal(object)
    _discover_requested = Signal()
    _start_requested = Signal(int)
    _pause_requested = Signal()
    _shutdown_requested = Signal()
    _preview_enabled_requested = Signal(bool)
    _preview_consumed_requested = Signal()

    def __init__(self, adapter: PoseAdapter) -> None:
        super().__init__()
        self._thread = QThread(self)
        self._worker = CameraWorker(adapter)
        self._worker.moveToThread(self._thread)
        self._discover_requested.connect(self._worker.discover_cameras)
        self._start_requested.connect(self._worker.start_camera)
        self._pause_requested.connect(self._worker.pause)
        self._preview_enabled_requested.connect(self._worker.set_preview_enabled)
        self._preview_consumed_requested.connect(self._worker.preview_consumed)
        self._shutdown_requested.connect(
            self._worker.shutdown,
            Qt.ConnectionType.BlockingQueuedConnection,
        )
        self._worker.cameras_ready.connect(self.cameras_ready.emit)
        self._worker.result_ready.connect(self.result_ready.emit)
        self._worker.preview_ready.connect(self.preview_ready.emit)
        self._closed = False
        self._thread.start()

    def discover_cameras(self) -> None:
        self._discover_requested.emit()

    def start_camera(self, camera_index: int) -> None:
        self._start_requested.emit(camera_index)

    def pause(self) -> None:
        self._pause_requested.emit()

    def set_preview_enabled(self, enabled: bool) -> None:
        self._preview_enabled_requested.emit(enabled)

    def preview_consumed(self) -> None:
        self._preview_consumed_requested.emit()

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread.isRunning():
            self._shutdown_requested.emit()
            self._thread.quit()
            self._thread.wait(5_000)
