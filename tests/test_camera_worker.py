from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from goodposture.adapters.runtime import AdapterResult, AdapterState, FramePreview
from goodposture.ui.camera_worker import CameraWorker


class FakePreviewAdapter:
    def __init__(self, preview: FramePreview) -> None:
        self.preview = preview
        self.preview_enabled = False

    def set_preview_enabled(self, enabled: bool) -> None:
        self.preview_enabled = enabled

    def poll(self, *, timestamp_ms: int) -> AdapterResult:
        del timestamp_ms
        return AdapterResult(
            state=AdapterState.READY,
            message="ready",
            preview=self.preview,
        )

    def enumerate_cameras(self) -> tuple[int, ...]:
        return ()

    def start(self, *, camera_index: int) -> AdapterResult:
        del camera_index
        return AdapterResult(state=AdapterState.READY, message="ready")

    def pause(self) -> AdapterResult:
        return AdapterResult(state=AdapterState.PAUSED, message="paused")

    def close(self) -> AdapterResult:
        return AdapterResult(state=AdapterState.STOPPED, message="stopped")


def application() -> QApplication:
    instance = QApplication.instance()
    if isinstance(instance, QApplication):
        return instance
    return QApplication([])


def test_worker_allows_at_most_one_unconsumed_preview() -> None:
    application()
    preview = FramePreview(
        width=2,
        height=1,
        bytes_per_line=6,
        rgb_bytes=b"\x00" * 6,
    )
    adapter = FakePreviewAdapter(preview)
    worker = CameraWorker(adapter)  # type: ignore[arg-type]
    previews: list[FramePreview] = []
    results: list[AdapterResult] = []
    worker.preview_ready.connect(previews.append)
    worker.result_ready.connect(results.append)
    worker.set_preview_enabled(True)

    worker._poll()
    worker._poll()

    assert adapter.preview_enabled is True
    assert previews == [preview]
    assert all(result.preview is None for result in results)

    worker.preview_consumed()
    worker._poll()

    assert previews == [preview, preview]
