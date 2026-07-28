from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from goodposture.adapters.runtime import (
    AdapterState,
    CameraOpenError,
    CameraOpenFailure,
    FramePreview,
    LatestFrameSlot,
    PoseAdapter,
    PoseModelSpec,
    verify_model,
)
from goodposture.app import AnalysisSession, SessionState
from goodposture.core.models import (
    Landmark,
    LandmarkName,
    ObservationFailure,
    PoseObservation,
)


class FakeCapture:
    def __init__(
        self,
        reads: list[tuple[bool, object]],
        *,
        read_error: Exception | None = None,
    ) -> None:
        self._reads = iter(reads)
        self._read_error = read_error
        self.released = False

    def read(self) -> tuple[bool, object]:
        if self._read_error is not None:
            raise self._read_error
        return next(self._reads, (False, None))

    def release(self) -> None:
        self.released = True


class FakeLandmarker:
    def __init__(
        self,
        detection: PoseObservation | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._detection = detection
        self._error = error
        self.closed = False

    def detect(self, frame_bgr: object, *, timestamp_ms: int) -> PoseObservation | None:
        del frame_bgr, timestamp_ms
        if self._error is not None:
            raise self._error
        return self._detection

    def close(self) -> None:
        self.closed = True


def _observation(timestamp_ms: int = 100) -> PoseObservation:
    landmark = Landmark(x=0.5, y=0.5, z=0.0)
    return PoseObservation(
        timestamp_ms=timestamp_ms,
        landmarks={name: landmark for name in LandmarkName},
    )


def _model_spec(tmp_path: Path) -> PoseModelSpec:
    model_path = tmp_path / "pose.task"
    payload = b"verified model bytes"
    model_path.write_bytes(payload)
    return PoseModelSpec(
        path=model_path,
        model_id="pose-test-v1",
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _adapter(
    spec: PoseModelSpec,
    captures: list[FakeCapture | Exception],
    landmarkers: list[FakeLandmarker | Exception],
) -> PoseAdapter:
    def open_capture(_: int) -> FakeCapture:
        item = captures.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def open_landmarker(_: Path) -> FakeLandmarker:
        item = landmarkers.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return PoseAdapter(
        model=spec,
        capture_factory=open_capture,
        landmarker_factory=open_landmarker,
    )


def test_model_verification_rejects_missing_empty_or_changed_assets(tmp_path: Path) -> None:
    missing = PoseModelSpec(
        path=tmp_path / "missing.task",
        model_id="pose-v1",
        sha256="0" * 64,
    )
    with pytest.raises(FileNotFoundError):
        verify_model(missing)

    empty_path = tmp_path / "empty.task"
    empty_path.write_bytes(b"")
    empty = PoseModelSpec(path=empty_path, model_id="pose-v1", sha256="0" * 64)
    with pytest.raises(ValueError, match="empty"):
        verify_model(empty)

    changed = _model_spec(tmp_path)
    changed.path.write_bytes(b"different")
    with pytest.raises(ValueError, match="integrity"):
        verify_model(changed)


@pytest.mark.parametrize(
    ("model_id", "sha256"),
    [("", "0" * 64), ("pose-v1", "bad")],
)
def test_model_spec_validates_identity_and_digest(
    tmp_path: Path, model_id: str, sha256: str
) -> None:
    with pytest.raises(ValueError):
        PoseModelSpec(path=tmp_path / "pose.task", model_id=model_id, sha256=sha256)


def test_latest_frame_slot_replaces_without_queueing() -> None:
    slot = LatestFrameSlot()
    first, second = object(), object()

    slot.publish(first)
    slot.publish(second)

    assert slot.take() is second
    assert slot.take() is None


def test_camera_enumeration_is_bounded_and_releases_every_probe(tmp_path: Path) -> None:
    captures = {0: FakeCapture([]), 2: FakeCapture([])}

    def open_capture(camera_index: int) -> FakeCapture:
        if camera_index not in captures:
            raise CameraOpenError(CameraOpenFailure.UNAVAILABLE)
        return captures[camera_index]

    adapter = PoseAdapter(
        model=_model_spec(tmp_path),
        capture_factory=open_capture,
        landmarker_factory=lambda _: FakeLandmarker(),
    )

    assert adapter.enumerate_cameras(maximum_devices=3) == (0, 2)
    assert all(capture.released for capture in captures.values())


def test_start_poll_pause_owns_and_releases_resources(tmp_path: Path) -> None:
    capture = FakeCapture([(True, object())])
    landmarker = FakeLandmarker(_observation())
    adapter = _adapter(_model_spec(tmp_path), [capture], [landmarker])

    started = adapter.start(camera_index=0)
    result = adapter.poll(timestamp_ms=100)
    paused = adapter.pause()

    assert started.state is AdapterState.READY
    assert result.observation == _observation()
    assert result.failure is None
    assert paused.state is AdapterState.PAUSED
    assert capture.released
    assert landmarker.closed
    assert adapter.buffered_frame_count == 0


def test_preview_is_opt_in_downscaled_throttled_and_not_retained(tmp_path: Path) -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :640] = (255, 0, 0)
    frame[:, 640:] = (0, 0, 255)
    capture = FakeCapture([(True, frame), (True, frame), (True, frame)])
    adapter = _adapter(
        _model_spec(tmp_path),
        [capture],
        [FakeLandmarker(_observation())],
    )
    adapter.start(camera_index=0)

    without_preview = adapter.poll(timestamp_ms=100)
    adapter.set_preview_enabled(True)
    first_preview = adapter.poll(timestamp_ms=200)
    throttled = adapter.poll(timestamp_ms=250)

    assert without_preview.preview is None
    assert isinstance(first_preview.preview, FramePreview)
    assert first_preview.preview.width == 480
    assert first_preview.preview.height == 270
    assert first_preview.preview.bytes_per_line == 1_440
    assert len(first_preview.preview.rgb_bytes) == 480 * 270 * 3
    assert first_preview.preview.rgb_bytes[:3] == b"\xff\x00\x00"
    assert "rgb_bytes" not in repr(first_preview.preview)
    assert throttled.preview is None
    assert adapter.buffered_frame_count == 0


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (CameraOpenFailure.PERMISSION_DENIED, AdapterState.PERMISSION_DENIED),
        (CameraOpenFailure.UNAVAILABLE, AdapterState.CAMERA_UNAVAILABLE),
    ],
)
def test_camera_open_failures_have_clear_states(
    tmp_path: Path,
    reason: CameraOpenFailure,
    expected: AdapterState,
) -> None:
    adapter = _adapter(
        _model_spec(tmp_path),
        [CameraOpenError(reason)],
        [FakeLandmarker()],
    )

    result = adapter.start(camera_index=0)

    assert result.state is expected
    assert result.failure is ObservationFailure.CAPTURE_UNAVAILABLE


def test_native_permission_error_maps_to_permission_denied(tmp_path: Path) -> None:
    adapter = _adapter(
        _model_spec(tmp_path),
        [PermissionError("native details")],
        [FakeLandmarker()],
    )

    result = adapter.start(camera_index=0)

    assert result.state is AdapterState.PERMISSION_DENIED
    assert result.failure is ObservationFailure.CAPTURE_UNAVAILABLE
    assert "native" not in result.message


def test_read_failure_disconnects_and_tears_down(tmp_path: Path) -> None:
    capture = FakeCapture([(False, None)])
    landmarker = FakeLandmarker()
    adapter = _adapter(_model_spec(tmp_path), [capture], [landmarker])
    adapter.start(camera_index=0)

    result = adapter.poll(timestamp_ms=100)

    assert result.state is AdapterState.CAMERA_DISCONNECTED
    assert result.failure is ObservationFailure.CAPTURE_READ_FAILED
    assert capture.released
    assert landmarker.closed
    assert adapter.buffered_frame_count == 0


def test_capture_exception_disconnects_and_tears_down(tmp_path: Path) -> None:
    capture = FakeCapture([], read_error=OSError("device details"))
    landmarker = FakeLandmarker()
    adapter = _adapter(_model_spec(tmp_path), [capture], [landmarker])
    adapter.start(camera_index=0)

    result = adapter.poll(timestamp_ms=100)

    assert result.state is AdapterState.CAMERA_DISCONNECTED
    assert result.failure is ObservationFailure.CAPTURE_READ_FAILED
    assert "device" not in result.message
    assert capture.released
    assert landmarker.closed


def test_inference_failure_is_recoverable_and_does_not_retain_frame(tmp_path: Path) -> None:
    capture = FakeCapture([(True, object())])
    landmarker = FakeLandmarker(error=RuntimeError("sensitive details"))
    adapter = _adapter(_model_spec(tmp_path), [capture], [landmarker])
    adapter.start(camera_index=0)

    result = adapter.poll(timestamp_ms=100)

    assert result.state is AdapterState.INFERENCE_FAILED
    assert result.failure is ObservationFailure.INFERENCE_FAILED
    assert "sensitive" not in result.message
    assert capture.released
    assert landmarker.closed
    assert adapter.buffered_frame_count == 0


def test_recover_reopens_resources_and_reports_recovered(tmp_path: Path) -> None:
    first_capture = FakeCapture([(False, None)])
    first_landmarker = FakeLandmarker()
    second_capture = FakeCapture([(True, object())])
    second_landmarker = FakeLandmarker(_observation(200))
    adapter = _adapter(
        _model_spec(tmp_path),
        [first_capture, second_capture],
        [first_landmarker, second_landmarker],
    )
    adapter.start(camera_index=2)
    adapter.poll(timestamp_ms=100)

    recovered = adapter.recover()
    result = adapter.poll(timestamp_ms=200)

    assert recovered.state is AdapterState.RECOVERED
    assert result.observation == _observation(200)


def test_invalid_inputs_and_non_monotonic_timestamps_are_rejected(tmp_path: Path) -> None:
    capture = FakeCapture([(True, object()), (True, object())])
    adapter = _adapter(
        _model_spec(tmp_path),
        [capture],
        [FakeLandmarker(_observation())],
    )
    with pytest.raises(ValueError, match="camera_index"):
        adapter.start(camera_index=-1)

    adapter.start(camera_index=0)
    adapter.poll(timestamp_ms=100)
    with pytest.raises(ValueError, match="strictly increasing"):
        adapter.poll(timestamp_ms=100)


def test_landmarker_initialization_failure_releases_open_camera(tmp_path: Path) -> None:
    capture = FakeCapture([])
    adapter = _adapter(
        _model_spec(tmp_path),
        [capture],
        [RuntimeError("incompatible model")],
    )

    result = adapter.start(camera_index=0)

    assert result.state is AdapterState.MODEL_INCOMPATIBLE
    assert capture.released


def test_adapter_results_feed_headless_session_without_ui(tmp_path: Path) -> None:
    capture = FakeCapture([(True, object())])
    adapter = _adapter(
        _model_spec(tmp_path),
        [capture],
        [FakeLandmarker(None)],
    )
    session = AnalysisSession(model_id="pose-test-v1")
    session.start(timestamp_ms=0)
    session.start_calibration(timestamp_ms=1)
    adapter.start(camera_index=0)

    result = adapter.poll(timestamp_ms=100)
    assert result.observation is not None
    update = session.process_observation(result.observation)

    assert update.state is SessionState.CALIBRATING
    assert result.observation.landmarks == {}
