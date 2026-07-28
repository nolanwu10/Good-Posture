"""Hardened local camera and pose-inference resource lifecycle."""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, cast

import cv2

from goodposture.adapters.landmark_mapping import DetectedPose
from goodposture.adapters.mediapipe_pose import MediaPipePoseLandmarker
from goodposture.core.models import ObservationFailure, PoseObservation

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_HASH_CHUNK_SIZE = 1024 * 1024


class AdapterState(StrEnum):
    """Operational states safe to expose without native exception details."""

    STOPPED = "stopped"
    READY = "ready"
    RECOVERED = "recovered"
    PAUSED = "paused"
    PERMISSION_DENIED = "permission_denied"
    CAMERA_UNAVAILABLE = "camera_unavailable"
    CAMERA_DISCONNECTED = "camera_disconnected"
    MODEL_INVALID = "model_invalid"
    MODEL_INCOMPATIBLE = "model_incompatible"
    INFERENCE_FAILED = "inference_failed"


class CameraOpenFailure(StrEnum):
    """Reasons a capture factory may report while opening a camera."""

    PERMISSION_DENIED = "permission_denied"
    UNAVAILABLE = "unavailable"


class CameraOpenError(RuntimeError):
    """Typed camera-open failure without device-specific details."""

    def __init__(self, reason: CameraOpenFailure) -> None:
        super().__init__(reason.value)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class PoseModelSpec:
    """Versioned local model identity plus its expected integrity digest."""

    path: Path
    model_id: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")


def verify_model(model: PoseModelSpec) -> None:
    """Verify that a local model exists, is non-empty, and matches its digest."""

    if not model.path.is_file():
        raise FileNotFoundError(f"Pose model asset is unavailable: {model.model_id}")
    if model.path.stat().st_size == 0:
        raise ValueError(f"Pose model asset is empty: {model.model_id}")
    digest = hashlib.sha256()
    with model.path.open("rb") as model_file:
        while chunk := model_file.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    if not hmac.compare_digest(digest.hexdigest().lower(), model.sha256.lower()):
        raise ValueError(f"Pose model integrity check failed: {model.model_id}")


class Capture(Protocol):
    """Small capture surface needed by the deterministic adapter."""

    def read(self) -> tuple[bool, object]: ...

    def release(self) -> None: ...


class Landmarker(Protocol):
    """Small inference surface needed by the deterministic adapter."""

    def detect(
        self, frame_bgr: object, *, timestamp_ms: int
    ) -> DetectedPose | PoseObservation | None: ...

    def close(self) -> None: ...


class LatestFrameSlot:
    """A single replaceable frame slot that cannot grow into a queue."""

    def __init__(self) -> None:
        self._frame: object | None = None
        self._lock = Lock()

    def publish(self, frame: object) -> None:
        with self._lock:
            self._frame = frame

    def take(self) -> object | None:
        with self._lock:
            frame = self._frame
            self._frame = None
            return frame

    def clear(self) -> None:
        with self._lock:
            self._frame = None

    @property
    def count(self) -> int:
        with self._lock:
            return int(self._frame is not None)


@dataclass(frozen=True, slots=True)
class FramePreview:
    """One bounded RGB preview kept only long enough for the setup UI to paint."""

    width: int
    height: int
    bytes_per_line: int
    rgb_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("preview dimensions must be positive")
        if self.bytes_per_line != self.width * 3:
            raise ValueError("preview must contain packed RGB pixels")
        if len(self.rgb_bytes) != self.bytes_per_line * self.height:
            raise ValueError("preview byte length does not match its dimensions")


@dataclass(frozen=True, slots=True)
class AdapterResult:
    """One bounded adapter result with an optional transient setup preview."""

    state: AdapterState
    message: str
    observation: PoseObservation | None = None
    failure: ObservationFailure | None = None
    preview: FramePreview | None = field(default=None, repr=False)


CaptureFactory = Callable[[int], Capture]
LandmarkerFactory = Callable[[Path], Landmarker]


def open_opencv_camera(camera_index: int) -> Capture:
    """Open one Windows camera or raise a sanitized availability failure."""

    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        raise CameraOpenError(CameraOpenFailure.UNAVAILABLE)
    return capture


def open_mediapipe_landmarker(model_path: Path) -> Landmarker:
    """Create the production in-memory MediaPipe landmarker."""

    return MediaPipePoseLandmarker(model_path)


class PoseAdapter:
    """Own camera/model resources and emit session-ready observations."""

    def __init__(
        self,
        *,
        model: PoseModelSpec,
        capture_factory: CaptureFactory = open_opencv_camera,
        landmarker_factory: LandmarkerFactory = open_mediapipe_landmarker,
        preview_max_width: int = 480,
        preview_interval_ms: int = 100,
    ) -> None:
        if preview_max_width <= 0:
            raise ValueError("preview_max_width must be positive")
        if preview_interval_ms <= 0:
            raise ValueError("preview_interval_ms must be positive")
        self._model = model
        self._capture_factory = capture_factory
        self._landmarker_factory = landmarker_factory
        self._capture: Capture | None = None
        self._landmarker: Landmarker | None = None
        self._camera_index: int | None = None
        self._last_timestamp_ms: int | None = None
        self._slot = LatestFrameSlot()
        self._state = AdapterState.STOPPED
        self._preview_enabled = False
        self._preview_max_width = preview_max_width
        self._preview_interval_ms = preview_interval_ms
        self._last_preview_timestamp_ms: int | None = None

    @property
    def state(self) -> AdapterState:
        return self._state

    @property
    def buffered_frame_count(self) -> int:
        return self._slot.count

    def set_preview_enabled(self, enabled: bool) -> None:
        """Enable or disable a throttled, downscaled in-memory setup preview."""

        self._preview_enabled = enabled
        self._last_preview_timestamp_ms = None

    def enumerate_cameras(self, *, maximum_devices: int = 8) -> tuple[int, ...]:
        """Probe a bounded index range and release every successful capture."""

        if (
            not isinstance(maximum_devices, int)
            or isinstance(maximum_devices, bool)
            or maximum_devices <= 0
        ):
            raise ValueError("maximum_devices must be a positive integer")
        available: list[int] = []
        for camera_index in range(maximum_devices):
            capture: Capture | None = None
            try:
                capture = self._capture_factory(camera_index)
                available.append(camera_index)
            except Exception:
                continue
            finally:
                if capture is not None:
                    with suppress(Exception):
                        capture.release()
        return tuple(available)

    def start(self, *, camera_index: int) -> AdapterResult:
        """Verify assets and acquire exactly one camera and model instance."""

        if (
            not isinstance(camera_index, int)
            or isinstance(camera_index, bool)
            or camera_index < 0
        ):
            raise ValueError("camera_index must be a non-negative integer")
        self._release_resources()
        self._camera_index = camera_index
        self._last_timestamp_ms = None
        self._last_preview_timestamp_ms = None
        try:
            verify_model(self._model)
        except (FileNotFoundError, OSError, ValueError):
            self._state = AdapterState.MODEL_INVALID
            return self._result("The local pose model did not pass its integrity check.")

        try:
            self._capture = self._capture_factory(camera_index)
        except PermissionError:
            self._state = AdapterState.PERMISSION_DENIED
            return self._result(
                "Camera permission was denied.",
                failure=ObservationFailure.CAPTURE_UNAVAILABLE,
            )
        except CameraOpenError as error:
            if error.reason is CameraOpenFailure.PERMISSION_DENIED:
                self._state = AdapterState.PERMISSION_DENIED
                message = "Camera permission was denied."
            else:
                self._state = AdapterState.CAMERA_UNAVAILABLE
                message = "The selected camera is unavailable."
            return self._result(message, failure=ObservationFailure.CAPTURE_UNAVAILABLE)
        except Exception:
            self._state = AdapterState.CAMERA_UNAVAILABLE
            return self._result(
                "The selected camera is unavailable.",
                failure=ObservationFailure.CAPTURE_UNAVAILABLE,
            )

        try:
            self._landmarker = self._landmarker_factory(self._model.path)
        except Exception:
            self._release_resources()
            self._state = AdapterState.MODEL_INCOMPATIBLE
            return self._result("The local pose model could not be initialized.")

        self._state = AdapterState.READY
        return self._result("Local camera and pose model are ready.")

    def poll(self, *, timestamp_ms: int) -> AdapterResult:
        """Read and infer one frame, then immediately discard every frame reference."""

        if self._state not in {AdapterState.READY, AdapterState.RECOVERED}:
            raise RuntimeError("adapter must be ready before polling")
        if (
            not isinstance(timestamp_ms, int)
            or isinstance(timestamp_ms, bool)
            or timestamp_ms < 0
        ):
            raise ValueError("timestamp_ms must be non-negative")
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("timestamp_ms must be strictly increasing")
        self._last_timestamp_ms = timestamp_ms
        assert self._capture is not None
        assert self._landmarker is not None

        try:
            ok, frame = self._capture.read()
        except Exception:
            self._release_resources()
            self._state = AdapterState.CAMERA_DISCONNECTED
            return self._result(
                "The selected camera stopped providing frames.",
                failure=ObservationFailure.CAPTURE_READ_FAILED,
            )
        if not ok or frame is None:
            self._release_resources()
            self._state = AdapterState.CAMERA_DISCONNECTED
            return self._result(
                "The selected camera stopped providing frames.",
                failure=ObservationFailure.CAPTURE_READ_FAILED,
            )

        self._slot.publish(frame)
        try:
            current_frame = self._slot.take()
            assert current_frame is not None
            preview = self._preview(current_frame, timestamp_ms=timestamp_ms)
            detected = self._landmarker.detect(current_frame, timestamp_ms=timestamp_ms)
            if isinstance(detected, DetectedPose):
                observation = detected.observation
            elif detected is None:
                observation = PoseObservation(timestamp_ms=timestamp_ms, landmarks={})
            else:
                observation = detected
            return self._result(
                "Local pose inference completed.",
                observation=observation,
                preview=preview,
            )
        except Exception:
            self._release_resources()
            self._state = AdapterState.INFERENCE_FAILED
            return self._result(
                "Local pose inference was temporarily unavailable.",
                failure=ObservationFailure.INFERENCE_FAILED,
            )
        finally:
            self._slot.clear()

    def pause(self) -> AdapterResult:
        """Release all native resources promptly and enter a resumable state."""

        self._release_resources()
        self._state = AdapterState.PAUSED
        return self._result("Local camera monitoring is paused.")

    def recover(self) -> AdapterResult:
        """Reacquire the last selected camera after pause or recoverable failure."""

        if self._camera_index is None:
            raise RuntimeError("adapter has not selected a camera")
        result = self.start(camera_index=self._camera_index)
        if result.state is AdapterState.READY:
            self._state = AdapterState.RECOVERED
            return self._result("Local camera monitoring recovered.")
        return result

    def close(self) -> AdapterResult:
        """Idempotently release all native resources."""

        self._release_resources()
        self._state = AdapterState.STOPPED
        return self._result("Local camera monitoring stopped.")

    def _release_resources(self) -> None:
        self._slot.clear()
        self._last_preview_timestamp_ms = None
        landmarker, self._landmarker = self._landmarker, None
        capture, self._capture = self._capture, None
        if landmarker is not None:
            with suppress(Exception):
                landmarker.close()
        if capture is not None:
            with suppress(Exception):
                capture.release()

    def _result(
        self,
        message: str,
        *,
        observation: PoseObservation | None = None,
        failure: ObservationFailure | None = None,
        preview: FramePreview | None = None,
    ) -> AdapterResult:
        return AdapterResult(
            state=self._state,
            message=message,
            observation=observation,
            failure=failure,
            preview=preview,
        )

    def _preview(self, frame_bgr: object, *, timestamp_ms: int) -> FramePreview | None:
        if not self._preview_enabled:
            return None
        if (
            self._last_preview_timestamp_ms is not None
            and timestamp_ms - self._last_preview_timestamp_ms
            < self._preview_interval_ms
        ):
            return None
        try:
            frame = cast(Any, frame_bgr)
            height = int(frame.shape[0])
            width = int(frame.shape[1])
            if height <= 0 or width <= 0:
                return None
            target_width = min(width, self._preview_max_width)
            target_height = max(1, round(height * target_width / width))
            if target_width != width:
                frame = cv2.resize(
                    frame,
                    (target_width, target_height),
                    interpolation=cv2.INTER_AREA,
                )
            rgb = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
            preview = FramePreview(
                width=target_width,
                height=target_height,
                bytes_per_line=target_width * 3,
                rgb_bytes=bytes(rgb.tobytes()),
            )
        except (AttributeError, IndexError, TypeError, ValueError, cv2.error):
            return None
        self._last_preview_timestamp_ms = timestamp_ms
        return preview

    def __enter__(self) -> PoseAdapter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
