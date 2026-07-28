"""Privacy-bounded operational diagnostic contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol


class DiagnosticEvent(StrEnum):
    """Allowlisted events that contain no observations or user data."""

    APP_STARTED = "app_started"
    APP_EXITED = "app_exited"
    UNHANDLED_FAILURE = "unhandled_failure"
    LOCAL_DATA_FAILURE = "local_data_failure"
    CAMERA_STARTED = "camera_started"
    CAMERA_READY = "camera_ready"
    CAMERA_PAUSED = "camera_paused"
    CAMERA_FAILURE = "camera_failure"
    TRACKING_UNKNOWN = "tracking_unknown"
    TRACKING_RECOVERED = "tracking_recovered"
    CHECKPOINT_FAILED = "checkpoint_failed"
    SESSION_STOPPED = "session_stopped"


class DiagnosticSink(Protocol):
    """Receive one allowlisted event without arbitrary detail fields."""

    def record(self, event: DiagnosticEvent) -> None: ...
