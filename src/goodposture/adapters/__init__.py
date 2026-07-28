"""Local device and model adapters."""

from goodposture.adapters.runtime import (
    AdapterResult,
    AdapterState,
    CameraOpenError,
    CameraOpenFailure,
    FramePreview,
    LatestFrameSlot,
    PoseAdapter,
    PoseModelSpec,
    verify_model,
)
from goodposture.adapters.sqlite_repository import (
    SQLITE_SCHEMA_VERSION,
    DailySummary,
    LocalSettings,
    SqliteAggregateSink,
    SqliteRepository,
)

__all__ = [
    "AdapterResult",
    "AdapterState",
    "CameraOpenError",
    "CameraOpenFailure",
    "DailySummary",
    "FramePreview",
    "LatestFrameSlot",
    "LocalSettings",
    "PoseAdapter",
    "PoseModelSpec",
    "SQLITE_SCHEMA_VERSION",
    "SqliteAggregateSink",
    "SqliteRepository",
    "verify_model",
]
