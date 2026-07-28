"""Framework-neutral data contracts for pose analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class LandmarkName(StrEnum):
    """The upper-body MediaPipe landmarks used by the feasibility metrics."""

    NOSE = "nose"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"


class ObservationFailure(StrEnum):
    """Sanitized adapter failures shared with the headless session."""

    CAPTURE_UNAVAILABLE = "capture_unavailable"
    CAPTURE_READ_FAILED = "capture_read_failed"
    INFERENCE_FAILED = "inference_failed"


@dataclass(frozen=True, slots=True)
class Landmark:
    """One normalized pose landmark, independent of MediaPipe types."""

    x: float
    y: float
    z: float
    visibility: float = 1.0
    presence: float = 1.0

    @property
    def confidence(self) -> float:
        """Return the conservative confidence supported by both signals."""

        return min(self.visibility, self.presence)


@dataclass(frozen=True, slots=True)
class PoseObservation:
    """A timestamped pose observation suitable for live or recorded sources."""

    timestamp_ms: int
    landmarks: Mapping[LandmarkName, Landmark]


@dataclass(frozen=True, slots=True)
class MetricReading:
    """A metric value plus the confidence that governs its availability."""

    value: float | None
    confidence: float
    unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class PostureMetrics:
    """Initial explainable measurements intended for personalized calibration."""

    shoulder_tilt_degrees: MetricReading
    torso_lean_degrees: MetricReading
    head_lateral_offset_ratio: MetricReading
    head_vertical_offset_ratio: MetricReading
    head_depth_ratio: MetricReading

    def as_dict(self) -> dict[str, float | None]:
        """Return display/evaluation values without exposing landmark data."""

        return {
            "shoulder_tilt_degrees": self.shoulder_tilt_degrees.value,
            "torso_lean_degrees": self.torso_lean_degrees.value,
            "head_lateral_offset_ratio": self.head_lateral_offset_ratio.value,
            "head_vertical_offset_ratio": self.head_vertical_offset_ratio.value,
            "head_depth_ratio": self.head_depth_ratio.value,
        }
