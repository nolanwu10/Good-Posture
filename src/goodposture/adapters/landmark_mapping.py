"""Mapping from pose-model output to framework-neutral application contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from goodposture.core.models import Landmark, LandmarkName, PoseObservation


class SourceLandmark(Protocol):
    """The small structural interface consumed from a pose model result."""

    x: float
    y: float
    z: float
    visibility: float | None
    presence: float | None


@dataclass(frozen=True, slots=True)
class DetectedPose:
    """A model result safe to pass outside the MediaPipe adapter."""

    observation: PoseObservation
    all_landmarks: tuple[Landmark, ...]


_CORE_LANDMARK_INDICES = {
    LandmarkName.NOSE: 0,
    LandmarkName.LEFT_EYE: 2,
    LandmarkName.RIGHT_EYE: 5,
    LandmarkName.LEFT_SHOULDER: 11,
    LandmarkName.RIGHT_SHOULDER: 12,
    LandmarkName.LEFT_HIP: 23,
    LandmarkName.RIGHT_HIP: 24,
}
_MINIMUM_REQUIRED_LANDMARKS = max(_CORE_LANDMARK_INDICES.values()) + 1


def _confidence_or_zero(value: float | None) -> float:
    return 0.0 if value is None else value


def map_pose_landmarks(
    source: Sequence[SourceLandmark],
    *,
    timestamp_ms: int,
) -> DetectedPose | None:
    """Copy one model pose into local, framework-neutral value objects."""

    if len(source) < _MINIMUM_REQUIRED_LANDMARKS:
        return None

    landmarks = tuple(
        Landmark(
            x=item.x,
            y=item.y,
            z=item.z,
            visibility=_confidence_or_zero(item.visibility),
            presence=_confidence_or_zero(item.presence),
        )
        for item in source
    )
    observation = PoseObservation(
        timestamp_ms=timestamp_ms,
        landmarks={
            name: landmarks[index] for name, index in _CORE_LANDMARK_INDICES.items()
        },
    )
    return DetectedPose(observation=observation, all_landmarks=landmarks)
