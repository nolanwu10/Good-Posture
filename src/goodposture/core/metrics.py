"""Deterministic, framework-neutral posture feature extraction."""

from __future__ import annotations

import math
from collections.abc import Iterable

from goodposture.core.models import (
    Landmark,
    LandmarkName,
    MetricReading,
    PoseObservation,
    PostureMetrics,
)

_GEOMETRY_EPSILON = 1e-9


def _landmarks(
    observation: PoseObservation,
    names: tuple[LandmarkName, ...],
) -> tuple[Landmark, ...] | None:
    try:
        return tuple(observation.landmarks[name] for name in names)
    except KeyError:
        return None


def _minimum_confidence(landmarks: Iterable[Landmark]) -> float:
    return min((landmark.confidence for landmark in landmarks), default=0.0)


def _unavailable(confidence: float, reason: str) -> MetricReading:
    return MetricReading(value=None, confidence=confidence, unavailable_reason=reason)


def _midpoint(first: Landmark, second: Landmark) -> tuple[float, float, float]:
    return (
        (first.x + second.x) / 2.0,
        (first.y + second.y) / 2.0,
        (first.z + second.z) / 2.0,
    )


def extract_posture_metrics(
    observation: PoseObservation,
    *,
    minimum_confidence: float = 0.5,
) -> PostureMetrics:
    """Extract calibration-ready metrics from one normalized pose observation.

    Ratios are normalized by shoulder width. Horizontal reflection preserves
    shoulder tilt and vertical/depth ratios while reversing torso lean and head
    lateral offset signs. Every metric is independently confidence-gated.
    """

    shoulders = _landmarks(
        observation,
        (LandmarkName.LEFT_SHOULDER, LandmarkName.RIGHT_SHOULDER),
    )
    hips = _landmarks(observation, (LandmarkName.LEFT_HIP, LandmarkName.RIGHT_HIP))
    nose_items = _landmarks(observation, (LandmarkName.NOSE,))

    shoulder_confidence = _minimum_confidence(shoulders or ())
    if shoulders is None:
        shoulder_tilt = _unavailable(0.0, "missing_landmarks")
        shoulder_width: float | None = None
        shoulder_midpoint: tuple[float, float, float] | None = None
    elif shoulder_confidence < minimum_confidence:
        shoulder_tilt = _unavailable(shoulder_confidence, "low_confidence")
        shoulder_width = None
        shoulder_midpoint = None
    else:
        left_shoulder, right_shoulder = shoulders
        shoulder_width = math.hypot(
            right_shoulder.x - left_shoulder.x,
            right_shoulder.y - left_shoulder.y,
        )
        shoulder_midpoint = _midpoint(left_shoulder, right_shoulder)
        if shoulder_width <= _GEOMETRY_EPSILON:
            shoulder_tilt = _unavailable(shoulder_confidence, "degenerate_geometry")
            shoulder_width = None
        else:
            tilt = math.degrees(
                math.atan2(
                    right_shoulder.y - left_shoulder.y,
                    abs(right_shoulder.x - left_shoulder.x),
                )
            )
            shoulder_tilt = MetricReading(value=tilt, confidence=shoulder_confidence)

    torso_points = (*shoulders, *hips) if shoulders is not None and hips is not None else None
    torso_confidence = _minimum_confidence(torso_points or ())
    if torso_points is None:
        torso_lean = _unavailable(0.0, "missing_landmarks")
    elif torso_confidence < minimum_confidence:
        torso_lean = _unavailable(torso_confidence, "low_confidence")
    else:
        left_shoulder, right_shoulder, left_hip, right_hip = torso_points
        shoulder_mid_x, shoulder_mid_y, _ = _midpoint(left_shoulder, right_shoulder)
        hip_mid_x, hip_mid_y, _ = _midpoint(left_hip, right_hip)
        torso_dx = shoulder_mid_x - hip_mid_x
        torso_dy = shoulder_mid_y - hip_mid_y
        if math.hypot(torso_dx, torso_dy) <= _GEOMETRY_EPSILON:
            torso_lean = _unavailable(torso_confidence, "degenerate_geometry")
        else:
            torso_lean = MetricReading(
                value=math.degrees(math.atan2(torso_dx, -torso_dy)),
                confidence=torso_confidence,
            )

    head_points = (*shoulders, *nose_items) if shoulders is not None and nose_items else None
    head_confidence = _minimum_confidence(head_points or ())
    if head_points is None:
        unavailable_head = _unavailable(0.0, "missing_landmarks")
        head_lateral = head_vertical = head_depth = unavailable_head
    elif head_confidence < minimum_confidence:
        unavailable_head = _unavailable(head_confidence, "low_confidence")
        head_lateral = head_vertical = head_depth = unavailable_head
    elif shoulder_width is None or shoulder_midpoint is None:
        unavailable_head = _unavailable(head_confidence, "degenerate_geometry")
        head_lateral = head_vertical = head_depth = unavailable_head
    else:
        nose = head_points[-1]
        shoulder_mid_x, shoulder_mid_y, shoulder_mid_z = shoulder_midpoint
        head_lateral = MetricReading(
            value=(nose.x - shoulder_mid_x) / shoulder_width,
            confidence=head_confidence,
        )
        head_vertical = MetricReading(
            value=(shoulder_mid_y - nose.y) / shoulder_width,
            confidence=head_confidence,
        )
        head_depth = MetricReading(
            value=(shoulder_mid_z - nose.z) / shoulder_width,
            confidence=head_confidence,
        )

    return PostureMetrics(
        shoulder_tilt_degrees=shoulder_tilt,
        torso_lean_degrees=torso_lean,
        head_lateral_offset_ratio=head_lateral,
        head_vertical_offset_ratio=head_vertical,
        head_depth_ratio=head_depth,
    )
