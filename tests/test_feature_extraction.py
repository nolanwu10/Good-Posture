from __future__ import annotations

import math

import pytest

from goodposture.core.metrics import extract_posture_metrics
from goodposture.core.models import Landmark, LandmarkName, PoseObservation


def upright_landmarks() -> dict[LandmarkName, Landmark]:
    return {
        LandmarkName.NOSE: Landmark(x=0.50, y=0.20, z=-0.18),
        LandmarkName.LEFT_SHOULDER: Landmark(x=0.35, y=0.42, z=0.00),
        LandmarkName.RIGHT_SHOULDER: Landmark(x=0.65, y=0.42, z=0.00),
        LandmarkName.LEFT_HIP: Landmark(x=0.40, y=0.75, z=0.04),
        LandmarkName.RIGHT_HIP: Landmark(x=0.60, y=0.75, z=0.04),
    }


def observation(
    landmarks: dict[LandmarkName, Landmark] | None = None,
) -> PoseObservation:
    return PoseObservation(timestamp_ms=123, landmarks=landmarks or upright_landmarks())


def test_extracts_initial_metrics_for_a_confident_pose() -> None:
    metrics = extract_posture_metrics(observation())

    assert metrics.shoulder_tilt_degrees.value == pytest.approx(0.0)
    assert metrics.torso_lean_degrees.value == pytest.approx(0.0)
    assert metrics.head_lateral_offset_ratio.value == pytest.approx(0.0)
    assert metrics.head_vertical_offset_ratio.value == pytest.approx(0.22 / 0.30)
    assert metrics.head_depth_ratio.value == pytest.approx(0.18 / 0.30)


def test_metrics_are_invariant_under_translation_and_uniform_scale() -> None:
    original = upright_landmarks()
    transformed = {
        name: Landmark(
            x=(landmark.x * 1.7) - 0.2,
            y=(landmark.y * 1.7) + 0.1,
            z=landmark.z * 1.7,
            visibility=landmark.visibility,
            presence=landmark.presence,
        )
        for name, landmark in original.items()
    }

    first = extract_posture_metrics(observation(original))
    second = extract_posture_metrics(observation(transformed))

    assert second.shoulder_tilt_degrees.value == pytest.approx(
        first.shoulder_tilt_degrees.value
    )
    assert second.torso_lean_degrees.value == pytest.approx(first.torso_lean_degrees.value)
    assert second.head_lateral_offset_ratio.value == pytest.approx(
        first.head_lateral_offset_ratio.value
    )
    assert second.head_vertical_offset_ratio.value == pytest.approx(
        first.head_vertical_offset_ratio.value
    )
    assert second.head_depth_ratio.value == pytest.approx(first.head_depth_ratio.value)


def test_low_confidence_nose_invalidates_only_head_metrics() -> None:
    landmarks = upright_landmarks()
    landmarks[LandmarkName.NOSE] = Landmark(
        x=0.50,
        y=0.20,
        z=-0.18,
        visibility=0.20,
        presence=0.90,
    )

    metrics = extract_posture_metrics(observation(landmarks), minimum_confidence=0.5)

    assert metrics.shoulder_tilt_degrees.value is not None
    assert metrics.torso_lean_degrees.value is not None
    assert metrics.head_lateral_offset_ratio.value is None
    assert metrics.head_vertical_offset_ratio.value is None
    assert metrics.head_depth_ratio.value is None
    assert metrics.head_lateral_offset_ratio.confidence == pytest.approx(0.20)


def test_missing_hips_invalidates_torso_lean_only() -> None:
    landmarks = upright_landmarks()
    del landmarks[LandmarkName.LEFT_HIP]
    del landmarks[LandmarkName.RIGHT_HIP]

    metrics = extract_posture_metrics(observation(landmarks))

    assert metrics.shoulder_tilt_degrees.value is not None
    assert metrics.torso_lean_degrees.value is None
    assert metrics.head_vertical_offset_ratio.value is not None


def test_degenerate_shoulder_width_does_not_produce_non_finite_ratios() -> None:
    landmarks = upright_landmarks()
    landmarks[LandmarkName.RIGHT_SHOULDER] = landmarks[LandmarkName.LEFT_SHOULDER]

    metrics = extract_posture_metrics(observation(landmarks))

    assert metrics.shoulder_tilt_degrees.value is None
    assert metrics.head_lateral_offset_ratio.value is None
    assert metrics.head_vertical_offset_ratio.value is None
    assert metrics.head_depth_ratio.value is None
    for metric in metrics.as_dict().values():
        assert metric is None or math.isfinite(metric)


def test_mirrored_pose_preserves_magnitudes_and_flips_lateral_signs() -> None:
    landmarks = upright_landmarks()
    landmarks[LandmarkName.NOSE] = Landmark(x=0.56, y=0.20, z=-0.18)
    landmarks[LandmarkName.LEFT_HIP] = Landmark(x=0.45, y=0.75, z=0.04)
    landmarks[LandmarkName.RIGHT_HIP] = Landmark(x=0.65, y=0.75, z=0.04)
    mirrored = {
        name: Landmark(
            x=1.0 - landmark.x,
            y=landmark.y,
            z=landmark.z,
            visibility=landmark.visibility,
            presence=landmark.presence,
        )
        for name, landmark in landmarks.items()
    }

    first = extract_posture_metrics(observation(landmarks))
    second = extract_posture_metrics(observation(mirrored))

    assert second.shoulder_tilt_degrees.value == pytest.approx(
        first.shoulder_tilt_degrees.value
    )
    assert second.torso_lean_degrees.value == pytest.approx(
        -first.torso_lean_degrees.value
    )
    assert second.head_lateral_offset_ratio.value == pytest.approx(
        -first.head_lateral_offset_ratio.value
    )
    assert second.head_vertical_offset_ratio.value == pytest.approx(
        first.head_vertical_offset_ratio.value
    )
    assert second.head_depth_ratio.value == pytest.approx(first.head_depth_ratio.value)
