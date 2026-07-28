from __future__ import annotations

from dataclasses import dataclass

import pytest

from goodposture.adapters.landmark_mapping import map_pose_landmarks
from goodposture.core.models import LandmarkName


@dataclass
class SourceLandmark:
    x: float
    y: float
    z: float
    visibility: float | None = 1.0
    presence: float | None = 1.0


def test_maps_mediapipe_indices_to_framework_neutral_observation() -> None:
    source = [
        SourceLandmark(
            x=index / 100,
            y=index / 200,
            z=-index / 300,
            visibility=0.9,
            presence=0.8,
        )
        for index in range(33)
    ]

    detected = map_pose_landmarks(source, timestamp_ms=456)

    assert detected is not None
    assert detected.observation.timestamp_ms == 456
    assert len(detected.all_landmarks) == 33
    assert detected.observation.landmarks[LandmarkName.NOSE].x == pytest.approx(0.00)
    assert detected.observation.landmarks[LandmarkName.LEFT_EYE].x == pytest.approx(0.02)
    assert detected.observation.landmarks[LandmarkName.RIGHT_EYE].x == pytest.approx(0.05)
    assert detected.observation.landmarks[LandmarkName.LEFT_SHOULDER].x == pytest.approx(
        0.11
    )
    assert detected.observation.landmarks[LandmarkName.RIGHT_SHOULDER].x == pytest.approx(
        0.12
    )
    assert detected.observation.landmarks[LandmarkName.LEFT_HIP].x == pytest.approx(0.23)
    assert detected.observation.landmarks[LandmarkName.RIGHT_HIP].x == pytest.approx(0.24)
    assert (
        detected.observation.landmarks[LandmarkName.LEFT_SHOULDER].confidence
        == pytest.approx(0.8)
    )


def test_none_confidence_values_are_conservatively_unreliable() -> None:
    source = [SourceLandmark(x=0.5, y=0.5, z=0.0) for _ in range(33)]
    source[0].visibility = None
    source[0].presence = None

    detected = map_pose_landmarks(source, timestamp_ms=1)

    assert detected is not None
    assert detected.observation.landmarks[LandmarkName.NOSE].confidence == 0.0


def test_rejects_incomplete_pose_output() -> None:
    source = [SourceLandmark(x=0.5, y=0.5, z=0.0) for _ in range(24)]

    assert map_pose_landmarks(source, timestamp_ms=1) is None
