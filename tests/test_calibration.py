from __future__ import annotations

from dataclasses import replace

import pytest

from goodposture.core.calibration import (
    CALIBRATION_CONFIG_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    CalibrationAccumulator,
    CalibrationBaseline,
    CalibrationConfig,
    CalibrationFailure,
    CalibrationValidity,
)
from goodposture.core.models import MetricReading, PostureMetrics


def metrics(
    *,
    shoulder_tilt: float = 0.0,
    torso_lean: float = 0.0,
    head_lateral: float = 0.0,
    head_vertical: float = 0.73,
    head_depth: float = 0.60,
    confidence: float = 0.95,
) -> PostureMetrics:
    return PostureMetrics(
        shoulder_tilt_degrees=MetricReading(shoulder_tilt, confidence),
        torso_lean_degrees=MetricReading(torso_lean, confidence),
        head_lateral_offset_ratio=MetricReading(head_lateral, confidence),
        head_vertical_offset_ratio=MetricReading(head_vertical, confidence),
        head_depth_ratio=MetricReading(head_depth, confidence),
    )


def config() -> CalibrationConfig:
    return CalibrationConfig(
        minimum_accepted_samples=5,
        minimum_sample_coverage=0.70,
        minimum_metric_confidence=0.80,
        maximum_sample_span_ms=10_000,
    )


def test_good_sequence_produces_a_versioned_derived_baseline() -> None:
    accumulator = CalibrationAccumulator(config())
    for index, adjustment in enumerate((-0.02, -0.01, 0.0, 0.01, 0.02)):
        accumulator.add(
            timestamp_ms=1_000 + (index * 100),
            metrics=metrics(
                shoulder_tilt=adjustment,
                torso_lean=adjustment,
                head_lateral=adjustment / 10,
                head_vertical=0.73 + adjustment / 10,
                head_depth=0.60 + adjustment / 10,
            ),
        )

    result = accumulator.finish(
        completed_at_ms=2_000,
        model_id="pose-landmarker-lite@sha256:abc",
    )

    assert result.failure is None
    assert result.baseline is not None
    assert result.baseline.calibration_schema_version == CALIBRATION_SCHEMA_VERSION
    assert result.baseline.calibration_config_version == CALIBRATION_CONFIG_VERSION
    assert result.baseline.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert result.baseline.accepted_sample_count == 5
    assert result.baseline.rejected_sample_count == 0
    assert result.baseline.feature("torso_lean_degrees").median == pytest.approx(0.0)
    assert result.baseline.feature("head_depth_ratio").dispersion > 0.0


def test_laptop_framing_calibrates_without_hips_or_torso_lean() -> None:
    accumulator = CalibrationAccumulator(config())
    without_torso = replace(
        metrics(),
        torso_lean_degrees=MetricReading(None, 0.0, "missing_landmarks"),
    )

    for index in range(5):
        assert accumulator.add(
            timestamp_ms=1_000 + index * 100,
            metrics=without_torso,
        )

    result = accumulator.finish(
        completed_at_ms=2_000,
        model_id="pose-landmarker-lite@sha256:abc",
    )

    assert result.baseline is not None
    assert result.baseline.accepted_sample_count == 5
    assert {feature.name for feature in result.baseline.features} == {
        "shoulder_tilt_degrees",
        "head_lateral_offset_ratio",
        "head_vertical_offset_ratio",
        "head_depth_ratio",
    }
    with pytest.raises(KeyError, match="torso_lean_degrees"):
        result.baseline.feature("torso_lean_degrees")


def test_brief_outliers_do_not_materially_shift_the_baseline() -> None:
    accumulator = CalibrationAccumulator(
        replace(config(), minimum_accepted_samples=7, minimum_sample_coverage=0.60)
    )
    stable_values = (-0.02, -0.01, 0.0, 0.0, 0.01, 0.02)
    for index, value in enumerate((*stable_values, 40.0)):
        accumulator.add(
            timestamp_ms=1_000 + (index * 100),
            metrics=metrics(
                shoulder_tilt=value,
                torso_lean=value,
                head_lateral=value / 10,
            ),
        )

    result = accumulator.finish(
        completed_at_ms=2_000,
        model_id="pose-landmarker-lite@sha256:abc",
    )

    assert result.baseline is not None
    assert result.baseline.feature("torso_lean_degrees").median == pytest.approx(0.0)
    assert result.baseline.feature("torso_lean_degrees").dispersion < 0.05


def test_low_confidence_and_missing_metrics_count_against_coverage() -> None:
    accumulator = CalibrationAccumulator(config())
    for index in range(5):
        accumulator.add(timestamp_ms=1_000 + index * 100, metrics=metrics())
    accumulator.add(timestamp_ms=1_600, metrics=metrics(confidence=0.20))
    missing_head = replace(
        metrics(),
        head_depth_ratio=MetricReading(None, 0.95, "missing_landmarks"),
    )
    accumulator.add(timestamp_ms=1_700, metrics=missing_head)

    result = accumulator.finish(
        completed_at_ms=2_000,
        model_id="pose-landmarker-lite@sha256:abc",
    )

    assert result.baseline is not None
    assert result.baseline.accepted_sample_count == 5
    assert result.baseline.rejected_sample_count == 2


def test_insufficient_confident_coverage_fails_without_a_partial_baseline() -> None:
    accumulator = CalibrationAccumulator(config())
    for index in range(4):
        accumulator.add(timestamp_ms=1_000 + index * 100, metrics=metrics())
    for index in range(3):
        accumulator.add(
            timestamp_ms=1_500 + index * 100,
            metrics=metrics(confidence=0.20),
        )

    result = accumulator.finish(
        completed_at_ms=2_000,
        model_id="pose-landmarker-lite@sha256:abc",
    )

    assert result.baseline is None
    assert result.failure is CalibrationFailure.INSUFFICIENT_CONFIDENT_SAMPLES
    assert "framing or lighting" in result.guidance


def test_unstable_sequence_fails_without_a_partial_baseline() -> None:
    accumulator = CalibrationAccumulator(config())
    for index, value in enumerate((-15.0, -8.0, 0.0, 8.0, 15.0)):
        accumulator.add(
            timestamp_ms=1_000 + index * 100,
            metrics=metrics(shoulder_tilt=value, torso_lean=value),
        )

    result = accumulator.finish(
        completed_at_ms=2_000,
        model_id="pose-landmarker-lite@sha256:abc",
    )

    assert result.baseline is None
    assert result.failure is CalibrationFailure.UNSTABLE_POSE
    assert "comfortable position" in result.guidance


def test_interrupted_sequence_cannot_be_finished_or_reused() -> None:
    accumulator = CalibrationAccumulator(config())
    for index in range(5):
        accumulator.add(timestamp_ms=1_000 + index * 100, metrics=metrics())

    accumulator.interrupt()
    result = accumulator.finish(
        completed_at_ms=2_000,
        model_id="pose-landmarker-lite@sha256:abc",
    )

    assert result.baseline is None
    assert result.failure is CalibrationFailure.INTERRUPTED
    with pytest.raises(RuntimeError, match="no longer collecting"):
        accumulator.add(timestamp_ms=2_100, metrics=metrics())


def test_baseline_round_trips_through_versioned_serialization() -> None:
    accumulator = CalibrationAccumulator(config())
    for index in range(5):
        accumulator.add(timestamp_ms=1_000 + index * 100, metrics=metrics())
    baseline = accumulator.finish(
        completed_at_ms=2_000,
        model_id="pose-landmarker-lite@sha256:abc",
    ).baseline
    assert baseline is not None

    restored = type(baseline).from_dict(baseline.to_dict())

    assert restored == baseline
    assert "landmarks" not in restored.to_dict()
    assert "samples" not in restored.to_dict()


def test_deserialization_rejects_inconsistent_quality_metadata() -> None:
    accumulator = CalibrationAccumulator(config())
    for index in range(5):
        accumulator.add(timestamp_ms=1_000 + index * 100, metrics=metrics())
    baseline = accumulator.finish(
        completed_at_ms=2_000,
        model_id="pose-landmarker-lite@sha256:abc",
    ).baseline
    assert baseline is not None
    data = baseline.to_dict()
    data["accepted_sample_count"] = -1

    with pytest.raises(ValueError, match="accepted_sample_count"):
        CalibrationBaseline.from_dict(data)


def test_accumulator_times_out_without_growing_past_its_bounded_window() -> None:
    accumulator = CalibrationAccumulator(replace(config(), maximum_sample_span_ms=300))
    for index in range(4):
        assert accumulator.add(timestamp_ms=1_000 + index * 100, metrics=metrics())

    assert not accumulator.add(timestamp_ms=1_401, metrics=metrics())
    with pytest.raises(RuntimeError, match="no longer collecting"):
        accumulator.add(timestamp_ms=1_500, metrics=metrics())

    result = accumulator.finish(
        completed_at_ms=2_000,
        model_id="pose-landmarker-lite@sha256:abc",
    )

    assert result.baseline is None
    assert result.failure is CalibrationFailure.CAPTURE_TIMED_OUT


def test_validity_policy_requires_fresh_compatible_calibration() -> None:
    accumulator = CalibrationAccumulator(config())
    for index in range(5):
        accumulator.add(timestamp_ms=1_000 + index * 100, metrics=metrics())
    baseline = accumulator.finish(
        completed_at_ms=2_000,
        model_id="pose-landmarker-lite@sha256:abc",
    ).baseline
    assert baseline is not None

    assert (
        baseline.validity(
            at_timestamp_ms=3_000,
            maximum_age_ms=10_000,
            expected_model_id=baseline.model_id,
            expected_feature_schema_version=baseline.feature_schema_version,
            expected_calibration_config_version=baseline.calibration_config_version,
        )
        is CalibrationValidity.VALID
    )
    assert (
        baseline.validity(
            at_timestamp_ms=20_001,
            maximum_age_ms=10_000,
            expected_model_id=baseline.model_id,
            expected_feature_schema_version=baseline.feature_schema_version,
            expected_calibration_config_version=baseline.calibration_config_version,
        )
        is CalibrationValidity.EXPIRED
    )
    assert (
        baseline.validity(
            at_timestamp_ms=3_000,
            maximum_age_ms=10_000,
            expected_model_id="different-model",
            expected_feature_schema_version=baseline.feature_schema_version,
            expected_calibration_config_version=baseline.calibration_config_version,
        )
        is CalibrationValidity.MODEL_CHANGED
    )
    assert (
        baseline.validity(
            at_timestamp_ms=3_000,
            maximum_age_ms=10_000,
            expected_model_id=baseline.model_id,
            expected_feature_schema_version=999,
            expected_calibration_config_version=baseline.calibration_config_version,
        )
        is CalibrationValidity.FEATURE_SCHEMA_CHANGED
    )
    assert (
        baseline.validity(
            at_timestamp_ms=3_000,
            maximum_age_ms=10_000,
            expected_model_id=baseline.model_id,
            expected_feature_schema_version=baseline.feature_schema_version,
            expected_calibration_config_version=999,
        )
        is CalibrationValidity.CONFIG_CHANGED
    )
