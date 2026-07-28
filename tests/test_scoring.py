from __future__ import annotations

import math
from dataclasses import replace

import pytest

from goodposture.core.calibration import (
    CALIBRATION_CONFIG_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    CalibrationBaseline,
    CalibrationFeature,
)
from goodposture.core.models import MetricReading, PostureMetrics
from goodposture.core.scoring import (
    DEFAULT_SCORING_CONFIG,
    SCORING_CONFIG_VERSION,
    ScoreState,
    TimeAwareScoreSmoother,
    score_posture,
)


def baseline() -> CalibrationBaseline:
    return CalibrationBaseline(
        calibration_schema_version=CALIBRATION_SCHEMA_VERSION,
        calibration_config_version=CALIBRATION_CONFIG_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_id="pose-landmarker-lite@sha256:abc",
        started_at_ms=0,
        completed_at_ms=1_000,
        accepted_sample_count=60,
        rejected_sample_count=0,
        features=(
            CalibrationFeature("shoulder_tilt_degrees", 0.0, 1.0),
            CalibrationFeature("torso_lean_degrees", 0.0, 1.0),
            CalibrationFeature("head_lateral_offset_ratio", 0.0, 0.02),
            CalibrationFeature("head_vertical_offset_ratio", 0.73, 0.02),
            CalibrationFeature("head_depth_ratio", 0.60, 0.02),
        ),
    )


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


def test_controlled_deviation_changes_score_monotonically() -> None:
    scores = [
        score_posture(metrics(torso_lean=value), baseline()).score
        for value in (0.0, 3.0, 6.0, 12.0)
    ]

    assert all(score is not None for score in scores)
    numeric_scores = [score for score in scores if score is not None]
    assert numeric_scores == sorted(numeric_scores)
    assert len(set(numeric_scores)) == len(numeric_scores)


def test_baseline_like_observation_scores_neutral() -> None:
    result = score_posture(metrics(), baseline())

    assert result.state is ScoreState.AVAILABLE
    assert result.score == pytest.approx(0.0)
    assert result.confidence == pytest.approx(0.95)


def test_combined_extreme_deviation_is_bounded_at_one_hundred() -> None:
    result = score_posture(
        metrics(
            shoulder_tilt=100.0,
            torso_lean=100.0,
            head_lateral=5.0,
            head_vertical=5.0,
            head_depth=5.0,
        ),
        baseline(),
    )

    assert result.score == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("confidence", "expected_state"),
    (
        (0.69, ScoreState.UNKNOWN),
        (0.70, ScoreState.AVAILABLE),
    ),
)
def test_overall_confidence_boundary_is_explicit(
    confidence: float,
    expected_state: ScoreState,
) -> None:
    result = score_posture(metrics(confidence=confidence), baseline())

    assert result.state is expected_state
    assert (result.score is None) is (expected_state is ScoreState.UNKNOWN)


def test_missing_feature_is_allowed_only_when_coverage_remains_sufficient() -> None:
    without_head_depth = replace(
        metrics(torso_lean=6.0),
        head_depth_ratio=MetricReading(None, 0.95, "missing_landmarks"),
    )

    enough_coverage = score_posture(without_head_depth, baseline())
    too_little_coverage = score_posture(
        replace(
            without_head_depth,
            head_vertical_offset_ratio=MetricReading(None, 0.95, "missing_landmarks"),
        ),
        baseline(),
    )

    assert enough_coverage.state is ScoreState.AVAILABLE
    assert too_little_coverage.state is ScoreState.UNKNOWN


def test_laptop_baseline_detects_directional_head_hunch_without_hips() -> None:
    laptop_baseline = replace(
        baseline(),
        features=tuple(
            feature
            for feature in baseline().features
            if feature.name != "torso_lean_degrees"
        ),
    )

    neutral = score_posture(metrics(torso_lean=100.0), laptop_baseline)
    head_forward = score_posture(
        metrics(torso_lean=100.0, head_depth=0.90),
        laptop_baseline,
    )
    moved_away = score_posture(
        metrics(torso_lean=100.0, head_depth=0.30),
        laptop_baseline,
    )

    assert neutral.state is ScoreState.AVAILABLE
    assert neutral.score == pytest.approx(0.0)
    assert head_forward.state is ScoreState.AVAILABLE
    assert head_forward.score is not None and head_forward.score >= 60.0
    assert head_forward.head_hunch_score == pytest.approx(100.0)
    assert moved_away.score is not None and moved_away.score < 60.0
    assert moved_away.head_hunch_score == pytest.approx(0.0)
    assert all(
        item.name != "torso_lean_degrees" for item in head_forward.feature_deviations
    )


def test_laptop_baseline_detects_head_lowering_without_forward_motion() -> None:
    laptop_baseline = replace(
        baseline(),
        features=tuple(
            feature
            for feature in baseline().features
            if feature.name != "torso_lean_degrees"
        ),
    )

    head_down = score_posture(
        metrics(head_vertical=0.45, head_depth=0.60),
        laptop_baseline,
    )
    head_up = score_posture(
        metrics(head_vertical=1.01, head_depth=0.60),
        laptop_baseline,
    )

    assert head_down.score is not None and head_down.score >= 60.0
    assert head_down.head_hunch_score == pytest.approx(100.0)
    assert head_up.score is not None and head_up.score < 60.0
    assert head_up.head_hunch_score == pytest.approx(0.0)


def test_unknown_input_cannot_worsen_smoothed_score() -> None:
    smoother = TimeAwareScoreSmoother(DEFAULT_SCORING_CONFIG)
    neutral = score_posture(metrics(), baseline())
    deviated = score_posture(metrics(torso_lean=12.0), baseline())
    unreliable = score_posture(metrics(torso_lean=100.0, confidence=0.20), baseline())

    first = smoother.update(timestamp_ms=0, reading=neutral)
    second = smoother.update(timestamp_ms=1_000, reading=deviated)
    unknown = smoother.update(timestamp_ms=1_100, reading=unreliable)
    recovered = smoother.update(timestamp_ms=1_200, reading=neutral)

    assert first.smoothed_score == pytest.approx(0.0)
    assert second.smoothed_score is not None and second.smoothed_score > 0.0
    assert unknown.state is ScoreState.UNKNOWN
    assert unknown.raw_score is None
    assert unknown.smoothed_score is None
    assert recovered.smoothed_score is not None
    assert recovered.smoothed_score < second.smoothed_score


def test_time_aware_smoothing_is_stable_across_frame_rates() -> None:
    neutral = score_posture(metrics(), baseline())
    deviated = score_posture(metrics(torso_lean=12.0), baseline())
    regular = TimeAwareScoreSmoother(DEFAULT_SCORING_CONFIG)
    irregular = TimeAwareScoreSmoother(DEFAULT_SCORING_CONFIG)
    regular.update(timestamp_ms=0, reading=neutral)
    irregular.update(timestamp_ms=0, reading=neutral)

    regular_result = None
    for timestamp_ms in range(100, 1_001, 100):
        regular_result = regular.update(timestamp_ms=timestamp_ms, reading=deviated)
    irregular_result = None
    for timestamp_ms in (125, 400, 550, 1_000):
        irregular_result = irregular.update(timestamp_ms=timestamp_ms, reading=deviated)

    assert regular_result is not None
    assert irregular_result is not None
    assert regular_result.smoothed_score == pytest.approx(
        irregular_result.smoothed_score,
        abs=1e-10,
    )


def test_long_gap_resets_stale_smoothed_score() -> None:
    config = replace(DEFAULT_SCORING_CONFIG, gap_reset_ms=5_000)
    smoother = TimeAwareScoreSmoother(config)
    deviated = score_posture(metrics(torso_lean=12.0), baseline(), config)
    neutral = score_posture(metrics(), baseline(), config)

    before_gap = smoother.update(timestamp_ms=0, reading=deviated)
    after_gap = smoother.update(timestamp_ms=5_000, reading=neutral)

    assert before_gap.smoothed_score is not None and before_gap.smoothed_score > 0.0
    assert after_gap.smoothed_score == pytest.approx(0.0)


def test_scoring_rejects_incompatible_or_non_finite_calibration() -> None:
    incompatible = replace(baseline(), feature_schema_version=999)
    invalid_features = list(baseline().features)
    invalid_features[0] = replace(invalid_features[0], dispersion=math.nan)
    non_finite = replace(baseline(), features=tuple(invalid_features))

    with pytest.raises(ValueError, match="feature schema"):
        score_posture(metrics(), incompatible)
    with pytest.raises(ValueError, match="statistics"):
        score_posture(metrics(), non_finite)


def test_scoring_configuration_is_versioned_and_rejects_invalid_values() -> None:
    assert DEFAULT_SCORING_CONFIG.version == SCORING_CONFIG_VERSION
    with pytest.raises(ValueError, match="smoothing_time_constant_ms"):
        replace(DEFAULT_SCORING_CONFIG, smoothing_time_constant_ms=0)
