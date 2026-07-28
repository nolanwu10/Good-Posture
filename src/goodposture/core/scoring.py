"""Confidence-aware posture scoring and time-aware smoothing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from goodposture.core.calibration import FEATURE_SCHEMA_VERSION, CalibrationBaseline
from goodposture.core.models import MetricReading, PostureMetrics

SCORING_CONFIG_VERSION: Final = 2


class ScoreState(StrEnum):
    """Whether a score is safe to use for awareness decisions."""

    AVAILABLE = "available"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FeatureScoringRule:
    """Weight and personalized tolerance policy for one feature."""

    name: str
    weight: float
    minimum_scale: float
    dispersion_multiplier: float = 3.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("feature rule name must be non-empty")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("feature rule weight must be positive and finite")
        if not math.isfinite(self.minimum_scale) or self.minimum_scale <= 0.0:
            raise ValueError("feature rule minimum_scale must be positive and finite")
        if (
            not math.isfinite(self.dispersion_multiplier)
            or self.dispersion_multiplier <= 0.0
        ):
            raise ValueError(
                "feature rule dispersion_multiplier must be positive and finite"
            )


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """Versioned weights, thresholds, and temporal smoothing parameters."""

    version: int
    rules: tuple[FeatureScoringRule, ...]
    minimum_metric_confidence: float = 0.50
    minimum_feature_coverage: float = 0.70
    minimum_overall_confidence: float = 0.70
    maximum_standardized_deviation: float = 4.0
    smoothing_time_constant_ms: int = 1_000
    gap_reset_ms: int = 5_000

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("scoring config version must be positive")
        if not self.rules:
            raise ValueError("scoring config must contain feature rules")
        names = tuple(rule.name for rule in self.rules)
        if len(set(names)) != len(names):
            raise ValueError("scoring feature rule names must be unique")
        for name, value in (
            ("minimum_metric_confidence", self.minimum_metric_confidence),
            ("minimum_feature_coverage", self.minimum_feature_coverage),
            ("minimum_overall_confidence", self.minimum_overall_confidence),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if (
            not math.isfinite(self.maximum_standardized_deviation)
            or self.maximum_standardized_deviation <= 0.0
        ):
            raise ValueError("maximum_standardized_deviation must be positive and finite")
        if self.smoothing_time_constant_ms <= 0:
            raise ValueError("smoothing_time_constant_ms must be positive")
        if self.gap_reset_ms <= 0:
            raise ValueError("gap_reset_ms must be positive")


DEFAULT_SCORING_CONFIG: Final = ScoringConfig(
    version=SCORING_CONFIG_VERSION,
    rules=(
        FeatureScoringRule("shoulder_tilt_degrees", weight=0.15, minimum_scale=1.5),
        FeatureScoringRule("torso_lean_degrees", weight=0.25, minimum_scale=1.5),
        FeatureScoringRule(
            "head_lateral_offset_ratio",
            weight=0.10,
            minimum_scale=0.04,
        ),
        FeatureScoringRule(
            "head_vertical_offset_ratio",
            weight=0.20,
            minimum_scale=0.04,
        ),
        FeatureScoringRule("head_depth_ratio", weight=0.30, minimum_scale=0.04),
    ),
)


@dataclass(frozen=True, slots=True)
class FeatureDeviation:
    """One explainable, bounded feature contribution."""

    name: str
    standardized_deviation: float
    confidence: float


@dataclass(frozen=True, slots=True)
class ScoreReading:
    """An instantaneous score before temporal smoothing."""

    state: ScoreState
    score: float | None
    confidence: float
    feature_deviations: tuple[FeatureDeviation, ...]
    config_version: int
    head_hunch_score: float | None = None


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """A timestamped score suitable for downstream state policies."""

    timestamp_ms: int
    state: ScoreState
    raw_score: float | None
    smoothed_score: float | None
    confidence: float
    feature_deviations: tuple[FeatureDeviation, ...]
    config_version: int
    head_hunch_score: float | None = None


def _metric_readings(metrics: PostureMetrics) -> dict[str, MetricReading]:
    return {
        "shoulder_tilt_degrees": metrics.shoulder_tilt_degrees,
        "torso_lean_degrees": metrics.torso_lean_degrees,
        "head_lateral_offset_ratio": metrics.head_lateral_offset_ratio,
        "head_vertical_offset_ratio": metrics.head_vertical_offset_ratio,
        "head_depth_ratio": metrics.head_depth_ratio,
    }


def score_posture(
    metrics: PostureMetrics,
    baseline: CalibrationBaseline,
    config: ScoringConfig = DEFAULT_SCORING_CONFIG,
) -> ScoreReading:
    """Compare derived metrics with a personalized baseline.

    Deviations are symmetric awareness signals around the person's baseline;
    they are not medical judgments or universal posture targets.
    """

    if baseline.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError("calibration feature schema is incompatible with scoring")
    readings = _metric_readings(metrics)
    calibration_by_name = {feature.name: feature for feature in baseline.features}
    calibrated_rules = tuple(
        rule for rule in config.rules if rule.name in calibration_by_name
    )
    if not calibrated_rules:
        raise ValueError("calibration contains no scoring features")
    total_weight = sum(rule.weight for rule in calibrated_rules)
    available_weight = 0.0
    confidence_weighted_sum = 0.0
    effective_weight = 0.0
    weighted_deviation = 0.0
    deviations: list[FeatureDeviation] = []
    directional_head_deviations: list[float] = []

    for rule in calibrated_rules:
        if rule.name not in readings:
            raise ValueError(f"unknown scoring feature: {rule.name}")
        reading = readings[rule.name]
        value = reading.value
        if (
            value is None
            or not math.isfinite(value)
            or not math.isfinite(reading.confidence)
            or not 0.0 <= reading.confidence <= 1.0
            or reading.confidence < config.minimum_metric_confidence
        ):
            continue

        calibration = calibration_by_name[rule.name]
        if (
            not math.isfinite(calibration.median)
            or not math.isfinite(calibration.dispersion)
            or calibration.dispersion <= 0.0
        ):
            raise ValueError(f"invalid calibration statistics for {rule.name}")
        scale = max(
            rule.minimum_scale,
            calibration.dispersion * rule.dispersion_multiplier,
        )
        standardized = min(
            abs(value - calibration.median) / scale,
            config.maximum_standardized_deviation,
        )
        if rule.name == "head_depth_ratio":
            directional_head_deviations.append(
                min(
                    max((value - calibration.median) / scale, 0.0),
                    config.maximum_standardized_deviation,
                )
            )
        elif rule.name == "head_vertical_offset_ratio":
            directional_head_deviations.append(
                min(
                    max((calibration.median - value) / scale, 0.0),
                    config.maximum_standardized_deviation,
                )
            )
        deviations.append(
            FeatureDeviation(
                name=rule.name,
                standardized_deviation=standardized,
                confidence=reading.confidence,
            )
        )
        available_weight += rule.weight
        confidence_weighted_sum += rule.weight * reading.confidence
        confidence_adjusted_weight = rule.weight * reading.confidence
        effective_weight += confidence_adjusted_weight
        weighted_deviation += standardized * confidence_adjusted_weight

    coverage = available_weight / total_weight
    overall_confidence = (
        confidence_weighted_sum / available_weight if available_weight else 0.0
    )
    if (
        coverage < config.minimum_feature_coverage
        or overall_confidence < config.minimum_overall_confidence
        or effective_weight == 0.0
    ):
        return ScoreReading(
            state=ScoreState.UNKNOWN,
            score=None,
            confidence=overall_confidence,
            feature_deviations=tuple(deviations),
            config_version=config.version,
            head_hunch_score=None,
        )

    weighted_score = (
        weighted_deviation
        / effective_weight
        / config.maximum_standardized_deviation
        * 100.0
    )
    head_hunch_score = (
        max(directional_head_deviations, default=0.0)
        / config.maximum_standardized_deviation
        * 100.0
    )
    return ScoreReading(
        state=ScoreState.AVAILABLE,
        score=min(max(weighted_score, head_hunch_score, 0.0), 100.0),
        confidence=overall_confidence,
        feature_deviations=tuple(deviations),
        config_version=config.version,
        head_hunch_score=head_hunch_score,
    )


class TimeAwareScoreSmoother:
    """Smooth valid scores by elapsed time and ignore uncertain observations."""

    def __init__(self, config: ScoringConfig = DEFAULT_SCORING_CONFIG) -> None:
        self._config = config
        self._smoothed_score: float | None = None
        self._last_valid_timestamp_ms: int | None = None
        self._last_observation_timestamp_ms: int | None = None

    def update(self, *, timestamp_ms: int, reading: ScoreReading) -> ScoreResult:
        """Update smoothing state using an irregular timestamped observation."""

        if timestamp_ms < 0:
            raise ValueError("timestamp_ms cannot be negative")
        if (
            self._last_observation_timestamp_ms is not None
            and timestamp_ms <= self._last_observation_timestamp_ms
        ):
            raise ValueError("score timestamps must increase")
        if reading.config_version != self._config.version:
            raise ValueError("score reading and smoother config versions must match")
        self._last_observation_timestamp_ms = timestamp_ms

        if reading.state is ScoreState.UNKNOWN:
            return ScoreResult(
                timestamp_ms=timestamp_ms,
                state=ScoreState.UNKNOWN,
                raw_score=None,
                smoothed_score=None,
                confidence=reading.confidence,
                feature_deviations=reading.feature_deviations,
                config_version=self._config.version,
                head_hunch_score=None,
            )
        if reading.score is None:
            raise ValueError("available score reading must contain a score")

        if (
            self._smoothed_score is None
            or self._last_valid_timestamp_ms is None
            or timestamp_ms - self._last_valid_timestamp_ms >= self._config.gap_reset_ms
        ):
            smoothed_score = reading.score
        else:
            elapsed_ms = timestamp_ms - self._last_valid_timestamp_ms
            alpha = 1.0 - math.exp(
                -elapsed_ms / self._config.smoothing_time_constant_ms
            )
            smoothed_score = self._smoothed_score + alpha * (
                reading.score - self._smoothed_score
            )

        self._smoothed_score = smoothed_score
        self._last_valid_timestamp_ms = timestamp_ms
        return ScoreResult(
            timestamp_ms=timestamp_ms,
            state=ScoreState.AVAILABLE,
            raw_score=reading.score,
            smoothed_score=smoothed_score,
            confidence=reading.confidence,
            feature_deviations=reading.feature_deviations,
            config_version=self._config.version,
            head_hunch_score=reading.head_hunch_score,
        )
