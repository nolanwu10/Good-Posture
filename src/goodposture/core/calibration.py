"""Robust personalized calibration using derived posture metrics only."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from goodposture.core.models import MetricReading, PostureMetrics

CALIBRATION_SCHEMA_VERSION: Final = 2
CALIBRATION_CONFIG_VERSION: Final = 2
FEATURE_SCHEMA_VERSION: Final = 2

_FEATURE_NAMES: Final = (
    "shoulder_tilt_degrees",
    "torso_lean_degrees",
    "head_lateral_offset_ratio",
    "head_vertical_offset_ratio",
    "head_depth_ratio",
)
_REQUIRED_FEATURE_NAMES: Final = (
    "shoulder_tilt_degrees",
    "head_lateral_offset_ratio",
    "head_vertical_offset_ratio",
    "head_depth_ratio",
)
_OPTIONAL_FEATURE_NAMES: Final = ("torso_lean_degrees",)
_STABILITY_LIMITS: Final = {
    "shoulder_tilt_degrees": 5.0,
    "torso_lean_degrees": 5.0,
    "head_lateral_offset_ratio": 0.15,
    "head_vertical_offset_ratio": 0.15,
    "head_depth_ratio": 0.15,
}
_DISPERSION_FLOORS: Final = {
    "shoulder_tilt_degrees": 0.01,
    "torso_lean_degrees": 0.01,
    "head_lateral_offset_ratio": 0.005,
    "head_vertical_offset_ratio": 0.005,
    "head_depth_ratio": 0.005,
}


class CalibrationFailure(StrEnum):
    """Reasons an attempted calibration produced no baseline."""

    INSUFFICIENT_CONFIDENT_SAMPLES = "insufficient_confident_samples"
    UNSTABLE_POSE = "unstable_pose"
    INTERRUPTED = "interrupted"
    CAPTURE_TIMED_OUT = "capture_timed_out"


class CalibrationValidity(StrEnum):
    """Compatibility and freshness decision for a saved calibration."""

    VALID = "valid"
    EXPIRED = "expired"
    MODEL_CHANGED = "model_changed"
    FEATURE_SCHEMA_CHANGED = "feature_schema_changed"
    CONFIG_CHANGED = "config_changed"


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Quality requirements for one calibration attempt."""

    minimum_accepted_samples: int = 60
    minimum_sample_coverage: float = 0.80
    minimum_metric_confidence: float = 0.70
    maximum_sample_span_ms: int = 15_000

    def __post_init__(self) -> None:
        if self.minimum_accepted_samples < 1:
            raise ValueError("minimum_accepted_samples must be positive")
        if not 0.0 <= self.minimum_sample_coverage <= 1.0:
            raise ValueError("minimum_sample_coverage must be between 0 and 1")
        if not 0.0 <= self.minimum_metric_confidence <= 1.0:
            raise ValueError("minimum_metric_confidence must be between 0 and 1")
        if self.maximum_sample_span_ms <= 0:
            raise ValueError("maximum_sample_span_ms must be positive")


@dataclass(frozen=True, slots=True)
class CalibrationFeature:
    """Robust baseline statistics for one derived feature."""

    name: str
    median: float
    dispersion: float

    def to_dict(self) -> dict[str, str | float]:
        return {
            "name": self.name,
            "median": self.median,
            "dispersion": self.dispersion,
        }

    @classmethod
    def from_dict(cls, data: object) -> CalibrationFeature:
        if not isinstance(data, dict):
            raise ValueError("calibration feature must be an object")
        name = data.get("name")
        median = data.get("median")
        dispersion = data.get("dispersion")
        if not isinstance(name, str):
            raise ValueError("calibration feature name must be a string")
        if (
            not isinstance(median, (int, float))
            or not isinstance(dispersion, (int, float))
            or not math.isfinite(float(median))
            or not math.isfinite(float(dispersion))
            or float(dispersion) <= 0.0
        ):
            raise ValueError("calibration feature statistics must be finite")
        return cls(name=name, median=float(median), dispersion=float(dispersion))


@dataclass(frozen=True, slots=True)
class CalibrationBaseline:
    """A persisted-safe calibration containing no observations or landmarks."""

    calibration_schema_version: int
    calibration_config_version: int
    feature_schema_version: int
    model_id: str
    started_at_ms: int
    completed_at_ms: int
    accepted_sample_count: int
    rejected_sample_count: int
    features: tuple[CalibrationFeature, ...]

    def feature(self, name: str) -> CalibrationFeature:
        """Return one feature statistic by its stable contract name."""

        for feature in self.features:
            if feature.name == name:
                return feature
        raise KeyError(name)

    def validity(
        self,
        *,
        at_timestamp_ms: int,
        maximum_age_ms: int,
        expected_model_id: str,
        expected_feature_schema_version: int,
        expected_calibration_config_version: int,
    ) -> CalibrationValidity:
        """Decide whether this baseline can be reused or needs recalibration."""

        if self.calibration_config_version != expected_calibration_config_version:
            return CalibrationValidity.CONFIG_CHANGED
        if self.feature_schema_version != expected_feature_schema_version:
            return CalibrationValidity.FEATURE_SCHEMA_CHANGED
        if self.model_id != expected_model_id:
            return CalibrationValidity.MODEL_CHANGED
        if maximum_age_ms < 0:
            raise ValueError("maximum_age_ms cannot be negative")
        age_ms = at_timestamp_ms - self.completed_at_ms
        if age_ms < 0:
            raise ValueError("at_timestamp_ms cannot predate calibration")
        if age_ms > maximum_age_ms:
            return CalibrationValidity.EXPIRED
        return CalibrationValidity.VALID

    def to_dict(self) -> dict[str, object]:
        """Serialize only derived calibration statistics and metadata."""

        return {
            "calibration_schema_version": self.calibration_schema_version,
            "calibration_config_version": self.calibration_config_version,
            "feature_schema_version": self.feature_schema_version,
            "model_id": self.model_id,
            "started_at_ms": self.started_at_ms,
            "completed_at_ms": self.completed_at_ms,
            "accepted_sample_count": self.accepted_sample_count,
            "rejected_sample_count": self.rejected_sample_count,
            "features": [feature.to_dict() for feature in self.features],
        }

    @classmethod
    def from_dict(cls, data: object) -> CalibrationBaseline:
        """Deserialize the supported calibration schema, rejecting ambiguity."""

        if not isinstance(data, dict):
            raise ValueError("calibration baseline must be an object")
        schema_version = data.get("calibration_schema_version")
        if schema_version != CALIBRATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported calibration schema version: {schema_version!r}")

        feature_items = data.get("features")
        if not isinstance(feature_items, list):
            raise ValueError("calibration features must be a list")
        features = tuple(CalibrationFeature.from_dict(item) for item in feature_items)
        feature_names = tuple(feature.name for feature in features)
        ordered_names = tuple(name for name in _FEATURE_NAMES if name in feature_names)
        if (
            feature_names != ordered_names
            or len(set(feature_names)) != len(feature_names)
            or not set(_REQUIRED_FEATURE_NAMES).issubset(feature_names)
        ):
            raise ValueError("calibration feature set is invalid or out of order")

        calibration_config_version = _required_positive_int(
            data, "calibration_config_version"
        )
        feature_schema_version = _required_positive_int(data, "feature_schema_version")
        model_id = data.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a non-empty string")
        started_at_ms = _required_nonnegative_int(data, "started_at_ms")
        completed_at_ms = _required_nonnegative_int(data, "completed_at_ms")
        if completed_at_ms < started_at_ms:
            raise ValueError("completed_at_ms cannot predate started_at_ms")
        return cls(
            calibration_schema_version=CALIBRATION_SCHEMA_VERSION,
            calibration_config_version=calibration_config_version,
            feature_schema_version=feature_schema_version,
            model_id=model_id,
            started_at_ms=started_at_ms,
            completed_at_ms=completed_at_ms,
            accepted_sample_count=_required_positive_int(data, "accepted_sample_count"),
            rejected_sample_count=_required_nonnegative_int(data, "rejected_sample_count"),
            features=features,
        )


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """The all-or-nothing result of one calibration attempt."""

    baseline: CalibrationBaseline | None
    failure: CalibrationFailure | None
    guidance: str


@dataclass(frozen=True, slots=True)
class CalibrationProgress:
    """Derived calibration progress containing no observations or landmarks."""

    accepted_sample_count: int
    rejected_sample_count: int
    required_sample_count: int
    elapsed_ms: int
    maximum_sample_span_ms: int
    timed_out: bool

    @property
    def is_complete(self) -> bool:
        return self.accepted_sample_count >= self.required_sample_count


def _required_int(data: dict[object, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _required_nonnegative_int(data: dict[object, object], key: str) -> int:
    value = _required_int(data, key)
    if value < 0:
        raise ValueError(f"{key} cannot be negative")
    return value


def _required_positive_int(data: dict[object, object], key: str) -> int:
    value = _required_int(data, key)
    if value < 1:
        raise ValueError(f"{key} must be positive")
    return value


def _readings(metrics: PostureMetrics) -> dict[str, MetricReading]:
    return {
        "shoulder_tilt_degrees": metrics.shoulder_tilt_degrees,
        "torso_lean_degrees": metrics.torso_lean_degrees,
        "head_lateral_offset_ratio": metrics.head_lateral_offset_ratio,
        "head_vertical_offset_ratio": metrics.head_vertical_offset_ratio,
        "head_depth_ratio": metrics.head_depth_ratio,
    }


def _robust_statistics(name: str, values: list[float]) -> CalibrationFeature:
    median = statistics.median(values)
    absolute_deviations = [abs(value - median) for value in values]
    scaled_mad = statistics.median(absolute_deviations) * 1.4826
    return CalibrationFeature(
        name=name,
        median=median,
        dispersion=max(scaled_mad, _DISPERSION_FLOORS[name]),
    )


class CalibrationAccumulator:
    """Collect a bounded sequence and produce an all-or-nothing baseline."""

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self._config = config or CalibrationConfig()
        self._accepted_values: dict[str, list[float]] = {
            name: [] for name in _FEATURE_NAMES
        }
        self._accepted_sample_count = 0
        self._rejected_count = 0
        self._first_timestamp_ms: int | None = None
        self._last_timestamp_ms: int | None = None
        self._state = "collecting"

    @property
    def progress(self) -> CalibrationProgress:
        first_timestamp_ms = self._first_timestamp_ms
        last_timestamp_ms = self._last_timestamp_ms
        elapsed_ms = (
            0
            if first_timestamp_ms is None or last_timestamp_ms is None
            else last_timestamp_ms - first_timestamp_ms
        )
        return CalibrationProgress(
            accepted_sample_count=self._accepted_sample_count,
            rejected_sample_count=self._rejected_count,
            required_sample_count=self._config.minimum_accepted_samples,
            elapsed_ms=elapsed_ms,
            maximum_sample_span_ms=self._config.maximum_sample_span_ms,
            timed_out=self._state == "timed_out",
        )

    def add(self, *, timestamp_ms: int, metrics: PostureMetrics) -> bool:
        """Add a sample if every derived metric is finite and confident."""

        if self._state != "collecting":
            raise RuntimeError("calibration is no longer collecting")
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("calibration timestamps must increase")
        if self._first_timestamp_ms is None:
            self._first_timestamp_ms = timestamp_ms
        elif timestamp_ms - self._first_timestamp_ms > self._config.maximum_sample_span_ms:
            self._state = "timed_out"
            self._clear_accepted()
            return False
        self._last_timestamp_ms = timestamp_ms

        readings = _readings(metrics)

        def is_usable(reading: MetricReading) -> bool:
            return (
                reading.value is not None
                and math.isfinite(reading.value)
                and reading.confidence >= self._config.minimum_metric_confidence
            )

        accepted = all(is_usable(readings[name]) for name in _REQUIRED_FEATURE_NAMES)
        if accepted:
            self._accepted_sample_count += 1
            for name, reading in readings.items():
                if is_usable(reading):
                    assert reading.value is not None
                    self._accepted_values[name].append(float(reading.value))
        else:
            self._rejected_count += 1
        return accepted

    def interrupt(self) -> None:
        """Invalidate this in-memory attempt without creating a baseline."""

        if self._state == "collecting":
            self._state = "interrupted"
            self._clear_accepted()

    def _clear_accepted(self) -> None:
        self._accepted_sample_count = 0
        for values in self._accepted_values.values():
            values.clear()

    def finish(self, *, completed_at_ms: int, model_id: str) -> CalibrationResult:
        """Finish once, returning either a complete baseline or guidance."""

        if self._state == "finished":
            raise RuntimeError("calibration has already finished")
        if not model_id:
            raise ValueError("model_id must be non-empty")
        if self._state == "interrupted":
            self._state = "finished"
            return CalibrationResult(
                baseline=None,
                failure=CalibrationFailure.INTERRUPTED,
                guidance="Calibration was interrupted. Try again when you are ready.",
            )
        if self._state == "timed_out":
            self._state = "finished"
            return CalibrationResult(
                baseline=None,
                failure=CalibrationFailure.CAPTURE_TIMED_OUT,
                guidance="Calibration took too long. Settle comfortably and try again.",
            )
        self._state = "finished"

        accepted_count = self._accepted_sample_count
        total_count = accepted_count + self._rejected_count
        coverage = accepted_count / total_count if total_count else 0.0
        if (
            accepted_count < self._config.minimum_accepted_samples
            or coverage < self._config.minimum_sample_coverage
        ):
            return CalibrationResult(
                baseline=None,
                failure=CalibrationFailure.INSUFFICIENT_CONFIDENT_SAMPLES,
                guidance=(
                    "We could not see enough of the pose reliably. "
                    "Try improving camera framing or lighting."
                ),
            )

        assert self._first_timestamp_ms is not None
        assert self._last_timestamp_ms is not None
        if completed_at_ms < self._last_timestamp_ms:
            raise ValueError("completed_at_ms cannot predate the last sample")

        included_names = tuple(
            name
            for name in _FEATURE_NAMES
            if name in _REQUIRED_FEATURE_NAMES
            or (
                name in _OPTIONAL_FEATURE_NAMES
                and len(self._accepted_values[name])
                >= self._config.minimum_accepted_samples
                and len(self._accepted_values[name]) / accepted_count
                >= self._config.minimum_sample_coverage
            )
        )
        features = tuple(
            _robust_statistics(name, self._accepted_values[name])
            for name in included_names
        )
        if any(
            feature.dispersion > _STABILITY_LIMITS[feature.name] for feature in features
        ):
            return CalibrationResult(
                baseline=None,
                failure=CalibrationFailure.UNSTABLE_POSE,
                guidance=(
                    "There was too much movement to form a baseline. "
                    "Settle into a comfortable position and try again."
                ),
            )

        return CalibrationResult(
            baseline=CalibrationBaseline(
                calibration_schema_version=CALIBRATION_SCHEMA_VERSION,
                calibration_config_version=CALIBRATION_CONFIG_VERSION,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                model_id=model_id,
                started_at_ms=self._first_timestamp_ms,
                completed_at_ms=completed_at_ms,
                accepted_sample_count=accepted_count,
                rejected_sample_count=self._rejected_count,
                features=features,
            ),
            failure=None,
            guidance="Calibration complete.",
        )
