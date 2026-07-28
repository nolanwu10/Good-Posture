"""Deterministic sustained-deviation, recovery, and cooldown policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from goodposture.core.scoring import ScoreResult, ScoreState

ALERT_POLICY_CONFIG_VERSION: Final = 6
SUPPORTIVE_PROMPT_MESSAGE: Final = (
    "A quick comfort check: would a small position change or movement break feel good?"
)


class AlertState(StrEnum):
    """User-visible alert policy states."""

    MONITORING = "monitoring"
    DEVIATION_PENDING = "deviation_pending"
    COOLDOWN = "cooldown"
    RECOVERED = "recovered"
    PAUSED = "paused"
    UNKNOWN = "unknown"


class AlertEvent(StrEnum):
    """One-shot events emitted by the policy."""

    PROMPT = "prompt"


class AlertTrigger(StrEnum):
    """Why the policy emitted a reminder."""

    CONTINUOUS = "continuous"
    DEBT = "debt"


class _PostureBucket(StrEnum):
    DEVIATION = "deviation"
    COMFORTABLE = "comfortable"
    NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class AlertPolicyConfig:
    """Versioned alert thresholds and time boundaries."""

    version: int = ALERT_POLICY_CONFIG_VERSION
    deviation_threshold: float = 75.0
    recovery_threshold: float = 45.0
    continuous_deviation_duration_ms: int = 7_000
    posture_debt_limit_ms: int = 12_000
    recovery_debt_decay_rate: float = 1.0
    recovery_duration_ms: int = 5_000
    cooldown_duration_ms: int = 20_000
    maximum_observation_gap_ms: int = 2_000

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("alert policy version must be positive")
        for name, value in (
            ("deviation_threshold", self.deviation_threshold),
            ("recovery_threshold", self.recovery_threshold),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                raise ValueError(f"{name} must be between 0 and 100")
        if self.recovery_threshold >= self.deviation_threshold:
            raise ValueError("recovery_threshold must be below deviation_threshold")
        if self.continuous_deviation_duration_ms <= 0:
            raise ValueError("continuous_deviation_duration_ms must be positive")
        if self.posture_debt_limit_ms <= 0:
            raise ValueError("posture_debt_limit_ms must be positive")
        if (
            not math.isfinite(self.recovery_debt_decay_rate)
            or self.recovery_debt_decay_rate <= 0.0
        ):
            raise ValueError("recovery_debt_decay_rate must be positive and finite")
        if self.recovery_duration_ms <= 0:
            raise ValueError("recovery_duration_ms must be positive")
        if self.cooldown_duration_ms <= 0:
            raise ValueError("cooldown_duration_ms must be positive")
        if self.maximum_observation_gap_ms <= 0:
            raise ValueError("maximum_observation_gap_ms must be positive")


DEFAULT_ALERT_POLICY_CONFIG: Final = AlertPolicyConfig()


@dataclass(frozen=True, slots=True)
class AlertDecision:
    """A bounded state snapshot with an optional one-shot prompt."""

    timestamp_ms: int
    state: AlertState
    event: AlertEvent | None
    prompt_message: str | None
    pending_since_ms: int | None
    cooldown_until_ms: int | None
    config_version: int
    continuous_deviation_ms: int
    posture_debt_ms: float
    trigger: AlertTrigger | None


class AlertPolicy:
    """Consume smoothed scores and emit at most one prompt per alert cycle."""

    def __init__(
        self,
        config: AlertPolicyConfig = DEFAULT_ALERT_POLICY_CONFIG,
    ) -> None:
        self._config = config
        self._state = AlertState.MONITORING
        self._paused = False
        self._pending_since_ms: int | None = None
        self._cooldown_until_ms: int | None = None
        self._requires_recovery = False
        self._recovery_since_ms: int | None = None
        self._unrecovered_deviation_since_ms: int | None = None
        self._last_input_timestamp_ms: int | None = None
        self._last_score_timestamp_ms: int | None = None
        self._last_posture_bucket: _PostureBucket | None = None
        self._continuous_deviation_ms = 0
        self._posture_debt_ms = 0.0

    def _record_timestamp(self, timestamp_ms: int) -> None:
        if timestamp_ms < 0:
            raise ValueError("timestamp_ms cannot be negative")
        if (
            self._last_input_timestamp_ms is not None
            and timestamp_ms <= self._last_input_timestamp_ms
        ):
            raise ValueError("alert policy timestamps must increase")
        self._last_input_timestamp_ms = timestamp_ms

    def _decision(
        self,
        timestamp_ms: int,
        *,
        event: AlertEvent | None = None,
        trigger: AlertTrigger | None = None,
    ) -> AlertDecision:
        return AlertDecision(
            timestamp_ms=timestamp_ms,
            state=self._state,
            event=event,
            prompt_message=SUPPORTIVE_PROMPT_MESSAGE if event is AlertEvent.PROMPT else None,
            pending_since_ms=self._pending_since_ms,
            cooldown_until_ms=self._cooldown_until_ms,
            config_version=self._config.version,
            continuous_deviation_ms=self._continuous_deviation_ms,
            posture_debt_ms=self._posture_debt_ms,
            trigger=trigger,
        )

    def _clear_deviation_progress(self) -> None:
        self._pending_since_ms = None
        self._continuous_deviation_ms = 0
        self._posture_debt_ms = 0.0
        self._last_posture_bucket = None

    def _prompt(
        self,
        timestamp_ms: int,
        *,
        trigger: AlertTrigger,
    ) -> AlertDecision:
        self._clear_deviation_progress()
        self._requires_recovery = True
        self._recovery_since_ms = None
        self._unrecovered_deviation_since_ms = timestamp_ms
        self._cooldown_until_ms = timestamp_ms + self._config.cooldown_duration_ms
        self._state = AlertState.COOLDOWN
        return self._decision(
            timestamp_ms,
            event=AlertEvent.PROMPT,
            trigger=trigger,
        )

    def pause(self, *, timestamp_ms: int) -> AlertDecision:
        """Pause prompting and cancel any pending sustained deviation."""

        self._record_timestamp(timestamp_ms)
        self._paused = True
        self._clear_deviation_progress()
        self._state = AlertState.PAUSED
        return self._decision(timestamp_ms)

    def resume(self, *, timestamp_ms: int) -> AlertDecision:
        """Resume monitoring without restoring cancelled pending time."""

        self._record_timestamp(timestamp_ms)
        self._paused = False
        if self._requires_recovery or (
            self._cooldown_until_ms is not None
            and timestamp_ms < self._cooldown_until_ms
        ):
            self._state = AlertState.COOLDOWN
        else:
            self._state = AlertState.MONITORING
        return self._decision(timestamp_ms)

    def update(self, score: ScoreResult) -> AlertDecision:
        """Advance the policy with one timestamped smoothed score."""

        timestamp_ms = score.timestamp_ms
        self._record_timestamp(timestamp_ms)
        gap_ms = (
            timestamp_ms - self._last_score_timestamp_ms
            if self._last_score_timestamp_ms is not None
            else None
        )
        self._last_score_timestamp_ms = timestamp_ms
        if (
            gap_ms is not None
            and gap_ms > self._config.maximum_observation_gap_ms
        ):
            self._continuous_deviation_ms = 0
            self._pending_since_ms = None
            self._recovery_since_ms = None
            self._unrecovered_deviation_since_ms = None

        if self._paused:
            self._clear_deviation_progress()
            self._recovery_since_ms = None
            self._unrecovered_deviation_since_ms = None
            self._state = AlertState.PAUSED
            return self._decision(timestamp_ms)

        if score.state is ScoreState.UNKNOWN:
            self._continuous_deviation_ms = 0
            self._pending_since_ms = None
            self._last_posture_bucket = None
            self._recovery_since_ms = None
            self._unrecovered_deviation_since_ms = None
            self._state = AlertState.UNKNOWN
            return self._decision(timestamp_ms)
        if score.smoothed_score is None:
            raise ValueError("available score result must contain a smoothed score")
        value = score.smoothed_score
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise ValueError("smoothed score must be finite and between 0 and 100")

        if value >= self._config.deviation_threshold:
            current_bucket = _PostureBucket.DEVIATION
        elif value <= self._config.recovery_threshold:
            current_bucket = _PostureBucket.COMFORTABLE
        elif self._last_posture_bucket is not None:
            current_bucket = self._last_posture_bucket
        else:
            current_bucket = _PostureBucket.NEUTRAL

        timed_cooldown = (
            self._cooldown_until_ms is not None
            and timestamp_ms < self._cooldown_until_ms
        )
        if self._requires_recovery:
            self._clear_deviation_progress()
            self._last_posture_bucket = current_bucket
            recovered = value <= self._config.recovery_threshold
            if recovered:
                self._recovery_since_ms = None
            elif value < self._config.deviation_threshold:
                if self._recovery_since_ms is None:
                    self._recovery_since_ms = timestamp_ms
                recovered = (
                    timestamp_ms - self._recovery_since_ms
                    >= self._config.recovery_duration_ms
                )
            else:
                self._recovery_since_ms = None
            if recovered:
                self._requires_recovery = False
                self._recovery_since_ms = None
                self._unrecovered_deviation_since_ms = None
                self._state = AlertState.RECOVERED
                return self._decision(timestamp_ms)
            if value >= self._config.deviation_threshold:
                if self._unrecovered_deviation_since_ms is None:
                    self._unrecovered_deviation_since_ms = timestamp_ms
            else:
                self._unrecovered_deviation_since_ms = None
            if (
                not timed_cooldown
                and self._unrecovered_deviation_since_ms is not None
                and timestamp_ms - self._unrecovered_deviation_since_ms
                >= self._config.cooldown_duration_ms
            ):
                return self._prompt(timestamp_ms, trigger=AlertTrigger.CONTINUOUS)
            self._state = AlertState.COOLDOWN
            return self._decision(timestamp_ms)

        usable_gap = (
            gap_ms
            if gap_ms is not None
            and gap_ms <= self._config.maximum_observation_gap_ms
            else None
        )
        if usable_gap is not None:
            if self._last_posture_bucket is _PostureBucket.DEVIATION:
                self._posture_debt_ms = min(
                    float(self._config.posture_debt_limit_ms),
                    self._posture_debt_ms + usable_gap,
                )
            elif self._last_posture_bucket is _PostureBucket.COMFORTABLE:
                self._posture_debt_ms = max(
                    0.0,
                    self._posture_debt_ms
                    - usable_gap * self._config.recovery_debt_decay_rate,
                )

        if current_bucket is _PostureBucket.DEVIATION:
            if (
                usable_gap is not None
                and self._last_posture_bucket is _PostureBucket.DEVIATION
            ):
                self._continuous_deviation_ms = min(
                    self._config.continuous_deviation_duration_ms,
                    self._continuous_deviation_ms + usable_gap,
                )
            else:
                self._continuous_deviation_ms = 0
                self._pending_since_ms = timestamp_ms
        else:
            self._continuous_deviation_ms = 0
            self._pending_since_ms = None

        self._last_posture_bucket = current_bucket
        trigger: AlertTrigger | None = None
        if current_bucket is _PostureBucket.DEVIATION:
            if (
                self._continuous_deviation_ms
                >= self._config.continuous_deviation_duration_ms
            ):
                trigger = AlertTrigger.CONTINUOUS
            elif self._posture_debt_ms >= self._config.posture_debt_limit_ms:
                trigger = AlertTrigger.DEBT
        if trigger is not None:
            if not timed_cooldown:
                return self._prompt(timestamp_ms, trigger=trigger)
            self._state = AlertState.COOLDOWN
            return self._decision(timestamp_ms)

        if timed_cooldown:
            self._state = (
                AlertState.RECOVERED
                if current_bucket is _PostureBucket.COMFORTABLE
                else AlertState.COOLDOWN
            )
            return self._decision(timestamp_ms)

        self._state = (
            AlertState.DEVIATION_PENDING
            if current_bucket is _PostureBucket.DEVIATION
            or self._posture_debt_ms > 0.0
            else AlertState.MONITORING
        )
        return self._decision(timestamp_ms)
