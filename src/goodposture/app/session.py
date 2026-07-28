"""Deterministic headless composition of the GoodPosture analysis core."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum

from goodposture.core.alert_policy import (
    DEFAULT_ALERT_POLICY_CONFIG,
    AlertDecision,
    AlertEvent,
    AlertPolicy,
    AlertPolicyConfig,
    AlertState,
)
from goodposture.core.calibration import (
    CalibrationAccumulator,
    CalibrationBaseline,
    CalibrationConfig,
    CalibrationProgress,
)
from goodposture.core.metrics import extract_posture_metrics
from goodposture.core.models import (
    MetricReading,
    ObservationFailure,
    PoseObservation,
    PostureMetrics,
)
from goodposture.core.scoring import (
    DEFAULT_SCORING_CONFIG,
    ScoreReading,
    ScoreResult,
    ScoreState,
    ScoringConfig,
    TimeAwareScoreSmoother,
    score_posture,
)


class SessionState(StrEnum):
    """Lifecycle states independent of any UI or operating-system adapter."""

    CREATED = "created"
    NEEDS_CALIBRATION = "needs_calibration"
    CALIBRATING = "calibrating"
    MONITORING = "monitoring"
    PAUSED = "paused"
    UNKNOWN = "unknown"
    ERROR = "error"
    STOPPED = "stopped"


class PostureAssessment(StrEnum):
    """Immediate confidence-aware posture assessment, separate from reminders."""

    GOOD = "good"
    NEEDS_ADJUSTMENT = "needs_adjustment"
    TRACKING_UNCERTAIN = "tracking_uncertain"


class SessionEventType(StrEnum):
    """Typed status and alert events emitted by the session."""

    STARTED = "started"
    CALIBRATION_STARTED = "calibration_started"
    CALIBRATION_COMPLETED = "calibration_completed"
    CALIBRATION_FAILED = "calibration_failed"
    CALIBRATION_CANCELLED = "calibration_cancelled"
    BASELINE_DELETED = "baseline_deleted"
    PAUSED = "paused"
    RESUMED = "resumed"
    QUIET_MODE_ENABLED = "quiet_mode_enabled"
    QUIET_MODE_DISABLED = "quiet_mode_disabled"
    TRACKING_UNKNOWN = "tracking_unknown"
    MONITORING_RECOVERED = "monitoring_recovered"
    RECOVERABLE_ERROR = "recoverable_error"
    PROMPT = "prompt"
    STOPPED = "stopped"


class _AggregateBucket(StrEnum):
    INELIGIBLE = "ineligible"
    ELIGIBLE_COMFORTABLE = "eligible_comfortable"
    ELIGIBLE_DEVIATION = "eligible_deviation"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SessionEvent:
    """A framework-neutral event safe for a UI or notification adapter."""

    timestamp_ms: int
    type: SessionEventType
    message: str


@dataclass(frozen=True, slots=True)
class SessionAggregates:
    """Derived in-memory rollups with no observations or landmarks."""

    started_at_ms: int | None
    ended_at_ms: int | None
    eligible_monitoring_ms: int
    unknown_ms: int
    deviation_ms: int
    prompt_count: int
    post_prompt_recovered_ms: int = 0


@dataclass(frozen=True, slots=True)
class SessionUpdate:
    """One bounded session result; the session retains no update history."""

    state: SessionState
    events: tuple[SessionEvent, ...]
    score: ScoreResult | None
    alert: AlertDecision | None
    aggregates: SessionAggregates
    posture_assessment: PostureAssessment = PostureAssessment.TRACKING_UNCERTAIN


_EVENT_MESSAGES = {
    SessionEventType.STARTED: "Monitoring session started.",
    SessionEventType.CALIBRATION_STARTED: (
        "Calibration started. Settle into a comfortable position."
    ),
    SessionEventType.CALIBRATION_COMPLETED: "Calibration complete.",
    SessionEventType.CALIBRATION_FAILED: (
        "Calibration was not completed. Adjust framing or lighting and try again."
    ),
    SessionEventType.CALIBRATION_CANCELLED: "Calibration cancelled.",
    SessionEventType.BASELINE_DELETED: "The current calibration baseline was deleted.",
    SessionEventType.PAUSED: "Monitoring paused.",
    SessionEventType.RESUMED: "Monitoring resumed.",
    SessionEventType.QUIET_MODE_ENABLED: (
        "Quiet mode is on. Monitoring continues without comfort prompts."
    ),
    SessionEventType.QUIET_MODE_DISABLED: "Quiet mode is off. Comfort prompts can resume.",
    SessionEventType.TRACKING_UNKNOWN: "Tracking is uncertain; no posture judgment made.",
    SessionEventType.MONITORING_RECOVERED: "Tracking recovered.",
    SessionEventType.RECOVERABLE_ERROR: (
        "Monitoring encountered a temporary local capture or inference problem."
    ),
    SessionEventType.PROMPT: (
        "A quick comfort check: would a small position change or movement break feel good?"
    ),
    SessionEventType.STOPPED: "Monitoring session stopped.",
}

_POST_PROMPT_OUTCOME_WINDOW_MS = 5 * 60_000

_FAILURE_MESSAGES = {
    ObservationFailure.CAPTURE_UNAVAILABLE: (
        "The local camera is temporarily unavailable; monitoring will retry safely."
    ),
    ObservationFailure.CAPTURE_READ_FAILED: (
        "The local camera did not provide a frame; monitoring will retry safely."
    ),
    ObservationFailure.INFERENCE_FAILED: (
        "Local pose inference was temporarily unavailable; monitoring will retry safely."
    ),
}


def _unknown_metrics(reason: str) -> PostureMetrics:
    unavailable = MetricReading(value=None, confidence=0.0, unavailable_reason=reason)
    return PostureMetrics(
        shoulder_tilt_degrees=unavailable,
        torso_lean_degrees=unavailable,
        head_lateral_offset_ratio=unavailable,
        head_vertical_offset_ratio=unavailable,
        head_depth_ratio=unavailable,
    )


class AnalysisSession:
    """Compose calibration, scoring, smoothing, alerts, and aggregate math."""

    def __init__(
        self,
        *,
        model_id: str,
        baseline: CalibrationBaseline | None = None,
        calibration_config: CalibrationConfig | None = None,
        scoring_config: ScoringConfig = DEFAULT_SCORING_CONFIG,
        alert_config: AlertPolicyConfig = DEFAULT_ALERT_POLICY_CONFIG,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must be non-empty")
        self._model_id = model_id
        self._baseline = baseline
        self._calibration_config = calibration_config or CalibrationConfig()
        self._scoring_config = scoring_config
        self._alert_config = alert_config
        self._maximum_confident_interval_ms = min(
            scoring_config.gap_reset_ms,
            alert_config.maximum_observation_gap_ms,
        )
        self._state = SessionState.CREATED
        self._calibrator: CalibrationAccumulator | None = None
        self._pre_calibration_state: SessionState | None = None
        self._smoother = TimeAwareScoreSmoother(scoring_config)
        self._alert_policy = AlertPolicy(alert_config)
        self._last_timestamp_ms: int | None = None
        self._bucket = _AggregateBucket.INELIGIBLE
        self._started_at_ms: int | None = None
        self._ended_at_ms: int | None = None
        self._eligible_monitoring_ms = 0
        self._unknown_ms = 0
        self._deviation_ms = 0
        self._prompt_count = 0
        self._post_prompt_recovered_ms = 0
        self._post_prompt_outcome_until_ms: int | None = None
        self._post_prompt_recovery_confirmed = False
        self._quiet_mode = False
        self._posture_assessment = PostureAssessment.TRACKING_UNCERTAIN

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def baseline(self) -> CalibrationBaseline | None:
        return self._baseline

    @property
    def calibration_progress(self) -> CalibrationProgress | None:
        if self._calibrator is None:
            return None
        return self._calibrator.progress

    @property
    def aggregates(self) -> SessionAggregates:
        return SessionAggregates(
            started_at_ms=self._started_at_ms,
            ended_at_ms=self._ended_at_ms,
            eligible_monitoring_ms=self._eligible_monitoring_ms,
            unknown_ms=self._unknown_ms,
            deviation_ms=self._deviation_ms,
            prompt_count=self._prompt_count,
            post_prompt_recovered_ms=self._post_prompt_recovered_ms,
        )

    @property
    def quiet_mode(self) -> bool:
        """Whether analysis continues with supportive prompts suppressed."""

        return self._quiet_mode

    @property
    def last_timestamp_ms(self) -> int | None:
        """Most recent validated monotonic input, for adapter handoffs."""

        return self._last_timestamp_ms

    @property
    def scoring_config(self) -> ScoringConfig:
        return self._scoring_config

    @property
    def alert_config(self) -> AlertPolicyConfig:
        return self._alert_config

    def snapshot(self) -> SessionUpdate:
        """Return the current bounded state without advancing the session clock."""

        return self._update()

    def _event(
        self,
        timestamp_ms: int,
        event_type: SessionEventType,
        *,
        message: str | None = None,
    ) -> SessionEvent:
        return SessionEvent(
            timestamp_ms=timestamp_ms,
            type=event_type,
            message=message or _EVENT_MESSAGES[event_type],
        )

    def _update(
        self,
        *,
        events: tuple[SessionEvent, ...] = (),
        score: ScoreResult | None = None,
        alert: AlertDecision | None = None,
    ) -> SessionUpdate:
        return SessionUpdate(
            state=self._state,
            events=events,
            score=score,
            alert=alert,
            aggregates=self.aggregates,
            posture_assessment=self._posture_assessment,
        )

    def _advance_time(self, timestamp_ms: int) -> None:
        if timestamp_ms < 0:
            raise ValueError("timestamp_ms cannot be negative")
        if self._last_timestamp_ms is not None:
            if timestamp_ms <= self._last_timestamp_ms:
                raise ValueError("session timestamps must increase")
            elapsed_ms = timestamp_ms - self._last_timestamp_ms
            if self._bucket in (
                _AggregateBucket.ELIGIBLE_COMFORTABLE,
                _AggregateBucket.ELIGIBLE_DEVIATION,
            ):
                if elapsed_ms > self._maximum_confident_interval_ms:
                    self._unknown_ms += elapsed_ms
                else:
                    self._eligible_monitoring_ms += elapsed_ms
                    if self._bucket is _AggregateBucket.ELIGIBLE_DEVIATION:
                        self._deviation_ms += elapsed_ms
                    elif (
                        self._post_prompt_recovery_confirmed
                        and self._post_prompt_outcome_until_ms is not None
                    ):
                        outcome_end_ms = min(
                            timestamp_ms,
                            self._post_prompt_outcome_until_ms,
                        )
                        self._post_prompt_recovered_ms += max(
                            0,
                            outcome_end_ms - self._last_timestamp_ms,
                        )
            elif self._bucket is _AggregateBucket.UNKNOWN:
                self._unknown_ms += elapsed_ms
        self._last_timestamp_ms = timestamp_ms

    def _ensure_started(self) -> None:
        if self._state is SessionState.CREATED:
            raise RuntimeError("session has not started")
        if self._state is SessionState.STOPPED:
            raise RuntimeError("session is stopped")

    def _reset_scoring(self) -> None:
        self._smoother = TimeAwareScoreSmoother(self._scoring_config)

    def _reset_analysis(self) -> None:
        self._reset_scoring()
        self._alert_policy = AlertPolicy(self._alert_config)
        self._posture_assessment = PostureAssessment.TRACKING_UNCERTAIN

    def start(self, *, timestamp_ms: int) -> SessionUpdate:
        if self._state is not SessionState.CREATED:
            raise RuntimeError("session has already started")
        self._advance_time(timestamp_ms)
        self._started_at_ms = timestamp_ms
        self._state = (
            SessionState.MONITORING
            if self._baseline is not None
            else SessionState.NEEDS_CALIBRATION
        )
        return self._update(
            events=(self._event(timestamp_ms, SessionEventType.STARTED),)
        )

    def start_calibration(self, *, timestamp_ms: int) -> SessionUpdate:
        self._ensure_started()
        if self._state is SessionState.CALIBRATING:
            raise RuntimeError("calibration is already active")
        self._advance_time(timestamp_ms)
        self._pre_calibration_state = self._state
        self._calibrator = CalibrationAccumulator(self._calibration_config)
        if self._state is not SessionState.NEEDS_CALIBRATION:
            self._alert_policy.pause(timestamp_ms=timestamp_ms)
        self._bucket = _AggregateBucket.INELIGIBLE
        self._posture_assessment = PostureAssessment.TRACKING_UNCERTAIN
        self._state = SessionState.CALIBRATING
        return self._update(
            events=(self._event(timestamp_ms, SessionEventType.CALIBRATION_STARTED),)
        )

    def _restore_after_calibration(self, timestamp_ms: int) -> None:
        previous = self._pre_calibration_state
        if self._baseline is None:
            self._state = SessionState.NEEDS_CALIBRATION
        elif previous is SessionState.PAUSED:
            self._state = SessionState.PAUSED
        else:
            if not self._quiet_mode:
                self._alert_policy.resume(timestamp_ms=timestamp_ms)
            self._reset_scoring()
            self._state = SessionState.MONITORING
        self._bucket = _AggregateBucket.INELIGIBLE

    def finish_calibration(self, *, timestamp_ms: int) -> SessionUpdate:
        self._ensure_started()
        if self._state is not SessionState.CALIBRATING or self._calibrator is None:
            raise RuntimeError("calibration is not active")
        self._advance_time(timestamp_ms)
        result = self._calibrator.finish(
            completed_at_ms=timestamp_ms,
            model_id=self._model_id,
        )
        self._calibrator = None
        if result.baseline is not None:
            self._baseline = result.baseline
            previous = self._pre_calibration_state
            self._reset_analysis()
            if previous is SessionState.PAUSED:
                self._alert_policy.pause(timestamp_ms=timestamp_ms)
                self._state = SessionState.PAUSED
            else:
                if self._quiet_mode:
                    self._alert_policy.pause(timestamp_ms=timestamp_ms)
                self._state = SessionState.MONITORING
            event_type = SessionEventType.CALIBRATION_COMPLETED
        else:
            self._restore_after_calibration(timestamp_ms)
            event_type = SessionEventType.CALIBRATION_FAILED
        self._pre_calibration_state = None
        return self._update(events=(self._event(timestamp_ms, event_type),))

    def cancel_calibration(self, *, timestamp_ms: int) -> SessionUpdate:
        self._ensure_started()
        if self._state is not SessionState.CALIBRATING or self._calibrator is None:
            raise RuntimeError("calibration is not active")
        self._advance_time(timestamp_ms)
        self._calibrator.interrupt()
        self._calibrator = None
        self._restore_after_calibration(timestamp_ms)
        self._pre_calibration_state = None
        return self._update(
            events=(self._event(timestamp_ms, SessionEventType.CALIBRATION_CANCELLED),)
        )

    def delete_baseline(self, *, timestamp_ms: int) -> SessionUpdate:
        """Forget the current derived baseline without retaining partial data."""

        self._ensure_started()
        if self._state is SessionState.CALIBRATING:
            raise RuntimeError("cancel calibration before deleting the baseline")
        self._advance_time(timestamp_ms)
        self._baseline = None
        self._reset_analysis()
        self._bucket = _AggregateBucket.INELIGIBLE
        self._state = SessionState.NEEDS_CALIBRATION
        return self._update(
            events=(self._event(timestamp_ms, SessionEventType.BASELINE_DELETED),)
        )

    def pause(self, *, timestamp_ms: int) -> SessionUpdate:
        self._ensure_started()
        if self._state is SessionState.CALIBRATING:
            raise RuntimeError("cancel calibration before pausing")
        self._advance_time(timestamp_ms)
        self._alert_policy.pause(timestamp_ms=timestamp_ms)
        self._bucket = _AggregateBucket.INELIGIBLE
        self._posture_assessment = PostureAssessment.TRACKING_UNCERTAIN
        self._state = SessionState.PAUSED
        return self._update(
            events=(self._event(timestamp_ms, SessionEventType.PAUSED),)
        )

    def resume(self, *, timestamp_ms: int) -> SessionUpdate:
        self._ensure_started()
        if self._state is not SessionState.PAUSED:
            raise RuntimeError("session is not paused")
        if self._baseline is None:
            raise RuntimeError("calibration is required before resuming")
        self._advance_time(timestamp_ms)
        if not self._quiet_mode:
            self._alert_policy.resume(timestamp_ms=timestamp_ms)
        self._reset_scoring()
        self._bucket = _AggregateBucket.INELIGIBLE
        self._posture_assessment = PostureAssessment.TRACKING_UNCERTAIN
        self._state = SessionState.MONITORING
        return self._update(
            events=(self._event(timestamp_ms, SessionEventType.RESUMED),)
        )

    def set_quiet_mode(
        self,
        *,
        enabled: bool,
        timestamp_ms: int,
    ) -> SessionUpdate:
        """Continue analysis while explicitly suppressing or restoring prompts."""

        self._ensure_started()
        if self._state is SessionState.CALIBRATING:
            raise RuntimeError("finish or cancel calibration before changing quiet mode")
        if enabled is self._quiet_mode:
            raise RuntimeError("quiet mode already has the requested setting")
        self._advance_time(timestamp_ms)
        if self._state is not SessionState.PAUSED:
            if enabled:
                self._alert_policy.pause(timestamp_ms=timestamp_ms)
            else:
                self._alert_policy.resume(timestamp_ms=timestamp_ms)
        self._quiet_mode = enabled
        event_type = (
            SessionEventType.QUIET_MODE_ENABLED
            if enabled
            else SessionEventType.QUIET_MODE_DISABLED
        )
        return self._update(events=(self._event(timestamp_ms, event_type),))

    def process_observation(self, observation: PoseObservation) -> SessionUpdate:
        self._ensure_started()
        timestamp_ms = observation.timestamp_ms
        self._advance_time(timestamp_ms)

        if self._state is SessionState.PAUSED:
            self._bucket = _AggregateBucket.INELIGIBLE
            return self._update()

        metrics = extract_posture_metrics(observation)
        if self._state is SessionState.CALIBRATING:
            assert self._calibrator is not None
            self._calibrator.add(timestamp_ms=timestamp_ms, metrics=metrics)
            self._bucket = _AggregateBucket.INELIGIBLE
            return self._update()
        if self._baseline is None:
            self._state = SessionState.NEEDS_CALIBRATION
            self._bucket = _AggregateBucket.INELIGIBLE
            return self._update()

        previous_state = self._state
        reading = score_posture(metrics, self._baseline, self._scoring_config)
        score = self._smoother.update(timestamp_ms=timestamp_ms, reading=reading)
        alert = self._alert_policy.update(score)
        events: tuple[SessionEvent, ...] = ()
        if score.state is ScoreState.UNKNOWN:
            self._state = SessionState.UNKNOWN
            self._bucket = _AggregateBucket.UNKNOWN
            self._posture_assessment = PostureAssessment.TRACKING_UNCERTAIN
            events = (
                self._event(timestamp_ms, SessionEventType.TRACKING_UNKNOWN),
            )
        else:
            self._state = SessionState.MONITORING
            assert score.smoothed_score is not None
            if score.smoothed_score >= self._alert_config.deviation_threshold:
                self._posture_assessment = PostureAssessment.NEEDS_ADJUSTMENT
            elif (
                score.smoothed_score <= self._alert_config.recovery_threshold
                or self._posture_assessment is PostureAssessment.TRACKING_UNCERTAIN
            ):
                self._posture_assessment = PostureAssessment.GOOD
            self._bucket = (
                _AggregateBucket.ELIGIBLE_DEVIATION
                if score.smoothed_score >= self._alert_config.deviation_threshold
                else _AggregateBucket.ELIGIBLE_COMFORTABLE
            )
            if alert.event is AlertEvent.PROMPT:
                self._prompt_count += 1
                self._post_prompt_outcome_until_ms = (
                    timestamp_ms + _POST_PROMPT_OUTCOME_WINDOW_MS
                )
                self._post_prompt_recovery_confirmed = False
                events = (
                    self._event(
                        timestamp_ms,
                        SessionEventType.PROMPT,
                        message=alert.prompt_message,
                    ),
                )
            elif (
                alert.state is AlertState.RECOVERED
                and self._post_prompt_outcome_until_ms is not None
            ):
                self._post_prompt_recovery_confirmed = True
            elif previous_state in (SessionState.UNKNOWN, SessionState.ERROR):
                events = (
                    self._event(timestamp_ms, SessionEventType.MONITORING_RECOVERED),
                )
        return self._update(events=events, score=score, alert=alert)

    def process_observations(
        self,
        observations: Iterable[PoseObservation],
    ) -> Iterator[SessionUpdate]:
        """Process a live or prerecorded source without buffering its outputs."""

        for observation in observations:
            yield self.process_observation(observation)

    def process_failure(
        self,
        *,
        timestamp_ms: int,
        failure: ObservationFailure,
    ) -> SessionUpdate:
        """Convert adapter failures into recoverable uncertainty events."""

        self._ensure_started()
        self._advance_time(timestamp_ms)
        if self._state is SessionState.PAUSED:
            self._bucket = _AggregateBucket.INELIGIBLE
            return self._update()
        if self._state is SessionState.CALIBRATING:
            assert self._calibrator is not None
            self._calibrator.add(
                timestamp_ms=timestamp_ms,
                metrics=_unknown_metrics(failure.value),
            )
            self._bucket = _AggregateBucket.INELIGIBLE
            return self._update(
                events=(
                    self._event(
                        timestamp_ms,
                        SessionEventType.RECOVERABLE_ERROR,
                        message=_FAILURE_MESSAGES[failure],
                    ),
                )
            )

        reading = ScoreReading(
            state=ScoreState.UNKNOWN,
            score=None,
            confidence=0.0,
            feature_deviations=(),
            config_version=self._scoring_config.version,
        )
        score = self._smoother.update(timestamp_ms=timestamp_ms, reading=reading)
        alert = self._alert_policy.update(score)
        self._state = SessionState.ERROR
        self._bucket = _AggregateBucket.UNKNOWN
        self._posture_assessment = PostureAssessment.TRACKING_UNCERTAIN
        return self._update(
            events=(
                self._event(
                    timestamp_ms,
                    SessionEventType.RECOVERABLE_ERROR,
                    message=_FAILURE_MESSAGES[failure],
                ),
            ),
            score=score,
            alert=alert,
        )

    def stop(self, *, timestamp_ms: int) -> SessionUpdate:
        self._ensure_started()
        self._advance_time(timestamp_ms)
        if self._calibrator is not None:
            self._calibrator.interrupt()
            self._calibrator = None
        self._bucket = _AggregateBucket.INELIGIBLE
        self._ended_at_ms = timestamp_ms
        self._state = SessionState.STOPPED
        return self._update(
            events=(self._event(timestamp_ms, SessionEventType.STOPPED),)
        )
