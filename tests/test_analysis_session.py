from __future__ import annotations

from dataclasses import replace

import pytest

from goodposture.app.session import (
    AnalysisSession,
    ObservationFailure,
    PostureAssessment,
    SessionEventType,
    SessionState,
)
from goodposture.core.alert_policy import DEFAULT_ALERT_POLICY_CONFIG, AlertEvent
from goodposture.core.calibration import (
    CALIBRATION_CONFIG_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    CalibrationBaseline,
    CalibrationConfig,
    CalibrationFeature,
)
from goodposture.core.models import Landmark, LandmarkName, PoseObservation
from goodposture.core.scoring import DEFAULT_SCORING_CONFIG


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


def observation(
    timestamp_ms: int,
    *,
    deviated: bool = False,
    confidence: float = 0.95,
) -> PoseObservation:
    if deviated:
        landmarks = {
            LandmarkName.NOSE: Landmark(
                x=0.95,
                y=0.55,
                z=-0.90,
                visibility=confidence,
                presence=confidence,
            ),
            LandmarkName.LEFT_SHOULDER: Landmark(
                x=0.30,
                y=0.25,
                z=0.00,
                visibility=confidence,
                presence=confidence,
            ),
            LandmarkName.RIGHT_SHOULDER: Landmark(
                x=0.70,
                y=0.60,
                z=0.00,
                visibility=confidence,
                presence=confidence,
            ),
            LandmarkName.LEFT_HIP: Landmark(
                x=0.05,
                y=0.80,
                z=0.04,
                visibility=confidence,
                presence=confidence,
            ),
            LandmarkName.RIGHT_HIP: Landmark(
                x=0.25,
                y=0.80,
                z=0.04,
                visibility=confidence,
                presence=confidence,
            ),
        }
    else:
        landmarks = {
            LandmarkName.NOSE: Landmark(
                x=0.50,
                y=0.20,
                z=-0.18,
                visibility=confidence,
                presence=confidence,
            ),
            LandmarkName.LEFT_SHOULDER: Landmark(
                x=0.35,
                y=0.42,
                z=0.00,
                visibility=confidence,
                presence=confidence,
            ),
            LandmarkName.RIGHT_SHOULDER: Landmark(
                x=0.65,
                y=0.42,
                z=0.00,
                visibility=confidence,
                presence=confidence,
            ),
            LandmarkName.LEFT_HIP: Landmark(
                x=0.40,
                y=0.75,
                z=0.04,
                visibility=confidence,
                presence=confidence,
            ),
            LandmarkName.RIGHT_HIP: Landmark(
                x=0.60,
                y=0.75,
                z=0.04,
                visibility=confidence,
                presence=confidence,
            ),
        }
    return PoseObservation(timestamp_ms=timestamp_ms, landmarks=landmarks)


def test_quiet_mode_survives_cancelled_recalibration() -> None:
    session = AnalysisSession(model_id=baseline().model_id, baseline=baseline())
    session.start(timestamp_ms=1_000)
    session.set_quiet_mode(enabled=True, timestamp_ms=1_100)
    session.start_calibration(timestamp_ms=1_200)
    session.cancel_calibration(timestamp_ms=1_300)

    update = session.process_observation(observation(1_400))

    assert session.quiet_mode is True
    assert update.alert is not None
    assert update.alert.state.value == "paused"


def session(existing_baseline: CalibrationBaseline | None = None) -> AnalysisSession:
    return AnalysisSession(
        model_id="pose-landmarker-lite@sha256:abc",
        baseline=existing_baseline,
        calibration_config=CalibrationConfig(
            minimum_accepted_samples=3,
            minimum_sample_coverage=0.70,
            minimum_metric_confidence=0.80,
            maximum_sample_span_ms=5_000,
        ),
        scoring_config=replace(
            DEFAULT_SCORING_CONFIG,
            smoothing_time_constant_ms=1,
            gap_reset_ms=5_000,
        ),
        alert_config=replace(
            DEFAULT_ALERT_POLICY_CONFIG,
            deviation_threshold=20.0,
            recovery_threshold=10.0,
            continuous_deviation_duration_ms=1_000,
            posture_debt_limit_ms=2_000,
            recovery_debt_decay_rate=0.5,
            cooldown_duration_ms=5_000,
            maximum_observation_gap_ms=2_000,
        ),
    )


def test_observations_flow_headlessly_from_landmarks_to_prompt_event() -> None:
    analysis = session(baseline())
    analysis.start(timestamp_ms=0)

    pending = analysis.process_observation(observation(100, deviated=True))
    prompted = analysis.process_observation(observation(1_100, deviated=True))

    assert pending.state is SessionState.MONITORING
    assert pending.score is not None and pending.score.smoothed_score is not None
    assert prompted.alert is not None
    assert prompted.alert.event is AlertEvent.PROMPT
    assert [event.type for event in prompted.events] == [SessionEventType.PROMPT]
    assert prompted.aggregates.prompt_count == 1


def test_recovered_time_counts_observed_upright_time_after_a_prompt() -> None:
    analysis = session(baseline())
    analysis.start(timestamp_ms=0)
    analysis.process_observation(observation(100, deviated=True))
    prompted = analysis.process_observation(observation(1_100, deviated=True))

    recovered = analysis.process_observation(observation(1_200))
    after_one_upright_second = analysis.process_observation(observation(2_200))

    assert prompted.alert is not None
    assert prompted.alert.event is AlertEvent.PROMPT
    assert recovered.alert is not None
    assert recovered.alert.state.value == "recovered"
    assert after_one_upright_second.aggregates.post_prompt_recovered_ms == 1_000


def test_default_policy_prompts_for_sustained_head_hunch_without_hip_calibration() -> None:
    laptop_baseline = replace(
        baseline(),
        features=tuple(
            feature
            for feature in baseline().features
            if feature.name != "torso_lean_degrees"
        ),
    )
    analysis = AnalysisSession(
        model_id=laptop_baseline.model_id,
        baseline=laptop_baseline,
    )
    analysis.start(timestamp_ms=0)
    hunch_landmarks = dict(observation(100).landmarks)
    nose = hunch_landmarks[LandmarkName.NOSE]
    hunch_landmarks[LandmarkName.NOSE] = replace(nose, z=-0.30)

    prompted = None
    for timestamp_ms in range(100, 7_101, 500):
        prompted = analysis.process_observation(
            PoseObservation(timestamp_ms=timestamp_ms, landmarks=hunch_landmarks)
        )

    assert prompted is not None
    assert prompted.score is not None
    assert prompted.score.head_hunch_score == pytest.approx(100.0)
    assert prompted.alert is not None
    assert prompted.alert.event is AlertEvent.PROMPT


def test_posture_assessment_changes_before_reminder_timers_complete() -> None:
    analysis = AnalysisSession(model_id=baseline().model_id, baseline=baseline())
    analysis.start(timestamp_ms=0)

    good = analysis.process_observation(observation(100))
    needs_adjustment = None
    for timestamp_ms in range(200, 1_501, 100):
        needs_adjustment = analysis.process_observation(
            observation(timestamp_ms, deviated=True)
        )

    assert good.posture_assessment is PostureAssessment.GOOD
    assert needs_adjustment is not None
    assert needs_adjustment.posture_assessment is PostureAssessment.NEEDS_ADJUSTMENT
    assert needs_adjustment.alert is not None
    assert needs_adjustment.alert.event is None
    assert (
        needs_adjustment.alert.continuous_deviation_ms
        < DEFAULT_ALERT_POLICY_CONFIG.continuous_deviation_duration_ms
    )


def test_unknown_confidence_is_safe_and_restarts_pending_time() -> None:
    analysis = session(baseline())
    analysis.start(timestamp_ms=0)
    analysis.process_observation(observation(100, deviated=True))

    unknown = analysis.process_observation(
        observation(500, deviated=True, confidence=0.20)
    )
    restarted = analysis.process_observation(observation(1_000, deviated=True))
    prompted = analysis.process_observation(observation(2_000, deviated=True))

    assert unknown.state is SessionState.UNKNOWN
    assert unknown.score is not None and unknown.score.smoothed_score is None
    assert unknown.alert is not None and unknown.alert.event is None
    assert restarted.alert is not None and restarted.alert.pending_since_ms == 1_000
    assert prompted.alert is not None
    assert prompted.alert.event is AlertEvent.PROMPT


def test_pause_and_resume_cancel_pending_without_processing_observations() -> None:
    analysis = session(baseline())
    analysis.start(timestamp_ms=0)
    analysis.process_observation(observation(100, deviated=True))

    paused = analysis.pause(timestamp_ms=500)
    ignored = analysis.process_observation(observation(1_000, deviated=True))
    resumed = analysis.resume(timestamp_ms=1_500)
    restarted = analysis.process_observation(observation(2_000, deviated=True))

    assert paused.state is SessionState.PAUSED
    assert ignored.state is SessionState.PAUSED
    assert ignored.score is None
    assert resumed.state is SessionState.MONITORING
    assert restarted.alert is not None and restarted.alert.pending_since_ms == 2_000


def test_calibration_lifecycle_enables_monitoring_without_persisting_samples() -> None:
    analysis = session()

    started = analysis.start(timestamp_ms=0)
    calibrating = analysis.start_calibration(timestamp_ms=100)
    for timestamp_ms in (200, 300, 400):
        analysis.process_observation(observation(timestamp_ms))
    finished = analysis.finish_calibration(timestamp_ms=500)

    assert started.state is SessionState.NEEDS_CALIBRATION
    assert calibrating.state is SessionState.CALIBRATING
    assert finished.state is SessionState.MONITORING
    assert finished.events[0].type is SessionEventType.CALIBRATION_COMPLETED
    assert analysis.baseline is not None
    assert not hasattr(analysis, "observations")


def test_failed_recalibration_keeps_the_last_complete_baseline() -> None:
    original = baseline()
    analysis = session(original)
    analysis.start(timestamp_ms=0)
    analysis.start_calibration(timestamp_ms=100)
    for timestamp_ms in (200, 300, 400):
        analysis.process_observation(observation(timestamp_ms, confidence=0.20))

    failed = analysis.finish_calibration(timestamp_ms=500)

    assert failed.state is SessionState.MONITORING
    assert failed.events[0].type is SessionEventType.CALIBRATION_FAILED
    assert analysis.baseline is original


def test_calibration_failures_count_against_sample_coverage() -> None:
    analysis = session()
    analysis.start(timestamp_ms=0)
    analysis.start_calibration(timestamp_ms=100)
    analysis.process_observation(observation(200))
    analysis.process_failure(
        timestamp_ms=300,
        failure=ObservationFailure.CAPTURE_READ_FAILED,
    )
    analysis.process_observation(observation(400))
    analysis.process_failure(
        timestamp_ms=500,
        failure=ObservationFailure.INFERENCE_FAILED,
    )
    analysis.process_observation(observation(600))

    finished = analysis.finish_calibration(timestamp_ms=700)

    assert finished.state is SessionState.NEEDS_CALIBRATION
    assert finished.events[0].type is SessionEventType.CALIBRATION_FAILED
    assert analysis.baseline is None


def test_cancelled_recalibration_preserves_the_previous_baseline() -> None:
    original = baseline()
    analysis = session(original)
    analysis.start(timestamp_ms=0)
    analysis.start_calibration(timestamp_ms=100)
    analysis.process_observation(observation(200))

    cancelled = analysis.cancel_calibration(timestamp_ms=300)

    assert cancelled.state is SessionState.MONITORING
    assert cancelled.events[0].type is SessionEventType.CALIBRATION_CANCELLED
    assert analysis.baseline is original


def test_typed_inference_failure_is_recoverable_and_cancels_pending() -> None:
    analysis = session(baseline())
    analysis.start(timestamp_ms=0)
    analysis.process_observation(observation(100, deviated=True))

    failed = analysis.process_failure(
        timestamp_ms=500,
        failure=ObservationFailure.INFERENCE_FAILED,
    )
    recovered = analysis.process_observation(observation(1_000, deviated=True))

    assert failed.state is SessionState.ERROR
    assert failed.events[0].type is SessionEventType.RECOVERABLE_ERROR
    assert "inference" in failed.events[0].message.lower()
    assert failed.score is not None and failed.score.smoothed_score is None
    assert recovered.state is SessionState.MONITORING
    assert recovered.alert is not None and recovered.alert.pending_since_ms == 1_000


def test_aggregates_count_only_confident_monitoring_intervals() -> None:
    analysis = session(baseline())
    analysis.start(timestamp_ms=0)
    analysis.process_observation(observation(100))
    analysis.process_observation(observation(1_100, deviated=True))
    analysis.process_observation(observation(2_100, confidence=0.20))
    analysis.process_observation(observation(2_600))
    analysis.pause(timestamp_ms=3_100)
    stopped = analysis.stop(timestamp_ms=4_100)

    assert stopped.aggregates.eligible_monitoring_ms == 2_500
    assert stopped.aggregates.unknown_ms == 500
    assert stopped.aggregates.deviation_ms == 1_000
    assert stopped.aggregates.prompt_count == 0
    assert stopped.aggregates.ended_at_ms == 4_100


def test_long_observation_gap_is_unknown_not_eligible_monitoring_time() -> None:
    analysis = session(baseline())
    analysis.start(timestamp_ms=0)
    analysis.process_observation(observation(100))

    after_gap = analysis.process_observation(observation(5_100))

    assert after_gap.aggregates.eligible_monitoring_ms == 0
    assert after_gap.aggregates.unknown_ms == 5_000
    assert after_gap.aggregates.deviation_ms == 0


def test_prerecorded_observation_iterable_uses_the_same_session_path() -> None:
    analysis = session(baseline())
    analysis.start(timestamp_ms=0)

    updates = tuple(
        analysis.process_observations(
            [observation(100), observation(200), observation(300)]
        )
    )

    assert len(updates) == 3
    assert all(update.state is SessionState.MONITORING for update in updates)


def test_timestamps_are_monotonic_across_observations_and_controls() -> None:
    analysis = session(baseline())
    analysis.start(timestamp_ms=0)
    analysis.process_observation(observation(100))

    with pytest.raises(ValueError, match="increase"):
        analysis.pause(timestamp_ms=100)


def test_stopped_session_rejects_further_observations() -> None:
    analysis = session(baseline())
    analysis.start(timestamp_ms=0)
    analysis.stop(timestamp_ms=100)

    with pytest.raises(RuntimeError, match="stopped"):
        analysis.process_observation(observation(200))
