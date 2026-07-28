from __future__ import annotations

from goodposture.app import AnalysisSession, SessionState
from goodposture.core.calibration import (
    CALIBRATION_CONFIG_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    CalibrationBaseline,
    CalibrationConfig,
    CalibrationFeature,
)
from goodposture.core.models import Landmark, LandmarkName, ObservationFailure, PoseObservation
from goodposture.ui.calibration_flow import CalibrationFlow, CalibrationUiState


def observation(timestamp_ms: int, *, confidence: float = 0.95) -> PoseObservation:
    return PoseObservation(
        timestamp_ms=timestamp_ms,
        landmarks={
            LandmarkName.NOSE: Landmark(
                x=0.50,
                y=0.35,
                z=-0.12,
                visibility=confidence,
                presence=confidence,
            ),
            LandmarkName.LEFT_SHOULDER: Landmark(
                x=0.35,
                y=0.50,
                z=0.00,
                visibility=confidence,
                presence=confidence,
            ),
            LandmarkName.RIGHT_SHOULDER: Landmark(
                x=0.65,
                y=0.50,
                z=0.00,
                visibility=confidence,
                presence=confidence,
            ),
            LandmarkName.LEFT_HIP: Landmark(
                x=0.40,
                y=0.78,
                z=0.04,
                visibility=confidence,
                presence=confidence,
            ),
            LandmarkName.RIGHT_HIP: Landmark(
                x=0.60,
                y=0.78,
                z=0.04,
                visibility=confidence,
                presence=confidence,
            ),
        },
    )


def saved_baseline() -> CalibrationBaseline:
    return CalibrationBaseline(
        calibration_schema_version=CALIBRATION_SCHEMA_VERSION,
        calibration_config_version=CALIBRATION_CONFIG_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_id="pose-test-v1",
        started_at_ms=100,
        completed_at_ms=200,
        accepted_sample_count=60,
        rejected_sample_count=0,
        features=(
            CalibrationFeature("shoulder_tilt_degrees", 0.0, 1.0),
            CalibrationFeature("head_lateral_offset_ratio", 0.0, 0.02),
            CalibrationFeature("head_vertical_offset_ratio", 0.73, 0.02),
            CalibrationFeature("head_depth_ratio", 0.60, 0.02),
        ),
    )


def flow(
    *,
    required_samples: int = 3,
    maximum_span_ms: int = 1_000,
    baseline: CalibrationBaseline | None = None,
) -> CalibrationFlow:
    session = AnalysisSession(
        model_id="pose-test-v1",
        baseline=baseline,
        calibration_config=CalibrationConfig(
            minimum_accepted_samples=required_samples,
            maximum_sample_span_ms=maximum_span_ms,
        ),
    )
    return CalibrationFlow(session=session, countdown_ms=300)


def ready_camera(calibration_flow: CalibrationFlow) -> None:
    calibration_flow.accept_privacy(timestamp_ms=0)
    calibration_flow.set_cameras((0, 2))
    calibration_flow.select_camera(2)
    calibration_flow.camera_ready()


def test_privacy_step_is_explicit_local_non_medical_consent() -> None:
    calibration_flow = flow()

    view = calibration_flow.view

    assert view.state is CalibrationUiState.PRIVACY
    assert "on this device" in view.body
    assert "not stored" in view.body
    assert "not medical" in view.body
    assert view.primary_action == "Continue"


def test_user_can_leave_before_granting_camera_consent() -> None:
    calibration_flow = flow()

    calibration_flow.leave(timestamp_ms=0)

    assert calibration_flow.view.state is CalibrationUiState.CLOSED


def test_camera_selection_requires_consent_and_valid_camera() -> None:
    calibration_flow = flow()

    calibration_flow.accept_privacy(timestamp_ms=0)
    calibration_flow.set_cameras((0, 2))

    assert calibration_flow.view.state is CalibrationUiState.CAMERA_SELECTION
    assert calibration_flow.view.camera_indices == (0, 2)
    assert "2 local camera sources" in calibration_flow.view.body
    assert "virtual cameras" in calibration_flow.view.body
    calibration_flow.select_camera(2)
    assert calibration_flow.selected_camera_index == 2


def test_compatible_saved_baseline_skips_recapture_after_camera_is_ready() -> None:
    calibration_flow = flow(baseline=saved_baseline())
    calibration_flow.accept_privacy(timestamp_ms=0)
    calibration_flow.set_cameras((0,))
    calibration_flow.select_camera(0)

    calibration_flow.camera_ready()

    assert calibration_flow.view.state is CalibrationUiState.SUCCESS
    assert "saved calibration" in calibration_flow.view.body.lower()
    assert calibration_flow.session.baseline == saved_baseline()


def test_no_camera_and_camera_failure_have_recoverable_guidance() -> None:
    calibration_flow = flow()
    calibration_flow.accept_privacy(timestamp_ms=0)
    calibration_flow.set_cameras(())

    assert calibration_flow.view.state is CalibrationUiState.CAMERA_ERROR
    assert "camera" in calibration_flow.view.body.lower()

    calibration_flow.retry_camera_selection()
    calibration_flow.set_cameras((0,))
    calibration_flow.select_camera(0)
    calibration_flow.camera_error("Camera permission was denied.")
    assert calibration_flow.view.state is CalibrationUiState.CAMERA_ERROR
    assert calibration_flow.view.body == "Camera permission was denied."


def test_framing_quality_starts_and_resets_countdown() -> None:
    calibration_flow = flow()
    ready_camera(calibration_flow)

    assert calibration_flow.view.title == "Set your comfortable upright baseline"
    assert "head centered above your shoulders" in calibration_flow.view.body.lower()
    assert "countdown" in calibration_flow.view.body.lower()

    calibration_flow.process_observation(observation(100))
    assert calibration_flow.view.state is CalibrationUiState.COUNTDOWN
    assert calibration_flow.view.countdown_seconds == 1

    calibration_flow.process_observation(observation(200, confidence=0.2))
    assert calibration_flow.view.state is CalibrationUiState.FRAMING
    assert "both shoulders" in calibration_flow.view.body.lower()


def test_framing_explains_when_camera_has_no_detected_person() -> None:
    calibration_flow = flow()
    ready_camera(calibration_flow)

    calibration_flow.process_observation(PoseObservation(timestamp_ms=100, landmarks={}))

    assert calibration_flow.view.state is CalibrationUiState.FRAMING
    assert "Camera is active" in calibration_flow.view.body
    assert "no person" in calibration_flow.view.body
    assert "another camera source" in calibration_flow.view.body


def test_laptop_framing_does_not_require_hips() -> None:
    calibration_flow = flow()
    ready_camera(calibration_flow)
    partial_observation = observation(100)
    partial_observation = PoseObservation(
        timestamp_ms=partial_observation.timestamp_ms,
        landmarks={
            name: landmark
            for name, landmark in partial_observation.landmarks.items()
            if name not in {LandmarkName.LEFT_HIP, LandmarkName.RIGHT_HIP}
        },
    )

    calibration_flow.process_observation(partial_observation)

    assert calibration_flow.view.state is CalibrationUiState.COUNTDOWN
    assert "Framing looks good" in calibration_flow.view.body


def test_user_can_choose_another_camera_from_framing() -> None:
    calibration_flow = flow()
    ready_camera(calibration_flow)

    assert "Choose another camera" in calibration_flow.view.secondary_actions

    calibration_flow.choose_another_camera(timestamp_ms=100)

    assert calibration_flow.view.state is CalibrationUiState.CAMERA_SELECTION
    assert calibration_flow.selected_camera_index is None
    assert calibration_flow.view.camera_indices == ()


def test_switching_camera_cancels_partial_calibration() -> None:
    calibration_flow = flow()
    ready_camera(calibration_flow)
    calibration_flow.process_observation(observation(100))
    calibration_flow.process_observation(observation(400))
    calibration_flow.process_observation(observation(500))

    calibration_flow.choose_another_camera(timestamp_ms=600)

    assert calibration_flow.view.state is CalibrationUiState.CAMERA_SELECTION
    assert calibration_flow.session.state is SessionState.NEEDS_CALIBRATION
    assert calibration_flow.session.calibration_progress is None
    assert calibration_flow.session.baseline is None


def test_stable_confident_sequence_completes_calibration() -> None:
    calibration_flow = flow(required_samples=3)
    ready_camera(calibration_flow)

    calibration_flow.process_observation(observation(100))
    calibration_flow.process_observation(observation(400))
    assert calibration_flow.view.state is CalibrationUiState.CALIBRATING

    calibration_flow.process_observation(observation(500))
    calibration_flow.process_observation(observation(600))
    calibration_flow.process_observation(observation(700))
    assert calibration_flow.view.accepted_samples == 3
    calibration_flow.process_observation(observation(800))

    assert calibration_flow.view.state is CalibrationUiState.SUCCESS
    assert calibration_flow.session.baseline is not None


def test_insufficient_quality_fails_without_partial_baseline_and_can_retry() -> None:
    calibration_flow = flow(required_samples=3, maximum_span_ms=300)
    ready_camera(calibration_flow)
    calibration_flow.process_observation(observation(100))
    calibration_flow.process_observation(observation(400))

    calibration_flow.process_observation(observation(500, confidence=0.2))
    calibration_flow.process_observation(observation(900, confidence=0.2))
    calibration_flow.process_observation(observation(1_000, confidence=0.2))

    assert calibration_flow.view.state is CalibrationUiState.FAILURE
    assert calibration_flow.session.baseline is None
    assert "try again" in calibration_flow.view.body.lower()

    calibration_flow.retry(timestamp_ms=1_100)
    assert calibration_flow.view.state is CalibrationUiState.FRAMING


def test_recalibrate_delete_and_leave_are_available_after_success() -> None:
    calibration_flow = flow(required_samples=1)
    ready_camera(calibration_flow)
    calibration_flow.process_observation(observation(100))
    calibration_flow.process_observation(observation(400))
    calibration_flow.process_observation(observation(500))
    calibration_flow.process_observation(observation(600))
    assert calibration_flow.view.state is CalibrationUiState.SUCCESS

    calibration_flow.recalibrate(timestamp_ms=700)
    assert calibration_flow.view.state is CalibrationUiState.FRAMING
    assert calibration_flow.session.baseline is not None

    calibration_flow.delete_baseline(timestamp_ms=800)
    assert calibration_flow.session.baseline is None
    assert calibration_flow.view.state is CalibrationUiState.CAMERA_SELECTION

    calibration_flow.leave(timestamp_ms=900)
    assert calibration_flow.view.state is CalibrationUiState.CLOSED


def test_capture_failure_during_calibration_never_becomes_bad_posture() -> None:
    calibration_flow = flow()
    ready_camera(calibration_flow)
    calibration_flow.process_observation(observation(100))
    calibration_flow.process_observation(observation(400))

    calibration_flow.process_failure(
        timestamp_ms=500,
        failure=ObservationFailure.CAPTURE_READ_FAILED,
        message="The camera disconnected.",
    )

    assert calibration_flow.view.state is CalibrationUiState.CAMERA_ERROR
    assert calibration_flow.session.baseline is None
    assert "disconnected" in calibration_flow.view.body
