"""Deterministic first-run and calibration presentation state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from goodposture.app import AnalysisSession, SessionState
from goodposture.core.metrics import extract_posture_metrics
from goodposture.core.models import LandmarkName, ObservationFailure, PoseObservation

_PRIVACY_BODY = (
    "Pose processing happens on this device. Raw camera images are used only "
    "for immediate analysis and are not stored or uploaded. GoodPosture is a "
    "comfort and movement awareness tool, not medical guidance."
)


class CalibrationUiState(StrEnum):
    """User-visible first-run states independent of Qt."""

    PRIVACY = "privacy"
    CAMERA_SELECTION = "camera_selection"
    CAMERA_ERROR = "camera_error"
    FRAMING = "framing"
    COUNTDOWN = "countdown"
    CALIBRATING = "calibrating"
    SUCCESS = "success"
    FAILURE = "failure"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class CalibrationView:
    """Complete render model with no camera frames or landmarks."""

    state: CalibrationUiState
    title: str
    body: str
    primary_action: str | None
    secondary_actions: tuple[str, ...] = ()
    camera_indices: tuple[int, ...] = ()
    countdown_seconds: int | None = None
    accepted_samples: int = 0
    required_samples: int = 0


class CalibrationFlow:
    """Coordinate consent, framing, countdown, and calibration quality."""

    def __init__(
        self,
        *,
        session: AnalysisSession,
        countdown_ms: int = 3_000,
        minimum_framing_confidence: float = 0.70,
    ) -> None:
        if countdown_ms <= 0:
            raise ValueError("countdown_ms must be positive")
        if not 0.0 <= minimum_framing_confidence <= 1.0:
            raise ValueError("minimum_framing_confidence must be between 0 and 1")
        self._session = session
        self._countdown_ms = countdown_ms
        self._minimum_framing_confidence = minimum_framing_confidence
        self._state = CalibrationUiState.PRIVACY
        self._camera_indices: tuple[int, ...] = ()
        self._selected_camera_index: int | None = None
        self._countdown_started_at_ms: int | None = None
        self._message = _PRIVACY_BODY

    @property
    def session(self) -> AnalysisSession:
        return self._session

    @property
    def selected_camera_index(self) -> int | None:
        return self._selected_camera_index

    @property
    def view(self) -> CalibrationView:
        progress = self._session.calibration_progress
        accepted = 0 if progress is None else progress.accepted_sample_count
        required = 0 if progress is None else progress.required_sample_count
        countdown_seconds: int | None = None
        if self._state is CalibrationUiState.COUNTDOWN:
            countdown_seconds = max(
                1,
                math.ceil(self._remaining_countdown_ms / 1_000),
            )
        title, primary, secondary = _presentation(self._state)
        return CalibrationView(
            state=self._state,
            title=title,
            body=self._message,
            primary_action=primary,
            secondary_actions=secondary,
            camera_indices=self._camera_indices,
            countdown_seconds=countdown_seconds,
            accepted_samples=accepted,
            required_samples=required,
        )

    @property
    def _remaining_countdown_ms(self) -> int:
        return getattr(self, "_last_countdown_remaining_ms", self._countdown_ms)

    def accept_privacy(self, *, timestamp_ms: int) -> None:
        self._require_state(CalibrationUiState.PRIVACY)
        self._session.start(timestamp_ms=timestamp_ms)
        self._state = CalibrationUiState.CAMERA_SELECTION
        self._message = "Choose a camera. It will turn on only after you continue."

    def set_cameras(self, camera_indices: tuple[int, ...]) -> None:
        if self._state not in {
            CalibrationUiState.CAMERA_SELECTION,
            CalibrationUiState.CAMERA_ERROR,
        }:
            raise RuntimeError("camera discovery is not active")
        if any(index < 0 for index in camera_indices):
            raise ValueError("camera indices must be non-negative")
        self._camera_indices = tuple(dict.fromkeys(camera_indices))
        if not self._camera_indices:
            self._state = CalibrationUiState.CAMERA_ERROR
            self._message = (
                "No available camera was found. Check Windows camera access, "
                "then try again."
            )
            return
        self._state = CalibrationUiState.CAMERA_SELECTION
        count = len(self._camera_indices)
        noun = "source" if count == 1 else "sources"
        self._message = (
            f"{count} local camera {noun} found. Windows may include virtual cameras. "
            "Choose one; you can return and try another if it cannot detect you."
        )

    def select_camera(self, camera_index: int) -> None:
        self._require_state(CalibrationUiState.CAMERA_SELECTION)
        if camera_index not in self._camera_indices:
            raise ValueError("camera_index is not an available camera")
        self._selected_camera_index = camera_index
        self._message = "Opening the selected camera locally…"

    def camera_ready(self) -> None:
        self._require_state(CalibrationUiState.CAMERA_SELECTION)
        if self._selected_camera_index is None:
            raise RuntimeError("select a camera before opening it")
        if self._session.baseline is not None:
            self._state = CalibrationUiState.SUCCESS
            self._message = (
                "Your saved calibration is ready for this camera. You can use it, "
                "recalibrate, or delete it. Raw camera images are not stored."
            )
            return
        self._state = CalibrationUiState.FRAMING
        self._message = (
            "Sit as upright as feels natural, with your head centered above your "
            "shoulders and your shoulders relaxed. Your hips do not need to be "
            "visible. When head-and-shoulder tracking stays clear, a short countdown "
            "will begin before the baseline is captured. No image is stored or uploaded."
        )

    def camera_error(self, message: str) -> None:
        if not message.strip():
            raise ValueError("camera error message must be non-empty")
        self._state = CalibrationUiState.CAMERA_ERROR
        self._message = message

    def retry_camera_selection(self) -> None:
        self._state = CalibrationUiState.CAMERA_SELECTION
        self._selected_camera_index = None
        self._camera_indices = ()
        self._message = "Checking available cameras…"

    def choose_another_camera(self, *, timestamp_ms: int) -> None:
        if self._state not in {
            CalibrationUiState.FRAMING,
            CalibrationUiState.COUNTDOWN,
            CalibrationUiState.CALIBRATING,
        }:
            raise RuntimeError("camera switching is not available")
        if self._session.state is SessionState.CALIBRATING:
            self._session.cancel_calibration(timestamp_ms=timestamp_ms)
        self._countdown_started_at_ms = None
        self.retry_camera_selection()

    def process_observation(self, observation: PoseObservation) -> None:
        if self._state is CalibrationUiState.FRAMING:
            well_framed, guidance = self._framing_feedback(observation)
            if well_framed:
                self._state = CalibrationUiState.COUNTDOWN
                self._countdown_started_at_ms = observation.timestamp_ms
                self._last_countdown_remaining_ms = self._countdown_ms
                self._message = "Framing looks good. Stay comfortably still."
            else:
                self._message = guidance
            return

        if self._state is CalibrationUiState.COUNTDOWN:
            well_framed, guidance = self._framing_feedback(observation)
            if not well_framed:
                self._state = CalibrationUiState.FRAMING
                self._countdown_started_at_ms = None
                self._message = guidance
                return
            assert self._countdown_started_at_ms is not None
            remaining = self._countdown_ms - (
                observation.timestamp_ms - self._countdown_started_at_ms
            )
            self._last_countdown_remaining_ms = max(0, remaining)
            if remaining <= 0:
                self._session.start_calibration(timestamp_ms=observation.timestamp_ms)
                self._state = CalibrationUiState.CALIBRATING
                self._message = (
                    "Keep the comfortable upright position you chose while clear, "
                    "stable head-and-shoulder measurements are collected."
                )
            return

        if self._state is not CalibrationUiState.CALIBRATING:
            return

        progress = self._session.calibration_progress
        assert progress is not None
        if progress.is_complete or progress.timed_out:
            update = self._session.finish_calibration(
                timestamp_ms=observation.timestamp_ms
            )
            if update.state in {SessionState.MONITORING, SessionState.PAUSED}:
                self._state = CalibrationUiState.SUCCESS
                self._message = (
                    "Calibration is ready. Your baseline reflects this comfortable "
                    "position and can be changed anytime."
                )
            else:
                self._state = CalibrationUiState.FAILURE
                self._message = (
                    update.events[-1].message
                    + " Improve framing or lighting, then try again."
                )
            return

        self._session.process_observation(observation)
        progress = self._session.calibration_progress
        assert progress is not None
        self._message = (
            f"Collecting a comfortable baseline: "
            f"{progress.accepted_sample_count}/{progress.required_sample_count} "
            "clear samples."
        )

    def process_failure(
        self,
        *,
        timestamp_ms: int,
        failure: ObservationFailure,
        message: str,
    ) -> None:
        if self._session.state is SessionState.CALIBRATING:
            self._session.process_failure(timestamp_ms=timestamp_ms, failure=failure)
            self._session.cancel_calibration(timestamp_ms=timestamp_ms + 1)
        self.camera_error(message)

    def retry(self, *, timestamp_ms: int) -> None:
        del timestamp_ms
        self._require_state(CalibrationUiState.FAILURE)
        self._state = CalibrationUiState.FRAMING
        self._countdown_started_at_ms = None
        self._message = _FRAMING_GUIDANCE

    def recalibrate(self, *, timestamp_ms: int) -> None:
        del timestamp_ms
        self._require_state(CalibrationUiState.SUCCESS)
        self._state = CalibrationUiState.FRAMING
        self._countdown_started_at_ms = None
        self._message = _FRAMING_GUIDANCE

    def delete_baseline(self, *, timestamp_ms: int) -> None:
        self._session.delete_baseline(timestamp_ms=timestamp_ms)
        self._state = CalibrationUiState.CAMERA_SELECTION
        self._selected_camera_index = None
        self._message = "The current baseline was deleted. Choose a camera to calibrate again."

    def leave(self, *, timestamp_ms: int) -> None:
        if self._session.state is SessionState.CREATED:
            self._state = CalibrationUiState.CLOSED
            self._message = "GoodPosture setup closed."
            return
        if self._session.state is SessionState.CALIBRATING:
            self._session.cancel_calibration(timestamp_ms=timestamp_ms)
            timestamp_ms += 1
        if self._session.state is not SessionState.STOPPED:
            self._session.stop(timestamp_ms=timestamp_ms)
        self._state = CalibrationUiState.CLOSED
        self._message = "GoodPosture setup closed."

    def _framing_feedback(self, observation: PoseObservation) -> tuple[bool, str]:
        metrics = extract_posture_metrics(observation)
        readings = (
            metrics.shoulder_tilt_degrees,
            metrics.head_lateral_offset_ratio,
            metrics.head_vertical_offset_ratio,
            metrics.head_depth_ratio,
        )
        if all(
            reading.value is not None
            and reading.confidence >= self._minimum_framing_confidence
            for reading in readings
        ):
            return True, "Framing looks good."

        if not observation.landmarks:
            return (
                False,
                "Camera is active, but no person is detected. Check that the camera "
                "light is on and nothing covers the lens, or choose another camera source.",
            )

        def landmarks_are_clear(names: tuple[LandmarkName, ...]) -> bool:
            return all(
                (landmark := observation.landmarks.get(name)) is not None
                and landmark.confidence >= self._minimum_framing_confidence
                for name in names
            )

        if not landmarks_are_clear(
            (LandmarkName.LEFT_SHOULDER, LandmarkName.RIGHT_SHOULDER)
        ):
            return (
                False,
                "Camera is active and a person is detected. Move until both shoulders "
                "are visible, or try another camera source.",
            )
        if not landmarks_are_clear((LandmarkName.NOSE,)):
            return (
                False,
                "Camera is active and your upper body is visible. Adjust slightly so "
                "your head is also detected clearly.",
            )
        return (
            False,
            "Camera is active and your upper body is detected, but tracking is still "
            "uncertain. Adjust lighting or shift slightly, then settle comfortably.",
        )

    def _require_state(self, expected: CalibrationUiState) -> None:
        if self._state is not expected:
            raise RuntimeError(f"expected {expected.value}, found {self._state.value}")


_FRAMING_GUIDANCE = (
    "Sit as upright as feels natural, with your head centered above your shoulders "
    "and both shoulders relaxed and visible. Your hips do not need to be visible. "
    "A short countdown starts only after tracking remains clear."
)


def _presentation(
    state: CalibrationUiState,
) -> tuple[str, str | None, tuple[str, ...]]:
    presentations = {
        CalibrationUiState.PRIVACY: ("Before we begin", "Continue", ("Leave",)),
        CalibrationUiState.CAMERA_SELECTION: ("Choose a camera", "Use this camera", ("Leave",)),
        CalibrationUiState.CAMERA_ERROR: ("Camera unavailable", "Try again", ("Leave",)),
        CalibrationUiState.FRAMING: (
            "Set your comfortable upright baseline",
            None,
            ("Choose another camera", "Leave"),
        ),
        CalibrationUiState.COUNTDOWN: (
            "Hold that comfortable position",
            None,
            ("Choose another camera", "Leave"),
        ),
        CalibrationUiState.CALIBRATING: (
            "Creating your baseline",
            None,
            ("Choose another camera", "Leave"),
        ),
        CalibrationUiState.SUCCESS: (
            "Calibration complete",
            "Finish",
            ("Recalibrate", "Delete baseline"),
        ),
        CalibrationUiState.FAILURE: ("Let's try that again", "Retry", ("Leave",)),
        CalibrationUiState.CLOSED: ("Setup closed", None, ()),
    }
    return presentations[state]
