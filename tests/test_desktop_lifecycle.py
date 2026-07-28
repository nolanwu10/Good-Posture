from __future__ import annotations

from dataclasses import replace

from goodposture.app.desktop_lifecycle import (
    DesktopLifecycle,
    DesktopState,
    _tray_state,
)
from goodposture.app.diagnostics import DiagnosticEvent
from goodposture.app.session import AnalysisSession, SessionAggregates, SessionState
from goodposture.core.alert_policy import AlertDecision, AlertState
from goodposture.core.calibration import (
    CALIBRATION_CONFIG_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    CalibrationBaseline,
    CalibrationFeature,
)
from goodposture.core.models import ObservationFailure, PoseObservation


def baseline() -> CalibrationBaseline:
    return CalibrationBaseline(
        calibration_schema_version=CALIBRATION_SCHEMA_VERSION,
        calibration_config_version=CALIBRATION_CONFIG_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_id="pose-landmarker-lite@sha256:abc",
        started_at_ms=0,
        completed_at_ms=100,
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


class FakeCamera:
    def __init__(
        self,
        *,
        fail_pause: bool = False,
        fail_shutdown: bool = False,
    ) -> None:
        self.calls: list[object] = []
        self.fail_pause = fail_pause
        self.fail_shutdown = fail_shutdown

    def start_camera(self, camera_index: int) -> None:
        self.calls.append(("start", camera_index))

    def pause(self) -> None:
        self.calls.append("pause")
        if self.fail_pause:
            raise OSError("camera pause failed")

    def shutdown(self) -> None:
        self.calls.append("shutdown")
        if self.fail_shutdown:
            raise OSError("camera shutdown failed")


class FakeSink:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.values: list[SessionAggregates] = []

    def flush(self, aggregates: SessionAggregates) -> None:
        self.values.append(aggregates)
        if self.fail:
            raise OSError("simulated persistence failure")


class FakeDiagnostics:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.events: list[DiagnosticEvent] = []

    def record(self, event: DiagnosticEvent) -> None:
        self.events.append(event)
        if self.fail:
            raise OSError("diagnostic log unavailable")


def lifecycle(
    *,
    sink: FakeSink | None = None,
    diagnostics: FakeDiagnostics | None = None,
    checkpoint_interval_ms: int = 60_000,
) -> tuple[DesktopLifecycle, FakeCamera, FakeSink]:
    active_session = AnalysisSession(model_id=baseline().model_id, baseline=baseline())
    active_session.start(timestamp_ms=1_000)
    camera = FakeCamera()
    actual_sink = sink or FakeSink()

    def session_factory(saved_baseline: CalibrationBaseline) -> AnalysisSession:
        return AnalysisSession(model_id=saved_baseline.model_id, baseline=saved_baseline)

    return (
        DesktopLifecycle(
            session=active_session,
            camera=camera,
            camera_index=2,
            session_factory=session_factory,
            aggregate_sink=actual_sink,
            diagnostics=diagnostics,
            aggregate_checkpoint_interval_ms=checkpoint_interval_ms,
        ),
        camera,
        actual_sink,
    )


def test_construction_never_starts_background_camera() -> None:
    controller, camera, _ = lifecycle()

    assert controller.state is DesktopState.STOPPED
    assert controller.tray.status == "Monitoring is stopped. The camera is off."
    assert controller.tray.recalibrate_enabled is False
    assert camera.calls == []


def test_start_adapter_readiness_unknown_and_error_stay_synchronized() -> None:
    controller, camera, _ = lifecycle()

    starting = controller.start(timestamp_ms=1_100)
    ready = controller.camera_ready()
    unknown = controller.process_observation(
        PoseObservation(timestamp_ms=1_200, landmarks={})
    )
    error = controller.process_failure(
        timestamp_ms=1_300,
        failure=ObservationFailure.CAPTURE_READ_FAILED,
    )

    assert camera.calls == [("start", 2)]
    assert starting.state is DesktopState.STARTING
    assert ready.state is DesktopState.MONITORING
    assert unknown.state is DesktopState.UNKNOWN
    assert "uncertain" in unknown.tray.status
    assert error.state is DesktopState.ERROR
    assert "camera is off" in error.tray.status.lower()


def test_pause_resume_quiet_stop_restart_and_exit_release_resources() -> None:
    controller, camera, sink = lifecycle()
    controller.start(timestamp_ms=1_100)
    controller.camera_ready()

    quiet = controller.set_quiet_mode(enabled=True, timestamp_ms=1_200)
    paused = controller.pause(timestamp_ms=1_300)
    resumed = controller.resume(timestamp_ms=1_400)
    stopped = controller.stop(timestamp_ms=1_500)
    restarted = controller.start(timestamp_ms=1_600)
    exited = controller.exit(timestamp_ms=1_700)

    assert quiet.tray.quiet_checked is True
    assert paused.state is DesktopState.PAUSED
    assert paused.tray.pause_resume_label == "Resume monitoring"
    assert resumed.state is DesktopState.STARTING
    assert stopped.state is DesktopState.STOPPED
    assert stopped.flush_succeeded is True
    assert restarted.state is DesktopState.STARTING
    assert exited.state is DesktopState.EXITED
    assert camera.calls == [
        ("start", 2),
        "pause",
        ("start", 2),
        "pause",
        ("start", 2),
        "pause",
        "shutdown",
    ]
    assert len(sink.values) == 2


def test_exit_is_safe_even_when_aggregate_flush_fails() -> None:
    controller, camera, sink = lifecycle(sink=FakeSink(fail=True))
    controller.start(timestamp_ms=1_100)

    result = controller.exit(timestamp_ms=1_200)

    assert result.state is DesktopState.EXITED
    assert result.flush_succeeded is False
    assert camera.calls[-2:] == ["pause", "shutdown"]
    assert len(sink.values) == 1


def test_lifecycle_records_state_changes_without_detailed_session_data() -> None:
    diagnostics = FakeDiagnostics()
    controller, camera, _sink = lifecycle(diagnostics=diagnostics)

    controller.start(timestamp_ms=1_100)
    controller.camera_ready()
    controller.pause(timestamp_ms=1_200)
    controller.resume(timestamp_ms=1_300)
    controller.stop(timestamp_ms=1_400)
    controller.exit(timestamp_ms=1_500)

    assert diagnostics.events == [
        DiagnosticEvent.CAMERA_STARTED,
        DiagnosticEvent.CAMERA_READY,
        DiagnosticEvent.CAMERA_PAUSED,
        DiagnosticEvent.CAMERA_STARTED,
        DiagnosticEvent.SESSION_STOPPED,
        DiagnosticEvent.APP_EXITED,
    ]
    assert camera.calls[-1] == "shutdown"


def test_diagnostic_write_failure_never_interrupts_lifecycle() -> None:
    controller, camera, _sink = lifecycle(diagnostics=FakeDiagnostics(fail=True))

    started = controller.start(timestamp_ms=1_100)
    exited = controller.exit(timestamp_ms=1_200)

    assert started.state is DesktopState.STARTING
    assert exited.state is DesktopState.EXITED
    assert camera.calls[-1] == "shutdown"


def test_exit_contains_camera_cleanup_failures_and_still_marks_exited() -> None:
    active_session = AnalysisSession(model_id=baseline().model_id, baseline=baseline())
    active_session.start(timestamp_ms=1_000)
    camera = FakeCamera(fail_pause=True, fail_shutdown=True)
    sink = FakeSink()
    diagnostics = FakeDiagnostics()

    controller = DesktopLifecycle(
        session=active_session,
        camera=camera,
        camera_index=0,
        session_factory=lambda saved: AnalysisSession(
            model_id=saved.model_id,
            baseline=saved,
        ),
        aggregate_sink=sink,
        diagnostics=diagnostics,
    )
    controller.start(timestamp_ms=1_100)

    result = controller.exit(timestamp_ms=1_200)

    assert result.state is DesktopState.EXITED
    assert camera.calls[-2:] == ["pause", "shutdown"]
    assert diagnostics.events[-1] is DiagnosticEvent.APP_EXITED


def test_recalibration_request_pauses_and_releases_camera() -> None:
    controller, camera, _ = lifecycle()
    controller.start(timestamp_ms=1_100)

    result = controller.request_recalibration(timestamp_ms=1_200)

    assert result.recalibration_requested is True
    assert result.state is DesktopState.PAUSED
    assert controller.session.state is SessionState.PAUSED
    assert camera.calls[-1] == "pause"


def test_quiet_mode_suppresses_alert_policy_while_analysis_continues() -> None:
    controller, _, _ = lifecycle()
    controller.start(timestamp_ms=1_100)
    controller.camera_ready()
    controller.set_quiet_mode(enabled=True, timestamp_ms=1_200)

    result = controller.process_observation(
        PoseObservation(timestamp_ms=1_300, landmarks={})
    )

    assert result.session_update is not None
    assert result.session_update.alert is not None
    assert result.session_update.alert.state.value == "paused"


def test_tray_explains_active_reminder_cooldown_without_a_popup() -> None:
    alert = AlertDecision(
        timestamp_ms=1_000,
        state=AlertState.COOLDOWN,
        event=None,
        prompt_message=None,
        pending_since_ms=None,
        cooldown_until_ms=601_000,
        config_version=4,
        continuous_deviation_ms=0,
        posture_debt_ms=0.0,
        trigger=None,
    )

    tray = _tray_state(DesktopState.MONITORING, quiet=False, alert=alert)

    assert "monitoring locally" in tray.status.lower()
    assert "next reminder in about 10 min" in tray.status.lower()


def test_tray_reports_short_refractory_period_in_seconds() -> None:
    alert = AlertDecision(
        timestamp_ms=1_000,
        state=AlertState.COOLDOWN,
        event=None,
        prompt_message=None,
        pending_since_ms=None,
        cooldown_until_ms=21_000,
        config_version=6,
        continuous_deviation_ms=0,
        posture_debt_ms=0.0,
        trigger=None,
    )

    tray = _tray_state(DesktopState.MONITORING, quiet=False, alert=alert)

    assert "next reminder in about 20 sec" in tray.status.lower()


def test_camera_error_without_core_failure_still_updates_tray() -> None:
    controller, _, _ = lifecycle()
    controller.start(timestamp_ms=1_100)

    result = controller.camera_error()

    assert result.state is DesktopState.ERROR
    assert "camera is off" in result.tray.status.lower()


def test_session_factory_preserves_calibration_baseline_on_restart() -> None:
    controller, _, _ = lifecycle()
    saved = controller.session.baseline
    controller.start(timestamp_ms=1_100)
    controller.stop(timestamp_ms=1_200)

    controller.start(timestamp_ms=1_300)

    assert controller.session.baseline == saved
    assert controller.session.state is SessionState.MONITORING


def test_baseline_fixture_is_not_mutated_by_lifecycle() -> None:
    original = baseline()
    modified = replace(original, completed_at_ms=101)

    assert original.completed_at_ms == 100
    assert modified.completed_at_ms == 101


def test_active_session_is_checkpointed_periodically_then_finalized() -> None:
    controller, _, sink = lifecycle(checkpoint_interval_ms=100)
    controller.start(timestamp_ms=1_100)
    controller.camera_ready()

    before_interval = controller.process_observation(
        PoseObservation(timestamp_ms=1_199, landmarks={})
    )
    at_interval = controller.process_observation(
        PoseObservation(timestamp_ms=1_200, landmarks={})
    )
    stopped = controller.stop(timestamp_ms=1_300)

    assert before_interval.flush_succeeded is None
    assert at_interval.flush_succeeded is True
    assert stopped.flush_succeeded is True
    assert len(sink.values) == 2
    assert sink.values[0].ended_at_ms is None
    assert sink.values[1].ended_at_ms == 1_300
