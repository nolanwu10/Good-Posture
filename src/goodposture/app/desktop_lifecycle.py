"""Deterministic tray-facing ownership of a headless analysis session."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from goodposture.app.diagnostics import DiagnosticEvent, DiagnosticSink
from goodposture.app.session import (
    AnalysisSession,
    SessionAggregates,
    SessionState,
    SessionUpdate,
)
from goodposture.core.alert_policy import AlertDecision, AlertState
from goodposture.core.calibration import CalibrationBaseline
from goodposture.core.models import ObservationFailure, PoseObservation


class CameraControl(Protocol):
    """Resource-owning camera surface needed by the desktop lifecycle."""

    def start_camera(self, camera_index: int) -> None: ...

    def pause(self) -> None: ...

    def shutdown(self) -> None: ...


class AggregateSink(Protocol):
    """Task-11-ready boundary that receives derived rollups only."""

    def flush(self, aggregates: SessionAggregates) -> None: ...


SessionFactory = Callable[[CalibrationBaseline], AnalysisSession]


class DesktopState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    MONITORING = "monitoring"
    UNKNOWN = "unknown"
    PAUSED = "paused"
    ERROR = "error"
    EXITED = "exited"


@dataclass(frozen=True, slots=True)
class TrayState:
    """Complete, framework-neutral tray render state."""

    status: str
    start_stop_label: str
    start_stop_enabled: bool
    pause_resume_label: str
    pause_resume_enabled: bool
    quiet_enabled: bool
    quiet_checked: bool
    recalibrate_enabled: bool
    exited: bool = False


@dataclass(frozen=True, slots=True)
class LifecycleResult:
    """One bounded lifecycle transition."""

    state: DesktopState
    tray: TrayState
    session_update: SessionUpdate | None = None
    flush_succeeded: bool | None = None
    recalibration_requested: bool = False


class MemoryAggregateSink:
    """Keep the latest derived rollup in memory until Task 11 adds SQLite."""

    def __init__(self) -> None:
        self.latest: SessionAggregates | None = None

    def flush(self, aggregates: SessionAggregates) -> None:
        self.latest = aggregates


class DesktopLifecycle:
    """Synchronize camera ownership, session state, and tray actions."""

    def __init__(
        self,
        *,
        session: AnalysisSession,
        camera: CameraControl,
        camera_index: int,
        session_factory: SessionFactory,
        aggregate_sink: AggregateSink,
        diagnostics: DiagnosticSink | None = None,
        aggregate_checkpoint_interval_ms: int = 60_000,
    ) -> None:
        if camera_index < 0:
            raise ValueError("camera_index must be non-negative")
        if session.baseline is None:
            raise ValueError("a calibrated session is required")
        if session.state not in {SessionState.MONITORING, SessionState.PAUSED}:
            raise ValueError("session must be ready to monitor")
        if aggregate_checkpoint_interval_ms <= 0:
            raise ValueError("aggregate_checkpoint_interval_ms must be positive")
        self._session = session
        self._camera = camera
        self._camera_index = camera_index
        self._session_factory = session_factory
        self._aggregate_sink = aggregate_sink
        self._diagnostics = diagnostics
        self._aggregate_checkpoint_interval_ms = aggregate_checkpoint_interval_ms
        self._state = DesktopState.STOPPED
        self._flushed_session = False
        self._last_checkpoint_ms: int | None = None

    @property
    def session(self) -> AnalysisSession:
        return self._session

    @property
    def state(self) -> DesktopState:
        return self._state

    @property
    def tray(self) -> TrayState:
        return _tray_state(self._state, quiet=self._session.quiet_mode)

    def current(self) -> LifecycleResult:
        return self._result()

    def start(self, *, timestamp_ms: int) -> LifecycleResult:
        self._ensure_not_exited()
        if self._state is not DesktopState.STOPPED:
            raise RuntimeError("monitoring is already active")
        if self._session.state is SessionState.STOPPED:
            baseline = self._session.baseline
            assert baseline is not None
            self._session = self._session_factory(baseline)
            update = self._session.start(timestamp_ms=timestamp_ms)
        elif self._session.state is SessionState.PAUSED:
            update = self._session.resume(timestamp_ms=timestamp_ms)
        else:
            update = None
        self._flushed_session = False
        self._last_checkpoint_ms = timestamp_ms
        self._camera.start_camera(self._camera_index)
        self._record(DiagnosticEvent.CAMERA_STARTED)
        self._state = DesktopState.STARTING
        return self._result(update=update)

    def camera_ready(self) -> LifecycleResult:
        self._ensure_active()
        self._state = DesktopState.MONITORING
        self._record(DiagnosticEvent.CAMERA_READY)
        return self._result(update=self._session.snapshot())

    def process_observation(self, observation: PoseObservation) -> LifecycleResult:
        self._ensure_active()
        previous_state = self._state
        update = self._session.process_observation(observation)
        self._state = _desktop_state(update.state)
        if self._state is DesktopState.UNKNOWN and previous_state is not DesktopState.UNKNOWN:
            self._record(DiagnosticEvent.TRACKING_UNKNOWN)
        elif (
            self._state is DesktopState.MONITORING
            and previous_state in {DesktopState.UNKNOWN, DesktopState.ERROR}
        ):
            self._record(DiagnosticEvent.TRACKING_RECOVERED)
        return self._result(
            update=update,
            flush_succeeded=self._checkpoint_if_due(observation.timestamp_ms),
        )

    def process_failure(
        self,
        *,
        timestamp_ms: int,
        failure: ObservationFailure,
    ) -> LifecycleResult:
        self._ensure_active()
        previous_state = self._state
        update = self._session.process_failure(
            timestamp_ms=timestamp_ms,
            failure=failure,
        )
        self._state = DesktopState.ERROR
        if previous_state is not DesktopState.ERROR:
            self._record(DiagnosticEvent.CAMERA_FAILURE)
        return self._result(
            update=update,
            flush_succeeded=self._checkpoint_if_due(timestamp_ms),
        )

    def camera_error(self) -> LifecycleResult:
        """Reflect an adapter error that has no core observation failure."""

        self._ensure_active()
        self._state = DesktopState.ERROR
        self._record(DiagnosticEvent.CAMERA_FAILURE)
        return self._result()

    def pause(self, *, timestamp_ms: int) -> LifecycleResult:
        self._ensure_active()
        self._camera.pause()
        self._record(DiagnosticEvent.CAMERA_PAUSED)
        update = self._session.pause(timestamp_ms=timestamp_ms)
        self._state = DesktopState.PAUSED
        return self._result(update=update)

    def resume(self, *, timestamp_ms: int) -> LifecycleResult:
        self._ensure_not_exited()
        if self._state is not DesktopState.PAUSED:
            raise RuntimeError("monitoring is not paused")
        update = self._session.resume(timestamp_ms=timestamp_ms)
        self._camera.start_camera(self._camera_index)
        self._record(DiagnosticEvent.CAMERA_STARTED)
        self._state = DesktopState.STARTING
        return self._result(update=update)

    def set_quiet_mode(
        self,
        *,
        enabled: bool,
        timestamp_ms: int,
    ) -> LifecycleResult:
        self._ensure_active()
        update = self._session.set_quiet_mode(
            enabled=enabled,
            timestamp_ms=timestamp_ms,
        )
        return self._result(update=update)

    def request_recalibration(self, *, timestamp_ms: int) -> LifecycleResult:
        self._ensure_active()
        self._camera.pause()
        self._record(DiagnosticEvent.CAMERA_PAUSED)
        update: SessionUpdate | None = None
        if self._session.state is not SessionState.PAUSED:
            update = self._session.pause(timestamp_ms=timestamp_ms)
        self._state = DesktopState.PAUSED
        result = self._result(update=update)
        return LifecycleResult(
            state=result.state,
            tray=result.tray,
            session_update=result.session_update,
            recalibration_requested=True,
        )

    def stop(self, *, timestamp_ms: int) -> LifecycleResult:
        self._ensure_not_exited()
        self._camera.pause()
        update: SessionUpdate | None = None
        if self._session.state is not SessionState.STOPPED:
            update = self._session.stop(timestamp_ms=timestamp_ms)
        flush_succeeded = self._flush_once()
        self._record(DiagnosticEvent.SESSION_STOPPED)
        self._state = DesktopState.STOPPED
        return self._result(update=update, flush_succeeded=flush_succeeded)

    def exit(self, *, timestamp_ms: int) -> LifecycleResult:
        self._ensure_not_exited()
        update: SessionUpdate | None = None
        flush_succeeded: bool | None = None
        try:
            if self._session.state is not SessionState.STOPPED:
                try:
                    self._camera.pause()
                except Exception:
                    self._record(DiagnosticEvent.CAMERA_FAILURE)
                try:
                    update = self._session.stop(timestamp_ms=timestamp_ms)
                except Exception:
                    self._record(DiagnosticEvent.UNHANDLED_FAILURE)
                flush_succeeded = self._flush_once()
        finally:
            try:
                self._camera.shutdown()
            except Exception:
                self._record(DiagnosticEvent.CAMERA_FAILURE)
            finally:
                self._state = DesktopState.EXITED
                self._record(DiagnosticEvent.APP_EXITED)
        return self._result(update=update, flush_succeeded=flush_succeeded)

    def _flush_once(self) -> bool:
        if self._flushed_session:
            return True
        try:
            self._aggregate_sink.flush(self._session.aggregates)
        except Exception:
            self._record(DiagnosticEvent.CHECKPOINT_FAILED)
            return False
        self._flushed_session = True
        return True

    def _checkpoint_if_due(self, timestamp_ms: int) -> bool | None:
        previous = self._last_checkpoint_ms
        if previous is None:
            self._last_checkpoint_ms = timestamp_ms
            return None
        if timestamp_ms - previous < self._aggregate_checkpoint_interval_ms:
            return None
        self._last_checkpoint_ms = timestamp_ms
        try:
            self._aggregate_sink.flush(self._session.aggregates)
        except Exception:
            self._record(DiagnosticEvent.CHECKPOINT_FAILED)
            return False
        return True

    def _record(self, event: DiagnosticEvent) -> None:
        if self._diagnostics is None:
            return
        try:
            self._diagnostics.record(event)
        except Exception:
            return

    def _ensure_active(self) -> None:
        self._ensure_not_exited()
        if self._state is DesktopState.STOPPED:
            raise RuntimeError("monitoring is stopped")

    def _ensure_not_exited(self) -> None:
        if self._state is DesktopState.EXITED:
            raise RuntimeError("desktop lifecycle has exited")

    def _result(
        self,
        *,
        update: SessionUpdate | None = None,
        flush_succeeded: bool | None = None,
    ) -> LifecycleResult:
        return LifecycleResult(
            state=self._state,
            tray=_tray_state(
                self._state,
                quiet=self._session.quiet_mode,
                alert=None if update is None else update.alert,
            ),
            session_update=update,
            flush_succeeded=flush_succeeded,
        )


def _desktop_state(state: SessionState) -> DesktopState:
    return {
        SessionState.MONITORING: DesktopState.MONITORING,
        SessionState.UNKNOWN: DesktopState.UNKNOWN,
        SessionState.ERROR: DesktopState.ERROR,
        SessionState.PAUSED: DesktopState.PAUSED,
        SessionState.STOPPED: DesktopState.STOPPED,
    }.get(state, DesktopState.ERROR)


def _tray_state(
    state: DesktopState,
    *,
    quiet: bool,
    alert: AlertDecision | None = None,
) -> TrayState:
    statuses = {
        DesktopState.STOPPED: "Monitoring is stopped. The camera is off.",
        DesktopState.STARTING: "Starting local camera monitoring…",
        DesktopState.MONITORING: (
            "Monitoring locally in quiet mode." if quiet else "Monitoring locally."
        ),
        DesktopState.UNKNOWN: "Tracking is uncertain; no posture judgment is being made.",
        DesktopState.PAUSED: "Monitoring is paused. The camera is off.",
        DesktopState.ERROR: "Camera or pose tracking needs attention. The camera is off.",
        DesktopState.EXITED: "GoodPosture has exited.",
    }
    if (
        state is DesktopState.MONITORING
        and not quiet
        and alert is not None
        and alert.state is AlertState.COOLDOWN
        and alert.cooldown_until_ms is not None
    ):
        remaining_ms = max(0, alert.cooldown_until_ms - alert.timestamp_ms)
        if remaining_ms < 60_000:
            remaining_seconds = max(1, (remaining_ms + 999) // 1_000)
            statuses[state] = (
                "Monitoring locally; next reminder in about "
                f"{remaining_seconds} sec."
            )
        else:
            remaining_minutes = max(1, (remaining_ms + 59_999) // 60_000)
            statuses[state] = (
                "Monitoring locally; next reminder in about "
                f"{remaining_minutes} min."
            )
    active = state not in {DesktopState.STOPPED, DesktopState.EXITED}
    return TrayState(
        status=statuses[state],
        start_stop_label="Stop monitoring" if active else "Start monitoring",
        start_stop_enabled=state is not DesktopState.EXITED,
        pause_resume_label=(
            "Resume monitoring"
            if state is DesktopState.PAUSED
            else "Pause monitoring"
        ),
        pause_resume_enabled=state in {
            DesktopState.STARTING,
            DesktopState.MONITORING,
            DesktopState.UNKNOWN,
            DesktopState.PAUSED,
        },
        quiet_enabled=active and state is not DesktopState.ERROR,
        quiet_checked=quiet,
        recalibrate_enabled=state in {
            DesktopState.MONITORING,
            DesktopState.UNKNOWN,
            DesktopState.PAUSED,
        },
        exited=state is DesktopState.EXITED,
    )
