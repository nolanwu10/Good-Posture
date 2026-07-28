"""Launch and coordinate the consent-to-tray local desktop vertical slice."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from types import TracebackType

from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QApplication, QMessageBox

from goodposture.adapters.local_diagnostics import LocalDiagnosticLog
from goodposture.adapters.runtime import (
    AdapterResult,
    AdapterState,
    FramePreview,
    PoseAdapter,
    PoseModelSpec,
)
from goodposture.adapters.sqlite_repository import (
    SqliteAggregateSink,
    SqliteRepository,
)
from goodposture.app import (
    AnalysisSession,
    DesktopLifecycle,
    DesktopState,
    LifecycleResult,
)
from goodposture.app.diagnostics import DiagnosticEvent, DiagnosticSink
from goodposture.core.calibration import CalibrationBaseline
from goodposture.ui.calibration_flow import CalibrationFlow
from goodposture.ui.calibration_window import CalibrationWindow
from goodposture.ui.camera_worker import CameraBackend
from goodposture.ui.companion import CompanionPolicy, CompanionPreferences
from goodposture.ui.companion_delivery import (
    CompanionPresenter,
    NotificationDispatcher,
)
from goodposture.ui.companion_window import CompanionWindow
from goodposture.ui.daily_summary_dialog import DailySummaryDialog
from goodposture.ui.detection_inspector import DetectionInspectorWindow
from goodposture.ui.notifications import QtNotificationSink
from goodposture.ui.settings_dialog import SettingsDialog
from goodposture.ui.tray import GoodPostureTrayIcon
from goodposture.ui.windows_startup import WindowsStartupSetting

_MODEL_SHA256 = "59929E1D1EE95287735DDD833B19CF4AC46D29BC7AFDDBBF6753C459690D574A"
_MODEL_ID = "pose_landmarker_lite@sha256:59929e1d"


def _startup_command(model_path: Path) -> tuple[str, ...]:
    """Build a launch command that works for source and frozen installs."""

    model_argument = str(model_path.resolve())
    if getattr(sys, "frozen", False):
        return (sys.executable, "desktop", "--model", model_argument)
    return (sys.executable, "-m", "goodposture", "desktop", "--model", model_argument)


class DesktopRuntime:
    """Own all Qt adapters after calibration while core policy stays headless."""

    def __init__(
        self,
        *,
        application: QApplication,
        model: PoseModelSpec,
        model_path: Path,
        repository: SqliteRepository,
        diagnostics: DiagnosticSink | None = None,
    ) -> None:
        self._application = application
        self._model = model
        self._model_path = model_path
        self._repository = repository
        self._diagnostics = diagnostics
        self._settings = repository.load_settings()
        try:
            saved_baseline = repository.load_compatible_calibration(
                model_id=model.model_id
            )
        except ValueError:
            repository.delete_calibration()
            saved_baseline = None
        self._backend = CameraBackend(PoseAdapter(model=model))
        self._flow = CalibrationFlow(
            session=AnalysisSession(
                model_id=model.model_id,
                baseline=saved_baseline,
            )
        )
        self._calibration_window: CalibrationWindow | None = None
        self._controller: DesktopLifecycle | None = None
        self._aggregate_sink = SqliteAggregateSink(repository)
        self._tray: GoodPostureTrayIcon | None = None
        self._companion = CompanionWindow()
        self._inspector: DetectionInspectorWindow | None = None
        self._history_dialog: DailySummaryDialog | None = None
        self._notification_sink: QtNotificationSink | None = None
        self._presenter = self._new_presenter()
        self._last_timestamp_ms = 0
        self._exiting = False
        self._startup = WindowsStartupSetting(_startup_command(model_path))
        self._backend.preview_ready.connect(self._handle_preview)

    def show_setup(self) -> None:
        self._calibration_window = CalibrationWindow(
            flow=self._flow,
            backend=self._backend,
            on_complete=self._complete_calibration,
            on_baseline_deleted=self._delete_saved_baseline,
        )
        self._calibration_window.show()

    def show_initial_surface(self) -> None:
        """Open the dashboard when a compatible local calibration is available."""

        baseline = self._flow.session.baseline
        camera_index = self._settings.camera_index
        if baseline is None or not isinstance(camera_index, int) or camera_index < 0:
            self.show_setup()
            return
        self._flow.session.start(timestamp_ms=self._timestamp_ms())
        self._initialize_controller(camera_index=camera_index)
        self._apply(self._require_controller().current())
        self.show_daily_summary()

    def _initialize_controller(self, *, camera_index: int) -> None:
        if self._controller is not None:
            return
        self._controller = DesktopLifecycle(
            session=self._flow.session,
            camera=self._backend,
            camera_index=camera_index,
            session_factory=self._new_session,
            aggregate_sink=self._aggregate_sink,
            diagnostics=self._diagnostics,
        )
        self._tray = GoodPostureTrayIcon(commands=self)
        self._tray.show()
        self._presenter = self._new_presenter()
        self._application.setQuitOnLastWindowClosed(False)
        self._application.aboutToQuit.connect(self._quit_safely)
        self._backend.result_ready.connect(self._handle_adapter_result)

    def _timestamp_ms(self) -> int:
        now = time.monotonic_ns() // 1_000_000
        self._last_timestamp_ms = max(now, self._last_timestamp_ms + 1)
        return self._last_timestamp_ms

    def _complete_calibration(self, window: CalibrationWindow) -> None:
        selected_camera = self._flow.selected_camera_index
        if selected_camera is None:
            raise RuntimeError("calibration completed without a selected camera")
        baseline = self._flow.session.baseline
        if baseline is None:
            raise RuntimeError("calibration completed without a baseline")
        self._settings = replace(self._settings, camera_index=selected_camera)
        try:
            self._repository.save_calibration(
                baseline,
                saved_at_utc_ms=time.time_ns() // 1_000_000,
            )
            self._repository.save_settings(self._settings)
        except Exception:
            QMessageBox.warning(
                None,
                "Local data could not be saved",
                "Monitoring can continue, but calibration or settings may need "
                "to be entered again after restart.",
            )
        self._last_timestamp_ms = max(
            self._last_timestamp_ms,
            self._flow.session.last_timestamp_ms or 0,
        )
        if self._controller is None:
            self._initialize_controller(camera_index=selected_camera)
            result = self._require_controller().start(timestamp_ms=self._timestamp_ms())
        else:
            self._backend.result_ready.connect(self._handle_adapter_result)
            result = self._require_controller().resume(timestamp_ms=self._timestamp_ms())
        self._apply(result)
        window.hide()
        self.show_daily_summary()

    def _new_session(self, baseline: CalibrationBaseline) -> AnalysisSession:
        return AnalysisSession(model_id=self._model.model_id, baseline=baseline)

    def _new_presenter(self) -> CompanionPresenter:
        notifications: NotificationDispatcher | None = None
        self._notification_sink = None
        if self._settings.notifications_enabled and self._tray is not None:
            self._notification_sink = QtNotificationSink(self._tray)
            notifications = NotificationDispatcher(self._notification_sink)
        return CompanionPresenter(
            policy=CompanionPolicy(
                CompanionPreferences(
                    companion_enabled=self._settings.companion_enabled,
                    notifications_enabled=self._settings.notifications_enabled,
                )
            ),
            view_sink=self._companion,
            notifications=notifications,
        )

    def _handle_adapter_result(self, value: object) -> None:
        if not isinstance(value, AdapterResult) or self._controller is None:
            return
        if value.observation is not None:
            self._last_timestamp_ms = max(
                self._last_timestamp_ms,
                value.observation.timestamp_ms,
            )
        if value.observation is not None:
            result = self._controller.process_observation(value.observation)
        elif value.failure is not None:
            result = self._controller.process_failure(
                timestamp_ms=self._timestamp_ms(),
                failure=value.failure,
            )
        elif value.state in {AdapterState.READY, AdapterState.RECOVERED}:
            result = self._controller.camera_ready()
        else:
            result = self._controller.camera_error()
        if (
            value.observation is not None
            and result.session_update is not None
            and self._inspector is not None
            and self._inspector.isVisible()
        ):
            baseline = self._controller.session.baseline
            if baseline is not None:
                self._inspector.update_analysis(
                    observation=value.observation,
                    update=result.session_update,
                    baseline=baseline,
                    scoring_config=self._controller.session.scoring_config,
                    alert_config=self._controller.session.alert_config,
                )
        self._apply(result)

    def _handle_preview(self, value: object) -> None:
        try:
            if (
                isinstance(value, FramePreview)
                and self._inspector is not None
                and self._inspector.isVisible()
            ):
                self._inspector.update_preview(value)
        finally:
            self._backend.preview_consumed()

    def _apply(self, result: LifecycleResult) -> None:
        if self._controller is not None:
            self._last_timestamp_ms = max(
                self._last_timestamp_ms,
                self._controller.session.last_timestamp_ms or 0,
            )
        if self._tray is not None:
            self._tray.apply_state(result.tray)
        if result.session_update is not None:
            self._presenter.handle(result.session_update)

    def toggle_monitoring(self) -> None:
        controller = self._require_controller()
        if controller.state is DesktopState.STOPPED:
            result = controller.start(timestamp_ms=self._timestamp_ms())
        else:
            result = controller.stop(timestamp_ms=self._timestamp_ms())
        self._apply(result)

    def toggle_pause(self) -> None:
        controller = self._require_controller()
        if controller.state is DesktopState.PAUSED:
            result = controller.resume(timestamp_ms=self._timestamp_ms())
        else:
            result = controller.pause(timestamp_ms=self._timestamp_ms())
        self._apply(result)

    def set_quiet_mode(self, enabled: bool) -> None:
        controller = self._require_controller()
        if controller.session.quiet_mode == enabled:
            return
        self._apply(
            controller.set_quiet_mode(
                enabled=enabled,
                timestamp_ms=self._timestamp_ms(),
            )
        )

    def show_status(self) -> None:
        controller = self._require_controller()
        QMessageBox.information(None, "GoodPosture status", controller.tray.status)

    def show_detection_inspector(self) -> None:
        self._require_controller()
        if self._inspector is None:
            self._inspector = DetectionInspectorWindow(
                on_closed=self._close_detection_inspector
            )
        self._backend.set_preview_enabled(True)
        self._inspector.show()
        self._inspector.raise_()
        self._inspector.activateWindow()

    def _close_detection_inspector(self) -> None:
        self._backend.set_preview_enabled(False)

    def recalibrate(self) -> None:
        controller = self._require_controller()
        if self._inspector is not None:
            self._inspector.close()
        if self._history_dialog is not None:
            self._history_dialog.hide()
        self._apply(
            controller.request_recalibration(timestamp_ms=self._timestamp_ms())
        )
        self._backend.result_ready.disconnect(self._handle_adapter_result)
        self._flow.recalibrate(timestamp_ms=self._timestamp_ms())
        self._calibration_window = CalibrationWindow(
            flow=self._flow,
            backend=self._backend,
            on_complete=self._complete_calibration,
            on_baseline_deleted=self._delete_saved_baseline,
        )
        self._calibration_window.show()
        selected = self._flow.selected_camera_index
        if selected is not None:
            self._backend.start_camera(selected)

    def show_settings(self) -> None:
        startup_enabled = False
        with suppress(OSError):
            startup_enabled = self._startup.is_enabled()
        try:
            has_saved_calibration = self._repository.load_calibration() is not None
        except Exception:
            has_saved_calibration = False
        dialog = SettingsDialog(
            settings=self._settings,
            startup_available=self._startup.available,
            startup_enabled=startup_enabled,
            has_saved_calibration=has_saved_calibration,
        )
        dialog.delete_calibration_requested.connect(
            lambda: self._confirm_delete_saved_calibration(dialog)
        )
        if dialog.exec() == SettingsDialog.DialogCode.Accepted:
            try:
                self._startup.set_enabled(dialog.startup_checkbox.isChecked())
            except OSError:
                QMessageBox.warning(
                    None,
                    "Startup setting unavailable",
                    "Windows could not update the current-user startup setting.",
                )
            self._settings = dialog.selected_settings()
            try:
                self._repository.save_settings(self._settings)
            except Exception:
                QMessageBox.warning(
                    None,
                    "Settings could not be saved",
                    "The current choices will not be restored after restart.",
                )
            self._presenter = self._new_presenter()

    def show_daily_summary(self) -> None:
        controller = self._require_controller()
        if controller.session.aggregates.started_at_ms is not None:
            try:
                self._aggregate_sink.flush(controller.session.aggregates)
            except Exception:
                QMessageBox.warning(
                    None,
                    "History temporarily unavailable",
                    "The current session could not be checkpointed locally.",
                )
        try:
            summaries = self._repository.recent_daily_summaries(limit=30)
        except Exception:
            QMessageBox.warning(
                None,
                "History temporarily unavailable",
                "Saved local summaries could not be read.",
            )
            return
        if self._history_dialog is None:
            self._history_dialog = DailySummaryDialog(summaries)
            self._history_dialog.delete_history_requested.connect(
                self._confirm_delete_history
            )
            self._history_dialog.open_inspector_requested.connect(
                self.show_detection_inspector
            )
            self._history_dialog.recalibrate_requested.connect(self.recalibrate)
        else:
            self._history_dialog.set_summaries(summaries)
        self._history_dialog.show()
        self._history_dialog.raise_()
        self._history_dialog.activateWindow()

    def _confirm_delete_history(self) -> None:
        answer = QMessageBox.question(
            self._history_dialog,
            "Delete local history?",
            "This permanently deletes saved session and daily totals. Monitoring "
            "will stop first so the deleted session does not reappear.",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        controller = self._require_controller()
        if controller.state is not DesktopState.STOPPED:
            self._apply(controller.stop(timestamp_ms=self._timestamp_ms()))
        try:
            self._repository.delete_history()
        except Exception:
            QMessageBox.warning(
                self._history_dialog,
                "History could not be deleted",
                "Saved local history is still present.",
            )
            return
        if self._history_dialog is not None:
            self._history_dialog.set_summaries(())

    def _confirm_delete_saved_calibration(self, dialog: SettingsDialog) -> None:
        answer = QMessageBox.question(
            dialog,
            "Delete saved calibration?",
            "This removes the derived baseline from local storage. The current "
            "session may continue with its in-memory baseline, but the next launch "
            "will ask you to calibrate again.",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._delete_saved_baseline()
        dialog.delete_calibration_button.setEnabled(False)

    def _delete_saved_baseline(self) -> None:
        try:
            self._repository.delete_calibration()
        except Exception:
            QMessageBox.warning(
                None,
                "Calibration could not be deleted",
                "The saved local calibration is still present.",
            )

    def exit(self) -> None:
        self._quit_safely()
        self._application.quit()

    def _quit_safely(self) -> None:
        if self._exiting:
            return
        self._exiting = True
        try:
            if (
                self._controller is not None
                and self._controller.state is not DesktopState.EXITED
            ):
                self._apply(self._controller.exit(timestamp_ms=self._timestamp_ms()))
            else:
                self._backend.shutdown()
        except Exception:
            self._record(DiagnosticEvent.UNHANDLED_FAILURE)
        finally:
            with suppress(Exception):
                self._backend.shutdown()
            for cleanup in (
                self._tray.hide if self._tray is not None else None,
                self._inspector.close if self._inspector is not None else None,
                (
                    self._history_dialog.close
                    if self._history_dialog is not None
                    else None
                ),
                self._companion.hide,
            ):
                if cleanup is None:
                    continue
                with suppress(Exception):
                    cleanup()
            try:
                self._repository.close()
            except Exception:
                self._record(DiagnosticEvent.LOCAL_DATA_FAILURE)
            self._record(DiagnosticEvent.APP_EXITED)

    def _record(self, event: DiagnosticEvent) -> None:
        if self._diagnostics is None:
            return
        with suppress(Exception):
            self._diagnostics.record(event)

    def _require_controller(self) -> DesktopLifecycle:
        if self._controller is None:
            raise RuntimeError("desktop monitoring is not initialized")
        return self._controller


def run_desktop(model_path: Path) -> int:
    """Launch setup, then keep GoodPosture available from the system tray."""

    application = QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QApplication(sys.argv)
    assert isinstance(application, QApplication)
    application.setApplicationName("GoodPosture")
    application.setOrganizationName("GoodPosture")
    diagnostics = LocalDiagnosticLog(default_diagnostics_path())
    with suppress(Exception):
        diagnostics.record(DiagnosticEvent.APP_STARTED)
    model = PoseModelSpec(path=model_path, model_id=_MODEL_ID, sha256=_MODEL_SHA256)
    try:
        repository = SqliteRepository(default_database_path())
    except Exception:
        with suppress(Exception):
            diagnostics.record(DiagnosticEvent.LOCAL_DATA_FAILURE)
        QMessageBox.critical(
            None,
            "GoodPosture local data unavailable",
            "GoodPosture could not open its private local database.",
        )
        return 2
    runtime = DesktopRuntime(
        application=application,
        model=model,
        model_path=model_path,
        repository=repository,
        diagnostics=diagnostics,
    )
    runtime.show_initial_surface()
    if owns_application:
        previous_hook = sys.excepthook
        sys.excepthook = _desktop_exception_hook(
            runtime=runtime,
            application=application,
            diagnostics=diagnostics,
        )
        try:
            return application.exec()
        finally:
            sys.excepthook = previous_hook
            runtime._quit_safely()
    return 0


def _desktop_exception_hook(
    *,
    runtime: DesktopRuntime,
    application: QApplication,
    diagnostics: DiagnosticSink,
) -> Callable[[type[BaseException], BaseException, TracebackType | None], None]:
    """Contain an unhandled Qt callback failure without logging its payload."""

    def handle(
        exception_type: type[BaseException],
        exception: BaseException,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        with suppress(Exception):
            diagnostics.record(DiagnosticEvent.UNHANDLED_FAILURE)
        with suppress(Exception):
            runtime._quit_safely()
        with suppress(Exception):
            application.exit(1)

    return handle


def default_database_path() -> Path:
    """Return the Qt-managed per-user local application data path."""

    location = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppLocalDataLocation
    )
    if not location:
        raise RuntimeError("a local application data directory is unavailable")
    return Path(location) / "goodposture.sqlite3"


def default_diagnostics_path() -> Path:
    """Return the bounded log path beside other private local app data."""

    return default_database_path().with_name("diagnostics.jsonl")
