from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from goodposture.adapters.runtime import PoseModelSpec
from goodposture.adapters.sqlite_repository import LocalSettings, SqliteRepository
from goodposture.app.desktop_lifecycle import DesktopState
from goodposture.app.diagnostics import DiagnosticEvent
from goodposture.core.calibration import (
    CALIBRATION_CONFIG_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    CalibrationBaseline,
    CalibrationFeature,
)
from goodposture.ui.desktop import (
    DesktopRuntime,
    _desktop_exception_hook,
    _startup_command,
    default_database_path,
    default_diagnostics_path,
)

_APPLICATION: QApplication | None = None


def application() -> QApplication:
    global _APPLICATION
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        _APPLICATION = existing
        return existing
    _APPLICATION = QApplication([])
    return _APPLICATION


def baseline(model_id: str) -> CalibrationBaseline:
    return CalibrationBaseline(
        calibration_schema_version=CALIBRATION_SCHEMA_VERSION,
        calibration_config_version=CALIBRATION_CONFIG_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_id=model_id,
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


class FakeDiagnostics:
    def __init__(self) -> None:
        self.events: list[DiagnosticEvent] = []

    def record(self, event: DiagnosticEvent) -> None:
        self.events.append(event)


class FailingController:
    state = DesktopState.MONITORING

    def exit(self, *, timestamp_ms: int) -> None:
        del timestamp_ms
        raise RuntimeError("simulated controller exit failure")


class FakeRuntime:
    def __init__(self) -> None:
        self.quit_count = 0

    def _quit_safely(self) -> None:
        self.quit_count += 1


class FakeApplication:
    def __init__(self) -> None:
        self.exit_codes: list[int] = []

    def exit(self, code: int) -> None:
        self.exit_codes.append(code)


def test_desktop_loads_compatible_local_calibration_and_releases_database(
    tmp_path: Path,
) -> None:
    app = application()
    model_id = "pose-test-v1"
    database_path = tmp_path / "goodposture.sqlite3"
    repository = SqliteRepository(database_path)
    repository.save_calibration(baseline(model_id), saved_at_utc_ms=1_000)
    repository.save_settings(
        LocalSettings(
            camera_index=2,
            companion_enabled=False,
            notifications_enabled=True,
        )
    )
    model_path = tmp_path / "pose.task"
    runtime = DesktopRuntime(
        application=app,
        model=PoseModelSpec(
            path=model_path,
            model_id=model_id,
            sha256="0" * 64,
        ),
        model_path=model_path,
        repository=repository,
    )

    assert runtime._flow.session.baseline == baseline(model_id)
    assert runtime._settings.camera_index == 2

    runtime._quit_safely()
    reopened = SqliteRepository(database_path)
    assert reopened.load_calibration() == baseline(model_id)
    reopened.close()


def test_saved_calibration_opens_dashboard_without_starting_camera(tmp_path: Path) -> None:
    app = application()
    model_id = "pose-test-v1"
    repository = SqliteRepository(tmp_path / "goodposture.sqlite3")
    repository.save_calibration(baseline(model_id), saved_at_utc_ms=1_000)
    repository.save_settings(LocalSettings(camera_index=0))
    runtime = DesktopRuntime(
        application=app,
        model=PoseModelSpec(
            path=tmp_path / "pose.task",
            model_id=model_id,
            sha256="0" * 64,
        ),
        model_path=tmp_path / "pose.task",
        repository=repository,
    )

    runtime.show_initial_surface()

    assert runtime._controller is not None
    assert runtime._controller.state is DesktopState.STOPPED
    assert runtime._tray is not None and runtime._tray.isVisible()
    assert runtime._history_dialog is not None and runtime._history_dialog.isVisible()
    assert runtime._calibration_window is None

    runtime._history_dialog.close()
    runtime.show_daily_summary()
    assert runtime._history_dialog.isVisible()

    runtime._quit_safely()


def test_missing_calibration_opens_setup(tmp_path: Path) -> None:
    runtime = DesktopRuntime(
        application=application(),
        model=PoseModelSpec(
            path=tmp_path / "pose.task",
            model_id="pose-test-v1",
            sha256="0" * 64,
        ),
        model_path=tmp_path / "pose.task",
        repository=SqliteRepository(tmp_path / "goodposture.sqlite3"),
    )

    runtime.show_initial_surface()

    assert runtime._calibration_window is not None
    assert runtime._controller is None

    runtime._quit_safely()


def test_default_database_path_uses_local_app_data_and_stable_filename() -> None:
    _ = application()

    path = default_database_path()

    assert path.name == "goodposture.sqlite3"
    assert path.parent != Path(".")


def test_frozen_startup_command_invokes_the_installed_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "GoodPosture.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))

    assert _startup_command(tmp_path / "models" / "pose.task") == (
        str(executable),
        "desktop",
        "--model",
        str((tmp_path / "models" / "pose.task").resolve()),
    )


def test_default_diagnostics_path_shares_private_local_app_directory() -> None:
    _ = application()

    diagnostics = default_diagnostics_path()

    assert diagnostics.name == "diagnostics.jsonl"
    assert diagnostics.parent == default_database_path().parent


def test_runtime_cleanup_contains_controller_failure_and_closes_database(
    tmp_path: Path,
) -> None:
    app = application()
    repository = SqliteRepository(tmp_path / "goodposture.sqlite3")
    diagnostics = FakeDiagnostics()
    runtime = DesktopRuntime(
        application=app,
        model=PoseModelSpec(
            path=tmp_path / "pose.task",
            model_id="pose-test-v1",
            sha256="0" * 64,
        ),
        model_path=tmp_path / "pose.task",
        repository=repository,
        diagnostics=diagnostics,
    )
    runtime._controller = FailingController()  # type: ignore[assignment]

    runtime._quit_safely()

    assert DiagnosticEvent.UNHANDLED_FAILURE in diagnostics.events
    assert diagnostics.events[-1] is DiagnosticEvent.APP_EXITED
    with pytest.raises(sqlite3.ProgrammingError):
        repository.load_settings()


def test_unhandled_exception_hook_logs_code_only_and_requests_clean_exit() -> None:
    runtime = FakeRuntime()
    application = FakeApplication()
    diagnostics = FakeDiagnostics()
    hook = _desktop_exception_hook(
        runtime=runtime,
        application=application,  # type: ignore[arg-type]
        diagnostics=diagnostics,
    )

    hook(RuntimeError, RuntimeError("sensitive detail"), None)

    assert diagnostics.events == [DiagnosticEvent.UNHANDLED_FAILURE]
    assert runtime.quit_count == 1
    assert application.exit_codes == [1]
