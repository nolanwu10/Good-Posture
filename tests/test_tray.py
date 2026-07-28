from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QSystemTrayIcon

from goodposture.app.desktop_lifecycle import TrayState
from goodposture.ui.tray import GoodPostureTrayIcon


class FakeCommands:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def toggle_monitoring(self) -> None:
        self.calls.append("monitor")

    def toggle_pause(self) -> None:
        self.calls.append("pause")

    def set_quiet_mode(self, enabled: bool) -> None:
        self.calls.append(("quiet", enabled))

    def show_status(self) -> None:
        self.calls.append("status")

    def show_detection_inspector(self) -> None:
        self.calls.append("inspector")

    def recalibrate(self) -> None:
        self.calls.append("recalibrate")

    def show_settings(self) -> None:
        self.calls.append("settings")

    def show_daily_summary(self) -> None:
        self.calls.append("summary")

    def exit(self) -> None:
        self.calls.append("exit")


def application() -> QApplication:
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


def tray_state(**overrides: object) -> TrayState:
    values: dict[str, object] = {
        "status": "Monitoring locally.",
        "start_stop_label": "Stop monitoring",
        "start_stop_enabled": True,
        "pause_resume_label": "Pause monitoring",
        "pause_resume_enabled": True,
        "quiet_enabled": True,
        "quiet_checked": False,
        "recalibrate_enabled": True,
        "exited": False,
    }
    values.update(overrides)
    return TrayState(**values)  # type: ignore[arg-type]


def test_tray_exposes_every_task_10_action() -> None:
    _ = application()
    commands = FakeCommands()
    tray = GoodPostureTrayIcon(commands=commands)

    tray.start_stop_action.trigger()
    tray.pause_resume_action.trigger()
    tray.quiet_action.trigger()
    tray.status_action.trigger()
    tray.inspector_action.trigger()
    tray.recalibrate_action.trigger()
    tray.settings_action.trigger()
    tray.summary_action.trigger()
    tray.exit_action.trigger()

    assert commands.calls == [
        "monitor",
        "pause",
        ("quiet", True),
        "status",
        "inspector",
        "recalibrate",
        "settings",
        "summary",
        "exit",
    ]


def test_tray_render_state_tracks_pause_quiet_error_and_exit() -> None:
    _ = application()
    tray = GoodPostureTrayIcon(commands=FakeCommands())

    tray.apply_state(
        tray_state(
            status="Monitoring is paused. The camera is off.",
            pause_resume_label="Resume monitoring",
            quiet_checked=True,
        )
    )
    assert tray.pause_resume_action.text() == "Resume monitoring"
    assert tray.quiet_action.isChecked()
    assert "camera is off" in tray.status_action.text()

    tray.apply_state(
        tray_state(
            status="GoodPosture has exited.",
            start_stop_enabled=False,
            pause_resume_enabled=False,
            quiet_enabled=False,
            recalibrate_enabled=False,
            exited=True,
        )
    )
    assert not tray.start_stop_action.isEnabled()
    assert not tray.settings_action.isEnabled()
    assert not tray.summary_action.isEnabled()


def test_left_click_opens_dashboard_and_quiet_mode_keeps_tray_visible() -> None:
    _ = application()
    commands = FakeCommands()
    tray = GoodPostureTrayIcon(commands=commands)
    tray.show()

    tray.apply_state(
        tray_state(
            status="Monitoring locally in quiet mode.",
            quiet_checked=True,
        )
    )
    tray.activated.emit(QSystemTrayIcon.ActivationReason.Trigger)

    assert tray.isVisible()
    assert commands.calls == ["summary"]
