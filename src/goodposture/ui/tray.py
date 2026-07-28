"""PySide6 system-tray controls backed by deterministic render state."""

from __future__ import annotations

from typing import Protocol

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QSystemTrayIcon

from goodposture.app.desktop_lifecycle import TrayState


class TrayCommands(Protocol):
    """User-intent callbacks; timestamps and lifecycle rules stay outside Qt."""

    def toggle_monitoring(self) -> None: ...

    def toggle_pause(self) -> None: ...

    def set_quiet_mode(self, enabled: bool) -> None: ...

    def show_status(self) -> None: ...

    def show_detection_inspector(self) -> None: ...

    def recalibrate(self) -> None: ...

    def show_settings(self) -> None: ...

    def show_daily_summary(self) -> None: ...

    def exit(self) -> None: ...


class GoodPostureTrayIcon(QSystemTrayIcon):
    """Expose the complete Task 10 action set without owning app policy."""

    def __init__(self, *, commands: TrayCommands, icon: QIcon | None = None) -> None:
        application = QApplication.instance()
        if not isinstance(application, QApplication):
            raise RuntimeError("a QApplication is required")
        tray_icon = icon or application.style().standardIcon(
            QStyle.StandardPixmap.SP_ComputerIcon
        )
        super().__init__(tray_icon)
        self._commands = commands
        self.setToolTip("GoodPosture — camera off")
        menu = QMenu()
        self.status_action = menu.addAction("Status: camera off")
        self.status_action.triggered.connect(commands.show_status)
        self.inspector_action = menu.addAction("Detection inspector (developer)…")
        self.inspector_action.triggered.connect(commands.show_detection_inspector)
        menu.addSeparator()
        self.start_stop_action = menu.addAction("Start monitoring")
        self.start_stop_action.triggered.connect(commands.toggle_monitoring)
        self.pause_resume_action = menu.addAction("Pause monitoring")
        self.pause_resume_action.triggered.connect(commands.toggle_pause)
        self.quiet_action = menu.addAction("Quiet mode")
        self.quiet_action.setCheckable(True)
        self.quiet_action.toggled.connect(commands.set_quiet_mode)
        self.recalibrate_action = menu.addAction("Recalibrate…")
        self.recalibrate_action.triggered.connect(commands.recalibrate)
        menu.addSeparator()
        self.settings_action = menu.addAction("Settings…")
        self.settings_action.triggered.connect(commands.show_settings)
        self.summary_action = menu.addAction("Open dashboard")
        self.summary_action.triggered.connect(commands.show_daily_summary)
        menu.addSeparator()
        self.exit_action = menu.addAction("Exit GoodPosture")
        self.exit_action.triggered.connect(commands.exit)
        self.setContextMenu(menu)
        self.activated.connect(self._handle_activation)

    def _handle_activation(
        self,
        reason: QSystemTrayIcon.ActivationReason,
    ) -> None:
        if reason is QSystemTrayIcon.ActivationReason.Trigger:
            self._commands.show_daily_summary()

    def apply_state(self, state: TrayState) -> None:
        """Atomically synchronize every state-dependent tray affordance."""

        self.status_action.setText(f"Status: {state.status}")
        self.setToolTip(f"GoodPosture — {state.status}")
        self.start_stop_action.setText(state.start_stop_label)
        self.start_stop_action.setEnabled(state.start_stop_enabled)
        self.pause_resume_action.setText(state.pause_resume_label)
        self.pause_resume_action.setEnabled(state.pause_resume_enabled)
        self.quiet_action.blockSignals(True)
        self.quiet_action.setChecked(state.quiet_checked)
        self.quiet_action.blockSignals(False)
        self.quiet_action.setEnabled(state.quiet_enabled)
        self.recalibrate_action.setEnabled(state.recalibrate_enabled)
        self.settings_action.setEnabled(not state.exited)
        self.summary_action.setEnabled(not state.exited)
        self.inspector_action.setEnabled(not state.exited)
