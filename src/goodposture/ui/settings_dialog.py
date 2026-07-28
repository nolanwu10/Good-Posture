"""Accessible local desktop settings dialog."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from goodposture.adapters.sqlite_repository import LocalSettings


class SettingsDialog(QDialog):
    """Edit local preferences and expose explicit calibration deletion."""

    delete_calibration_requested = Signal()

    def __init__(
        self,
        *,
        settings: LocalSettings,
        startup_available: bool,
        startup_enabled: bool,
        has_saved_calibration: bool,
    ) -> None:
        super().__init__()
        self._camera_index = settings.camera_index
        self.setWindowTitle("GoodPosture settings")
        self.setAccessibleName("GoodPosture local settings")
        self.setModal(True)
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        explanation = QLabel(
            "GoodPosture starts monitoring only when you choose to start it. "
            "Windows startup never turns on the camera automatically."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self.startup_checkbox = QCheckBox("Start GoodPosture with Windows")
        self.startup_checkbox.setChecked(startup_enabled)
        self.startup_checkbox.setEnabled(startup_available)
        layout.addWidget(self.startup_checkbox)

        self.companion_checkbox = QCheckBox("Show brief posture reminder bubbles")
        self.companion_checkbox.setChecked(settings.companion_enabled)
        layout.addWidget(self.companion_checkbox)

        self.notifications_checkbox = QCheckBox(
            "Allow Windows notifications for comfort reminders"
        )
        self.notifications_checkbox.setChecked(settings.notifications_enabled)
        layout.addWidget(self.notifications_checkbox)

        self.privacy_label = QLabel(
            "Preferences and derived calibration stay on this device. "
            "Camera images and raw landmark history are not saved."
        )
        self.privacy_label.setWordWrap(True)
        layout.addWidget(self.privacy_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.delete_calibration_button = QPushButton("Delete saved calibration…")
        self.delete_calibration_button.setAccessibleName("Delete saved calibration")
        self.delete_calibration_button.setEnabled(has_saved_calibration)
        self.delete_calibration_button.clicked.connect(
            self.delete_calibration_requested
        )
        buttons.addButton(
            self.delete_calibration_button,
            QDialogButtonBox.ButtonRole.DestructiveRole,
        )
        layout.addWidget(buttons)

    def selected_settings(self) -> LocalSettings:
        return LocalSettings(
            camera_index=self._camera_index,
            companion_enabled=self.companion_checkbox.isChecked(),
            notifications_enabled=self.notifications_checkbox.isChecked(),
        )
