from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from goodposture.adapters.sqlite_repository import LocalSettings
from goodposture.ui.settings_dialog import SettingsDialog


def application() -> QApplication:
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


def test_settings_dialog_round_trips_local_preferences() -> None:
    application()
    dialog = SettingsDialog(
        settings=LocalSettings(
            camera_index=2,
            companion_enabled=False,
            notifications_enabled=True,
        ),
        startup_available=True,
        startup_enabled=True,
        has_saved_calibration=True,
    )

    assert dialog.startup_checkbox.isChecked()
    assert not dialog.companion_checkbox.isChecked()
    assert dialog.notifications_checkbox.isChecked()

    dialog.companion_checkbox.setChecked(True)
    selected = dialog.selected_settings()

    assert selected == LocalSettings(
        camera_index=2,
        companion_enabled=True,
        notifications_enabled=True,
    )


def test_unavailable_startup_and_missing_calibration_disable_related_controls() -> None:
    application()
    dialog = SettingsDialog(
        settings=LocalSettings(),
        startup_available=False,
        startup_enabled=False,
        has_saved_calibration=False,
    )

    assert not dialog.startup_checkbox.isEnabled()
    assert not dialog.delete_calibration_button.isEnabled()
    assert "on this device" in dialog.privacy_label.text().lower()


def test_delete_saved_calibration_is_an_explicit_signal() -> None:
    application()
    dialog = SettingsDialog(
        settings=LocalSettings(),
        startup_available=True,
        startup_enabled=False,
        has_saved_calibration=True,
    )
    requests: list[bool] = []
    dialog.delete_calibration_requested.connect(lambda: requests.append(True))

    dialog.delete_calibration_button.click()

    assert requests == [True]
