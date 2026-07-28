from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractButton, QApplication

from goodposture.adapters.sqlite_repository import LocalSettings
from goodposture.ui.daily_summary_dialog import DailySummaryDialog
from goodposture.ui.settings_dialog import SettingsDialog

_APPLICATION: QApplication | None = None


def application() -> QApplication:
    global _APPLICATION
    existing = QApplication.instance()
    _APPLICATION = existing if isinstance(existing, QApplication) else QApplication([])
    return _APPLICATION


def test_settings_controls_have_keyboard_focus_and_accessible_labels() -> None:
    _ = application()
    dialog = SettingsDialog(
        settings=LocalSettings(),
        startup_available=True,
        startup_enabled=False,
        has_saved_calibration=True,
    )

    for button in dialog.findChildren(QAbstractButton):
        assert button.text() or button.accessibleName()
        assert button.focusPolicy() != Qt.FocusPolicy.NoFocus
    assert dialog.minimumWidth() >= 500


def test_dashboard_chart_and_controls_are_named_keyboard_operable_and_resizable() -> None:
    _ = application()
    dialog = DailySummaryDialog(())

    assert dialog.chart_card.accessibleName()
    for button in (
        dialog.day_button,
        dialog.week_button,
        dialog.previous_week_button,
        dialog.next_week_button,
        dialog.info_button,
    ):
        assert button.text() or button.accessibleName()
        assert button.focusPolicy() != Qt.FocusPolicy.NoFocus
    assert dialog.minimumWidth() >= 700
    assert dialog.minimumHeight() >= 400
