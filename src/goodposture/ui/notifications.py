"""Failure-reporting Qt system-notification adapter."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication, QStyle, QSystemTrayIcon


class QtNotificationSink:
    """Deliver a system notification without owning tray controls."""

    def __init__(self, tray_icon: QSystemTrayIcon | None = None) -> None:
        self._tray_icon = tray_icon
        self._owns_tray_icon = tray_icon is None

    def show(self, *, title: str, message: str) -> None:
        if not title.strip() or not message.strip():
            raise ValueError("notification title and message must be non-empty")
        if not QSystemTrayIcon.isSystemTrayAvailable():
            raise RuntimeError("system notifications are unavailable")
        application = QApplication.instance()
        if not isinstance(application, QApplication):
            raise RuntimeError("a Qt application is required for notifications")
        if self._tray_icon is None:
            icon = application.style().standardIcon(
                QStyle.StandardPixmap.SP_MessageBoxInformation
            )
            self._tray_icon = QSystemTrayIcon(icon)
            self._tray_icon.setToolTip("GoodPosture")
            self._tray_icon.show()
        self._tray_icon.showMessage(
            title,
            message,
            QSystemTrayIcon.MessageIcon.Information,
            6_000,
        )

    def close(self) -> None:
        if self._tray_icon is not None and self._owns_tray_icon:
            self._tray_icon.hide()
            self._tray_icon = None
