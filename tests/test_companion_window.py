from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from goodposture.ui.companion import CompanionMode, CompanionView
from goodposture.ui.companion_window import CompanionWindow, companion_position


def application() -> QApplication:
    instance = QApplication.instance()
    if isinstance(instance, QApplication):
        return instance
    return QApplication([])


def test_corner_position_respects_each_screen_available_geometry() -> None:
    assert companion_position(
        QRect(1920, 0, 1280, 1024),
        QSize(320, 120),
        margin=24,
    ) == (2856, 880)


def test_window_never_accepts_focus_and_renders_textual_state() -> None:
    application()
    window = CompanionWindow()

    assert window.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
    assert window.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert window.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert window.focusPolicy() is Qt.FocusPolicy.NoFocus

    window.apply_view(
        CompanionView(
            mode=CompanionMode.UNKNOWN,
            title="Tracking uncertain",
            message="No posture judgment is being made.",
            visible=True,
            accent="quiet",
        )
    )

    assert window.title_label.text() == "Tracking uncertain"
    assert window.message_label.text() == "No posture judgment is being made."
    window.close()


def test_prompt_bubble_is_compact_circular_and_persists_until_hidden() -> None:
    application()
    window = CompanionWindow()

    window.apply_view(
        CompanionView(
            mode=CompanionMode.PROMPT,
            title="Posture check",
            message="Shift when ready",
            visible=True,
            accent="attention",
        )
    )

    assert window.isVisible()
    assert 150 <= window.width() <= 180
    assert 150 <= window.height() <= 180
    assert window.width() == window.height()
    pixmap = window.illustration_label.pixmap()
    assert not pixmap.isNull()
    assert window.illustration_label.accessibleName()
    for label in (window.title_label, window.message_label):
        assert label.heightForWidth(label.width()) <= label.height()
        assert label.geometry().left() >= 24
        assert label.geometry().right() <= window.width() - 24

    QTest.qWait(30)

    assert window.isVisible()

    window.apply_view(
        CompanionView(
            mode=CompanionMode.HIDDEN,
            title="",
            message="",
            visible=False,
            accent="quiet",
        )
    )

    assert not window.isVisible()
