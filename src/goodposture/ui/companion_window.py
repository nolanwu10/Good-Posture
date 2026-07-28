"""Non-focus-stealing PySide6 corner companion."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QCursor, QGuiApplication, QPixmap, QRegion, QScreen
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from goodposture.ui.companion import CompanionView


def companion_position(
    available_geometry: QRect,
    window_size: QSize,
    *,
    margin: int = 24,
) -> tuple[int, int]:
    """Return a bottom-right position within one screen's usable geometry."""

    if margin < 0:
        raise ValueError("margin cannot be negative")
    return (
        available_geometry.right() - window_size.width() - margin + 1,
        available_geometry.bottom() - window_size.height() - margin + 1,
    )


class CompanionWindow(QWidget):
    """A passive status surface that never accepts input focus."""

    def __init__(self) -> None:
        flags = (
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        super().__init__(None, flags)
        self.setObjectName("companion")
        self.setAccessibleName("GoodPosture status companion")
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedSize(180, 180)
        self.setMask(QRegion(self.rect(), QRegion.RegionType.Ellipse))
        card = QFrame(self)
        card.setObjectName("card")
        card.setGeometry(0, 0, 180, 180)
        self._card = card
        layout = QVBoxLayout(card)
        layout.setContentsMargins(26, 21, 26, 22)
        layout.setSpacing(3)
        self.illustration_label = QLabel()
        self.illustration_label.setObjectName("illustration")
        self.illustration_label.setAccessibleName(
            "Playful line drawing of a person adjusting their posture"
        )
        self.illustration_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.illustration_label.setFixedHeight(54)
        illustration = QPixmap(
            str(
                Path(__file__).resolve().parents[1]
                / "assets"
                / "illustrations"
                / "posture-slouch.png"
            )
        )
        if not illustration.isNull():
            self.illustration_label.setPixmap(
                illustration.scaled(
                    58,
                    58,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self.title_label = QLabel()
        self.title_label.setObjectName("title")
        self.title_label.setWordWrap(True)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setTextFormat(Qt.TextFormat.PlainText)
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred,
        )
        self.message_label = QLabel()
        self.message_label.setObjectName("message")
        self.message_label.setWordWrap(True)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.setTextFormat(Qt.TextFormat.PlainText)
        self.message_label.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred,
        )
        layout.addWidget(self.illustration_label)
        layout.addWidget(self.title_label)
        layout.addWidget(self.message_label)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(_STYLE)

    def apply_view(self, view: CompanionView) -> None:
        self.title_label.setText(view.title)
        self.message_label.setText(view.message)
        self.ensurePolished()
        for label in (self.title_label, self.message_label):
            required_height = label.heightForWidth(label.width())
            label.setMinimumHeight(
                max(label.fontMetrics().height(), required_height)
            )
        self._card.setProperty("accent", view.accent)
        style = self._card.style()
        style.unpolish(self._card)
        style.polish(self._card)
        if not view.visible:
            self.hide()
            return
        self.place_on_screen()
        self.show()

    def place_on_screen(self, screen: QScreen | None = None, *, margin: int = 24) -> None:
        target = screen or QGuiApplication.screenAt(QCursor.pos())
        if target is None:
            target = QGuiApplication.primaryScreen()
        if target is None:
            return
        x, y = companion_position(target.availableGeometry(), self.size(), margin=margin)
        self.move(QPoint(x, y))


_STYLE = """
QWidget#companion { background: transparent; font-family: "Segoe UI"; }
QFrame#card {
    background: #fffaf0;
    border: 2px solid #8a6b2d;
    border-radius: 90px;
}
QFrame#card[accent="attention"] {
    background: #fffaf0;
    border-color: #8a6b2d;
}
QLabel#title {
    color: #18201b;
    font-size: 12px;
    font-weight: 700;
    qproperty-alignment: AlignCenter;
}
QLabel#message {
    color: #35423a;
    font-size: 11px;
    qproperty-alignment: AlignCenter;
}
QLabel#illustration { background: transparent; }
"""
