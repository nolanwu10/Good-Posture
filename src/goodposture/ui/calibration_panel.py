"""Presentation-only widgets and styling for calibration setup."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from goodposture.ui.calibration_flow import CalibrationUiState, CalibrationView


class CalibrationPanel(QWidget):
    """Build and render native controls without owning flow behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("page")
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(48, 40, 48, 40)
        page_layout.setSpacing(20)

        brand = QLabel("GOODPOSTURE")
        brand.setObjectName("brand")
        brand.setAccessibleName("GoodPosture")
        page_layout.addWidget(brand)

        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(32, 32, 32, 32)
        card_layout.setSpacing(18)

        self.title_label = QLabel()
        self.title_label.setObjectName("title")
        self.title_label.setWordWrap(True)
        card_layout.addWidget(self.title_label)

        self.body_label = QLabel()
        self.body_label.setObjectName("body")
        self.body_label.setWordWrap(True)
        self.body_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByKeyboard)
        card_layout.addWidget(self.body_label)

        self.preview_label = QLabel("Live preview starting…")
        self.preview_label.setObjectName("preview")
        self.preview_label.setAccessibleName("Live local camera preview")
        self.preview_label.setAccessibleDescription(
            "A temporary on-device camera preview used only while framing and calibrating."
        )
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(220)
        self.preview_label.setMaximumHeight(300)
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        card_layout.addWidget(self.preview_label, stretch=1)

        self.countdown_label = QLabel()
        self.countdown_label.setObjectName("countdown")
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.countdown_label.setAccessibleName("Calibration countdown")
        card_layout.addWidget(self.countdown_label)

        self.camera_combo = QComboBox()
        self.camera_combo.setAccessibleName("Available cameras")
        self.camera_combo.setMinimumHeight(40)
        card_layout.addWidget(self.camera_combo)

        self.progress_bar = QProgressBar()
        self.progress_bar.setAccessibleName("Calibration sample progress")
        self.progress_bar.setTextVisible(True)
        card_layout.addWidget(self.progress_bar)

        card_layout.addStretch()
        button_row = QHBoxLayout()
        button_row.setSpacing(12)
        self.secondary_buttons = (QPushButton(), QPushButton())
        for button in self.secondary_buttons:
            button.setObjectName("secondaryButton")
            button_row.addWidget(button)
        button_row.addStretch()
        self.primary_button = QPushButton()
        self.primary_button.setObjectName("primaryButton")
        self.primary_button.setDefault(True)
        button_row.addWidget(self.primary_button)
        card_layout.addLayout(button_row)
        page_layout.addWidget(card, stretch=1)

        footer = QLabel(
            "Local processing • Raw camera images are not stored • "
            "Comfort awareness, not medical advice"
        )
        footer.setObjectName("footer")
        footer.setWordWrap(True)
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        page_layout.addWidget(footer)
        self.setStyleSheet(_STYLE)

    def apply_view(self, view: CalibrationView) -> None:
        self.title_label.setText(view.title)
        self.body_label.setText(view.body)
        show_cameras = view.state is CalibrationUiState.CAMERA_SELECTION
        self.camera_combo.setVisible(show_cameras)
        self.camera_combo.setEnabled(bool(view.camera_indices))

        show_preview = view.state in {
            CalibrationUiState.FRAMING,
            CalibrationUiState.COUNTDOWN,
            CalibrationUiState.CALIBRATING,
        }
        self.preview_label.setVisible(show_preview)
        if show_preview:
            if self.preview_label.pixmap().isNull():
                self.preview_label.setText("Live preview starting…")
        else:
            self.preview_label.clear()

        self.countdown_label.setVisible(view.state is CalibrationUiState.COUNTDOWN)
        self.countdown_label.setText(
            "" if view.countdown_seconds is None else str(view.countdown_seconds)
        )

        self.progress_bar.setVisible(view.state is CalibrationUiState.CALIBRATING)
        self.progress_bar.setRange(0, max(1, view.required_samples))
        self.progress_bar.setValue(view.accepted_samples)
        self.progress_bar.setFormat(
            f"{view.accepted_samples} of {view.required_samples} clear samples"
        )

        self.primary_button.setVisible(view.primary_action is not None)
        if view.primary_action is not None:
            self.primary_button.setText(view.primary_action)
            self.primary_button.setAccessibleName(view.primary_action)
            self.primary_button.setEnabled(
                view.state is not CalibrationUiState.CAMERA_SELECTION
                or bool(view.camera_indices)
            )

        for index, button in enumerate(self.secondary_buttons):
            visible = index < len(view.secondary_actions)
            button.setVisible(visible)
            if visible:
                label = view.secondary_actions[index]
                button.setText(label)
                button.setAccessibleName(label)

        if self.primary_button.isVisible() and self.primary_button.isEnabled():
            self.primary_button.setFocus(Qt.FocusReason.OtherFocusReason)


_STYLE = """
QWidget#page { background: #f2f5f1; color: #18201b; font-family: "Segoe UI"; font-size: 16px; }
QLabel#brand { color: #315b43; font-size: 13px; font-weight: 700; letter-spacing: 2px; }
QFrame#card { background: #ffffff; border: 1px solid #cdd8d0; border-radius: 10px; }
QLabel#title { color: #18201b; font-size: 30px; font-weight: 650; }
QLabel#body { color: #35423a; font-size: 17px; line-height: 1.4; }
QLabel#preview {
    background: #111713; color: #dce9df; border: 1px solid #91a298;
    border-radius: 8px; font-size: 15px;
}
QLabel#countdown { color: #315b43; font-size: 64px; font-weight: 700; }
QLabel#footer { color: #55645a; font-size: 13px; }
QComboBox, QProgressBar {
    border: 1px solid #91a298; border-radius: 6px; background: #ffffff;
    color: #18201b; padding: 8px;
}
QComboBox QAbstractItemView {
    background: #ffffff; color: #18201b; border: 1px solid #91a298;
    selection-background-color: #dce9df; selection-color: #18201b;
    outline: 0;
}
QComboBox QAbstractItemView::item { min-height: 36px; padding: 6px 10px; }
QProgressBar::chunk { background: #3f7454; border-radius: 4px; }
QPushButton {
    min-height: 38px; padding: 0 18px; border-radius: 6px; font-weight: 600;
}
QPushButton#primaryButton {
    background: #315b43; color: #ffffff; border: 2px solid #315b43;
}
QPushButton#primaryButton:hover { background: #274936; }
QPushButton#secondaryButton {
    background: #ffffff; color: #274936; border: 1px solid #71847a;
}
QPushButton:focus, QComboBox:focus { border: 3px solid #174d9b; }
QPushButton:disabled {
    background: #d8ded9; color: #6a746d; border-color: #c3cbc5;
}
"""
