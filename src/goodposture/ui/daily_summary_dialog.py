"""Compact interactive dashboard for privacy-bounded posture outcomes."""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    Qt,
    Signal,
)
from PySide6.QtGui import QCloseEvent, QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from goodposture.adapters.sqlite_repository import DailySummary

_DAILY_GOAL_MINIMUM_MS = 30 * 60_000
_COMPARISON_MINIMUM_MS = 30 * 60_000
_ABOUT_SAME_PERCENTAGE_POINTS = 3.0


@dataclass(frozen=True, slots=True)
class _PeriodTotals:
    upright_ms: int = 0
    slouch_ms: int = 0
    reminders: int = 0
    recovered_ms: int = 0
    morning_slouch_ms: int = 0
    afternoon_slouch_ms: int = 0
    evening_slouch_ms: int = 0
    late_night_slouch_ms: int = 0

    @property
    def monitored_ms(self) -> int:
        return self.upright_ms + self.slouch_ms


class _DayBarButton(QAbstractButton):
    """Keyboard-operable daily bar with visible selection and exact a11y text."""

    def __init__(
        self,
        *,
        local_day: date,
        totals: _PeriodTotals,
        maximum_ms: int,
    ) -> None:
        super().__init__()
        self.local_day = local_day
        self.totals = totals
        self._maximum_ms = max(1, maximum_ms)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFixedHeight(190)
        self.setMinimumWidth(74)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName(
            f"{local_day.strftime('%A, %B')} {local_day.day}: "
            f"upright {_format_duration(totals.upright_ms)}, "
            f"slouch time {_format_duration(totals.slouch_ms)}, "
            f"{totals.reminders} reminders"
        )

    def paintEvent(self, event: object) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.isChecked() or self.underMouse():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#e9f1ec" if self.isChecked() else "#f2f6f3"))
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(2, 2, -2, -2), 10, 10)

        plot_bottom = self.height() - 43
        maximum_height = 118
        total_height = round(
            maximum_height * min(self.totals.monitored_ms, self._maximum_ms)
            / self._maximum_ms
        )
        total_height = max(5 if self.totals.monitored_ms else 0, total_height)
        bar = QRectF(
            (self.width() - 34) / 2,
            plot_bottom - total_height,
            34,
            total_height,
        )
        if self.totals.monitored_ms:
            slouch_height = (
                bar.height() * self.totals.slouch_ms / self.totals.monitored_ms
            )
            upright_rect = QRectF(
                bar.left(),
                bar.top(),
                bar.width(),
                bar.height() - slouch_height,
            )
            slouch_rect = QRectF(
                bar.left(),
                upright_rect.bottom(),
                bar.width(),
                slouch_height,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#5d9b78"))
            painter.drawRoundedRect(upright_rect, 6, 6)
            if slouch_rect.height() > 0:
                painter.setBrush(QColor("#e2a52e"))
                painter.drawRoundedRect(slouch_rect, 6, 6)
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#dce4df"))
            painter.drawRoundedRect(
                QRectF((self.width() - 34) / 2, plot_bottom - 5, 34, 5),
                3,
                3,
            )

        font = QFont(self.font())
        font.setPointSize(9)
        font.setBold(self.isChecked())
        painter.setFont(font)
        painter.setPen(QColor("#213f32"))
        painter.drawText(
            QRectF(0, self.height() - 36, self.width(), 18),
            Qt.AlignmentFlag.AlignCenter,
            self.local_day.strftime("%a"),
        )
        painter.setPen(QColor("#52665c"))
        painter.drawText(
            QRectF(0, self.height() - 20, self.width(), 16),
            Qt.AlignmentFlag.AlignCenter,
            str(self.local_day.day),
        )
        if self.hasFocus():
            painter.setPen(QPen(QColor("#245b43"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(
                QRectF(self.rect()).adjusted(3, 3, -3, -3),
                9,
                9,
            )


class DailySummaryDialog(QDialog):
    """Show daily and weekly posture outcomes with compact direct manipulation."""

    delete_history_requested = Signal()
    open_inspector_requested = Signal()
    recalibrate_requested = Signal()

    def __init__(
        self,
        summaries: tuple[DailySummary, ...],
        *,
        reduced_motion: bool | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("GoodPosture dashboard")
        self.setAccessibleName("GoodPosture interactive day and week dashboard")
        self.setObjectName("dashboard")
        self.setMinimumSize(800, 620)
        self.resize(920, 700)
        self.setStyleSheet(_STYLE)
        self._reduced_motion = (
            _system_reduced_motion() if reduced_motion is None else reduced_motion
        )
        self._selection_animation: QPropertyAnimation | None = None
        self._summaries: tuple[DailySummary, ...] = ()
        self._summaries_by_day: dict[date, DailySummary] = {}
        self._week_start = date.today() - timedelta(days=date.today().weekday())
        self._selected_day = date.today()
        self._mode = "day"
        self._bar_group = QButtonGroup(self)
        self._bar_group.setExclusive(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(22, 18, 22, 14)
        outer.setSpacing(12)

        header = QHBoxLayout()
        heading = QLabel("Your posture")
        heading.setObjectName("pageTitle")
        header.addWidget(heading)
        header.addStretch(1)
        self.info_button = QToolButton()
        self.info_button.setText("i")
        self.info_button.setObjectName("infoButton")
        self.info_button.setCheckable(True)
        self.info_button.setAccessibleName("Show metric definitions")
        self.info_button.setToolTip("How these metrics are calculated")
        self.info_button.toggled.connect(self._toggle_info)
        header.addWidget(self.info_button)
        outer.addLayout(header)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.day_button = QPushButton("Day")
        self.week_button = QPushButton("Week")
        for button in (self.day_button, self.week_button):
            button.setCheckable(True)
            button.setObjectName("modeButton")
            controls.addWidget(button)
        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)
        mode_group.addButton(self.day_button)
        mode_group.addButton(self.week_button)
        self.day_button.setChecked(True)
        self.day_button.clicked.connect(lambda: self._set_mode("day"))
        self.week_button.clicked.connect(lambda: self._set_mode("week"))
        controls.addStretch(1)
        self.previous_week_button = QToolButton()
        self.previous_week_button.setText("‹")
        self.previous_week_button.setAccessibleName("Previous week")
        self.previous_week_button.clicked.connect(lambda: self._navigate_week(-1))
        controls.addWidget(self.previous_week_button)
        self.period_label = QLabel()
        self.period_label.setObjectName("periodHeading")
        self.period_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.period_label.setMinimumWidth(150)
        controls.addWidget(self.period_label)
        self.next_week_button = QToolButton()
        self.next_week_button.setText("›")
        self.next_week_button.setAccessibleName("Next week")
        self.next_week_button.clicked.connect(lambda: self._navigate_week(1))
        controls.addWidget(self.next_week_button)
        outer.addLayout(controls)

        self.info_panel = QFrame()
        self.info_panel.setObjectName("infoPanel")
        info_layout = QVBoxLayout(self.info_panel)
        info_layout.setContentsMargins(12, 9, 12, 9)
        self.info_label = QLabel(
            "Streak goal: at least 30 minutes recorded and more upright than "
            "slouch time on consecutive recorded days. Last-week comparison uses "
            "slouch share and calls changes under 3 percentage points about the "
            "same. Recovered time is observed upright time within five minutes "
            "after a reminder and confirmed recovery; it is an estimate, not proof "
            "of time that would otherwise have been spent slouching. Slouch timing "
            "uses only coarse local dayparts."
        )
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        self.info_panel.hide()
        outer.addWidget(self.info_panel)

        insights = QGridLayout()
        insights.setHorizontalSpacing(10)
        self.recovered_value_label = self._insight_card(
            insights, 0, "Recovered after reminders", prominent=True
        )
        self.streak_value_label = self._insight_card(insights, 1, "Posture streak")
        self.comparison_value_label = self._insight_card(
            insights, 2, "Compared with last week"
        )
        self.common_time_value_label = self._insight_card(
            insights, 3, "Common slouch time", secondary=True
        )
        insights.setColumnStretch(0, 2)
        for column in range(1, 4):
            insights.setColumnStretch(column, 1)
        outer.addLayout(insights)

        self.chart_card = QFrame()
        self.chart_card.setObjectName("chartCard")
        self.chart_card.setAccessibleName("Selectable daily upright and slouch bars")
        chart_layout = QVBoxLayout(self.chart_card)
        chart_layout.setContentsMargins(14, 10, 14, 10)
        chart_layout.setSpacing(4)
        legend = QHBoxLayout()
        legend.addWidget(_legend("Upright time", "#5d9b78"))
        legend.addWidget(_legend("Slouch time", "#e2a52e"))
        legend.addStretch(1)
        chart_layout.addLayout(legend)
        self.bar_layout = QHBoxLayout()
        self.bar_layout.setSpacing(5)
        chart_layout.addLayout(self.bar_layout)
        outer.addWidget(self.chart_card, 1)

        self.empty_label = QLabel("No daily history yet")
        self.empty_label.setObjectName("emptyState")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self.empty_label)

        self.detail_card = QFrame()
        self.detail_card.setObjectName("detailCard")
        detail_layout = QHBoxLayout(self.detail_card)
        detail_layout.setContentsMargins(16, 10, 16, 10)
        self.detail_illustration = QLabel()
        self.detail_illustration.setAccessibleName(
            "Line drawing of a person sitting upright"
        )
        self.detail_illustration.setFixedSize(76, 76)
        pixmap = QPixmap(str(_illustration_path("posture-upright.png")))
        if not pixmap.isNull():
            self.detail_illustration.setPixmap(
                pixmap.scaled(
                    74,
                    74,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        detail_layout.addWidget(self.detail_illustration)
        self.detail_title_label = QLabel()
        self.detail_title_label.setObjectName("detailTitle")
        self.detail_title_label.setMinimumWidth(150)
        detail_layout.addWidget(self.detail_title_label)
        detail_layout.addStretch(1)
        self.upright_value_label = _detail_metric(detail_layout, "Upright time")
        self.slouch_value_label = _detail_metric(detail_layout, "Slouch time")
        self.reminders_value_label = _detail_metric(detail_layout, "Reminders")
        outer.addWidget(self.detail_card)
        self._detail_effect = QGraphicsOpacityEffect(self.detail_card)
        self.detail_card.setGraphicsEffect(self._detail_effect)
        self._detail_effect.setOpacity(1.0)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.open_inspector_button = QPushButton("Developer inspector...")
        self.open_inspector_button.setObjectName("secondaryButton")
        self.open_inspector_button.setAccessibleName(
            "Open developer detection inspector"
        )
        self.open_inspector_button.clicked.connect(self.open_inspector_requested)
        buttons.addButton(
            self.open_inspector_button,
            QDialogButtonBox.ButtonRole.ActionRole,
        )
        self.recalibrate_button = QPushButton("Recalibrate...")
        self.recalibrate_button.setObjectName("secondaryButton")
        self.recalibrate_button.setAccessibleName("Recalibrate your local baseline")
        self.recalibrate_button.clicked.connect(self.recalibrate_requested)
        buttons.addButton(
            self.recalibrate_button,
            QDialogButtonBox.ButtonRole.ActionRole,
        )
        self.delete_history_button = QPushButton("Delete local history...")
        self.delete_history_button.setObjectName("destructiveButton")
        self.delete_history_button.setAccessibleName("Delete saved local history")
        self.delete_history_button.clicked.connect(self.delete_history_requested)
        buttons.addButton(
            self.delete_history_button,
            QDialogButtonBox.ButtonRole.DestructiveRole,
        )
        outer.addWidget(buttons)
        self.set_summaries(summaries)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Keep the app available from the system tray when the dashboard closes."""

        self.hide()
        event.ignore()

    @staticmethod
    def _insight_card(
        layout: QGridLayout,
        column: int,
        title: str,
        *,
        prominent: bool = False,
        secondary: bool = False,
    ) -> QLabel:
        card = QFrame()
        card.setObjectName(
            "prominentInsight" if prominent else "secondaryInsight" if secondary else "insight"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 9, 12, 9)
        card_layout.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("insightTitle")
        value = QLabel("—")
        value.setObjectName("insightValue")
        value.setWordWrap(True)
        card_layout.addWidget(title_label)
        card_layout.addWidget(value)
        layout.addWidget(card, 0, column)
        return value

    def set_summaries(self, summaries: tuple[DailySummary, ...]) -> None:
        self._summaries = tuple(
            sorted(summaries, key=lambda item: item.local_day, reverse=True)
        )
        self._summaries_by_day = {
            date.fromisoformat(summary.local_day): summary for summary in summaries
        }
        if self._summaries:
            self._selected_day = date.fromisoformat(self._summaries[0].local_day)
            self._week_start = self._selected_day - timedelta(
                days=self._selected_day.weekday()
            )
        has_history = bool(self._summaries)
        self.chart_card.setVisible(has_history)
        self.detail_card.setVisible(has_history)
        self.empty_label.setVisible(not has_history)
        self.delete_history_button.setEnabled(has_history)
        self._refresh()

    def _toggle_info(self, visible: bool) -> None:
        self.info_panel.setVisible(visible)
        self.info_button.setAccessibleName(
            "Hide metric definitions" if visible else "Show metric definitions"
        )

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self._refresh_detail(animate=True)

    def _navigate_week(self, direction: int) -> None:
        self._week_start += timedelta(days=7 * direction)
        week_days = self._week_days()
        summaries_in_week = [day for day in week_days if day in self._summaries_by_day]
        self._selected_day = (
            max(summaries_in_week) if summaries_in_week else week_days[0]
        )
        self._refresh()

    def _select_day(self, selected_day: date) -> None:
        self._selected_day = selected_day
        self._mode = "day"
        self.day_button.setChecked(True)
        self._refresh_bar_selection()
        self._refresh_detail(animate=True)

    def _week_days(self) -> tuple[date, ...]:
        return tuple(self._week_start + timedelta(days=offset) for offset in range(7))

    def _refresh(self) -> None:
        week_end = self._week_start + timedelta(days=6)
        self.period_label.setText(_format_period(self._week_start, week_end))
        latest_week = (
            self._week_start
            if not self._summaries
            else date.fromisoformat(self._summaries[0].local_day)
            - timedelta(days=date.fromisoformat(self._summaries[0].local_day).weekday())
        )
        self.next_week_button.setEnabled(self._week_start < latest_week)
        _clear_layout(self.bar_layout)
        week_days = self._week_days()
        totals_by_day = {
            day: _totals((self._summaries_by_day[day],))
            if day in self._summaries_by_day
            else _PeriodTotals()
            for day in week_days
        }
        maximum_ms = max(
            (totals.monitored_ms for totals in totals_by_day.values()),
            default=1,
        )
        self._bar_group = QButtonGroup(self)
        self._bar_group.setExclusive(True)
        for day in week_days:
            button = _DayBarButton(
                local_day=day,
                totals=totals_by_day[day],
                maximum_ms=maximum_ms,
            )
            button.clicked.connect(
                lambda checked=False, selected_day=day: self._select_day(selected_day)
            )
            self._bar_group.addButton(button)
            self.bar_layout.addWidget(button)
        self._refresh_bar_selection()
        self._refresh_insights()
        self._refresh_detail(animate=False)

    def _refresh_bar_selection(self) -> None:
        for button in self._bar_group.buttons():
            if isinstance(button, _DayBarButton):
                button.setChecked(
                    self._mode == "day" and button.local_day == self._selected_day
                )

    def _refresh_insights(self) -> None:
        latest = self._summaries[:7]
        recent_totals = _totals(latest)
        self.recovered_value_label.setText(
            _format_duration(recent_totals.recovered_ms)
            if recent_totals.reminders >= 2 and recent_totals.recovered_ms > 0
            else "Not enough data"
        )
        streak = _posture_streak(self._summaries)
        self.streak_value_label.setText(
            f"{streak} day{'s' if streak != 1 else ''}"
        )
        self.comparison_value_label.setText(_week_comparison(self._summaries))
        self.common_time_value_label.setText(_common_slouch_daypart(recent_totals))
        self.recovered_value_label.setAccessibleName(
            f"Recovered after reminders estimate: {self.recovered_value_label.text()}"
        )

    def _refresh_detail(self, *, animate: bool) -> None:
        self._refresh_bar_selection()
        if self._mode == "week":
            summaries = tuple(
                self._summaries_by_day[day]
                for day in self._week_days()
                if day in self._summaries_by_day
            )
            totals = _totals(summaries)
            self.detail_title_label.setText("This week")
        else:
            summary = self._summaries_by_day.get(self._selected_day)
            totals = _totals(()) if summary is None else _totals((summary,))
            self.detail_title_label.setText(
                f"{self._selected_day.strftime('%A, %b')} {self._selected_day.day}"
            )
        self.upright_value_label.setText(_format_duration(totals.upright_ms))
        self.slouch_value_label.setText(_format_duration(totals.slouch_ms))
        self.reminders_value_label.setText(str(totals.reminders))
        self.detail_card.setAccessibleName(
            f"{self.detail_title_label.text()}: upright time "
            f"{self.upright_value_label.text()}, slouch time "
            f"{self.slouch_value_label.text()}, reminders "
            f"{self.reminders_value_label.text()}"
        )
        if animate:
            self._animate_detail()

    def _animate_detail(self) -> None:
        if self._reduced_motion:
            self._selection_animation = None
            self._detail_effect.setOpacity(1.0)
            return
        animation = QPropertyAnimation(self._detail_effect, b"opacity", self)
        animation.setDuration(140)
        animation.setStartValue(0.72)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._selection_animation = animation
        animation.start()


def _detail_metric(layout: QHBoxLayout, title: str) -> QLabel:
    group = QVBoxLayout()
    group.setSpacing(1)
    label = QLabel(title)
    label.setObjectName("detailMetricTitle")
    value = QLabel("—")
    value.setObjectName("detailMetricValue")
    value.setAlignment(Qt.AlignmentFlag.AlignRight)
    group.addWidget(label)
    group.addWidget(value)
    layout.addLayout(group)
    return value


def _legend(text: str, color: str) -> QWidget:
    item = QWidget()
    layout = QHBoxLayout(item)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)
    swatch = QFrame()
    swatch.setFixedSize(11, 11)
    swatch.setStyleSheet(f"background: {color}; border-radius: 5px;")
    layout.addWidget(swatch)
    label = QLabel(text)
    label.setObjectName("legendText")
    layout.addWidget(label)
    return item


def _totals(summaries: tuple[DailySummary, ...]) -> _PeriodTotals:
    return _PeriodTotals(
        upright_ms=sum(
            max(0, summary.eligible_monitoring_ms - summary.deviation_ms)
            for summary in summaries
        ),
        slouch_ms=sum(summary.deviation_ms for summary in summaries),
        reminders=sum(summary.prompt_count for summary in summaries),
        recovered_ms=sum(summary.post_prompt_recovered_ms for summary in summaries),
        morning_slouch_ms=sum(summary.morning_deviation_ms for summary in summaries),
        afternoon_slouch_ms=sum(
            summary.afternoon_deviation_ms for summary in summaries
        ),
        evening_slouch_ms=sum(summary.evening_deviation_ms for summary in summaries),
        late_night_slouch_ms=sum(
            summary.late_night_deviation_ms for summary in summaries
        ),
    )


def _daily_goal_met(summary: DailySummary) -> bool:
    upright_ms = max(0, summary.eligible_monitoring_ms - summary.deviation_ms)
    return (
        summary.eligible_monitoring_ms >= _DAILY_GOAL_MINIMUM_MS
        and upright_ms > summary.deviation_ms
    )


def _posture_streak(summaries: tuple[DailySummary, ...]) -> int:
    if not summaries:
        return 0
    by_day = {date.fromisoformat(item.local_day): item for item in summaries}
    current = max(by_day)
    streak = 0
    while current in by_day and _daily_goal_met(by_day[current]):
        streak += 1
        current -= timedelta(days=1)
    return streak


def _week_comparison(summaries: tuple[DailySummary, ...]) -> str:
    if not summaries:
        return "Not enough data"
    by_day = {date.fromisoformat(item.local_day): item for item in summaries}
    latest = max(by_day)
    current_start = latest - timedelta(days=latest.weekday())
    current = _totals(
        tuple(
            summary
            for day, summary in by_day.items()
            if current_start <= day <= current_start + timedelta(days=6)
        )
    )
    previous_start = current_start - timedelta(days=7)
    previous = _totals(
        tuple(
            summary
            for day, summary in by_day.items()
            if previous_start <= day <= previous_start + timedelta(days=6)
        )
    )
    if (
        current.monitored_ms < _COMPARISON_MINIMUM_MS
        or previous.monitored_ms < _COMPARISON_MINIMUM_MS
    ):
        return "Not enough data"
    current_share = 100 * current.slouch_ms / current.monitored_ms
    previous_share = 100 * previous.slouch_ms / previous.monitored_ms
    delta = current_share - previous_share
    if abs(delta) < _ABOUT_SAME_PERCENTAGE_POINTS:
        return f"About the same · {abs(delta):.0f} pp"
    if delta < 0:
        return f"Better · {abs(delta):.0f} pp less"
    return f"Worse · {delta:.0f} pp more"


def _common_slouch_daypart(totals: _PeriodTotals) -> str:
    values = (
        ("Morning", totals.morning_slouch_ms),
        ("Afternoon", totals.afternoon_slouch_ms),
        ("Evening", totals.evening_slouch_ms),
        ("Late night", totals.late_night_slouch_ms),
    )
    label, duration_ms = max(values, key=lambda item: item[1])
    return label if duration_ms > 0 else "Not enough data"


def _format_period(start: date, end: date) -> str:
    if start.month == end.month:
        return f"{start.strftime('%b')} {start.day}–{end.day}"
    return f"{start.strftime('%b')} {start.day}–{end.strftime('%b')} {end.day}"


def _format_duration(duration_ms: int) -> str:
    total_seconds = max(0, duration_ms) // 1_000
    hours, remaining = divmod(total_seconds, 3_600)
    minutes, seconds = divmod(remaining, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m"
    return f"{seconds}s"


def _clear_layout(layout: QHBoxLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item is None:
            continue
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


def _illustration_path(filename: str) -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "illustrations" / filename


def _system_reduced_motion() -> bool:
    if sys.platform != "win32":
        return True
    animations_enabled = ctypes.c_int()
    try:
        succeeded = ctypes.windll.user32.SystemParametersInfoW(
            0x1042,
            0,
            ctypes.byref(animations_enabled),
            0,
        )
    except (AttributeError, OSError):
        return True
    return not bool(succeeded and animations_enabled.value)


_STYLE = """
QDialog#dashboard {
    background: #f4f6f2;
    color: #18251f;
    font-family: "Segoe UI";
    font-size: 13px;
}
QLabel#pageTitle { color: #173c2d; font-size: 26px; font-weight: 700; }
QToolButton {
    min-width: 30px;
    min-height: 30px;
    color: #294c3d;
    background: #ffffff;
    border: 1px solid #c6d2ca;
    border-radius: 8px;
    font-size: 17px;
    font-weight: 700;
}
QToolButton:hover, QToolButton:focus { background: #eaf1ed; border-color: #6e9180; }
QToolButton#infoButton { border-radius: 15px; font-size: 13px; }
QToolButton#infoButton:checked { background: #dcebe3; }
QPushButton {
    min-height: 30px;
    padding: 3px 12px;
    color: #203c30;
    background: #ffffff;
    border: 1px solid #aebdb4;
    border-radius: 7px;
}
QPushButton:hover { background: #eef4f0; border-color: #789486; }
QPushButton:focus { border: 2px solid #356f56; }
QPushButton#modeButton:checked {
    color: #ffffff;
    background: #356f56;
    border-color: #356f56;
}
QLabel#periodHeading { color: #29483a; font-size: 14px; font-weight: 700; }
QFrame#infoPanel {
    color: #40544a;
    background: #edf1ee;
    border: 1px solid #d4ddd7;
    border-radius: 8px;
}
QFrame#prominentInsight, QFrame#insight, QFrame#secondaryInsight {
    background: #ffffff;
    border: 1px solid #d5ded8;
    border-radius: 10px;
}
QFrame#prominentInsight { background: #e7f1e9; border-color: #bcd3c3; }
QFrame#secondaryInsight { background: #f7f9f7; }
QLabel#insightTitle { color: #52635b; font-size: 10px; font-weight: 650; }
QLabel#insightValue { color: #173c2d; font-size: 16px; font-weight: 700; }
QFrame#prominentInsight QLabel#insightValue { font-size: 20px; }
QFrame#secondaryInsight QLabel#insightValue { color: #475b51; font-size: 14px; }
QFrame#chartCard {
    background: #ffffff;
    border: 1px solid #d5ded8;
    border-radius: 12px;
}
QLabel#legendText { color: #344a40; font-size: 11px; font-weight: 600; }
QLabel#emptyState {
    color: #52635b;
    background: #ffffff;
    border: 1px dashed #b8c7be;
    border-radius: 10px;
    padding: 48px;
}
QFrame#detailCard {
    background: #ffffff;
    border: 1px solid #d5ded8;
    border-radius: 12px;
}
QLabel#detailTitle { color: #173c2d; font-size: 17px; font-weight: 700; }
QLabel#detailMetricTitle { color: #52635b; font-size: 10px; }
QLabel#detailMetricValue { color: #173c2d; font-size: 18px; font-weight: 700; }
QPushButton#secondaryButton { color: #294c3d; }
QPushButton#destructiveButton { color: #7a302c; }
"""
