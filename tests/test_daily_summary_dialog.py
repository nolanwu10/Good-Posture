from __future__ import annotations

import os
from datetime import date, timedelta

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QScrollArea

from goodposture.adapters.sqlite_repository import DailySummary
from goodposture.ui.daily_summary_dialog import (
    DailySummaryDialog,
    _common_slouch_daypart,
    _DayBarButton,
    _posture_streak,
    _totals,
    _week_comparison,
)


def application() -> QApplication:
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


def summary(
    local_day: str,
    *,
    eligible_ms: int = 3_600_000,
    deviation_ms: int = 600_000,
    reminders: int = 1,
    recovered_ms: int = 0,
    morning_ms: int = 0,
    afternoon_ms: int = 0,
    evening_ms: int = 0,
    late_night_ms: int = 0,
) -> DailySummary:
    return DailySummary(
        local_day=local_day,
        session_count=1,
        eligible_monitoring_ms=eligible_ms,
        unknown_ms=0,
        deviation_ms=deviation_ms,
        prompt_count=reminders,
        post_prompt_recovered_ms=recovered_ms,
        morning_deviation_ms=morning_ms,
        afternoon_deviation_ms=afternoon_ms,
        evening_deviation_ms=evening_ms,
        late_night_deviation_ms=late_night_ms,
    )


def recent_week() -> tuple[DailySummary, ...]:
    return tuple(
        summary(
            (date(2026, 7, 20) + timedelta(days=offset)).isoformat(),
            reminders=offset % 3,
            recovered_ms=120_000 if offset % 3 else 0,
            afternoon_ms=300_000,
        )
        for offset in range(7)
    )


def test_dashboard_is_compact_interactive_and_not_a_scrolling_report() -> None:
    application()
    dialog = DailySummaryDialog(recent_week(), reduced_motion=True)

    assert dialog.windowTitle() == "GoodPosture dashboard"
    assert dialog.findChildren(QScrollArea) == []
    assert dialog.day_button.isChecked()
    assert dialog.chart_card.accessibleName()
    assert len(dialog.findChildren(_DayBarButton)) == 7
    visible_copy = " ".join(
        label.text() for label in dialog.findChildren(QLabel) if label.isVisibleTo(dialog)
    ).lower()
    assert "confidently monitored" not in visible_copy
    assert "tracking coverage" not in visible_copy


def test_clicking_or_keyboard_selecting_a_day_reveals_exact_details() -> None:
    application()
    dialog = DailySummaryDialog(recent_week(), reduced_motion=True)
    buttons = dialog.findChildren(_DayBarButton)
    target = next(button for button in buttons if button.local_day == date(2026, 7, 22))

    target.click()

    assert target.isChecked()
    assert dialog.day_button.isChecked()
    assert "Wednesday" in dialog.detail_title_label.text()
    assert dialog.upright_value_label.text() == "50m"
    assert dialog.slouch_value_label.text() == "10m"
    assert dialog.reminders_value_label.text() == "2"

    target.setFocus()
    QTest.keyClick(target, Qt.Key.Key_Space)
    assert target.isChecked()


def test_week_toggle_and_navigation_update_the_selected_period() -> None:
    application()
    summaries = recent_week() + (
        summary("2026-07-19"),
    )
    dialog = DailySummaryDialog(summaries, reduced_motion=True)

    dialog.week_button.click()

    assert dialog.week_button.isChecked()
    assert dialog.detail_title_label.text() == "This week"
    assert dialog.upright_value_label.text() == "5h 50m"

    latest_period = dialog.period_label.text()
    dialog.previous_week_button.click()
    assert dialog.period_label.text() != latest_period
    assert dialog.next_week_button.isEnabled()


def test_info_affordance_hides_methodology_until_requested() -> None:
    application()
    dialog = DailySummaryDialog(recent_week(), reduced_motion=True)

    assert dialog.info_panel.isHidden()
    dialog.info_button.click()

    assert dialog.info_panel.isVisibleTo(dialog)
    methodology = dialog.info_label.text().lower()
    assert "30 minutes" in methodology
    assert "five minutes after a reminder" in methodology
    assert "estimate" in methodology
    assert "coarse local dayparts" in methodology


def test_streak_comparison_daypart_and_recovered_estimate_are_defensible() -> None:
    current_start = date(2026, 7, 20)
    current = tuple(
        summary(
            (current_start + timedelta(days=offset)).isoformat(),
            deviation_ms=360_000,
            reminders=1,
            recovered_ms=90_000,
            evening_ms=360_000,
        )
        for offset in range(7)
    )
    previous = tuple(
        summary(
            (current_start - timedelta(days=7) + timedelta(days=offset)).isoformat(),
            deviation_ms=900_000,
        )
        for offset in range(7)
    )
    summaries = current + previous

    assert _posture_streak(summaries) == 14
    assert _week_comparison(summaries).startswith("Better")
    assert "pp less" in _week_comparison(summaries)
    assert _common_slouch_daypart(_totals(current)) == "Evening"

    application()
    dialog = DailySummaryDialog(summaries, reduced_motion=True)
    assert dialog.recovered_value_label.text() == "10m"
    assert dialog.streak_value_label.text() == "14 days"
    assert dialog.common_time_value_label.text() == "Evening"


def test_recovered_estimate_waits_for_enough_reminder_outcomes() -> None:
    application()
    dialog = DailySummaryDialog(
        (
            summary(
                "2026-07-26",
                reminders=1,
                recovered_ms=120_000,
            ),
        ),
        reduced_motion=True,
    )

    assert dialog.recovered_value_label.text() == "Not enough data"


def test_selection_animation_is_short_and_disabled_for_reduced_motion() -> None:
    application()
    reduced = DailySummaryDialog(recent_week(), reduced_motion=True)
    reduced.week_button.click()
    assert reduced._selection_animation is None

    animated = DailySummaryDialog(recent_week(), reduced_motion=False)
    animated.week_button.click()
    assert animated._selection_animation is not None
    assert animated._selection_animation.duration() == 140


def test_empty_history_and_explicit_actions_remain_accessible() -> None:
    application()
    dialog = DailySummaryDialog((), reduced_motion=True)
    delete_calls: list[bool] = []
    inspector_calls: list[bool] = []
    dialog.delete_history_requested.connect(lambda: delete_calls.append(True))
    dialog.open_inspector_requested.connect(lambda: inspector_calls.append(True))

    assert dialog.empty_label.isVisibleTo(dialog)
    assert dialog.chart_card.isHidden()
    assert not dialog.delete_history_button.isEnabled()

    dialog.open_inspector_button.click()
    dialog.set_summaries(recent_week())
    dialog.delete_history_button.click()

    assert inspector_calls == [True]
    assert delete_calls == [True]


def test_dashboard_close_hides_it_and_recalibrate_is_explicit() -> None:
    application()
    dialog = DailySummaryDialog(recent_week(), reduced_motion=True)
    recalibrate_calls: list[bool] = []
    dialog.recalibrate_requested.connect(lambda: recalibrate_calls.append(True))
    dialog.show()

    dialog.close()
    dialog.show()
    dialog.recalibrate_button.click()

    assert dialog.isVisible()
    assert dialog.recalibrate_button.accessibleName()
    assert recalibrate_calls == [True]
