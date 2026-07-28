"""Failure-isolated delivery of companion decisions."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from goodposture.app import SessionEventType, SessionUpdate
from goodposture.core.alert_policy import AlertState
from goodposture.ui.companion import (
    CompanionDecision,
    CompanionPolicy,
    CompanionView,
)


class NotificationSink(Protocol):
    def show(self, *, title: str, message: str) -> None: ...


class CompanionViewSink(Protocol):
    def apply_view(self, view: CompanionView) -> None: ...


class NotificationDispatcher:
    """Isolate native notification failures from session processing."""

    def __init__(self, sink: NotificationSink) -> None:
        self._sink = sink

    def deliver(self, *, title: str, message: str) -> bool:
        try:
            self._sink.show(title=title, message=message)
        except Exception:
            return False
        return True


class CompanionPresenter:
    """Apply decisions without allowing notification delivery to interrupt UI."""

    def __init__(
        self,
        *,
        policy: CompanionPolicy,
        view_sink: CompanionViewSink,
        notifications: NotificationDispatcher | None = None,
    ) -> None:
        self._policy = policy
        self._view_sink = view_sink
        self._notifications = notifications

    def handle(self, update: SessionUpdate) -> CompanionDecision:
        decision = self._policy.update(update)
        dismiss_requested = any(
            event.type
            in {
                SessionEventType.QUIET_MODE_ENABLED,
                SessionEventType.PAUSED,
                SessionEventType.STOPPED,
                SessionEventType.CALIBRATION_STARTED,
            }
            for event in update.events
        ) or (
            update.alert is not None
            and update.alert.state is AlertState.RECOVERED
        )
        if decision.view.visible or dismiss_requested:
            self._view_sink.apply_view(decision.view)
        if decision.notification is not None and self._notifications is not None:
            delivered = self._notifications.deliver(
                title=decision.notification.title,
                message=decision.notification.message,
            )
            if not delivered and not decision.view.visible:
                fallback_view = replace(decision.view, visible=True)
                self._view_sink.apply_view(fallback_view)
                decision = CompanionDecision(
                    view=fallback_view,
                    notification=decision.notification,
                )
        return decision
