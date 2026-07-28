"""Deterministic session-event presentation for the corner companion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from goodposture.app import SessionEventType, SessionUpdate


class CompanionMode(StrEnum):
    """User-visible companion modes."""

    MONITORING = "monitoring"
    NEEDS_ADJUSTMENT = "needs_adjustment"
    UNKNOWN = "unknown"
    PAUSED = "paused"
    PROMPT = "prompt"
    ERROR = "error"
    HIDDEN = "hidden"


@dataclass(frozen=True, slots=True)
class CompanionPreferences:
    """Independent presentation choices with quiet defaults."""

    companion_enabled: bool = True
    notifications_enabled: bool = False


@dataclass(frozen=True, slots=True)
class CompanionView:
    """Small render model containing no scores, observations, or landmarks."""

    mode: CompanionMode
    title: str
    message: str
    visible: bool
    accent: str


@dataclass(frozen=True, slots=True)
class NotificationRequest:
    """One sanitized system-notification request."""

    title: str
    message: str


@dataclass(frozen=True, slots=True)
class CompanionDecision:
    """One bounded presentation decision."""

    view: CompanionView
    notification: NotificationRequest | None


class CompanionPolicy:
    """Map typed session updates to gentle companion and notification output."""

    def __init__(
        self,
        preferences: CompanionPreferences | None = None,
    ) -> None:
        self._preferences = preferences or CompanionPreferences()
        self._last_notified_prompt_timestamp_ms: int | None = None

    def update(self, update: SessionUpdate) -> CompanionDecision:
        prompt = next(
            (
                event
                for event in update.events
                if event.type is SessionEventType.PROMPT
            ),
            None,
        )
        if prompt is not None:
            view = CompanionView(
                mode=CompanionMode.PROMPT,
                title="Posture check",
                message="Shift when ready",
                visible=self._preferences.companion_enabled,
                accent="attention",
            )
            should_notify = (
                self._preferences.notifications_enabled
                and prompt.timestamp_ms != self._last_notified_prompt_timestamp_ms
            )
            if should_notify:
                self._last_notified_prompt_timestamp_ms = prompt.timestamp_ms
            return CompanionDecision(
                view=view,
                notification=(
                    NotificationRequest(
                        title="GoodPosture comfort check",
                        message=prompt.message,
                    )
                    if should_notify
                    else None
                ),
            )

        return CompanionDecision(
            view=CompanionView(
                mode=CompanionMode.HIDDEN,
                title="",
                message="",
                visible=False,
                accent="quiet",
            ),
            notification=None,
        )
