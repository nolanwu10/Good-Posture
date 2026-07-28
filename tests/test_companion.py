from __future__ import annotations

from dataclasses import dataclass

from goodposture.app import (
    PostureAssessment,
    SessionAggregates,
    SessionEvent,
    SessionEventType,
    SessionState,
    SessionUpdate,
)
from goodposture.core.alert_policy import AlertDecision, AlertState
from goodposture.ui.companion import (
    CompanionMode,
    CompanionPolicy,
    CompanionPreferences,
    CompanionView,
)
from goodposture.ui.companion_delivery import CompanionPresenter, NotificationDispatcher


def update(
    state: SessionState,
    *,
    event: SessionEvent | None = None,
    alert: AlertDecision | None = None,
    posture_assessment: PostureAssessment = PostureAssessment.TRACKING_UNCERTAIN,
) -> SessionUpdate:
    return SessionUpdate(
        state=state,
        events=() if event is None else (event,),
        score=None,
        alert=alert,
        aggregates=SessionAggregates(
            started_at_ms=0,
            ended_at_ms=None,
            eligible_monitoring_ms=0,
            unknown_ms=0,
            deviation_ms=0,
            prompt_count=0,
        ),
        posture_assessment=posture_assessment,
    )


def alert(timestamp_ms: int, state: AlertState) -> AlertDecision:
    return AlertDecision(
        timestamp_ms=timestamp_ms,
        state=state,
        event=None,
        prompt_message=None,
        pending_since_ms=None,
        cooldown_until_ms=None,
        config_version=6,
        continuous_deviation_ms=0,
        posture_debt_ms=0.0,
        trigger=None,
    )


def test_normal_background_states_never_show_a_persistent_companion() -> None:
    policy = CompanionPolicy()

    for state in (
        SessionState.MONITORING,
        SessionState.UNKNOWN,
        SessionState.PAUSED,
        SessionState.ERROR,
    ):
        decision = policy.update(update(state))
        assert decision.view.mode is CompanionMode.HIDDEN
        assert decision.view.visible is False


def test_needs_adjustment_stays_hidden_until_sustained_policy_emits_prompt() -> None:
    policy = CompanionPolicy(
        CompanionPreferences(companion_enabled=True, notifications_enabled=True)
    )

    decision = policy.update(
        update(
            SessionState.MONITORING,
            posture_assessment=PostureAssessment.NEEDS_ADJUSTMENT,
        )
    )

    assert decision.view.mode is CompanionMode.HIDDEN
    assert decision.view.visible is False
    assert decision.notification is None


def test_prompt_uses_supportive_session_copy_and_requests_one_notification() -> None:
    prompt = SessionEvent(
        timestamp_ms=30_000,
        type=SessionEventType.PROMPT,
        message="A quick comfort check: would a small position change feel good?",
    )
    policy = CompanionPolicy(
        CompanionPreferences(companion_enabled=True, notifications_enabled=True)
    )

    decision = policy.update(update(SessionState.MONITORING, event=prompt))
    duplicate = policy.update(update(SessionState.MONITORING, event=prompt))

    assert decision.view.mode is CompanionMode.PROMPT
    assert decision.view.title == "Posture check"
    assert decision.view.message == "Shift when ready"
    assert decision.notification is not None
    assert decision.notification.message == prompt.message
    assert duplicate.notification is None


def test_notifications_and_companion_are_independently_configurable() -> None:
    prompt = SessionEvent(
        timestamp_ms=1,
        type=SessionEventType.PROMPT,
        message="Would a brief movement break feel comfortable?",
    )
    policy = CompanionPolicy(
        CompanionPreferences(companion_enabled=False, notifications_enabled=True)
    )

    decision = policy.update(update(SessionState.MONITORING, event=prompt))

    assert decision.view.visible is False
    assert decision.notification is not None


def test_non_monitoring_lifecycle_states_hide_companion() -> None:
    policy = CompanionPolicy()

    for state in (
        SessionState.CREATED,
        SessionState.NEEDS_CALIBRATION,
        SessionState.CALIBRATING,
        SessionState.STOPPED,
    ):
        assert policy.update(update(state)).view.visible is False


@dataclass
class FakeNotificationSink:
    should_fail: bool = False
    calls: int = 0

    def show(self, *, title: str, message: str) -> None:
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("native notification detail")


def test_notification_failure_is_isolated_from_monitoring() -> None:
    sink = FakeNotificationSink(should_fail=True)
    dispatcher = NotificationDispatcher(sink)

    delivered = dispatcher.deliver(
        title="GoodPosture comfort check",
        message="Would a small movement feel good?",
    )

    assert delivered is False
    assert sink.calls == 1


@dataclass
class FakeViewSink:
    current: CompanionView | None = None

    def apply_view(self, view: CompanionView) -> None:
        self.current = view


def test_presenter_updates_companion_even_when_notification_fails() -> None:
    prompt = SessionEvent(
        timestamp_ms=5_000,
        type=SessionEventType.PROMPT,
        message="Would a brief movement break feel comfortable?",
    )
    view_sink = FakeViewSink()
    presenter = CompanionPresenter(
        policy=CompanionPolicy(
            CompanionPreferences(companion_enabled=True, notifications_enabled=True)
        ),
        view_sink=view_sink,
        notifications=NotificationDispatcher(FakeNotificationSink(should_fail=True)),
    )

    decision = presenter.handle(update(SessionState.MONITORING, event=prompt))

    assert decision.view.mode is CompanionMode.PROMPT
    assert view_sink.current == decision.view


def test_notification_only_mode_falls_back_to_companion_on_delivery_failure() -> None:
    prompt = SessionEvent(
        timestamp_ms=6_000,
        type=SessionEventType.PROMPT,
        message="Would a small movement feel good?",
    )
    view_sink = FakeViewSink()
    presenter = CompanionPresenter(
        policy=CompanionPolicy(
            CompanionPreferences(companion_enabled=False, notifications_enabled=True)
        ),
        view_sink=view_sink,
        notifications=NotificationDispatcher(FakeNotificationSink(should_fail=True)),
    )

    decision = presenter.handle(update(SessionState.MONITORING, event=prompt))

    assert decision.view.visible is True
    assert view_sink.current is not None
    assert view_sink.current.visible is True


def test_prompt_bubble_is_not_cleared_by_next_frame_but_quiet_mode_dismisses_it() -> None:
    prompt = SessionEvent(
        timestamp_ms=6_000,
        type=SessionEventType.PROMPT,
        message="Would a small movement feel good?",
    )
    quiet = SessionEvent(
        timestamp_ms=6_100,
        type=SessionEventType.QUIET_MODE_ENABLED,
        message="Quiet mode enabled.",
    )
    view_sink = FakeViewSink()
    presenter = CompanionPresenter(
        policy=CompanionPolicy(),
        view_sink=view_sink,
    )

    presenter.handle(update(SessionState.MONITORING, event=prompt))
    presenter.handle(update(SessionState.MONITORING))

    assert view_sink.current is not None
    assert view_sink.current.mode is CompanionMode.PROMPT

    presenter.handle(update(SessionState.MONITORING, event=quiet))

    hidden_view = view_sink.current
    assert hidden_view is not None
    assert hidden_view.mode is CompanionMode.HIDDEN
    assert hidden_view.visible is False


def test_prompt_persists_through_tracking_uncertainty_and_hides_on_valid_recovery() -> None:
    prompt = SessionEvent(
        timestamp_ms=6_000,
        type=SessionEventType.PROMPT,
        message="Would a small movement feel good?",
    )
    view_sink = FakeViewSink()
    presenter = CompanionPresenter(policy=CompanionPolicy(), view_sink=view_sink)

    presenter.handle(update(SessionState.MONITORING, event=prompt))
    presenter.handle(
        update(
            SessionState.UNKNOWN,
            alert=alert(7_000, AlertState.UNKNOWN),
        )
    )

    assert view_sink.current is not None
    assert view_sink.current.mode is CompanionMode.PROMPT

    presenter.handle(
        update(
            SessionState.MONITORING,
            alert=alert(12_000, AlertState.RECOVERED),
            posture_assessment=PostureAssessment.GOOD,
        )
    )

    assert view_sink.current is not None
    assert view_sink.current.mode is CompanionMode.HIDDEN
    assert view_sink.current.visible is False


def test_pause_and_stop_dismiss_a_persistent_prompt() -> None:
    prompt = SessionEvent(
        timestamp_ms=6_000,
        type=SessionEventType.PROMPT,
        message="Would a small movement feel good?",
    )
    view_sink = FakeViewSink()
    presenter = CompanionPresenter(policy=CompanionPolicy(), view_sink=view_sink)
    presenter.handle(update(SessionState.MONITORING, event=prompt))

    for timestamp_ms, event_type, state in (
        (7_000, SessionEventType.PAUSED, SessionState.PAUSED),
        (8_000, SessionEventType.STOPPED, SessionState.STOPPED),
    ):
        presenter.handle(
            update(
                state,
                event=SessionEvent(
                    timestamp_ms=timestamp_ms,
                    type=event_type,
                    message=event_type.value,
                ),
            )
        )
        assert view_sink.current is not None
        assert view_sink.current.visible is False
