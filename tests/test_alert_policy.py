from __future__ import annotations

from dataclasses import replace

import pytest

from goodposture.core.alert_policy import (
    ALERT_POLICY_CONFIG_VERSION,
    DEFAULT_ALERT_POLICY_CONFIG,
    AlertEvent,
    AlertPolicy,
    AlertPolicyConfig,
    AlertState,
    AlertTrigger,
)
from goodposture.core.scoring import SCORING_CONFIG_VERSION, ScoreResult, ScoreState


def config() -> AlertPolicyConfig:
    return AlertPolicyConfig(
        version=ALERT_POLICY_CONFIG_VERSION,
        deviation_threshold=60.0,
        recovery_threshold=40.0,
        continuous_deviation_duration_ms=1_000,
        posture_debt_limit_ms=2_000,
        recovery_debt_decay_rate=0.5,
        cooldown_duration_ms=5_000,
        maximum_observation_gap_ms=1_000,
    )


def available_score(timestamp_ms: int, score: float) -> ScoreResult:
    return ScoreResult(
        timestamp_ms=timestamp_ms,
        state=ScoreState.AVAILABLE,
        raw_score=score,
        smoothed_score=score,
        confidence=0.95,
        feature_deviations=(),
        config_version=SCORING_CONFIG_VERSION,
    )


def unknown_score(timestamp_ms: int) -> ScoreResult:
    return ScoreResult(
        timestamp_ms=timestamp_ms,
        state=ScoreState.UNKNOWN,
        raw_score=None,
        smoothed_score=None,
        confidence=0.20,
        feature_deviations=(),
        config_version=SCORING_CONFIG_VERSION,
    )


def test_brief_deviation_never_prompts() -> None:
    policy = AlertPolicy(config())

    started = policy.update(available_score(0, 70.0))
    still_pending = policy.update(available_score(900, 70.0))
    recovered = policy.update(available_score(1_000, 35.0))

    assert started.state is AlertState.DEVIATION_PENDING
    assert still_pending.state is AlertState.DEVIATION_PENDING
    assert recovered.state is AlertState.DEVIATION_PENDING
    assert recovered.continuous_deviation_ms == 0
    assert recovered.posture_debt_ms == pytest.approx(1_000.0)
    assert all(
        decision.event is None for decision in (started, still_pending, recovered)
    )


def test_continuous_deviation_prompts_once_at_sustained_duration() -> None:
    policy = AlertPolicy(config())

    policy.update(available_score(0, 70.0))
    policy.update(available_score(500, 70.0))
    prompted = policy.update(available_score(1_000, 70.0))
    suppressed = policy.update(available_score(1_500, 70.0))

    assert prompted.state is AlertState.COOLDOWN
    assert prompted.event is AlertEvent.PROMPT
    assert prompted.trigger is AlertTrigger.CONTINUOUS
    assert prompted.prompt_message is not None
    assert "comfort" in prompted.prompt_message.lower()
    assert "bad" not in prompted.prompt_message.lower()
    assert "correct" not in prompted.prompt_message.lower()
    assert suppressed.state is AlertState.COOLDOWN
    assert suppressed.event is None


def test_brief_correction_preserves_debt_while_resetting_continuous_time() -> None:
    policy = AlertPolicy(config())

    policy.update(available_score(0, 70.0))
    policy.update(available_score(600, 70.0))
    corrected = policy.update(available_score(800, 35.0))
    decayed = policy.update(available_score(1_000, 35.0))

    assert corrected.continuous_deviation_ms == 0
    assert corrected.posture_debt_ms == pytest.approx(800.0)
    assert decayed.posture_debt_ms == pytest.approx(700.0)
    assert decayed.event is None


def test_repeated_short_deviations_can_trigger_posture_debt() -> None:
    policy = AlertPolicy(config())

    policy.update(available_score(0, 70.0))
    policy.update(available_score(800, 70.0))
    policy.update(available_score(900, 35.0))
    policy.update(available_score(1_100, 35.0))
    policy.update(available_score(1_200, 70.0))
    policy.update(available_score(2_000, 70.0))
    policy.update(available_score(2_100, 35.0))
    policy.update(available_score(2_300, 35.0))
    policy.update(available_score(2_400, 70.0))
    prompted = policy.update(available_score(3_000, 70.0))

    assert prompted.event is AlertEvent.PROMPT
    assert prompted.trigger is AlertTrigger.DEBT


def test_unknown_confidence_preserves_debt_but_cannot_add_or_prompt() -> None:
    policy = AlertPolicy(config())

    policy.update(available_score(0, 70.0))
    policy.update(available_score(500, 70.0))
    unknown = policy.update(unknown_score(700))
    restarted = policy.update(available_score(900, 70.0))

    assert unknown.state is AlertState.UNKNOWN
    assert unknown.event is None
    assert unknown.posture_debt_ms == pytest.approx(500.0)
    assert restarted.posture_debt_ms == pytest.approx(500.0)
    assert restarted.event is None


def test_pause_cancels_pending_and_blocks_score_updates_until_resume() -> None:
    policy = AlertPolicy(config())

    policy.update(available_score(0, 70.0))
    paused = policy.pause(timestamp_ms=500)
    ignored = policy.update(available_score(1_000, 70.0))
    resumed = policy.resume(timestamp_ms=1_500)
    restarted = policy.update(available_score(2_000, 70.0))

    assert paused.state is AlertState.PAUSED
    assert paused.posture_debt_ms == 0.0
    assert ignored.state is AlertState.PAUSED
    assert ignored.event is None
    assert resumed.state is AlertState.MONITORING
    assert restarted.state is AlertState.DEVIATION_PENDING
    assert restarted.pending_since_ms == 2_000


def test_recovery_during_cooldown_does_not_bypass_cooldown() -> None:
    policy = AlertPolicy(config())
    policy.update(available_score(0, 70.0))
    policy.update(available_score(1_000, 70.0))

    recovered = policy.update(available_score(1_500, 35.0))
    suppressed = policy.update(available_score(2_000, 70.0))
    restarted = policy.update(available_score(6_000, 70.0))
    prompted_again = policy.update(available_score(7_000, 70.0))

    assert recovered.state is AlertState.RECOVERED
    assert suppressed.state is AlertState.COOLDOWN
    assert suppressed.event is None
    assert restarted.state is AlertState.DEVIATION_PENDING
    assert prompted_again.event is AlertEvent.PROMPT


def test_sustained_subthreshold_recovery_rearms_a_new_deviation_episode() -> None:
    recovery_config = AlertPolicyConfig(
        version=ALERT_POLICY_CONFIG_VERSION,
        deviation_threshold=60.0,
        recovery_threshold=40.0,
        continuous_deviation_duration_ms=1_000,
        posture_debt_limit_ms=2_000,
        recovery_debt_decay_rate=0.5,
        recovery_duration_ms=1_000,
        cooldown_duration_ms=5_000,
        maximum_observation_gap_ms=1_000,
    )
    policy = AlertPolicy(recovery_config)
    policy.update(available_score(0, 70.0))
    first_prompt = policy.update(available_score(1_000, 70.0))

    recovery_started = policy.update(available_score(1_500, 50.0))
    recovered = policy.update(available_score(2_500, 50.0))
    restarted = policy.update(available_score(6_001, 70.0))
    second_prompt = policy.update(available_score(7_001, 70.0))

    assert first_prompt.event is AlertEvent.PROMPT
    assert recovery_started.state is AlertState.COOLDOWN
    assert recovered.state is AlertState.RECOVERED
    assert restarted.state is AlertState.DEVIATION_PENDING
    assert second_prompt.event is AlertEvent.PROMPT


def test_fluctuating_subthreshold_scores_do_not_rearm_after_cooldown() -> None:
    recovery_config = replace(config(), recovery_duration_ms=1_000)
    policy = AlertPolicy(recovery_config)
    policy.update(available_score(0, 70.0))
    policy.update(available_score(1_000, 70.0))

    policy.update(available_score(1_500, 50.0))
    policy.update(available_score(2_000, 70.0))
    policy.update(available_score(2_500, 50.0))
    policy.update(available_score(3_000, 70.0))
    suppressed = policy.update(available_score(6_001, 70.0))

    assert suppressed.state is AlertState.COOLDOWN
    assert suppressed.event is None


def test_default_policy_realerts_at_refractory_end_for_a_recovered_new_episode() -> None:
    policy = AlertPolicy()

    first_prompt = None
    for timestamp_ms in range(0, 7_001, 1_000):
        first_prompt = policy.update(available_score(timestamp_ms, 80.0))
    assert first_prompt is not None
    assert first_prompt.event is AlertEvent.PROMPT

    recovered = policy.update(available_score(8_000, 40.0))
    assert recovered.state is AlertState.RECOVERED

    repeat_decisions = [
        policy.update(available_score(timestamp_ms, 80.0))
        for timestamp_ms in range(15_000, 27_001, 1_000)
    ]

    second_prompt = repeat_decisions[-1]
    assert all(decision.event is None for decision in repeat_decisions[:-1])
    assert second_prompt.event is AlertEvent.PROMPT
    assert second_prompt.timestamp_ms - first_prompt.timestamp_ms == 20_000


def test_default_policy_realerts_every_20_seconds_during_continuous_deviation() -> None:
    policy = AlertPolicy()

    first_prompt = None
    for timestamp_ms in range(0, 7_001, 1_000):
        first_prompt = policy.update(available_score(timestamp_ms, 80.0))
    assert first_prompt is not None
    assert first_prompt.event is AlertEvent.PROMPT

    repeat_decisions = [
        policy.update(available_score(timestamp_ms, 80.0))
        for timestamp_ms in range(8_000, 27_001, 1_000)
    ]

    second_prompt = repeat_decisions[-1]
    assert all(decision.event is None for decision in repeat_decisions[:-1])
    assert second_prompt.event is AlertEvent.PROMPT
    assert second_prompt.timestamp_ms - first_prompt.timestamp_ms == 20_000


def test_default_policy_threshold_oscillation_does_not_create_popup_storm() -> None:
    policy = AlertPolicy()
    for timestamp_ms in range(0, 7_001, 1_000):
        policy.update(available_score(timestamp_ms, 80.0))

    decisions = [
        policy.update(
            available_score(
                timestamp_ms,
                80.0 if timestamp_ms % 2_000 else 70.0,
            )
        )
        for timestamp_ms in range(8_000, 40_001, 1_000)
    ]

    assert all(decision.event is None for decision in decisions)


def test_cooldown_expiry_still_requires_recovery_before_rearming() -> None:
    policy = AlertPolicy(config())
    policy.update(available_score(0, 70.0))
    policy.update(available_score(1_000, 70.0))

    not_recovered = policy.update(available_score(7_000, 70.0))
    recovered = policy.update(available_score(7_500, 35.0))
    restarted = policy.update(available_score(8_000, 70.0))
    prompted_again = policy.update(available_score(9_000, 70.0))

    assert not_recovered.state is AlertState.COOLDOWN
    assert not_recovered.event is None
    assert recovered.state is AlertState.RECOVERED
    assert restarted.state is AlertState.DEVIATION_PENDING
    assert prompted_again.event is AlertEvent.PROMPT


def test_unknown_during_cooldown_does_not_count_as_recovery() -> None:
    policy = AlertPolicy(config())
    policy.update(available_score(0, 70.0))
    policy.update(available_score(1_000, 70.0))

    unknown = policy.update(unknown_score(1_500))
    still_not_rearmed = policy.update(available_score(7_000, 70.0))
    recovered = policy.update(available_score(7_500, 35.0))

    assert unknown.state is AlertState.UNKNOWN
    assert still_not_rearmed.state is AlertState.COOLDOWN
    assert still_not_rearmed.event is None
    assert recovered.state is AlertState.RECOVERED


def test_long_observation_gap_breaks_continuous_time_without_erasing_debt() -> None:
    policy = AlertPolicy(config())

    policy.update(available_score(0, 70.0))
    policy.update(available_score(500, 70.0))
    after_gap = policy.update(available_score(1_501, 70.0))

    assert after_gap.state is AlertState.DEVIATION_PENDING
    assert after_gap.continuous_deviation_ms == 0
    assert after_gap.posture_debt_ms == pytest.approx(500.0)
    assert after_gap.event is None


def test_timestamps_must_increase_across_scores_and_pause_commands() -> None:
    policy = AlertPolicy(config())
    policy.update(available_score(100, 20.0))

    with pytest.raises(ValueError, match="increase"):
        policy.pause(timestamp_ms=100)


def test_alert_policy_configuration_is_versioned_and_hysteretic() -> None:
    assert ALERT_POLICY_CONFIG_VERSION == 6
    assert DEFAULT_ALERT_POLICY_CONFIG.version == ALERT_POLICY_CONFIG_VERSION
    assert DEFAULT_ALERT_POLICY_CONFIG.deviation_threshold == pytest.approx(75.0)
    assert DEFAULT_ALERT_POLICY_CONFIG.recovery_threshold == pytest.approx(45.0)
    assert DEFAULT_ALERT_POLICY_CONFIG.continuous_deviation_duration_ms == 7_000
    assert DEFAULT_ALERT_POLICY_CONFIG.posture_debt_limit_ms == 12_000
    assert DEFAULT_ALERT_POLICY_CONFIG.recovery_debt_decay_rate == pytest.approx(1.0)
    assert DEFAULT_ALERT_POLICY_CONFIG.recovery_duration_ms == 5_000
    assert DEFAULT_ALERT_POLICY_CONFIG.cooldown_duration_ms == 20_000
    with pytest.raises(ValueError, match="recovery_threshold"):
        replace(DEFAULT_ALERT_POLICY_CONFIG, recovery_threshold=75.0)
