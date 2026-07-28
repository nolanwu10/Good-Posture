from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from goodposture.core.alert_policy import DEFAULT_ALERT_POLICY_CONFIG
from goodposture.evaluation import (
    DEFAULT_EVALUATION_CORPUS_PATH,
    EvaluationGate,
    compare_evaluation_reports,
    evaluate_corpus,
    load_evaluation_corpus,
    render_evaluation_markdown,
    run_evaluation_protocol,
)


def test_versioned_corpus_covers_required_synthetic_scenarios() -> None:
    corpus = load_evaluation_corpus(DEFAULT_EVALUATION_CORPUS_PATH)

    assert corpus.schema_version == 1
    assert corpus.corpus_version == "synthetic-landmarks-v1"
    assert corpus.provenance == "synthetic"
    assert corpus.consent_reference is None
    assert {scenario.label for scenario in corpus.scenarios} >= {
        "upright",
        "slouch_like_deviation",
        "lean",
        "intentional_movement",
        "occlusion",
        "out_of_frame",
    }


def test_consented_derived_corpus_requires_explicit_deidentified_consent(
    tmp_path: Path,
) -> None:
    raw = json.loads(DEFAULT_EVALUATION_CORPUS_PATH.read_text(encoding="utf-8"))
    raw["provenance"] = "consented_derived"
    raw["deidentified"] = True
    raw["consent_reference"] = None
    path = tmp_path / "missing-consent.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="consent_reference"):
        load_evaluation_corpus(path)


def test_corpus_rejects_raw_media_fields(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_EVALUATION_CORPUS_PATH.read_text(encoding="utf-8"))
    raw["video_path"] = "never-allowed.mp4"
    path = tmp_path / "raw-media.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="raw media"):
        load_evaluation_corpus(path)


def test_tuned_profile_removes_false_prompts_without_missing_sustained_events() -> None:
    corpus = load_evaluation_corpus(DEFAULT_EVALUATION_CORPUS_PATH)
    before_config = replace(
        DEFAULT_ALERT_POLICY_CONFIG,
        version=3,
        deviation_threshold=60.0,
        recovery_threshold=40.0,
        continuous_deviation_duration_ms=5_000,
        posture_debt_limit_ms=8_000,
        recovery_debt_decay_rate=0.5,
    )
    tuned_config = replace(
        DEFAULT_ALERT_POLICY_CONFIG,
        version=4,
        deviation_threshold=75.0,
        recovery_threshold=45.0,
        continuous_deviation_duration_ms=7_000,
        posture_debt_limit_ms=12_000,
        recovery_debt_decay_rate=1.0,
    )

    before = evaluate_corpus(corpus, profile_name="task11-default-v3", alert_config=before_config)
    after = evaluate_corpus(corpus, profile_name="task12-synthetic-v1", alert_config=tuned_config)

    assert before.metrics.false_prompt_count > 0
    assert after.metrics.false_prompt_count == 0
    assert after.metrics.missed_sustained_deviations == 0
    assert after.metrics.prompt_precision == pytest.approx(1.0)
    assert after.metrics.frame_rate_robustness == pytest.approx(1.0)
    assert after.metrics.maximum_prompt_latency_spread_ms <= 500
    assert after.metrics.unknown_coverage > 0.0

    comparison = compare_evaluation_reports(
        before,
        after,
        gate=EvaluationGate(
            maximum_added_false_prompts=0,
            maximum_added_misses=0,
            maximum_prompt_latency_increase_ms=2_500,
            maximum_unknown_coverage_increase=0.0,
            minimum_frame_rate_robustness=1.0,
        ),
    )
    assert comparison.passed is True
    assert comparison.failures == ()


def test_report_is_repeatable_and_contains_only_aggregate_evaluation_data() -> None:
    corpus = load_evaluation_corpus(DEFAULT_EVALUATION_CORPUS_PATH)

    first = evaluate_corpus(corpus, profile_name="repeatable")
    second = evaluate_corpus(corpus, profile_name="repeatable")
    payload = first.to_json()
    markdown = render_evaluation_markdown(first)

    assert first == second
    assert payload == second.to_json()
    assert '"landmarks"' not in payload
    assert '"templates"' not in payload
    assert "Prompt precision" in markdown
    assert "Known limitations" in markdown


def test_regression_gate_reports_missed_sustained_deviations() -> None:
    corpus = load_evaluation_corpus(DEFAULT_EVALUATION_CORPUS_PATH)
    before = evaluate_corpus(corpus, profile_name="before")
    misses_everything = evaluate_corpus(
        corpus,
        profile_name="misses",
        alert_config=replace(
            DEFAULT_ALERT_POLICY_CONFIG,
            deviation_threshold=100.0,
            continuous_deviation_duration_ms=60_000,
            posture_debt_limit_ms=60_000,
        ),
    )

    comparison = compare_evaluation_reports(before, misses_everything)

    assert comparison.passed is False
    assert any("missed" in failure for failure in comparison.failures)


def test_default_protocol_writes_before_after_and_gate_reports(tmp_path: Path) -> None:
    result = run_evaluation_protocol(
        corpus_path=DEFAULT_EVALUATION_CORPUS_PATH,
        output_directory=tmp_path,
    )

    assert result.comparison.passed is True
    assert result.before.alert_config_version == 3
    assert result.after.alert_config_version == 6
    assert result.before.metrics.false_prompt_count > 0
    assert result.after.metrics.false_prompt_count == 0
    assert {path.name for path in tmp_path.iterdir()} == {
        "before-task11-default-v3.json",
        "before-task11-default-v3.md",
        "after-task12-synthetic-v1.json",
        "after-task12-synthetic-v1.md",
        "comparison.json",
    }
    comparison_text = (tmp_path / "comparison.json").read_text()
    comparison_payload = json.loads(comparison_text)
    assert comparison_payload["passed"] is True
    assert '"landmarks":' not in comparison_text
    assert '"templates":' not in comparison_text
