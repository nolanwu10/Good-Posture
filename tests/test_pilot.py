from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from goodposture.pilot import PilotEntry, append_pilot_entry, summarize_pilot


def entry(day: int, *, false_prompts: int = 0) -> PilotEntry:
    return PilotEntry(
        local_day=date(2026, 7, day),
        monitoring_minutes=120,
        peak_cpu_percent=8.5,
        peak_memory_mb=240.0,
        false_prompt_count=false_prompts,
        missed_prompt_count=0,
        failure_count=0,
        usability_rating=4,
    )


def test_pilot_log_has_a_fixed_aggregate_only_schema(tmp_path: Path) -> None:
    path = tmp_path / "pilot.jsonl"

    append_pilot_entry(path, entry(26))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "failure_count",
        "false_prompt_count",
        "local_day",
        "missed_prompt_count",
        "monitoring_minutes",
        "peak_cpu_percent",
        "peak_memory_mb",
        "schema_version",
        "usability_rating",
    }
    assert "frame" not in path.read_text(encoding="utf-8").lower()
    assert "landmark" not in path.read_text(encoding="utf-8").lower()


def test_pilot_summary_requires_three_distinct_days(tmp_path: Path) -> None:
    path = tmp_path / "pilot.jsonl"
    append_pilot_entry(path, entry(26, false_prompts=1))
    append_pilot_entry(path, entry(27))
    append_pilot_entry(path, entry(28))

    summary = summarize_pilot(path)

    assert summary.distinct_days == 3
    assert summary.collection_complete is True
    assert summary.false_prompt_count == 1
    assert summary.monitoring_minutes == 360


def test_pilot_entry_rejects_invalid_or_unbounded_metrics() -> None:
    with pytest.raises(ValueError):
        PilotEntry(
            local_day=date(2026, 7, 26),
            monitoring_minutes=-1,
            peak_cpu_percent=0,
            peak_memory_mb=0,
            false_prompt_count=0,
            missed_prompt_count=0,
            failure_count=0,
            usability_rating=4,
        )
