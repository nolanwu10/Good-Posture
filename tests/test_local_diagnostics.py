from __future__ import annotations

import json
from pathlib import Path

import pytest

from goodposture.adapters.local_diagnostics import LocalDiagnosticLog
from goodposture.app.diagnostics import DiagnosticEvent


def test_local_diagnostic_log_contains_only_allowlisted_operational_fields(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "diagnostics.jsonl"
    diagnostics = LocalDiagnosticLog(log_path, utc_now_ms=lambda: 1_234)

    diagnostics.record(DiagnosticEvent.APP_STARTED)

    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload == {
        "event": "app_started",
        "schema_version": 1,
        "timestamp_utc_ms": 1_234,
    }
    serialized = json.dumps(payload).lower()
    assert all(
        forbidden not in serialized
        for forbidden in (
            "landmark",
            "frame",
            "image",
            "video",
            "database",
            "path",
            "observation",
        )
    )


def test_local_diagnostic_log_rejects_arbitrary_values(tmp_path: Path) -> None:
    diagnostics = LocalDiagnosticLog(tmp_path / "diagnostics.jsonl")

    with pytest.raises(TypeError, match="DiagnosticEvent"):
        diagnostics.record("camera_started")  # type: ignore[arg-type]


def test_local_diagnostic_log_rotates_to_a_bounded_number_of_files(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "diagnostics.jsonl"
    diagnostics = LocalDiagnosticLog(
        log_path,
        maximum_bytes=100,
        backup_count=2,
        utc_now_ms=lambda: 1,
    )

    for _ in range(10):
        diagnostics.record(DiagnosticEvent.CAMERA_STARTED)

    files = sorted(path.name for path in tmp_path.iterdir())
    assert files == [
        "diagnostics.jsonl",
        "diagnostics.jsonl.1",
        "diagnostics.jsonl.2",
    ]
    assert all(path.stat().st_size <= 100 for path in tmp_path.iterdir())
