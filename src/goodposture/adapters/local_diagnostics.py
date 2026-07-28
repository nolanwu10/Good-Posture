"""Bounded local JSON-lines diagnostics with an allowlisted payload."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

from goodposture.app.diagnostics import DiagnosticEvent

DIAGNOSTIC_SCHEMA_VERSION = 1


class LocalDiagnosticLog:
    """Write operational event codes without frames, landmarks, or user data."""

    def __init__(
        self,
        path: Path,
        *,
        maximum_bytes: int = 256 * 1024,
        backup_count: int = 2,
        utc_now_ms: Callable[[], int] | None = None,
    ) -> None:
        if maximum_bytes < 64:
            raise ValueError("maximum_bytes must be at least 64")
        if backup_count < 1:
            raise ValueError("backup_count must be positive")
        self._path = path
        self._maximum_bytes = maximum_bytes
        self._backup_count = backup_count
        self._utc_now_ms = utc_now_ms or _system_utc_now_ms
        self._lock = threading.Lock()

    def record(self, event: DiagnosticEvent) -> None:
        """Append one fixed-schema event and rotate before exceeding the bound."""

        if not isinstance(event, DiagnosticEvent):
            raise TypeError("event must be a DiagnosticEvent")
        timestamp_utc_ms = self._utc_now_ms()
        if (
            isinstance(timestamp_utc_ms, bool)
            or not isinstance(timestamp_utc_ms, int)
            or timestamp_utc_ms < 0
        ):
            raise ValueError("diagnostic timestamp must be a non-negative integer")
        line = (
            json.dumps(
                {
                    "event": event.value,
                    "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
                    "timestamp_utc_ms": timestamp_utc_ms,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        encoded_length = len(line.encode("utf-8"))
        if encoded_length > self._maximum_bytes:
            raise ValueError("diagnostic entry exceeds maximum_bytes")
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            current_size = self._path.stat().st_size if self._path.exists() else 0
            if current_size + encoded_length > self._maximum_bytes:
                self._rotate()
            with self._path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line)

    def _rotate(self) -> None:
        for index in range(self._backup_count, 0, -1):
            source = (
                self._path
                if index == 1
                else Path(f"{self._path}.{index - 1}")
            )
            destination = Path(f"{self._path}.{index}")
            if destination.exists():
                destination.unlink()
            if source.exists():
                source.replace(destination)


def _system_utc_now_ms() -> int:
    return time.time_ns() // 1_000_000
