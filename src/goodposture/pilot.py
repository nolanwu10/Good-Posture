"""Aggregate-only records for the explicitly local lived-use pilot."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

PILOT_SCHEMA_VERSION = 1
MINIMUM_PILOT_DAYS = 3


@dataclass(frozen=True, slots=True)
class PilotEntry:
    local_day: date
    monitoring_minutes: int
    peak_cpu_percent: float
    peak_memory_mb: float
    false_prompt_count: int
    missed_prompt_count: int
    failure_count: int
    usability_rating: int

    def __post_init__(self) -> None:
        counts = (
            self.monitoring_minutes,
            self.false_prompt_count,
            self.missed_prompt_count,
            self.failure_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("pilot counts cannot be negative")
        if not math.isfinite(self.peak_cpu_percent) or not 0 <= self.peak_cpu_percent <= 100:
            raise ValueError("peak CPU percent must be between 0 and 100")
        if not math.isfinite(self.peak_memory_mb) or self.peak_memory_mb < 0:
            raise ValueError("peak memory must be a finite non-negative value")
        if not 1 <= self.usability_rating <= 5:
            raise ValueError("usability rating must be between 1 and 5")


@dataclass(frozen=True, slots=True)
class PilotSummary:
    distinct_days: int
    monitoring_minutes: int
    peak_cpu_percent: float
    peak_memory_mb: float
    false_prompt_count: int
    missed_prompt_count: int
    failure_count: int
    average_usability_rating: float
    collection_complete: bool


def append_pilot_entry(path: Path, entry: PilotEntry) -> None:
    """Append one fixed-schema day record without free text or observation data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(entry)
    payload["local_day"] = entry.local_day.isoformat()
    payload["schema_version"] = PILOT_SCHEMA_VERSION
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def summarize_pilot(path: Path) -> PilotSummary:
    entries = _read_entries(path)
    days = {item.local_day for item in entries}
    ratings = [item.usability_rating for item in entries]
    return PilotSummary(
        distinct_days=len(days),
        monitoring_minutes=sum(item.monitoring_minutes for item in entries),
        peak_cpu_percent=max((item.peak_cpu_percent for item in entries), default=0.0),
        peak_memory_mb=max((item.peak_memory_mb for item in entries), default=0.0),
        false_prompt_count=sum(item.false_prompt_count for item in entries),
        missed_prompt_count=sum(item.missed_prompt_count for item in entries),
        failure_count=sum(item.failure_count for item in entries),
        average_usability_rating=sum(ratings) / len(ratings) if ratings else 0.0,
        collection_complete=len(days) >= MINIMUM_PILOT_DAYS,
    )


def _read_entries(path: Path) -> tuple[PilotEntry, ...]:
    if not path.is_file():
        return ()
    entries: list[PilotEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload.pop("schema_version", None) != PILOT_SCHEMA_VERSION:
            raise ValueError("unsupported pilot schema version")
        entries.append(
            PilotEntry(
                local_day=date.fromisoformat(payload["local_day"]),
                monitoring_minutes=payload["monitoring_minutes"],
                peak_cpu_percent=payload["peak_cpu_percent"],
                peak_memory_mb=payload["peak_memory_mb"],
                false_prompt_count=payload["false_prompt_count"],
                missed_prompt_count=payload["missed_prompt_count"],
                failure_count=payload["failure_count"],
                usability_rating=payload["usability_rating"],
            )
        )
    return tuple(entries)
