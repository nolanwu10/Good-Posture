"""Versioned local SQLite persistence for derived GoodPosture data only."""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Final

from goodposture.app.session import SessionAggregates
from goodposture.core.calibration import (
    CALIBRATION_CONFIG_VERSION,
    FEATURE_SCHEMA_VERSION,
    CalibrationBaseline,
)

SQLITE_SCHEMA_VERSION: Final = 3


@dataclass(frozen=True, slots=True)
class LocalSettings:
    """Small typed preference set with no camera content or posture history."""

    camera_index: int | None = None
    companion_enabled: bool = True
    notifications_enabled: bool = False

    def __post_init__(self) -> None:
        if self.camera_index is not None and self.camera_index < 0:
            raise ValueError("camera_index must be non-negative")


@dataclass(frozen=True, slots=True)
class DailySummary:
    """One local-day rollup containing only derived durations and counts."""

    local_day: str
    session_count: int
    eligible_monitoring_ms: int
    unknown_ms: int
    deviation_ms: int
    prompt_count: int
    post_prompt_recovered_ms: int = 0
    morning_deviation_ms: int = 0
    afternoon_deviation_ms: int = 0
    evening_deviation_ms: int = 0
    late_night_deviation_ms: int = 0


class SqliteAggregateSink:
    """Map repeatable session checkpoints onto stable local SQLite rows."""

    def __init__(
        self,
        repository: SqliteRepository,
        *,
        run_id: str | None = None,
        utc_now_ms: Callable[[], int] | None = None,
        local_day: Callable[[], str] | None = None,
        local_hour: Callable[[], int] | None = None,
    ) -> None:
        self._repository = repository
        self._run_id = run_id or uuid.uuid4().hex
        _validate_session_id(self._run_id)
        self._utc_now_ms = utc_now_ms or _system_utc_now_ms
        self._local_day = local_day or _system_local_day
        self._local_hour = local_hour or _system_local_hour
        self._session_days: dict[int, str] = {}
        self._last_deviation_ms: dict[int, int] = {}
        self._daypart_deviation_ms: dict[int, list[int]] = {}

    def flush(self, aggregates: SessionAggregates) -> None:
        started_at_ms = aggregates.started_at_ms
        if started_at_ms is None:
            raise ValueError("cannot persist a session before it starts")
        local_day = self._session_days.setdefault(started_at_ms, self._local_day())
        previous_deviation_ms = self._last_deviation_ms.get(started_at_ms, 0)
        dayparts = self._daypart_deviation_ms.setdefault(started_at_ms, [0, 0, 0, 0])
        if aggregates.deviation_ms < previous_deviation_ms:
            dayparts[:] = [0, 0, 0, 0]
            previous_deviation_ms = 0
        dayparts[_daypart_index(self._local_hour())] += (
            aggregates.deviation_ms - previous_deviation_ms
        )
        self._last_deviation_ms[started_at_ms] = aggregates.deviation_ms
        self._repository.save_session(
            session_id=f"{self._run_id}:{started_at_ms}",
            local_day=local_day,
            aggregates=aggregates,
            updated_at_utc_ms=self._utc_now_ms(),
            deviation_by_daypart_ms=(
                dayparts[0],
                dayparts[1],
                dayparts[2],
                dayparts[3],
            ),
        )


class SqliteRepository:
    """Transactional repository with an explicit privacy-bounded schema."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path)
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._migrate()
        except Exception:
            self._connection.close()
            raise

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> SqliteRepository:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _migrate(self) -> None:
        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if version > SQLITE_SCHEMA_VERSION:
            raise RuntimeError(
                f"database schema {version} is newer than supported "
                f"schema {SQLITE_SCHEMA_VERSION}"
            )
        with self._connection:
            if version == 0:
                self._create_current_schema()
            elif version == 1:
                self._create_session_schema()
            elif version == 2:
                self._migrate_outcome_schema()
            self._connection.execute(f"PRAGMA user_version = {SQLITE_SCHEMA_VERSION}")

    def _create_current_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE settings (
                singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                camera_index INTEGER CHECK (camera_index IS NULL OR camera_index >= 0),
                companion_enabled INTEGER NOT NULL CHECK (companion_enabled IN (0, 1)),
                notifications_enabled INTEGER NOT NULL
                    CHECK (notifications_enabled IN (0, 1))
            );
            CREATE TABLE calibration (
                singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                baseline_json TEXT NOT NULL,
                saved_at_utc_ms INTEGER NOT NULL CHECK (saved_at_utc_ms >= 0)
            );
            """
        )
        self._create_session_schema()

    def _create_session_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY NOT NULL,
                local_day TEXT NOT NULL,
                started_at_monotonic_ms INTEGER,
                ended_at_monotonic_ms INTEGER,
                eligible_monitoring_ms INTEGER NOT NULL
                    CHECK (eligible_monitoring_ms >= 0),
                unknown_ms INTEGER NOT NULL CHECK (unknown_ms >= 0),
                deviation_ms INTEGER NOT NULL CHECK (deviation_ms >= 0),
                prompt_count INTEGER NOT NULL CHECK (prompt_count >= 0),
                post_prompt_recovered_ms INTEGER NOT NULL DEFAULT 0
                    CHECK (post_prompt_recovered_ms >= 0),
                morning_deviation_ms INTEGER NOT NULL DEFAULT 0
                    CHECK (morning_deviation_ms >= 0),
                afternoon_deviation_ms INTEGER NOT NULL DEFAULT 0
                    CHECK (afternoon_deviation_ms >= 0),
                evening_deviation_ms INTEGER NOT NULL DEFAULT 0
                    CHECK (evening_deviation_ms >= 0),
                late_night_deviation_ms INTEGER NOT NULL DEFAULT 0
                    CHECK (late_night_deviation_ms >= 0),
                updated_at_utc_ms INTEGER NOT NULL CHECK (updated_at_utc_ms >= 0),
                finalized INTEGER NOT NULL CHECK (finalized IN (0, 1))
            );
            CREATE INDEX sessions_local_day_index ON sessions(local_day);
            CREATE TABLE daily_aggregates (
                local_day TEXT PRIMARY KEY NOT NULL,
                session_count INTEGER NOT NULL CHECK (session_count >= 0),
                eligible_monitoring_ms INTEGER NOT NULL
                    CHECK (eligible_monitoring_ms >= 0),
                unknown_ms INTEGER NOT NULL CHECK (unknown_ms >= 0),
                deviation_ms INTEGER NOT NULL CHECK (deviation_ms >= 0),
                prompt_count INTEGER NOT NULL CHECK (prompt_count >= 0),
                post_prompt_recovered_ms INTEGER NOT NULL DEFAULT 0
                    CHECK (post_prompt_recovered_ms >= 0),
                morning_deviation_ms INTEGER NOT NULL DEFAULT 0
                    CHECK (morning_deviation_ms >= 0),
                afternoon_deviation_ms INTEGER NOT NULL DEFAULT 0
                    CHECK (afternoon_deviation_ms >= 0),
                evening_deviation_ms INTEGER NOT NULL DEFAULT 0
                    CHECK (evening_deviation_ms >= 0),
                late_night_deviation_ms INTEGER NOT NULL DEFAULT 0
                    CHECK (late_night_deviation_ms >= 0),
                updated_at_utc_ms INTEGER NOT NULL CHECK (updated_at_utc_ms >= 0)
            );
            """
        )

    def _migrate_outcome_schema(self) -> None:
        for table in ("sessions", "daily_aggregates"):
            for column in (
                "post_prompt_recovered_ms",
                "morning_deviation_ms",
                "afternoon_deviation_ms",
                "evening_deviation_ms",
                "late_night_deviation_ms",
            ):
                self._connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} "
                    "INTEGER NOT NULL DEFAULT 0 CHECK "
                    f"({column} >= 0)"
                )

    @staticmethod
    def serialize_calibration(baseline: CalibrationBaseline) -> str:
        """Return canonical JSON containing derived baseline statistics only."""

        return json.dumps(
            baseline.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def save_calibration(
        self,
        baseline: CalibrationBaseline,
        *,
        saved_at_utc_ms: int,
    ) -> None:
        _validate_nonnegative(saved_at_utc_ms, "saved_at_utc_ms")
        payload = self.serialize_calibration(baseline)
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO calibration(singleton_id, baseline_json, saved_at_utc_ms)
                VALUES (1, ?, ?)
                ON CONFLICT(singleton_id) DO UPDATE SET
                    baseline_json = excluded.baseline_json,
                    saved_at_utc_ms = excluded.saved_at_utc_ms
                """,
                (payload, saved_at_utc_ms),
            )

    def load_calibration(self) -> CalibrationBaseline | None:
        row = self._connection.execute(
            "SELECT baseline_json FROM calibration WHERE singleton_id = 1"
        ).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(str(row["baseline_json"]))
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError("saved calibration is not valid JSON") from error
        return CalibrationBaseline.from_dict(value)

    def load_compatible_calibration(
        self,
        *,
        model_id: str,
    ) -> CalibrationBaseline | None:
        """Return a reusable baseline only when its interpretation still matches."""

        if not model_id:
            raise ValueError("model_id must be non-empty")
        baseline = self.load_calibration()
        if baseline is None:
            return None
        if (
            baseline.model_id != model_id
            or baseline.feature_schema_version != FEATURE_SCHEMA_VERSION
            or baseline.calibration_config_version != CALIBRATION_CONFIG_VERSION
        ):
            return None
        return baseline

    def delete_calibration(self) -> None:
        with self._connection:
            self._connection.execute("DELETE FROM calibration")

    def save_settings(self, settings: LocalSettings) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO settings(
                    singleton_id,
                    camera_index,
                    companion_enabled,
                    notifications_enabled
                )
                VALUES (1, ?, ?, ?)
                ON CONFLICT(singleton_id) DO UPDATE SET
                    camera_index = excluded.camera_index,
                    companion_enabled = excluded.companion_enabled,
                    notifications_enabled = excluded.notifications_enabled
                """,
                (
                    settings.camera_index,
                    int(settings.companion_enabled),
                    int(settings.notifications_enabled),
                ),
            )

    def load_settings(self) -> LocalSettings:
        row = self._connection.execute(
            """
            SELECT camera_index, companion_enabled, notifications_enabled
            FROM settings
            WHERE singleton_id = 1
            """
        ).fetchone()
        if row is None:
            return LocalSettings()
        camera_index = row["camera_index"]
        return LocalSettings(
            camera_index=None if camera_index is None else int(camera_index),
            companion_enabled=bool(row["companion_enabled"]),
            notifications_enabled=bool(row["notifications_enabled"]),
        )

    def save_session(
        self,
        *,
        session_id: str,
        local_day: str,
        aggregates: SessionAggregates,
        updated_at_utc_ms: int,
        deviation_by_daypart_ms: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        _validate_session_id(session_id)
        _validate_local_day(local_day)
        _validate_aggregates(aggregates)
        _validate_nonnegative(updated_at_utc_ms, "updated_at_utc_ms")
        for value in deviation_by_daypart_ms:
            _validate_nonnegative(value, "deviation_by_daypart_ms")
        with self._connection:
            existing = self._connection.execute(
                "SELECT local_day FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if existing is not None and str(existing["local_day"]) != local_day:
                raise ValueError("an existing session cannot move to another local day")
            self._connection.execute(
                """
                INSERT INTO sessions(
                    session_id,
                    local_day,
                    started_at_monotonic_ms,
                    ended_at_monotonic_ms,
                    eligible_monitoring_ms,
                    unknown_ms,
                    deviation_ms,
                    prompt_count,
                    post_prompt_recovered_ms,
                    morning_deviation_ms,
                    afternoon_deviation_ms,
                    evening_deviation_ms,
                    late_night_deviation_ms,
                    updated_at_utc_ms,
                    finalized
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    ended_at_monotonic_ms = excluded.ended_at_monotonic_ms,
                    eligible_monitoring_ms = excluded.eligible_monitoring_ms,
                    unknown_ms = excluded.unknown_ms,
                    deviation_ms = excluded.deviation_ms,
                    prompt_count = excluded.prompt_count,
                    post_prompt_recovered_ms = excluded.post_prompt_recovered_ms,
                    morning_deviation_ms = excluded.morning_deviation_ms,
                    afternoon_deviation_ms = excluded.afternoon_deviation_ms,
                    evening_deviation_ms = excluded.evening_deviation_ms,
                    late_night_deviation_ms = excluded.late_night_deviation_ms,
                    updated_at_utc_ms = excluded.updated_at_utc_ms,
                    finalized = excluded.finalized
                """,
                (
                    session_id,
                    local_day,
                    aggregates.started_at_ms,
                    aggregates.ended_at_ms,
                    aggregates.eligible_monitoring_ms,
                    aggregates.unknown_ms,
                    aggregates.deviation_ms,
                    aggregates.prompt_count,
                    aggregates.post_prompt_recovered_ms,
                    *deviation_by_daypart_ms,
                    updated_at_utc_ms,
                    int(aggregates.ended_at_ms is not None),
                ),
            )
            self._rebuild_daily_summary(local_day, updated_at_utc_ms)

    def _rebuild_daily_summary(self, local_day: str, updated_at_utc_ms: int) -> None:
        row = self._connection.execute(
            """
            SELECT
                COUNT(*) AS session_count,
                COALESCE(SUM(eligible_monitoring_ms), 0) AS eligible_monitoring_ms,
                COALESCE(SUM(unknown_ms), 0) AS unknown_ms,
                COALESCE(SUM(deviation_ms), 0) AS deviation_ms,
                COALESCE(SUM(prompt_count), 0) AS prompt_count,
                COALESCE(SUM(post_prompt_recovered_ms), 0)
                    AS post_prompt_recovered_ms,
                COALESCE(SUM(morning_deviation_ms), 0) AS morning_deviation_ms,
                COALESCE(SUM(afternoon_deviation_ms), 0) AS afternoon_deviation_ms,
                COALESCE(SUM(evening_deviation_ms), 0) AS evening_deviation_ms,
                COALESCE(SUM(late_night_deviation_ms), 0)
                    AS late_night_deviation_ms
            FROM sessions
            WHERE local_day = ?
            """,
            (local_day,),
        ).fetchone()
        assert row is not None
        self._connection.execute(
            """
            INSERT INTO daily_aggregates(
                local_day,
                session_count,
                eligible_monitoring_ms,
                unknown_ms,
                deviation_ms,
                prompt_count,
                post_prompt_recovered_ms,
                morning_deviation_ms,
                afternoon_deviation_ms,
                evening_deviation_ms,
                late_night_deviation_ms,
                updated_at_utc_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(local_day) DO UPDATE SET
                session_count = excluded.session_count,
                eligible_monitoring_ms = excluded.eligible_monitoring_ms,
                unknown_ms = excluded.unknown_ms,
                deviation_ms = excluded.deviation_ms,
                prompt_count = excluded.prompt_count,
                post_prompt_recovered_ms = excluded.post_prompt_recovered_ms,
                morning_deviation_ms = excluded.morning_deviation_ms,
                afternoon_deviation_ms = excluded.afternoon_deviation_ms,
                evening_deviation_ms = excluded.evening_deviation_ms,
                late_night_deviation_ms = excluded.late_night_deviation_ms,
                updated_at_utc_ms = excluded.updated_at_utc_ms
            """,
            (
                local_day,
                int(row["session_count"]),
                int(row["eligible_monitoring_ms"]),
                int(row["unknown_ms"]),
                int(row["deviation_ms"]),
                int(row["prompt_count"]),
                int(row["post_prompt_recovered_ms"]),
                int(row["morning_deviation_ms"]),
                int(row["afternoon_deviation_ms"]),
                int(row["evening_deviation_ms"]),
                int(row["late_night_deviation_ms"]),
                updated_at_utc_ms,
            ),
        )

    def daily_summary(self, local_day: str) -> DailySummary | None:
        _validate_local_day(local_day)
        row = self._connection.execute(
            """
            SELECT
                local_day,
                session_count,
                eligible_monitoring_ms,
                unknown_ms,
                deviation_ms,
                prompt_count,
                post_prompt_recovered_ms,
                morning_deviation_ms,
                afternoon_deviation_ms,
                evening_deviation_ms,
                late_night_deviation_ms
            FROM daily_aggregates
            WHERE local_day = ?
            """,
            (local_day,),
        ).fetchone()
        if row is None:
            return None
        return DailySummary(
            local_day=str(row["local_day"]),
            session_count=int(row["session_count"]),
            eligible_monitoring_ms=int(row["eligible_monitoring_ms"]),
            unknown_ms=int(row["unknown_ms"]),
            deviation_ms=int(row["deviation_ms"]),
            prompt_count=int(row["prompt_count"]),
            post_prompt_recovered_ms=int(row["post_prompt_recovered_ms"]),
            morning_deviation_ms=int(row["morning_deviation_ms"]),
            afternoon_deviation_ms=int(row["afternoon_deviation_ms"]),
            evening_deviation_ms=int(row["evening_deviation_ms"]),
            late_night_deviation_ms=int(row["late_night_deviation_ms"]),
        )

    def recent_daily_summaries(self, *, limit: int = 7) -> tuple[DailySummary, ...]:
        if not 1 <= limit <= 366:
            raise ValueError("limit must be between 1 and 366")
        rows = self._connection.execute(
            """
            SELECT
                local_day,
                session_count,
                eligible_monitoring_ms,
                unknown_ms,
                deviation_ms,
                prompt_count,
                post_prompt_recovered_ms,
                morning_deviation_ms,
                afternoon_deviation_ms,
                evening_deviation_ms,
                late_night_deviation_ms
            FROM daily_aggregates
            ORDER BY local_day DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return tuple(
            DailySummary(
                local_day=str(row["local_day"]),
                session_count=int(row["session_count"]),
                eligible_monitoring_ms=int(row["eligible_monitoring_ms"]),
                unknown_ms=int(row["unknown_ms"]),
                deviation_ms=int(row["deviation_ms"]),
                prompt_count=int(row["prompt_count"]),
                post_prompt_recovered_ms=int(row["post_prompt_recovered_ms"]),
                morning_deviation_ms=int(row["morning_deviation_ms"]),
                afternoon_deviation_ms=int(row["afternoon_deviation_ms"]),
                evening_deviation_ms=int(row["evening_deviation_ms"]),
                late_night_deviation_ms=int(row["late_night_deviation_ms"]),
            )
            for row in rows
        )

    def delete_history(self) -> None:
        with self._connection:
            self._connection.execute("DELETE FROM sessions")
            self._connection.execute("DELETE FROM daily_aggregates")

    def delete_all(self) -> None:
        with self._connection:
            self._connection.execute("DELETE FROM daily_aggregates")
            self._connection.execute("DELETE FROM sessions")
            self._connection.execute("DELETE FROM calibration")
            self._connection.execute("DELETE FROM settings")


def _validate_nonnegative(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_session_id(session_id: str) -> None:
    if not session_id or len(session_id) > 200:
        raise ValueError("session_id must contain 1 to 200 characters")


def _validate_local_day(local_day: str) -> None:
    try:
        parsed = date.fromisoformat(local_day)
    except ValueError as error:
        raise ValueError("local_day must use YYYY-MM-DD format") from error
    if parsed.isoformat() != local_day:
        raise ValueError("local_day must use canonical YYYY-MM-DD format")


def _validate_aggregates(aggregates: SessionAggregates) -> None:
    for name in (
        "eligible_monitoring_ms",
        "unknown_ms",
        "deviation_ms",
        "prompt_count",
        "post_prompt_recovered_ms",
    ):
        _validate_nonnegative(getattr(aggregates, name), name)
    if aggregates.started_at_ms is not None:
        _validate_nonnegative(aggregates.started_at_ms, "started_at_ms")
    if aggregates.ended_at_ms is not None:
        _validate_nonnegative(aggregates.ended_at_ms, "ended_at_ms")
        if (
            aggregates.started_at_ms is None
            or aggregates.ended_at_ms < aggregates.started_at_ms
        ):
            raise ValueError("ended_at_ms requires and cannot predate started_at_ms")
    if aggregates.deviation_ms > aggregates.eligible_monitoring_ms:
        raise ValueError("deviation_ms cannot exceed eligible_monitoring_ms")
    if not all(
        math.isfinite(float(value))
        for value in (
            aggregates.eligible_monitoring_ms,
            aggregates.unknown_ms,
            aggregates.deviation_ms,
            aggregates.prompt_count,
            aggregates.post_prompt_recovered_ms,
        )
    ):
        raise ValueError("aggregate values must be finite")


def _system_utc_now_ms() -> int:
    return time.time_ns() // 1_000_000


def _system_local_day() -> str:
    return date.today().isoformat()


def _system_local_hour() -> int:
    return datetime.now().astimezone().hour


def _daypart_index(hour: int) -> int:
    if not 0 <= hour <= 23:
        raise ValueError("local hour must be between 0 and 23")
    if 5 <= hour < 12:
        return 0
    if 12 <= hour < 17:
        return 1
    if 17 <= hour < 22:
        return 2
    return 3
