from __future__ import annotations

import sqlite3
from pathlib import Path

from goodposture.adapters.sqlite_repository import (
    SQLITE_SCHEMA_VERSION,
    LocalSettings,
    SqliteAggregateSink,
    SqliteRepository,
)
from goodposture.app.session import SessionAggregates
from goodposture.core.calibration import (
    CALIBRATION_CONFIG_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    CalibrationBaseline,
    CalibrationFeature,
)


def baseline() -> CalibrationBaseline:
    return CalibrationBaseline(
        calibration_schema_version=CALIBRATION_SCHEMA_VERSION,
        calibration_config_version=CALIBRATION_CONFIG_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_id="pose-test-v1",
        started_at_ms=100,
        completed_at_ms=200,
        accepted_sample_count=60,
        rejected_sample_count=2,
        features=(
            CalibrationFeature("shoulder_tilt_degrees", 0.5, 0.25),
            CalibrationFeature("head_lateral_offset_ratio", 0.0, 0.01),
            CalibrationFeature("head_vertical_offset_ratio", 0.7, 0.02),
            CalibrationFeature("head_depth_ratio", 0.6, 0.02),
        ),
    )


def aggregates(
    *,
    started_at_ms: int = 1_000,
    ended_at_ms: int | None = None,
    eligible_monitoring_ms: int = 10_000,
    unknown_ms: int = 2_000,
    deviation_ms: int = 3_000,
    prompt_count: int = 1,
    post_prompt_recovered_ms: int = 0,
) -> SessionAggregates:
    return SessionAggregates(
        started_at_ms=started_at_ms,
        ended_at_ms=ended_at_ms,
        eligible_monitoring_ms=eligible_monitoring_ms,
        unknown_ms=unknown_ms,
        deviation_ms=deviation_ms,
        prompt_count=prompt_count,
        post_prompt_recovered_ms=post_prompt_recovered_ms,
    )


def test_fresh_database_has_versioned_privacy_bounded_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "goodposture.sqlite3"

    repository = SqliteRepository(database_path)
    repository.close()

    with sqlite3.connect(database_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        columns = {
            row[1]
            for table in tables
            if not table.startswith("sqlite_")
            for row in connection.execute(f"PRAGMA table_info({table})")
        }

    assert version == SQLITE_SCHEMA_VERSION
    assert {"settings", "calibration", "sessions", "daily_aggregates"} <= tables
    assert not {
        "frame",
        "image",
        "video",
        "landmark",
        "observation",
        "rgb_bytes",
    } & columns


def test_calibration_and_typed_settings_round_trip(tmp_path: Path) -> None:
    repository = SqliteRepository(tmp_path / "goodposture.sqlite3")
    saved_settings = LocalSettings(
        camera_index=2,
        companion_enabled=False,
        notifications_enabled=True,
    )

    repository.save_calibration(baseline(), saved_at_utc_ms=1_000)
    repository.save_settings(saved_settings)

    assert repository.load_calibration() == baseline()
    assert repository.load_compatible_calibration(model_id="pose-test-v1") == baseline()
    assert repository.load_settings() == saved_settings


def test_incompatible_saved_calibration_is_not_reused(tmp_path: Path) -> None:
    repository = SqliteRepository(tmp_path / "goodposture.sqlite3")
    repository.save_calibration(baseline(), saved_at_utc_ms=1_000)

    assert repository.load_compatible_calibration(model_id="other-model") is None
    assert repository.load_calibration() == baseline()


def test_session_checkpoint_is_idempotent_and_rebuilds_daily_rollup(
    tmp_path: Path,
) -> None:
    repository = SqliteRepository(tmp_path / "goodposture.sqlite3")
    first = aggregates()
    final = aggregates(
        ended_at_ms=12_000,
        eligible_monitoring_ms=11_000,
        unknown_ms=2_500,
        deviation_ms=4_000,
        prompt_count=2,
    )

    repository.save_session(
        session_id="run-a:1000",
        local_day="2026-07-26",
        aggregates=first,
        updated_at_utc_ms=10_000,
    )
    repository.save_session(
        session_id="run-a:1000",
        local_day="2026-07-26",
        aggregates=first,
        updated_at_utc_ms=10_000,
    )
    repository.save_session(
        session_id="run-a:1000",
        local_day="2026-07-26",
        aggregates=final,
        updated_at_utc_ms=12_000,
    )

    summary = repository.daily_summary("2026-07-26")

    assert summary is not None
    assert summary.session_count == 1
    assert summary.eligible_monitoring_ms == 11_000
    assert summary.unknown_ms == 2_500
    assert summary.deviation_ms == 4_000
    assert summary.prompt_count == 2


def test_multiple_sessions_roll_up_and_delete_history_preserves_calibration(
    tmp_path: Path,
) -> None:
    repository = SqliteRepository(tmp_path / "goodposture.sqlite3")
    repository.save_calibration(baseline(), saved_at_utc_ms=1_000)
    repository.save_session(
        session_id="run-a:1000",
        local_day="2026-07-26",
        aggregates=aggregates(ended_at_ms=11_000),
        updated_at_utc_ms=11_000,
    )
    repository.save_session(
        session_id="run-a:2000",
        local_day="2026-07-26",
        aggregates=aggregates(
            started_at_ms=2_000,
            ended_at_ms=8_000,
            eligible_monitoring_ms=6_000,
            unknown_ms=1_000,
            deviation_ms=2_000,
            prompt_count=0,
        ),
        updated_at_utc_ms=12_000,
    )

    summary = repository.daily_summary("2026-07-26")
    repository.delete_history()

    assert summary is not None
    assert summary.session_count == 2
    assert summary.eligible_monitoring_ms == 16_000
    assert repository.daily_summary("2026-07-26") is None
    assert repository.load_calibration() == baseline()


def test_delete_all_removes_settings_calibration_sessions_and_rollups(
    tmp_path: Path,
) -> None:
    repository = SqliteRepository(tmp_path / "goodposture.sqlite3")
    repository.save_calibration(baseline(), saved_at_utc_ms=1_000)
    repository.save_settings(LocalSettings(camera_index=1))
    repository.save_session(
        session_id="run-a:1000",
        local_day="2026-07-26",
        aggregates=aggregates(ended_at_ms=11_000),
        updated_at_utc_ms=11_000,
    )

    repository.delete_all()

    assert repository.load_calibration() is None
    assert repository.load_settings() == LocalSettings()
    assert repository.daily_summary("2026-07-26") is None


def test_version_one_database_migrates_without_losing_saved_calibration(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "goodposture.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            PRAGMA user_version = 1;
            CREATE TABLE settings (
                singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                camera_index INTEGER,
                companion_enabled INTEGER NOT NULL,
                notifications_enabled INTEGER NOT NULL
            );
            CREATE TABLE calibration (
                singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                baseline_json TEXT NOT NULL,
                saved_at_utc_ms INTEGER NOT NULL
            );
            """
        )
        connection.execute(
            """
            INSERT INTO calibration(singleton_id, baseline_json, saved_at_utc_ms)
            VALUES (1, ?, 1000)
            """,
            (SqliteRepository.serialize_calibration(baseline()),),
        )

    repository = SqliteRepository(database_path)

    assert repository.load_calibration() == baseline()
    assert repository.daily_summary("2026-07-26") is None
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == (
            SQLITE_SCHEMA_VERSION
        )


def test_version_two_history_migrates_with_zeroed_private_outcomes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "goodposture.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            PRAGMA user_version = 2;
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY NOT NULL,
                local_day TEXT NOT NULL,
                started_at_monotonic_ms INTEGER,
                ended_at_monotonic_ms INTEGER,
                eligible_monitoring_ms INTEGER NOT NULL,
                unknown_ms INTEGER NOT NULL,
                deviation_ms INTEGER NOT NULL,
                prompt_count INTEGER NOT NULL,
                updated_at_utc_ms INTEGER NOT NULL,
                finalized INTEGER NOT NULL
            );
            CREATE TABLE daily_aggregates (
                local_day TEXT PRIMARY KEY NOT NULL,
                session_count INTEGER NOT NULL,
                eligible_monitoring_ms INTEGER NOT NULL,
                unknown_ms INTEGER NOT NULL,
                deviation_ms INTEGER NOT NULL,
                prompt_count INTEGER NOT NULL,
                updated_at_utc_ms INTEGER NOT NULL
            );
            INSERT INTO daily_aggregates VALUES
                ('2026-07-26', 1, 3600000, 0, 600000, 2, 1000);
            """
        )

    repository = SqliteRepository(database_path)
    migrated = repository.daily_summary("2026-07-26")

    assert migrated is not None
    assert migrated.deviation_ms == 600_000
    assert migrated.prompt_count == 2
    assert migrated.post_prompt_recovered_ms == 0
    assert migrated.afternoon_deviation_ms == 0
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == (
            SQLITE_SCHEMA_VERSION
        )


def test_failed_session_write_rolls_back_existing_summary(
    tmp_path: Path,
) -> None:
    repository = SqliteRepository(tmp_path / "goodposture.sqlite3")
    repository.save_session(
        session_id="run-a:1000",
        local_day="2026-07-26",
        aggregates=aggregates(ended_at_ms=11_000),
        updated_at_utc_ms=11_000,
    )
    original = repository.daily_summary("2026-07-26")

    try:
        repository.save_session(
            session_id="",
            local_day="not-a-date",
            aggregates=aggregates(ended_at_ms=12_000),
            updated_at_utc_ms=12_000,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid session should have been rejected")

    assert repository.daily_summary("2026-07-26") == original


def test_aggregate_sink_checkpoints_one_session_id_and_rotates_for_next_session(
    tmp_path: Path,
) -> None:
    repository = SqliteRepository(tmp_path / "goodposture.sqlite3")
    sink = SqliteAggregateSink(
        repository,
        run_id="desktop-run",
        utc_now_ms=lambda: 50_000,
        local_day=lambda: "2026-07-26",
    )

    sink.flush(aggregates(eligible_monitoring_ms=5_000))
    sink.flush(aggregates(eligible_monitoring_ms=8_000, deviation_ms=2_000))
    sink.flush(
        aggregates(
            started_at_ms=20_000,
            ended_at_ms=25_000,
            eligible_monitoring_ms=5_000,
            unknown_ms=0,
            deviation_ms=1_000,
            prompt_count=0,
        )
    )

    summary = repository.daily_summary("2026-07-26")

    assert summary is not None
    assert summary.session_count == 2
    assert summary.eligible_monitoring_ms == 13_000


def test_sink_persists_recovered_time_and_coarse_slouch_daypart(
    tmp_path: Path,
) -> None:
    repository = SqliteRepository(tmp_path / "goodposture.sqlite3")
    sink = SqliteAggregateSink(
        repository,
        run_id="desktop-run",
        utc_now_ms=lambda: 50_000,
        local_day=lambda: "2026-07-26",
        local_hour=lambda: 14,
    )

    sink.flush(
        aggregates(
            deviation_ms=1_000,
            post_prompt_recovered_ms=2_000,
        )
    )
    sink.flush(
        aggregates(
            deviation_ms=3_000,
            post_prompt_recovered_ms=4_000,
        )
    )

    summary = repository.daily_summary("2026-07-26")

    assert summary is not None
    assert summary.post_prompt_recovered_ms == 4_000
    assert summary.afternoon_deviation_ms == 3_000
    assert summary.morning_deviation_ms == 0
    assert summary.evening_deviation_ms == 0
    assert summary.late_night_deviation_ms == 0
