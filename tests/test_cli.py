from __future__ import annotations

import sys
from pathlib import Path

from goodposture.cli import build_parser, default_model_path, main
from goodposture.evaluation import DEFAULT_EVALUATION_CORPUS_PATH


def test_prototype_defaults_to_local_model_and_primary_camera() -> None:
    args = build_parser().parse_args(["prototype"])

    assert args.model == Path("models/pose_landmarker_lite.task")
    assert args.camera == 0
    assert args.no_mirror is False


def test_prototype_accepts_camera_and_mirroring_overrides() -> None:
    args = build_parser().parse_args(
        ["prototype", "--camera", "2", "--no-mirror", "--model", "other.task"]
    )

    assert args.model == Path("other.task")
    assert args.camera == 2
    assert args.no_mirror is True


def test_calibrate_defaults_to_pinned_local_model() -> None:
    args = build_parser().parse_args(["calibrate"])

    assert args.model == Path("models/pose_landmarker_lite.task")


def test_desktop_defaults_to_pinned_local_model() -> None:
    args = build_parser().parse_args(["desktop"])

    assert args.model == Path("models/pose_landmarker_lite.task")


def test_evaluate_defaults_to_versioned_local_corpus() -> None:
    args = build_parser().parse_args(["evaluate"])

    assert args.corpus == Path("evaluation/corpus-v1.json")
    assert args.output == Path("evaluation/reports")


def test_evaluate_command_runs_local_protocol(tmp_path: Path) -> None:
    exit_code = main(
        [
            "evaluate",
            "--corpus",
            str(DEFAULT_EVALUATION_CORPUS_PATH),
            "--output",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "comparison.json").is_file()


def test_frozen_default_model_path_is_relative_to_packaged_executable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "GoodPosture.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))

    assert default_model_path() == tmp_path / "models" / "pose_landmarker_lite.task"


def test_package_smoke_parser_uses_packaged_model_default() -> None:
    args = build_parser().parse_args(["package-smoke"])

    assert args.model == default_model_path()
