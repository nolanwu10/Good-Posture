"""Command-line entry points for local GoodPosture development."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path


def default_model_path() -> Path:
    """Resolve the bundled model beside a frozen executable when packaged."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "models" / "pose_landmarker_lite.task"
    return Path("models/pose_landmarker_lite.task")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="goodposture",
        description="Local-first posture-awareness feasibility tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prototype = subparsers.add_parser(
        "prototype",
        help="Open the local webcam landmark and posture-metrics viewer.",
    )
    prototype.add_argument(
        "--model",
        type=Path,
        default=default_model_path(),
        help="Path to a local MediaPipe Pose Landmarker .task model.",
    )
    prototype.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0).",
    )
    prototype.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not mirror the webcam image before local inference.",
    )
    calibrate = subparsers.add_parser(
        "calibrate",
        help="Open the local first-run privacy and calibration flow.",
    )
    calibrate.add_argument(
        "--model",
        type=Path,
        default=default_model_path(),
        help="Path to the checksum-pinned local Pose Landmarker model.",
    )
    desktop = subparsers.add_parser(
        "desktop",
        help="Run consent, calibration, and the local system-tray experience.",
    )
    desktop.add_argument(
        "--model",
        type=Path,
        default=default_model_path(),
        help="Path to the checksum-pinned local Pose Landmarker model.",
    )
    evaluate = subparsers.add_parser(
        "evaluate",
        help="Replay the versioned local evaluation corpus and write aggregate reports.",
    )
    evaluate.add_argument(
        "--corpus",
        type=Path,
        default=Path("evaluation/corpus-v1.json"),
        help="Path to a synthetic or explicitly consented derived corpus.",
    )
    evaluate.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/reports"),
        help="Directory for aggregate JSON and Markdown reports.",
    )
    package_smoke = subparsers.add_parser("package-smoke", help=argparse.SUPPRESS)
    package_smoke.add_argument(
        "--model",
        type=Path,
        default=default_model_path(),
        help=argparse.SUPPRESS,
    )
    pilot = subparsers.add_parser(
        "pilot-record",
        help="Append one aggregate-only local lived-use record.",
    )
    pilot.add_argument("--date", type=date.fromisoformat, required=True)
    pilot.add_argument("--monitoring-minutes", type=int, required=True)
    pilot.add_argument("--peak-cpu-percent", type=float, required=True)
    pilot.add_argument("--peak-memory-mb", type=float, required=True)
    pilot.add_argument("--false-prompts", type=int, required=True)
    pilot.add_argument("--missed-prompts", type=int, required=True)
    pilot.add_argument("--failures", type=int, required=True)
    pilot.add_argument("--usability-rating", type=int, required=True)
    pilot.add_argument("--output", type=Path, required=True)
    pilot_report = subparsers.add_parser(
        "pilot-report",
        help="Summarize aggregate-only local lived-use records.",
    )
    pilot_report.add_argument("--input", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prototype":
        from goodposture.prototype import PrototypeOptions, run_prototype

        try:
            run_prototype(
                PrototypeOptions(
                    model_path=args.model,
                    camera_index=args.camera,
                    mirror=not args.no_mirror,
                )
            )
        except (FileNotFoundError, RuntimeError) as error:
            print(f"GoodPosture could not start: {error}", file=sys.stderr)
            return 2
    elif args.command == "calibrate":
        from goodposture.ui.qt_calibration import run_calibration_ui

        return run_calibration_ui(args.model)
    elif args.command == "desktop":
        from goodposture.ui.desktop import run_desktop

        return run_desktop(args.model)
    elif args.command == "evaluate":
        from goodposture.evaluation import run_evaluation_protocol

        try:
            result = run_evaluation_protocol(
                corpus_path=args.corpus,
                output_directory=args.output,
            )
        except (OSError, ValueError) as error:
            print(f"GoodPosture evaluation could not run: {error}", file=sys.stderr)
            return 2
        print(f"Evaluation reports written to {args.output}")
        if not result.comparison.passed:
            print("Evaluation regression gate failed.", file=sys.stderr)
            return 1
    elif args.command == "package-smoke":
        from goodposture.package_smoke import run_package_smoke

        return run_package_smoke(args.model)
    elif args.command == "pilot-record":
        from goodposture.pilot import PilotEntry, append_pilot_entry

        try:
            append_pilot_entry(
                args.output,
                PilotEntry(
                    local_day=args.date,
                    monitoring_minutes=args.monitoring_minutes,
                    peak_cpu_percent=args.peak_cpu_percent,
                    peak_memory_mb=args.peak_memory_mb,
                    false_prompt_count=args.false_prompts,
                    missed_prompt_count=args.missed_prompts,
                    failure_count=args.failures,
                    usability_rating=args.usability_rating,
                ),
            )
        except (OSError, ValueError) as error:
            print(f"Pilot record was not saved: {error}", file=sys.stderr)
            return 2
    elif args.command == "pilot-report":
        from goodposture.pilot import summarize_pilot

        try:
            print(json.dumps(asdict(summarize_pilot(args.input)), sort_keys=True))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"Pilot report could not be read: {error}", file=sys.stderr)
            return 2
    return 0
