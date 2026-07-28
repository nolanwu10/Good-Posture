from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "goodposture"


def test_production_code_has_no_raw_media_write_api() -> None:
    forbidden_calls = {"VideoWriter", "imwrite", "imencode"}
    findings: list[str] = []

    for path in SOURCE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else function.id
                if isinstance(function, ast.Name)
                else None
            )
            if name in forbidden_calls:
                findings.append(f"{path.relative_to(ROOT)}:{node.lineno}:{name}")

    assert findings == []


def test_realtime_application_has_no_network_client_imports() -> None:
    forbidden_roots = {
        "aiohttp",
        "boto3",
        "botocore",
        "httpx",
        "openai",
        "requests",
        "socket",
        "urllib",
    }
    findings: list[str] = []

    for path in SOURCE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = (node.module,)
            for name in names:
                if name.split(".", 1)[0] in forbidden_roots:
                    findings.append(f"{path.relative_to(ROOT)}:{node.lineno}:{name}")

    assert findings == []


def test_repository_contains_no_raw_camera_fixture_files() -> None:
    allowed_generated_art = {
        "src/goodposture/assets/illustrations/posture-slouch.png",
        "src/goodposture/assets/illustrations/posture-upright.png",
    }
    forbidden_extensions = {
        ".avi",
        ".bmp",
        ".gif",
        ".jpeg",
        ".jpg",
        ".mkv",
        ".mov",
        ".mp4",
        ".png",
        ".webm",
    }

    findings = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in forbidden_extensions
        and path.relative_to(ROOT).as_posix() not in allowed_generated_art
        and ".venv" not in path.parts
        and ".git" not in path.parts
        and "tmp" not in path.parts
        and not any(part == "build" or part.startswith("build-") for part in path.parts)
        and not any(part == "dist" or part.startswith("dist-") for part in path.parts)
    ]

    assert findings == []
