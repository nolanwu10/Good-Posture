from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_windows_build_is_reproducible_onedir_and_runs_offline_smoke() -> None:
    script = (ROOT / "scripts" / "build_windows_package.ps1").read_text(
        encoding="utf-8"
    )

    assert "pyinstaller" in script.lower()
    assert "--onedir" in script
    assert "--windowed" in script
    assert "--contents-directory" in script
    assert "package-smoke" in script
    assert "59929E1D1EE95287735DDD833B19CF4AC46D29BC7AFDDBBF6753C459690D574A" in script
    assert "Compress-Archive" in script


def test_installer_never_enables_startup_and_uninstaller_preserves_data_by_default() -> None:
    installer = (ROOT / "packaging" / "Install-GoodPosture.ps1").read_text(
        encoding="utf-8"
    )
    uninstaller = (ROOT / "packaging" / "Uninstall-GoodPosture.ps1").read_text(
        encoding="utf-8"
    )

    assert "CurrentVersion\\Run" not in installer
    assert "[switch]$RemoveUserData" in uninstaller
    assert "if ($RemoveUserData)" in uninstaller
    assert "CurrentVersion\\Run" in uninstaller
    assert "GoodPosture\\GoodPosture" in uninstaller


def test_installer_refuses_running_upgrade_and_verifies_release_identity() -> None:
    installer = (ROOT / "packaging" / "Install-GoodPosture.ps1").read_text(
        encoding="utf-8"
    )
    build_script = (ROOT / "scripts" / "build_windows_package.ps1").read_text(
        encoding="utf-8"
    )

    assert "release-manifest.json" in installer
    assert "Get-Process" in installer
    assert "Exit GoodPosture from the system tray" in installer
    assert "Get-FileHash" in installer
    assert "appExecutableSha256" in installer
    assert "Move-Item" in installer
    assert "DisplayVersion -Value $releaseManifest.version" in installer

    assert "release-manifest.json" in build_script
    assert "appExecutableSha256" in build_script


@pytest.mark.skipif(os.name != "nt", reason="Windows installer integration")
def test_installer_stages_and_verifies_complete_replacement(tmp_path: Path) -> None:
    release_directory = tmp_path / "GoodPosture-9.8.7-windows-x64"
    app_directory = release_directory / "App"
    app_directory.mkdir(parents=True)
    executable_bytes = b"verified replacement executable"
    (app_directory / "GoodPosture.exe").write_bytes(executable_bytes)
    shutil.copy2(
        ROOT / "packaging" / "Install-GoodPosture.ps1",
        release_directory / "Install-GoodPosture.ps1",
    )
    shutil.copy2(
        ROOT / "packaging" / "Uninstall-GoodPosture.ps1",
        release_directory / "Uninstall-GoodPosture.ps1",
    )
    (release_directory / "release-manifest.json").write_text(
        json.dumps(
            {
                "version": "9.8.7",
                "appExecutableSha256": hashlib.sha256(executable_bytes).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    install_directory = tmp_path / "installed" / "GoodPosture"
    install_directory.mkdir(parents=True)
    (install_directory / "stale-version.txt").write_text("0.1.0", encoding="utf-8")
    start_menu_directory = tmp_path / "StartMenu"

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(release_directory / "Install-GoodPosture.ps1"),
            "-InstallDirectory",
            str(install_directory),
            "-StartMenuDirectory",
            str(start_menu_directory),
            "-SkipRegistry",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "GoodPosture 9.8.7 installed and verified" in result.stdout
    assert (install_directory / "GoodPosture.exe").read_bytes() == executable_bytes
    assert (install_directory / "release-manifest.json").is_file()
    assert not (install_directory / "stale-version.txt").exists()
    assert (start_menu_directory / "GoodPosture.lnk").is_file()
