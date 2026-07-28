"""Explicit opt-in Windows startup registration."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence

_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_VALUE_NAME = "GoodPosture"


class WindowsStartupSetting:
    """Manage one current-user Run entry; construction never enables it."""

    def __init__(self, command: Sequence[str]) -> None:
        if not command or any(not item for item in command):
            raise ValueError("startup command items must be non-empty")
        self._command = subprocess.list2cmdline(list(command))

    @property
    def available(self) -> bool:
        return sys.platform == "win32"

    def is_enabled(self) -> bool:
        if not self.available:
            return False
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as key:
                value, _ = winreg.QueryValueEx(key, _VALUE_NAME)
        except FileNotFoundError:
            return False
        return bool(value == self._command)

    def set_enabled(self, enabled: bool) -> None:
        if not self.available:
            raise RuntimeError("Windows startup registration is unavailable")
        import winreg

        if enabled:
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as key:
                winreg.SetValueEx(key, _VALUE_NAME, 0, winreg.REG_SZ, self._command)
            return
        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                _RUN_KEY,
                0,
                winreg.KEY_SET_VALUE,
            ) as key:
                winreg.DeleteValue(key, _VALUE_NAME)
        except FileNotFoundError:
            return
