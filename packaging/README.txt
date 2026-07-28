GoodPosture 0.1.6 - Windows local pilot package

1. Extract the entire ZIP.
2. Exit any running GoodPosture instance from its system-tray menu.
3. Run Install-GoodPosture.ps1 in PowerShell. Do not continue unless it reports
   "GoodPosture 0.1.6 installed and verified".
4. Launch GoodPosture from the GoodPosture Start-menu shortcut.

The installer refuses to modify a running installation. It stages and verifies
the complete replacement before swapping it into place, preventing a mixed old
and new installation.

GoodPosture processes camera frames locally and does not retain raw images or
video. It stores calibration, settings, diagnostics event codes, and daily
aggregates under:
  %LOCALAPPDATA%\GoodPosture\GoodPosture

The uninstaller preserves that local data by default. To explicitly remove it:
  powershell.exe -ExecutionPolicy Bypass -File .\Uninstall-GoodPosture.ps1 -RemoveUserData

This pilot is a wellness awareness tool, not a medical device. It does not
diagnose, treat, prevent, or correct medical conditions.
