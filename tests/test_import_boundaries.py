from __future__ import annotations

import subprocess
import sys


def test_core_import_does_not_load_framework_or_adapter_dependencies() -> None:
    forbidden = ("cv2", "mediapipe", "sqlite3", "PySide6")
    script = (
        "import sys; import goodposture.core; "
        f"forbidden={forbidden!r}; "
        "loaded=[name for name in forbidden if name in sys.modules]; "
        "sys.exit(','.join(loaded) if loaded else 0)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
