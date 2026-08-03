"""Tell yoyo where the data lives before anything imports a module that needs it.

yoyo holds code; the data, models and artefacts stay in the fable-trading tree.
Several modules resolve real paths at import time (log locations, key files), so
the root has to be set before collection rather than inside a fixture.

Also pins the native import order: lightgbm and torch each ship an OpenMP runtime
and on macOS loading lightgbm's first kills the process (exit 139). Measured, not
guessed -- see fable-trading's tests/conftest.py for the evidence.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_ROOT = Path.home() / "fable-trading"


def pytest_configure(config) -> None:  # noqa: ARG001
    os.environ.setdefault("YOYO_DATA_ROOT", str(DEFAULT_DATA_ROOT))
    try:
        import torchvision  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
