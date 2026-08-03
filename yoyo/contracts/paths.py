"""Where the data lives. Never inferred from where the code lives.

Until 2026-08-03 eight modules computed the project root as
``Path(__file__).parents[2]``, which was right only because code and data sat in
the same repository. Splitting yoyo out of fable-trading made every one of them
point at a directory containing no data/ and no models/ -- and the failure mode was
the bad kind, quiet: "bundle file does not exist", "keys not found", as if the
artefacts were missing rather than the root being wrong.

So the root is resolved explicitly and, when it cannot be, the error says which
question failed rather than letting a caller act on an empty directory.

Resolution order:
  1. $YOYO_DATA_ROOT -- explicit wins, always
  2. the nearest ancestor of the working directory that holds data/ or models/
  3. raise

Deliberately no fallback to the package directory. That fallback is the bug.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "YOYO_DATA_ROOT"
MARKERS = ("data", "models")


class DataRootError(RuntimeError):
    """Raised when the data root cannot be established. Never downgraded."""


def data_root(start: Path | None = None) -> Path:
    """Absolute path of the tree holding data/, models/, analysis/ and the rest.

    `start` is for tests; production passes nothing and gets cwd-based discovery.
    """
    explicit = os.environ.get(ENV_VAR, "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise DataRootError(f"{ENV_VAR}={explicit!r} is not a directory")
        return path

    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if any((candidate / m).is_dir() for m in MARKERS):
            return candidate
    raise DataRootError(
        f"cannot locate the data root from {here}: no ancestor contains "
        f"{' or '.join(MARKERS)}/. Set {ENV_VAR} to the tree that does "
        f"(for this project, the fable-trading checkout)."
    )


def data_path(*parts: str) -> Path:
    """A path under the data root. Existence is the caller's business."""
    return data_root().joinpath(*parts)
