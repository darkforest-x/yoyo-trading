"""The one structural rule, enforced instead of documented.

layers/ may not import each other. They talk through contracts/ and data/.

This is not tidiness. On 2026-08-03 a single fault spanned forward_scan, frozen
and executor because one layer's fact -- which coordinate system a model was
trained in -- was being decided by another layer's fact -- whether this trade is
long or short. Nothing stopped that import, so nothing stopped the bug, and it
reached the live path. A rule written in a README would not have caught it; this
does, at the moment someone types the import.

Deliberately AST-based rather than import-based: a module that is broken or slow
to import still gets checked, and the test does not need the whole dependency
tree installed to say something useful.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

YOYO = Path(__file__).resolve().parents[1] / "yoyo"
LAYERS = YOYO / "layers"
# What a layer is allowed to reach for. Anything else in yoyo.* is a violation.
ALLOWED_PREFIXES = ("yoyo.contracts", "yoyo.data")


def _layer_modules() -> list[tuple[str, Path]]:
    out = []
    if not LAYERS.exists():
        return out
    for layer_dir in sorted(p for p in LAYERS.iterdir() if p.is_dir()):
        for path in sorted(layer_dir.rglob("*.py")):
            out.append((layer_dir.name, path))
    return out


def _yoyo_imports(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):  # pragma: no cover - a broken file fails elsewhere
        return []
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("yoyo."):
                found.append(node.module)
        elif isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.startswith("yoyo.")]
    return found


@pytest.mark.parametrize(
    ("layer", "path"),
    _layer_modules() or [("<none yet>", Path(__file__))],
    ids=lambda v: v.name if isinstance(v, Path) else str(v),
)
def test_a_layer_does_not_import_another_layer(layer: str, path: Path) -> None:
    if layer == "<none yet>":
        pytest.skip("no layer modules migrated yet")
    offenders = []
    for module in _yoyo_imports(path):
        if module.startswith(f"yoyo.layers.{layer}"):
            continue  # its own layer is fine
        if module.startswith(ALLOWED_PREFIXES):
            continue
        if module.startswith("yoyo.layers."):
            offenders.append(module)
    assert not offenders, (
        f"{path.relative_to(YOYO.parent)} reaches into another layer: {offenders}. "
        "Route it through yoyo.contracts instead -- that is the whole point of the split."
    )


def test_contracts_depend_on_no_layer() -> None:
    """A contract that imports a layer is not a contract, it is a back door."""
    contracts = YOYO / "contracts"
    if not contracts.exists():
        pytest.skip("contracts not created yet")
    offenders = {
        str(p.relative_to(YOYO.parent)): [
            m for m in _yoyo_imports(p) if m.startswith("yoyo.layers")
        ]
        for p in contracts.rglob("*.py")
    }
    bad = {k: v for k, v in offenders.items() if v}
    assert not bad, f"contracts must not import layers: {bad}"


def test_the_rule_is_actually_testable(tmp_path: Path) -> None:
    """Guards the guard: prove the checker fires on a real violation.

    Without this, an empty layers/ directory would make the suite green and say
    nothing at all.
    """
    bad = tmp_path / "bad.py"
    bad.write_text("from yoyo.layers.l4_execution import executor\n", encoding="utf-8")
    assert _yoyo_imports(bad) == ["yoyo.layers.l4_execution"]
