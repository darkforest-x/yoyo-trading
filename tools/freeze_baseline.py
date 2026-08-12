#!/usr/bin/env python3
"""Freeze the R1/R2 comparison baseline before any R3 arm is trained.

Without this, "R3 is better" has nothing to be better than: the V5 brief makes
the freeze a precondition (section 8.1), and the numbers it pins down -- weight
SHAs, the training recipe, static val metrics, and the continuous canary
contract -- are exactly the ones a later run could otherwise drift against
without anyone noticing.

Everything is read out of fable-trading at freeze time from the pointer file in
``configs/frozen_baseline_r1_r2_v1.json``; this module carries no metric values
of its own. A missing path is an error, never a silently skipped field, because
a baseline with holes is worse than no baseline.

Read-only with respect to the source repo: it opens files there and writes only
under ``manifests/`` in this repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CONFIG = ROOT / "configs/source_repo.json"
DEFAULT_SPEC = ROOT / "configs/frozen_baseline_r1_r2_v1.json"
DEFAULT_OUT = ROOT / "manifests/frozen_baseline_r1_r2_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"frozen baseline needs {what}, missing: {path}")
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar_yaml(path: Path, keys: list[str]) -> dict[str, Any]:
    """Pull flat ``key: value`` scalars out of an ultralytics args.yaml.

    A three-line reader instead of PyYAML: args.yaml is flat, and pyproject
    deliberately declares no dependencies (iron rule 12).
    """
    wanted = set(keys)
    found: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line or line.startswith((" ", "\t", "#")):
            continue
        key, _, raw = line.partition(":")
        if key.strip() not in wanted:
            continue
        value: Any = raw.strip()
        if value in {"null", ""}:
            value = None
        elif value in {"true", "false"}:
            value = value == "true"
        else:
            try:
                value = int(value) if value.isdigit() else float(value)
            except ValueError:
                value = value.strip("'\"")
        found[key.strip()] = value
    missing = wanted - set(found)
    if missing:
        raise KeyError(f"{path} is missing recipe keys: {sorted(missing)}")
    return found


REQUIRED_METRICS = ("precision", "recall", "map50", "map50_95")


def static_metrics(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    """Accept both val-metric shapes written by this project.

    The Mac re-validation runs wrap their numbers in ``metrics``; the earlier
    baseline run wrote them flat. Guessing wrong here would freeze an empty
    baseline, so unwrap explicitly and insist the four headline metrics exist.
    """
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else payload
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing:
        raise KeyError(f"{path} has no {missing} to freeze")
    return {key: metrics[key] for key in metrics if isinstance(metrics[key], (int, float))}


def dataset_facts(root: Path) -> dict[str, Any]:
    facts: dict[str, Any] = {"path": str(root), "data_yaml_sha256": None, "manifests": {}}
    data_yaml = require(root / "data.yaml", f"data.yaml of {root.name}")
    facts["data_yaml_sha256"] = sha256_file(data_yaml)
    for name in ("positive_manifest", "negative_manifest", "hard_negative_manifest"):
        path = root / f"{name}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        splits: dict[str, int] = {}
        lengths: dict[str, int] = {}
        for row in rows:
            splits[str(row.get("split"))] = splits.get(str(row.get("split")), 0) + 1
            lengths[str(row.get("win_len"))] = lengths.get(str(row.get("win_len")), 0) + 1
        facts["manifests"][name] = {
            "sha256": sha256_file(path),
            "rows": len(rows),
            "by_split": dict(sorted(splits.items())),
            "by_win_len": dict(sorted(lengths.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else -1)),
        }
    return facts


def canary_facts(path: Path, contract_keys: list[str]) -> dict[str, Any]:
    summary = read_json(path)
    facts = {key: summary.get(key) for key in contract_keys}
    facts.update(
        {
            "weights_sha256": summary.get("weights_sha256"),
            "scanned_symbols": len(summary.get("scanned_symbols", [])),
            "raw_detections": summary.get("raw_detections"),
            "deduplicated_events": summary.get("deduplicated_events"),
            "summary_path": str(path),
            "summary_sha256": sha256_file(path),
        }
    )
    return facts


def freeze(source_config: Path, spec_path: Path, out_path: Path) -> dict[str, Any]:
    source = read_json(source_config)
    spec = read_json(spec_path)
    repo = Path(source["source_repo"])
    require(repo, "the fable-trading source repo")

    weights: dict[str, Any] = {}
    for name, rel in spec["weights"].items():
        path = require(repo / rel, f"{name} weights")
        weights[name] = {"path": rel, "sha256": sha256_file(path), "bytes": path.stat().st_size}

    recipe_path = require(repo / spec["training_recipe_source"], "the R1 training args.yaml")
    recipe = scalar_yaml(recipe_path, spec["recipe_keys"])
    augmentation_off = all(
        recipe[key] in (0, 0.0, False, None)
        for key in (
            "hsv_h", "hsv_s", "hsv_v", "degrees", "shear", "perspective",
            "flipud", "fliplr", "mosaic", "mixup", "cutmix", "copy_paste", "erasing",
        )
    )

    static: dict[str, Any] = {}
    for name, rel in spec["static_metrics"].items():
        path = require(repo / rel, f"{name} static val metrics")
        payload = read_json(path)
        static[name] = {
            "weights_sha256": payload.get("weights_sha256"),
            "metrics": static_metrics(payload, path),
            "evaluation_scope": payload.get("evaluation_scope"),
            "data_yaml_sha256": payload.get("data_yaml_sha256"),
            "path": rel,
        }

    datasets = {
        name: dataset_facts(require(repo / rel, f"dataset {name}"))
        for name, rel in spec["datasets"].items()
    }

    canary: dict[str, Any] = {}
    for snapshot, models in spec["continuous_canary"].items():
        canary[snapshot] = {
            model: canary_facts(require(repo / rel, f"canary {snapshot}/{model}"), spec["canary_contract_keys"])
            for model, rel in models.items()
        }
    for snapshot, models in canary.items():
        contracts = {
            model: json.dumps([facts[key] for key in spec["canary_contract_keys"]], sort_keys=True)
            for model, facts in models.items()
        }
        if len(set(contracts.values())) > 1:
            raise ValueError(
                f"canary snapshot {snapshot} compares models under different contracts: {contracts}"
            )

    gold = require(repo / spec["gold_geometry_source"], "the owner gold geometry sheet")

    review_packs: dict[str, Any] = {}
    for name, rel in spec["owner_review_packs"].items():
        path = repo / rel
        if not path.exists():
            review_packs[name] = {"path": rel, "present": False}
            continue
        payload = read_json(path)
        counts = payload.get("counts") or {}
        overall = payload.get("overall") or {}
        review_packs[name] = {
            "path": rel,
            "present": True,
            "sha256": sha256_file(path),
            "rows": payload.get("rows") or overall.get("reviewed"),
            "owner_yes": overall.get("YES", counts.get("target")),
            "owner_no": overall.get("NO", counts.get("hard_negative")),
            "protocol": payload.get("protocol"),
        }

    frozen = {
        "protocol": spec["protocol"],
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "detection_class": spec["detection_class"],
        "direction_scope": spec["direction_scope"],
        "window_protocol": spec["window_protocol"],
        "repos": {
            "fable_trading": {"path": str(repo), "commit": git_commit(repo)},
            "yoyo_trading": {"path": str(ROOT), "commit": git_commit(ROOT)},
        },
        "holdout_start": source["holdout_start"],
        "holdout_read": False,
        "weights": weights,
        "training_recipe": {
            "source": spec["training_recipe_source"],
            "values": recipe,
            "augmentation_fully_disabled": augmentation_off,
        },
        "static_val_metrics": static,
        "datasets": datasets,
        "gold_geometry_source": {"path": spec["gold_geometry_source"], "sha256": sha256_file(gold)},
        "continuous_canary": canary,
        "owner_review_packs": review_packs,
        "config_sha256": {
            "source_repo": sha256_text(source_config.read_text(encoding="utf-8")),
            "spec": sha256_text(spec_path.read_text(encoding="utf-8")),
        },
        "promoted": False,
        "orders_placed": False,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    frozen["self_sha256"] = sha256_file(out_path)
    return frozen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    frozen = freeze(args.source_config, args.spec, args.out)
    print(json.dumps({k: v for k, v in frozen.items() if k != "datasets"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
