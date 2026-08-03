"""One exact bundle decides what production runs. No discovery, no fallback.

The failure this exists to prevent is not hypothetical. frozen.py finds the newest
loadable JSON matching a glob, skips a corrupt one, and quietly serves an older
model; runtime never reads models/ACTIVE at all. That the pointer and the default
config happen to agree today is luck, not governance -- and on 2026-08-03 a
related gap did fire for real, with a short model served six sign-flipped
features because the extractor was chosen from trade side rather than from what
the model was trained on (analysis/p0_baseline_audit_20260803.md).

So a bundle names everything that has to agree, and every one of them is checked:
identity by sha256 of the actual files, semantics by explicit enum. A field that
is absent is an error rather than a default, because every default in this area
has a direction, and the wrong direction is what breaks serving.

Two invariants are enforced beyond field presence, because they cannot be true
together and both have already been assumed at some point:

  legacy_unaligned semantics can never be execution eligible -- that model was
  fitted in a different coordinate system, and "just fix the live features"
  is precisely the half-step that produced the 2026-08-03 fault

  paper_only and execution_eligible cannot both be true -- a bundle that says it
  is paper must not be reachable by the order path

This module does NOT activate anything. Presence of models/active_bundle.json is
the owner's switch. Production calls ``require_active_bundle`` and refuses to run
when the file is absent; research/audit callers may use ``load_active_bundle`` to
observe that absence. Refusing is not promotion and does not touch models/ACTIVE.

Takeover plan: docs/protocol_repair/P0_SAFETY_SPEC.md section 3, acceptance
C-01..C-08 and A-05/A-06.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from yoyo.contracts.outcomes import GAP_POLICIES, RETURN_CONVENTIONS
from yoyo.contracts.paths import data_path, data_root  # noqa: F401

# Resolved lazily. A default computed at import time freezes whatever the working
# directory happened to be when the module was first touched, which is precisely
# the class of bug that splitting code from data exposed.
ACTIVE_BUNDLE_NAME = "active_bundle.json"

SIDES = ("long", "short")
FEATURE_SEMANTICS = ("legacy_unaligned", "side_aligned_v1")
FEATURE_SCHEMAS = ("judgment_28_v1",)
MODEL_OBJECTIVES = ("regression", "binary")
THRESHOLD_OPERATORS = (">", ">=")
SAME_BAR_POLICIES = ("conservative_sl",)
CANDIDATE_SOURCES = ("yolo",)
TARGET_SEMANTICS = ("gross", "net_taker", "net_maker")
REPORTING_ROUTES = ("gross", "taker", "maker")
SELECTOR_STATUSES = ("calibrated", "abnormal_tie_mass_audit_only")
MAX_PRODUCTION_TIP_AGE_BARS = 2
MAX_SELECTOR_PASS_RATE_DEVIATION = 0.02
MAX_THRESHOLD_EQUAL_RATE = 0.02

# Every one is required. Absent is an error, not a default: see module docstring.
REQUIRED_FIELDS = (
    "bundle_version", "protocol_version", "strategy_id", "side", "timeframe",
    "window_bars", "candidate_source", "max_tip_age_bars",
    "feature_schema", "feature_semantics", "model_objective",
    "model_num_iteration", "score_semantics",
    "threshold", "threshold_operator", "tie_policy",
    "calibration_quantile", "calibration_pass_rate", "threshold_equal_rate",
    "selector_status",
    "research_entry_mode", "live_entry_mode",
    "tp_atr_mult", "sl_atr_mult", "horizon_bars", "same_bar_policy", "gap_policy",
    "return_convention", "cost_route", "target_ret_column", "target_semantics",
    "target_cost_included", "reporting_route",
    "detector_path", "detector_sha256",
    "model_path", "model_sha256",
    "dataset_path", "dataset_sha256",
    "execution_eligible", "paper_only",
)

# (field carrying the path, field carrying the digest)
HASHED_ARTEFACTS = (
    ("model_path", "model_sha256"),
    ("dataset_path", "dataset_sha256"),
    ("detector_path", "detector_sha256"),
)


def file_sha256(path: Path) -> str:
    """Digest of one artefact. Lives here rather than in a layer on purpose.

    A contract that has to import a layer to verify identity is not a contract --
    it is a back door, and tests/test_layer_boundaries.py rejects it.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BundleError(RuntimeError):
    """Raised for any bundle that cannot be trusted. Never downgraded to a warning."""

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


@dataclass(frozen=True)
class StrategyProtocol:
    """The whole contract, flat and typed. Construct only via load_bundle."""

    path: Path
    bundle_version: int
    protocol_version: str
    strategy_id: str
    side: str
    timeframe: str
    window_bars: int
    candidate_source: str
    max_tip_age_bars: int
    feature_schema: str
    feature_semantics: str
    model_objective: str
    model_num_iteration: int
    score_semantics: str
    threshold: float
    threshold_operator: str
    tie_policy: str
    calibration_quantile: float
    calibration_pass_rate: float
    threshold_equal_rate: float
    selector_status: str
    research_entry_mode: str
    live_entry_mode: str
    tp_atr_mult: float
    sl_atr_mult: float
    horizon_bars: int
    same_bar_policy: str
    gap_policy: str
    return_convention: str
    cost_route: str
    target_ret_column: str
    target_semantics: str
    target_cost_included: bool
    reporting_route: str
    detector_path: Path
    detector_sha256: str
    model_path: Path
    model_sha256: str
    dataset_path: Path
    dataset_sha256: str
    execution_eligible: bool
    paper_only: bool

    def passes_threshold(self, score: float) -> bool:
        """Apply the bundle's own operator. > and >= differ exactly where ties sit.

        The takeover plan reports the production q90 gate letting through about
        91.2% of val because scores tie on the boundary, so which operator a
        bundle declares is load-bearing, not cosmetic.
        """
        return score > self.threshold if self.threshold_operator == ">" else score >= self.threshold

    def accepts_row_side(self, row_side: object) -> bool:
        """Acceptance A-05: a row whose side disagrees with the strategy is a mismatch.

        Anything absent or unparseable is refused rather than assumed, matching
        executor.signal_trade_side since 2026-08-03.
        """
        if row_side is None:
            return False
        return str(row_side).strip().lower() == self.side


def _require(raw: Mapping[str, Any], path: Path) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        raise BundleError(path, f"missing required field(s): {', '.join(sorted(missing))}")


def _enum(raw: Mapping[str, Any], path: Path, field: str, allowed: tuple[str, ...]) -> str:
    value = str(raw[field]).strip()
    if value not in allowed:
        raise BundleError(path, f"{field}={value!r} not in {allowed}")
    return value


def _resolve(project_dir: Path, value: Any) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else project_dir / p


def load_bundle(path: Path, project_dir: Path | None = None) -> StrategyProtocol:
    """Load and fully verify one bundle. Any doubt raises; nothing is skipped.

    Deliberately has no sibling that "finds" a bundle. Discovery is the defect
    being removed, so the caller must name the file.
    """
    project_dir = data_root() if project_dir is None else Path(project_dir)
    path = Path(path)
    if not path.exists():
        raise BundleError(path, "bundle file does not exist")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BundleError(path, f"unreadable bundle: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise BundleError(path, "bundle must be a JSON object")

    _require(raw, path)
    side = _enum(raw, path, "side", SIDES)
    semantics = _enum(raw, path, "feature_semantics", FEATURE_SEMANTICS)
    feature_schema = _enum(raw, path, "feature_schema", FEATURE_SCHEMAS)
    model_objective = _enum(raw, path, "model_objective", MODEL_OBJECTIVES)
    operator = _enum(raw, path, "threshold_operator", THRESHOLD_OPERATORS)
    _enum(raw, path, "same_bar_policy", SAME_BAR_POLICIES)
    gap_policy = _enum(raw, path, "gap_policy", GAP_POLICIES)
    return_convention = _enum(raw, path, "return_convention", RETURN_CONVENTIONS)
    target_semantics = _enum(raw, path, "target_semantics", TARGET_SEMANTICS)
    reporting_route = _enum(raw, path, "reporting_route", REPORTING_ROUTES)
    selector_status = _enum(raw, path, "selector_status", SELECTOR_STATUSES)
    _enum(raw, path, "candidate_source", CANDIDATE_SOURCES)

    for field in ("execution_eligible", "paper_only", "target_cost_included"):
        if not isinstance(raw[field], bool):
            raise BundleError(path, f"{field} must be a JSON boolean, got {raw[field]!r}")
    eligible = bool(raw["execution_eligible"])
    paper_only = bool(raw["paper_only"])
    target_cost_included = bool(raw["target_cost_included"])

    if eligible and semantics == "legacy_unaligned":
        raise BundleError(
            path,
            "execution_eligible with feature_semantics=legacy_unaligned: that model "
            "was fitted in a different coordinate system; repairing the live "
            "extractor does not make it executable (plan D-03)",
        )
    if eligible and paper_only:
        raise BundleError(path, "execution_eligible and paper_only cannot both be true")
    if target_cost_included != (target_semantics != "gross"):
        raise BundleError(
            path,
            "target_cost_included contradicts target_semantics; gross excludes cost, "
            "net_taker/net_maker include it",
        )
    if side == "long" and return_convention != "linear_long":
        raise BundleError(path, "long side requires return_convention=linear_long")
    if side == "short" and return_convention == "linear_long":
        raise BundleError(path, "short side requires an explicit short return convention")

    try:
        numbers = {
            "bundle_version": int(raw["bundle_version"]),
            "window_bars": int(raw["window_bars"]),
            "max_tip_age_bars": int(raw["max_tip_age_bars"]),
            "horizon_bars": int(raw["horizon_bars"]),
            "model_num_iteration": int(raw["model_num_iteration"]),
            "calibration_quantile": float(raw["calibration_quantile"]),
            "calibration_pass_rate": float(raw["calibration_pass_rate"]),
            "threshold_equal_rate": float(raw["threshold_equal_rate"]),
            "threshold": float(raw["threshold"]),
            "tp_atr_mult": float(raw["tp_atr_mult"]),
            "sl_atr_mult": float(raw["sl_atr_mult"]),
        }
    except (TypeError, ValueError) as exc:
        raise BundleError(path, f"non-numeric field: {exc}") from exc
    finite_fields = (
        "threshold", "tp_atr_mult", "sl_atr_mult", "calibration_quantile",
        "calibration_pass_rate", "threshold_equal_rate",
    )
    if any(not math.isfinite(numbers[field]) for field in finite_fields):
        raise BundleError(path, "threshold, selector rates, and barrier multipliers must be finite")
    for field in ("bundle_version", "window_bars", "horizon_bars", "model_num_iteration"):
        if numbers[field] <= 0:
            raise BundleError(path, f"{field} must be positive")
    if numbers["max_tip_age_bars"] < 0:
        raise BundleError(path, "max_tip_age_bars must be non-negative")
    if numbers["max_tip_age_bars"] > MAX_PRODUCTION_TIP_AGE_BARS:
        raise BundleError(
            path,
            f"max_tip_age_bars may not exceed {MAX_PRODUCTION_TIP_AGE_BARS} in a production bundle",
        )
    quantile = numbers["calibration_quantile"]
    pass_rate = numbers["calibration_pass_rate"]
    equal_rate = numbers["threshold_equal_rate"]
    if not 0.0 < quantile < 1.0:
        raise BundleError(path, "calibration_quantile must be between zero and one")
    if not 0.0 <= pass_rate <= 1.0 or not 0.0 <= equal_rate <= 1.0:
        raise BundleError(path, "selector rates must be between zero and one")
    expected_pass_rate = 1.0 - quantile
    selector_abnormal = (
        abs(pass_rate - expected_pass_rate) > MAX_SELECTOR_PASS_RATE_DEVIATION
        or equal_rate > MAX_THRESHOLD_EQUAL_RATE
    )
    if eligible and selector_status != "calibrated":
        raise BundleError(path, "execution_eligible bundle requires selector_status=calibrated")
    if eligible and selector_abnormal:
        raise BundleError(
            path,
            "execution_eligible selector has abnormal pass/equality rates; recalibrate in P2",
        )

    for field in REQUIRED_FIELDS:
        if isinstance(raw[field], str) and not raw[field].strip():
            raise BundleError(path, f"{field} must not be blank")


    resolved: dict[str, Path] = {}
    for path_field, hash_field in HASHED_ARTEFACTS:
        target = _resolve(project_dir, raw[path_field])
        if not target.exists():
            raise BundleError(path, f"{path_field} does not exist: {target}")
        declared = str(raw[hash_field]).strip().lower()
        actual = file_sha256(target)
        if declared != actual:
            raise BundleError(
                path,
                f"{hash_field} mismatch for {target.name}: "
                f"declared {declared[:16]}… actual {actual[:16]}…",
            )
        resolved[path_field] = target

    return StrategyProtocol(
        path=path,
        protocol_version=str(raw["protocol_version"]),
        strategy_id=str(raw["strategy_id"]),
        side=side,
        timeframe=str(raw["timeframe"]),
        candidate_source=str(raw["candidate_source"]),
        feature_schema=feature_schema,
        feature_semantics=semantics,
        model_objective=model_objective,
        score_semantics=str(raw["score_semantics"]),
        threshold_operator=operator,
        tie_policy=str(raw["tie_policy"]),
        selector_status=selector_status,
        research_entry_mode=str(raw["research_entry_mode"]),
        live_entry_mode=str(raw["live_entry_mode"]),
        same_bar_policy=str(raw["same_bar_policy"]),
        gap_policy=gap_policy,
        return_convention=return_convention,
        cost_route=str(raw["cost_route"]),
        target_ret_column=str(raw["target_ret_column"]),
        target_semantics=target_semantics,
        target_cost_included=target_cost_included,
        reporting_route=reporting_route,
        detector_path=resolved["detector_path"],
        detector_sha256=str(raw["detector_sha256"]).strip().lower(),
        model_path=resolved["model_path"],
        model_sha256=str(raw["model_sha256"]).strip().lower(),
        dataset_path=resolved["dataset_path"],
        dataset_sha256=str(raw["dataset_sha256"]).strip().lower(),
        execution_eligible=eligible,
        paper_only=paper_only,
        **numbers,
    )


def load_active_bundle(project_dir: Path | None = None) -> StrategyProtocol | None:
    """Read the exact configured bundle, or report that none is configured.

    Audit/research code may need to distinguish absence from corruption. Production
    must call :func:`require_active_bundle`, because absence is not authority to
    discover a model elsewhere.
    """
    project_dir = data_root() if project_dir is None else Path(project_dir)
    bundle = project_dir / "models" / "active_bundle.json"
    if not bundle.exists():
        return None
    return load_bundle(bundle, project_dir=project_dir)


def require_active_bundle(project_dir: Path | None = None) -> StrategyProtocol:
    """Production authority: exactly one verified bundle, otherwise fail closed."""
    root = data_root() if project_dir is None else Path(project_dir)
    protocol = load_active_bundle(project_dir=root)
    if protocol is None:
        raise BundleError(
            root / "models" / "active_bundle.json",
            "production requires an explicit active bundle; models/ACTIVE and "
            "latest-artifact discovery are research/legacy authorities only",
        )
    return protocol


def runtime_artifact(protocol: StrategyProtocol):
    """Adapt a verified bundle to the existing scorer without reading a sidecar.

    The adapter deliberately takes threshold, side, dataset, feature semantics and
    inference iteration from the bundle itself. Loading the old JSON sidecar here
    would create a second authority and reintroduce C-07 through the back door.
    """
    from src.judgment.features import FEATURE_COLUMNS
    from src.judgment.frozen import FrozenArtifact, FrozenConfig

    config = FrozenConfig(
        name=protocol.protocol_version,
        project_dir=protocol.path.parent.parent,
        dataset_path=protocol.dataset_path,
        models_dir=protocol.model_path.parent,
        score_quantile=0.0,
        horizon_bars=protocol.horizon_bars,
        objective=protocol.model_objective,
        side=protocol.side,
    )
    return FrozenArtifact(
        config=config,
        model_path=protocol.model_path,
        metadata_path=protocol.path,
        dataset_path=protocol.dataset_path,
        relative_model_path=str(protocol.model_path),
        relative_dataset_path=str(protocol.dataset_path),
        threshold=protocol.threshold,
        feature_columns=tuple(FEATURE_COLUMNS),
        dataset_sha256=protocol.dataset_sha256,
        dataset_size_bytes=protocol.dataset_path.stat().st_size,
        best_iteration=protocol.model_num_iteration,
        sizing_tiers=None,
        feature_semantics=protocol.feature_semantics,
    )
