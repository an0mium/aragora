#!/usr/bin/env python3
"""Check that CANONICAL_GOALS.md's headline numbers match live repo state.

Implements TCP-1 from docs/plans/2026-04-17-trust-compound-plan.md. Each
claim in docs/status/claims/canonical_metrics.yaml delegates to this script
via the existing DIC-14 ClaimVerifier infrastructure.

Usage:
    python3 scripts/check_canonical_metrics.py --claim <claim_id>
    python3 scripts/check_canonical_metrics.py --all
    python3 scripts/check_canonical_metrics.py --all --json

Exit codes:
    0  all requested claims pass (or are within tolerance)
    1  at least one claim drifted beyond tolerance
    2  usage error / unknown claim

Prints structured JSON per claim to stdout so the ClaimVerifier (or CI)
can capture the result.

Design notes:

- The script is intentionally small and uses only stdlib + regex so it
  runs on every push and on a schedule without pulling heavy deps.
- Each check is a pure function of (doc text, live repo state); no
  network calls, no LLM invocations, no flakiness sources.
- Tolerances are per-claim so fast-growing counts (tests, modules) can
  drift within a band without failing CI, while exact claims (version)
  must match strictly.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_GOALS = REPO_ROOT / "docs" / "CANONICAL_GOALS.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"
ADAPTERS_DIR = REPO_ROOT / "aragora" / "knowledge" / "mound" / "adapters"
OUTPUT_PATH = REPO_ROOT / "docs" / "status" / "generated" / "canonical_metrics" / "latest.json"

PRECOMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
MODEL_PINS = REPO_ROOT / "aragora" / "config" / "model_pins.py"
INCIDENT_LOG = REPO_ROOT / "benchmarks" / "bench_readiness" / "incident_2026-04-07_high-gravity.md"
ROTATION_SCHEDULE = REPO_ROOT / "benchmarks" / "bench_readiness" / "rotation-schedule.yaml"
ANTHROPIC_AGENT = REPO_ROOT / "aragora" / "agents" / "api_agents" / "anthropic.py"
OPENAI_AGENT = REPO_ROOT / "aragora" / "agents" / "api_agents" / "openai.py"
GEMINI_AGENT = REPO_ROOT / "aragora" / "agents" / "api_agents" / "gemini.py"

CRUX_DETECTOR = REPO_ROOT / "aragora" / "reasoning" / "crux_detector.py"
BELIEF_NETWORK = REPO_ROOT / "aragora" / "reasoning" / "belief.py"


@dataclass
class ClaimCheck:
    claim_id: str
    status: str  # "pass" | "fail" | "warn"
    claimed: str
    observed: str
    tolerance: str
    message: str


# ---------------------------------------------------------------------------
# Observers — recompute values from live state
# ---------------------------------------------------------------------------


def _observe_km_adapters_count() -> int:
    """Count KM adapter files under aragora/knowledge/mound/adapters/.

    Definition: any .py file in that directory whose name ends in
    ``_adapter.py`` OR which defines a class ending in ``Adapter`` that
    registers via the adapter factory. We use the filename heuristic
    because the factory import is heavy; the filename count is a
    reasonable first-pass approximation.
    """
    if not ADAPTERS_DIR.is_dir():
        return 0
    count = 0
    for child in ADAPTERS_DIR.iterdir():
        if child.is_file() and child.name.endswith("_adapter.py"):
            count += 1
    return count


def _observe_python_modules_count() -> int:
    """Count non-test Python files under aragora/."""
    aragora_dir = REPO_ROOT / "aragora"
    if not aragora_dir.is_dir():
        return 0
    count = 0
    for path in aragora_dir.rglob("*.py"):
        # Skip obvious non-module artifacts
        parts = path.relative_to(aragora_dir).parts
        if any(part.startswith(".") or part == "__pycache__" for part in parts):
            continue
        count += 1
    return count


def _observe_test_definitions_count() -> int:
    """Count ``def test_`` and ``async def test_`` occurrences across tests/.

    Mirrors the canonical method documented in ``docs/METRICS.md``:

        git grep -E '^[[:space:]]*(async )?def test_' -- tests | wc -l

    The earlier sync-only regex (``^\\s*def test_``) missed
    ``async def test_`` entries and caused the canonical-metrics check
    to falsely report stale-docs drift when the underlying issue was a
    counter bug. Including ``async def test_`` matches the method
    documented in METRICS.md and the count produced by pytest's
    collection on this repo.
    """

    tests_dir = REPO_ROOT / "tests"
    if not tests_dir.is_dir():
        return 0
    pattern = re.compile(r"^\s*(?:async )?def test_", re.MULTILINE)
    count = 0
    for path in tests_dir.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        count += len(pattern.findall(text))
    return count


def _observe_pyproject_version() -> str:
    try:
        text = PYPROJECT.read_text(encoding="utf-8")
    except OSError:
        return ""
    match = re.search(r'^\s*version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else ""


# ---------------------------------------------------------------------------
# Claim extractors — parse the value from CANONICAL_GOALS.md
# ---------------------------------------------------------------------------

_GOALS_TEXT_CACHE: str | None = None


def _goals_text() -> str:
    global _GOALS_TEXT_CACHE
    if _GOALS_TEXT_CACHE is None:
        _GOALS_TEXT_CACHE = CANONICAL_GOALS.read_text(encoding="utf-8")
    return _GOALS_TEXT_CACHE


def _claimed_km_adapter_count() -> int | None:
    """Parse 'Knowledge Mound adapters | <n> registered adapter specs'."""
    match = re.search(
        r"Knowledge Mound adapters\s*\|\s*(\d+)",
        _goals_text(),
    )
    return int(match.group(1)) if match else None


def _claimed_python_modules_count() -> int | None:
    """Parse 'Python modules | 3,800+' → 3800."""
    match = re.search(
        r"Python modules\s*\|\s*([\d,]+)\+?",
        _goals_text(),
    )
    if match is None:
        return None
    return int(match.group(1).replace(",", ""))


def _claimed_test_definitions_count() -> int | None:
    """Parse 'Automated tests | 210,000+' → 210000."""
    match = re.search(
        r"Automated tests\s*\|\s*([\d,]+)\+?",
        _goals_text(),
    )
    if match is None:
        return None
    return int(match.group(1).replace(",", ""))


def _claimed_version() -> str | None:
    match = re.search(
        r"Version\s*\|\s*([\d.]+)",
        _goals_text(),
    )
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Claim checks
# ---------------------------------------------------------------------------


def _check_km_adapters_count() -> ClaimCheck:
    claimed = _claimed_km_adapter_count()
    observed = _observe_km_adapters_count()
    if claimed is None:
        return ClaimCheck(
            claim_id="canonical.km_adapters.count",
            status="fail",
            claimed="<missing>",
            observed=str(observed),
            tolerance="exact",
            message="Could not parse adapter count from CANONICAL_GOALS.md",
        )
    # Tolerance of +/-2 because adapter-registration naming may
    # legitimately slip during refactors and the filename heuristic is
    # imperfect. Anything bigger than that is real drift.
    delta = abs(claimed - observed)
    if delta <= 2:
        return ClaimCheck(
            claim_id="canonical.km_adapters.count",
            status="pass",
            claimed=str(claimed),
            observed=str(observed),
            tolerance="+/-2",
            message=f"docs claim {claimed} adapters; live count is {observed}",
        )
    return ClaimCheck(
        claim_id="canonical.km_adapters.count",
        status="fail",
        claimed=str(claimed),
        observed=str(observed),
        tolerance="+/-2",
        message=(
            f"docs claim {claimed} adapters but live count is {observed} — "
            f"drift of {delta}. Update CANONICAL_GOALS.md or fix adapter registration."
        ),
    )


def _check_python_modules_count() -> ClaimCheck:
    claimed = _claimed_python_modules_count()
    observed = _observe_python_modules_count()
    if claimed is None:
        return ClaimCheck(
            claim_id="canonical.python_modules.count",
            status="fail",
            claimed="<missing>",
            observed=str(observed),
            tolerance="+/-20%",
            message="Could not parse module count from CANONICAL_GOALS.md",
        )
    # Modules grow naturally; tolerate +/-20% drift before flagging.
    # The doc uses "3,800+" notation, so observed >= claimed is fine.
    tolerance_band = max(int(claimed * 0.2), 100)
    if observed >= claimed - tolerance_band:
        return ClaimCheck(
            claim_id="canonical.python_modules.count",
            status="pass",
            claimed=f"{claimed}+",
            observed=str(observed),
            tolerance="+/-20%",
            message=f"docs claim {claimed}+ modules; live count is {observed} — within tolerance",
        )
    return ClaimCheck(
        claim_id="canonical.python_modules.count",
        status="warn",
        claimed=f"{claimed}+",
        observed=str(observed),
        tolerance="+/-20%",
        message=(
            f"docs claim {claimed}+ modules; live count is {observed}. "
            f"Refresh CANONICAL_GOALS.md when drift exceeds 20%."
        ),
    )


def _check_test_definitions_count() -> ClaimCheck:
    claimed = _claimed_test_definitions_count()
    observed = _observe_test_definitions_count()
    if claimed is None:
        return ClaimCheck(
            claim_id="canonical.test_definitions.count",
            status="fail",
            claimed="<missing>",
            observed=str(observed),
            tolerance="+/-20%",
            message="Could not parse test count from CANONICAL_GOALS.md",
        )
    tolerance_band = max(int(claimed * 0.2), 5000)
    if observed >= claimed - tolerance_band:
        return ClaimCheck(
            claim_id="canonical.test_definitions.count",
            status="pass",
            claimed=f"{claimed}+",
            observed=str(observed),
            tolerance="+/-20%",
            message=f"docs claim {claimed}+ tests; live count is {observed}",
        )
    return ClaimCheck(
        claim_id="canonical.test_definitions.count",
        status="warn",
        claimed=f"{claimed}+",
        observed=str(observed),
        tolerance="+/-20%",
        message=(
            f"docs claim {claimed}+ tests; live count is {observed}. "
            f"Refresh CANONICAL_GOALS.md when drift exceeds 20%."
        ),
    )


def _check_version_matches_pyproject() -> ClaimCheck:
    claimed = _claimed_version()
    observed = _observe_pyproject_version()
    if claimed is None:
        return ClaimCheck(
            claim_id="canonical.version.matches_pyproject",
            status="fail",
            claimed="<missing>",
            observed=observed or "<missing>",
            tolerance="exact",
            message="Could not parse version from CANONICAL_GOALS.md",
        )
    if observed == claimed:
        return ClaimCheck(
            claim_id="canonical.version.matches_pyproject",
            status="pass",
            claimed=claimed,
            observed=observed,
            tolerance="exact",
            message=f"docs and pyproject.toml both report version {claimed}",
        )
    return ClaimCheck(
        claim_id="canonical.version.matches_pyproject",
        status="fail",
        claimed=claimed,
        observed=observed or "<missing>",
        tolerance="exact",
        message=(
            f"version drift: CANONICAL_GOALS.md says {claimed!r} but "
            f"pyproject.toml says {observed!r}. Reconcile before release."
        ),
    )


# ---------------------------------------------------------------------------
# Security claim checks — added in Phase 14c, see docs/status/claims/canonical_metrics.yaml.
# These checks are tolerant of the underlying artefacts being missing (they
# report fail/warn rather than raising) so the manifest stays evaluable on any
# commit; missing artefacts are themselves drift to surface, not a script bug.
# ---------------------------------------------------------------------------


def _check_gitleaks_dual_stage() -> ClaimCheck:
    claim_id = "security.gitleaks.dual_stage"
    if not PRECOMMIT_CONFIG.is_file():
        return ClaimCheck(
            claim_id=claim_id,
            status="fail",
            claimed="gitleaks at [pre-commit, pre-push]",
            observed="<.pre-commit-config.yaml missing>",
            tolerance="exact",
            message=".pre-commit-config.yaml is missing — restore from PR #6194 (Droid foundation).",
        )
    text = PRECOMMIT_CONFIG.read_text(encoding="utf-8")
    # Match the gitleaks block and look for both stages within ~10 lines.
    pattern = re.compile(
        r"(?m)^\s*-\s*id:\s*gitleaks\b[\s\S]{0,400}?stages:\s*\[\s*pre-commit\s*,\s*pre-push\s*\]",
    )
    if pattern.search(text):
        return ClaimCheck(
            claim_id=claim_id,
            status="pass",
            claimed="gitleaks at [pre-commit, pre-push]",
            observed="gitleaks block declares both stages",
            tolerance="exact",
            message="gitleaks runs at both pre-commit and pre-push — bypass via --no-verify is caught at push time.",
        )
    return ClaimCheck(
        claim_id=claim_id,
        status="fail",
        claimed="gitleaks at [pre-commit, pre-push]",
        observed="gitleaks block missing one of the required stages",
        tolerance="exact",
        message=(
            "gitleaks is not configured for both pre-commit AND pre-push. "
            "This regresses the 2026-04-07 incident response — restore the dual-stage config."
        ),
    )


def _check_model_pins_frontier_aligned() -> ClaimCheck:
    claim_id = "security.model_pins.frontier_aligned"
    if not MODEL_PINS.is_file():
        return ClaimCheck(
            claim_id=claim_id,
            status="fail",
            claimed="OPUS_4_7, GPT_5_4, GEMINI_3_1_PRO exported",
            observed="<aragora/config/model_pins.py missing>",
            tolerance="exact",
            message="aragora/config/model_pins.py is missing — likely PR #6194 has not merged yet.",
        )
    text = MODEL_PINS.read_text(encoding="utf-8")
    required = ("OPUS_4_7", "GPT_5_4", "GEMINI_3_1_PRO")
    missing = [
        name for name in required if not re.search(rf"^\s*{name}\s*[:=]", text, re.MULTILINE)
    ]
    if not missing:
        return ClaimCheck(
            claim_id=claim_id,
            status="pass",
            claimed="OPUS_4_7, GPT_5_4, GEMINI_3_1_PRO exported",
            observed="all three frontier constants present",
            tolerance="exact",
            message="model_pins registry exports the three canonical frontier IDs.",
        )
    return ClaimCheck(
        claim_id=claim_id,
        status="fail",
        claimed="OPUS_4_7, GPT_5_4, GEMINI_3_1_PRO exported",
        observed=f"missing: {', '.join(missing)}",
        tolerance="exact",
        message=(
            f"model_pins.py is missing exports: {', '.join(missing)}. "
            "Restore them — consumers across 66+ files import from this single registry."
        ),
    )


def _check_incident_log_present() -> ClaimCheck:
    claim_id = "security.incident_log.present"
    missing = [p.name for p in (INCIDENT_LOG, ROTATION_SCHEDULE) if not p.is_file()]
    if not missing:
        return ClaimCheck(
            claim_id=claim_id,
            status="pass",
            claimed="incident log + rotation schedule present",
            observed="both files present",
            tolerance="exact",
            message="2026-04-07 incident write-up and 90-day rotation schedule are intact.",
        )
    return ClaimCheck(
        claim_id=claim_id,
        status="fail" if len(missing) == 2 else "warn",
        claimed="incident log + rotation schedule present",
        observed=f"missing: {', '.join(missing)}",
        tolerance="exact",
        message=(
            f"Audit-trail file(s) missing: {', '.join(missing)}. "
            "These document the 2026-04-07 Anthropic-key leak response and rotation cadence — do not delete."
        ),
    )


def _check_openrouter_fallback_wired() -> ClaimCheck:
    claim_id = "security.openrouter_fallback.wired"
    targets = {"anthropic": ANTHROPIC_AGENT, "openai": OPENAI_AGENT, "gemini": GEMINI_AGENT}
    missing_files = [name for name, path in targets.items() if not path.is_file()]
    if missing_files:
        return ClaimCheck(
            claim_id=claim_id,
            status="fail",
            claimed="QuotaFallbackMixin on anthropic/openai/gemini agents",
            observed=f"agent file(s) missing: {', '.join(missing_files)}",
            tolerance="exact",
            message=f"Provider agent file(s) missing — repo layout regression: {', '.join(missing_files)}.",
        )
    # Either QuotaFallbackMixin directly OR OpenAICompatibleMixin (which itself
    # inherits from QuotaFallbackMixin) in the class declaration line counts as
    # wired — both routes give the agent the OpenRouter quota-fallback path.
    pattern = re.compile(
        r"^class\s+\w+\s*\([^)]*(?:QuotaFallbackMixin|OpenAICompatibleMixin)",
        re.MULTILINE,
    )
    not_wired = [
        name
        for name, path in targets.items()
        if not pattern.search(path.read_text(encoding="utf-8"))
    ]
    if not not_wired:
        return ClaimCheck(
            claim_id=claim_id,
            status="pass",
            claimed="QuotaFallbackMixin on anthropic/openai/gemini agents",
            observed="all three frontier-provider agents inherit QuotaFallbackMixin",
            tolerance="exact",
            message="OpenRouter universal fallback is wired on all three frontier providers.",
        )
    return ClaimCheck(
        claim_id=claim_id,
        status="fail",
        claimed="QuotaFallbackMixin on anthropic/openai/gemini agents",
        observed=f"missing on: {', '.join(not_wired)}",
        tolerance="exact",
        message=(
            f"QuotaFallbackMixin not declared on: {', '.join(not_wired)}. "
            "Removing it re-introduces the 'missing direct key blocks debates' failure mode."
        ),
    )


# ---------------------------------------------------------------------------
# Proof-carrying claim checks — folded in from the Epistemic Runtime vision.
# These verify that the substrate required by DIC-13/14 (Epistemic CI),
# DIC-15 (Crux Engine), and DIC-23..28 (Dialectical Runtime synthesis) is
# importable with the public API those downstream plans depend on. Small,
# concrete, executable — not a new subsystem.
# ---------------------------------------------------------------------------


def _check_crux_detector_wired() -> ClaimCheck:
    claim_id = "proof_carrying.crux_detector.wired"
    if not CRUX_DETECTOR.is_file():
        return ClaimCheck(
            claim_id=claim_id,
            status="fail",
            claimed="CruxDetector + CruxClaim importable with detect_cruxes()",
            observed="<aragora/reasoning/crux_detector.py missing>",
            tolerance="exact",
            message="crux_detector.py missing — the Crux Engine (DIC-15) substrate is gone.",
        )
    text = CRUX_DETECTOR.read_text(encoding="utf-8")
    has_crux_claim = re.search(r"^class\s+CruxClaim\b", text, re.MULTILINE) is not None
    has_detector = re.search(r"^class\s+CruxDetector\b", text, re.MULTILINE) is not None
    has_detect_cruxes = (
        re.search(r"^\s*(?:async\s+)?def\s+detect_cruxes\b", text, re.MULTILINE) is not None
    )
    missing = []
    if not has_crux_claim:
        missing.append("CruxClaim")
    if not has_detector:
        missing.append("CruxDetector")
    if not has_detect_cruxes:
        missing.append("detect_cruxes()")
    if not missing:
        return ClaimCheck(
            claim_id=claim_id,
            status="pass",
            claimed="CruxDetector + CruxClaim importable with detect_cruxes()",
            observed="all three symbols present",
            tolerance="exact",
            message="Crux Engine substrate is wired — CruxDetector.detect_cruxes and CruxClaim dataclass both exist.",
        )
    return ClaimCheck(
        claim_id=claim_id,
        status="fail",
        claimed="CruxDetector + CruxClaim importable with detect_cruxes()",
        observed=f"missing: {', '.join(missing)}",
        tolerance="exact",
        message=(
            f"Crux substrate incomplete — missing: {', '.join(missing)}. "
            "DIC-15 (Crux Engine) and DIC-23..28 (Dialectical Runtime loop) both assume these."
        ),
    )


def _check_belief_network_wired() -> ClaimCheck:
    claim_id = "proof_carrying.belief_network.wired"
    if not BELIEF_NETWORK.is_file():
        return ClaimCheck(
            claim_id=claim_id,
            status="fail",
            claimed="BeliefNetwork, BeliefNode, BeliefStatus importable",
            observed="<aragora/reasoning/belief.py missing>",
            tolerance="exact",
            message="belief.py missing — provenance substrate for Epistemic CI + Dialectical Runtime is gone.",
        )
    text = BELIEF_NETWORK.read_text(encoding="utf-8")
    required = ("BeliefNetwork", "BeliefNode", "BeliefStatus")
    missing = [
        name for name in required if not re.search(rf"^class\s+{name}\b", text, re.MULTILINE)
    ]
    if not missing:
        return ClaimCheck(
            claim_id=claim_id,
            status="pass",
            claimed="BeliefNetwork, BeliefNode, BeliefStatus importable",
            observed="all three symbols present",
            tolerance="exact",
            message="Provenance substrate is wired — belief.py exports the three claim-graph primitives.",
        )
    return ClaimCheck(
        claim_id=claim_id,
        status="fail",
        claimed="BeliefNetwork, BeliefNode, BeliefStatus importable",
        observed=f"missing: {', '.join(missing)}",
        tolerance="exact",
        message=(
            f"belief.py missing: {', '.join(missing)}. "
            "DIC-13/14 ClaimVerifier + DIC-24 genealogy ledger both depend on this."
        ),
    )


CHECKS: dict[str, Callable[[], ClaimCheck]] = {
    "canonical.km_adapters.count": _check_km_adapters_count,
    "canonical.python_modules.count": _check_python_modules_count,
    "canonical.test_definitions.count": _check_test_definitions_count,
    "canonical.version.matches_pyproject": _check_version_matches_pyproject,
    "security.gitleaks.dual_stage": _check_gitleaks_dual_stage,
    "security.model_pins.frontier_aligned": _check_model_pins_frontier_aligned,
    "security.incident_log.present": _check_incident_log_present,
    "security.openrouter_fallback.wired": _check_openrouter_fallback_wired,
    "proof_carrying.crux_detector.wired": _check_crux_detector_wired,
    "proof_carrying.belief_network.wired": _check_belief_network_wired,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify canonical metrics claims")
    parser.add_argument("--claim", help="Verify a single claim by id")
    parser.add_argument("--all", action="store_true", help="Verify every claim")
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit the canonical metrics receipt JSON to stdout. This is the default "
            "output format; the flag exists for fan-out tooling compatibility."
        ),
    )
    parser.add_argument(
        "--write-receipt",
        action="store_true",
        help=("Also write a receipt file to docs/status/generated/canonical_metrics/latest.json"),
    )
    args = parser.parse_args()

    if not args.claim and not args.all:
        print("error: must pass --claim <id> or --all", file=sys.stderr)
        return 2

    if args.claim:
        if args.claim not in CHECKS:
            print(
                f"error: unknown claim {args.claim!r}; known claims: {', '.join(sorted(CHECKS))}",
                file=sys.stderr,
            )
            return 2
        results = [CHECKS[args.claim]()]
    else:
        results = [check() for check in CHECKS.values()]

    summary = {
        "pass": sum(1 for r in results if r.status == "pass"),
        "warn": sum(1 for r in results if r.status == "warn"),
        "fail": sum(1 for r in results if r.status == "fail"),
    }
    payload = {
        "manifest_id": "canonical_metrics",
        "results": [asdict(r) for r in results],
        "summary": summary,
    }
    print(json.dumps(payload, sort_keys=True, indent=2))

    if args.write_receipt:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    if summary["fail"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
