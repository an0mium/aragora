#!/usr/bin/env python3
"""Check that CANONICAL_GOALS.md's headline numbers match live repo state.

Implements TCP-1 from docs/plans/2026-04-17-trust-compound-plan.md. Each
claim in docs/status/claims/canonical_metrics.yaml delegates to this script
via the existing DIC-14 ClaimVerifier infrastructure.

Usage:
    python3 scripts/check_canonical_metrics.py --claim <claim_id>
    python3 scripts/check_canonical_metrics.py --all

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
    """Count `def test_` occurrences across tests/."""
    tests_dir = REPO_ROOT / "tests"
    if not tests_dir.is_dir():
        return 0
    pattern = re.compile(r"^\s*def test_", re.MULTILINE)
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


CHECKS: dict[str, Callable[[], ClaimCheck]] = {
    "canonical.km_adapters.count": _check_km_adapters_count,
    "canonical.python_modules.count": _check_python_modules_count,
    "canonical.test_definitions.count": _check_test_definitions_count,
    "canonical.version.matches_pyproject": _check_version_matches_pyproject,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify canonical metrics claims")
    parser.add_argument("--claim", help="Verify a single claim by id")
    parser.add_argument("--all", action="store_true", help="Verify every claim")
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

    payload = {
        "manifest_id": "canonical_metrics",
        "results": [asdict(r) for r in results],
        "summary": {
            "pass": sum(1 for r in results if r.status == "pass"),
            "warn": sum(1 for r in results if r.status == "warn"),
            "fail": sum(1 for r in results if r.status == "fail"),
        },
    }
    print(json.dumps(payload, sort_keys=True, indent=2))

    if args.write_receipt:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    if payload["summary"]["fail"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
