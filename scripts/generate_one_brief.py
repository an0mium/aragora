#!/usr/bin/env python3
"""Generate a single Mode 3 PDB brief for one PR from the command line.

Skips the web UI and server entirely — runs the full
``run_protocol_b`` pipeline in-process, writes the resulting brief to
``.aragora/review-queue/briefs/pr-{N}-{sha}.json`` via the same
storage layer the server uses, and prints a summary.

Primary use case: dogfooding the Mode 3 pipeline without starting
``aragora serve``. Useful for founder-facing end-to-end validation
of the first real heterogeneous-panel brief.

Requires (as env vars):
    ANTHROPIC_API_KEY    — for claude_core slot
    OPENAI_API_KEY       — for gpt_core slot
    ARAGORA_PDB_BRIEF_GENERATION_ENABLED=1  — feature flag

Optional:
    ARAGORA_PDB_PANEL_ID (default: protocol_b_default)

Exit codes:
    0  brief generated successfully
    1  input loader failure (PR not found, gh CLI issue)
    2  provider not configured (missing API keys for required slots)
    3  budget or execution failure
    4  brief generation disabled (feature flag off)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Offline-friendly — don't reach out to Secrets Manager during a local run.
os.environ.setdefault("ARAGORA_USE_SECRETS_MANAGER", "false")

from aragora.pdb import storage
from aragora.pdb.brief_state import BriefLifecycleState
from aragora.pdb.input_loader import (
    InputLoaderError,
    InputLoaderErrorReason,
    load_execution_input,
)
from aragora.pdb.protocol import (
    PDBExecutionResult,
    PDBExecutionStatus,
    run_protocol_b,
)


FEATURE_FLAG = "ARAGORA_PDB_BRIEF_GENERATION_ENABLED"


def _feature_enabled() -> bool:
    return os.environ.get(FEATURE_FLAG, "").strip() in {"1", "true", "yes"}


def _build_invoker():
    """Import and construct the default provider invoker.

    Imported lazily because the invoker factory ships in PR #6404
    (``aragora/pdb/invoker_factory.py``). If it isn't available, we
    print a clear error and exit 2.
    """
    try:
        from aragora.pdb.invoker_factory import build_default_invoker
    except ImportError:
        print(
            "error: aragora.pdb.invoker_factory not found. "
            "This CLI requires the ProviderInvoker module landed in "
            "#6404 (codex/pdb-real-invoker-phase-a). Pull latest main.",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        return build_default_invoker()
    except Exception as exc:
        print(f"error: provider invoker setup failed: {exc}", file=sys.stderr)
        print(
            "Ensure ANTHROPIC_API_KEY and OPENAI_API_KEY are set in env.",
            file=sys.stderr,
        )
        sys.exit(2)


def _parse_repo(value: str) -> str:
    if "/" not in value or value.count("/") != 1:
        raise argparse.ArgumentTypeError(f"invalid repo {value!r}; expected 'owner/name'")
    return value


def _summarize_result(result: PDBExecutionResult, *, quiet: bool) -> None:
    brief = result.brief
    print(f"status:        {result.status.value}")
    print(f"active roster: {', '.join(result.active_roster) or '(none)'}")
    if result.missing_slots:
        print(f"missing slots: {', '.join(result.missing_slots)}")
    if result.degrade_reasons:
        print(f"degraded:      {'; '.join(result.degrade_reasons)}")
    print(f"cost (USD):    ${result.actual_cost_usd:.4f}")

    if brief is None:
        if result.failure_reason:
            print(f"failure:       {result.failure_reason}")
        return

    print(f"\nverdict:       {brief.recommendation.value}")
    print(f"confidence:    {brief.overall_confidence}/5")
    print(f"disagreement:  {brief.disagreement_score:.2f}")
    print(f"top line:      {brief.top_line}")
    if brief.dissent:
        print(f"\ndissent ({len(brief.dissent)} view(s)):")
        for view in brief.dissent:
            print(f"  - {view.slot_id}: {view.position.value} ({view.confidence}/5)")
            print(f"    {view.reason[:200]}")

    if quiet:
        return

    print("\nrole findings:")
    for rf in brief.role_findings[:3]:
        print(f"  - {rf.role} ({rf.agent}): {rf.summary[:200]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a single Mode 3 PDB brief for one PR.")
    parser.add_argument(
        "repo",
        type=_parse_repo,
        help="GitHub repo as owner/name (e.g. synaptent/aragora)",
    )
    parser.add_argument("pr_number", type=int, help="PR number")
    parser.add_argument(
        "--panel-id",
        default=os.environ.get("ARAGORA_PDB_PANEL_ID", "protocol_b_default"),
        help="Panel config id (default: protocol_b_default)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only verdict + cost; skip role-findings preview",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Run the panel but don't write to .aragora/review-queue/briefs/",
    )
    args = parser.parse_args()

    if not _feature_enabled():
        print(
            f"error: {FEATURE_FLAG}=1 must be set in env before running.",
            file=sys.stderr,
        )
        return 4

    print(f"loading PR input for {args.repo}#{args.pr_number}...")
    try:
        loaded = load_execution_input(
            pr_number=args.pr_number,
            repo=args.repo,
            panel_id=args.panel_id,
        )
    except InputLoaderError as exc:
        print(f"error: input loader failed: {exc.reason.value}: {exc}", file=sys.stderr)
        if exc.reason is InputLoaderErrorReason.GH_MISSING:
            print("hint: install gh CLI + `gh auth login`.", file=sys.stderr)
        return 1

    print(f"head SHA:      {loaded.head_sha}")
    print(f"panel:         {args.panel_id}")

    invoker = _build_invoker()

    print("\nrunning protocol B (findings → critique → synthesis)...")
    t0 = time.monotonic()
    try:
        result = run_protocol_b(input=loaded.execution_input, invoker=invoker)
    except Exception as exc:
        print(f"error: execution failed: {exc}", file=sys.stderr)
        return 3
    elapsed = time.monotonic() - t0

    print(f"execution:     {elapsed:.1f}s wall-clock")
    _summarize_result(result, quiet=args.quiet)

    if result.status is not PDBExecutionStatus.READY or result.brief is None:
        return 3

    if not args.no_persist:
        brief_dict = json.loads(
            json.dumps(result.brief, default=lambda o: getattr(o, "__dict__", str(o)))
        )
        storage.mark_ready(
            pr_number=args.pr_number,
            head_sha=loaded.head_sha,
            brief_json=brief_dict,
            signature="cli-local-run",  # local dogfood; not cryptographically signed
        )
        storage.append_index_event(
            pr_number=args.pr_number,
            head_sha=loaded.head_sha,
            event_type="pdb_brief_generated",
            fields={
                "source": "scripts/generate_one_brief.py",
                "cost_usd": result.actual_cost_usd,
                "wall_clock_ms": int(elapsed * 1000),
            },
        )
        ready_path = storage._ready_path(args.pr_number, loaded.head_sha)
        print(f"\nbrief saved:   {ready_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
