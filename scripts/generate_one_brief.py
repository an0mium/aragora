#!/usr/bin/env python3
"""Generate a single Mode 3 PDB brief for one PR from the command line.

Skips the web UI and server entirely — runs the full
``run_protocol_b`` pipeline in-process, writes the resulting brief to
``.aragora/review-queue/briefs/pr-{N}-{sha}.json`` via the same
storage layer the server uses, and prints a summary.

Primary use case: dogfooding the Mode 3 pipeline without starting
``aragora serve``. Useful for founder-facing end-to-end validation
of the first real heterogeneous-panel brief.

## Credential sources (pick one)

**AWS Secrets Manager** (recommended — keys never in shell history):

    # One-time: store keys in AWS Secrets Manager, configure AWS creds:
    aws secretsmanager create-secret --name aragora/ANTHROPIC_API_KEY --secret-string "..."
    aws secretsmanager create-secret --name aragora/OPENAI_API_KEY --secret-string "..."
    aws configure  # or SSO/IAM role

    # Each run:
    export ARAGORA_USE_SECRETS_MANAGER=true
    export ARAGORA_PDB_BRIEF_GENERATION_ENABLED=1
    python scripts/generate_one_brief.py synaptent/aragora <pr>

The CLI calls ``hydrate_env_from_secrets`` before building the invoker,
which fetches the keys from Secrets Manager and sets them on the
process's env (not exported to the parent shell). Keys never touch
shell history or disk outside the OS's credential store for
``aws configure``.

**macOS Keychain** (simplest for a laptop):

    security add-generic-password -a "$USER" -s "aragora-anthropic-api-key" -w
    security add-generic-password -a "$USER" -s "aragora-openai-api-key" -w

    export ANTHROPIC_API_KEY=$(security find-generic-password -a "$USER" -s "aragora-anthropic-api-key" -w)
    export OPENAI_API_KEY=$(security find-generic-password -a "$USER" -s "aragora-openai-api-key" -w)
    export ARAGORA_PDB_BRIEF_GENERATION_ENABLED=1

**Plain env vars** (dev-only; keys visible in shell history):

    export ANTHROPIC_API_KEY=... OPENAI_API_KEY=...
    export ARAGORA_PDB_BRIEF_GENERATION_ENABLED=1

## Optional slots (heterodox + regulatory panel)

Set any of these in env (or Secrets Manager under the same name) to
expand the panel beyond the core Claude + GPT roster:

    GEMINI_API_KEY, GROK_API_KEY (or XAI_API_KEY),
    OPENROUTER_API_KEY, MISTRAL_API_KEY

## Optional config

    ARAGORA_PDB_PANEL_ID (default: protocol_b_default)
    ARAGORA_USE_SECRETS_MANAGER=true  (to enable Secrets Manager path)

## Exit codes

    0  brief generated successfully
    1  input loader failure (PR not found, gh CLI issue)
    2  provider not configured (missing API keys for required slots)
    3  budget or execution failure
    4  brief generation disabled (feature flag off)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Hydrate credentials from the configured source BEFORE importing aragora
# modules that snapshot env at import time. ``hydrate_env_from_secrets`` is
# best-effort: if ``ARAGORA_USE_SECRETS_MANAGER`` is unset or false, this is
# a no-op and plain env vars are used. If it IS set, the keys get pulled
# from AWS Secrets Manager into the process's env (not the parent shell's).
_CREDENTIAL_KEYS = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GROK_API_KEY",
    "XAI_API_KEY",
    "OPENROUTER_API_KEY",
    "MISTRAL_API_KEY",
]
try:
    from aragora.config.secrets import hydrate_env_from_secrets

    hydrate_env_from_secrets(_CREDENTIAL_KEYS)
except Exception:  # noqa: BLE001 — never block CLI startup on hydration issues
    # If Secrets Manager isn't reachable, the CLI still works with plain env
    # vars set in the parent shell. _build_invoker() will surface a clear
    # error if no keys are available for the required core slots.
    logging.getLogger(__name__).debug("hydrate_env_from_secrets unavailable; falling back to env")

from aragora.pdb import storage
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
    print(f"confidence:    {_format_confidence(brief.overall_confidence)}")
    print(f"disagreement:  {brief.disagreement_score:.2f}")
    print(f"top line:      {brief.top_line}")
    if brief.dissent:
        print(f"\ndissent ({len(brief.dissent)} view(s)):")
        for view in brief.dissent:
            position = getattr(view.position, "value", view.position)
            agent = getattr(view, "agent", getattr(view, "slot_id", "?"))
            print(f"  - {agent}: {position}")
            reason = getattr(view, "reason", "")
            if reason:
                print(f"    {reason[:200]}")

    if quiet:
        return

    print("\nrole findings:")
    for rf in brief.role_findings[:3]:
        role_name = getattr(rf.role, "value", rf.role)
        text = getattr(rf, "finding_text", "") or getattr(rf, "summary", "")
        print(f"  - {role_name} ({rf.agent}): {text[:200]}")


def _format_confidence(raw: float | int, *, include_raw: bool = True) -> str:
    """Return a ``n/5``-shaped string for a confidence value.

    Briefs carry confidence as a float in the ``0.0..1.0`` range; the
    previous CLI printed the raw float next to ``/5`` which made it
    look like a 5-point scale (e.g. ``0.82/5``). This helper buckets
    the float into a 1..5 integer and, when ``include_raw`` is true,
    appends the original float in parentheses so the underlying score
    remains visible. Values outside the unit range are printed
    verbatim as ``<raw>/5`` to surface unexpected inputs rather than
    silently clip them.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return f"{raw}/5"
    if 0.0 <= value <= 1.0:
        bucket = max(1, min(5, round(value * 5)))
        if include_raw:
            return f"{bucket}/5 (raw={value:.2f})"
        return f"{bucket}/5"
    return f"{raw}/5"


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
        result = run_protocol_b(input=loaded.input, invoker=invoker)
    except Exception as exc:
        print(f"error: execution failed: {exc}", file=sys.stderr)
        return 3
    elapsed = time.monotonic() - t0

    print(f"execution:     {elapsed:.1f}s wall-clock")
    _summarize_result(result, quiet=args.quiet)

    successful_statuses = (PDBExecutionStatus.SUCCESS, PDBExecutionStatus.DEGRADED)
    if result.status not in successful_statuses or result.brief is None:
        return 3

    if not args.no_persist:
        try:
            ready_path = storage.persist_ready_from_executor(
                result,
                pr_number=args.pr_number,
                head_sha=loaded.head_sha,
                source="scripts/generate_one_brief.py",
                signature="cli-local-run",
                cost_usd=result.actual_cost_usd,
                wall_clock_ms=int(elapsed * 1000),
            )
            print(f"\nbrief saved:   {ready_path}")
        except Exception as exc:  # noqa: BLE001 — surface but don't abort
            print(f"warning: brief display OK but persist failed: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
