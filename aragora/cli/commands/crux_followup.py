"""CLI command: ``aragora crux-followup``.

Reads a CruxSet JSON file (or stdin) and emits DIC-17 FollowupProposals for
each load-bearing crux above the score threshold.  Default: dry-run (print
proposals, do not file anything).  Issue filing requires
``ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED=1`` and ``--file-issues``.

Advances: issue #6027 (DIC-17 — unresolved cruxes → bounded follow-up).
Live queue effect: none — read-only unless ``--file-issues`` with flag set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from aragora.epistemic.followup import (
    DEFAULT_CRUX_LOAD_BEARING_THRESHOLD,
    FollowupProposal,
    propose_followup_for_cruxset,
)
from aragora.reasoning.cruxset import CruxSet

_FLAG = "ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED"


def _followup_enabled() -> bool:
    return str(os.environ.get(_FLAG) or "").strip().lower() in {"1", "true", "yes", "on"}


def _load_cruxset(source: str | None) -> CruxSet | None:
    if source and source != "-":
        path = Path(source).expanduser()
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return None
        raw = path.read_text(encoding="utf-8")
    else:
        raw = sys.stdin.read()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: input is not JSON: {exc}", file=sys.stderr)
        return None

    try:
        return CruxSet.from_json(payload)
    except (KeyError, ValueError, TypeError) as exc:
        print(f"error: payload is not a CruxSet: {exc}", file=sys.stderr)
        return None


def _render_text(proposals: list[FollowupProposal]) -> str:
    if not proposals:
        return "No qualifying cruxes above threshold — no follow-up proposals."
    lines: list[str] = [f"{len(proposals)} follow-up proposal(s):"]
    for i, p in enumerate(proposals, 1):
        lines.extend(
            [
                f"\n[{i}] {p.title}",
                f"    source_key: {p.source_key}",
                f"    rationale:  {p.rationale}",
                f"    labels:     {', '.join(p.labels)}",
            ]
        )
    return "\n".join(lines)


def _proposal_to_dict(p: FollowupProposal, *, repo: str) -> dict:
    return {
        "source_key": p.source_key,
        "source_kind": p.source_kind,
        "title": p.title,
        "rationale": p.rationale,
        "labels": list(p.labels),
        "provenance": p.provenance,
        "gh_args": p.to_gh_create_args(repo=repo) if repo else [],
    }


def cmd_crux_followup(args: argparse.Namespace) -> int:
    """Generate DIC-17 follow-up proposals from a CruxSet JSON file."""
    cruxset = _load_cruxset(getattr(args, "cruxset_file", None))
    if cruxset is None:
        return 2

    threshold = float(getattr(args, "threshold", DEFAULT_CRUX_LOAD_BEARING_THRESHOLD))
    top_k = int(getattr(args, "top_k", 5))
    as_json = bool(getattr(args, "json", False))
    file_issues = bool(getattr(args, "file_issues", False))
    repo = str(getattr(args, "repo", "") or "")

    if file_issues and not _followup_enabled():
        print(
            f"error: --file-issues requires {_FLAG}=1\n"
            "       Use --dry-run (or omit --file-issues) to preview proposals without filing.",
            file=sys.stderr,
        )
        return 1

    proposals = propose_followup_for_cruxset(
        cruxset,
        top_k=top_k,
        load_bearing_threshold=threshold,
    )

    if as_json:
        print(json.dumps([_proposal_to_dict(p, repo=repo) for p in proposals], indent=2))
    else:
        print(_render_text(proposals))

    if file_issues and proposals:
        if not repo:
            print("error: --file-issues requires --repo owner/name", file=sys.stderr)
            return 1
        print(
            f"\nwould file {len(proposals)} issue(s) to {repo} "
            f"({_FLAG}=1; commands shown, not executed)"
        )
        for p in proposals:
            print(" ", "gh", " ".join(p.to_gh_create_args(repo=repo)))

    return 0


__all__ = ["cmd_crux_followup"]
