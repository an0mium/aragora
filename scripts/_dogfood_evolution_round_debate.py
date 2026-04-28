"""Dogfood Phase 1 — run an Arena debate to select the next bounded PR.

Read-only against the codebase. Persists artifacts to docs/plans/.
Not a real CLI command; lives under scripts/ as a one-shot helper for
the 2026-04-28 evolution round.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

# Load .env so XAI_API_KEY / GEMINI_API_KEY etc. are visible.
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


CANDIDATES = [
    {
        "id": "A_agent_readable_receipts",
        "title": "Agent-readable receipts envelope (Worker A, P8)",
        "loc": 250,
        "pillar": "P8 + P3",
        "summary": (
            "Add a structured machine-parseable receipt envelope alongside "
            "the existing human-readable one. Aragora's own agents become "
            "able to consume Aragora's outputs."
        ),
    },
    {
        "id": "B_crux_receipt_unification",
        "title": "CruxReceipt unification bridge (Worker B, dialectical)",
        "loc": 250,
        "pillar": "P5 + dialectical thesis",
        "summary": (
            "CruxReceipt is split between gauntlet runner and epistemic "
            "engine with two diverging dataclass shapes. A bridge module "
            "canonicalizes the shape and lets both sides emit and consume "
            "the same receipt."
        ),
    },
    {
        "id": "C_backlog_audit_consolidation",
        "title": "Backlog-audit classifier consolidation (Worker C, P7)",
        "loc": 250,
        "pillar": "P7",
        "summary": (
            "The backlog-audit classifier has been patched 20 times in 4 "
            "weeks. Consolidate into a single function with explicit "
            "invariants, paralleling handoff_contract."
        ),
    },
    {
        "id": "D_handoff_contract_migration",
        "title": "Migrate one legacy script to handoff_contract (#6785 follow-up)",
        "loc": 200,
        "pillar": "P5",
        "summary": (
            "Migrate the read-only archive-satisfied path of "
            "audit_codex_branch_backlog.py to delegate to "
            "aragora.swarm.handoff_contract. Validates #6785 against real "
            "code, no behavior change."
        ),
    },
    {
        "id": "E_p3_bridge_inspector",
        "title": "Bridge Run Inspector first PR (P3 dark-pillar exit)",
        "loc": 350,
        "pillar": "P3",
        "summary": (
            "Initial CLI route + JSON renderer for `aragora bridge inspect`. "
            "Exits P3 from dark-pillar status."
        ),
    },
    {
        "id": "F_km_bridge_wiring",
        "title": "Wire KMMetricsHealthBridge into postgres-store factory",
        "loc": 150,
        "pillar": "P4",
        "summary": (
            "Wire the (now-shipped) KMMetricsHealthBridge into the "
            "postgres-store factory plus add `aragora km status` CLI. "
            "Closes P4 from spec to live."
        ),
    },
]


TASK = """\
Aragora is in Profile-3 polish week. Operator review queue already contains 4 \
overnight-filed PRs: #6784 (KM resilience tests), #6785 (handoff_contract \
skeleton), #6786 (KMMetricsHealthBridge), #6787 (overnight docs). All BLOCKED \
by pre-existing main CI red, not by regressions. Carve-outs: Codex owns \
parser/automation, Codex Desktop paused, Claude owns docs and red-CI watch, \
no new red-CI patches, no automation orchestration script edits beyond bounded \
delegation, outbox stays clean.

You are choosing the SINGLE next bounded PR (≤300 LOC, additive only) that \
will be filed in the next slot. Six candidates, each summarized below. Each \
proposal is reversible by closing its branch.

You must select EXACTLY ONE candidate by id. Justify on three axes: \
(1) thesis alignment (evolution / dialectical / Aragora-as-its-own-user); \
(2) coordination risk (does it cross any carve-out?); \
(3) measurable improvement on a Pillar (P1..P8).

Candidates:

%s

Output your final answer as a single line beginning with "WINNER:" followed \
by the candidate id.
""" % "\n\n".join(
    f"- id={c['id']}\n  title: {c['title']}\n  loc: {c['loc']}\n  pillar: {c['pillar']}\n  summary: {c['summary']}"
    for c in CANDIDATES
)


async def run_debate() -> dict:
    from aragora.agents import create_agent
    from aragora.core import Environment
    from aragora.debate import Arena, DebateProtocol

    agents = []
    agent_specs = []

    # Note: GEMINI_API_KEY observed expired in this environment. Skipping.
    if os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY"):
        try:
            agents.append(
                create_agent(
                    "grok",
                    name="grok_proposer",
                    role="proposer",
                    model="grok-4-latest",
                )
            )
            agent_specs.append({"type": "grok", "role": "proposer", "model": "grok-4-latest"})
        except Exception as e:  # noqa: BLE001
            print(f"WARN: grok proposer failed: {e}", file=sys.stderr)

        try:
            agents.append(
                create_agent(
                    "grok",
                    name="grok_critic",
                    role="critic",
                    model="grok-4-latest",
                )
            )
            agent_specs.append({"type": "grok", "role": "critic", "model": "grok-4-latest"})
        except Exception as e:  # noqa: BLE001
            print(f"WARN: grok critic failed: {e}", file=sys.stderr)

    # Always include demo as offline judge.
    try:
        agents.append(create_agent("demo", name="demo_judge", role="synthesizer"))
        agent_specs.append({"type": "demo", "role": "synthesizer"})
    except Exception as e:  # noqa: BLE001
        print(f"WARN: demo agent failed: {e}", file=sys.stderr)

    if len(agents) < 2:
        raise RuntimeError(f"Insufficient agents available ({len(agents)}); cannot run debate")

    env = Environment(
        task=TASK,
        max_rounds=2,
        roles=["proposer", "critic", "synthesizer"],
    )
    protocol = DebateProtocol(rounds=2, consensus="majority")

    arena = Arena(
        environment=env,
        agents=agents,
        protocol=protocol,
        memory=None,
    )
    started_at = datetime.now(UTC)
    result = await arena.run()
    finished_at = datetime.now(UTC)

    return {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_s": (finished_at - started_at).total_seconds(),
        "agents": agent_specs,
        "candidates": CANDIDATES,
        "task": TASK,
        "winner_text": _extract_winner(result),
        "consensus": _safe_attr(result, "consensus", default=None),
        "final_answer": _safe_attr(result, "final_answer", default=None)
        or _safe_attr(result, "answer", default=None),
        "rounds_completed": _safe_attr(result, "rounds_completed", default=None),
        "vote_counts": _safe_attr(result, "vote_counts", default=None),
    }


def _safe_attr(obj, name, default=None):
    try:
        v = getattr(obj, name, default)
        if hasattr(v, "to_dict"):
            return v.to_dict()
        if hasattr(v, "__dict__"):
            return {k: str(val) for k, val in v.__dict__.items() if not k.startswith("_")}
        return v
    except Exception:  # noqa: BLE001
        return default


def _extract_winner(result) -> str | None:
    text_candidates = []
    for attr in ("final_answer", "answer", "consensus_text", "winning_proposal"):
        v = getattr(result, attr, None)
        if isinstance(v, str):
            text_candidates.append(v)

    proposals = getattr(result, "proposals", None) or {}
    if isinstance(proposals, dict):
        text_candidates.extend(str(v) for v in proposals.values() if v)

    for txt in text_candidates:
        for line in txt.splitlines():
            line = line.strip()
            if line.upper().startswith("WINNER:"):
                return line[len("WINNER:") :].strip()
    return None


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "plans"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "2026-04-28-evolution-round-debate-receipt.json"
    out_md = out_dir / "2026-04-28-evolution-round-debate-summary.md"

    try:
        receipt = asyncio.run(run_debate())
        receipt["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        receipt = {
            "status": "error",
            "error_class": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "candidates": CANDIDATES,
            "fallback_winner": "B_crux_receipt_unification",
        }

    out_json.write_text(json.dumps(receipt, indent=2, default=str))
    print(f"wrote: {out_json}")

    summary_lines = [
        "# 2026-04-28 Evolution Round — Debate Summary",
        "",
        f"**Status:** {receipt.get('status')}",
        "",
    ]
    if receipt.get("status") == "ok":
        summary_lines += [
            f"**Duration:** {receipt.get('duration_s'):.1f}s",
            f"**Agents:** {receipt.get('agents')}",
            f"**Winner extracted:** {receipt.get('winner_text')}",
            f"**Consensus:** {receipt.get('consensus')}",
            "",
            "## Final answer",
            "",
            "```",
            str(receipt.get("final_answer") or "(no final_answer attribute)")[:4000],
            "```",
            "",
        ]
    else:
        summary_lines += [
            f"**Error class:** {receipt.get('error_class')}",
            f"**Error message:** {receipt.get('error_message')}",
            f"**Fallback winner:** {receipt.get('fallback_winner')}",
            "",
            "## Traceback (truncated)",
            "",
            "```",
            (receipt.get("traceback") or "")[:2000],
            "```",
            "",
        ]
    out_md.write_text("\n".join(summary_lines))
    print(f"wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
