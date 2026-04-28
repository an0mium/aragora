"""Dogfood Phase 2 — run the gauntlet against the winning candidate.

Reads the debate receipt, takes the winner, runs an adversarial
gauntlet against its proposal text, persists a gauntlet receipt to
docs/plans/.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

# Load .env
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


PROPOSAL_TEXT = """\
Aragora bounded next-PR proposal: Agent-Readable Receipts Envelope.

Goal: Aragora generates receipts (decision receipts, gauntlet receipts,
debate result receipts) for human inspection, but Aragora's own agents
cannot reliably parse them. This violates the Aragora-as-its-own-user
thesis (P8 parity gap surfaced by Worker A).

Scope (≤300 LOC, additive only):

1. New module aragora/receipts/agent_envelope.py introducing:
   - @dataclass AgentReceiptEnvelope { schema_version, receipt_kind,
     summary_text, structured_payload, metadata }
   - to_dict() / from_dict() with strict schema validation
   - Helper: wrap_human_receipt(human_text, structured_payload, kind)

2. Tests in tests/receipts/test_agent_envelope.py (~12-15 tests):
   - to_dict / from_dict roundtrip
   - structured_payload type contract
   - schema_version mismatch handling
   - reject InvalidEnvelope for malformed input

3. NO changes to existing receipt emitters in this PR. Migration of
   one or two emitters is a follow-up PR per the spec sequencing.
   The envelope is consumed only by the test fixture in this PR.

Constraints honored:
- ≤300 LOC additive only.
- No CI workflow patches.
- No automation orchestration script edits.
- No production receipt store writes.
- No callers wired in this PR.
- Operator review required, no auto-merge.
"""


def _agent_factory(name: str):
    """Map gauntlet agent-name strings to live agents.

    Falls back to grok for any name when only XAI_API_KEY is available.
    """
    from aragora.agents import create_agent

    return create_agent("grok", name=name, role="critic", model="grok-4-latest")


async def run_gauntlet() -> dict:
    from aragora.gauntlet.runner import GauntletRunner
    from aragora.gauntlet.config import GauntletConfig, AttackCategory, ProbeCategory

    # Use grok-only agent list since that's the only working API key.
    config = GauntletConfig(
        name="Evolution Round Gauntlet — Agent-Readable Receipts",
        domain="aragora-bounded-pr",
        attack_categories=[
            AttackCategory.LOGIC,
            AttackCategory.ARCHITECTURE,
            AttackCategory.OPERATIONAL,
        ],
        attack_rounds=1,
        attacks_per_category=2,
        probe_categories=[
            ProbeCategory.CONTRADICTION,
            ProbeCategory.HALLUCINATION,
        ],
        probes_per_category=1,
        max_total_probes=4,
        run_scenario_matrix=False,  # keep budget tight
        enable_scenario_analysis=False,
        enable_adversarial_probing=True,
        enable_formal_verification=False,
        enable_deep_audit=False,
        agents=["grok"],
        max_agents=1,
        timeout_seconds=180,
        attack_timeout=30,
        probe_timeout=20,
        sign_receipt=False,  # offline only
    )

    runner = GauntletRunner(config=config, agent_factory=_agent_factory)
    started = datetime.now(UTC)
    result = await runner.run(
        input_content=PROPOSAL_TEXT,
        context=(
            "Context: Aragora is in Profile-3 polish week. This proposal "
            "is the winning candidate from a parallel debate. The PR is "
            "<=300 LOC, additive only, no caller migration. Operator "
            "review required, no auto-merge."
        ),
    )
    finished = datetime.now(UTC)

    return {
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_s": (finished - started).total_seconds(),
        "config_used": result.config_used,
        "verdict": getattr(result, "verdict", None),
        "robustness_score": getattr(result, "robustness_score", None),
        "consensus_score": getattr(result, "consensus_score", None),
        "findings_count": len(getattr(result, "findings", [])),
        "findings": [
            {
                "severity": str(f.severity) if hasattr(f, "severity") else str(f.get("severity")),
                "category": str(f.category) if hasattr(f, "category") else str(f.get("category")),
                "title": getattr(f, "title", None) or f.get("title", "")
                if hasattr(f, "get")
                else getattr(f, "title", "?"),
                "description": (
                    getattr(f, "description", None)
                    or (f.get("description", "") if hasattr(f, "get") else "")
                )[:500],
            }
            for f in getattr(result, "findings", [])
        ],
        "halt_class": _has_halt_class(result),
    }


def _has_halt_class(result) -> bool:
    """Detect halt-class findings: critical-severity that the operator
    should treat as a stop signal."""
    for f in getattr(result, "findings", []):
        sev = str(getattr(f, "severity", "") or (f.get("severity") if hasattr(f, "get") else ""))
        if sev.lower() in ("critical", "severity.critical", "gauntletseverity.critical"):
            return True
    return False


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "plans"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "2026-04-28-evolution-round-gauntlet-receipt.json"
    out_md = out_dir / "2026-04-28-evolution-round-gauntlet-summary.md"

    try:
        receipt = asyncio.run(run_gauntlet())
        receipt["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        receipt = {
            "status": "error",
            "error_class": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }

    out_json.write_text(json.dumps(receipt, indent=2, default=str))
    print(f"wrote: {out_json}")

    lines = [
        "# 2026-04-28 Evolution Round — Gauntlet Vet Summary",
        "",
        f"**Status:** {receipt.get('status')}",
        "",
    ]
    if receipt.get("status") == "ok":
        lines += [
            f"**Duration:** {receipt.get('duration_s'):.1f}s",
            f"**Verdict:** {receipt.get('verdict')}",
            f"**Robustness score:** {receipt.get('robustness_score')}",
            f"**Findings:** {receipt.get('findings_count')}",
            f"**Halt-class:** {receipt.get('halt_class')}",
            "",
            "## Findings",
            "",
        ]
        for f in receipt.get("findings", []):
            lines += [
                f"- **{f['severity']} / {f['category']}**: {f.get('title')}",
                f"  - {f.get('description')[:300]}",
            ]
    else:
        lines += [
            f"**Error class:** {receipt.get('error_class')}",
            f"**Error message:** {receipt.get('error_message')}",
            "",
            "## Traceback (truncated)",
            "",
            "```",
            (receipt.get("traceback") or "")[:2000],
            "```",
            "",
        ]
    out_md.write_text("\n".join(lines))
    print(f"wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
