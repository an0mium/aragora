#!/usr/bin/env python3
"""End-to-end smoke test for the AGT-05 reputation signal flow.

Exercises the full path:

    MetaculusQuestion (resolved) + agent prediction
        → bridge_from_metaculus_question
        → StakeableClaim + ResolvedClaim
        → settle_claim (Brier-proper scoring rule)
        → ReputationDelta

with the ``ARAGORA_REPUTATION_FLOW_ENABLED`` flag set to ``1``. Uses
synthetic-but-realistic data so the run is hermetic and deterministic.

Run::

    ARAGORA_REPUTATION_FLOW_ENABLED=1 \\
        python3 scripts/agt_05_reputation_flow_smoke.py

The script prints a per-agent per-question table of reputation deltas
and a per-agent total. No live API calls. No mutation of any ledger.
This is purely a calibration exercise to prove the signal flow runs
end-to-end on the same code path a live Arena debate would use once
the predictor/resolver hooks are wired.

Out of scope (deliberate):

- Calling :func:`aragora.reputation.ledger.apply_delta` or any
  on-chain registry — this script only computes deltas.
- Wiring into ``Arena`` — that is the production integration step
  ``AGT-05`` sub-deliverable that follows this calibration.
- Choosing between ``brier_proper`` and ``binary`` for live use —
  this script demonstrates both for comparison.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from typing import Any

# Ensure repo root is importable when run from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aragora.connectors.prediction_markets.metaculus import MetaculusQuestion
from aragora.reputation.metaculus_bridge import (
    bridge_from_metaculus_question,
    reputation_flow_enabled,
)
from aragora.reputation.settlement import settle_claim


def _resolved_question(qid: int, title: str, resolution: float) -> MetaculusQuestion:
    now = datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")
    return MetaculusQuestion(
        question_id=qid,
        title=title,
        question_type="binary",
        created_time=now,
        close_time=now,
        resolve_time=now,
        active_state="resolved",
        resolution=resolution,
        community_q2=0.5,
        raw={},
    )


def _smoke() -> int:
    if not reputation_flow_enabled():
        print(
            "ARAGORA_REPUTATION_FLOW_ENABLED is not set; nothing to demonstrate.",
            file=sys.stderr,
        )
        return 2

    questions = [
        ("manifold-1001", "Will the H1-01 rev-4 corpus be promoted by 2026-05-15?", 1.0),
        ("manifold-1002", "Will the proof-loop alerter PR (#7156) land by 2026-05-20?", 1.0),
        ("manifold-1003", "Will boss_metrics.jsonl reach 10k rows by 2026-06-01?", 0.0),
    ]
    # Agents and their predicted probabilities per question.
    # Calibrated (=outcome): high score; opposite: low/negative score.
    agents = {
        "claude-opus-4-7": [0.90, 0.85, 0.10],  # well-calibrated
        "gpt-4.1": [0.60, 0.55, 0.45],  # roughly indifferent
        "demo-anti": [0.10, 0.05, 0.90],  # systematically wrong
    }

    print()
    print("AGT-05 Reputation Flow Smoke Test")
    print("=" * 78)
    print(f"  ARAGORA_REPUTATION_FLOW_ENABLED={os.environ.get('ARAGORA_REPUTATION_FLOW_ENABLED')}")
    print("  scoring_rule=brier_proper")
    print(f"  agents={list(agents.keys())}")
    print(f"  questions={[q[0] for q in questions]}")
    print()

    per_agent_total: dict[str, float] = dict.fromkeys(agents, 0.0)
    rows: list[dict[str, Any]] = []

    for q_id, q_title, resolution in questions:
        q = _resolved_question(qid=int(q_id.split("-")[1]), title=q_title, resolution=resolution)
        for agent_id, predictions in agents.items():
            idx = [q[0] for q in questions].index(q_id)
            p = predictions[idx]
            claim, resolved = bridge_from_metaculus_question(
                q,
                agent_id=agent_id,
                predicted_probability=p,
                stake_units=10,
            )
            delta = settle_claim(claim, resolved)
            per_agent_total[agent_id] += delta.delta
            rows.append(
                {
                    "question_id": q_id,
                    "agent": agent_id,
                    "predicted_p": p,
                    "outcome": resolved.outcome,
                    "brier": delta.reason.get("brier"),
                    "payout_fraction": delta.reason.get("payout_fraction"),
                    "delta": round(delta.delta, 3),
                }
            )

    # Pretty print
    print(
        f"{'question':<14} {'agent':<24} {'pred_p':>7} {'outcome':<6} "
        f"{'brier':>7} {'payout':>8} {'delta':>7}"
    )
    print("-" * 78)
    for r in rows:
        print(
            f"{r['question_id']:<14} {r['agent']:<24} "
            f"{r['predicted_p']:>7.2f} {r['outcome']:<6} "
            f"{(r['brier'] or 0):>7.3f} {(r['payout_fraction'] or 0):>8.3f} "
            f"{r['delta']:>7.2f}"
        )
    print("-" * 78)
    print()
    print("Per-agent total reputation delta (Brier-proper, stake_units=10 per Q):")
    for a in sorted(per_agent_total):
        print(f"  {a:<24} {per_agent_total[a]:>+8.3f}")
    print()
    print("Interpretation:")
    print("  delta > 0  → agent gained reputation (calibrated towards outcome)")
    print("  delta < 0  → agent lost reputation (systematically miscalibrated)")
    print("  delta ≈ 0  → agent indifferent (predicted near 0.5)")
    print()
    print("This proves the end-to-end signal flow runs cleanly with the flag enabled.")
    print("Production wiring: have an Arena post-debate hook call this same path")
    print("with debate-derived predictions on a real resolved Manifold/Metaculus")
    print("question — that is the next AGT-05 production integration step.")
    print()

    summary = {
        "scoring_rule": "brier_proper",
        "flag": os.environ.get("ARAGORA_REPUTATION_FLOW_ENABLED"),
        "rows": rows,
        "per_agent_total": per_agent_total,
        "ran_at": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
    }
    if "--json" in sys.argv[1:]:
        print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(_smoke())
