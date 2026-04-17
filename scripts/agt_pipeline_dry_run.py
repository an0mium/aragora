#!/usr/bin/env python3
"""End-to-end AGT pipeline dry-run.

Runs a fully synthetic trace through the 11 AGT-* modules landed
Apr 17 2026 and writes the complete trace as JSON. This is the first
proof that the wire works end-to-end — it exercises each module's
output shape against the next module's input shape without any
external dependencies (no debate LLM calls, no GitHub API, no chain).

Flow:

    synthetic CruxSet  ─┐
                        ├→ AgentReceipt (AGT-02)
                        │
                        └→ DIC-17 FollowupProposal

    synthetic Market + MarketPosition + ResolutionEvent (AGT-04)
                        │
                        ├→ bridge_from_market_position (AGT-05)
                        │      ↓
                        │  (StakeableClaim, ResolvedClaim)
                        │      ↓
                        │  settle_claim (AGT-05)
                        │      ↓
                        │  ReputationDelta
                        │      ↓
                        │  anchor_delta dry-run (AGT-05)
                        │      ↓
                        └→ AnchorReceipt

    ShiftLedger fixture → compute_viah (AGT-06) → ViahReport

Usage:
    python3 scripts/agt_pipeline_dry_run.py [--out PATH]

Default output: docs/status/generated/agt_e2e_trace.json

The script is deterministic — given the same inputs it produces the
same JSON (timestamps are the one exception; ``--pin-timestamp``
forces a fixed reference time for fixture generation).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from aragora.epistemic.followup import (  # noqa: E402
    propose_followup_for_crux,
    propose_followup_for_cruxset,
    propose_followup_for_failed_claim,
)

# Deterministic test signing key (32 bytes, fixed for fixture reproducibility).
# NOT for production use — this is a public, committed seed.
_FIXTURE_SIGNING_SEED = bytes.fromhex(
    "a3c9d1e07b2f45680123456789abcdef0123456789abcdef0123456789abcdef"
)
from aragora.markets.store import MarketStore  # noqa: E402
from aragora.markets.types import (  # noqa: E402
    Market,
    MarketPosition,
    ResolutionEvent as MarketResolution,
)
from aragora.metrics.viah import ViahReport, compute_viah  # noqa: E402
from aragora.protocols.a2a.receipts import (  # noqa: E402
    AgentReceipt,
    DissentEntry,
)
from aragora.reasoning.cruxset import Crux, CruxPosition, CruxSet  # noqa: E402
from aragora.reputation.anchor import anchor_delta  # noqa: E402
from aragora.reputation.bridge import bridge_from_market_position  # noqa: E402
from aragora.reputation.settlement import settle_claim  # noqa: E402
from aragora.swarm.shift_ledger import ShiftLedger  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "docs" / "status" / "generated" / "agt_e2e_trace.json"


def _pinned_now() -> datetime:
    """A fixed reference timestamp for reproducible fixture generation."""
    return datetime(2026, 4, 17, 12, 0, tzinfo=UTC)


def _build_synthetic_cruxset(now: datetime) -> CruxSet:
    """Hand-constructed CruxSet as if a real debate produced it."""
    return CruxSet.build(
        question="Should Aragora activate the AGT-* upper-layer flags now?",
        cruxes=[
            Crux(
                crux_id="c_benchmark_evidence",
                statement=(
                    "Is three consecutive green BC-12 soaks the right evidence "
                    "standard before flipping the upper-layer flags?"
                ),
                positions=(
                    CruxPosition(
                        side="for",
                        agents=("codex-strategic",),
                        rationale="substrate-first discipline",
                    ),
                    CruxPosition(
                        side="against",
                        agents=("claude-sonnet",),
                        rationale="productive soak is stronger evidence",
                    ),
                ),
                load_bearing_score=0.82,
                evidence_gaps=("no settled policy on productive-vs-idle soak equivalence",),
                counterfactual="If one productive soak is sufficient, flip order of AGT activation vs soak",
                candidate_verifier="docs/status/BC12_SOAK_POLICY.md (not yet written)",
            ),
            Crux(
                crux_id="c_reputation_bootstrap",
                statement="How do new agents earn initial reputation without prior settled claims?",
                positions=(
                    CruxPosition(
                        side="for_sandbox",
                        agents=("claude-sonnet",),
                        rationale="sandbox reputation with reduced stake caps",
                    ),
                    CruxPosition(
                        side="for_oracle",
                        agents=("codex-strategic",),
                        rationale="bootstrap from trusted oracle delegations",
                    ),
                ),
                load_bearing_score=0.65,
                evidence_gaps=("no cold-start policy specified",),
                counterfactual="Without a policy, AGT-05 flows cannot accept new-agent submissions",
            ),
            Crux(
                crux_id="c_tangential_detail",
                statement="Which logger level should DIC-17 use for proposal-filing events?",
                positions=(CruxPosition(side="info", agents=("claude-sonnet",)),),
                load_bearing_score=0.18,
            ),
        ],
        decision=None,  # Crux-finder debates may intentionally not converge
        evidence_gaps=("BC-12 policy gap", "cold-start reputation policy gap"),
        counterfactual_notes=(
            "avg_uncertainty=0.52",
            "convergence_barrier=0.41",
        ),
        verifier_candidates=("c_benchmark_evidence", "c_reputation_bootstrap"),
        receipt_id="rcpt_e2e_synthetic",
        provenance={"debate_id": "debate_e2e_synthetic", "arena_run_id": "run_001"},
        created_at=now.isoformat().replace("+00:00", "Z"),
    )


def _build_synthetic_agent_receipt(
    cruxset: CruxSet,
    *,
    now: datetime,
) -> AgentReceipt:
    """Wrap the CruxSet in an A2A AgentReceipt envelope.

    Uses a deterministic Ed25519 test key so the fixture hash is
    reproducible across runs. Real production receipts must use a
    properly managed signing key.
    """
    signing_key = Ed25519PrivateKey.from_private_bytes(_FIXTURE_SIGNING_SEED)
    return AgentReceipt.build(
        issuer="aragora.ai",
        subject_kind="debate_outcome",
        subject={
            "question": cruxset.question,
            "decision": cruxset.decision,
            "cruxset_id": cruxset.cruxset_id,
        },
        cruxset=cruxset.to_json(),
        dissent=(
            DissentEntry(
                agent_id="codex-strategic",
                statement="Three consecutive green soaks remain the right bar",
                confidence=0.7,
            ),
            DissentEntry(
                agent_id="claude-sonnet",
                statement="Productive soak is sufficient evidence",
                confidence=0.6,
            ),
        ),
        reputation_deltas_applied=(),
        freshness_sla_seconds=24 * 3600,
        settlement_window_seconds=7 * 24 * 3600,
        provenance={"debate_id": "debate_e2e_synthetic"},
        issued_at=now.isoformat().replace("+00:00", "Z"),
        signing_key=signing_key,
    )


def _build_synthetic_market(now: datetime) -> Market:
    return Market.create(
        question_kind="pr_merge",
        target={"repo": "synaptent/aragora", "number": 6116},
        description="Will the AGT CLI verbs PR merge within 3 days?",
        resolution_window_days=3,
        created_at=now - timedelta(days=4),
    )


def _build_synthetic_position(market: Market, now: datetime) -> MarketPosition:
    return MarketPosition.create(
        market_id=market.market_id,
        agent_id="alice-predictor",
        probability=0.92,
        stake=50,
        submitted_at=now - timedelta(days=3, hours=12),
        rationale="PR has all 5 checks green and admin privileges available",
    )


def _build_synthetic_resolution(
    market: Market, now: datetime, outcome: str = "yes"
) -> MarketResolution:
    if outcome == "yes":
        return MarketResolution.yes(
            market_id=market.market_id,
            resolution_source="github_pr_state",
            evidence={"state": "MERGED", "merged": True, "merged_at": "2026-04-17T05:22:00Z"},
            resolved_at=now,
        )
    return MarketResolution.no(
        market_id=market.market_id,
        resolution_source="github_pr_state",
        evidence={"state": "CLOSED", "merged": False},
        resolved_at=now,
    )


def _build_synthetic_ledger(now: datetime, tmp_dir: Path) -> ShiftLedger:
    """Seed a temporary ShiftLedger with one shift worth of entries."""
    ledger_path = tmp_dir / "shift_ledger.jsonl"
    entries = [
        {
            "entry_type": "shift_start",
            "timestamp": (now - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "payload": {
                "shift_id": "e2e-shift-1",
                "max_hours": 3.0,
                "benchmark_mode": "hybrid",
                "queue_size": 0,
            },
        },
        {
            "entry_type": "pr_merged",
            "timestamp": (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "payload": {"pr_number": 6080, "title": "substrate hardening"},
        },
        {
            "entry_type": "pr_merged",
            "timestamp": (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "payload": {"pr_number": 6116, "title": "AGT CLI verbs"},
        },
        {
            "entry_type": "cycle_tick",
            "timestamp": (now - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "payload": {
                "queue_size": 0,
                "queue_removed": 0,
                "open_prs": 0,
                "boss_running": False,
                "merge_running": True,
                "benchmark_fresh": True,
                "actions": [],
                "rescue_count": 0,
                "stop_reason": "",
            },
        },
        {
            "entry_type": "shift_stop",
            "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "payload": {
                "shift_id": "e2e-shift-1",
                "reason": "completed",
                "cycles": 1,
                "duration_seconds": 10800.0,
            },
        },
    ]
    with ledger_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
    return ShiftLedger(path=ledger_path)


def run_pipeline(*, now: datetime | None = None, tmp_dir: Path | None = None) -> dict:
    """Execute the full AGT pipeline on synthetic inputs and return the trace."""
    if now is None:
        now = _pinned_now()
    if tmp_dir is None:
        tmp_dir = Path.cwd() / ".agt_e2e_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # === AGT-01: build CruxSet + wrap in AgentReceipt ===
    cruxset = _build_synthetic_cruxset(now)
    receipt = _build_synthetic_agent_receipt(cruxset, now=now)

    # === DIC-17: propose follow-ups for load-bearing cruxes ===
    proposals = propose_followup_for_cruxset(cruxset, top_k=3)

    # === AGT-04: build market + position + resolution ===
    market = _build_synthetic_market(now)
    position = _build_synthetic_position(market, now)
    resolution = _build_synthetic_resolution(market, now, outcome="yes")

    store = MarketStore(tmp_dir / "markets")
    store.add_market(market)
    store.add_position(position)
    store.record_resolution(resolution)

    # === AGT-05: bridge → settle → anchor (dry-run) ===
    claim, resolved = bridge_from_market_position(position, market, resolution)
    pinned_iso = now.isoformat().replace("+00:00", "Z")
    delta = settle_claim(claim, resolved, scoring_rule="brier_proper", applied_at=pinned_iso)
    anchor_receipt = anchor_delta(delta, agent_id=42, dry_run=True)
    # AnchorReceipt.submitted_at uses datetime.now() internally; normalize
    # to the pinned reference so the trace JSON is reproducible.
    anchor_receipt_json = anchor_receipt.to_json()
    anchor_receipt_json["submitted_at"] = pinned_iso

    # === AGT-05 failure branch: settle a losing claim, propose DIC-17 follow-up ===
    losing_position = MarketPosition.create(
        market_id=market.market_id + "-alt",  # avoid the "already resolved" store guard
        agent_id="bob-overconfident",
        probability=0.95,
        stake=50,
        submitted_at=now - timedelta(days=3),
    )
    # Fabricate a distinct market for the alt position
    losing_market = Market.create(
        question_kind="pr_merge",
        target={"repo": "synaptent/aragora", "number": 9999},
        description="Will a non-existent PR merge?",
        resolution_window_days=3,
        created_at=now - timedelta(days=4),
    )
    losing_position_real = MarketPosition.create(
        market_id=losing_market.market_id,
        agent_id="bob-overconfident",
        probability=0.95,
        stake=50,
        submitted_at=now - timedelta(days=3),
    )
    losing_resolution = MarketResolution.no(
        market_id=losing_market.market_id,
        resolution_source="github_pr_state",
        evidence={"state": "CLOSED", "merged": False},
        resolved_at=now,
    )
    losing_claim, losing_resolved = bridge_from_market_position(
        losing_position_real, losing_market, losing_resolution
    )
    losing_delta = settle_claim(
        losing_claim, losing_resolved, scoring_rule="brier_proper", applied_at=pinned_iso
    )
    losing_proposal = propose_followup_for_failed_claim(
        losing_claim, losing_resolved, losing_delta, delta_loss_threshold=-10.0
    )

    # === AGT-06: VIAH over synthetic ShiftLedger ===
    ledger = _build_synthetic_ledger(now, tmp_dir)
    viah = compute_viah(ledger=ledger, window_hours=24.0, now=now)
    # Normalize the environment-dependent ledger path so the trace is
    # reproducible across machines and tmp-dirs
    viah_dict = viah.to_dict()
    viah_dict["inputs"]["ledger_path"] = "<pinned:shift_ledger.jsonl>"

    # === Assemble full trace ===
    return {
        "pipeline": "agt_e2e_dry_run",
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "cruxset": cruxset.to_json(),
        "agent_receipt": receipt.to_json(),
        "dic17_proposals_from_cruxset": [
            {
                "source_key": p.source_key,
                "title": p.title,
                "labels": list(p.labels),
                "rationale": p.rationale,
                "provenance": dict(p.provenance),
            }
            for p in proposals
        ],
        "market": market.to_json(),
        "position": position.to_json(),
        "resolution": resolution.to_json(),
        "claim": claim.to_json(),
        "resolved_claim": resolved.to_json(),
        "reputation_delta": delta.to_json(),
        "anchor_receipt": anchor_receipt_json,
        "failure_branch": {
            "losing_claim": losing_claim.to_json(),
            "losing_resolved": losing_resolved.to_json(),
            "losing_delta": losing_delta.to_json(),
            "dic17_proposal_for_failed_claim": (
                None
                if losing_proposal is None
                else {
                    "source_key": losing_proposal.source_key,
                    "title": losing_proposal.title,
                    "labels": list(losing_proposal.labels),
                    "rationale": losing_proposal.rationale,
                }
            ),
        },
        "viah_report": viah_dict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the AGT e2e pipeline dry-run")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Trace JSON output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--pin-timestamp",
        action="store_true",
        help="Use the canonical pinned timestamp for fixture-stable output",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Directory for temporary market/ledger state (default: ./.agt_e2e_tmp)",
    )
    args = parser.parse_args()

    now = _pinned_now() if args.pin_timestamp else datetime.now(tz=UTC)
    trace = run_pipeline(now=now, tmp_dir=args.tmp_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(trace, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote trace to {args.out}")
    print(f"  cruxset_id: {trace['cruxset']['cruxset_id']}")
    print(f"  agent_receipt: {trace['agent_receipt']['receipt_id']}")
    print(f"  dic17 proposals (from cruxset): {len(trace['dic17_proposals_from_cruxset'])}")
    print(f"  reputation_delta: {trace['reputation_delta']['delta']:+.3f}")
    print(
        f"  anchor_receipt: dry_run={trace['anchor_receipt']['dry_run']} value={trace['anchor_receipt']['value']}"
    )
    print(f"  viah: {trace['viah_report']['viah']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
