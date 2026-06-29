"""Bridge: a merge-quorum PR-review outcome -> a portable DecisionReceipt.

This is the M2 core. It turns a :class:`CollectOutcome` — the result of a
heterogeneous-model PR review produced by ``aragora/swarm/quorum_evidence.py`` —
into a gauntlet :class:`DecisionReceipt`, so the review can be exported to a
portable Open Decision Receipt (``aragora/gauntlet/odr_export.py``) and verified
independently by ``aragora-verify``.

Design rules (mirroring the emitter's "never fabricate" contract):

- **Pure and side-effect-free.** It copies fields off the outcome; it makes no
  network calls and mutates nothing. The Action / settlement layer decides what
  to do with the receipt.
- **No invented clearance.** The verdict and quorum reflect only posted,
  supportive evidence when no reviewer dissent is present. Prepared evidence
  must remain merge-blocking in the portable receipt.
- **Internally consistent quorum.** Posted supporting and dissenting families
  are carried into ``consensus_proof``; ``odr_export`` records all reviewers as
  participants, so the verifier's quorum-consistency check holds
  (supporting/dissenting agents are always a subset of participants).
- **Family is the disclosed axis.** Each evidence item's model *family* is
  recorded as its participant provider, so ODR independence (distinct model
  families) is disclosed rather than guessed.
"""

from __future__ import annotations

import hashlib

from aragora.gauntlet.receipt_models import (
    AgentResponseRecord,
    ConsensusProof,
    DecisionReceipt,
)
from aragora.swarm.quorum_evidence import FAMILY_PROVIDERS, CollectOutcome, tier_quorum_rule

__all__ = ["collect_outcome_to_decision_receipt"]


def collect_outcome_to_decision_receipt(outcome: CollectOutcome) -> DecisionReceipt:
    """Map a merge-quorum :class:`CollectOutcome` onto a :class:`DecisionReceipt`.

    The returned receipt is ready for ``decision_receipt_to_odr`` and carries the
    PR's provenance (repo, number, head SHA, tier) under ``settlement_metadata``.
    """
    supportive = list(outcome.supportive_families)
    dissenting = list(outcome.dissenting_families)
    counting = list(outcome.counting_families)
    posted = {str(family).strip() for family in outcome.posted if str(family).strip()}
    posted_supportive = [family for family in supportive if family in posted]
    posted_quorum = bool(posted) and tier_quorum_rule(
        outcome.tier,
        tiered_gate=outcome.tiered_gate,
    ).is_satisfied_by(posted_supportive)
    reached = outcome.action == "post" and posted_quorum and not dissenting

    confidence = (len(supportive) / len(counting)) if counting else 0.0
    verdict = "PASS" if reached else "CHANGES_REQUESTED"

    head = (outcome.head_sha or "").strip()
    input_hash = hashlib.sha256(f"{outcome.repo}#{outcome.pr}@{head}".encode()).hexdigest()

    agent_responses = [
        AgentResponseRecord(
            agent=item.family,
            response=item.body,
            provider=FAMILY_PROVIDERS.get(item.family, item.family),
        )
        for item in outcome.items
        if (item.family or "").strip()
    ]

    dissenting_views = [f"{item.family}: {item.body}" for item in outcome.items if item.dissenting]

    return DecisionReceipt(
        receipt_id=f"pr-{outcome.pr}-{head[:12]}" if head else f"pr-{outcome.pr}",
        gauntlet_id=f"merge-quorum/{outcome.repo}#{outcome.pr}",
        timestamp=outcome.head_committed_at or "",
        input_summary=f"Merge {outcome.repo}#{outcome.pr} @ {head[:12]}",
        input_hash=input_hash,
        risk_summary={
            "counting": len(counting),
            "supportive": len(supportive),
            "dissenting": len(dissenting),
            "total": len(outcome.items),
        },
        attacks_attempted=0,
        attacks_successful=0,
        probes_run=0,
        vulnerabilities_found=0,
        verdict=verdict,
        confidence=confidence,
        robustness_score=confidence,
        verdict_reasoning=outcome.action_reason,
        dissenting_views=dissenting_views,
        consensus_proof=ConsensusProof(
            reached=reached,
            confidence=confidence,
            supporting_agents=posted_supportive,
            dissenting_agents=dissenting,
            method="merge-quorum",
        ),
        agent_responses=agent_responses,
        settlement_metadata={
            "repo": outcome.repo,
            "pr": outcome.pr,
            "head_sha": head,
            "tier": outcome.tier,
            "action": outcome.action,
            "tiered_gate": outcome.tiered_gate,
        },
    )
