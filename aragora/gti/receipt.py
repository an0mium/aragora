"""Belief provenance + freshness fields for GTI DecisionReceipts.

Extends the existing receipt provenance concept (aragora/gauntlet/receipt_models.py
:ProvenanceRecord, and ReceiptVerification whose verification_status already
includes "stale") with the freshness contract this benchmark requires: every
load-bearing belief must carry a source + as_of timestamp and must not be used
past its TTL without revalidation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class BeliefProvenance:
    belief_id: str
    source: str
    as_of: str  # ISO 8601
    verification_method: str
    freshness_ttl_seconds: float
    was_revalidated_at_decision: bool


def validate_belief_provenance(beliefs: list[BeliefProvenance], now_iso: str) -> list[str]:
    """Return a list of problems; empty list means the receipt is valid."""
    problems: list[str] = []
    now = datetime.fromisoformat(now_iso)
    for b in beliefs:
        if not b.source or not b.as_of:
            problems.append(f"{b.belief_id}: missing source/as_of provenance")
            continue
        age = (now - datetime.fromisoformat(b.as_of)).total_seconds()
        if age > b.freshness_ttl_seconds and not b.was_revalidated_at_decision:
            problems.append(
                f"{b.belief_id}: belief used past TTL "
                f"({age:.0f}s > {b.freshness_ttl_seconds:.0f}s) without revalidation"
            )
    return problems
