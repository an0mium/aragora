"""AGT-05 reputation flow — claim → stake → resolution → delta.

This package is the in-memory Python layer of the AGT-05 spec in
``docs/plans/SKIN_IN_THE_GAME_REPUTATION.md``. It unifies the AGT-04
synthetic-market resolution stream and the AGT-01 CruxSet resolution
stream into a single shape — :class:`StakeableClaim` plus
:class:`ResolvedClaim` — from which :func:`settle_claim` computes a
:class:`ReputationDelta` using a proper scoring rule.

The on-chain anchoring via
:class:`aragora.blockchain.contracts.reputation.ReputationRegistry`
lives in a downstream PR. This package computes deltas in memory and
leaves the registry write to callers, so the flow can be tested and
validated before touching chain state.

Feature flag: ``ARAGORA_REPUTATION_FLOW_ENABLED``. Dormant by default.
"""

from __future__ import annotations

import os

from aragora.reputation.anchor import (
    AnchorError,
    AnchorReceipt,
    anchor_delta,
    anchoring_enabled,
    delta_to_feedback_args,
    enable_anchoring,
)
from aragora.reputation.bridge import bridge_from_market_position
from aragora.reputation.crux_bridge import (
    CruxPositionRecord,
    CruxResolutionEvent,
    bridge_from_crux_position,
)
from aragora.reputation.settlement import settle_claim
from aragora.reputation.store import (
    AgentScore,
    ReputationStore,
    ReputationStoreError,
    enable_store,
    store_enabled,
)
from aragora.reputation.types import (
    DOMAIN_CODE_PR,
    DOMAIN_CRUX_RESOLUTION,
    DOMAIN_DEBATE_POSITION,
    DOMAIN_KM_CONTRIBUTION,
    DOMAIN_PREDICTION_MARKET,
    ReputationDelta,
    ResolvedClaim,
    StakeableClaim,
)

__all__ = [
    "DOMAIN_CODE_PR",
    "DOMAIN_CRUX_RESOLUTION",
    "DOMAIN_DEBATE_POSITION",
    "DOMAIN_KM_CONTRIBUTION",
    "DOMAIN_PREDICTION_MARKET",
    "AnchorError",
    "AnchorReceipt",
    "ReputationDelta",
    "ResolvedClaim",
    "StakeableClaim",
    "anchor_delta",
    "anchoring_enabled",
    "bridge_from_market_position",
    "CruxPositionRecord",
    "CruxResolutionEvent",
    "bridge_from_crux_position",
    "delta_to_feedback_args",
    "enable_anchoring",
    "enable_reputation_flow",
    "reputation_flow_enabled",
    "settle_claim",
    # store
    "AgentScore",
    "ReputationStore",
    "ReputationStoreError",
    "enable_store",
    "store_enabled",
]


def reputation_flow_enabled() -> bool:
    """Return True if the AGT-05 reputation flow is enabled."""
    raw = str(os.environ.get("ARAGORA_REPUTATION_FLOW_ENABLED") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def enable_reputation_flow() -> None:
    """Enable the AGT-05 reputation flow for the current process."""
    os.environ["ARAGORA_REPUTATION_FLOW_ENABLED"] = "1"
