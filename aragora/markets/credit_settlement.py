"""Credit settlement bridge for resolved synthetic-market positions (AGT-04 SD-3).

Connects ``evaluate_position_payout`` to ``ComputeBudgetManager``, closing
AGT-04 sub-deliverable 3: internal credit bookkeeping (stake forfeit/refund).

Gated behind ``ARAGORA_SYNTHETIC_MARKETS_ENABLED`` (the existing AGT-04 flag).
``require_enabled=False`` bypasses the guard for callers that checked the flag.

Credit mapping: payout in [-stake, +stake].
  payout > 0 → reward_accuracy(epistemic_score=payout/stake)
  payout < 0 → penalize_inaccuracy(epistemic_score=1 - abs(payout)/stake)
  payout == 0 → no-op (inconclusive / random; stake logically refunded)

Out of scope: durable ledger persistence, on-chain anchoring (AGT-05),
batch scheduling.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

from aragora.blockchain.compute_budget import ComputeBudgetManager
from aragora.markets.scoring import evaluate_position_payout
from aragora.markets.types import MarketPosition, ResolutionEvent

logger = logging.getLogger(__name__)

_FLAG = "ARAGORA_SYNTHETIC_MARKETS_ENABLED"


def _enabled() -> bool:
    return str(os.environ.get(_FLAG) or "").strip().lower() in {"1", "true", "yes", "on"}


class CreditSettlementError(RuntimeError):
    """Raised when the feature flag is unset or on settlement invariant violations."""


def settle_position_credit(
    position: MarketPosition,
    resolution: ResolutionEvent,
    manager: ComputeBudgetManager,
    *,
    require_enabled: bool = True,
) -> int:
    """Apply a resolved prediction's Brier payout to the agent's compute budget.

    Returns the signed credit delta (positive = reward, negative = penalty,
    0 = inconclusive / no movement).
    """
    if require_enabled and not _enabled():
        raise CreditSettlementError(
            f"synthetic market credit settlement is disabled; set {_FLAG}=1 to enable"
        )

    payout = evaluate_position_payout(position=position, resolution=resolution)

    if payout > 0:
        score = payout / position.stake
        manager.reward_accuracy(position.agent_id, epistemic_score=score)
        logger.debug(
            "credit_settlement reward agent=%s payout=%d score=%.4f",
            position.agent_id,
            payout,
            score,
        )
    elif payout < 0:
        # penalize_inaccuracy expects accuracy (high=calibrated), so invert wrongness.
        accuracy = 1.0 - abs(payout) / position.stake
        manager.penalize_inaccuracy(position.agent_id, epistemic_score=accuracy)
        logger.debug(
            "credit_settlement penalty agent=%s payout=%d accuracy=%.4f",
            position.agent_id,
            payout,
            accuracy,
        )
    else:
        logger.debug(
            "credit_settlement no-op agent=%s outcome=%s", position.agent_id, resolution.outcome
        )

    return payout


def settle_batch_credits(
    positions: Iterable[MarketPosition],
    resolutions: dict[str, ResolutionEvent],
    manager: ComputeBudgetManager,
    *,
    require_enabled: bool = True,
) -> list[tuple[str, int]]:
    """Settle all positions that have a matching resolved market.

    Returns ``[(position_id, credit_delta)]`` for each settled position.
    Positions with no matching resolution are skipped silently.
    """
    if require_enabled and not _enabled():
        raise CreditSettlementError(
            f"synthetic market credit settlement is disabled; set {_FLAG}=1 to enable"
        )

    settled: list[tuple[str, int]] = []
    for position in positions:
        resolution = resolutions.get(position.market_id)
        if resolution is None:
            continue
        delta = settle_position_credit(position, resolution, manager, require_enabled=False)
        settled.append((position.position_id, delta))
    return settled


__all__ = ["CreditSettlementError", "settle_batch_credits", "settle_position_credit"]
