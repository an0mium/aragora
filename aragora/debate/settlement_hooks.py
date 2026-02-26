"""Settlement hooks — callbacks fired when claims are created or resolved.

Provides a pluggable hook system for the SettlementTracker so that
settlement events can trigger side-effects (blockchain reputation updates,
event bus emissions, audit logging) without coupling the tracker to those
systems.

Usage:
    from aragora.debate.settlement_hooks import (
        SettlementHookRegistry,
        ERC8004SettlementHook,
        EventBusSettlementHook,
    )

    hooks = SettlementHookRegistry()
    hooks.register(ERC8004SettlementHook())
    hooks.register(EventBusSettlementHook(event_bus))

    tracker = SettlementTracker(hooks=hooks)
    # Now settle() and extract_verifiable_claims() fire hooks automatically.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Protocol

from aragora.debate.settlement import (
    SettlementBatch,
    SettlementRecord,
    SettleResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hook protocol
# ---------------------------------------------------------------------------


class SettlementHook(Protocol):
    """Interface for settlement lifecycle hooks.

    Implementations receive callbacks when claims are extracted from a
    debate and when they are later settled against reality.  Hooks should
    be fast and non-blocking; failures are logged but do not prevent the
    settlement from proceeding.
    """

    def on_claims_extracted(self, batch: SettlementBatch) -> None:
        """Called after verifiable claims are extracted from a debate."""
        ...

    def on_settled(self, record: SettlementRecord, result: SettleResult) -> None:
        """Called after a claim is settled (correct / incorrect / partial)."""
        ...


# ---------------------------------------------------------------------------
# Hook registry
# ---------------------------------------------------------------------------


class SettlementHookRegistry:
    """Collects hooks and dispatches settlement events to all registered hooks.

    Thread-safe for reads (dispatching) but not for concurrent registration,
    which is fine because hooks are registered at startup.
    """

    def __init__(self) -> None:
        self._hooks: list[SettlementHook] = []

    def register(self, hook: SettlementHook) -> None:
        """Register a settlement hook."""
        self._hooks.append(hook)

    def fire_claims_extracted(self, batch: SettlementBatch) -> None:
        """Dispatch on_claims_extracted to all hooks."""
        for hook in self._hooks:
            try:
                hook.on_claims_extracted(batch)
            except (ValueError, TypeError, AttributeError, RuntimeError, OSError) as e:
                logger.warning(
                    "Settlement hook %s.on_claims_extracted failed: %s",
                    type(hook).__name__,
                    e,
                )

    def fire_settled(self, record: SettlementRecord, result: SettleResult) -> None:
        """Dispatch on_settled to all hooks."""
        for hook in self._hooks:
            try:
                hook.on_settled(record, result)
            except (ValueError, TypeError, AttributeError, RuntimeError, OSError) as e:
                logger.warning(
                    "Settlement hook %s.on_settled failed: %s",
                    type(hook).__name__,
                    e,
                )

    @property
    def hook_count(self) -> int:
        return len(self._hooks)


# ---------------------------------------------------------------------------
# ERC-8004 settlement hook
# ---------------------------------------------------------------------------


class ERC8004SettlementHook:
    """Pushes settlement outcomes to ERC-8004 reputation and validation registries.

    On settlement:
    - Submits a ReputationFeedback entry with the agent's Brier score component
      and outcome (tag1="settlement", tag2=outcome).
    - Creates a ValidationRecord with PASS/FAIL based on the settlement outcome.

    The adapter is resolved lazily so the hook can be instantiated even if
    the blockchain module is not available (it will log and skip).
    """

    def __init__(self, adapter: Any | None = None) -> None:
        self._adapter = adapter

    def _get_adapter(self) -> Any | None:
        if self._adapter is not None:
            return self._adapter
        try:
            from aragora.knowledge.mound.adapters.erc8004_adapter import ERC8004Adapter

            self._adapter = ERC8004Adapter()
            return self._adapter
        except ImportError:
            logger.debug("ERC8004Adapter not available")
            return None

    def on_claims_extracted(self, batch: SettlementBatch) -> None:
        """No-op on extraction — we only push on settlement resolution."""

    def on_settled(self, record: SettlementRecord, result: SettleResult) -> None:
        """Push reputation feedback and validation record to ERC-8004."""
        adapter = self._get_adapter()
        if adapter is None:
            return

        agent = record.claim.author
        brier_component = (record.claim.confidence - record.score) ** 2
        # Convert Brier component to reputation: 0=worst, 100=perfect
        reputation = max(0, min(100, int((1.0 - brier_component) * 100)))

        # Build a content-addressable hash of the settlement for on-chain reference
        content_hash = hashlib.sha256(
            json.dumps(
                {
                    "settlement_id": record.settlement_id,
                    "claim": record.claim.statement,
                    "outcome": record.outcome.value if record.outcome else "",
                    "score": record.score,
                    "evidence": record.outcome_evidence,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        try:
            if hasattr(adapter, "push_reputation"):
                adapter.push_reputation(
                    agent_id=agent,
                    score=reputation,
                    domain="settlement",
                    metadata={
                        "settlement_id": record.settlement_id,
                        "debate_id": record.claim.debate_id,
                        "outcome": record.outcome.value if record.outcome else "",
                        "brier_component": round(brier_component, 4),
                        "confidence": record.claim.confidence,
                        "content_hash": content_hash,
                    },
                )
                logger.info(
                    "ERC-8004 reputation pushed: agent=%s score=%d outcome=%s",
                    agent,
                    reputation,
                    record.outcome.value if record.outcome else "none",
                )
        except (ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("ERC-8004 reputation push failed for %s: %s", agent, e)

        try:
            if hasattr(adapter, "push_validation"):
                # Map outcome to validation response
                from aragora.blockchain.models import ValidationResponse

                response = (
                    ValidationResponse.PASS if record.score >= 0.5 else ValidationResponse.FAIL
                )
                adapter.push_validation(
                    agent_id=agent,
                    request_hash=content_hash,
                    response=response,
                    tag="settlement",
                    metadata={
                        "settlement_id": record.settlement_id,
                        "debate_id": record.claim.debate_id,
                    },
                )
        except (ImportError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.debug("ERC-8004 validation push failed for %s: %s", agent, e)


# ---------------------------------------------------------------------------
# EventBus settlement hook
# ---------------------------------------------------------------------------


class EventBusSettlementHook:
    """Emits settlement events on the debate EventBus for real-time dashboards.

    Events emitted:
    - ``settlement_claims_extracted``: when claims are extracted from a debate
    - ``settlement_resolved``: when a claim is settled against reality
    """

    def __init__(self, event_bus: Any) -> None:
        self._event_bus = event_bus

    def on_claims_extracted(self, batch: SettlementBatch) -> None:
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(
                "settlement_claims_extracted",
                debate_id=batch.debate_id,
                settlements_created=batch.settlements_created,
                settlement_ids=batch.settlement_ids,
                claims_skipped=batch.claims_skipped,
            )
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.debug("EventBus emit failed for claims_extracted: %s", e)

    def on_settled(self, record: SettlementRecord, result: SettleResult) -> None:
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit(
                "settlement_resolved",
                settlement_id=record.settlement_id,
                debate_id=record.claim.debate_id,
                agent=record.claim.author,
                outcome=result.outcome.value,
                score=result.score,
                elo_updates=result.elo_updates,
                calibration_recorded=result.calibration_recorded,
            )
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.debug("EventBus emit failed for settlement_resolved: %s", e)


# ---------------------------------------------------------------------------
# Logging settlement hook (lightweight, always-on audit trail)
# ---------------------------------------------------------------------------


class LoggingSettlementHook:
    """Structured log output for settlement events — useful as a baseline hook."""

    def on_claims_extracted(self, batch: SettlementBatch) -> None:
        logger.info(
            "Settlement hook: %d claims extracted from debate %s",
            batch.settlements_created,
            batch.debate_id,
        )

    def on_settled(self, record: SettlementRecord, result: SettleResult) -> None:
        logger.info(
            "Settlement hook: %s settled as %s (score=%.1f, agent=%s, debate=%s)",
            record.settlement_id,
            result.outcome.value,
            result.score,
            record.claim.author,
            record.claim.debate_id,
        )


__all__ = [
    "ERC8004SettlementHook",
    "EventBusSettlementHook",
    "LoggingSettlementHook",
    "SettlementHook",
    "SettlementHookRegistry",
]
