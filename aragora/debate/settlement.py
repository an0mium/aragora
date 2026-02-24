"""Settlement Tracker -- maps debate claims to measurable future outcomes.

When a debate produces verifiable predictions (claims with measurable criteria),
the settlement tracker:

1. Extracts verifiable claims from debate results
2. Stores them with expected resolution dates and verification criteria
3. Provides settle() to score claims as correct / incorrect / partial
4. Updates agent ELO ratings based on prediction accuracy
5. Feeds calibration data back to CalibrationTracker
6. Persists settlement history to the Knowledge Mound

Usage:
    tracker = SettlementTracker()

    # After a debate completes, extract pending settlements
    pending = tracker.extract_verifiable_claims(debate_id, debate_result)

    # Later, when outcomes are known, settle them
    result = tracker.settle(settlement_id, outcome=True, evidence="...")

    # Query pending settlements
    pending = tracker.get_pending(debate_id="abc-123")
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class SettlementStatus(str, Enum):
    """Status of a settlement claim."""

    PENDING = "pending"
    SETTLED_CORRECT = "settled_correct"
    SETTLED_INCORRECT = "settled_incorrect"
    SETTLED_PARTIAL = "settled_partial"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class SettlementOutcome(str, Enum):
    """Outcome of a settlement resolution."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"


@dataclass
class VerifiableClaim:
    """A claim extracted from a debate that can be verified against future outcomes.

    Attributes:
        claim_id: Unique identifier for this claim.
        debate_id: The debate that produced this claim.
        statement: The claim text.
        author: The agent that made the claim.
        confidence: The confidence expressed by the agent (0-1).
        verification_criteria: How to determine if the claim is correct.
        expected_resolution_date: When the claim should be resolvable.
        domain: Problem domain for calibration bucketing.
        metadata: Additional context from the debate.
    """

    claim_id: str
    debate_id: str
    statement: str
    author: str
    confidence: float
    verification_criteria: str = ""
    expected_resolution_date: str | None = None
    domain: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SettlementRecord:
    """A settlement record tracking a verifiable claim through resolution.

    Attributes:
        settlement_id: Unique identifier.
        claim: The verifiable claim being tracked.
        status: Current settlement status.
        outcome: Resolution outcome (set after settlement).
        outcome_evidence: Evidence for the outcome.
        score: Numeric score (1.0 = correct, 0.5 = partial, 0.0 = incorrect).
        settled_at: When the settlement was resolved.
        settled_by: Who/what resolved the settlement.
    """

    settlement_id: str
    claim: VerifiableClaim
    status: SettlementStatus = SettlementStatus.PENDING
    outcome: SettlementOutcome | None = None
    outcome_evidence: str = ""
    score: float = 0.0
    settled_at: str | None = None
    settled_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["outcome"] = self.outcome.value if self.outcome else None
        return d


@dataclass
class SettlementBatch:
    """Result of a batch settlement operation."""

    debate_id: str
    settlements_created: int
    settlement_ids: list[str]
    claims_skipped: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SettleResult:
    """Result of settling a single claim."""

    settlement_id: str
    outcome: SettlementOutcome
    score: float
    elo_updates: dict[str, float] = field(default_factory=dict)
    calibration_recorded: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d


# ---------------------------------------------------------------------------
# Claim extraction helpers
# ---------------------------------------------------------------------------

# Patterns that indicate a verifiable/predictive claim
_VERIFIABLE_KEYWORDS = [
    "will ",
    "should result in",
    "predict",
    "expect",
    "forecast",
    "estimate",
    "likely to",
    "probability",
    "within ",
    "by ",
    "increase",
    "decrease",
    "improve",
    "reduce",
    "achieve",
    "reach",
    "exceed",
]


def _is_verifiable(statement: str) -> bool:
    """Check if a claim statement contains verifiable/predictive language."""
    lower = statement.lower()
    return any(kw in lower for kw in _VERIFIABLE_KEYWORDS)


def _generate_settlement_id(debate_id: str, claim_text: str) -> str:
    """Generate a deterministic settlement ID from debate + claim."""
    content = f"{debate_id}:{claim_text}"
    h = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"stl-{h}"


# ---------------------------------------------------------------------------
# Settlement Tracker
# ---------------------------------------------------------------------------


class SettlementTracker:
    """Tracks verifiable claims from debates and settles them against outcomes.

    In-memory store with optional persistence to Knowledge Mound.  The tracker
    is designed to be instantiated once per server and shared across debates.

    Args:
        elo_system: Optional EloSystem instance for rating updates.
        calibration_tracker: Optional CalibrationTracker for calibration data.
        knowledge_mound: Optional KnowledgeMound for persistent storage.
    """

    def __init__(
        self,
        elo_system: Any | None = None,
        calibration_tracker: Any | None = None,
        knowledge_mound: Any | None = None,
    ) -> None:
        self._elo_system = elo_system
        self._calibration_tracker = calibration_tracker
        self._knowledge_mound = knowledge_mound

        # In-memory store: settlement_id -> SettlementRecord
        self._records: dict[str, SettlementRecord] = {}
        # Index: debate_id -> list of settlement_ids
        self._debate_index: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_verifiable_claims(
        self,
        debate_id: str,
        debate_result: Any,
        *,
        min_confidence: float = 0.3,
        domain: str = "general",
    ) -> SettlementBatch:
        """Extract verifiable claims from a debate result and register them.

        Scans the debate result for claims that contain predictive or
        measurable language, then creates settlement records for each.

        Args:
            debate_id: The debate that produced the claims.
            debate_result: The debate result object (has messages, claims, etc.).
            min_confidence: Minimum confidence to consider a claim verifiable.
            domain: Problem domain for calibration bucketing.

        Returns:
            SettlementBatch summarising what was created.
        """
        claims = self._extract_claims_from_result(debate_result, debate_id, domain)

        settlement_ids: list[str] = []
        skipped = 0

        for claim in claims:
            if claim.confidence < min_confidence:
                skipped += 1
                continue
            if not _is_verifiable(claim.statement):
                skipped += 1
                continue

            sid = _generate_settlement_id(debate_id, claim.statement)

            # Avoid duplicates
            if sid in self._records:
                skipped += 1
                continue

            record = SettlementRecord(
                settlement_id=sid,
                claim=claim,
            )
            self._records[sid] = record
            self._debate_index.setdefault(debate_id, []).append(sid)
            settlement_ids.append(sid)

        batch = SettlementBatch(
            debate_id=debate_id,
            settlements_created=len(settlement_ids),
            settlement_ids=settlement_ids,
            claims_skipped=skipped,
        )

        if settlement_ids:
            logger.info(
                "Extracted %d verifiable claims from debate %s (%d skipped)",
                len(settlement_ids),
                debate_id,
                skipped,
            )

        return batch

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle(
        self,
        settlement_id: str,
        outcome: str | SettlementOutcome,
        evidence: str = "",
        settled_by: str = "manual",
    ) -> SettleResult:
        """Settle a claim by providing the actual outcome.

        Args:
            settlement_id: The settlement to resolve.
            outcome: "correct", "incorrect", or "partial".
            evidence: Supporting evidence for the outcome.
            settled_by: Who/what resolved the settlement.

        Returns:
            SettleResult with score and any ELO/calibration updates.

        Raises:
            KeyError: If settlement_id is not found.
            ValueError: If the settlement is already resolved.
        """
        if settlement_id not in self._records:
            raise KeyError(f"Settlement not found: {settlement_id}")

        record = self._records[settlement_id]
        if record.status != SettlementStatus.PENDING:
            raise ValueError(f"Settlement {settlement_id} already resolved: {record.status.value}")

        # Parse outcome
        if isinstance(outcome, str):
            outcome = SettlementOutcome(outcome)

        # Score: 1.0 for correct, 0.5 for partial, 0.0 for incorrect
        score_map = {
            SettlementOutcome.CORRECT: 1.0,
            SettlementOutcome.PARTIAL: 0.5,
            SettlementOutcome.INCORRECT: 0.0,
        }
        score = score_map[outcome]

        # Update record
        status_map = {
            SettlementOutcome.CORRECT: SettlementStatus.SETTLED_CORRECT,
            SettlementOutcome.PARTIAL: SettlementStatus.SETTLED_PARTIAL,
            SettlementOutcome.INCORRECT: SettlementStatus.SETTLED_INCORRECT,
        }
        record.status = status_map[outcome]
        record.outcome = outcome
        record.outcome_evidence = evidence
        record.score = score
        record.settled_at = datetime.now(timezone.utc).isoformat()
        record.settled_by = settled_by

        # Feed back to systems
        elo_updates = self._update_elo(record)
        calibration_recorded = self._update_calibration(record)
        self._persist_to_knowledge_mound(record)

        logger.info(
            "Settled %s as %s (score=%.1f, agent=%s, debate=%s)",
            settlement_id,
            outcome.value,
            score,
            record.claim.author,
            record.claim.debate_id,
        )

        return SettleResult(
            settlement_id=settlement_id,
            outcome=outcome,
            score=score,
            elo_updates=elo_updates,
            calibration_recorded=calibration_recorded,
        )

    def settle_batch(
        self,
        settlements: list[dict[str, Any]],
        settled_by: str = "manual",
    ) -> list[SettleResult]:
        """Settle multiple claims at once.

        Args:
            settlements: List of dicts with keys: settlement_id, outcome, evidence.
            settled_by: Who/what resolved the settlements.

        Returns:
            List of SettleResult for each settlement.
        """
        results = []
        for entry in settlements:
            sid = entry.get("settlement_id", "")
            outcome = entry.get("outcome", "")
            evidence = entry.get("evidence", "")
            try:
                result = self.settle(sid, outcome, evidence, settled_by)
                results.append(result)
            except (KeyError, ValueError) as e:
                logger.warning("Batch settle failed for %s: %s", sid, e)
        return results

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_pending(
        self,
        debate_id: str | None = None,
        domain: str | None = None,
        limit: int = 100,
    ) -> list[SettlementRecord]:
        """Get pending (unsettled) settlements.

        Args:
            debate_id: Filter by debate ID.
            domain: Filter by domain.
            limit: Maximum number of records to return.

        Returns:
            List of pending SettlementRecord objects.
        """
        results: list[SettlementRecord] = []

        if debate_id:
            sids = self._debate_index.get(debate_id, [])
            candidates = [self._records[sid] for sid in sids if sid in self._records]
        else:
            candidates = list(self._records.values())

        for record in candidates:
            if record.status != SettlementStatus.PENDING:
                continue
            if domain and record.claim.domain != domain:
                continue
            results.append(record)
            if len(results) >= limit:
                break

        return results

    def get_settlement(self, settlement_id: str) -> SettlementRecord | None:
        """Get a specific settlement record."""
        return self._records.get(settlement_id)

    def get_history(
        self,
        debate_id: str | None = None,
        author: str | None = None,
        limit: int = 100,
    ) -> list[SettlementRecord]:
        """Get settlement history (resolved settlements).

        Args:
            debate_id: Filter by debate ID.
            author: Filter by claim author.
            limit: Maximum number of records to return.

        Returns:
            List of resolved SettlementRecord objects.
        """
        results: list[SettlementRecord] = []

        if debate_id:
            sids = self._debate_index.get(debate_id, [])
            candidates = [self._records[sid] for sid in sids if sid in self._records]
        else:
            candidates = list(self._records.values())

        for record in candidates:
            if record.status == SettlementStatus.PENDING:
                continue
            if author and record.claim.author != author:
                continue
            results.append(record)
            if len(results) >= limit:
                break

        return results

    def get_agent_accuracy(self, agent: str) -> dict[str, Any]:
        """Get accuracy statistics for a specific agent.

        Returns:
            Dict with total, correct, incorrect, partial counts and accuracy ratio.
        """
        total = 0
        correct = 0
        incorrect = 0
        partial = 0
        score_sum = 0.0

        for record in self._records.values():
            if record.claim.author != agent:
                continue
            if record.status == SettlementStatus.PENDING:
                continue
            total += 1
            score_sum += record.score
            if record.outcome == SettlementOutcome.CORRECT:
                correct += 1
            elif record.outcome == SettlementOutcome.INCORRECT:
                incorrect += 1
            elif record.outcome == SettlementOutcome.PARTIAL:
                partial += 1

        return {
            "agent": agent,
            "total_settled": total,
            "correct": correct,
            "incorrect": incorrect,
            "partial": partial,
            "accuracy": score_sum / total if total > 0 else 0.0,
            "brier_score": self._compute_brier_score(agent),
        }

    def _compute_brier_score(self, agent: str) -> float:
        """Compute Brier score for an agent's predictions.

        Brier score measures calibration: how well confidence matches outcomes.
        Lower is better (0 = perfect calibration).
        """
        n = 0
        brier_sum = 0.0

        for record in self._records.values():
            if record.claim.author != agent:
                continue
            if record.status == SettlementStatus.PENDING:
                continue

            # Brier score: (confidence - outcome)^2
            outcome_val = record.score  # 1.0, 0.5, or 0.0
            brier_sum += (record.claim.confidence - outcome_val) ** 2
            n += 1

        return brier_sum / n if n > 0 else 0.0

    # ------------------------------------------------------------------
    # Integration: ELO
    # ------------------------------------------------------------------

    def _update_elo(self, record: SettlementRecord) -> dict[str, float]:
        """Update ELO ratings based on settlement outcome.

        Uses the settlement score as a match result against a virtual
        "ground truth" opponent. Correct predictions boost rating,
        incorrect predictions lower it.

        Returns:
            Dict mapping agent name to ELO change.
        """
        if self._elo_system is None:
            return {}

        try:
            agent = record.claim.author
            domain = record.claim.domain

            # Use record_match with a virtual "ground_truth" opponent.
            # Score of 1.0 means the agent "won" (correct prediction).
            scores = {agent: record.score, "__ground_truth__": 1.0 - record.score}
            changes = self._elo_system.record_match(
                debate_id=record.claim.debate_id,
                participants=[agent, "__ground_truth__"],
                scores=scores,
                domain=domain,
                confidence_weight=record.claim.confidence,
            )
            # Only return the agent's change, not the virtual opponent's
            return {agent: changes.get(agent, 0.0)}
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.debug("ELO update failed for settlement %s: %s", record.settlement_id, e)
            return {}

    # ------------------------------------------------------------------
    # Integration: Calibration
    # ------------------------------------------------------------------

    def _update_calibration(self, record: SettlementRecord) -> bool:
        """Feed settlement result to the CalibrationTracker.

        Records the agent's expressed confidence against the binary outcome
        so the calibration system can compute calibration curves.

        Returns:
            True if calibration was recorded successfully.
        """
        if self._calibration_tracker is None:
            return False

        try:
            # For calibration, we use a binary correct/incorrect.
            # Partial outcomes are recorded as correct (score >= 0.5).
            correct = record.score >= 0.5
            self._calibration_tracker.record_prediction(
                agent=record.claim.author,
                confidence=record.claim.confidence,
                correct=correct,
                domain=record.claim.domain,
                debate_id=record.claim.debate_id,
                prediction_type="settlement",
            )
            return True
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(
                "Calibration update failed for settlement %s: %s",
                record.settlement_id,
                e,
            )
            return False

    # ------------------------------------------------------------------
    # Integration: Knowledge Mound
    # ------------------------------------------------------------------

    def _persist_to_knowledge_mound(self, record: SettlementRecord) -> bool:
        """Persist a settled record to the Knowledge Mound for long-term storage.

        Stores the settlement as a knowledge item with the outcome and
        calibration metadata so future debates can reference past predictions.

        Returns:
            True if persistence succeeded.
        """
        if self._knowledge_mound is None:
            return False

        try:
            from aragora.knowledge.unified.types import (
                ConfidenceLevel,
                KnowledgeItem,
                KnowledgeSource,
            )

            confidence_level = ConfidenceLevel.from_float(record.claim.confidence)

            item = KnowledgeItem(
                id=f"settlement:{record.settlement_id}",
                content=json.dumps(
                    {
                        "statement": record.claim.statement,
                        "outcome": record.outcome.value if record.outcome else None,
                        "score": record.score,
                        "author": record.claim.author,
                        "evidence": record.outcome_evidence,
                    }
                ),
                source=KnowledgeSource.DEBATE,
                confidence=confidence_level,
                metadata={
                    "type": "settlement",
                    "debate_id": record.claim.debate_id,
                    "settlement_id": record.settlement_id,
                    "domain": record.claim.domain,
                    "settled_at": record.settled_at or "",
                    "brier_component": (record.claim.confidence - record.score) ** 2,
                },
            )

            # Use sync-safe store if available
            if hasattr(self._knowledge_mound, "store_sync"):
                self._knowledge_mound.store_sync(item)
            elif hasattr(self._knowledge_mound, "store_item"):
                self._knowledge_mound.store_item(item)
            else:
                logger.debug("Knowledge Mound has no sync store method")
                return False

            return True
        except ImportError:
            logger.debug("Knowledge Mound types not available")
            return False
        except (AttributeError, TypeError, ValueError, RuntimeError, OSError) as e:
            logger.debug(
                "Knowledge Mound persistence failed for settlement %s: %s",
                record.settlement_id,
                e,
            )
            return False

    # ------------------------------------------------------------------
    # Internal: claim extraction from debate result
    # ------------------------------------------------------------------

    def _extract_claims_from_result(
        self,
        debate_result: Any,
        debate_id: str,
        domain: str,
    ) -> list[VerifiableClaim]:
        """Extract VerifiableClaim objects from a debate result.

        Tries multiple strategies:
        1. If result has a claims_kernel with typed claims, use those
        2. If result has messages, extract claims from message text
        3. If result has a final_answer, extract claims from that
        """
        claims: list[VerifiableClaim] = []

        # Strategy 1: ClaimsKernel typed claims
        kernel = getattr(debate_result, "claims_kernel", None)
        if kernel is not None:
            try:
                typed_claims = kernel.get_claims() if hasattr(kernel, "get_claims") else []
                for tc in typed_claims:
                    claims.append(
                        VerifiableClaim(
                            claim_id=getattr(tc, "claim_id", str(uuid.uuid4())),
                            debate_id=debate_id,
                            statement=getattr(tc, "statement", ""),
                            author=getattr(tc, "author", "unknown"),
                            confidence=getattr(tc, "confidence", 0.5),
                            domain=domain,
                            metadata={
                                "claim_type": getattr(tc, "claim_type", "assertion"),
                                "round_num": getattr(tc, "round_num", 0),
                            },
                        )
                    )
            except (AttributeError, TypeError) as e:
                logger.debug("ClaimsKernel extraction failed: %s", e)

        # Strategy 2: Messages with proposals/assertions
        messages = getattr(debate_result, "messages", None) or []
        if not claims and messages:
            try:
                from aragora.reasoning.claims import fast_extract_claims

                for msg in messages:
                    content = getattr(msg, "content", "") or getattr(msg, "text", "")
                    author = getattr(msg, "agent", "") or getattr(msg, "author", "unknown")
                    if not content:
                        continue
                    extracted = fast_extract_claims(str(content), str(author))
                    for ec in extracted:
                        claims.append(
                            VerifiableClaim(
                                claim_id=str(uuid.uuid4()),
                                debate_id=debate_id,
                                statement=ec.get("text", ""),
                                author=ec.get("author", "unknown"),
                                confidence=ec.get("confidence", 0.5),
                                domain=domain,
                            )
                        )
            except ImportError:
                logger.debug("fast_extract_claims not available")
            except (AttributeError, TypeError) as e:
                logger.debug("Message extraction failed: %s", e)

        # Strategy 3: Final answer
        final_answer = getattr(debate_result, "final_answer", None)
        if not claims and final_answer:
            try:
                from aragora.reasoning.claims import fast_extract_claims

                extracted = fast_extract_claims(str(final_answer), "consensus")
                for ec in extracted:
                    claims.append(
                        VerifiableClaim(
                            claim_id=str(uuid.uuid4()),
                            debate_id=debate_id,
                            statement=ec.get("text", ""),
                            author="consensus",
                            confidence=getattr(debate_result, "confidence", 0.5),
                            domain=domain,
                        )
                    )
            except ImportError:
                pass
            except (AttributeError, TypeError):
                pass

        return claims

    # ------------------------------------------------------------------
    # Summary / Stats
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Get an overall summary of settlement activity."""
        total = len(self._records)
        pending = sum(1 for r in self._records.values() if r.status == SettlementStatus.PENDING)
        settled = total - pending

        correct = sum(1 for r in self._records.values() if r.outcome == SettlementOutcome.CORRECT)
        incorrect = sum(
            1 for r in self._records.values() if r.outcome == SettlementOutcome.INCORRECT
        )
        partial = sum(1 for r in self._records.values() if r.outcome == SettlementOutcome.PARTIAL)

        # Collect per-agent stats
        agents: set[str] = set()
        for r in self._records.values():
            agents.add(r.claim.author)

        return {
            "total_settlements": total,
            "pending": pending,
            "settled": settled,
            "correct": correct,
            "incorrect": incorrect,
            "partial": partial,
            "accuracy": correct / settled if settled > 0 else 0.0,
            "agents_tracked": len(agents),
            "debates_tracked": len(self._debate_index),
        }


__all__ = [
    "SettlementBatch",
    "SettlementOutcome",
    "SettlementRecord",
    "SettlementStatus",
    "SettlementTracker",
    "SettleResult",
    "VerifiableClaim",
]
