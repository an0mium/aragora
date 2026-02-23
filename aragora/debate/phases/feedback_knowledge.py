"""
Knowledge feedback methods for FeedbackPhase.

Extracted from feedback_phase.py for maintainability.
Handles knowledge mound ingestion, knowledge extraction from debates,
evidence storage, culture observation, KM validation, and confidence
reinforcement.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

if TYPE_CHECKING:
    from aragora.debate.context import DebateContext

logger = logging.getLogger(__name__)


class KnowledgeFeedback:
    """Handles knowledge-related feedback operations."""

    def __init__(
        self,
        knowledge_mound: Any | None = None,
        enable_knowledge_ingestion: bool = True,
        ingest_debate_outcome: Callable[[Any], Any] | None = None,
        knowledge_bridge_hub: Any | None = None,
        enable_knowledge_extraction: bool = False,
        extraction_min_confidence: float = 0.3,
        extraction_promote_threshold: float = 0.6,
    ):
        self.knowledge_mound = knowledge_mound
        self.enable_knowledge_ingestion = enable_knowledge_ingestion
        self._ingest_debate_outcome = ingest_debate_outcome
        self.knowledge_bridge_hub = knowledge_bridge_hub
        self.enable_knowledge_extraction = enable_knowledge_extraction
        self.extraction_min_confidence = extraction_min_confidence
        self.extraction_promote_threshold = extraction_promote_threshold

    async def ingest_knowledge_outcome(self, ctx: DebateContext) -> None:
        """Ingest debate outcome into Knowledge Mound for future retrieval.

        Stores high-confidence debate conclusions in the unified knowledge
        superstructure so they can inform future debates on related topics.
        """
        if not self.knowledge_mound or not self.enable_knowledge_ingestion:
            return

        if not self._ingest_debate_outcome:
            return

        result = ctx.result
        if not result:
            return

        try:
            await self._ingest_debate_outcome(result)
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            logger.warning("[knowledge_mound] Failed to ingest outcome: %s", e)

    async def extract_knowledge_from_debate(self, ctx: DebateContext) -> None:
        """Extract structured knowledge (claims, relationships) from debate.

        Uses the Knowledge Mound's extraction capabilities to identify:
        - Facts and definitions stated by agents
        - Relationships between concepts
        - Consensus-backed conclusions (boosted confidence)

        Extracted knowledge is optionally promoted to the mound if it meets
        the promotion threshold.
        """
        if not self.knowledge_mound or not self.enable_knowledge_extraction:
            return

        result = ctx.result
        if not result:
            return

        # Check minimum confidence threshold
        confidence = getattr(result, "confidence", 0.0)
        if confidence < self.extraction_min_confidence:
            logger.debug(
                "[knowledge_extraction] Skipping: confidence %.2f < threshold %.2f",
                confidence,
                self.extraction_min_confidence,
            )
            return

        # Convert messages to extraction format
        messages = getattr(result, "messages", [])
        if not messages:
            return

        extraction_messages = []
        for msg in messages:
            extraction_messages.append(
                {
                    "agent_id": getattr(msg, "agent", None),
                    "content": getattr(msg, "content", ""),
                    "round": getattr(msg, "round", 0),
                }
            )

        # Get consensus text if available
        consensus_text = getattr(result, "final_answer", None) if result.consensus_reached else None

        # Get topic from environment
        topic = getattr(ctx.env, "task", None)

        try:
            # Extract knowledge using mound's extraction mixin
            extraction_result = await self.knowledge_mound.extract_from_debate(
                debate_id=ctx.debate_id,
                messages=extraction_messages,
                consensus_text=consensus_text,
                topic=topic,
            )

            claims_count = len(extraction_result.claims) if extraction_result.claims else 0
            rels_count = (
                len(extraction_result.relationships) if extraction_result.relationships else 0
            )

            logger.info(
                "[knowledge_extraction] Extracted %d claims and %d relationships from debate %s",
                claims_count,
                rels_count,
                ctx.debate_id,
            )

            # Optionally promote high-confidence claims to mound
            if claims_count > 0 and hasattr(self.knowledge_mound, "promote_extracted_knowledge"):
                workspace_id = getattr(ctx, "workspace_id", "default")
                promoted = await self.knowledge_mound.promote_extracted_knowledge(
                    workspace_id=workspace_id,
                    claims=extraction_result.claims,
                    min_confidence=self.extraction_promote_threshold,
                )
                if promoted > 0:
                    logger.info(
                        "[knowledge_extraction] Promoted %d claims to Knowledge Mound",
                        promoted,
                    )

        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            logger.warning("[knowledge_extraction] Failed to extract knowledge: %s", e)
        except Exception as e:  # noqa: BLE001 - phase isolation
            # Catch-all for unexpected errors; don't fail the feedback phase
            logger.error("[knowledge_extraction] Unexpected error during extraction: %s", e)

    async def store_evidence_in_mound(self, ctx: DebateContext) -> None:
        """Store collected evidence in Knowledge Mound via EvidenceBridge.

        Persists evidence gathered during the debate (sources, quotes, data)
        in the Knowledge Mound so it can inform future debates on related topics.
        """
        if not self.knowledge_bridge_hub:
            return

        evidence_items = getattr(ctx, "collected_evidence", [])
        if not evidence_items:
            return

        try:
            evidence_bridge = self.knowledge_bridge_hub.evidence
            stored_count = 0

            for evidence in evidence_items:
                try:
                    await evidence_bridge.store_from_collector_evidence(evidence)
                    stored_count += 1
                except (TypeError, ValueError, AttributeError) as e:
                    logger.debug("[evidence] Failed to store single item: %s", e)

            if stored_count > 0:
                logger.info(
                    "[evidence] Stored %d/%d evidence items in mound for debate %s",
                    stored_count,
                    len(evidence_items),
                    ctx.debate_id,
                )

        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            logger.warning("[evidence] Failed to store evidence: %s", e)

    async def observe_debate_culture(self, ctx: DebateContext) -> None:
        """Observe debate for organizational culture patterns.

        Extracts patterns from debate outcomes to build institutional knowledge
        about effective debate strategies, common arguments, and consensus patterns.
        This feeds into the CultureAccumulator for organizational learning.
        """
        if not self.knowledge_mound or not ctx.result:
            return

        try:
            # Check if the knowledge mound has observe_debate method
            if not hasattr(self.knowledge_mound, "observe_debate"):
                return

            patterns = await self.knowledge_mound.observe_debate(ctx.result)

            if patterns:
                logger.info(
                    "[culture] Extracted %d patterns from debate %s",
                    len(patterns),
                    ctx.debate_id,
                )

        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            logger.debug("[culture] Pattern observation failed: %s", e)

    async def validate_km_outcome(self, ctx: DebateContext) -> None:
        """Validate Knowledge Mound entries against debate outcome.

        Creates a feedback loop where knowledge that contributed to successful
        debate outcomes gets confidence boosts, and knowledge that contributed
        to failures gets penalties.  This makes the organizational memory
        self-correcting over time.
        """
        if not self.knowledge_mound:
            return

        # Get KM item IDs that were used during this debate
        km_item_ids: list[str] = getattr(ctx, "_km_item_ids_used", [])
        if not km_item_ids:
            return

        result = ctx.result
        if not result:
            return

        try:
            from aragora.debate.km_outcome_bridge import KMOutcomeBridge
            from aragora.debate.outcome_tracker import ConsensusOutcome

            # Create a ConsensusOutcome from the debate result
            outcome = ConsensusOutcome(
                debate_id=ctx.debate_id,
                consensus_text=getattr(result, "consensus_text", "") or "",
                consensus_confidence=getattr(result, "confidence", 0.5),
                implementation_attempted=True,
                implementation_succeeded=result.consensus_reached,
            )

            bridge = KMOutcomeBridge(
                outcome_tracker=None,
                knowledge_mound=self.knowledge_mound,
            )

            validations = await bridge.validate_knowledge_from_outcome(
                outcome=outcome,
                km_item_ids=km_item_ids,
            )

            if validations:
                logger.info(
                    "[km_outcome] Validated %d KM items (debate=%s, success=%s)",
                    len(validations),
                    ctx.debate_id,
                    outcome.implementation_succeeded,
                )

        except ImportError:
            logger.debug("[km_outcome] KMOutcomeBridge not available")
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.debug("[km_outcome] KM outcome validation failed: %s", e)

    async def reinforce_km_confidence(self, ctx: DebateContext) -> None:
        """Reinforce Knowledge Mound item confidence based on debate outcome.

        Implements the backward flow of the KM feedback loop: items that
        contributed to high-confidence consensus get a confidence boost,
        while items associated with low-confidence outcomes get a slight
        decrease.  This creates a reinforcement signal that makes the
        organizational knowledge self-improving over time.

        This is complementary to ``validate_km_outcome`` which uses the
        KMOutcomeBridge (when available).  This method provides a direct,
        lightweight fallback that always works when a KnowledgeMound and
        MemoryManager are available.
        """
        if not self.knowledge_mound:
            return

        km_item_ids: list[str] = getattr(ctx, "_km_item_ids_used", [])
        if not km_item_ids:
            return

        result = ctx.result
        if not result:
            return

        try:
            from aragora.debate.memory_manager import MemoryManager

            # Create a minimal MemoryManager just for the confidence update
            mm = MemoryManager()
            await mm.update_km_item_confidence(
                result=result,
                km_item_ids=km_item_ids,
                knowledge_mound=self.knowledge_mound,
            )
        except ImportError:
            logger.debug("[km_feedback] MemoryManager not available for confidence reinforcement")
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.debug("[km_feedback] KM confidence reinforcement failed: %s", e)


__all__ = ["KnowledgeFeedback"]
