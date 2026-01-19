"""
Knowledge Mound Operations for debate context and outcome storage.

Handles the integration between debate orchestration and the Knowledge Mound:
- Fetching relevant knowledge for debate context
- Ingesting debate outcomes for future retrieval
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.core import DebateResult, Environment
    from aragora.knowledge.mound.types import KnowledgeMound

logger = logging.getLogger(__name__)


class KnowledgeMoundOperations:
    """Operations for integrating Knowledge Mound with debate orchestration.

    Provides methods for:
    - Fetching relevant knowledge to inform debate context
    - Ingesting high-confidence debate outcomes into organizational memory
    """

    def __init__(
        self,
        knowledge_mound: Optional["KnowledgeMound"] = None,
        enable_retrieval: bool = True,
        enable_ingestion: bool = True,
    ):
        """Initialize Knowledge Mound operations.

        Args:
            knowledge_mound: The Knowledge Mound instance (optional)
            enable_retrieval: Whether to enable knowledge retrieval
            enable_ingestion: Whether to enable outcome ingestion
        """
        self.knowledge_mound = knowledge_mound
        self.enable_retrieval = enable_retrieval
        self.enable_ingestion = enable_ingestion

    async def fetch_knowledge_context(
        self, task: str, limit: int = 10
    ) -> Optional[str]:
        """Fetch relevant knowledge from Knowledge Mound for debate context.

        Queries the unified knowledge superstructure for semantically related
        knowledge items to inform the debate.

        Args:
            task: The debate task to find relevant knowledge for
            limit: Maximum number of knowledge items to retrieve

        Returns:
            Formatted string with knowledge context, or None if unavailable
        """
        if not self.knowledge_mound or not self.enable_retrieval:
            return None

        try:
            # Query mound for semantically related knowledge
            results = await self.knowledge_mound.query_semantic(
                query=task,
                limit=limit,
                min_confidence=0.5,
            )

            if not results or not results.items:
                return None

            # Format knowledge for agent context
            lines = ["## KNOWLEDGE MOUND CONTEXT"]
            lines.append("Relevant knowledge from organizational memory:\n")

            for item in results.items[:limit]:
                source = getattr(item, "source", "unknown")
                confidence = getattr(item, "confidence", 0.0)
                content = getattr(item, "content", str(item))[:300]
                lines.append(f"**[{source}]** (confidence: {confidence:.0%})")
                lines.append(f"{content}")
                lines.append("")

            logger.info(f"  [knowledge_mound] Retrieved {len(results.items)} items for context")
            return "\n".join(lines)

        except Exception as e:
            logger.warning(f"  [knowledge_mound] Failed to fetch context: {e}")
            return None

    async def ingest_debate_outcome(
        self,
        result: "DebateResult",
        env: Optional["Environment"] = None,
    ) -> None:
        """Store debate outcome in Knowledge Mound for future retrieval.

        Ingests the consensus conclusion and key claims from high-confidence
        debates into the organizational knowledge superstructure.

        Args:
            result: The debate result to ingest
            env: The debate environment with task details
        """
        if not self.knowledge_mound or not self.enable_ingestion:
            return

        # Only ingest high-quality outcomes (consensus with decent confidence)
        if not result.final_answer or result.confidence < 0.5:
            logger.debug("  [knowledge_mound] Skipping low-confidence debate outcome")
            return

        try:
            from aragora.knowledge.mound.types import IngestionRequest, KnowledgeSource

            # Build metadata from debate result
            metadata = {
                "debate_id": result.id,
                "task": env.task[:500] if env else "",
                "confidence": result.confidence,
                "consensus_reached": result.consensus_reached,
                "rounds_used": result.rounds_used,
                "participants": result.participants[:10] if result.participants else [],
                "winner": result.winner,
            }

            # Add belief cruxes if available
            if hasattr(result, "debate_cruxes") and result.debate_cruxes:
                metadata["crux_claims"] = [
                    str(c.get("claim", c))[:200] for c in result.debate_cruxes[:5]
                ]

            # Ingest the consensus conclusion
            ingestion_result = await self.knowledge_mound.store(
                IngestionRequest(
                    content=result.final_answer,
                    source=KnowledgeSource.DEBATE,
                    workspace_id=self.knowledge_mound.workspace_id,
                    metadata=metadata,
                )
            )

            if ingestion_result and ingestion_result.node_id:
                logger.info(
                    f"  [knowledge_mound] Ingested debate outcome (node_id={ingestion_result.node_id})"
                )

                # Record analytics if tracker available
                if hasattr(self, "analytics_tracker") and self.analytics_tracker:
                    self.analytics_tracker.record_event(
                        "knowledge_mound_ingestion",
                        details="Stored debate conclusion in Knowledge Mound",
                        debate_id=result.id,
                    )

        except Exception as e:
            logger.warning(f"  [knowledge_mound] Failed to ingest outcome: {e}")
