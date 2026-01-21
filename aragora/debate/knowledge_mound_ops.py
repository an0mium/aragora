"""
Knowledge Mound Operations for debate context and outcome storage.

Handles the integration between debate orchestration and the Knowledge Mound:
- Fetching relevant knowledge for debate context
- Ingesting debate outcomes for future retrieval
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from aragora.core import DebateResult, Environment
    from aragora.knowledge.mound.types import KnowledgeMound  # type: ignore[attr-defined]
    from aragora.knowledge.mound.metrics import KMMetrics

logger = logging.getLogger(__name__)

# Type alias for notification callback
NotifyCallback = Callable[[str, Any], None]


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
        notify_callback: Optional[NotifyCallback] = None,
        metrics: Optional["KMMetrics"] = None,
    ):
        """Initialize Knowledge Mound operations.

        Args:
            knowledge_mound: The Knowledge Mound instance (optional)
            enable_retrieval: Whether to enable knowledge retrieval
            enable_ingestion: Whether to enable outcome ingestion
            notify_callback: Optional callback for spectator/dashboard notifications
            metrics: Optional KMMetrics instance for observability
        """
        self.knowledge_mound = knowledge_mound
        self.enable_retrieval = enable_retrieval
        self.enable_ingestion = enable_ingestion
        self._notify_callback = notify_callback
        self._metrics = metrics

    async def fetch_knowledge_context(self, task: str, limit: int = 10) -> Optional[str]:
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

        start_time = time.time()
        success = True
        error_msg = None

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
            success = False
            error_msg = str(e)
            logger.warning(f"  [knowledge_mound] Failed to fetch context: {e}")
            return None

        finally:
            # Record metrics if available
            if self._metrics:
                latency_ms = (time.time() - start_time) * 1000
                try:
                    from aragora.knowledge.mound.metrics import OperationType

                    self._metrics.record(
                        OperationType.QUERY,
                        latency_ms,
                        success=success,
                        error=error_msg,
                        metadata={"operation": "fetch_knowledge_context"},
                    )
                except ImportError:
                    pass

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

        # Only ingest high-quality outcomes (consensus with strong confidence)
        # Use 0.85 threshold to ensure knowledge mound contains reliable conclusions
        if not result.final_answer or result.confidence < 0.85:
            logger.debug(
                "  [knowledge_mound] Skipping low-confidence debate outcome (need >= 0.85)"
            )
            return

        start_time = time.time()
        success = False
        error_msg = None

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
                    content=f"Debate Conclusion: {result.final_answer[:2000]}",
                    source_type=KnowledgeSource.DEBATE,
                    workspace_id=self.knowledge_mound.workspace_id,
                    metadata=metadata,
                )
            )

            success = ingestion_result and getattr(ingestion_result, "node_id", None)
            if success:
                logger.info(
                    f"  [knowledge_mound] Ingested debate outcome (node_id={ingestion_result.node_id})"
                )

                # Emit event for dashboard if callback provided
                if self._notify_callback:
                    self._notify_callback(
                        "knowledge_ingested",
                        {
                            "details": "Stored debate conclusion in Knowledge Mound",
                            "metric": result.confidence,
                        },
                    )

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"  [knowledge_mound] Failed to ingest outcome: {e}")

        finally:
            # Record metrics if available
            if self._metrics:
                latency_ms = (time.time() - start_time) * 1000
                try:
                    from aragora.knowledge.mound.metrics import OperationType

                    self._metrics.record(
                        OperationType.STORE,
                        latency_ms,
                        success=success,
                        error=error_msg,
                        metadata={"operation": "ingest_debate_outcome"},
                    )
                except ImportError:
                    pass
