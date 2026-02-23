"""
Knowledge Mound Operations for debate context and outcome storage.

Handles the integration between debate orchestration and the Knowledge Mound:
- Fetching relevant knowledge for debate context
- Ingesting debate outcomes for future retrieval
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

if TYPE_CHECKING:
    from aragora.core import DebateResult, Environment
    from aragora.knowledge.mound.facade import KnowledgeMound
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
        knowledge_mound: KnowledgeMound | None = None,
        enable_retrieval: bool = True,
        enable_ingestion: bool = True,
        notify_callback: NotifyCallback | None = None,
        metrics: KMMetrics | None = None,
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
        self._last_km_item_ids: list[str] = []  # IDs of KM items used in last fetch

    async def fetch_knowledge_context(
        self,
        task: str,
        limit: int = 10,
        auth_context: Any | None = None,
    ) -> str | None:
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
            results: Any = None
            try:
                if auth_context and hasattr(self.knowledge_mound, "query_with_visibility"):
                    actor_id = getattr(auth_context, "user_id", "") or ""
                    workspace_id = getattr(auth_context, "workspace_id", "") or ""
                    org_id = getattr(auth_context, "org_id", None)
                    if actor_id and workspace_id:
                        results = await self.knowledge_mound.query_with_visibility(
                            task,
                            actor_id=actor_id,
                            actor_workspace_id=workspace_id,
                            actor_org_id=org_id,
                            limit=limit,
                        )
                    else:
                        results = await self.knowledge_mound.query_semantic(
                            text=task,
                            limit=limit,
                            min_confidence=0.5,
                        )
                else:
                    results = await self.knowledge_mound.query_semantic(
                        text=task,
                        limit=limit,
                        min_confidence=0.5,
                    )
            except RuntimeError as e:
                if "not initialized" in str(e).lower() and hasattr(
                    self.knowledge_mound, "initialize"
                ):
                    await self.knowledge_mound.initialize()
                    if auth_context and hasattr(self.knowledge_mound, "query_with_visibility"):
                        actor_id = getattr(auth_context, "user_id", "") or ""
                        workspace_id = getattr(auth_context, "workspace_id", "") or ""
                        org_id = getattr(auth_context, "org_id", None)
                        if actor_id and workspace_id:
                            results = await self.knowledge_mound.query_with_visibility(
                                task,
                                actor_id=actor_id,
                                actor_workspace_id=workspace_id,
                                actor_org_id=org_id,
                                limit=limit,
                            )
                        else:
                            results = await self.knowledge_mound.query_semantic(
                                text=task,
                                limit=limit,
                                min_confidence=0.5,
                            )
                    else:
                        results = await self.knowledge_mound.query_semantic(
                            text=task,
                            limit=limit,
                            min_confidence=0.5,
                        )
                else:
                    raise

            items = None
            if isinstance(results, list):
                items = results
            elif hasattr(results, "items"):
                items = results.items

            if not items:
                self._last_km_item_ids = []
                return None

            # Track item IDs for outcome validation feedback loop
            self._last_km_item_ids = [
                getattr(item, "id", None) or getattr(item, "item_id", "")
                for item in items[:limit]
                if getattr(item, "id", None) or getattr(item, "item_id", "")
            ]

            # Format knowledge for agent context
            lines = ["## KNOWLEDGE MOUND CONTEXT"]
            lines.append("Relevant knowledge from organizational memory:\n")

            confidence_map = {
                "verified": 0.95,
                "high": 0.8,
                "medium": 0.6,
                "low": 0.3,
                "unverified": 0.2,
            }

            def _confidence_to_float(value: Any) -> float:
                if isinstance(value, (int, float)):
                    return float(value)
                if hasattr(value, "value"):
                    value = value.value
                if isinstance(value, str):
                    return confidence_map.get(value.lower(), 0.5)
                return 0.5

            for item in items[:limit]:
                source = getattr(item, "source", "unknown")
                confidence = _confidence_to_float(getattr(item, "confidence", 0.0))
                content = getattr(item, "content", str(item))[:300]
                lines.append(f"**[{source}]** (confidence: {confidence:.0%})")
                lines.append(f"{content}")
                lines.append("")

            logger.info("  [knowledge_mound] Retrieved %s items for context", len(items))
            return "\n".join(lines)

        except (
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            OSError,
            ConnectionError,
        ) as e:
            success = False
            error_msg = f"query_error:{type(e).__name__}"
            logger.warning("  [knowledge_mound] Failed to fetch context: %s", e)
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
                    logger.debug("KM telemetry not available")

    async def ingest_debate_outcome(
        self,
        result: DebateResult,
        env: Environment | None = None,
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

            success = bool(ingestion_result and getattr(ingestion_result, "node_id", None))
            if success:
                logger.info(
                    "  [knowledge_mound] Ingested debate outcome (node_id=%s)",
                    ingestion_result.node_id,
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

        except (RuntimeError, ValueError, OSError, KeyError) as e:
            error_msg = f"ingest_error:{type(e).__name__}"
            logger.warning("  [knowledge_mound] Failed to ingest outcome: %s", e)

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
                    logger.debug("KM telemetry not available")
