"""
RLM Operations Mixin for Knowledge Mound.

Provides RLM (Recursive Language Models) integration:
- query_with_rlm: Build hierarchical context from knowledge
- is_rlm_available: Check RLM availability
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Protocol

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import KnowledgeItem, MoundConfig
    from aragora.rlm.types import RLMContext

# Check for RLM availability
try:
    from aragora.rlm import get_rlm, RLMConfig, AbstractionLevel, RLMContext as _RLMContext, HAS_OFFICIAL_RLM
    HAS_RLM = True
except ImportError:
    HAS_RLM = False
    HAS_OFFICIAL_RLM = False
    get_rlm = None  # type: ignore[misc,assignment]
    RLMConfig = None  # type: ignore[misc,assignment]
    AbstractionLevel = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


class RLMProtocol(Protocol):
    """Protocol defining expected interface for RLM mixin."""

    config: "MoundConfig"
    workspace_id: str
    _initialized: bool

    def _ensure_initialized(self) -> None: ...
    async def query_semantic(
        self,
        text: str,
        limit: int = 10,
        min_confidence: float = 0.0,
        workspace_id: Optional[str] = None,
    ) -> List["KnowledgeItem"]: ...


class RLMOperationsMixin:
    """Mixin providing RLM operations for KnowledgeMound."""

    async def query_with_rlm(
        self: RLMProtocol,
        query: str,
        limit: int = 50,
        workspace_id: Optional[str] = None,
        agent_call: Optional[Any] = None,
    ) -> Optional["RLMContext"]:
        """
        Query knowledge and build hierarchical RLM context for navigation.

        Based on the "Recursive Language Models" paper (arXiv:2512.24601),
        this method builds a hierarchical representation of query results
        that enables efficient navigation from summaries to details.

        Args:
            query: Semantic query text
            limit: Maximum knowledge items to include
            workspace_id: Workspace to query
            agent_call: Optional callback for LLM-based compression

        Returns:
            RLMContext with hierarchical representation of knowledge,
            or None if RLM is not available.

        Example:
            ctx = await mound.query_with_rlm("contract requirements", limit=30)
            if ctx:
                # Get high-level overview
                abstract = ctx.get_at_level(AbstractionLevel.ABSTRACT)

                # Drill into specific node
                details = ctx.drill_down("type_fact")
        """
        if not HAS_RLM:
            logger.warning("RLM not available, use query_semantic instead")
            return None

        self._ensure_initialized()

        ws_id = workspace_id or self.workspace_id

        # Fetch relevant knowledge items
        items = await self.query_semantic(
            text=query,
            limit=limit,
            workspace_id=ws_id,
        )

        if not items:
            logger.debug("No knowledge items found for RLM context")
            return None

        # Build text content from knowledge items
        content_parts = []
        for item in items:
            source = getattr(item, 'source', None) or getattr(item, 'source_type', None)
            source_str = source.value if hasattr(source, 'value') else str(source) if source else 'unknown'
            confidence = getattr(item, 'confidence', 0.0)
            if hasattr(confidence, 'value'):
                confidence_val = 0.5  # Default if it's an enum string
            elif isinstance(confidence, (int, float)):
                confidence_val = confidence
            else:
                confidence_val = 0.5
            item_text = f"[{item.id}] ({source_str})\n"
            item_text += f"**Confidence**: {confidence_val:.0%}\n"
            item_text += f"{item.content}\n"
            content_parts.append(item_text)

        full_content = "\n---\n".join(content_parts)

        # Get AragoraRLM instance (routes to TRUE RLM when available)
        if get_rlm is None:
            logger.warning("RLM factory not available")
            return None

        try:
            config = RLMConfig() if RLMConfig else None
            rlm = get_rlm(config=config)
        except Exception as e:
            logger.warning(f"Failed to get RLM instance: {e}")
            return None

        # Query using AragoraRLM (routes to TRUE RLM if available)
        try:
            result = await rlm.compress_and_query(
                query=f"Summarize the key knowledge from these {len(items)} items",
                content=full_content,
                source_type="knowledge",
            )

            if result and result.answer:
                # Log which approach was used
                if result.used_true_rlm:
                    logger.info(
                        "[rlm] Built knowledge context using TRUE RLM from %d items "
                        "(model wrote code to examine content)",
                        len(items),
                    )
                elif result.used_compression_fallback:
                    logger.info(
                        "[rlm] Built knowledge context using compression from %d items",
                        len(items),
                    )

                # Return the RLM context from the result
                return result.context if hasattr(result, 'context') and result.context else None

            return None

        except Exception as e:
            logger.error(f"RLM query failed: {e}")
            return None

    def is_rlm_available(self: RLMProtocol) -> bool:
        """Check if RLM features are available."""
        return HAS_RLM
