"""
RLM Operations Mixin for Knowledge Mound.

Provides RLM (Recursive Language Models) integration:
- query_with_rlm: Build hierarchical context from knowledge
- query_with_true_rlm: Use TRUE RLM REPL for programmatic queries (PREFERRED)
- is_rlm_available: Check RLM availability

TRUE RLM Prioritization (Phase 12):
When the official `rlm` package is installed, TRUE RLM uses a REPL-based
approach where the model writes code to examine context. This is preferred
over compression-based methods.

Install TRUE RLM: pip install aragora[rlm]

NOTE: This is a mixin class designed to be composed with KnowledgeMound.
Attribute accesses like self._ensure_initialized, self.workspace_id, self.query_semantic, etc.
are provided by the composed class. A Protocol class (RLMProtocol) is used with cast()
to provide type-safe access to these attributes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import KnowledgeItem, MoundConfig
    from aragora.rlm.types import RLMContext

# Check for RLM availability
try:
    from aragora.rlm import (
        get_rlm,
        RLMConfig,
        RLMMode,
        AbstractionLevel,
        HAS_OFFICIAL_RLM,
    )

    HAS_RLM = True
except ImportError:
    HAS_RLM = False
    HAS_OFFICIAL_RLM = False
    get_rlm: Any = None  # type: ignore[no-redef]
    RLMConfig: Any = None  # type: ignore[no-redef]
    RLMMode: Any = None  # type: ignore[no-redef]
    AbstractionLevel: Any = None  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class RLMProtocol(Protocol):
    """Protocol defining expected interface for RLM mixin."""

    config: MoundConfig
    workspace_id: str
    _initialized: bool

    def _ensure_initialized(self) -> None: ...
    async def query_semantic(
        self,
        text: str,
        limit: int = 10,
        min_confidence: float = 0.0,
        workspace_id: str | None = None,
    ) -> list[KnowledgeItem]: ...


class RLMOperationsMixin:
    """Mixin providing RLM operations for KnowledgeMound."""

    async def query_with_rlm(
        self,
        query: str,
        limit: int = 50,
        workspace_id: str | None = None,
        agent_call: Any | None = None,
    ) -> RLMContext | None:
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

        # Cast self to Protocol to access methods provided by composed KnowledgeMound class
        host = cast(RLMProtocol, self)
        host._ensure_initialized()

        ws_id = workspace_id or host.workspace_id

        # Fetch relevant knowledge items via QueryOperationsMixin
        items = await host.query_semantic(
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
            source = getattr(item, "source", None) or getattr(item, "source_type", None)
            source_str = (
                source.value if hasattr(source, "value") else str(source) if source else "unknown"
            )
            confidence = getattr(item, "confidence", 0.0)
            if hasattr(confidence, "value"):
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
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("Failed to get RLM instance: %s", e)
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
                return result.context if hasattr(result, "context") and result.context else None

            return None

        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.error("RLM query failed: %s", e)
            return None

    def is_rlm_available(self) -> bool:
        """Check if RLM features are available."""
        return HAS_RLM

    def is_true_rlm_available(self) -> bool:
        """Check if TRUE RLM (REPL-based) features are available."""
        return HAS_RLM and HAS_OFFICIAL_RLM

    async def query_with_true_rlm(
        self,
        query: str,
        limit: int = 50,
        workspace_id: str | None = None,
        prefer_true_rlm: bool = True,
    ) -> str | None:
        """
        Query knowledge using TRUE RLM (REPL-based) when available.

        TRUE RLM (based on arXiv:2512.24601) uses a REPL environment where
        the model writes code to examine context programmatically:
        - Context stored as Python variables (not stuffed into prompts)
        - Model writes code like: `facts = get_facts(km, "topic", min_confidence=0.8)`
        - No information loss from truncation or compression

        This is the PREFERRED method when `pip install aragora[rlm]` is installed.

        Falls back to compression-based query if TRUE RLM not available.

        Args:
            query: Semantic query text
            limit: Maximum knowledge items to include
            workspace_id: Workspace to query
            prefer_true_rlm: If True, require TRUE RLM (warn if not available)

        Returns:
            Answer synthesized from knowledge items, or None if unavailable
        """
        if not HAS_RLM:
            logger.warning(
                "[rlm] RLM not available for knowledge query. "
                "Install with: pip install aragora[rlm]"
            )
            return None

        # Cast self to Protocol to access methods provided by composed KnowledgeMound class
        host = cast(RLMProtocol, self)
        host._ensure_initialized()
        ws_id = workspace_id or host.workspace_id

        # Check if TRUE RLM is available
        if prefer_true_rlm and not HAS_OFFICIAL_RLM:
            logger.warning(
                "[rlm] TRUE RLM preferred but not available. "
                "Will use compression fallback. "
                "Install with: pip install aragora[rlm] for better results."
            )

        # Fetch relevant knowledge items via QueryOperationsMixin
        items = await host.query_semantic(
            text=query,
            limit=limit,
            workspace_id=ws_id,
        )

        if not items:
            logger.debug("[rlm] No knowledge items found for TRUE RLM query")
            return None

        # Build text content from knowledge items
        content_parts = []
        for item in items:
            source = getattr(item, "source", None) or getattr(item, "source_type", None)
            source_str = (
                source.value if hasattr(source, "value") else str(source) if source else "unknown"
            )
            confidence = getattr(item, "confidence", 0.0)
            if hasattr(confidence, "value"):
                confidence_val = 0.5
            elif isinstance(confidence, (int, float)):
                confidence_val = confidence
            else:
                confidence_val = 0.5

            item_text = f"[{item.id}] ({source_str}, {confidence_val:.0%})\n{item.content}"
            content_parts.append(item_text)

        full_content = "\n---\n".join(content_parts)

        # Get AragoraRLM with TRUE RLM priority
        if get_rlm is None:
            logger.warning("[rlm] RLM factory not available")
            return None

        try:
            # Configure for TRUE RLM priority
            config = (
                RLMConfig(
                    prefer_true_rlm=True,
                    warn_on_compression_fallback=prefer_true_rlm,
                )
                if RLMConfig
                else None
            )

            # Get RLM with AUTO mode (prefers TRUE RLM)
            mode = RLMMode.AUTO if RLMMode else None
            rlm = get_rlm(config=config, mode=mode)
        except RuntimeError as e:
            # TRUE RLM required but not available
            logger.error("[rlm] TRUE RLM initialization failed: %s", e)
            return None
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("[rlm] Failed to get RLM instance: %s", e)
            return None

        # Try TRUE RLM query first (if available)
        if HAS_OFFICIAL_RLM and hasattr(rlm, "query"):
            try:
                result = await rlm.query(
                    query=query,
                    context=cast(Any, full_content),
                    strategy="auto",
                )

                if result.used_true_rlm and result.answer:
                    logger.info(
                        "[rlm] TRUE RLM query successful on %d knowledge items "
                        "(REPL-based, model wrote code to examine content)",
                        len(items),
                    )
                    return result.answer
            except (RuntimeError, ValueError, OSError) as e:
                logger.debug("[rlm] TRUE RLM query failed, trying compress_and_query: %s", e)

        # Fall back to compress_and_query
        try:
            result = await rlm.compress_and_query(
                query=query,
                content=full_content,
                source_type="knowledge",
            )

            if result and result.answer:
                approach = "TRUE RLM" if result.used_true_rlm else "compression fallback"
                logger.info(
                    "[rlm] Knowledge query via %s on %d items",
                    approach,
                    len(items),
                )
                return result.answer

            return None

        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.error("[rlm] Knowledge query failed: %s", e)
            return None

    async def create_knowledge_repl(
        self,
        workspace_id: str | None = None,
        content_id: str | None = None,
    ) -> Any | None:
        """
        Create a TRUE RLM REPL environment for interactive knowledge queries.

        Returns a REPL environment where an agent can write code to query
        knowledge programmatically:

            >>> facts = get_facts(km, "rate limiting", min_confidence=0.8)
            >>> related = get_related(km, facts[0].id, depth=2)
            >>> FINAL(f"Found {len(facts)} facts with {len(related)} related items")

        Args:
            workspace_id: Workspace to load knowledge from
            content_id: Optional ID for the REPL environment

        Returns:
            REPL environment, or None if TRUE RLM not available
        """
        if not HAS_RLM or not HAS_OFFICIAL_RLM:
            logger.warning(
                "[rlm] TRUE RLM REPL not available. Install with: pip install aragora[rlm]"
            )
            return None

        # Cast self to Protocol to access methods provided by composed KnowledgeMound class
        host = cast(RLMProtocol, self)
        host._ensure_initialized()
        ws_id = workspace_id or host.workspace_id

        try:
            from aragora.rlm import get_repl_adapter

            adapter = get_repl_adapter()

            # Create REPL environment with knowledge context
            env = adapter.create_repl_for_knowledge(
                mound=self,
                workspace_id=ws_id,
                content_id=content_id,
            )

            if env:
                logger.info(
                    "[rlm] Created TRUE RLM REPL environment for knowledge (workspace=%s)",
                    ws_id,
                )

            return env

        except ImportError:
            logger.debug("[rlm] REPL adapter not available")
            return None
        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.warning("[rlm] Failed to create knowledge REPL: %s", e)
            return None
