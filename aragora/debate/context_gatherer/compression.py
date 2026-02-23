"""
RLM compression mixin for ContextGatherer.

Contains methods for compressing large content using RLM (Recursive Language Models)
and TRUE RLM (REPL-based) when available.
"""

import asyncio
import logging
import sys
from typing import Any

from .constants import (
    HAS_RLM,
    HAS_OFFICIAL_RLM,
)

logger = logging.getLogger(__name__)


def _has_official_rlm() -> bool:
    """
    Resolve the official-RLM feature flag with package-level override support.

    Tests patch ``aragora.debate.context_gatherer.HAS_OFFICIAL_RLM``; consult
    that export first so patching the package behaves consistently.
    """
    package_mod = sys.modules.get("aragora.debate.context_gatherer")
    if package_mod is not None and hasattr(package_mod, "HAS_OFFICIAL_RLM"):
        return bool(getattr(package_mod, "HAS_OFFICIAL_RLM"))
    return bool(HAS_OFFICIAL_RLM)


class CompressionMixin:
    """Mixin providing RLM compression methods."""

    # Type hints for attributes defined in main class
    _enable_rlm: bool
    _rlm_compressor: Any
    _aragora_rlm: Any
    _rlm_threshold: int
    _enable_knowledge_grounding: bool
    _knowledge_mound: Any
    _knowledge_workspace_id: str

    def _get_task_hash(self, task: str) -> str:
        """Generate a cache key from task to prevent cache leaks between debates."""
        raise NotImplementedError("Must be implemented by main class")

    async def gather_knowledge_mound_context(self, task: str) -> str | None:
        """Query Knowledge Mound for relevant facts and evidence."""
        raise NotImplementedError("Must be implemented by main class")

    async def _compress_with_rlm(
        self,
        content: str,
        source_type: str = "documentation",
        max_chars: int = 3000,
    ) -> str:
        """
        Compress large content using RLM.

        Prioritizes TRUE RLM (REPL-based) via AragoraRLM when available:
        - Model writes code to examine/summarize content
        - Model has agency in deciding how to compress

        Falls back to HierarchicalCompressor (compression-only) if:
        - Official RLM not installed
        - AragoraRLM fails

        Falls back to truncation if all else fails.

        Args:
            content: The content to compress
            source_type: Type of content (for compression hints)
            max_chars: Target character limit

        Returns:
            Compressed or truncated content
        """
        # If content is under threshold, return as-is
        if len(content) <= self._rlm_threshold:
            return content[:max_chars] if len(content) > max_chars else content

        # If RLM is not enabled, use simple truncation
        if not self._enable_rlm:
            return (
                content[: max_chars - 30] + "... [truncated]"
                if len(content) > max_chars
                else content
            )

        # PRIMARY: Try AragoraRLM (routes to TRUE RLM if available)
        if self._aragora_rlm:
            try:
                logger.debug(
                    "[rlm] Using AragoraRLM for compression (routes to TRUE RLM if available)"
                )
                result = await asyncio.wait_for(
                    self._aragora_rlm.compress_and_query(
                        query=f"Summarize this {source_type} in under {max_chars} characters",
                        content=content,
                        source_type=source_type,
                    ),
                    timeout=15.0,
                )

                if result.answer and len(result.answer) < len(content):
                    approach = "TRUE RLM" if result.used_true_rlm else "compression fallback"
                    logger.debug(
                        "[rlm] Compressed %s -> %s chars (%s%%) via %s",
                        len(content),
                        len(result.answer),
                        int(len(result.answer) / len(content) * 100),
                        approach,
                    )
                    return (
                        result.answer[:max_chars]
                        if len(result.answer) > max_chars
                        else result.answer
                    )

            except asyncio.TimeoutError:
                logger.debug("[rlm] AragoraRLM compression timed out")
            except (ValueError, RuntimeError) as e:
                logger.debug("[rlm] AragoraRLM compression failed: %s", e)
            except (
                TypeError,
                AttributeError,
                KeyError,
                OSError,
                ConnectionError,
                ImportError,
            ) as e:
                logger.warning("[rlm] Unexpected error in AragoraRLM compression: %s", e)

        # FALLBACK: Try direct HierarchicalCompressor (compression-only)
        if self._rlm_compressor:
            try:
                logger.debug(
                    "[rlm] Falling back to HierarchicalCompressor (compression-only, no TRUE RLM)"
                )
                compression_result = await asyncio.wait_for(
                    self._rlm_compressor.compress(
                        content=content,
                        source_type=source_type,
                        max_levels=2,  # ABSTRACT and SUMMARY for faster compression
                    ),
                    timeout=10.0,
                )

                # Get summary level (or abstract if summary is too long)
                try:
                    from aragora.rlm.types import AbstractionLevel

                    summary = compression_result.context.get_at_level(AbstractionLevel.SUMMARY)
                    if summary and len(summary) > max_chars:
                        summary = compression_result.context.get_at_level(AbstractionLevel.ABSTRACT)
                except (ImportError, AttributeError):
                    summary = None

                if summary and len(summary) < len(content):
                    logger.debug(
                        "[rlm] Compressed %s -> %s chars (%s%%) via HierarchicalCompressor",
                        len(content),
                        len(summary),
                        int(len(summary) / len(content) * 100),
                    )
                    return summary[:max_chars] if len(summary) > max_chars else summary

            except asyncio.TimeoutError:
                logger.debug("[rlm] HierarchicalCompressor timed out")
            except (ValueError, RuntimeError) as e:
                logger.debug("[rlm] HierarchicalCompressor failed: %s", e)
            except (
                TypeError,
                AttributeError,
                KeyError,
                OSError,
                ConnectionError,
                ImportError,
            ) as e:
                logger.warning("[rlm] Unexpected error in HierarchicalCompressor: %s", e)

        # FINAL FALLBACK: Simple truncation
        logger.debug("[rlm] All RLM approaches failed, using simple truncation")
        return (
            content[: max_chars - 30] + "... [truncated]" if len(content) > max_chars else content
        )

    async def _query_with_true_rlm(
        self,
        query: str,
        content: str,
        source_type: str = "documentation",
    ) -> str | None:
        """
        Query content using TRUE RLM (REPL-based) when available.

        TRUE RLM allows the model to write code to examine context stored
        as Python variables in a REPL environment, rather than having the
        context stuffed into prompts.

        This is the PREFERRED method when the official `rlm` package is installed:
        - Model has agency in deciding how to query content
        - No information loss from truncation or compression
        - Model writes code like: `search_debate(context, r"consensus")`

        Falls back to compression-based query if TRUE RLM not available.

        Args:
            query: The question to answer about the content
            content: The content to query
            source_type: Type of content (for context hints)

        Returns:
            Answer from TRUE RLM, or None if not available
        """
        if not self._enable_rlm or not self._aragora_rlm:
            return None

        try:
            # Check if TRUE RLM is available (not just compression fallback)
            if HAS_RLM and _has_official_rlm():
                logger.debug(
                    "[rlm] Using TRUE RLM for query: '%s...' on %s chars of %s",
                    query[:50],
                    len(content),
                    source_type,
                )

                result = await asyncio.wait_for(
                    self._aragora_rlm.query(
                        query=query,
                        context=content,
                        strategy="auto",  # Let RLM decide: grep, partition, peek, etc.
                    ),
                    timeout=20.0,
                )

                if result.used_true_rlm and result.answer:
                    logger.debug(
                        "[rlm] TRUE RLM query successful: %s chars, confidence=%s",
                        len(result.answer),
                        result.confidence,
                    )
                    return result.answer

            # TRUE RLM not available - fall back to compress_and_query
            logger.debug("[rlm] TRUE RLM not available for query, using compress_and_query")
            result = await asyncio.wait_for(
                self._aragora_rlm.compress_and_query(
                    query=query,
                    content=content,
                    source_type=source_type,
                ),
                timeout=15.0,
            )

            if result.answer:
                approach = "TRUE RLM" if result.used_true_rlm else "compression"
                logger.debug("[rlm] Query via %s: %s chars", approach, len(result.answer))
                return result.answer

        except asyncio.TimeoutError:
            logger.debug("[rlm] TRUE RLM query timed out for: '%s...'", query[:30])
        except (ValueError, RuntimeError) as e:
            logger.debug("[rlm] TRUE RLM query failed: %s", e)
        except (TypeError, AttributeError, KeyError, OSError, ConnectionError) as e:
            logger.warning("[rlm] Unexpected error in TRUE RLM query: %s", e)

        return None

    async def query_knowledge_with_true_rlm(
        self,
        task: str,
        max_items: int = 10,
    ) -> str | None:
        """
        Query Knowledge Mound using TRUE RLM for better answer quality.

        When TRUE RLM is available, creates a REPL environment where the
        model can write code to navigate and query knowledge items.

        Args:
            task: The debate task/query
            max_items: Maximum knowledge items to include in context

        Returns:
            Synthesized answer from knowledge, or None if unavailable
        """
        if not self._enable_knowledge_grounding or not self._knowledge_mound:
            return None

        if not (HAS_RLM and _has_official_rlm()):
            # TRUE RLM not available - use standard query
            return await self.gather_knowledge_mound_context(task)

        try:
            from aragora.rlm import get_repl_adapter

            adapter = get_repl_adapter()

            # Create REPL environment for knowledge queries
            env = adapter.create_repl_for_knowledge(
                mound=self._knowledge_mound,
                workspace_id=self._knowledge_workspace_id,
                content_id=f"km_{self._get_task_hash(task)}",
            )

            if not env:
                # TRUE RLM REPL failed - fall back to standard
                return await self.gather_knowledge_mound_context(task)

            # Get REPL prompt for agent
            repl_prompt = adapter.get_repl_prompt(
                content_id=f"km_{self._get_task_hash(task)}",
                content_type="knowledge",
            )

            logger.info(
                "[rlm] Created TRUE RLM REPL environment for knowledge query: '%s...'",
                task[:50],
            )

            # Return the REPL prompt - the agent will use it to write code
            # that queries the knowledge programmatically
            return (
                "## KNOWLEDGE MOUND CONTEXT (TRUE RLM)\n"
                f"A REPL environment is available for knowledge queries.\n\n"
                f"{repl_prompt}\n"
            )

        except ImportError:
            logger.debug("[rlm] REPL adapter not available for knowledge queries")
        except (TypeError, AttributeError, KeyError, RuntimeError, OSError, ConnectionError) as e:
            logger.warning("[rlm] Failed to create knowledge REPL: %s", e)

        # Fall back to standard knowledge query
        return await self.gather_knowledge_mound_context(task)
