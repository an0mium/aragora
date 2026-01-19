"""
RLM Context Adapter - programmatic access to context as external environment.

Based on Prime Intellect's RLM paper (arXiv:2512.24601): instead of stuffing
everything into the prompt, treat context as an external environment that
the LLM can query programmatically.

Key RLM principles:
1. **External Environment**: Data lives outside the prompt, accessed via code
2. **Query-based Access**: LLM requests specific data when needed
3. **Drill-down**: Start with summaries, drill into details on demand
4. **Compression as Fallback**: Only compress when programmatic access unavailable

Usage:
    from aragora.rlm.adapter import RLMContextAdapter
    from aragora.rlm.types import AbstractionLevel

    # Create adapter with external context store
    adapter = RLMContextAdapter()

    # Register content in external store (not in prompt)
    context_id = adapter.register_content("evidence_123", long_evidence_text)

    # Get summary for prompt (minimal context)
    summary = adapter.get_summary(context_id, max_chars=200)

    # LLM can request more detail via query
    detail = await adapter.query(context_id, "What does the evidence say about X?")

    # Drill down to specific section
    section = adapter.drill_down(context_id, section="conclusion")
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from .types import AbstractionLevel, RLMContext, RLMQuery, RLMResult

if TYPE_CHECKING:
    from .compressor import HierarchicalCompressor

logger = logging.getLogger(__name__)


@dataclass
class RegisteredContent:
    """Content registered in the external environment."""

    id: str
    """Unique identifier for this content."""

    full_content: str
    """The complete original content."""

    content_type: str
    """Type: evidence, dissent, debate, historical, etc."""

    summary: str = ""
    """Pre-computed summary for prompt inclusion."""

    sections: dict[str, str] = field(default_factory=dict)
    """Named sections for drill-down access."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (source, timestamp, etc.)."""


class RLMContextAdapter:
    """
    RLM-style adapter: context as external environment, not prompt stuffing.

    Based on Prime Intellect's RLM paper (arXiv:2512.24601), this adapter
    implements the external environment pattern where content is stored
    externally and accessed programmatically rather than stuffed into prompts.

    Key principles:
    1. **External Environment**: Full content stored in registry, not prompts
    2. **Query-based Access**: LLM requests specific data when needed
    3. **Drill-down**: Start with summaries, drill into details on demand

    **Priority Order** (highest to lowest):
    1. TRUE RLM: Use LLM to query/summarize content programmatically
    2. COMPRESSION: Use hierarchical compressor if available
    3. TRUNCATION: Heuristic extraction as LAST RESORT only

    Async methods (generate_summary_async, format_for_prompt_async) use
    the full priority chain. Sync methods fall back to compression/truncation.
    """

    def __init__(
        self,
        compressor: Optional["HierarchicalCompressor"] = None,
        agent_call: Optional[Callable[[str, str], Any]] = None,
    ):
        """
        Initialize the adapter.

        Args:
            compressor: Optional compressor for fallback compression
            agent_call: Optional LLM call function for queries
        """
        self._compressor = compressor
        self._agent_call = agent_call
        self._registry: dict[str, RegisteredContent] = {}

    # =========================================================================
    # External Environment: Register and Access Content
    # =========================================================================

    def register_content(
        self,
        content_id: str,
        content: str,
        content_type: str = "text",
        summary: Optional[str] = None,
        sections: Optional[dict[str, str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Register content in the external environment.

        The content is stored externally rather than stuffed into prompts.
        LLMs can query it programmatically when needed.

        Args:
            content_id: Unique identifier (or auto-generate)
            content: Full content to register
            content_type: Type for context-aware handling
            summary: Pre-computed summary (auto-generates if None)
            sections: Named sections for drill-down
            metadata: Additional metadata

        Returns:
            The content_id for later access
        """
        if not content_id:
            content_id = self._generate_id(content)

        # Auto-generate summary if not provided
        if summary is None:
            summary = self._extract_summary(content, content_type)

        # Auto-extract sections if not provided
        if sections is None:
            sections = self._extract_sections(content, content_type)

        self._registry[content_id] = RegisteredContent(
            id=content_id,
            full_content=content,
            content_type=content_type,
            summary=summary,
            sections=sections,
            metadata=metadata or {},
        )

        logger.debug(
            f"adapter_register id={content_id} type={content_type} "
            f"size={len(content)} sections={list(sections.keys())}"
        )

        return content_id

    def get_summary(
        self,
        content_id: str,
        max_chars: Optional[int] = None,
    ) -> str:
        """
        Get a minimal summary for prompt inclusion.

        This is the RLM pattern: include only a summary in the prompt,
        not the full content. The LLM can query for details if needed.

        Args:
            content_id: ID of registered content
            max_chars: Optional character limit

        Returns:
            Summary text (or empty string if not found)
        """
        registered = self._registry.get(content_id)
        if not registered:
            logger.warning(f"adapter_get_summary not_found id={content_id}")
            return ""

        summary = registered.summary
        if max_chars and len(summary) > max_chars:
            summary = self._smart_truncate(summary, max_chars)

        return summary

    def get_full_content(self, content_id: str) -> str:
        """
        Get the full registered content (for queries, not prompts).

        Args:
            content_id: ID of registered content

        Returns:
            Full content text
        """
        registered = self._registry.get(content_id)
        return registered.full_content if registered else ""

    def drill_down(
        self,
        content_id: str,
        section: Optional[str] = None,
        query: Optional[str] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        """
        Drill down to specific content section.

        RLM pattern: start with summary, drill into details on demand.

        Args:
            content_id: ID of registered content
            section: Named section to retrieve (e.g., "conclusion", "evidence")
            query: Search query to find relevant portion
            max_chars: Optional character limit

        Returns:
            Section content or query-matched portion
        """
        registered = self._registry.get(content_id)
        if not registered:
            return ""

        # Get specific section
        if section and section in registered.sections:
            content = registered.sections[section]
            if max_chars and len(content) > max_chars:
                content = self._smart_truncate(content, max_chars)
            return content

        # Search for query match
        if query:
            return self._search_content(registered.full_content, query, max_chars)

        # Return full content (truncated if needed)
        content = registered.full_content
        if max_chars and len(content) > max_chars:
            content = self._smart_truncate(content, max_chars)
        return content

    async def query(
        self,
        content_id: str,
        question: str,
        max_response_chars: int = 500,
    ) -> RLMResult:
        """
        Query registered content with a question.

        Uses LLM to answer questions about the content without
        including the full content in the main conversation.

        Args:
            content_id: ID of registered content
            question: Question to answer about the content
            max_response_chars: Maximum response length

        Returns:
            RLMResult with answer and provenance
        """
        registered = self._registry.get(content_id)
        if not registered:
            return RLMResult(
                answer="Content not found",
                ready=True,
                confidence=0.0,
            )

        if not self._agent_call:
            # No LLM available - return relevant snippet
            snippet = self._search_content(
                registered.full_content, question, max_response_chars
            )
            return RLMResult(
                answer=snippet,
                ready=True,
                confidence=0.5,
                nodes_examined=[content_id],
            )

        # Build query prompt
        prompt = f"""Based on the following content, answer the question concisely.

Content ({registered.content_type}):
{registered.full_content[:4000]}

Question: {question}

Answer (be specific and cite relevant parts):"""

        try:
            response = await self._agent_call(prompt, "sub_model")
            return RLMResult(
                answer=str(response)[:max_response_chars],
                ready=True,
                confidence=0.8,
                nodes_examined=[content_id],
            )
        except Exception as e:
            logger.warning(f"adapter_query_failed error={e}")
            return RLMResult(
                answer=self._search_content(
                    registered.full_content, question, max_response_chars
                ),
                ready=True,
                confidence=0.3,
            )

    def list_registered(self) -> list[str]:
        """List all registered content IDs."""
        return list(self._registry.keys())

    def unregister(self, content_id: str) -> bool:
        """Remove content from the registry."""
        if content_id in self._registry:
            del self._registry[content_id]
            return True
        return False

    # =========================================================================
    # Fallback: Truncation (when RLM access not available)
    # =========================================================================

    def smart_truncate(
        self,
        content: str,
        max_chars: int,
        content_type: str = "text",
    ) -> str:
        """
        Smart truncation as fallback when RLM access not available.

        Unlike naive [:max_chars], this method:
        - Respects sentence boundaries when possible
        - Adds ellipsis indicator when truncated
        - Preserves key structural elements

        IMPORTANT: Use register_content + get_summary instead when possible.
        This is a fallback for cases where content must fit in a prompt.

        Args:
            content: Content to truncate
            max_chars: Maximum characters
            content_type: Type for context-aware truncation

        Returns:
            Truncated content with ellipsis if needed
        """
        return self._smart_truncate(content, max_chars, content_type)

    def format_for_prompt(
        self,
        content: str,
        max_chars: int,
        content_type: str = "text",
        include_hint: bool = True,
    ) -> str:
        """
        Format content for prompt inclusion with RLM hint (sync version).

        Uses heuristic summary. For TRUE RLM with LLM summarization,
        use format_for_prompt_async() instead.

        Args:
            content: Full content
            max_chars: Max chars for prompt
            content_type: Type of content
            include_hint: Include drill-down hint

        Returns:
            Formatted summary for prompt
        """
        if not content:
            return ""

        # Register content for later access
        content_id = self.register_content(
            content_id="",  # Auto-generate
            content=content,
            content_type=content_type,
        )

        summary = self.get_summary(content_id, max_chars - 50 if include_hint else max_chars)

        if include_hint and len(content) > max_chars:
            summary += f"\n[Full {content_type} available: {content_id}]"

        return summary

    async def format_for_prompt_async(
        self,
        content: str,
        max_chars: int,
        content_type: str = "text",
        include_hint: bool = True,
    ) -> str:
        """
        Format content for prompt inclusion - TRUE RLM with LLM summarization.

        This is the preferred method: registers content externally and uses
        LLM to generate a meaningful summary. Falls back to heuristics only
        when no LLM is available.

        Args:
            content: Full content
            max_chars: Max chars for prompt
            content_type: Type of content
            include_hint: Include drill-down hint

        Returns:
            LLM-generated summary with drill-down hint
        """
        if not content:
            return ""

        # Register content for later access (external environment)
        content_id = self.register_content(
            content_id="",  # Auto-generate
            content=content,
            content_type=content_type,
        )

        # TRUE RLM: Use LLM to generate summary
        summary = await self.generate_summary_async(
            content_id, max_chars - 50 if include_hint else max_chars
        )

        if include_hint and len(content) > max_chars:
            summary += f"\n[Full {content_type} available: {content_id}]"

        return summary

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for content."""
        return hashlib.sha256(f"{content[:100]}:{len(content)}".encode()).hexdigest()[:12]

    async def generate_summary_async(
        self,
        content_id: str,
        max_chars: Optional[int] = None,
    ) -> str:
        """
        Generate summary using TRUE RLM pattern.

        Priority order (highest to lowest):
        1. TRUE RLM: Use LLM to query external content
        2. COMPRESSION: Use hierarchical compressor if available
        3. TRUNCATION: Heuristic extraction as last resort

        Args:
            content_id: ID of registered content
            max_chars: Optional character limit

        Returns:
            Summary (LLM > compression > truncation fallback)
        """
        registered = self._registry.get(content_id)
        if not registered:
            return ""

        # Short content doesn't need summarization
        if len(registered.full_content) <= 200:
            return registered.summary or registered.full_content

        # PRIORITY 1: TRUE RLM - Use LLM to generate summary
        if self._agent_call:
            try:
                prompt = f"""Summarize this {registered.content_type} in 2-3 concise sentences.
Focus on the key points and conclusions.

Content:
{registered.full_content[:4000]}

Summary:"""
                response = await self._agent_call(prompt, "sub_model")
                summary = str(response).strip()

                # Cache the LLM-generated summary
                registered.summary = summary

                if max_chars and len(summary) > max_chars:
                    summary = self._smart_truncate(summary, max_chars)
                return summary
            except Exception as e:
                logger.warning(f"adapter_llm_summary_failed error={e}, trying compression")

        # PRIORITY 2: COMPRESSION - Use compressor if available
        if self._compressor:
            try:
                result = await self._compressor.compress(
                    registered.full_content,
                    source_type=registered.content_type,
                )
                summary = result.context.get_at_level(
                    result.context.abstraction_levels.get("summary", "SUMMARY")
                ) if hasattr(result, 'context') else str(result)
                registered.summary = summary

                if max_chars and len(summary) > max_chars:
                    summary = self._smart_truncate(summary, max_chars)
                return summary
            except Exception as e:
                logger.warning(f"adapter_compress_failed error={e}, using truncation")

        # PRIORITY 3: TRUNCATION - Heuristic extraction as last resort
        summary = registered.summary or self._heuristic_summary(
            registered.full_content, registered.content_type
        )
        if max_chars and len(summary) > max_chars:
            summary = self._smart_truncate(summary, max_chars)
        return summary

    def _heuristic_summary(self, content: str, content_type: str) -> str:
        """
        LAST RESORT: Heuristic summary extraction via truncation.

        Priority order:
        1. TRUE RLM (LLM queries) - PREFERRED
        2. COMPRESSION (hierarchical) - FALLBACK
        3. TRUNCATION (this method) - LAST RESORT ONLY

        This method should only be used when both LLM and compression
        are unavailable. Use generate_summary_async() for true RLM behavior.
        """
        if not content:
            return ""

        # For short content, return as-is
        if len(content) <= 200:
            return content

        # Try to extract first paragraph or sentence
        paragraphs = content.split("\n\n")
        if paragraphs:
            first_para = paragraphs[0].strip()
            if len(first_para) <= 300:
                return first_para

        # Extract first 2-3 sentences
        sentences = re.split(r"(?<=[.!?])\s+", content[:500])
        if sentences:
            summary_sentences = []
            total_len = 0
            for s in sentences[:3]:
                if total_len + len(s) > 250:
                    break
                summary_sentences.append(s)
                total_len += len(s)
            if summary_sentences:
                return " ".join(summary_sentences)

        # Fallback to truncation
        return self._smart_truncate(content, 200)

    def _extract_summary(self, content: str, content_type: str) -> str:
        """Alias for _heuristic_summary for backwards compatibility."""
        return self._heuristic_summary(content, content_type)

    def _extract_sections(
        self, content: str, content_type: str
    ) -> dict[str, str]:
        """Extract named sections from content."""
        sections: dict[str, str] = {}

        # Look for markdown headers
        header_pattern = r"^##?\s+(.+)$"
        lines = content.split("\n")
        current_section = "intro"
        current_content: list[str] = []

        for line in lines:
            match = re.match(header_pattern, line.strip())
            if match:
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = match.group(1).lower().replace(" ", "_")
                current_content = []
            else:
                current_content.append(line)

        # Save final section
        if current_content:
            sections[current_section] = "\n".join(current_content).strip()

        # For specific content types, extract key sections
        if content_type == "evidence":
            # Look for conclusion/finding
            for keyword in ["conclusion", "finding", "result"]:
                for line in lines:
                    if keyword in line.lower() and ":" in line:
                        sections["conclusion"] = line.split(":", 1)[-1].strip()
                        break

        elif content_type == "dissent":
            # Look for core disagreement
            for keyword in ["disagree", "concern", "issue", "however"]:
                for i, line in enumerate(lines):
                    if keyword in line.lower():
                        # Extract surrounding context
                        start = max(0, i - 1)
                        end = min(len(lines), i + 3)
                        sections["core"] = "\n".join(lines[start:end]).strip()
                        break

        return sections

    def _search_content(
        self, content: str, query: str, max_chars: Optional[int] = None
    ) -> str:
        """Search content for query-relevant portion."""
        if not query or not content:
            return content[:max_chars] if max_chars else content

        # Simple keyword search
        query_words = set(query.lower().split())
        lines = content.split("\n")

        # Score each line by keyword matches
        scored_lines: list[tuple[int, int, str]] = []
        for i, line in enumerate(lines):
            line_lower = line.lower()
            score = sum(1 for word in query_words if word in line_lower)
            if score > 0:
                scored_lines.append((score, i, line))

        if not scored_lines:
            # No matches - return beginning
            return content[:max_chars] if max_chars else content

        # Sort by score and get best matches with context
        scored_lines.sort(reverse=True)
        best_line_idx = scored_lines[0][1]

        # Include surrounding context
        start = max(0, best_line_idx - 2)
        end = min(len(lines), best_line_idx + 3)
        result = "\n".join(lines[start:end])

        if max_chars and len(result) > max_chars:
            result = self._smart_truncate(result, max_chars)

        return result

    def _smart_truncate(
        self,
        content: str,
        max_chars: int,
        content_type: str = "text",
    ) -> str:
        """Truncate preserving sentence boundaries."""
        if not content or len(content) <= max_chars:
            return content

        truncated = content[:max_chars]

        # Try to break at sentence boundary
        for i in range(len(truncated) - 1, int(max_chars * 0.5), -1):
            if truncated[i] in ".!?" and (
                i + 1 >= len(truncated) or truncated[i + 1] in " \n"
            ):
                return content[: i + 1]

        # Break at word boundary
        if " " in truncated:
            last_space = truncated.rfind(" ")
            if last_space > max_chars * 0.7:
                return content[:last_space] + "..."

        return truncated + "..."


# Global adapter instance
_global_adapter: Optional[RLMContextAdapter] = None


def get_adapter() -> RLMContextAdapter:
    """Get the global RLMContextAdapter instance."""
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = RLMContextAdapter()
    return _global_adapter


__all__ = [
    "RLMContextAdapter",
    "RegisteredContent",
    "get_adapter",
]
