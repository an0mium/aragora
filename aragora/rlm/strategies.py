"""
Decomposition strategies for RLM processing.

These strategies emerge from the RLM paper as common patterns
for navigating long context:
- Peek: Inspect initial sections
- Grep: Use regex/keyword search
- Partition+Map: Chunk and process in parallel
- Summarize: Recursive summarization
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

from .types import (
    AbstractionLevel,
    DecompositionStrategy,
    RLMConfig,
    RLMContext,
    RLMQuery,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    """Result from a strategy execution."""

    answer: str
    confidence: float
    nodes_used: list[str]
    tokens_examined: int
    sub_calls: int


class BaseStrategy(ABC):
    """Base class for decomposition strategies."""

    def __init__(
        self,
        config: RLMConfig,
        agent_call: Optional[Callable[[str, str, str], str]] = None,
    ):
        self.config = config
        self.agent_call = agent_call

    @abstractmethod
    async def execute(
        self,
        query: RLMQuery,
        context: RLMContext,
    ) -> StrategyResult:
        """Execute the strategy to answer the query."""
        pass

    @property
    @abstractmethod
    def strategy_type(self) -> DecompositionStrategy:
        """Return the strategy type."""
        pass


class PeekStrategy(BaseStrategy):
    """
    Peek strategy: Examine initial sections to understand structure.

    Best for:
    - Understanding document structure
    - Finding table of contents / outlines
    - Quick orientation queries
    """

    @property
    def strategy_type(self) -> DecompositionStrategy:
        return DecompositionStrategy.PEEK

    async def execute(
        self,
        query: RLMQuery,
        context: RLMContext,
    ) -> StrategyResult:
        """Peek at the start of each abstraction level."""
        examined_tokens = 0
        nodes_used = []
        peek_results = []

        # Peek at each level starting from abstract
        for level in [
            AbstractionLevel.ABSTRACT,
            AbstractionLevel.SUMMARY,
            AbstractionLevel.DETAILED,
        ]:
            if level not in context.levels:
                continue

            nodes = context.levels[level]
            if nodes:
                # Take first node at each level
                node = nodes[0]
                preview = node.content[:500]
                peek_results.append(f"[{level.name}] {preview}")
                nodes_used.append(node.id)
                examined_tokens += min(node.token_count, 125)  # ~500 chars

        # Build answer
        if not peek_results:
            # Fall back to original content peek
            preview = context.original_content[:2000]
            answer = f"Document preview:\n{preview}"
            examined_tokens = 500
        else:
            answer = "\n\n".join(peek_results)

        return StrategyResult(
            answer=answer,
            confidence=0.6,  # Peek gives structural understanding, not answers
            nodes_used=nodes_used,
            tokens_examined=examined_tokens,
            sub_calls=0,
        )


class GrepStrategy(BaseStrategy):
    """
    Grep strategy: Use regex/keyword search to find relevant sections.

    Best for:
    - Specific fact lookup
    - Finding mentions of entities
    - Locating specific sections
    """

    @property
    def strategy_type(self) -> DecompositionStrategy:
        return DecompositionStrategy.GREP

    async def execute(
        self,
        query: RLMQuery,
        context: RLMContext,
    ) -> StrategyResult:
        """Search for query terms across context."""
        # Extract search terms from query
        search_terms = self._extract_search_terms(query.query)

        matches = []
        nodes_used = []
        examined_tokens = 0

        # Search from summary level down to detailed
        for level in [AbstractionLevel.SUMMARY, AbstractionLevel.DETAILED, AbstractionLevel.FULL]:
            if level not in context.levels:
                continue

            for node in context.levels[level]:
                for term in search_terms:
                    if re.search(term, node.content, re.IGNORECASE):
                        # Found match
                        snippet = self._extract_snippet(node.content, term)
                        matches.append(
                            {
                                "node_id": node.id,
                                "level": level.name,
                                "term": term,
                                "snippet": snippet,
                            }
                        )
                        nodes_used.append(node.id)
                        examined_tokens += node.token_count

                        if len(matches) >= 10:
                            break
                if len(matches) >= 10:
                    break
            if len(matches) >= 10:
                break

        if not matches:
            # Try original content as fallback
            for term in search_terms:
                for match in re.finditer(term, context.original_content, re.IGNORECASE):
                    start = max(0, match.start() - 200)
                    end = min(len(context.original_content), match.end() + 200)
                    snippet = context.original_content[start:end]
                    matches.append(
                        {
                            "node_id": "original",
                            "level": "FULL",
                            "term": term,
                            "snippet": f"...{snippet}...",
                        }
                    )
                    if len(matches) >= 10:
                        break

        # Format answer
        if matches:
            answer_parts = [f"Found {len(matches)} matches for query:\n"]
            for m in matches:
                answer_parts.append(f"[{m['level']}:{m['node_id']}] {m['snippet']}")
            answer = "\n\n".join(answer_parts)
            confidence = min(0.9, 0.5 + len(matches) * 0.05)
        else:
            answer = f"No matches found for terms: {search_terms}"
            confidence = 0.3

        return StrategyResult(
            answer=answer,
            confidence=confidence,
            nodes_used=list(set(nodes_used)),
            tokens_examined=examined_tokens,
            sub_calls=0,
        )

    def _extract_search_terms(self, query: str) -> list[str]:
        """Extract search terms from natural language query."""
        # Remove common question words
        stopwords = {
            "what",
            "where",
            "when",
            "who",
            "why",
            "how",
            "is",
            "are",
            "was",
            "were",
            "the",
            "a",
            "an",
            "of",
            "in",
            "to",
            "for",
            "with",
            "on",
            "at",
            "by",
            "about",
            "does",
            "did",
            "do",
            "can",
            "could",
            "would",
            "should",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
        }

        # Tokenize and filter
        words = re.findall(r"\b\w+\b", query.lower())
        terms = [w for w in words if w not in stopwords and len(w) > 2]

        # Also extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        terms.extend(quoted)

        return terms[:5]  # Limit to top 5 terms

    def _extract_snippet(self, content: str, term: str, context_chars: int = 200) -> str:
        """Extract snippet around a match."""
        match = re.search(term, content, re.IGNORECASE)
        if not match:
            return content[:300] + "..."

        start = max(0, match.start() - context_chars)
        end = min(len(content), match.end() + context_chars)
        snippet = content[start:end]

        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet


class PartitionMapStrategy(BaseStrategy):
    """
    Partition+Map strategy: Chunk context and process in parallel.

    Best for:
    - Aggregation queries ("how many...", "list all...")
    - Comprehensive analysis
    - When answer might be anywhere in content
    """

    @property
    def strategy_type(self) -> DecompositionStrategy:
        return DecompositionStrategy.PARTITION_MAP

    async def execute(
        self,
        query: RLMQuery,
        context: RLMContext,
    ) -> StrategyResult:
        """Partition context and map query to each partition."""
        if not self.agent_call:
            # Fallback without LLM
            return await self._fallback_execute(query, context)

        # Get appropriate level based on query complexity
        source_level = query.start_level
        if source_level not in context.levels:
            # Fall back to available levels
            for level in [
                AbstractionLevel.SUMMARY,
                AbstractionLevel.DETAILED,
                AbstractionLevel.FULL,
            ]:
                if level in context.levels:
                    source_level = level
                    break

        nodes = context.levels.get(source_level, [])
        if not nodes:
            # Use original content chunks
            chunk_size = self.config.target_tokens * 4
            chunks = [
                context.original_content[i : i + chunk_size]
                for i in range(0, len(context.original_content), chunk_size)
            ]
        else:
            chunks = [node.content for node in nodes]

        # Map query to each chunk
        map_prompt = f"""Answer this question based ONLY on the provided content.
If the answer is not in the content, say "Not found in this section."

Question: {query.query}

Content:
{{chunk}}

Answer:"""

        # Process chunks (in parallel if configured)
        partial_answers = []
        sub_calls = 0

        if self.config.parallel_sub_calls:
            tasks = [
                self._process_chunk(map_prompt.format(chunk=chunk), i)
                for i, chunk in enumerate(chunks[: self.config.max_sub_calls])
            ]
            results = await asyncio.gather(*tasks)
            for result in results:
                if result:
                    partial_answers.append(result)
                    sub_calls += 1
        else:
            for i, chunk in enumerate(chunks[: self.config.max_sub_calls]):
                result = await self._process_chunk(map_prompt.format(chunk=chunk), i)
                if result:
                    partial_answers.append(result)
                    sub_calls += 1

        # Reduce: Combine partial answers
        if len(partial_answers) > 1:
            reduce_prompt = f"""Combine these partial answers into a comprehensive response.
Remove duplicates and resolve any contradictions.

Question: {query.query}

Partial answers:
{chr(10).join(f"[{i + 1}] {ans}" for i, ans in enumerate(partial_answers))}

Combined answer:"""

            try:
                final_answer = self.agent_call(reduce_prompt, self.config.root_model, "")
                sub_calls += 1
            except Exception as e:
                logger.error(f"Reduce step failed: {e}")
                final_answer = "\n\n".join(partial_answers)
        elif partial_answers:
            final_answer = partial_answers[0]
        else:
            final_answer = "No relevant information found in the content."

        # Calculate stats
        nodes_used = [node.id for node in nodes[: len(chunks)]] if nodes else []
        examined_tokens = sum(len(c) // 4 for c in chunks[: self.config.max_sub_calls])

        return StrategyResult(
            answer=final_answer,
            confidence=min(0.9, 0.4 + 0.1 * len(partial_answers)),
            nodes_used=nodes_used,
            tokens_examined=examined_tokens,
            sub_calls=sub_calls,
        )

    async def _process_chunk(self, prompt: str, index: int) -> Optional[str]:
        """Process a single chunk."""
        if not self.agent_call:
            return None

        try:
            response = self.agent_call(prompt, self.config.sub_model, "")
            # Filter out "not found" responses
            if "not found" in response.lower() or "no information" in response.lower():
                return None
            return response
        except Exception as e:
            logger.error(f"Chunk {index} processing failed: {e}")
            return None

    async def _fallback_execute(
        self,
        query: RLMQuery,
        context: RLMContext,
    ) -> StrategyResult:
        """Fallback when no LLM available."""
        # Use grep strategy as fallback
        grep = GrepStrategy(self.config, self.agent_call)
        return await grep.execute(query, context)


class SummarizeStrategy(BaseStrategy):
    """
    Summarize strategy: Recursive summarization for overview queries.

    Best for:
    - "What is this about?"
    - "Summarize the key points"
    - Understanding the overall content
    """

    @property
    def strategy_type(self) -> DecompositionStrategy:
        return DecompositionStrategy.SUMMARIZE

    async def execute(
        self,
        query: RLMQuery,
        context: RLMContext,
    ) -> StrategyResult:
        """Use hierarchical summaries to answer."""
        # Start at abstract level and work down if needed
        answer_parts: list[str] = []
        nodes_used: list[str] = []
        examined_tokens = 0

        # Try abstract level first
        if AbstractionLevel.ABSTRACT in context.levels:
            abstract_nodes = context.levels[AbstractionLevel.ABSTRACT]
            abstract_content = "\n\n".join(n.content for n in abstract_nodes)
            answer_parts.append(f"**Overview:**\n{abstract_content}")
            nodes_used.extend(n.id for n in abstract_nodes)
            examined_tokens += sum(n.token_count for n in abstract_nodes)

        # Add summary level for more detail
        if AbstractionLevel.SUMMARY in context.levels:
            summary_nodes = context.levels[AbstractionLevel.SUMMARY]
            summary_content = "\n\n".join(n.content for n in summary_nodes[:3])
            answer_parts.append(f"**Key Points:**\n{summary_content}")
            nodes_used.extend(n.id for n in summary_nodes[:3])
            examined_tokens += sum(n.token_count for n in summary_nodes[:3])

        # If no hierarchical summaries, use LLM to summarize
        if not answer_parts:
            if self.agent_call:
                prompt = f"""Provide a brief summary of this content.

Content:
{context.original_content[:8000]}

Summary:"""
                try:
                    summary = self.agent_call(prompt, self.config.root_model, "")
                    answer_parts.append(summary)
                except Exception as e:
                    logger.error(f"Summarization failed: {e}")
                    answer_parts.append(context.original_content[:1000] + "...")
            else:
                answer_parts.append(context.original_content[:1000] + "...")

        return StrategyResult(
            answer="\n\n".join(answer_parts),
            confidence=0.8 if AbstractionLevel.ABSTRACT in context.levels else 0.5,
            nodes_used=nodes_used,
            tokens_examined=examined_tokens,
            sub_calls=1 if (not answer_parts and self.agent_call) else 0,
        )


class HierarchicalStrategy(BaseStrategy):
    """
    Hierarchical strategy: Navigate pre-built abstraction levels.

    Best for:
    - Queries that need both overview and detail
    - Complex questions requiring multi-level understanding
    - When context is already well-compressed
    """

    @property
    def strategy_type(self) -> DecompositionStrategy:
        return DecompositionStrategy.HIERARCHICAL

    async def execute(
        self,
        query: RLMQuery,
        context: RLMContext,
    ) -> StrategyResult:
        """Navigate hierarchy based on query."""
        if not self.agent_call:
            # Fallback to summarize
            summarize = SummarizeStrategy(self.config, self.agent_call)
            return await summarize.execute(query, context)

        nodes_used = []
        examined_tokens = 0
        sub_calls = 0

        # Step 1: Examine abstract level to identify relevant areas
        relevant_nodes = []

        if AbstractionLevel.ABSTRACT in context.levels:
            abstract_content = "\n".join(
                f"[{n.id}] {n.content}" for n in context.levels[AbstractionLevel.ABSTRACT]
            )

            identify_prompt = f"""Based on this overview, which sections are most relevant to the question?
Return the node IDs (e.g., L3_0, L3_1) that are relevant.

Question: {query.query}

Overview:
{abstract_content}

Relevant node IDs (comma-separated):"""

            try:
                relevant_ids = self.agent_call(identify_prompt, self.config.sub_model, "")
                relevant_nodes = [
                    id.strip()
                    for id in relevant_ids.split(",")
                    if id.strip() in context.nodes_by_id
                ]
                sub_calls += 1
                examined_tokens += len(abstract_content) // 4
            except Exception as e:
                logger.error(f"Relevance identification failed: {e}")

        # Step 2: Drill down into relevant sections
        detailed_content = []

        for node_id in relevant_nodes[:3]:  # Limit drilling
            node = context.nodes_by_id.get(node_id)
            if not node:
                continue

            nodes_used.append(node_id)

            # Get children for more detail
            children = context.drill_down(node_id)
            if children:
                for child in children[:2]:
                    detailed_content.append(f"[{child.id}]\n{child.content}")
                    nodes_used.append(child.id)
                    examined_tokens += child.token_count
            else:
                detailed_content.append(f"[{node_id}]\n{node.content}")
                examined_tokens += node.token_count

        # Step 3: Answer using detailed content
        if detailed_content:
            answer_prompt = f"""Answer this question using the provided context.
Include citations to the relevant sections using [node_id].

Question: {query.query}

Context:
{chr(10).join(detailed_content)}

Answer with citations:"""

            try:
                answer = self.agent_call(answer_prompt, self.config.root_model, "")
                sub_calls += 1
            except Exception as e:
                logger.error(f"Answer generation failed: {e}")
                answer = f"Based on the content: {detailed_content[0][:500]}..."
        else:
            # Fall back to summary-based answer
            summary = context.get_at_level(AbstractionLevel.SUMMARY)
            answer = f"Based on available summaries: {summary[:1000]}..."

        return StrategyResult(
            answer=answer,
            confidence=0.85 if len(detailed_content) > 0 else 0.5,
            nodes_used=list(set(nodes_used)),
            tokens_examined=examined_tokens,
            sub_calls=sub_calls,
        )


class AutoStrategy(BaseStrategy):
    """
    Auto strategy: Automatically select the best strategy.

    Analyzes the query to determine the most appropriate approach.
    """

    @property
    def strategy_type(self) -> DecompositionStrategy:
        return DecompositionStrategy.AUTO

    async def execute(
        self,
        query: RLMQuery,
        context: RLMContext,
    ) -> StrategyResult:
        """Automatically select and execute the best strategy."""
        strategy = self._select_strategy(query, context)
        return await strategy.execute(query, context)

    def _select_strategy(self, query: RLMQuery, context: RLMContext) -> BaseStrategy:
        """Select strategy based on query characteristics."""
        q = query.query.lower()

        # Summarize strategy for overview queries
        if any(word in q for word in ["summary", "summarize", "overview", "about", "main"]):
            return SummarizeStrategy(self.config, self.agent_call)

        # Grep strategy for specific lookups
        if any(word in q for word in ["where", "find", "locate", "which", "when"]):
            return GrepStrategy(self.config, self.agent_call)

        # Partition+Map for aggregation queries
        if any(word in q for word in ["all", "every", "list", "how many", "count"]):
            return PartitionMapStrategy(self.config, self.agent_call)

        # Peek for structural queries
        if any(word in q for word in ["structure", "sections", "outline", "format"]):
            return PeekStrategy(self.config, self.agent_call)

        # Default to hierarchical for complex queries
        if len(context.levels) > 2:
            return HierarchicalStrategy(self.config, self.agent_call)

        # Fall back to partition+map
        return PartitionMapStrategy(self.config, self.agent_call)


# Strategy registry
STRATEGIES: dict[DecompositionStrategy, type[BaseStrategy]] = {
    DecompositionStrategy.PEEK: PeekStrategy,
    DecompositionStrategy.GREP: GrepStrategy,
    DecompositionStrategy.PARTITION_MAP: PartitionMapStrategy,
    DecompositionStrategy.SUMMARIZE: SummarizeStrategy,
    DecompositionStrategy.HIERARCHICAL: HierarchicalStrategy,
    DecompositionStrategy.AUTO: AutoStrategy,
}


def get_strategy(
    strategy_type: DecompositionStrategy,
    config: RLMConfig,
    agent_call: Optional[Callable[[str, str, str], str]] = None,
) -> BaseStrategy:
    """Get a strategy instance by type."""
    strategy_class = STRATEGIES.get(strategy_type, AutoStrategy)
    return strategy_class(config, agent_call)
