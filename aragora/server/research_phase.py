"""
Pre-debate research phase for current events.

Performs web search to gather current information before debates
on time-sensitive topics.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import httpx

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)

# Default timeout for web requests
DEFAULT_TIMEOUT = 30.0


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    snippet: str
    source: str = ""


@dataclass
class ResearchResult:
    """Result of pre-debate research."""

    query: str
    results: list[SearchResult] = field(default_factory=list)
    summary: str = ""
    sources: list[str] = field(default_factory=list)
    is_current_event: bool = False

    def to_context(self) -> str:
        """Convert to debate context string."""
        if not self.results and not self.summary:
            return ""

        parts = ["## Background Research\n"]

        if self.summary:
            parts.append(f"{self.summary}\n")

        if self.results:
            parts.append("\n### Key Sources:\n")
            for i, result in enumerate(self.results[:5], 1):
                parts.append(f"{i}. [{result.title}]({result.url})")
                if result.snippet:
                    parts.append(f"   {result.snippet[:200]}...")
                parts.append("")

        return "\n".join(parts)


class PreDebateResearcher:
    """Performs web search for current events before debates."""

    # Keywords that suggest a current event question
    CURRENT_EVENT_INDICATORS = [
        "today",
        "yesterday",
        "this week",
        "this month",
        "recent",
        "latest",
        "breaking",
        "news",
        "announced",
        "just",
        "new",
        "2024",
        "2025",
        "2026",
        "lawsuit",
        "ruling",
        "decision",
        "election",
        "happening",
        "update",
        "currently",
    ]

    def __init__(
        self,
        brave_api_key: Optional[str] = None,
        serper_api_key: Optional[str] = None,
        anthropic_client: Optional["anthropic.Anthropic"] = None,
    ):
        """Initialize the researcher.

        Args:
            brave_api_key: Brave Search API key (from env BRAVE_API_KEY)
            serper_api_key: Serper API key (from env SERPER_API_KEY)
            anthropic_client: Optional Anthropic client for summarization
        """
        self.brave_api_key = brave_api_key or os.getenv("BRAVE_API_KEY")
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        self._anthropic_client = anthropic_client

    @property
    def anthropic_client(self) -> "anthropic.Anthropic":
        """Get or create the Anthropic client."""
        if self._anthropic_client is None:
            import anthropic

            self._anthropic_client = anthropic.Anthropic()
        return self._anthropic_client

    def is_current_event(self, question: str) -> bool:
        """Check if the question relates to current events.

        Uses simple keyword matching for fast classification.
        """
        question_lower = question.lower()

        # Check for year references
        if any(year in question_lower for year in ["2024", "2025", "2026"]):
            return True

        # Check for current event indicators
        match_count = sum(
            1 for indicator in self.CURRENT_EVENT_INDICATORS if indicator in question_lower
        )

        return match_count >= 2

    async def _classify_with_llm(self, question: str) -> bool:
        """Use Claude to determine if question requires current info.

        More accurate but requires API call.
        """
        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Does this debate question require current/recent information to answer well?

Question: {question}

Respond with just "yes" or "no".""",
                    }
                ],
            )
            content = response.content[0].text.strip().lower()
            return content.startswith("yes")
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return self.is_current_event(question)

    async def search_brave(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search using Brave Search API."""
        if not self.brave_api_key:
            return []

        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": max_results},
                    headers={"X-Subscription-Token": self.brave_api_key},
                )
                response.raise_for_status()
                data = response.json()

                results = []
                for item in data.get("web", {}).get("results", [])[:max_results]:
                    results.append(
                        SearchResult(
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            snippet=item.get("description", ""),
                            source="brave",
                        )
                    )
                return results

        except Exception as e:
            logger.warning(f"Brave search failed: {e}")
            return []

    async def search_serper(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search using Serper (Google Search) API."""
        if not self.serper_api_key:
            return []

        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                response = await client.post(
                    "https://google.serper.dev/search",
                    json={"q": query, "num": max_results},
                    headers={
                        "X-API-KEY": self.serper_api_key,
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                data = response.json()

                results = []
                for item in data.get("organic", [])[:max_results]:
                    results.append(
                        SearchResult(
                            title=item.get("title", ""),
                            url=item.get("link", ""),
                            snippet=item.get("snippet", ""),
                            source="serper",
                        )
                    )
                return results

        except Exception as e:
            logger.warning(f"Serper search failed: {e}")
            return []

    def _extract_search_query(self, question: str) -> str:
        """Extract an effective search query from the debate question."""
        # Remove common debate framing words
        framing_words = [
            "debate",
            "discuss",
            "argue",
            "should",
            "could",
            "would",
            "pros and cons",
            "implications",
            "what are",
            "why is",
            "how does",
        ]

        query = question
        for word in framing_words:
            query = re.sub(rf"\b{word}\b", "", query, flags=re.IGNORECASE)

        # Clean up whitespace
        query = " ".join(query.split())

        # Limit length
        words = query.split()
        if len(words) > 10:
            query = " ".join(words[:10])

        return query.strip()

    async def search(
        self, question: str, max_results: int = 5
    ) -> ResearchResult:
        """Perform web search for the question.

        Tries available search APIs in order of preference.
        """
        query = self._extract_search_query(question)
        logger.info(f"Searching for: {query}")

        results: list[SearchResult] = []

        # Try Brave first, then Serper
        if self.brave_api_key:
            results = await self.search_brave(query, max_results)

        if not results and self.serper_api_key:
            results = await self.search_serper(query, max_results)

        if not results:
            logger.info("No search results (no API keys configured or search failed)")
            return ResearchResult(
                query=query,
                is_current_event=self.is_current_event(question),
            )

        # Extract unique sources
        sources = list(set(r.url for r in results if r.url))

        return ResearchResult(
            query=query,
            results=results,
            sources=sources,
            is_current_event=True,
        )

    async def research_and_summarize(
        self, question: str, max_results: int = 5
    ) -> ResearchResult:
        """Search and summarize results using Claude.

        This provides a synthesized context for the debate.
        """
        # First do the search
        result = await self.search(question, max_results)

        if not result.results:
            return result

        # Summarize the results
        try:
            snippets = "\n\n".join(
                f"Source: {r.title}\n{r.snippet}" for r in result.results[:5]
            )

            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Summarize these search results about: {question}

{snippets}

Provide a brief, factual summary (2-3 paragraphs) of the current situation.
Focus on facts, not opinions. Include relevant dates and specifics.""",
                    }
                ],
            )

            result.summary = response.content[0].text

        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            # Return without summary
            pass

        return result


async def research_question(
    question: str,
    summarize: bool = True,
    max_results: int = 5,
) -> Optional[ResearchResult]:
    """Convenience function to research a debate question.

    Args:
        question: The debate question
        summarize: Whether to use Claude to summarize results
        max_results: Maximum search results to fetch

    Returns:
        ResearchResult if the question needs current info, None otherwise
    """
    researcher = PreDebateResearcher()

    # Quick check if this needs research
    if not researcher.is_current_event(question):
        logger.debug("Question does not require current event research")
        return None

    if summarize:
        return await researcher.research_and_summarize(question, max_results)
    else:
        return await researcher.search(question, max_results)
