"""
Comprehensive tests for the aragora.server.research_phase module.

Tests cover:
- SearchResult and ResearchResult dataclasses
- PreDebateResearcher initialization and configuration
- Current event detection (keyword and LLM-based)
- Search functionality (Brave, Serper, Claude web search)
- Result aggregation and summarization
- Error handling and timeouts
- Concurrency management
- Integration with knowledge systems
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# SearchResult Tests
# =============================================================================


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_minimal_creation(self):
        """SearchResult can be created with required fields only."""
        from aragora.server.research_phase import SearchResult

        result = SearchResult(
            title="Test Article",
            url="https://example.com/article",
            snippet="This is a test snippet.",
        )

        assert result.title == "Test Article"
        assert result.url == "https://example.com/article"
        assert result.snippet == "This is a test snippet."
        assert result.source == ""

    def test_full_creation(self):
        """SearchResult can be created with all fields."""
        from aragora.server.research_phase import SearchResult

        result = SearchResult(
            title="Full Article",
            url="https://example.com/full",
            snippet="Full snippet text.",
            source="brave",
        )

        assert result.title == "Full Article"
        assert result.source == "brave"

    def test_different_sources(self):
        """SearchResult supports different source values."""
        from aragora.server.research_phase import SearchResult

        sources = ["brave", "serper", "claude_web_search", ""]

        for source in sources:
            result = SearchResult(
                title="Test",
                url="https://test.com",
                snippet="Test",
                source=source,
            )
            assert result.source == source


# =============================================================================
# ResearchResult Tests
# =============================================================================


class TestResearchResult:
    """Tests for ResearchResult dataclass."""

    def test_minimal_creation(self):
        """ResearchResult can be created with query only."""
        from aragora.server.research_phase import ResearchResult

        result = ResearchResult(query="test query")

        assert result.query == "test query"
        assert result.results == []
        assert result.summary == ""
        assert result.sources == []
        assert result.is_current_event is False

    def test_full_creation(self):
        """ResearchResult can be created with all fields."""
        from aragora.server.research_phase import ResearchResult, SearchResult

        search_results = [
            SearchResult(
                title="Article 1",
                url="https://example.com/1",
                snippet="Snippet 1",
                source="brave",
            )
        ]

        result = ResearchResult(
            query="full query",
            results=search_results,
            summary="This is a summary.",
            sources=["https://example.com/1"],
            is_current_event=True,
        )

        assert result.query == "full query"
        assert len(result.results) == 1
        assert result.summary == "This is a summary."
        assert result.sources == ["https://example.com/1"]
        assert result.is_current_event is True

    def test_to_context_empty(self):
        """to_context returns empty string when no results or summary."""
        from aragora.server.research_phase import ResearchResult

        result = ResearchResult(query="empty query")

        assert result.to_context() == ""

    def test_to_context_with_summary_only(self):
        """to_context includes summary when present."""
        from aragora.server.research_phase import ResearchResult

        result = ResearchResult(
            query="summary query",
            summary="This is a research summary.",
        )

        context = result.to_context()

        assert "## Background Research" in context
        assert "This is a research summary." in context

    def test_to_context_with_results(self):
        """to_context includes formatted results."""
        from aragora.server.research_phase import ResearchResult, SearchResult

        result = ResearchResult(
            query="results query",
            results=[
                SearchResult(
                    title="Source 1",
                    url="https://source1.com",
                    snippet="Description of source 1",
                    source="brave",
                ),
                SearchResult(
                    title="Source 2",
                    url="https://source2.com",
                    snippet="Description of source 2",
                    source="serper",
                ),
            ],
            summary="Summary text here.",
        )

        context = result.to_context()

        assert "### Key Sources:" in context
        assert "[Source 1](https://source1.com)" in context
        assert "[Source 2](https://source2.com)" in context
        assert "Description of source 1" in context

    def test_to_context_limits_results_to_five(self):
        """to_context only shows first 5 results."""
        from aragora.server.research_phase import ResearchResult, SearchResult

        results = [
            SearchResult(
                title=f"Source {i}",
                url=f"https://source{i}.com",
                snippet=f"Snippet {i}",
            )
            for i in range(10)
        ]

        result = ResearchResult(
            query="many results",
            results=results,
            summary="Summary",
        )

        context = result.to_context()

        assert "[Source 0]" in context
        assert "[Source 4]" in context
        assert "[Source 5]" not in context
        assert "[Source 9]" not in context

    def test_to_context_truncates_long_snippets(self):
        """to_context truncates snippets over 200 characters."""
        from aragora.server.research_phase import ResearchResult, SearchResult

        long_snippet = "x" * 500
        result = ResearchResult(
            query="long snippet",
            results=[
                SearchResult(
                    title="Long Source",
                    url="https://long.com",
                    snippet=long_snippet,
                )
            ],
            summary="Summary",
        )

        context = result.to_context()

        assert "..." in context
        assert long_snippet not in context


# =============================================================================
# PreDebateResearcher Initialization Tests
# =============================================================================


class TestPreDebateResearcherInit:
    """Tests for PreDebateResearcher initialization."""

    def test_default_initialization(self):
        """PreDebateResearcher initializes with environment variables."""
        from aragora.server.research_phase import PreDebateResearcher

        with patch.dict(os.environ, {}, clear=True):
            researcher = PreDebateResearcher()

        assert researcher.brave_api_key is None
        assert researcher.serper_api_key is None
        assert researcher._anthropic_client is None

    def test_initialization_with_api_keys(self):
        """PreDebateResearcher accepts API keys as parameters."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher(
            brave_api_key="brave-key-123",
            serper_api_key="serper-key-456",
        )

        assert researcher.brave_api_key == "brave-key-123"
        assert researcher.serper_api_key == "serper-key-456"

    def test_initialization_from_environment(self):
        """PreDebateResearcher reads API keys from environment."""
        from aragora.server.research_phase import PreDebateResearcher

        with patch.dict(
            os.environ,
            {
                "BRAVE_API_KEY": "env-brave-key",
                "SERPER_API_KEY": "env-serper-key",
            },
        ):
            researcher = PreDebateResearcher()

        assert researcher.brave_api_key == "env-brave-key"
        assert researcher.serper_api_key == "env-serper-key"

    def test_initialization_with_anthropic_client(self):
        """PreDebateResearcher accepts an Anthropic client."""
        from aragora.server.research_phase import PreDebateResearcher

        mock_client = MagicMock()
        researcher = PreDebateResearcher(anthropic_client=mock_client)

        assert researcher._anthropic_client is mock_client

    def test_anthropic_client_lazy_initialization(self):
        """Anthropic client is lazily initialized on first access."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher()
        assert researcher._anthropic_client is None

        # Create a mock anthropic module for environments where anthropic isn't installed
        mock_anthropic_mod = MagicMock()
        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_mod}):
            client = researcher.anthropic_client

        assert client is mock_client
        mock_anthropic_mod.Anthropic.assert_called_once()


# =============================================================================
# Current Event Detection Tests
# =============================================================================


class TestIsCurrentEvent:
    """Tests for is_current_event method."""

    def test_detects_year_references(self):
        """is_current_event returns True for year references."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher()

        assert researcher.is_current_event("What happened in 2024?") is True
        assert researcher.is_current_event("News from 2025") is True
        assert researcher.is_current_event("Predictions for 2026") is True

    def test_detects_temporal_keywords(self):
        """is_current_event returns True for multiple temporal keywords."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher()

        # Needs 2+ indicators to return True
        assert (
            researcher.is_current_event("What is happening today now?") is True
        )  # "happening" + "today"
        assert (
            researcher.is_current_event("Recent news developments in AI") is True
        )  # "recent" + "news"
        assert (
            researcher.is_current_event("The latest news update on climate") is True
        )  # "latest" + "news" + "update"

    def test_requires_multiple_indicators(self):
        """is_current_event requires 2+ indicators (or year)."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher()

        # Single indicator should return False
        assert researcher.is_current_event("Today is nice") is False  # "today" alone = 1 indicator
        assert (
            researcher.is_current_event("What is the capital of France?") is False
        )  # No indicators

        # Multiple indicators should return True
        assert (
            researcher.is_current_event("Recent news update") is True
        )  # "recent" + "news" + "update"

    def test_case_insensitive(self):
        """is_current_event is case insensitive."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher()

        assert researcher.is_current_event("WHAT IS THE LATEST NEWS?") is True
        assert researcher.is_current_event("Recent Update on Elections") is True

    def test_historical_questions(self):
        """is_current_event returns False for historical questions."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher()

        assert researcher.is_current_event("What caused World War I?") is False
        assert researcher.is_current_event("History of ancient Rome") is False


class TestClassifyWithLLM:
    """Tests for LLM-based classification."""

    @pytest.mark.asyncio
    async def test_classifies_yes_response(self):
        """_classify_with_llm returns True for 'yes' responses."""
        from aragora.server.research_phase import PreDebateResearcher

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="yes")]
        mock_client.messages.create.return_value = mock_response

        researcher = PreDebateResearcher(anthropic_client=mock_client)

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_response
            result = await researcher._classify_with_llm("What is the latest news?")

        assert result is True

    @pytest.mark.asyncio
    async def test_classifies_no_response(self):
        """_classify_with_llm returns False for 'no' responses."""
        from aragora.server.research_phase import PreDebateResearcher

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="no")]

        researcher = PreDebateResearcher()

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_response
            result = await researcher._classify_with_llm("What is 2+2?")

        assert result is False

    @pytest.mark.asyncio
    async def test_falls_back_on_error(self):
        """_classify_with_llm falls back to keyword detection on error."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher()

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = RuntimeError("API error")
            result = await researcher._classify_with_llm("Latest news update 2025")

        # Falls back to is_current_event which should return True
        assert result is True


# =============================================================================
# Search Functionality Tests
# =============================================================================


class TestSearchBrave:
    """Tests for Brave Search API integration."""

    @pytest.mark.asyncio
    async def test_returns_empty_without_api_key(self):
        """search_brave returns empty list without API key."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher(brave_api_key=None)
        results = await researcher.search_brave("test query")

        assert results == []

    @pytest.mark.asyncio
    async def test_returns_results_on_success(self):
        """search_brave returns results on successful API call."""
        from aragora.server.research_phase import PreDebateResearcher

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Result 1",
                        "url": "https://result1.com",
                        "description": "Description 1",
                    },
                    {
                        "title": "Result 2",
                        "url": "https://result2.com",
                        "description": "Description 2",
                    },
                ]
            }
        }

        researcher = PreDebateResearcher(brave_api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            results = await researcher.search_brave("test query", max_results=5)

        assert len(results) == 2
        assert results[0].title == "Result 1"
        assert results[0].source == "brave"
        assert results[1].url == "https://result2.com"

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self):
        """search_brave returns empty list on API error."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher(brave_api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=RuntimeError("API error")
            )

            results = await researcher.search_brave("test query")

        assert results == []

    @pytest.mark.asyncio
    async def test_respects_max_results(self):
        """search_brave respects max_results parameter."""
        from aragora.server.research_phase import PreDebateResearcher

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {"title": f"Result {i}", "url": f"https://r{i}.com", "description": f"Desc {i}"}
                    for i in range(10)
                ]
            }
        }

        researcher = PreDebateResearcher(brave_api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            results = await researcher.search_brave("test", max_results=3)

        assert len(results) == 3


class TestSearchSerper:
    """Tests for Serper/Google Search API integration."""

    @pytest.mark.asyncio
    async def test_returns_empty_without_api_key(self):
        """search_serper returns empty list without API key."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher(serper_api_key=None)
        results = await researcher.search_serper("test query")

        assert results == []

    @pytest.mark.asyncio
    async def test_returns_results_on_success(self):
        """search_serper returns results on successful API call."""
        from aragora.server.research_phase import PreDebateResearcher

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "organic": [
                {
                    "title": "Serper Result 1",
                    "link": "https://serper1.com",
                    "snippet": "Snippet 1",
                },
                {
                    "title": "Serper Result 2",
                    "link": "https://serper2.com",
                    "snippet": "Snippet 2",
                },
            ]
        }

        researcher = PreDebateResearcher(serper_api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            results = await researcher.search_serper("test query", max_results=5)

        assert len(results) == 2
        assert results[0].title == "Serper Result 1"
        assert results[0].source == "serper"
        assert results[1].url == "https://serper2.com"

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self):
        """search_serper returns empty list on API error."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher(serper_api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=RuntimeError("API error")
            )

            results = await researcher.search_serper("test query")

        assert results == []


class TestSearchWithClaude:
    """Tests for Claude web search integration."""

    @pytest.mark.asyncio
    async def test_returns_result_on_success(self):
        """search_with_claude returns ResearchResult on success."""
        from aragora.server.research_phase import PreDebateResearcher

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text="Research summary with https://example.com/source URL.")
        ]

        researcher = PreDebateResearcher()

        with patch("asyncio.wait_for", new_callable=AsyncMock) as mock_wait:
            mock_wait.return_value = mock_response

            result = await researcher.search_with_claude("test question")

        assert result.query == "test question"
        assert "Research summary" in result.summary
        assert result.is_current_event is True
        assert "https://example.com/source" in result.sources

    @pytest.mark.asyncio
    async def test_extracts_urls_from_summary(self):
        """search_with_claude extracts URLs from response."""
        from aragora.server.research_phase import PreDebateResearcher

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text="Summary with multiple sources: https://source1.com and https://source2.com/path"
            )
        ]

        researcher = PreDebateResearcher()

        with patch("asyncio.wait_for", new_callable=AsyncMock) as mock_wait:
            mock_wait.return_value = mock_response

            result = await researcher.search_with_claude("test")

        assert len(result.sources) == 2
        assert "https://source1.com" in result.sources
        assert "https://source2.com/path" in result.sources

    @pytest.mark.asyncio
    async def test_returns_empty_on_timeout(self):
        """search_with_claude returns empty result on timeout."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher()

        with patch("asyncio.wait_for", new_callable=AsyncMock) as mock_wait:
            mock_wait.side_effect = asyncio.TimeoutError()

            result = await researcher.search_with_claude("test question")

        assert result.query == "test question"
        assert result.summary == ""
        assert result.results == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self):
        """search_with_claude returns empty result on error."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher()

        with patch("asyncio.wait_for", new_callable=AsyncMock) as mock_wait:
            mock_wait.side_effect = RuntimeError("API error")

            result = await researcher.search_with_claude("test question")

        assert result.query == "test question"
        assert result.summary == ""


# =============================================================================
# Query Extraction Tests
# =============================================================================


class TestExtractSearchQuery:
    """Tests for search query extraction."""

    def test_removes_framing_words(self):
        """_extract_search_query removes debate framing words."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher()

        query = researcher._extract_search_query("Debate the implications of AI")
        assert "debate" not in query.lower()
        assert "implications" not in query.lower()
        assert "AI" in query

    def test_limits_query_length(self):
        """_extract_search_query limits query to 10 words."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher()

        long_question = " ".join(["word"] * 20)
        query = researcher._extract_search_query(long_question)

        assert len(query.split()) <= 10

    def test_preserves_key_terms(self):
        """_extract_search_query preserves key terms."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher()

        query = researcher._extract_search_query("What are the pros and cons of electric vehicles?")
        assert "electric" in query.lower()
        assert "vehicles" in query.lower()


# =============================================================================
# Domain Extraction Tests
# =============================================================================


class TestExtractDomain:
    """Tests for domain extraction from URLs."""

    def test_extracts_domain(self):
        """_extract_domain extracts domain from URL."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher()

        assert researcher._extract_domain("https://example.com/path") == "example.com"
        assert researcher._extract_domain("https://sub.domain.org/page") == "sub.domain.org"

    def test_handles_invalid_urls(self):
        """_extract_domain handles invalid URLs gracefully."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher()

        result = researcher._extract_domain("not-a-url")
        assert result == "not-a-url"  # Returns truncated original


# =============================================================================
# Search Method Tests
# =============================================================================


class TestSearch:
    """Tests for the main search method."""

    @pytest.mark.asyncio
    async def test_tries_brave_first(self):
        """search tries Brave API first when available."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher(brave_api_key="brave-key")

        mock_results = [MagicMock(url="https://brave.com")]

        with patch.object(
            researcher, "search_brave", new_callable=AsyncMock, return_value=mock_results
        ) as mock_brave:
            with patch.object(researcher, "search_serper", new_callable=AsyncMock) as mock_serper:
                result = await researcher.search("test query")

        mock_brave.assert_called_once()
        mock_serper.assert_not_called()
        assert len(result.results) == 1

    @pytest.mark.asyncio
    async def test_falls_back_to_serper(self):
        """search falls back to Serper when Brave returns no results."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher(brave_api_key="brave-key", serper_api_key="serper-key")

        mock_serper_results = [MagicMock(url="https://serper.com")]

        with patch.object(researcher, "search_brave", new_callable=AsyncMock, return_value=[]):
            with patch.object(
                researcher,
                "search_serper",
                new_callable=AsyncMock,
                return_value=mock_serper_results,
            ) as mock_serper:
                result = await researcher.search("test query")

        mock_serper.assert_called_once()
        assert len(result.results) == 1

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_apis(self):
        """search returns empty result when no APIs are configured."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher()

        result = await researcher.search("test query")

        assert result.results == []


# =============================================================================
# Research and Summarize Tests
# =============================================================================


class TestResearchAndSummarize:
    """Tests for research_and_summarize method."""

    @pytest.mark.asyncio
    async def test_uses_claude_search_by_default(self):
        """research_and_summarize uses Claude web search by default."""
        from aragora.server.research_phase import PreDebateResearcher

        mock_result = MagicMock()
        mock_result.summary = "Claude search summary"

        researcher = PreDebateResearcher()

        with patch.object(
            researcher, "search_with_claude", new_callable=AsyncMock, return_value=mock_result
        ) as mock_claude:
            result = await researcher.research_and_summarize("test question")

        mock_claude.assert_called_once()
        assert result.summary == "Claude search summary"

    @pytest.mark.asyncio
    async def test_falls_back_to_external_apis(self):
        """research_and_summarize falls back when Claude search fails."""
        from aragora.server.research_phase import PreDebateResearcher, ResearchResult

        empty_claude_result = ResearchResult(query="test", summary="")

        mock_search_result = ResearchResult(
            query="test",
            results=[MagicMock(title="External", snippet="External result", url="https://ext.com")],
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Summarized content")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        researcher = PreDebateResearcher(anthropic_client=mock_client)

        with patch.object(
            researcher,
            "search_with_claude",
            new_callable=AsyncMock,
            return_value=empty_claude_result,
        ):
            with patch.object(
                researcher, "search", new_callable=AsyncMock, return_value=mock_search_result
            ):
                result = await researcher.research_and_summarize("test question")

        assert result.summary == "Summarized content"

    @pytest.mark.asyncio
    async def test_uses_claude_knowledge_fallback(self):
        """research_and_summarize uses Claude knowledge when no external results."""
        from aragora.server.research_phase import PreDebateResearcher, ResearchResult

        empty_result = ResearchResult(query="test", summary="", results=[])

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Knowledge-based summary")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        researcher = PreDebateResearcher(anthropic_client=mock_client)

        with patch.object(
            researcher, "search_with_claude", new_callable=AsyncMock, return_value=empty_result
        ):
            with patch.object(
                researcher, "search", new_callable=AsyncMock, return_value=empty_result
            ):
                result = await researcher.research_and_summarize("test question")

        assert "Background Context" in result.summary or "Knowledge-based" in result.summary


# =============================================================================
# OpenRouter Fallback Tests
# =============================================================================


class TestOpenRouterFallback:
    """Tests for OpenRouter fallback behavior in research paths."""

    @pytest.mark.asyncio
    async def test_generate_text_with_fallback_uses_openrouter_on_billing_failure(self):
        """Billing/API failures should fall back to OpenRouter when available."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher()
        researcher._enable_openrouter_fallback = True

        mock_openrouter = MagicMock()
        mock_openrouter.generate = AsyncMock(return_value="Fallback summary text")

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = RuntimeError("credit balance is too low")
            with patch.object(researcher, "_get_openrouter_agent", return_value=mock_openrouter):
                text = await researcher._generate_text_with_fallback(
                    "Summarize this topic",
                    max_tokens=300,
                    timeout_seconds=5.0,
                )

        assert text == "Fallback summary text"
        mock_openrouter.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_text_with_fallback_raises_when_fallback_unavailable(self):
        """If fallback is unavailable, the original provider error should surface."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher()
        researcher._enable_openrouter_fallback = True

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = RuntimeError("credit balance is too low")
            with patch.object(researcher, "_get_openrouter_agent", return_value=None):
                with pytest.raises(RuntimeError, match="credit balance is too low"):
                    await researcher._generate_text_with_fallback(
                        "Summarize this topic",
                        max_tokens=300,
                        timeout_seconds=5.0,
                    )


# =============================================================================
# Module-Level Function Tests
# =============================================================================


class TestResearchQuestion:
    """Tests for research_question module function."""

    @pytest.mark.asyncio
    async def test_force_research_always_searches(self):
        """research_question always searches when force_research=True."""
        from aragora.server.research_phase import research_question

        with patch("aragora.server.research_phase.PreDebateResearcher") as mock_researcher_class:
            mock_researcher = MagicMock()
            mock_researcher.is_current_event.return_value = False
            mock_researcher.research_and_summarize = AsyncMock(return_value=MagicMock())
            mock_researcher_class.return_value = mock_researcher

            await research_question("What is 2+2?", force_research=True)

        mock_researcher.research_and_summarize.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_non_current_events(self):
        """research_question skips non-current event questions when not forced."""
        from aragora.server.research_phase import research_question

        with patch("aragora.server.research_phase.PreDebateResearcher") as mock_researcher_class:
            mock_researcher = MagicMock()
            mock_researcher.is_current_event.return_value = False
            mock_researcher_class.return_value = mock_researcher

            result = await research_question("What is 2+2?", force_research=False)

        assert result is None

    @pytest.mark.asyncio
    async def test_researches_current_events(self):
        """research_question researches current event questions."""
        from aragora.server.research_phase import research_question

        with patch("aragora.server.research_phase.PreDebateResearcher") as mock_researcher_class:
            mock_researcher = MagicMock()
            mock_researcher.is_current_event.return_value = True
            mock_researcher.research_and_summarize = AsyncMock(
                return_value=MagicMock(summary="Summary")
            )
            mock_researcher_class.return_value = mock_researcher

            result = await research_question("Latest news 2025", force_research=False)

        assert result is not None


class TestResearchForDebate:
    """Tests for research_for_debate module function."""

    @pytest.mark.asyncio
    async def test_returns_formatted_context(self):
        """research_for_debate returns formatted context string."""
        from aragora.server.research_phase import ResearchResult, research_for_debate

        mock_result = ResearchResult(
            query="test",
            summary="Research summary here.",
        )

        with patch(
            "aragora.server.research_phase.research_question",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            context = await research_for_debate("test question")

        assert "Background Research" in context
        assert "Research summary here." in context

    @pytest.mark.asyncio
    async def test_returns_empty_on_no_results(self):
        """research_for_debate returns empty string when no results."""
        from aragora.server.research_phase import research_for_debate

        with patch(
            "aragora.server.research_phase.research_question",
            new_callable=AsyncMock,
            return_value=None,
        ):
            context = await research_for_debate("test question")

        assert context == ""

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self):
        """research_for_debate returns empty string on exception."""
        from aragora.server.research_phase import research_for_debate

        with patch(
            "aragora.server.research_phase.research_question",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Research failed"),
        ):
            context = await research_for_debate("test question")

        assert context == ""


# =============================================================================
# Timeout and Error Handling Tests
# =============================================================================


class TestTimeoutHandling:
    """Tests for timeout handling in research operations."""

    @pytest.mark.asyncio
    async def test_claude_search_timeout(self):
        """Claude search handles timeout gracefully."""
        from aragora.server.research_phase import (
            CLAUDE_SEARCH_TIMEOUT,
            PreDebateResearcher,
        )

        researcher = PreDebateResearcher()

        with patch("asyncio.wait_for", new_callable=AsyncMock) as mock_wait:
            mock_wait.side_effect = asyncio.TimeoutError()

            result = await researcher.search_with_claude("test")

        assert result.summary == ""
        assert result.results == []

    @pytest.mark.asyncio
    async def test_brave_search_timeout(self):
        """Brave search handles timeout gracefully."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher(brave_api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=asyncio.TimeoutError()
            )

            results = await researcher.search_brave("test")

        assert results == []


class TestErrorHandling:
    """Tests for error handling in research operations."""

    @pytest.mark.asyncio
    async def test_handles_api_errors(self):
        """Research methods handle API errors gracefully."""
        from aragora.server.research_phase import PreDebateResearcher

        researcher = PreDebateResearcher(brave_api_key="test", serper_api_key="test")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=RuntimeError("API error")
            )
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=RuntimeError("API error")
            )

            brave_results = await researcher.search_brave("test")
            serper_results = await researcher.search_serper("test")

        assert brave_results == []
        assert serper_results == []

    @pytest.mark.asyncio
    async def test_handles_malformed_responses(self):
        """Research methods handle malformed API responses."""
        from aragora.server.research_phase import PreDebateResearcher

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {}  # Missing expected keys

        researcher = PreDebateResearcher(brave_api_key="test")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            results = await researcher.search_brave("test")

        assert results == []


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestConcurrency:
    """Tests for concurrent research operations."""

    @pytest.mark.asyncio
    async def test_concurrent_searches(self):
        """Multiple concurrent searches work correctly."""
        from aragora.server.research_phase import PreDebateResearcher

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [{"title": "Result", "url": "https://test.com", "description": "Desc"}]
            }
        }

        researcher = PreDebateResearcher(brave_api_key="test")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            # Run multiple searches concurrently
            tasks = [researcher.search_brave(f"query {i}") for i in range(5)]
            results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(len(r) == 1 for r in results)

    @pytest.mark.asyncio
    async def test_thread_pool_usage(self):
        """Claude API calls use thread pool correctly."""
        from aragora.server.research_phase import PreDebateResearcher

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Summary")]

        researcher = PreDebateResearcher()

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_response

            with patch("asyncio.wait_for", new_callable=AsyncMock) as mock_wait:
                mock_wait.return_value = mock_response

                await researcher.search_with_claude("test")

        # Verify wait_for was called (which wraps to_thread)
        mock_wait.assert_called_once()


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_timeout_values(self):
        """Timeout constants have reasonable values."""
        from aragora.server.research_phase import (
            CLAUDE_SEARCH_TIMEOUT,
            DEFAULT_TIMEOUT,
            SUMMARIZATION_TIMEOUT,
        )

        assert DEFAULT_TIMEOUT >= 30.0
        assert SUMMARIZATION_TIMEOUT >= 60.0
        assert CLAUDE_SEARCH_TIMEOUT >= 120.0
        assert CLAUDE_SEARCH_TIMEOUT > DEFAULT_TIMEOUT
        assert SUMMARIZATION_TIMEOUT >= DEFAULT_TIMEOUT

    def test_research_model(self):
        """Research model constant is set correctly."""
        from aragora.server.research_phase import RESEARCH_MODEL

        assert "claude" in RESEARCH_MODEL.lower() or "opus" in RESEARCH_MODEL.lower()

    def test_current_event_indicators(self):
        """Current event indicators list is populated."""
        from aragora.server.research_phase import PreDebateResearcher

        assert len(PreDebateResearcher.CURRENT_EVENT_INDICATORS) > 10
        assert "today" in PreDebateResearcher.CURRENT_EVENT_INDICATORS
        assert "2025" in PreDebateResearcher.CURRENT_EVENT_INDICATORS
