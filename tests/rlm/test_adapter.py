"""
Tests for RLMContextAdapter.

Tests the external environment pattern for context access:
- Content registration and retrieval
- Summary generation
- Drill-down to sections
- Query-based access
- Smart truncation fallback
- Circuit breaker and timeout handling (robustness)
"""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.rlm.adapter import (
    RLMContextAdapter,
    RegisteredContent,
    get_adapter,
)
from aragora.rlm.exceptions import (
    RLMCircuitOpenError,
    RLMContentNotFoundError,
    RLMError,
    RLMProviderError,
    RLMTimeoutError,
)
from aragora.rlm.types import RLMResult


class TestRLMContextAdapter:
    """Tests for RLMContextAdapter."""

    def test_init_default(self):
        """Test default initialization."""
        adapter = RLMContextAdapter()
        assert adapter._registry == {}
        assert adapter._compressor is None
        assert adapter._agent_call is None

    def test_init_with_params(self):
        """Test initialization with parameters."""
        mock_compressor = MagicMock()
        mock_agent_call = AsyncMock()
        adapter = RLMContextAdapter(
            compressor=mock_compressor,
            agent_call=mock_agent_call,
        )
        assert adapter._compressor == mock_compressor
        assert adapter._agent_call == mock_agent_call


class TestContentRegistration:
    """Tests for content registration."""

    def test_register_content_basic(self):
        """Test basic content registration."""
        adapter = RLMContextAdapter()
        content_id = adapter.register_content(
            content_id="test_001",
            content="This is test content for the adapter.",
            content_type="evidence",
        )
        assert content_id == "test_001"
        assert "test_001" in adapter._registry

    def test_register_content_auto_id(self):
        """Test auto-generated content ID."""
        adapter = RLMContextAdapter()
        content_id = adapter.register_content(
            content_id="",
            content="Auto-generate an ID for this content.",
        )
        assert content_id  # Should have generated an ID
        assert len(content_id) == 12  # MD5 hash prefix

    def test_register_content_auto_summary(self):
        """Test auto-generated summary."""
        adapter = RLMContextAdapter()
        long_content = "First sentence here. Second sentence. " * 20
        adapter.register_content(
            content_id="test",
            content=long_content,
        )
        registered = adapter._registry["test"]
        assert registered.summary
        assert len(registered.summary) < len(long_content)

    def test_register_content_custom_summary(self):
        """Test custom summary."""
        adapter = RLMContextAdapter()
        adapter.register_content(
            content_id="test",
            content="Long content here...",
            summary="Custom summary",
        )
        assert adapter._registry["test"].summary == "Custom summary"

    def test_register_content_sections(self):
        """Test section extraction."""
        adapter = RLMContextAdapter()
        content = """# Introduction
This is the intro.

## Methods
These are the methods.

## Results
These are the results."""

        adapter.register_content(
            content_id="test",
            content=content,
        )
        sections = adapter._registry["test"].sections
        assert "introduction" in sections or "intro" in sections
        assert "methods" in sections
        assert "results" in sections

    def test_list_registered(self):
        """Test listing registered content."""
        adapter = RLMContextAdapter()
        adapter.register_content("a", "Content A")
        adapter.register_content("b", "Content B")
        assert set(adapter.list_registered()) == {"a", "b"}

    def test_unregister(self):
        """Test unregistering content."""
        adapter = RLMContextAdapter()
        adapter.register_content("test", "Content")
        assert adapter.unregister("test")
        assert "test" not in adapter._registry
        assert not adapter.unregister("nonexistent")


class TestGetSummary:
    """Tests for get_summary method."""

    def test_get_summary_basic(self):
        """Test basic summary retrieval."""
        adapter = RLMContextAdapter()
        adapter.register_content(
            content_id="test",
            content="Short content",
            summary="Test summary",
        )
        summary = adapter.get_summary("test")
        assert summary == "Test summary"

    def test_get_summary_with_max_chars(self):
        """Test summary with character limit."""
        adapter = RLMContextAdapter()
        adapter.register_content(
            content_id="test",
            content="x",
            summary="This is a longer summary that needs truncation.",
        )
        summary = adapter.get_summary("test", max_chars=20)
        assert len(summary) <= 23  # 20 + "..."

    def test_get_summary_not_found(self):
        """Test summary for non-existent content."""
        adapter = RLMContextAdapter()
        summary = adapter.get_summary("nonexistent")
        assert summary == ""


class TestGetFullContent:
    """Tests for get_full_content method."""

    def test_get_full_content(self):
        """Test full content retrieval."""
        adapter = RLMContextAdapter()
        original = "This is the full content."
        adapter.register_content("test", original)
        assert adapter.get_full_content("test") == original

    def test_get_full_content_not_found(self):
        """Test full content for non-existent."""
        adapter = RLMContextAdapter()
        assert adapter.get_full_content("nonexistent") == ""


class TestDrillDown:
    """Tests for drill_down method."""

    def test_drill_down_section(self):
        """Test drill-down to specific section."""
        adapter = RLMContextAdapter()
        adapter._registry["test"] = RegisteredContent(
            id="test",
            full_content="Full content here",
            content_type="text",
            sections={"intro": "Introduction text", "conclusion": "Conclusion text"},
        )
        result = adapter.drill_down("test", section="intro")
        assert result == "Introduction text"

    def test_drill_down_query(self):
        """Test drill-down with query."""
        adapter = RLMContextAdapter()
        adapter.register_content(
            content_id="test",
            content="""Line one about apples.
Line two about oranges.
Line three about bananas.
Line four about apples again.""",
        )
        result = adapter.drill_down("test", query="apples")
        assert "apples" in result.lower()

    def test_drill_down_max_chars(self):
        """Test drill-down with max chars."""
        adapter = RLMContextAdapter()
        adapter.register_content("test", "x" * 1000)
        result = adapter.drill_down("test", max_chars=50)
        assert len(result) <= 53  # 50 + "..."

    def test_drill_down_not_found(self):
        """Test drill-down for non-existent."""
        adapter = RLMContextAdapter()
        assert adapter.drill_down("nonexistent") == ""


class TestQuery:
    """Tests for async query method."""

    @pytest.mark.asyncio
    async def test_query_no_llm(self):
        """Test query without LLM (uses search)."""
        adapter = RLMContextAdapter()
        adapter.register_content(
            content_id="test",
            content="The sky is blue. The grass is green. Water is wet.",
        )
        result = await adapter.query("test", "What color is the sky?")
        assert isinstance(result, RLMResult)
        assert "blue" in result.answer.lower() or result.confidence > 0

    @pytest.mark.asyncio
    async def test_query_not_found(self):
        """Test query for non-existent content raises RLMContentNotFoundError."""
        adapter = RLMContextAdapter(enable_circuit_breaker=False)
        with pytest.raises(RLMContentNotFoundError):
            await adapter.query("nonexistent", "Question?")

    @pytest.mark.asyncio
    async def test_query_with_llm(self):
        """Test query with mock LLM."""
        mock_agent = AsyncMock(return_value="The sky is blue.")
        adapter = RLMContextAdapter(agent_call=mock_agent)
        adapter.register_content("test", "The sky is blue. The grass is green.")

        result = await adapter.query("test", "What color is the sky?")
        assert result.answer == "The sky is blue."
        assert result.confidence == 0.8
        mock_agent.assert_called_once()


class TestGenerateSummaryAsync:
    """Tests for TRUE RLM: LLM-first summary generation."""

    @pytest.mark.asyncio
    async def test_llm_first_summary_generation(self):
        """Test that LLM is used FIRST for summary generation (TRUE RLM)."""
        mock_agent = AsyncMock(return_value="LLM-generated summary of the content.")
        adapter = RLMContextAdapter(agent_call=mock_agent)
        long_content = "This is a long document. " * 50

        adapter.register_content("test", long_content)
        summary = await adapter.generate_summary_async("test")

        # LLM should be called for long content
        mock_agent.assert_called_once()
        assert summary == "LLM-generated summary of the content."

    @pytest.mark.asyncio
    async def test_llm_summary_cached(self):
        """Test that LLM summary is cached in registered content."""
        mock_agent = AsyncMock(return_value="Cached LLM summary.")
        adapter = RLMContextAdapter(agent_call=mock_agent)
        long_content = "Long document content. " * 50

        adapter.register_content("test", long_content)
        await adapter.generate_summary_async("test")

        # Summary should be cached
        assert adapter._registry["test"].summary == "Cached LLM summary."

    @pytest.mark.asyncio
    async def test_heuristic_fallback_when_no_llm(self):
        """Test fallback to heuristic when no LLM available."""
        adapter = RLMContextAdapter()  # No agent_call
        long_content = "First sentence here. Second sentence. Third sentence. " * 10

        adapter.register_content("test", long_content)
        summary = await adapter.generate_summary_async("test")

        # Should use heuristic fallback
        assert summary  # Should have some summary
        assert len(summary) < len(long_content)

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_compression(self):
        """Test fallback to compression when LLM fails (priority 2)."""
        mock_agent = AsyncMock(side_effect=Exception("LLM error"))
        mock_compressor = AsyncMock()
        mock_result = MagicMock()
        mock_result.context.get_at_level.return_value = "Compressed summary"
        mock_compressor.compress.return_value = mock_result

        adapter = RLMContextAdapter(agent_call=mock_agent, compressor=mock_compressor)
        long_content = "First sentence. Second sentence. Third sentence. " * 10

        adapter.register_content("test", long_content)
        summary = await adapter.generate_summary_async("test")

        # Should try LLM first, then fall back to compression
        mock_agent.assert_called_once()
        mock_compressor.compress.assert_called_once()
        assert summary == "Compressed summary"

    @pytest.mark.asyncio
    async def test_compression_failure_falls_back_to_truncation(self):
        """Test fallback to truncation when both LLM and compression fail."""
        mock_agent = AsyncMock(side_effect=Exception("LLM error"))
        mock_compressor = AsyncMock(side_effect=Exception("Compress error"))

        adapter = RLMContextAdapter(agent_call=mock_agent, compressor=mock_compressor)
        long_content = "First sentence. Second sentence. Third sentence. " * 10

        adapter.register_content("test", long_content)
        summary = await adapter.generate_summary_async("test")

        # Should try LLM, then compression, then fall back to truncation
        mock_agent.assert_called_once()
        mock_compressor.compress.assert_called_once()
        assert summary  # Should still have truncated summary

    @pytest.mark.asyncio
    async def test_short_content_no_llm_needed(self):
        """Test that short content doesn't need LLM."""
        mock_agent = AsyncMock(return_value="Should not be called")
        adapter = RLMContextAdapter(agent_call=mock_agent)

        adapter.register_content("test", "Short content", summary="Short content")
        summary = await adapter.generate_summary_async("test")

        # LLM should NOT be called for short content
        mock_agent.assert_not_called()
        assert summary == "Short content"

    @pytest.mark.asyncio
    async def test_summary_with_max_chars(self):
        """Test summary respects max_chars limit."""
        mock_agent = AsyncMock(return_value="A very long LLM summary that exceeds the limit.")
        adapter = RLMContextAdapter(agent_call=mock_agent)
        long_content = "Long content. " * 50

        adapter.register_content("test", long_content)
        summary = await adapter.generate_summary_async("test", max_chars=20)

        assert len(summary) <= 23  # 20 + "..."

    @pytest.mark.asyncio
    async def test_not_found_returns_empty(self):
        """Test summary for non-existent content."""
        adapter = RLMContextAdapter()
        summary = await adapter.generate_summary_async("nonexistent")
        assert summary == ""


class TestFormatForPromptAsync:
    """Tests for TRUE RLM: async format with LLM summarization."""

    @pytest.mark.asyncio
    async def test_llm_first_formatting(self):
        """Test that async format uses LLM for summary (TRUE RLM)."""
        mock_agent = AsyncMock(return_value="LLM summary for prompt.")
        adapter = RLMContextAdapter(agent_call=mock_agent)
        long_content = "Long document content. " * 50

        result = await adapter.format_for_prompt_async(long_content, max_chars=200)

        # LLM should be used for summary
        mock_agent.assert_called_once()
        assert "LLM summary for prompt" in result

    @pytest.mark.asyncio
    async def test_includes_drill_down_hint(self):
        """Test that hint is included for long content."""
        mock_agent = AsyncMock(return_value="Summary.")
        adapter = RLMContextAdapter(agent_call=mock_agent)
        long_content = "x" * 500

        result = await adapter.format_for_prompt_async(
            long_content, max_chars=100, include_hint=True
        )

        assert "[Full" in result or "available" in result

    @pytest.mark.asyncio
    async def test_registers_content(self):
        """Test that content is registered for later access."""
        mock_agent = AsyncMock(return_value="Summary.")
        adapter = RLMContextAdapter(agent_call=mock_agent)

        await adapter.format_for_prompt_async("Content to register", max_chars=200)

        assert len(adapter.list_registered()) == 1

    @pytest.mark.asyncio
    async def test_empty_content_returns_empty(self):
        """Test empty content handling."""
        adapter = RLMContextAdapter()
        result = await adapter.format_for_prompt_async("", max_chars=100)
        assert result == ""

    @pytest.mark.asyncio
    async def test_fallback_when_no_llm(self):
        """Test heuristic fallback in async format."""
        adapter = RLMContextAdapter()  # No agent_call
        content = "First sentence. Second sentence. Third sentence. " * 20

        result = await adapter.format_for_prompt_async(content, max_chars=100)

        # Should use heuristic fallback
        assert result
        assert len(result) <= 100 + 50  # max_chars + hint allowance


class TestSmartTruncate:
    """Tests for smart_truncate method."""

    def test_short_content_unchanged(self):
        """Test that short content passes through unchanged."""
        adapter = RLMContextAdapter()
        content = "Short text."
        assert adapter.smart_truncate(content, max_chars=100) == content

    def test_truncate_at_sentence(self):
        """Test truncation at sentence boundary."""
        adapter = RLMContextAdapter()
        content = "First sentence. Second sentence. Third sentence."
        result = adapter.smart_truncate(content, max_chars=30)
        # Should end at a sentence boundary
        assert result.endswith(".") or result.endswith("...")

    def test_truncate_at_word(self):
        """Test truncation at word boundary."""
        adapter = RLMContextAdapter()
        content = "word " * 100
        result = adapter.smart_truncate(content, max_chars=30)
        # Should not cut mid-word
        assert result.endswith("...") or result.endswith(" ")

    def test_truncate_empty(self):
        """Test truncation of empty content."""
        adapter = RLMContextAdapter()
        assert adapter.smart_truncate("", max_chars=100) == ""


class TestFormatForPrompt:
    """Tests for format_for_prompt method."""

    def test_format_short_content(self):
        """Test formatting short content."""
        adapter = RLMContextAdapter()
        content = "Short content."
        result = adapter.format_for_prompt(content, max_chars=100)
        assert "Short content" in result

    def test_format_with_hint(self):
        """Test formatting with drill-down hint."""
        adapter = RLMContextAdapter()
        content = "x" * 500
        result = adapter.format_for_prompt(content, max_chars=100, include_hint=True)
        assert "[Full" in result or "available" in result

    def test_format_registers_content(self):
        """Test that format_for_prompt registers the content."""
        adapter = RLMContextAdapter()
        content = "Some content to register."
        adapter.format_for_prompt(content, max_chars=200)
        assert len(adapter.list_registered()) == 1


class TestSectionExtraction:
    """Tests for section extraction."""

    def test_extract_evidence_conclusion(self):
        """Test extraction of conclusion from evidence."""
        adapter = RLMContextAdapter()
        content = """Study results show...
Conclusion: The treatment was effective.
More details here."""

        adapter.register_content("test", content, content_type="evidence")
        sections = adapter._registry["test"].sections
        assert "conclusion" in sections
        assert "effective" in sections["conclusion"].lower()

    def test_extract_dissent_core(self):
        """Test extraction of core disagreement from dissent."""
        adapter = RLMContextAdapter()
        content = """Some context.
However, this approach has problems.
More issues here."""

        adapter.register_content("test", content, content_type="dissent")
        sections = adapter._registry["test"].sections
        assert "core" in sections or "intro" in sections


class TestGlobalAdapter:
    """Tests for global adapter instance."""

    def test_get_adapter_singleton(self):
        """Test that get_adapter returns singleton."""
        adapter1 = get_adapter()
        adapter2 = get_adapter()
        assert adapter1 is adapter2

    def test_get_adapter_type(self):
        """Test get_adapter return type."""
        adapter = get_adapter()
        assert isinstance(adapter, RLMContextAdapter)


class TestSearchContent:
    """Tests for internal search functionality."""

    def test_search_finds_relevant_lines(self):
        """Test search finds relevant content."""
        adapter = RLMContextAdapter()
        content = """Line about cats.
Line about dogs.
Line about birds.
Another line about dogs."""

        result = adapter._search_content(content, "dogs", max_chars=500)
        assert "dogs" in result.lower()

    def test_search_no_matches(self):
        """Test search with no matches returns beginning."""
        adapter = RLMContextAdapter()
        content = "Line one.\nLine two.\nLine three."
        result = adapter._search_content(content, "xyz", max_chars=100)
        assert result.startswith("Line")


class TestHeuristicFallback:
    """Tests for heuristic methods (FALLBACK only, not TRUE RLM)."""

    def test_heuristic_summary_is_last_resort(self):
        """Verify _heuristic_summary is documented as LAST RESORT."""
        adapter = RLMContextAdapter()
        # The method should exist and be clearly marked as last resort
        assert hasattr(adapter, "_heuristic_summary")
        docstring = adapter._heuristic_summary.__doc__
        assert "LAST RESORT" in docstring

    def test_extract_summary_is_alias(self):
        """Test _extract_summary is alias for backwards compat."""
        adapter = RLMContextAdapter()
        content = "First sentence. Second sentence."
        heuristic = adapter._heuristic_summary(content, "text")
        extract = adapter._extract_summary(content, "text")
        assert heuristic == extract

    def test_sync_get_summary_uses_heuristic(self):
        """Test sync get_summary uses heuristic (not LLM)."""
        mock_agent = AsyncMock(return_value="LLM response")
        adapter = RLMContextAdapter(agent_call=mock_agent)
        long_content = "Long content. " * 50

        adapter.register_content("test", long_content)
        summary = adapter.get_summary("test")  # Sync call

        # LLM should NOT be called for sync method
        mock_agent.assert_not_called()
        assert summary  # Should still have heuristic summary

    def test_sync_format_uses_heuristic(self):
        """Test sync format_for_prompt uses heuristic (not LLM)."""
        mock_agent = AsyncMock(return_value="LLM response")
        adapter = RLMContextAdapter(agent_call=mock_agent)

        result = adapter.format_for_prompt("Long content. " * 50, max_chars=100)

        # LLM should NOT be called for sync method
        mock_agent.assert_not_called()
        assert result  # Should still have result


class TestRegisteredContentDataclass:
    """Tests for RegisteredContent dataclass."""

    def test_registered_content_creation(self):
        """Test creating RegisteredContent."""
        rc = RegisteredContent(
            id="test",
            full_content="Content",
            content_type="text",
        )
        assert rc.id == "test"
        assert rc.full_content == "Content"
        assert rc.summary == ""
        assert rc.sections == {}
        assert rc.metadata == {}

    def test_registered_content_with_all_fields(self):
        """Test RegisteredContent with all fields."""
        rc = RegisteredContent(
            id="test",
            full_content="Content",
            content_type="evidence",
            summary="Summary",
            sections={"intro": "Intro text"},
            metadata={"source": "test"},
        )
        assert rc.summary == "Summary"
        assert rc.sections["intro"] == "Intro text"
        assert rc.metadata["source"] == "test"


class TestRobustnessInitialization:
    """Tests for robustness initialization options."""

    def test_init_with_timeout(self):
        """Test initialization with custom timeout."""
        adapter = RLMContextAdapter(timeout_seconds=60.0)
        assert adapter._timeout_seconds == 60.0

    def test_init_default_timeout(self):
        """Test default timeout is set."""
        adapter = RLMContextAdapter()
        assert adapter._timeout_seconds == 30.0  # DEFAULT_TIMEOUT_SECONDS

    def test_init_with_circuit_breaker_enabled(self):
        """Test circuit breaker is created by default."""
        adapter = RLMContextAdapter(enable_circuit_breaker=True)
        assert adapter._circuit_breaker is not None

    def test_init_with_circuit_breaker_disabled(self):
        """Test circuit breaker can be disabled."""
        adapter = RLMContextAdapter(enable_circuit_breaker=False)
        assert adapter._circuit_breaker is None

    def test_init_with_custom_circuit_breaker_config(self):
        """Test custom circuit breaker configuration."""
        adapter = RLMContextAdapter(
            enable_circuit_breaker=True,
            failure_threshold=10,
            cooldown_seconds=120.0,
        )
        assert adapter._circuit_breaker is not None
        assert adapter._circuit_breaker.failure_threshold == 10
        assert adapter._circuit_breaker.cooldown_seconds == 120.0


class TestExceptions:
    """Tests for RLM exceptions."""

    def test_rlm_error_basic(self):
        """Test basic RLMError creation."""
        err = RLMError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.message == "Something went wrong"
        assert err.operation is None
        assert err.content_id is None

    def test_rlm_error_with_context(self):
        """Test RLMError with operation and content_id."""
        err = RLMError("Failed", operation="query", content_id="test123")
        assert "[query]" in str(err)
        assert "test123" in str(err)

    def test_rlm_timeout_error(self):
        """Test RLMTimeoutError creation."""
        err = RLMTimeoutError("Timed out", timeout_seconds=30.0, operation="query")
        assert err.timeout_seconds == 30.0
        assert "30.0s" in str(err)

    def test_rlm_circuit_open_error(self):
        """Test RLMCircuitOpenError creation."""
        err = RLMCircuitOpenError("Circuit open", cooldown_remaining=45.0, operation="query")
        assert err.cooldown_remaining == 45.0
        assert "45.0s" in str(err)

    def test_rlm_content_not_found_error(self):
        """Test RLMContentNotFoundError creation."""
        err = RLMContentNotFoundError("Not found", content_id="missing")
        assert err.content_id == "missing"

    def test_rlm_provider_error(self):
        """Test RLMProviderError creation."""
        err = RLMProviderError(
            "API error",
            provider="openai",
            status_code=429,
            is_transient=True,
        )
        assert err.provider == "openai"
        assert err.status_code == 429
        assert err.is_transient


class TestQueryRobustness:
    """Tests for query method robustness features."""

    @pytest.mark.asyncio
    async def test_query_content_not_found_raises(self):
        """Test that query raises RLMContentNotFoundError for missing content."""
        adapter = RLMContextAdapter(enable_circuit_breaker=False)
        with pytest.raises(RLMContentNotFoundError) as exc_info:
            await adapter.query("nonexistent", "Question?")
        assert exc_info.value.content_id == "nonexistent"

    @pytest.mark.asyncio
    async def test_query_circuit_breaker_open_raises(self):
        """Test that query raises when circuit breaker is open."""
        adapter = RLMContextAdapter(
            agent_call=AsyncMock(),
            enable_circuit_breaker=True,
            failure_threshold=1,
        )
        adapter.register_content("test", "Content")

        # Force circuit open
        adapter._circuit_breaker.is_open = True

        with pytest.raises(RLMCircuitOpenError):
            await adapter.query("test", "Question?")

    @pytest.mark.asyncio
    async def test_query_timeout_raises(self):
        """Test that query raises RLMTimeoutError on timeout."""

        async def slow_agent(*args):
            await asyncio.sleep(10)
            return "Never returned"

        adapter = RLMContextAdapter(
            agent_call=slow_agent,
            timeout_seconds=0.01,  # Very short timeout
            enable_circuit_breaker=False,
        )
        adapter.register_content("test", "Content")

        with pytest.raises(RLMTimeoutError) as exc_info:
            await adapter.query("test", "Question?")
        assert exc_info.value.operation == "query"

    @pytest.mark.asyncio
    async def test_query_connection_error_raises(self):
        """Test that query raises RLMProviderError on connection error."""

        async def failing_agent(*args):
            raise ConnectionError("Network unreachable")

        adapter = RLMContextAdapter(
            agent_call=failing_agent,
            enable_circuit_breaker=False,
        )
        adapter.register_content("test", "Content")

        with pytest.raises(RLMProviderError) as exc_info:
            await adapter.query("test", "Question?")
        assert exc_info.value.is_transient

    @pytest.mark.asyncio
    async def test_query_records_success(self):
        """Test that query records success with circuit breaker."""
        adapter = RLMContextAdapter(
            agent_call=AsyncMock(return_value="Answer"),
            enable_circuit_breaker=True,
        )
        adapter.register_content("test", "Content")

        # Trigger some failures first
        adapter._circuit_breaker._single_failures = 2

        await adapter.query("test", "Question?")

        # Success should reset failures
        assert adapter._circuit_breaker._single_failures == 0

    @pytest.mark.asyncio
    async def test_query_records_failure(self):
        """Test that query records failure with circuit breaker on timeout."""

        async def slow_agent(*args):
            await asyncio.sleep(10)
            return "Never returned"

        adapter = RLMContextAdapter(
            agent_call=slow_agent,
            timeout_seconds=0.01,
            enable_circuit_breaker=True,
        )
        adapter.register_content("test", "Content")

        try:
            await adapter.query("test", "Question?")
        except RLMTimeoutError:
            pass

        # Failure should be recorded
        assert adapter._circuit_breaker._single_failures == 1

    @pytest.mark.asyncio
    async def test_query_custom_timeout_override(self):
        """Test that custom timeout can be passed to query."""

        async def medium_agent(*args):
            await asyncio.sleep(0.05)
            return "Answer"

        adapter = RLMContextAdapter(
            agent_call=medium_agent,
            timeout_seconds=0.01,  # Default would timeout
            enable_circuit_breaker=False,
        )
        adapter.register_content("test", "Content")

        # Should succeed with longer timeout override
        result = await adapter.query("test", "Question?", timeout_seconds=1.0)
        assert result.answer == "Answer"


class TestGenerateSummaryRobustness:
    """Tests for generate_summary_async robustness features."""

    @pytest.mark.asyncio
    async def test_summary_timeout_falls_back_to_compression(self):
        """Test that timeout falls back to compression."""

        async def slow_agent(*args):
            await asyncio.sleep(10)
            return "Never returned"

        mock_compressor = AsyncMock()
        mock_result = MagicMock()
        mock_result.context.get_at_level.return_value = "Compressed summary"
        mock_compressor.compress.return_value = mock_result

        adapter = RLMContextAdapter(
            agent_call=slow_agent,
            compressor=mock_compressor,
            timeout_seconds=0.01,
            enable_circuit_breaker=False,
        )
        long_content = "Long content. " * 50
        adapter.register_content("test", long_content)

        summary = await adapter.generate_summary_async("test")

        # Should fall back to compression
        mock_compressor.compress.assert_called_once()
        assert summary == "Compressed summary"

    @pytest.mark.asyncio
    async def test_summary_circuit_open_skips_to_fallback(self):
        """Test that open circuit skips LLM and uses fallback."""
        mock_agent = AsyncMock(return_value="LLM summary")
        adapter = RLMContextAdapter(
            agent_call=mock_agent,
            enable_circuit_breaker=True,
        )
        long_content = "Long content. " * 50
        adapter.register_content("test", long_content)

        # Force circuit open
        adapter._circuit_breaker.is_open = True

        summary = await adapter.generate_summary_async("test")

        # LLM should NOT be called when circuit is open
        mock_agent.assert_not_called()
        # Should still have a summary (from heuristic)
        assert summary

    @pytest.mark.asyncio
    async def test_summary_records_success_on_circuit_breaker(self):
        """Test that successful summary records success."""
        mock_agent = AsyncMock(return_value="LLM summary")
        adapter = RLMContextAdapter(
            agent_call=mock_agent,
            enable_circuit_breaker=True,
        )
        long_content = "Long content. " * 50
        adapter.register_content("test", long_content)

        # Add some failures
        adapter._circuit_breaker._single_failures = 2

        await adapter.generate_summary_async("test")

        # Success should reset failures
        assert adapter._circuit_breaker._single_failures == 0

    @pytest.mark.asyncio
    async def test_summary_records_failure_on_timeout(self):
        """Test that timeout records failure with circuit breaker."""

        async def slow_agent(*args):
            await asyncio.sleep(10)
            return "Never returned"

        adapter = RLMContextAdapter(
            agent_call=slow_agent,
            timeout_seconds=0.01,
            enable_circuit_breaker=True,
        )
        long_content = "Long content. " * 50
        adapter.register_content("test", long_content)

        # Should not raise - falls back to heuristic
        await adapter.generate_summary_async("test")

        # But failure should be recorded
        assert adapter._circuit_breaker._single_failures == 1

    @pytest.mark.asyncio
    async def test_summary_custom_timeout(self):
        """Test custom timeout for generate_summary_async."""

        async def medium_agent(*args):
            await asyncio.sleep(0.05)
            return "LLM summary"

        adapter = RLMContextAdapter(
            agent_call=medium_agent,
            timeout_seconds=0.01,  # Would timeout with default
            enable_circuit_breaker=False,
        )
        long_content = "Long content. " * 50
        adapter.register_content("test", long_content)

        # Should succeed with longer timeout
        summary = await adapter.generate_summary_async("test", timeout_seconds=1.0)
        assert summary == "LLM summary"


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold_failures(self):
        """Test that circuit opens after threshold failures."""
        call_count = 0

        async def failing_agent(*args):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        adapter = RLMContextAdapter(
            agent_call=failing_agent,
            enable_circuit_breaker=True,
            failure_threshold=3,
        )
        adapter.register_content("test", "Content")

        # Make failures up to threshold
        for _ in range(3):
            try:
                await adapter.query("test", "Question?")
            except RLMProviderError:
                pass

        # Circuit should now be open
        with pytest.raises(RLMCircuitOpenError):
            await adapter.query("test", "Question?")

        # Verify agent was only called 3 times (not on the 4th due to open circuit)
        assert call_count == 3

    def test_circuit_breaker_name(self):
        """Test circuit breaker has correct name."""
        adapter = RLMContextAdapter(enable_circuit_breaker=True)
        assert adapter._circuit_breaker.name == "rlm_adapter"
