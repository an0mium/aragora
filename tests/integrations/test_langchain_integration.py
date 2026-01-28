"""
Tests for LangChain integration.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.integrations.langchain import (
    AragoraTool,
    AragoraDebateTool,
    AragoraRetriever,
    AragoraCallbackHandler,
    is_langchain_available,
    LANGCHAIN_AVAILABLE,
)
from aragora.integrations.langchain.tools import (
    AragoraToolInput,
    get_langchain_version,
)


# =============================================================================
# Tool Tests
# =============================================================================


class TestAragoraTool:
    """Tests for AragoraTool."""

    def test_tool_initialization(self):
        """Test tool initialization with default values."""
        tool = AragoraTool()

        assert tool.name == "aragora_debate"
        assert tool.aragora_url == "http://localhost:8080"
        assert len(tool.default_agents) == 3

    def test_tool_initialization_custom(self):
        """Test tool initialization with custom values."""
        tool = AragoraTool(
            aragora_url="https://custom.api.com",
            api_token="test-key",
            timeout_seconds=60.0,
        )

        assert tool.aragora_url == "https://custom.api.com"
        assert tool.api_token == "test-key"
        assert tool.timeout_seconds == 60.0

    def test_tool_has_description(self):
        """Test that tool has a description."""
        tool = AragoraTool()
        assert len(tool.description) > 50
        assert "multi-agent" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_arun_connection_error(self):
        """Test async run handles connection errors gracefully."""
        tool = AragoraTool(aragora_url="https://invalid.nonexistent.domain.test")

        # This will fail to connect but should return a valid error JSON or message
        result = await tool._arun("Test question")

        # Should contain some kind of error indication
        assert "error" in result.lower() or "failed" in result.lower()

    def test_run_calls_arun(self):
        """Test that synchronous run calls async run."""
        tool = AragoraTool(aragora_url="https://api.test.com", api_token="test-key")

        # Mock the _arun method
        expected_result = '{"answer": "test", "confidence": 0.9}'
        with patch.object(tool, "_arun", new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = expected_result

            result = tool._run("Test question")

            assert result == expected_result
            mock_arun.assert_called_once()


# =============================================================================
# Retriever Tests
# =============================================================================


class TestAragoraRetriever:
    """Tests for AragoraRetriever."""

    def test_retriever_initialization(self):
        """Test retriever initialization with default values."""
        retriever = AragoraRetriever()

        assert retriever.aragora_url == "http://localhost:8080"
        assert retriever.max_results == 5

    def test_retriever_initialization_custom(self):
        """Test retriever initialization with custom values."""
        retriever = AragoraRetriever(
            aragora_url="https://custom.api.com",
            api_token="test-key",
            max_results=10,
            include_metadata=False,
        )

        assert retriever.aragora_url == "https://custom.api.com"
        assert retriever.api_token == "test-key"
        assert retriever.max_results == 10
        assert retriever.include_metadata is False

    @pytest.mark.asyncio
    async def test_aget_relevant_documents_connection_error(self):
        """Test async document retrieval handles connection errors gracefully."""
        retriever = AragoraRetriever(aragora_url="https://invalid.nonexistent.domain.test")

        # Should return empty list on connection error
        docs = await retriever._aget_relevant_documents("test query")
        assert len(docs) == 0


# =============================================================================
# Callback Handler Tests
# =============================================================================


class TestAragoraCallbackHandler:
    """Tests for AragoraCallbackHandler."""

    def test_handler_initialization(self):
        """Test callback handler initialization."""
        handler = AragoraCallbackHandler()
        assert handler.aragora_url == "http://localhost:8080"

    def test_handler_initialization_with_debate_id(self):
        """Test callback handler initialization with debate_id."""
        handler = AragoraCallbackHandler(
            aragora_url="https://api.test.com",
            debate_id="debate-123",
            session_id="session-456",
        )

        assert handler.debate_id == "debate-123"
        assert handler.session_id == "session-456"

    def test_get_events(self):
        """Test get_events returns recorded events."""
        handler = AragoraCallbackHandler()

        # Handler should have get_events method
        events = handler.get_events()
        assert isinstance(events, list)

    def test_clear_events(self):
        """Test clear_events clears recorded events."""
        handler = AragoraCallbackHandler()

        # Handler should have clear_events method
        handler.clear_events()
        events = handler.get_events()
        assert len(events) == 0


# =============================================================================
# Tool Input Tests
# =============================================================================


class TestAragoraToolInput:
    """Tests for AragoraToolInput schema."""

    def test_tool_input_defaults(self):
        """Test tool input with default values."""
        input_obj = AragoraToolInput(question="Test question")
        assert input_obj.question == "Test question"
        assert input_obj.rounds == 3
        assert input_obj.consensus_threshold == 0.8

    def test_tool_input_custom(self):
        """Test tool input with custom values."""
        input_obj = AragoraToolInput(
            question="Test question",
            agents=["claude", "gpt"],
            rounds=5,
            consensus_threshold=0.9,
            include_evidence=False,
        )

        assert input_obj.question == "Test question"
        assert input_obj.agents == ["claude", "gpt"]
        assert input_obj.rounds == 5
        assert input_obj.consensus_threshold == 0.9
        assert input_obj.include_evidence is False


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_is_langchain_available(self):
        """Test LangChain availability check."""
        result = is_langchain_available()
        assert result is True

    def test_get_langchain_version(self):
        """Test getting LangChain version."""
        version = get_langchain_version()
        assert version is not None
        assert isinstance(version, str)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
class TestLangChainIntegration:
    """Integration tests that require LangChain."""

    def test_tool_inherits_from_base_tool(self):
        """Test that AragoraTool inherits from LangChain BaseTool."""
        try:
            from langchain_core.tools import BaseTool
        except ImportError:
            from langchain.tools import BaseTool

        tool = AragoraTool()
        assert isinstance(tool, BaseTool)

    def test_retriever_inherits_from_base_retriever(self):
        """Test that AragoraRetriever inherits from LangChain BaseRetriever."""
        try:
            from langchain_core.retrievers import BaseRetriever
        except ImportError:
            from langchain.schema import BaseRetriever

        retriever = AragoraRetriever()
        assert isinstance(retriever, BaseRetriever)

    def test_callback_handler_inherits_from_base_callback(self):
        """Test that AragoraCallbackHandler inherits from BaseCallbackHandler."""
        try:
            from langchain_core.callbacks.base import BaseCallbackHandler
        except ImportError:
            from langchain.callbacks.base import BaseCallbackHandler

        handler = AragoraCallbackHandler()
        assert isinstance(handler, BaseCallbackHandler)

    def test_tool_has_args_schema(self):
        """Test that tool has proper args_schema."""
        from pydantic import BaseModel

        tool = AragoraTool()
        assert hasattr(tool, "args_schema")
        assert issubclass(tool.args_schema, BaseModel)
