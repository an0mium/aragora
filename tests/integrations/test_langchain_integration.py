"""
Tests for LangChain integration.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.integrations.langchain import (
    AragoraTool,
    AragoraRetriever,
    AragoraCallbackHandler,
    AragoraToolInput,
    is_langchain_available,
    get_langchain_version,
    LANGCHAIN_AVAILABLE,
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
        assert tool.api_base == "https://api.aragora.ai"
        assert tool.default_rounds == 3
        assert len(tool.default_agents) == 3

    def test_tool_initialization_custom(self):
        """Test tool initialization with custom values."""
        tool = AragoraTool(
            api_base="https://custom.api.com",
            api_key="test-key",
            default_agents=["claude", "gpt"],
            default_rounds=5,
            timeout_seconds=60.0,
        )

        assert tool.api_base == "https://custom.api.com"
        assert tool.api_key == "test-key"
        assert tool.default_agents == ["claude", "gpt"]
        assert tool.default_rounds == 5
        assert tool.timeout_seconds == 60.0

    def test_tool_has_description(self):
        """Test that tool has a description."""
        tool = AragoraTool()
        assert len(tool.description) > 50
        assert "multi-agent" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_arun_connection_error(self):
        """Test async run handles connection errors gracefully."""
        tool = AragoraTool(api_base="https://invalid.nonexistent.domain.test")

        # This will fail to connect but should return a valid error JSON
        result = await tool._arun("Test question")

        result_data = json.loads(result)
        assert "error" in result_data
        # Should contain some kind of connection/network error message
        assert "Failed to run debate" in result_data["error"]

    def test_run_calls_arun(self):
        """Test that synchronous run calls async run."""
        tool = AragoraTool(api_base="https://api.test.com", api_key="test-key")

        # Mock the _arun method
        expected_result = '{"answer": "test", "confidence": 0.9}'
        with patch.object(tool, "_arun", new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = expected_result

            result = tool._run("Test question")

            assert result == expected_result
            mock_arun.assert_called_once()

    def test_format_result(self):
        """Test result formatting."""
        tool = AragoraTool()

        result = {
            "final_answer": "Test answer",
            "confidence": 0.9,
            "consensus_reached": True,
            "rounds_used": 2,
            "participants": ["claude", "gpt"],
            "reasoning": ["Point 1", "Point 2", "Point 3", "Point 4"],
        }

        formatted = tool._format_result(result)
        formatted_data = json.loads(formatted)

        assert formatted_data["answer"] == "Test answer"
        assert formatted_data["confidence"] == 0.9
        assert len(formatted_data.get("key_points", [])) <= 3  # Max 3 points


# =============================================================================
# Retriever Tests
# =============================================================================


class TestAragoraRetriever:
    """Tests for AragoraRetriever."""

    def test_retriever_initialization(self):
        """Test retriever initialization with default values."""
        retriever = AragoraRetriever()

        assert retriever.api_base == "https://api.aragora.ai"
        assert retriever.top_k == 5
        assert retriever.min_confidence == 0.0

    def test_retriever_initialization_custom(self):
        """Test retriever initialization with custom values."""
        retriever = AragoraRetriever(
            api_base="https://custom.api.com",
            api_key="test-key",
            top_k=10,
            min_confidence=0.5,
            include_metadata=False,
        )

        assert retriever.api_base == "https://custom.api.com"
        assert retriever.api_key == "test-key"
        assert retriever.top_k == 10
        assert retriever.min_confidence == 0.5
        assert retriever.include_metadata is False

    @pytest.mark.asyncio
    async def test_aget_relevant_documents_connection_error(self):
        """Test async document retrieval handles connection errors gracefully."""
        retriever = AragoraRetriever(api_base="https://invalid.nonexistent.domain.test")

        # Should return empty list on connection error
        docs = await retriever._aget_relevant_documents("test query")
        assert len(docs) == 0

    def test_get_relevant_documents_sync(self):
        """Test synchronous document retrieval interface."""
        retriever = AragoraRetriever(api_base="https://api.test.com")

        # Mock the async method
        mock_docs = [MagicMock(page_content="Test", metadata={})]
        with patch.object(retriever, "_aget_relevant_documents", new_callable=AsyncMock) as mock_aget:
            mock_aget.return_value = mock_docs

            docs = retriever.get_relevant_documents("test query")

            assert len(docs) == 1
            mock_aget.assert_called_once_with("test query")

    def test_convert_to_documents_with_metadata(self):
        """Test document conversion with metadata."""
        retriever = AragoraRetriever(include_metadata=True)

        nodes = [
            {
                "id": "node-1",
                "content": "Test content",
                "node_type": "fact",
                "confidence": 0.9,
                "created_at": "2024-01-01T00:00:00Z",
                "topics": ["topic1", "topic2"],
            }
        ]

        docs = retriever._convert_to_documents(nodes)

        assert len(docs) == 1
        assert docs[0].page_content == "Test content"
        assert docs[0].metadata["node_id"] == "node-1"
        assert docs[0].metadata["node_type"] == "fact"
        assert docs[0].metadata["confidence"] == 0.9
        assert "topics" in docs[0].metadata

    def test_convert_to_documents_without_metadata(self):
        """Test document conversion without metadata."""
        retriever = AragoraRetriever(include_metadata=False)

        nodes = [
            {
                "id": "node-1",
                "content": "Test content",
                "confidence": 0.9,
            }
        ]

        docs = retriever._convert_to_documents(nodes)

        assert len(docs) == 1
        assert docs[0].page_content == "Test content"
        assert len(docs[0].metadata) == 0


# =============================================================================
# Callback Handler Tests
# =============================================================================


class TestAragoraCallbackHandler:
    """Tests for AragoraCallbackHandler."""

    def test_handler_initialization(self):
        """Test callback handler initialization."""
        handler = AragoraCallbackHandler()
        assert handler.verbose is False

    def test_handler_initialization_with_callbacks(self):
        """Test callback handler initialization with callbacks."""
        on_start = MagicMock()
        on_end = MagicMock()
        on_error = MagicMock()

        handler = AragoraCallbackHandler(
            on_debate_start=on_start,
            on_debate_end=on_end,
            on_error=on_error,
            verbose=True,
        )

        assert handler.verbose is True
        assert handler._on_debate_start is on_start
        assert handler._on_debate_end is on_end
        assert handler._on_error is on_error

    def test_on_tool_start(self):
        """Test on_tool_start callback."""
        on_start = MagicMock()
        handler = AragoraCallbackHandler(on_debate_start=on_start)

        handler.on_tool_start(
            serialized={"name": "aragora_debate"},
            input_str="Test question",
        )

        on_start.assert_called_once()
        call_args = on_start.call_args[0][0]
        assert call_args["input"] == "Test question"

    def test_on_tool_start_non_aragora(self):
        """Test on_tool_start is not triggered for non-Aragora tools."""
        on_start = MagicMock()
        handler = AragoraCallbackHandler(on_debate_start=on_start)

        handler.on_tool_start(
            serialized={"name": "other_tool"},
            input_str="Test input",
        )

        on_start.assert_not_called()

    def test_on_tool_end(self):
        """Test on_tool_end callback."""
        on_end = MagicMock()
        handler = AragoraCallbackHandler(on_debate_end=on_end)

        output = json.dumps({"answer": "Test answer", "confidence": 0.9})
        handler.on_tool_end(output=output)

        on_end.assert_called_once()
        call_args = on_end.call_args[0][0]
        assert call_args["answer"] == "Test answer"

    def test_on_tool_end_invalid_json(self):
        """Test on_tool_end with invalid JSON."""
        on_end = MagicMock()
        handler = AragoraCallbackHandler(on_debate_end=on_end)

        handler.on_tool_end(output="Invalid JSON")

        on_end.assert_called_once()
        call_args = on_end.call_args[0][0]
        assert "raw_output" in call_args

    def test_on_tool_error(self):
        """Test on_tool_error callback."""
        on_error = MagicMock()
        handler = AragoraCallbackHandler(on_error=on_error)

        error = Exception("Test error")
        handler.on_tool_error(error=error)

        on_error.assert_called_once_with(error)


# =============================================================================
# Tool Input Tests
# =============================================================================


class TestAragoraToolInput:
    """Tests for AragoraToolInput schema."""

    def test_tool_input_defaults(self):
        """Test tool input with default values."""
        if LANGCHAIN_AVAILABLE:
            # Pydantic model
            input_obj = AragoraToolInput(question="Test question")
            assert input_obj.question == "Test question"
            assert input_obj.rounds == 3
            assert input_obj.consensus_threshold == 0.8
        else:
            # Dataclass fallback
            input_obj = AragoraToolInput(question="Test question")
            assert input_obj.question == "Test question"
            assert input_obj.rounds == 3

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
        # Result depends on whether LangChain is installed
        assert isinstance(result, bool)
        assert result == LANGCHAIN_AVAILABLE

    def test_get_langchain_version(self):
        """Test getting LangChain version."""
        version = get_langchain_version()

        if LANGCHAIN_AVAILABLE:
            assert version is not None
            assert isinstance(version, str)
        else:
            assert version is None


# =============================================================================
# Integration Tests (skipped if LangChain not available)
# =============================================================================


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
class TestLangChainIntegration:
    """Integration tests that require LangChain."""

    def test_tool_inherits_from_base_tool(self):
        """Test that AragoraTool inherits from LangChain BaseTool."""
        from langchain_core.tools import BaseTool

        tool = AragoraTool()
        assert isinstance(tool, BaseTool)

    def test_retriever_inherits_from_base_retriever(self):
        """Test that AragoraRetriever inherits from LangChain BaseRetriever."""
        from langchain_core.retrievers import BaseRetriever

        retriever = AragoraRetriever()
        assert isinstance(retriever, BaseRetriever)

    def test_callback_handler_inherits_from_base_callback(self):
        """Test that AragoraCallbackHandler inherits from BaseCallbackHandler."""
        from langchain_core.callbacks.base import BaseCallbackHandler

        handler = AragoraCallbackHandler()
        assert isinstance(handler, BaseCallbackHandler)

    def test_tool_has_args_schema(self):
        """Test that tool has proper args_schema."""
        from pydantic import BaseModel

        tool = AragoraTool()
        assert hasattr(tool, "args_schema")
        assert issubclass(tool.args_schema, BaseModel)
