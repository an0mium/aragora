"""Tests for integrations/langchain/tools.py — LangChain tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from aragora.integrations.langchain.tools import (
    AragoraDebateInput,
    AragoraDebateTool,
    AragoraDecisionInput,
    AragoraDecisionTool,
    AragoraKnowledgeInput,
    AragoraKnowledgeTool,
    AragoraToolInput,
    get_aragora_tools,
    get_langchain_version,
)


# =============================================================================
# Input schemas
# =============================================================================


class TestInputSchemas:
    def test_aragora_tool_input(self):
        inp = AragoraToolInput(question="Why?")
        assert inp.question == "Why?"
        assert inp.rounds == 3
        assert inp.consensus_threshold == 0.8
        assert inp.include_evidence is True

    def test_aragora_debate_input(self):
        inp = AragoraDebateInput(task="test")
        assert inp.task == "test"
        assert inp.agents is None
        assert inp.max_rounds is None

    def test_aragora_knowledge_input(self):
        inp = AragoraKnowledgeInput(query="search term")
        assert inp.query == "search term"
        assert inp.limit == 5

    def test_aragora_decision_input(self):
        inp = AragoraDecisionInput(question="approve?")
        assert inp.question == "approve?"
        assert inp.options is None


# =============================================================================
# get_langchain_version
# =============================================================================


class TestGetLangchainVersion:
    def test_returns_string_or_none(self):
        result = get_langchain_version()
        assert result is None or isinstance(result, str)


# =============================================================================
# AragoraDebateTool
# =============================================================================


class TestDebateTool:
    def test_defaults(self):
        tool = AragoraDebateTool()
        assert tool.name == "aragora_debate"
        assert tool.aragora_url == "http://localhost:8080"
        assert tool.api_token is None
        assert tool.default_agents == ["claude", "gpt-4", "gemini"]
        assert tool.default_max_rounds == 5

    def test_custom_config(self):
        tool = AragoraDebateTool(aragora_url="http://example.com", api_token="tok-123")
        assert tool.aragora_url == "http://example.com"
        assert tool.api_token == "tok-123"

    @pytest.mark.asyncio
    async def test_arun_consensus(self):
        tool = AragoraDebateTool(api_token="test-tok")
        mock_resp = httpx.Response(
            200,
            json={
                "consensus_reached": True,
                "confidence": 0.85,
                "final_answer": "Use PostgreSQL",
            },
            request=httpx.Request("POST", "http://localhost:8080/api/debate/start"),
        )
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await tool._arun("Which DB?")
        assert "PostgreSQL" in result
        assert "85%" in result

    @pytest.mark.asyncio
    async def test_arun_no_consensus(self):
        tool = AragoraDebateTool()
        mock_resp = httpx.Response(
            200,
            json={"consensus_reached": False, "rounds": 5},
            request=httpx.Request("POST", "http://localhost:8080/api/debate/start"),
        )
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await tool._arun("Which DB?")
        assert "No consensus" in result

    @pytest.mark.asyncio
    async def test_arun_error(self):
        tool = AragoraDebateTool()
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            result = await tool._arun("Which DB?")
        assert "Error" in result


# =============================================================================
# AragoraKnowledgeTool
# =============================================================================


class TestKnowledgeTool:
    def test_defaults(self):
        tool = AragoraKnowledgeTool()
        assert tool.name == "aragora_knowledge"
        assert tool.timeout_seconds == 30.0

    @pytest.mark.asyncio
    async def test_arun_results(self):
        tool = AragoraKnowledgeTool(api_token="tok")
        mock_resp = httpx.Response(
            200,
            json={
                "items": [
                    {"title": "Migration Guide", "content": "Step 1...", "confidence": 0.9},
                    {"title": "Best Practices", "content": "Always...", "confidence": 0.7},
                ]
            },
            request=httpx.Request("GET", "http://localhost:8080/api/v1/knowledge/search"),
        )
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            result = await tool._arun("migrations")
        assert "Migration Guide" in result
        assert "Best Practices" in result

    @pytest.mark.asyncio
    async def test_arun_no_results(self):
        tool = AragoraKnowledgeTool()
        mock_resp = httpx.Response(
            200,
            json={"items": []},
            request=httpx.Request("GET", "http://localhost:8080/api/v1/knowledge/search"),
        )
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            result = await tool._arun("nothing")
        assert "No knowledge found" in result

    @pytest.mark.asyncio
    async def test_arun_error(self):
        tool = AragoraKnowledgeTool()
        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            result = await tool._arun("test")
        assert "Error" in result


# =============================================================================
# AragoraDecisionTool
# =============================================================================


class TestDecisionTool:
    def test_defaults(self):
        tool = AragoraDecisionTool()
        assert tool.name == "aragora_decision"
        assert tool.timeout_seconds == 120.0

    @pytest.mark.asyncio
    async def test_arun_decision(self):
        tool = AragoraDecisionTool(api_token="tok")
        mock_resp = httpx.Response(
            200,
            json={
                "decision": "Approve",
                "confidence": 0.92,
                "rationale": "Solid plan",
                "receipt_id": "r-123",
            },
            request=httpx.Request("POST", "http://localhost:8080/api/v1/decisions/make"),
        )
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await tool._arun("Approve budget?")
        assert "Approve" in result
        assert "r-123" in result
        assert "92%" in result

    @pytest.mark.asyncio
    async def test_arun_with_options(self):
        tool = AragoraDecisionTool()
        mock_resp = httpx.Response(
            200,
            json={
                "decision": "Option A",
                "confidence": 0.75,
                "rationale": "Better fit",
                "receipt_id": "r-456",
            },
            request=httpx.Request("POST", "http://localhost:8080/api/v1/decisions/make"),
        )
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await tool._arun("Pick one", options=["Option A", "Option B"])
        assert "Option A" in result

    @pytest.mark.asyncio
    async def test_arun_error(self):
        tool = AragoraDecisionTool()
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=RuntimeError("timeout"),
        ):
            result = await tool._arun("test?")
        assert "Error" in result


# =============================================================================
# get_aragora_tools
# =============================================================================


class TestGetAragoraTools:
    def test_returns_three_tools(self):
        tools = get_aragora_tools()
        assert len(tools) == 3

    def test_custom_url(self):
        tools = get_aragora_tools(aragora_url="http://custom:9090", api_token="tok")
        for tool in tools:
            assert tool.aragora_url == "http://custom:9090"
            assert tool.api_token == "tok"
