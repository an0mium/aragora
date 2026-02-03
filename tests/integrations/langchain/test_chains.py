"""Tests for integrations/langchain/chains.py — LangChain chains."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from aragora.integrations.langchain.chains import (
    AragoraDebateChain,
    AragoraResearchDebateChain,
)


# =============================================================================
# AragoraDebateChain
# =============================================================================


class TestAragoraDebateChain:
    def test_defaults(self):
        chain = AragoraDebateChain()
        assert chain.aragora_url == "http://localhost:8080"
        assert chain.pre_research is True
        assert chain.post_verify is True
        assert chain.max_rounds == 5
        assert chain.default_agents == ["claude", "gpt-4", "gemini"]

    def test_custom_config(self):
        chain = AragoraDebateChain(
            aragora_url="http://example.com",
            api_token="tok",
            pre_research=False,
            post_verify=False,
        )
        assert chain.aragora_url == "http://example.com"
        assert chain.api_token == "tok"
        assert chain.pre_research is False

    def test_input_keys(self):
        chain = AragoraDebateChain()
        assert chain.input_keys == ["question"]

    def test_output_keys(self):
        chain = AragoraDebateChain()
        assert "answer" in chain.output_keys
        assert "confidence" in chain.output_keys

    def test_chain_type(self):
        chain = AragoraDebateChain()
        assert chain._chain_type == "aragora_debate_chain"

    @pytest.mark.asyncio
    async def test_acall_consensus_no_research(self):
        chain = AragoraDebateChain(pre_research=False, post_verify=False)
        debate_resp = httpx.Response(
            200,
            json={
                "consensus_reached": True,
                "confidence": 0.9,
                "final_answer": "Microservices for this scale",
            },
            request=httpx.Request("POST", "http://localhost:8080/api/debate/start"),
        )
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=debate_resp):
            result = await chain._acall({"question": "Microservices?"})
        assert result["answer"] == "Microservices for this scale"
        assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_acall_with_research(self):
        chain = AragoraDebateChain(pre_research=True, post_verify=False, api_token="tok")
        knowledge_resp = httpx.Response(
            200,
            json={"items": [{"title": "Doc", "content": "Relevant info"}]},
            request=httpx.Request("GET", "http://localhost:8080/api/v1/knowledge/search"),
        )
        debate_resp = httpx.Response(
            200,
            json={
                "consensus_reached": True,
                "confidence": 0.85,
                "final_answer": "Yes",
            },
            request=httpx.Request("POST", "http://localhost:8080/api/debate/start"),
        )
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=knowledge_resp),
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=debate_resp),
        ):
            result = await chain._acall({"question": "Migrate?", "context": "50 consumers"})
        assert result["answer"] == "Yes"
        assert (
            "knowledge" in result["reasoning"].lower() or "research" in result["reasoning"].lower()
        )

    @pytest.mark.asyncio
    async def test_acall_debate_error(self):
        chain = AragoraDebateChain(pre_research=False)
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            result = await chain._acall({"question": "test"})
        assert "failed" in result["answer"].lower()
        assert result["confidence"] == 0

    @pytest.mark.asyncio
    async def test_acall_no_consensus(self):
        chain = AragoraDebateChain(pre_research=False, post_verify=False)
        debate_resp = httpx.Response(
            200,
            json={
                "consensus_reached": False,
                "confidence": 0.4,
                "final_answer": "Split decision",
                "rounds": 5,
            },
            request=httpx.Request("POST", "http://localhost:8080/api/debate/start"),
        )
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=debate_resp):
            result = await chain._acall({"question": "test"})
        assert result["answer"] == "Split decision"
        assert "no consensus" in result["reasoning"].lower()


# =============================================================================
# AragoraResearchDebateChain
# =============================================================================


class TestAragoraResearchDebateChain:
    def test_defaults(self):
        chain = AragoraResearchDebateChain()
        assert chain.search_sources == ["knowledge", "web", "documents"]
        assert chain.max_search_results == 10

    def test_input_keys(self):
        chain = AragoraResearchDebateChain()
        assert chain.input_keys == ["topic"]

    def test_output_keys(self):
        chain = AragoraResearchDebateChain()
        assert "conclusion" in chain.output_keys
        assert "sources" in chain.output_keys

    def test_chain_type(self):
        chain = AragoraResearchDebateChain()
        assert chain._chain_type == "aragora_research_debate_chain"

    @pytest.mark.asyncio
    async def test_acall_full_flow(self):
        chain = AragoraResearchDebateChain(api_token="tok")
        knowledge_resp = httpx.Response(
            200,
            json={"items": [{"title": "Prior Art", "content": "Context here"}]},
            request=httpx.Request("GET", "http://localhost:8080/api/v1/knowledge/search"),
        )
        debate_resp = httpx.Response(
            200,
            json={
                "consensus_reached": True,
                "confidence": 0.88,
                "final_answer": "Go with approach B",
                "rounds": 3,
            },
            request=httpx.Request("POST", "http://localhost:8080/api/debate/start"),
        )
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=knowledge_resp),
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=debate_resp),
        ):
            result = await chain._acall({"topic": "API versioning"})
        assert result["conclusion"] == "Go with approach B"
        assert len(result["sources"]) == 1
        assert "Consensus: True" in result["debate_summary"]

    @pytest.mark.asyncio
    async def test_acall_debate_failure(self):
        chain = AragoraResearchDebateChain(search_sources=[])  # skip research
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=RuntimeError("timeout"),
        ):
            result = await chain._acall({"topic": "test"})
        assert "failed" in result["conclusion"].lower()
        assert result["debate_summary"] == "Debate failed"
