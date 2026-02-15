"""Tests for VerticalsAPI client resource."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.client.client import AragoraClient
from aragora.client.resources.verticals import VerticalsAPI


@pytest.fixture
def mock_client() -> AragoraClient:
    client = MagicMock(spec=AragoraClient)
    return client


@pytest.fixture
def api(mock_client: AragoraClient) -> VerticalsAPI:
    return VerticalsAPI(mock_client)


SAMPLE_VERTICAL = {
    "id": "healthcare",
    "name": "Healthcare",
    "description": "Healthcare vertical with HIPAA compliance",
    "status": "active",
    "tools": ["fhir_reader", "clinical_lookup"],
    "compliance_frameworks": ["hipaa", "hitech"],
}

SAMPLE_VERTICAL_2 = {
    "id": "financial",
    "name": "Financial Services",
    "description": "Financial services vertical with SOX compliance",
    "status": "active",
    "tools": ["risk_calculator", "audit_trail"],
    "compliance_frameworks": ["sox", "pci_dss"],
}

SAMPLE_TOOLS = {
    "vertical_id": "healthcare",
    "tools": [
        {"name": "fhir_reader", "type": "data_access", "enabled": True},
        {"name": "clinical_lookup", "type": "reference", "enabled": True},
    ],
}

SAMPLE_COMPLIANCE = {
    "vertical_id": "healthcare",
    "frameworks": [
        {"name": "HIPAA", "status": "compliant", "controls": 42},
        {"name": "HITECH", "status": "partial", "controls": 18},
    ],
}

SAMPLE_SUGGESTION = {
    "suggested_vertical": "healthcare",
    "confidence": 0.92,
    "reasoning": "Task involves patient data and clinical decisions",
}

SAMPLE_AGENT = {
    "agent_id": "vert-agent-001",
    "vertical_id": "healthcare",
    "status": "created",
    "specialization": "clinical_reviewer",
}

SAMPLE_DEBATE = {
    "debate_id": "vert-debate-001",
    "vertical_id": "healthcare",
    "status": "running",
    "task": "Evaluate treatment protocol",
}

SAMPLE_CONFIG_UPDATED = {
    "vertical_id": "healthcare",
    "config": {"max_agents": 5, "compliance_level": "strict"},
    "updated": True,
}


class TestVerticalsList:
    def test_list_default(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"verticals": [SAMPLE_VERTICAL, SAMPLE_VERTICAL_2]}
        results = api.list()
        assert len(results) == 2
        assert results[0]["id"] == "healthcare"
        assert results[1]["id"] == "financial"
        mock_client._get.assert_called_once_with("/api/v1/verticals", {})

    def test_list_with_keyword(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"verticals": [SAMPLE_VERTICAL]}
        results = api.list(keyword="health")
        assert len(results) == 1
        assert results[0]["name"] == "Healthcare"
        mock_client._get.assert_called_once_with("/api/v1/verticals", {"keyword": "health"})

    def test_list_empty(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"verticals": []}
        results = api.list()
        assert results == []

    def test_list_no_keyword_sends_empty_params(
        self, api: VerticalsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"verticals": []}
        api.list(keyword=None)
        mock_client._get.assert_called_once_with("/api/v1/verticals", {})

    def test_list_missing_verticals_key(
        self, api: VerticalsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {}
        results = api.list()
        assert results == []

    @pytest.mark.asyncio
    async def test_list_async(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"verticals": [SAMPLE_VERTICAL]})
        results = await api.list_async()
        assert len(results) == 1
        assert results[0]["id"] == "healthcare"

    @pytest.mark.asyncio
    async def test_list_async_with_keyword(
        self, api: VerticalsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"verticals": [SAMPLE_VERTICAL]})
        results = await api.list_async(keyword="health")
        assert len(results) == 1
        mock_client._get_async.assert_called_once_with("/api/v1/verticals", {"keyword": "health"})

    @pytest.mark.asyncio
    async def test_list_async_missing_verticals_key(
        self, api: VerticalsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={})
        results = await api.list_async()
        assert results == []


class TestVerticalsGet:
    def test_get(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_VERTICAL
        result = api.get("healthcare")
        assert result["id"] == "healthcare"
        assert result["name"] == "Healthcare"
        mock_client._get.assert_called_once_with("/api/v1/verticals/healthcare")

    def test_get_returns_full_response(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_VERTICAL
        result = api.get("healthcare")
        assert result["compliance_frameworks"] == ["hipaa", "hitech"]
        assert result["tools"] == ["fhir_reader", "clinical_lookup"]

    @pytest.mark.asyncio
    async def test_get_async(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_VERTICAL)
        result = await api.get_async("healthcare")
        assert result["id"] == "healthcare"
        mock_client._get_async.assert_called_once_with("/api/v1/verticals/healthcare")


class TestVerticalsTools:
    def test_tools(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_TOOLS
        result = api.tools("healthcare")
        assert result["vertical_id"] == "healthcare"
        assert len(result["tools"]) == 2
        assert result["tools"][0]["name"] == "fhir_reader"
        mock_client._get.assert_called_once_with("/api/v1/verticals/healthcare/tools")

    def test_tools_different_vertical(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {
            "vertical_id": "financial",
            "tools": [{"name": "risk_calculator", "type": "analysis", "enabled": True}],
        }
        result = api.tools("financial")
        assert result["vertical_id"] == "financial"
        mock_client._get.assert_called_once_with("/api/v1/verticals/financial/tools")

    @pytest.mark.asyncio
    async def test_tools_async(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_TOOLS)
        result = await api.tools_async("healthcare")
        assert result["vertical_id"] == "healthcare"
        assert len(result["tools"]) == 2
        mock_client._get_async.assert_called_once_with("/api/v1/verticals/healthcare/tools")


class TestVerticalsCompliance:
    def test_compliance(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_COMPLIANCE
        result = api.compliance("healthcare")
        assert result["vertical_id"] == "healthcare"
        assert len(result["frameworks"]) == 2
        assert result["frameworks"][0]["name"] == "HIPAA"
        mock_client._get.assert_called_once_with("/api/v1/verticals/healthcare/compliance")

    def test_compliance_checks_framework_details(
        self, api: VerticalsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = SAMPLE_COMPLIANCE
        result = api.compliance("healthcare")
        hipaa = result["frameworks"][0]
        assert hipaa["status"] == "compliant"
        assert hipaa["controls"] == 42

    @pytest.mark.asyncio
    async def test_compliance_async(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_COMPLIANCE)
        result = await api.compliance_async("healthcare")
        assert result["vertical_id"] == "healthcare"
        mock_client._get_async.assert_called_once_with("/api/v1/verticals/healthcare/compliance")


class TestVerticalsSuggest:
    def test_suggest(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_SUGGESTION
        result = api.suggest("Review patient treatment records")
        assert result["suggested_vertical"] == "healthcare"
        assert result["confidence"] == 0.92
        mock_client._get.assert_called_once_with(
            "/api/v1/verticals/suggest",
            {"task": "Review patient treatment records"},
        )

    def test_suggest_passes_task_param(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {
            "suggested_vertical": "financial",
            "confidence": 0.85,
            "reasoning": "Task involves financial analysis",
        }
        result = api.suggest("Analyze quarterly earnings report")
        assert result["suggested_vertical"] == "financial"
        mock_client._get.assert_called_once_with(
            "/api/v1/verticals/suggest",
            {"task": "Analyze quarterly earnings report"},
        )

    @pytest.mark.asyncio
    async def test_suggest_async(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_SUGGESTION)
        result = await api.suggest_async("Review patient treatment records")
        assert result["suggested_vertical"] == "healthcare"
        mock_client._get_async.assert_called_once_with(
            "/api/v1/verticals/suggest",
            {"task": "Review patient treatment records"},
        )


class TestVerticalsUpdateConfig:
    def test_update_config(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._put.return_value = SAMPLE_CONFIG_UPDATED
        config = {"max_agents": 5, "compliance_level": "strict"}
        result = api.update_config("healthcare", config)
        assert result["updated"] is True
        assert result["vertical_id"] == "healthcare"
        mock_client._put.assert_called_once_with(
            "/api/v1/verticals/healthcare/config",
            {"max_agents": 5, "compliance_level": "strict"},
        )

    def test_update_config_empty_config(
        self, api: VerticalsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._put.return_value = {"vertical_id": "legal", "config": {}, "updated": True}
        result = api.update_config("legal", {})
        assert result["updated"] is True
        mock_client._put.assert_called_once_with("/api/v1/verticals/legal/config", {})

    def test_update_config_nested_data(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._put.return_value = SAMPLE_CONFIG_UPDATED
        nested_config = {
            "agents": {"max": 10, "types": ["reviewer", "analyst"]},
            "compliance": {"level": "strict", "frameworks": ["hipaa"]},
        }
        api.update_config("healthcare", nested_config)
        mock_client._put.assert_called_once_with(
            "/api/v1/verticals/healthcare/config", nested_config
        )

    @pytest.mark.asyncio
    async def test_update_config_async(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._put_async = AsyncMock(return_value=SAMPLE_CONFIG_UPDATED)
        config = {"max_agents": 5}
        result = await api.update_config_async("healthcare", config)
        assert result["updated"] is True
        mock_client._put_async.assert_called_once_with(
            "/api/v1/verticals/healthcare/config", {"max_agents": 5}
        )


class TestVerticalsCreateAgent:
    def test_create_agent(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_AGENT
        payload = {"specialization": "clinical_reviewer", "model": "claude"}
        result = api.create_agent("healthcare", payload)
        assert result["agent_id"] == "vert-agent-001"
        assert result["vertical_id"] == "healthcare"
        assert result["status"] == "created"
        mock_client._post.assert_called_once_with("/api/v1/verticals/healthcare/agent", payload)

    def test_create_agent_minimal_payload(
        self, api: VerticalsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = SAMPLE_AGENT
        api.create_agent("healthcare", {})
        mock_client._post.assert_called_once_with("/api/v1/verticals/healthcare/agent", {})

    def test_create_agent_different_vertical(
        self, api: VerticalsAPI, mock_client: AragoraClient
    ) -> None:
        expected = {
            "agent_id": "vert-agent-002",
            "vertical_id": "legal",
            "status": "created",
            "specialization": "contract_analyst",
        }
        mock_client._post.return_value = expected
        payload = {"specialization": "contract_analyst"}
        result = api.create_agent("legal", payload)
        assert result["vertical_id"] == "legal"
        assert result["specialization"] == "contract_analyst"
        mock_client._post.assert_called_once_with("/api/v1/verticals/legal/agent", payload)

    @pytest.mark.asyncio
    async def test_create_agent_async(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value=SAMPLE_AGENT)
        payload = {"specialization": "clinical_reviewer"}
        result = await api.create_agent_async("healthcare", payload)
        assert result["agent_id"] == "vert-agent-001"
        mock_client._post_async.assert_called_once_with(
            "/api/v1/verticals/healthcare/agent", payload
        )


class TestVerticalsCreateDebate:
    def test_create_debate(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_DEBATE
        payload = {"task": "Evaluate treatment protocol", "rounds": 3}
        result = api.create_debate("healthcare", payload)
        assert result["debate_id"] == "vert-debate-001"
        assert result["vertical_id"] == "healthcare"
        assert result["status"] == "running"
        mock_client._post.assert_called_once_with("/api/v1/verticals/healthcare/debate", payload)

    def test_create_debate_with_agents(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_DEBATE
        payload = {
            "task": "Evaluate treatment protocol",
            "agents": ["clinical_reviewer", "pharmacologist"],
            "rounds": 5,
        }
        api.create_debate("healthcare", payload)
        call_args = mock_client._post.call_args
        assert call_args[0][1]["agents"] == ["clinical_reviewer", "pharmacologist"]
        assert call_args[0][1]["rounds"] == 5

    def test_create_debate_minimal_payload(
        self, api: VerticalsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = SAMPLE_DEBATE
        api.create_debate("healthcare", {"task": "minimal"})
        mock_client._post.assert_called_once_with(
            "/api/v1/verticals/healthcare/debate", {"task": "minimal"}
        )

    @pytest.mark.asyncio
    async def test_create_debate_async(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value=SAMPLE_DEBATE)
        payload = {"task": "Evaluate treatment protocol"}
        result = await api.create_debate_async("healthcare", payload)
        assert result["debate_id"] == "vert-debate-001"
        mock_client._post_async.assert_called_once_with(
            "/api/v1/verticals/healthcare/debate", payload
        )


class TestVerticalsInit:
    def test_init_stores_client(self, mock_client: AragoraClient) -> None:
        api = VerticalsAPI(mock_client)
        assert api._client is mock_client

    def test_api_uses_correct_base_paths(
        self, api: VerticalsAPI, mock_client: AragoraClient
    ) -> None:
        """Verify all methods use /api/v1/verticals base path."""
        mock_client._get.return_value = {"verticals": []}
        api.list()
        assert mock_client._get.call_args[0][0] == "/api/v1/verticals"

        mock_client._get.return_value = SAMPLE_VERTICAL
        api.get("test")
        assert mock_client._get.call_args[0][0] == "/api/v1/verticals/test"


class TestVerticalsEdgeCases:
    def test_list_empty_keyword_string(self, api: VerticalsAPI, mock_client: AragoraClient) -> None:
        """Empty string keyword should not be included in params."""
        mock_client._get.return_value = {"verticals": []}
        api.list(keyword="")
        # Empty string is falsy, so params should be empty
        mock_client._get.assert_called_once_with("/api/v1/verticals", {})

    def test_get_special_characters_in_id(
        self, api: VerticalsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"id": "health-care_v2"}
        api.get("health-care_v2")
        mock_client._get.assert_called_once_with("/api/v1/verticals/health-care_v2")

    def test_suggest_long_task_description(
        self, api: VerticalsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = SAMPLE_SUGGESTION
        long_task = "A" * 1000
        api.suggest(long_task)
        mock_client._get.assert_called_once_with("/api/v1/verticals/suggest", {"task": long_task})

    def test_update_config_preserves_config_dict(
        self, api: VerticalsAPI, mock_client: AragoraClient
    ) -> None:
        """Ensure the config dict passed is not mutated."""
        mock_client._put.return_value = SAMPLE_CONFIG_UPDATED
        original_config = {"max_agents": 5}
        config_copy = original_config.copy()
        api.update_config("healthcare", original_config)
        assert original_config == config_copy
