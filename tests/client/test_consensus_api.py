"""Tests for ConsensusAPI client resource."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.client.client import AragoraClient
from aragora.client.resources.consensus import (
    ConsensusAPI,
    ConsensusStats,
    Dissent,
    RiskWarning,
    SettledTopic,
    SimilarDebate,
)


@pytest.fixture
def mock_client() -> AragoraClient:
    client = MagicMock(spec=AragoraClient)
    return client


@pytest.fixture
def api(mock_client: AragoraClient) -> ConsensusAPI:
    return ConsensusAPI(mock_client)


# ---------------------------------------------------------------------------
# Sample response payloads
# ---------------------------------------------------------------------------

SAMPLE_SIMILAR_DEBATE = {
    "id": "deb-001",
    "topic": "Rate limiter design",
    "conclusion": "Use token bucket",
    "strength": "strong",
    "confidence": 0.92,
    "similarity": 0.87,
    "timestamp": "2026-02-10T08:00:00Z",
    "dissent_count": 2,
}

SAMPLE_SETTLED_TOPIC = {
    "topic": "Database choice for analytics",
    "conclusion": "Use ClickHouse for OLAP workloads",
    "confidence": 0.95,
    "strength": "unanimous",
    "last_debated": "2026-02-09T12:00:00Z",
    "debate_count": 3,
}

SAMPLE_DISSENT = {
    "id": "dis-001",
    "debate_id": "deb-001",
    "agent_id": "agent-critic",
    "dissent_type": "alternative_approach",
    "content": "Consider sliding window instead",
    "reasoning": "Better for bursty traffic patterns",
    "confidence": 0.78,
    "acknowledged": True,
    "rebuttal": "Token bucket handles bursts with bucket size",
    "timestamp": "2026-02-10T08:05:00Z",
}

SAMPLE_RISK_WARNING = {
    "id": "rw-001",
    "debate_id": "deb-001",
    "agent_id": "agent-risk",
    "content": "Rate limiter may cause cascading failures",
    "reasoning": "If upstream retries on 429, load amplifies",
    "severity": "high",
    "acknowledged": False,
    "timestamp": "2026-02-10T08:10:00Z",
}

SAMPLE_STATS = {
    "total_consensuses": 42,
    "total_dissents": 15,
    "by_strength": {"strong": 20, "moderate": 15, "weak": 7},
    "by_domain": {"architecture": 25, "security": 10, "operations": 7},
    "avg_confidence": 0.85,
}


# ---------------------------------------------------------------------------
# SimilarDebate dataclass tests
# ---------------------------------------------------------------------------


class TestSimilarDebateDataclass:
    def test_from_dict_full(self) -> None:
        debate = SimilarDebate.from_dict(SAMPLE_SIMILAR_DEBATE)
        assert debate.id == "deb-001"
        assert debate.topic == "Rate limiter design"
        assert debate.conclusion == "Use token bucket"
        assert debate.strength == "strong"
        assert debate.confidence == 0.92
        assert debate.similarity == 0.87
        assert debate.timestamp == "2026-02-10T08:00:00Z"
        assert debate.dissent_count == 2

    def test_from_dict_empty(self) -> None:
        debate = SimilarDebate.from_dict({})
        assert debate.id == ""
        assert debate.topic == ""
        assert debate.conclusion == ""
        assert debate.strength == "unknown"
        assert debate.confidence == 0.0
        assert debate.similarity == 0.0
        assert debate.timestamp == ""
        assert debate.dissent_count == 0

    def test_from_dict_partial(self) -> None:
        debate = SimilarDebate.from_dict({"id": "x", "topic": "y"})
        assert debate.id == "x"
        assert debate.topic == "y"
        assert debate.confidence == 0.0
        assert debate.dissent_count == 0

    def test_defaults(self) -> None:
        debate = SimilarDebate(
            id="a",
            topic="t",
            conclusion="c",
            strength="s",
            confidence=0.5,
            similarity=0.6,
            timestamp="ts",
        )
        assert debate.dissent_count == 0


# ---------------------------------------------------------------------------
# SettledTopic dataclass tests
# ---------------------------------------------------------------------------


class TestSettledTopicDataclass:
    def test_from_dict_full(self) -> None:
        topic = SettledTopic.from_dict(SAMPLE_SETTLED_TOPIC)
        assert topic.topic == "Database choice for analytics"
        assert topic.conclusion == "Use ClickHouse for OLAP workloads"
        assert topic.confidence == 0.95
        assert topic.strength == "unanimous"
        assert topic.last_debated == "2026-02-09T12:00:00Z"
        assert topic.debate_count == 3

    def test_from_dict_empty(self) -> None:
        topic = SettledTopic.from_dict({})
        assert topic.topic == ""
        assert topic.conclusion == ""
        assert topic.confidence == 0.0
        assert topic.strength == "unknown"
        assert topic.last_debated == ""
        assert topic.debate_count == 1

    def test_defaults(self) -> None:
        topic = SettledTopic(
            topic="t",
            conclusion="c",
            confidence=0.8,
            strength="strong",
            last_debated="2026-01-01",
        )
        assert topic.debate_count == 1


# ---------------------------------------------------------------------------
# Dissent dataclass tests
# ---------------------------------------------------------------------------


class TestDissentDataclass:
    def test_from_dict_full(self) -> None:
        dissent = Dissent.from_dict(SAMPLE_DISSENT)
        assert dissent.id == "dis-001"
        assert dissent.debate_id == "deb-001"
        assert dissent.agent_id == "agent-critic"
        assert dissent.dissent_type == "alternative_approach"
        assert dissent.content == "Consider sliding window instead"
        assert dissent.reasoning == "Better for bursty traffic patterns"
        assert dissent.confidence == 0.78
        assert dissent.acknowledged is True
        assert dissent.rebuttal == "Token bucket handles bursts with bucket size"
        assert dissent.timestamp == "2026-02-10T08:05:00Z"

    def test_from_dict_empty(self) -> None:
        dissent = Dissent.from_dict({})
        assert dissent.id == ""
        assert dissent.debate_id == ""
        assert dissent.agent_id == ""
        assert dissent.dissent_type == ""
        assert dissent.content == ""
        assert dissent.reasoning == ""
        assert dissent.confidence == 0.0
        assert dissent.acknowledged is False
        assert dissent.rebuttal == ""
        assert dissent.timestamp == ""

    def test_defaults(self) -> None:
        dissent = Dissent(
            id="d",
            debate_id="deb",
            agent_id="a",
            dissent_type="minor_quibble",
            content="c",
            reasoning="r",
            confidence=0.5,
        )
        assert dissent.acknowledged is False
        assert dissent.rebuttal == ""
        assert dissent.timestamp == ""


# ---------------------------------------------------------------------------
# RiskWarning dataclass tests
# ---------------------------------------------------------------------------


class TestRiskWarningDataclass:
    def test_from_dict_full(self) -> None:
        warning = RiskWarning.from_dict(SAMPLE_RISK_WARNING)
        assert warning.id == "rw-001"
        assert warning.debate_id == "deb-001"
        assert warning.agent_id == "agent-risk"
        assert warning.content == "Rate limiter may cause cascading failures"
        assert warning.reasoning == "If upstream retries on 429, load amplifies"
        assert warning.severity == "high"
        assert warning.acknowledged is False
        assert warning.timestamp == "2026-02-10T08:10:00Z"

    def test_from_dict_empty(self) -> None:
        warning = RiskWarning.from_dict({})
        assert warning.id == ""
        assert warning.debate_id == ""
        assert warning.agent_id == ""
        assert warning.content == ""
        assert warning.reasoning == ""
        assert warning.severity == "medium"
        assert warning.acknowledged is False
        assert warning.timestamp == ""

    def test_defaults(self) -> None:
        warning = RiskWarning(
            id="w",
            debate_id="deb",
            agent_id="a",
            content="c",
            reasoning="r",
        )
        assert warning.severity == "medium"
        assert warning.acknowledged is False
        assert warning.timestamp == ""


# ---------------------------------------------------------------------------
# ConsensusStats dataclass tests
# ---------------------------------------------------------------------------


class TestConsensusStatsDataclass:
    def test_from_dict_full(self) -> None:
        stats = ConsensusStats.from_dict(SAMPLE_STATS)
        assert stats.total_consensuses == 42
        assert stats.total_dissents == 15
        assert stats.by_strength == {"strong": 20, "moderate": 15, "weak": 7}
        assert stats.by_domain == {"architecture": 25, "security": 10, "operations": 7}
        assert stats.avg_confidence == 0.85

    def test_from_dict_empty(self) -> None:
        stats = ConsensusStats.from_dict({})
        assert stats.total_consensuses == 0
        assert stats.total_dissents == 0
        assert stats.by_strength == {}
        assert stats.by_domain == {}
        assert stats.avg_confidence == 0.0


# ---------------------------------------------------------------------------
# ConsensusAPI.find_similar / find_similar_async
# ---------------------------------------------------------------------------


class TestFindSimilar:
    def test_find_similar_basic(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"debates": [SAMPLE_SIMILAR_DEBATE]}
        results = api.find_similar("rate limiting")
        assert len(results) == 1
        assert isinstance(results[0], SimilarDebate)
        assert results[0].id == "deb-001"
        mock_client._get.assert_called_once_with(
            "/api/consensus/similar",
            params={"topic": "rate limiting", "limit": 10, "min_confidence": 0.0},
        )

    def test_find_similar_with_domain(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"debates": []}
        api.find_similar("caching", domain="architecture")
        params = mock_client._get.call_args[1]["params"]
        assert params["domain"] == "architecture"
        assert params["topic"] == "caching"

    def test_find_similar_with_all_params(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"debates": []}
        api.find_similar("topic", domain="security", min_confidence=0.5, limit=5)
        params = mock_client._get.call_args[1]["params"]
        assert params["topic"] == "topic"
        assert params["domain"] == "security"
        assert params["min_confidence"] == 0.5
        assert params["limit"] == 5

    def test_find_similar_no_domain_omits_key(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"debates": []}
        api.find_similar("topic")
        params = mock_client._get.call_args[1]["params"]
        assert "domain" not in params

    def test_find_similar_empty_response(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"debates": []}
        results = api.find_similar("no matches")
        assert results == []

    def test_find_similar_missing_debates_key(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {}
        results = api.find_similar("topic")
        assert results == []

    def test_find_similar_multiple_results(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        debate2 = {**SAMPLE_SIMILAR_DEBATE, "id": "deb-002", "similarity": 0.75}
        mock_client._get.return_value = {"debates": [SAMPLE_SIMILAR_DEBATE, debate2]}
        results = api.find_similar("rate limiting")
        assert len(results) == 2
        assert results[0].id == "deb-001"
        assert results[1].id == "deb-002"
        assert results[1].similarity == 0.75

    @pytest.mark.asyncio
    async def test_find_similar_async(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(
            return_value={"debates": [SAMPLE_SIMILAR_DEBATE]}
        )
        results = await api.find_similar_async("rate limiting")
        assert len(results) == 1
        assert results[0].topic == "Rate limiter design"
        mock_client._get_async.assert_called_once_with(
            "/api/consensus/similar",
            params={"topic": "rate limiting", "limit": 10, "min_confidence": 0.0},
        )

    @pytest.mark.asyncio
    async def test_find_similar_async_with_domain(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"debates": []})
        await api.find_similar_async("topic", domain="ops")
        params = mock_client._get_async.call_args[1]["params"]
        assert params["domain"] == "ops"

    @pytest.mark.asyncio
    async def test_find_similar_async_missing_key(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={})
        results = await api.find_similar_async("topic")
        assert results == []


# ---------------------------------------------------------------------------
# ConsensusAPI.get_settled / get_settled_async
# ---------------------------------------------------------------------------


class TestGetSettled:
    def test_get_settled_default(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"topics": [SAMPLE_SETTLED_TOPIC]}
        results = api.get_settled()
        assert len(results) == 1
        assert isinstance(results[0], SettledTopic)
        assert results[0].topic == "Database choice for analytics"
        mock_client._get.assert_called_once_with(
            "/api/consensus/settled",
            params={"limit": 20, "min_confidence": 0.7},
        )

    def test_get_settled_with_domain(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"topics": []}
        api.get_settled(domain="architecture")
        params = mock_client._get.call_args[1]["params"]
        assert params["domain"] == "architecture"

    def test_get_settled_custom_params(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"topics": []}
        api.get_settled(domain="security", min_confidence=0.9, limit=5)
        params = mock_client._get.call_args[1]["params"]
        assert params["min_confidence"] == 0.9
        assert params["limit"] == 5
        assert params["domain"] == "security"

    def test_get_settled_no_domain_omits_key(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"topics": []}
        api.get_settled()
        params = mock_client._get.call_args[1]["params"]
        assert "domain" not in params

    def test_get_settled_empty_response(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"topics": []}
        results = api.get_settled()
        assert results == []

    def test_get_settled_missing_topics_key(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {}
        results = api.get_settled()
        assert results == []

    @pytest.mark.asyncio
    async def test_get_settled_async(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(
            return_value={"topics": [SAMPLE_SETTLED_TOPIC]}
        )
        results = await api.get_settled_async()
        assert len(results) == 1
        assert results[0].confidence == 0.95

    @pytest.mark.asyncio
    async def test_get_settled_async_with_domain(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"topics": []})
        await api.get_settled_async(domain="ops")
        params = mock_client._get_async.call_args[1]["params"]
        assert params["domain"] == "ops"

    @pytest.mark.asyncio
    async def test_get_settled_async_missing_key(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={})
        results = await api.get_settled_async()
        assert results == []


# ---------------------------------------------------------------------------
# ConsensusAPI.get_dissents / get_dissents_async
# ---------------------------------------------------------------------------


class TestGetDissents:
    def test_get_dissents_default(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"dissents": [SAMPLE_DISSENT]}
        results = api.get_dissents()
        assert len(results) == 1
        assert isinstance(results[0], Dissent)
        assert results[0].id == "dis-001"
        mock_client._get.assert_called_once_with(
            "/api/consensus/dissents",
            params={"limit": 20},
        )

    def test_get_dissents_with_topic(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"dissents": []}
        api.get_dissents(topic="rate limiting")
        params = mock_client._get.call_args[1]["params"]
        assert params["topic"] == "rate limiting"

    def test_get_dissents_with_type(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"dissents": []}
        api.get_dissents(dissent_type="fundamental_disagreement")
        params = mock_client._get.call_args[1]["params"]
        assert params["type"] == "fundamental_disagreement"

    def test_get_dissents_with_all_params(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"dissents": []}
        api.get_dissents(topic="caching", dissent_type="risk_warning", limit=5)
        params = mock_client._get.call_args[1]["params"]
        assert params["topic"] == "caching"
        assert params["type"] == "risk_warning"
        assert params["limit"] == 5

    def test_get_dissents_no_optional_params_omits_keys(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"dissents": []}
        api.get_dissents()
        params = mock_client._get.call_args[1]["params"]
        assert "topic" not in params
        assert "type" not in params

    def test_get_dissents_empty_response(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"dissents": []}
        results = api.get_dissents()
        assert results == []

    def test_get_dissents_missing_key(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {}
        results = api.get_dissents()
        assert results == []

    def test_get_dissents_multiple(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        dissent2 = {**SAMPLE_DISSENT, "id": "dis-002", "dissent_type": "edge_case_concern"}
        mock_client._get.return_value = {"dissents": [SAMPLE_DISSENT, dissent2]}
        results = api.get_dissents()
        assert len(results) == 2
        assert results[1].dissent_type == "edge_case_concern"

    @pytest.mark.asyncio
    async def test_get_dissents_async(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(
            return_value={"dissents": [SAMPLE_DISSENT]}
        )
        results = await api.get_dissents_async()
        assert len(results) == 1
        assert results[0].content == "Consider sliding window instead"

    @pytest.mark.asyncio
    async def test_get_dissents_async_with_filters(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"dissents": []})
        await api.get_dissents_async(topic="t", dissent_type="minor_quibble", limit=3)
        params = mock_client._get_async.call_args[1]["params"]
        assert params["topic"] == "t"
        assert params["type"] == "minor_quibble"
        assert params["limit"] == 3

    @pytest.mark.asyncio
    async def test_get_dissents_async_missing_key(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={})
        results = await api.get_dissents_async()
        assert results == []


# ---------------------------------------------------------------------------
# ConsensusAPI.get_risk_warnings / get_risk_warnings_async
# ---------------------------------------------------------------------------


class TestGetRiskWarnings:
    def test_get_risk_warnings_default(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"warnings": [SAMPLE_RISK_WARNING]}
        results = api.get_risk_warnings()
        assert len(results) == 1
        assert isinstance(results[0], RiskWarning)
        assert results[0].severity == "high"
        mock_client._get.assert_called_once_with(
            "/api/consensus/risk-warnings",
            params={"limit": 10},
        )

    def test_get_risk_warnings_with_topic(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"warnings": []}
        api.get_risk_warnings(topic="scaling")
        params = mock_client._get.call_args[1]["params"]
        assert params["topic"] == "scaling"

    def test_get_risk_warnings_custom_limit(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"warnings": []}
        api.get_risk_warnings(limit=50)
        params = mock_client._get.call_args[1]["params"]
        assert params["limit"] == 50

    def test_get_risk_warnings_no_topic_omits_key(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"warnings": []}
        api.get_risk_warnings()
        params = mock_client._get.call_args[1]["params"]
        assert "topic" not in params

    def test_get_risk_warnings_empty(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"warnings": []}
        results = api.get_risk_warnings()
        assert results == []

    def test_get_risk_warnings_missing_key(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {}
        results = api.get_risk_warnings()
        assert results == []

    @pytest.mark.asyncio
    async def test_get_risk_warnings_async(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(
            return_value={"warnings": [SAMPLE_RISK_WARNING]}
        )
        results = await api.get_risk_warnings_async()
        assert len(results) == 1
        assert results[0].id == "rw-001"

    @pytest.mark.asyncio
    async def test_get_risk_warnings_async_with_topic(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"warnings": []})
        await api.get_risk_warnings_async(topic="scaling")
        params = mock_client._get_async.call_args[1]["params"]
        assert params["topic"] == "scaling"

    @pytest.mark.asyncio
    async def test_get_risk_warnings_async_missing_key(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={})
        results = await api.get_risk_warnings_async()
        assert results == []


# ---------------------------------------------------------------------------
# ConsensusAPI.get_contrarian_views / get_contrarian_views_async
# ---------------------------------------------------------------------------


class TestGetContrarianViews:
    def test_get_contrarian_views_default(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        contrarian = {**SAMPLE_DISSENT, "dissent_type": "fundamental_disagreement"}
        mock_client._get.return_value = {"views": [contrarian]}
        results = api.get_contrarian_views()
        assert len(results) == 1
        assert isinstance(results[0], Dissent)
        mock_client._get.assert_called_once_with(
            "/api/consensus/contrarian",
            params={"limit": 10},
        )

    def test_get_contrarian_views_custom_limit(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"views": []}
        api.get_contrarian_views(limit=25)
        params = mock_client._get.call_args[1]["params"]
        assert params["limit"] == 25

    def test_get_contrarian_views_empty(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"views": []}
        results = api.get_contrarian_views()
        assert results == []

    def test_get_contrarian_views_missing_key(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {}
        results = api.get_contrarian_views()
        assert results == []

    @pytest.mark.asyncio
    async def test_get_contrarian_views_async(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        contrarian = {**SAMPLE_DISSENT, "dissent_type": "fundamental_disagreement"}
        mock_client._get_async = AsyncMock(return_value={"views": [contrarian]})
        results = await api.get_contrarian_views_async()
        assert len(results) == 1
        assert results[0].dissent_type == "fundamental_disagreement"

    @pytest.mark.asyncio
    async def test_get_contrarian_views_async_custom_limit(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"views": []})
        await api.get_contrarian_views_async(limit=3)
        params = mock_client._get_async.call_args[1]["params"]
        assert params["limit"] == 3

    @pytest.mark.asyncio
    async def test_get_contrarian_views_async_missing_key(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={})
        results = await api.get_contrarian_views_async()
        assert results == []


# ---------------------------------------------------------------------------
# ConsensusAPI.get_stats / get_stats_async
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_get_stats(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_STATS
        stats = api.get_stats()
        assert isinstance(stats, ConsensusStats)
        assert stats.total_consensuses == 42
        assert stats.total_dissents == 15
        assert stats.avg_confidence == 0.85
        assert stats.by_strength["strong"] == 20
        assert stats.by_domain["architecture"] == 25
        mock_client._get.assert_called_once_with("/api/consensus/stats")

    def test_get_stats_empty_response(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {}
        stats = api.get_stats()
        assert stats.total_consensuses == 0
        assert stats.total_dissents == 0
        assert stats.by_strength == {}
        assert stats.by_domain == {}
        assert stats.avg_confidence == 0.0

    @pytest.mark.asyncio
    async def test_get_stats_async(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_STATS)
        stats = await api.get_stats_async()
        assert stats.total_consensuses == 42
        assert stats.avg_confidence == 0.85

    @pytest.mark.asyncio
    async def test_get_stats_async_empty(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={})
        stats = await api.get_stats_async()
        assert stats.total_consensuses == 0


# ---------------------------------------------------------------------------
# Integration-like workflow tests
# ---------------------------------------------------------------------------


class TestWorkflows:
    """Integration-like tests that exercise multiple methods in sequence."""

    def test_find_and_inspect_dissents(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        """Find similar debates, then fetch dissents for the most similar one."""
        mock_client._get.side_effect = [
            {"debates": [SAMPLE_SIMILAR_DEBATE]},
            {"dissents": [SAMPLE_DISSENT]},
        ]
        similar = api.find_similar("rate limiting")
        assert len(similar) == 1
        assert similar[0].dissent_count == 2

        dissents = api.get_dissents(topic=similar[0].topic)
        assert len(dissents) == 1
        assert dissents[0].debate_id == similar[0].id

    def test_settled_topics_with_stats(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        """Get stats, then pull settled topics for the dominant domain."""
        mock_client._get.side_effect = [
            SAMPLE_STATS,
            {"topics": [SAMPLE_SETTLED_TOPIC]},
        ]
        stats = api.get_stats()
        top_domain = max(stats.by_domain, key=stats.by_domain.get)  # type: ignore[arg-type]
        assert top_domain == "architecture"

        settled = api.get_settled(domain=top_domain)
        assert len(settled) == 1

    def test_risk_assessment_workflow(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        """Get risk warnings and contrarian views for a full risk picture."""
        mock_client._get.side_effect = [
            {"warnings": [SAMPLE_RISK_WARNING]},
            {
                "views": [
                    {**SAMPLE_DISSENT, "dissent_type": "fundamental_disagreement"}
                ]
            },
        ]
        warnings = api.get_risk_warnings(topic="rate limiting")
        contrarian = api.get_contrarian_views()

        assert len(warnings) == 1
        assert warnings[0].severity == "high"
        assert len(contrarian) == 1
        assert contrarian[0].dissent_type == "fundamental_disagreement"

    @pytest.mark.asyncio
    async def test_async_workflow(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        """Async workflow: find similar, get dissents, get stats."""
        mock_client._get_async = AsyncMock(
            side_effect=[
                {"debates": [SAMPLE_SIMILAR_DEBATE]},
                {"dissents": [SAMPLE_DISSENT]},
                SAMPLE_STATS,
            ]
        )
        similar = await api.find_similar_async("rate limiting")
        assert len(similar) == 1

        dissents = await api.get_dissents_async(topic=similar[0].topic)
        assert len(dissents) == 1

        stats = await api.get_stats_async()
        assert stats.total_consensuses == 42

    def test_empty_knowledge_base(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        """All endpoints return empty when knowledge base has no data."""
        mock_client._get.return_value = {}
        assert api.find_similar("anything") == []
        assert api.get_settled() == []
        assert api.get_dissents() == []
        assert api.get_risk_warnings() == []
        assert api.get_contrarian_views() == []
        stats = api.get_stats()
        assert stats.total_consensuses == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_from_dict_with_extra_fields_ignored(self) -> None:
        """Extra fields in API response should not cause errors."""
        data = {**SAMPLE_SIMILAR_DEBATE, "extra_field": "ignored", "nested": {"a": 1}}
        debate = SimilarDebate.from_dict(data)
        assert debate.id == "deb-001"
        assert not hasattr(debate, "extra_field")

    def test_from_dict_with_wrong_types_uses_raw_value(self) -> None:
        """from_dict does not coerce types; it passes values through."""
        data = {**SAMPLE_SIMILAR_DEBATE, "confidence": "not_a_float"}
        debate = SimilarDebate.from_dict(data)
        assert debate.confidence == "not_a_float"

    def test_dissent_type_param_maps_to_type_key(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        """The dissent_type Python param maps to 'type' in the API params."""
        mock_client._get.return_value = {"dissents": []}
        api.get_dissents(dissent_type="edge_case_concern")
        params = mock_client._get.call_args[1]["params"]
        assert "type" in params
        assert "dissent_type" not in params

    def test_find_similar_zero_limit(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"debates": []}
        api.find_similar("topic", limit=0)
        params = mock_client._get.call_args[1]["params"]
        assert params["limit"] == 0

    def test_settled_high_confidence_threshold(self, api: ConsensusAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"topics": []}
        api.get_settled(min_confidence=1.0)
        params = mock_client._get.call_args[1]["params"]
        assert params["min_confidence"] == 1.0

    def test_api_stores_client_reference(self, mock_client: AragoraClient) -> None:
        api = ConsensusAPI(mock_client)
        assert api._client is mock_client
