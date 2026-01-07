"""
Tests for insights endpoint handlers.

Tests InsightsHandler covering:
- Route matching (can_handle)
- GET /api/insights/recent - Recent insights listing
- POST /api/insights/extract-detailed - Detailed insight extraction
- Claim extraction heuristics
- Evidence extraction patterns
- Argumentation pattern detection
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, AsyncMock
from dataclasses import dataclass
from enum import Enum

from aragora.server.handlers.insights import InsightsHandler


class MockInsightType(Enum):
    """Mock insight type enum."""
    PATTERN = "pattern"
    CLAIM = "claim"
    EVIDENCE = "evidence"


@dataclass
class MockInsight:
    """Mock insight for testing."""
    id: str
    type: MockInsightType
    title: str
    description: str
    confidence: float
    agents_involved: list
    evidence: list


class MockInsightStore:
    """Mock insight store for testing."""

    def __init__(self, insights: list[MockInsight] | None = None):
        self._insights = insights or []

    async def get_recent_insights(self, limit: int = 20) -> list[MockInsight]:
        """Get recent insights."""
        return self._insights[:limit]


@pytest.fixture
def insights_handler():
    """Create InsightsHandler instance."""
    return InsightsHandler({})


@pytest.fixture
def handler_with_store():
    """Create handler with mock insight store."""
    store = MockInsightStore([
        MockInsight(
            id="insight-001",
            type=MockInsightType.PATTERN,
            title="Consensus Pattern",
            description="Agents reached agreement via negotiation",
            confidence=0.85,
            agents_involved=["claude", "gpt"],
            evidence=["evidence 1", "evidence 2", "evidence 3", "evidence 4"],
        ),
        MockInsight(
            id="insight-002",
            type=MockInsightType.CLAIM,
            title="Key Claim",
            description="Main argument about efficiency",
            confidence=0.72,
            agents_involved=["claude"],
            evidence=["single evidence"],
        ),
    ])
    return InsightsHandler({"insight_store": store})


class TestInsightsHandlerRouting:
    """Test route matching."""

    def test_can_handle_insights_routes(self, insights_handler):
        """Should handle /api/insights/* paths."""
        assert insights_handler.can_handle("/api/insights/recent")
        assert insights_handler.can_handle("/api/insights/extract-detailed")
        assert insights_handler.can_handle("/api/insights/other")

    def test_cannot_handle_other_routes(self, insights_handler):
        """Should not handle non-insights routes."""
        assert not insights_handler.can_handle("/api/debates")
        assert not insights_handler.can_handle("/api/agents")
        assert not insights_handler.can_handle("/insights/recent")


class TestRecentInsights:
    """Test GET /api/insights/recent endpoint."""

    def test_get_recent_insights_success(self, handler_with_store):
        """Should return recent insights."""
        result = handler_with_store.handle_get(
            "/api/insights/recent", {}, None, handler_with_store.ctx
        )

        assert result is not None
        data = json.loads(result.body)
        assert "insights" in data
        assert len(data["insights"]) == 2
        assert data["count"] == 2

    def test_get_recent_insights_fields(self, handler_with_store):
        """Should include expected fields in insights."""
        result = handler_with_store.handle_get(
            "/api/insights/recent", {}, None, handler_with_store.ctx
        )

        data = json.loads(result.body)
        insight = data["insights"][0]

        assert insight["id"] == "insight-001"
        assert insight["type"] == "pattern"
        assert insight["title"] == "Consensus Pattern"
        assert insight["description"] == "Agents reached agreement via negotiation"
        assert insight["confidence"] == 0.85
        assert insight["agents_involved"] == ["claude", "gpt"]

    def test_get_recent_insights_limits_evidence(self, handler_with_store):
        """Should limit evidence to first 3 items."""
        result = handler_with_store.handle_get(
            "/api/insights/recent", {}, None, handler_with_store.ctx
        )

        data = json.loads(result.body)
        evidence = data["insights"][0]["evidence"]

        assert len(evidence) == 3  # Limited to first 3

    def test_get_recent_insights_respects_limit(self, handler_with_store):
        """Should respect limit parameter."""
        result = handler_with_store.handle_get(
            "/api/insights/recent", {"limit": "1"}, None, handler_with_store.ctx
        )

        data = json.loads(result.body)
        assert len(data["insights"]) == 1

    def test_get_recent_insights_caps_limit(self, handler_with_store):
        """Should cap limit at 100."""
        result = handler_with_store.handle_get(
            "/api/insights/recent", {"limit": "1000"}, None, handler_with_store.ctx
        )

        # Handler should cap at 100, but we only have 2 insights
        assert result is not None

    def test_get_recent_insights_no_store(self, insights_handler):
        """Should return error when no insight store configured."""
        result = insights_handler.handle_get(
            "/api/insights/recent", {}, None, {}
        )

        assert result is not None
        data = json.loads(result.body)
        assert "error" in data
        assert data["insights"] == []


class TestExtractDetailedInsights:
    """Test POST /api/insights/extract-detailed endpoint."""

    def test_extract_insights_success(self, insights_handler):
        """Should extract insights from content."""
        mock_handler = Mock()
        mock_handler.rfile.read.return_value = json.dumps({
            "content": "Therefore, we should implement caching because it improves performance."
        }).encode()
        mock_handler.headers = {"Content-Length": "100"}

        result = insights_handler.handle_post(
            "/api/insights/extract-detailed", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "claims" in data
        assert "evidence_chains" in data
        assert "patterns" in data

    def test_extract_insights_missing_content(self, insights_handler):
        """Should return error when content is missing."""
        mock_handler = Mock()
        mock_handler.rfile.read.return_value = json.dumps({}).encode()
        mock_handler.headers = {"Content-Length": "2"}

        result = insights_handler.handle_post(
            "/api/insights/extract-detailed", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 400

    def test_extract_insights_invalid_json(self, insights_handler):
        """Should return error for invalid JSON."""
        mock_handler = Mock()
        mock_handler.rfile.read.return_value = b"not json"
        mock_handler.headers = {"Content-Length": "8"}

        result = insights_handler.handle_post(
            "/api/insights/extract-detailed", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 400


class TestClaimExtraction:
    """Test claim extraction heuristics."""

    def test_extract_claims_with_therefore(self, insights_handler):
        """Should extract claims with 'therefore'."""
        content = "The data shows improvement. Therefore, we should continue."
        claims = insights_handler._extract_claims_from_content(content)

        assert len(claims) > 0
        assert any("Therefore" in c["text"] or "therefore" in c["text"] for c in claims)

    def test_extract_claims_with_should(self, insights_handler):
        """Should extract claims with 'should'."""
        content = "We should implement better testing practices."
        claims = insights_handler._extract_claims_from_content(content)

        assert len(claims) > 0
        assert claims[0]["type"] == "argument"

    def test_extract_claims_with_i_believe(self, insights_handler):
        """Should extract claims with 'I believe'."""
        content = "I believe this approach is more efficient."
        claims = insights_handler._extract_claims_from_content(content)

        assert len(claims) > 0

    def test_extract_claims_limits_results(self, insights_handler):
        """Should limit claims to 20."""
        # Generate content with many potential claims
        content = ". ".join([f"Therefore item {i} is important" for i in range(30)])
        claims = insights_handler._extract_claims_from_content(content)

        assert len(claims) <= 20

    def test_extract_claims_skips_short_sentences(self, insights_handler):
        """Should skip very short sentences."""
        content = "Yes. No. Therefore x."
        claims = insights_handler._extract_claims_from_content(content)

        # Short sentences should be skipped
        assert all(len(c["text"]) >= 10 for c in claims)


class TestEvidenceExtraction:
    """Test evidence extraction patterns."""

    def test_extract_evidence_according_to(self, insights_handler):
        """Should extract 'according to' citations."""
        content = "According to Smith, this is effective."
        evidence = insights_handler._extract_evidence_from_content(content)

        assert len(evidence) > 0
        assert evidence[0]["type"] == "citation"
        assert "Smith" in evidence[0]["source"]

    def test_extract_evidence_research_shows(self, insights_handler):
        """Should extract 'research shows' patterns."""
        content = "Research shows that early intervention helps."
        evidence = insights_handler._extract_evidence_from_content(content)

        assert len(evidence) > 0
        assert evidence[0]["type"] == "research"

    def test_extract_evidence_for_example(self, insights_handler):
        """Should extract examples."""
        content = "For example, Python is widely used."
        evidence = insights_handler._extract_evidence_from_content(content)

        assert len(evidence) > 0
        assert evidence[0]["type"] == "example"

    def test_extract_evidence_limits_results(self, insights_handler):
        """Should limit evidence to 15 items."""
        content = ". ".join([f"According to source {i}, fact {i}" for i in range(20)])
        evidence = insights_handler._extract_evidence_from_content(content)

        assert len(evidence) <= 15


class TestPatternExtraction:
    """Test argumentation pattern detection."""

    def test_extract_pattern_balanced_comparison(self, insights_handler):
        """Should detect balanced comparison pattern."""
        content = "On one hand, A is good. On the other hand, B is also good."
        patterns = insights_handler._extract_patterns_from_content(content)

        types = [p["type"] for p in patterns]
        assert "balanced_comparison" in types

    def test_extract_pattern_concession_rebuttal(self, insights_handler):
        """Should detect concession-rebuttal pattern."""
        content = "While this is true, however the counterargument is stronger."
        patterns = insights_handler._extract_patterns_from_content(content)

        types = [p["type"] for p in patterns]
        assert "concession_rebuttal" in types

    def test_extract_pattern_enumerated_argument(self, insights_handler):
        """Should detect enumerated argument pattern."""
        content = "First, we need planning. Second, we need execution."
        patterns = insights_handler._extract_patterns_from_content(content)

        types = [p["type"] for p in patterns]
        assert "enumerated_argument" in types

    def test_extract_pattern_conditional_reasoning(self, insights_handler):
        """Should detect conditional reasoning pattern."""
        content = "If we invest now, then we will see returns later."
        patterns = insights_handler._extract_patterns_from_content(content)

        types = [p["type"] for p in patterns]
        assert "conditional_reasoning" in types

    def test_extract_pattern_causal_reasoning(self, insights_handler):
        """Should detect causal reasoning pattern."""
        content = "This works because it is simple. It succeeds because users like it."
        patterns = insights_handler._extract_patterns_from_content(content)

        types = [p["type"] for p in patterns]
        assert "causal_reasoning" in types

        # Should track instance count
        causal = next(p for p in patterns if p["type"] == "causal_reasoning")
        assert causal["instances"] == 2

    def test_extract_pattern_multiple(self, insights_handler):
        """Should detect multiple patterns."""
        content = """
        On one hand, caching is fast. On the other hand, it uses memory.
        If we enable caching, then performance improves.
        This works because caching reduces database load.
        """
        patterns = insights_handler._extract_patterns_from_content(content)

        types = [p["type"] for p in patterns]
        assert "balanced_comparison" in types
        assert "conditional_reasoning" in types
        assert "causal_reasoning" in types


class TestIntegration:
    """Integration tests for full request flow."""

    def test_full_extract_flow(self, insights_handler):
        """Should handle full extraction flow."""
        content = """
        According to recent studies, machine learning improves accuracy.
        Therefore, we should adopt ML techniques because they are effective.
        On one hand, ML requires data. On the other hand, it provides automation.
        """

        mock_handler = Mock()
        mock_handler.rfile.read.return_value = json.dumps({
            "content": content,
            "debate_id": "test-debate-001",
        }).encode()
        mock_handler.headers = {"Content-Length": str(len(content) + 50)}

        result = insights_handler.handle_post(
            "/api/insights/extract-detailed", {}, mock_handler
        )

        assert result.status_code == 200
        data = json.loads(result.body)

        assert data["debate_id"] == "test-debate-001"
        assert len(data["claims"]) > 0
        assert len(data["evidence_chains"]) > 0
        assert len(data["patterns"]) > 0

    def test_selective_extraction(self, insights_handler):
        """Should respect extraction flags."""
        content = "Therefore, we should do this because it works."

        mock_handler = Mock()
        mock_handler.rfile.read.return_value = json.dumps({
            "content": content,
            "extract_claims": True,
            "extract_evidence": False,
            "extract_patterns": False,
        }).encode()
        mock_handler.headers = {"Content-Length": "200"}

        result = insights_handler.handle_post(
            "/api/insights/extract-detailed", {}, mock_handler
        )

        data = json.loads(result.body)
        assert "claims" in data
        assert "evidence_chains" not in data
        assert "patterns" not in data
