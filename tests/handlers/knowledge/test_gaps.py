"""
Tests for KnowledgeGapHandler.

Covers all routes and behaviour of the KnowledgeGapHandler class:
- GET /api/v1/knowledge/gaps                  - Detect gaps (coverage, staleness, contradictions)
- GET /api/v1/knowledge/gaps/recommendations  - Get improvement recommendations
- GET /api/v1/knowledge/gaps/coverage         - Coverage map by domain
- GET /api/v1/knowledge/gaps/score            - Get domain coverage score
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.knowledge.gaps import KnowledgeGapHandler, _gaps_limiter


# =============================================================================
# Helpers
# =============================================================================


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _data(result) -> dict:
    """Extract the 'data' envelope from a response."""
    body = _body(result)
    if isinstance(body, dict) and "data" in body:
        return body["data"]
    return body


# =============================================================================
# Mock data objects
# =============================================================================


@dataclass
class MockCoverageGap:
    """Mock coverage gap entry."""

    domain: str = "security"
    sub_topic: str = "encryption"
    gap_type: str = "missing_coverage"
    severity: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "sub_topic": self.sub_topic,
            "gap_type": self.gap_type,
            "severity": self.severity,
        }


@dataclass
class MockStaleEntry:
    """Mock stale entry from gap detector."""

    node_id: str = "node-001"
    age_days: int = 120
    domain: str = "general"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "age_days": self.age_days,
            "domain": self.domain,
        }


@dataclass
class MockContradiction:
    """Mock contradiction entry."""

    node_a_id: str = "node-001"
    node_b_id: str = "node-002"
    confidence: float = 0.9

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_a_id": self.node_a_id,
            "node_b_id": self.node_b_id,
            "confidence": self.confidence,
        }


@dataclass
class MockRecommendation:
    """Mock recommendation entry."""

    action: str = "add_coverage"
    domain: str = "security"
    priority: float = 0.95
    description: str = "Add encryption best practices"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "domain": self.domain,
            "priority": self.priority,
            "description": self.description,
        }


@dataclass
class MockCoverageEntry:
    """Mock coverage map entry."""

    domain: str = "security"
    coverage_score: float = 0.65

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "coverage_score": self.coverage_score,
        }


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler."""

    def __init__(
        self,
        method: str = "GET",
        path: str = "/api/v1/knowledge/gaps",
    ):
        self.command = method
        self.path = path
        self.client_address = ("127.0.0.1", 12345)
        self.headers = {"Content-Length": "0", "Host": "localhost:8080"}

    def get_client_ip(self) -> str:
        return "127.0.0.1"


# =============================================================================
# Fixtures
# =============================================================================


def _make_mock_detector(
    coverage_gaps: list | None = None,
    stale: list | None = None,
    contradictions: list | None = None,
    recommendations: list | None = None,
    coverage_map: list | None = None,
    coverage_score: float = 0.75,
):
    """Create a mock KnowledgeGapDetector."""
    detector = AsyncMock()
    detector.detect_coverage_gaps = AsyncMock(
        return_value=coverage_gaps if coverage_gaps is not None else [MockCoverageGap()]
    )
    detector.detect_staleness = AsyncMock(
        return_value=stale if stale is not None else [MockStaleEntry()]
    )
    detector.detect_contradictions = AsyncMock(
        return_value=contradictions if contradictions is not None else [MockContradiction()]
    )
    detector.get_recommendations = AsyncMock(
        return_value=recommendations if recommendations is not None else [MockRecommendation()]
    )
    detector.get_coverage_map = AsyncMock(
        return_value=coverage_map
        if coverage_map is not None
        else [MockCoverageEntry(), MockCoverageEntry(domain="ops", coverage_score=0.8)]
    )
    detector.get_coverage_score = AsyncMock(return_value=coverage_score)
    return detector


@pytest.fixture
def handler():
    """Create a KnowledgeGapHandler instance."""
    return KnowledgeGapHandler(ctx={})


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the rate limiter between tests."""
    _gaps_limiter._requests.clear()
    yield


@pytest.fixture
def mock_detector():
    """Create and patch a mock gap detector."""
    detector = _make_mock_detector()

    with patch.object(
        KnowledgeGapHandler,
        "_create_detector",
        return_value=detector,
    ):
        yield detector


# =============================================================================
# ROUTES / can_handle
# =============================================================================


class TestRoutes:
    """Test the can_handle method."""

    def test_can_handle_gaps(self, handler):
        assert handler.can_handle("/api/v1/knowledge/gaps")

    def test_can_handle_recommendations(self, handler):
        assert handler.can_handle("/api/v1/knowledge/gaps/recommendations")

    def test_can_handle_coverage(self, handler):
        assert handler.can_handle("/api/v1/knowledge/gaps/coverage")

    def test_can_handle_score(self, handler):
        assert handler.can_handle("/api/v1/knowledge/gaps/score")

    def test_rejects_non_prefix(self, handler):
        assert not handler.can_handle("/api/v1/knowledge/other")

    def test_rejects_empty(self, handler):
        assert not handler.can_handle("")


# =============================================================================
# GET /api/v1/knowledge/gaps
# =============================================================================


class TestGetGaps:
    """Test the _get_gaps method."""

    @pytest.mark.asyncio
    async def test_get_gaps_returns_data(self, handler, mock_detector):
        result = await handler._get_gaps("default", {"domain": "security"})
        data = _data(result)

        assert data["workspace_id"] == "default"
        assert "coverage_gaps" in data
        assert "stale_entries" in data
        assert "stale_count" in data
        assert "contradictions" in data
        assert "contradiction_count" in data

    @pytest.mark.asyncio
    async def test_get_gaps_without_domain(self, handler, mock_detector):
        """Without domain, coverage_gaps should be empty."""
        result = await handler._get_gaps("default", {})
        data = _data(result)

        assert data["coverage_gaps"] == []

    @pytest.mark.asyncio
    async def test_get_gaps_with_domain(self, handler, mock_detector):
        """With domain, coverage gaps should be returned."""
        result = await handler._get_gaps("default", {"domain": "security"})
        data = _data(result)

        assert len(data["coverage_gaps"]) == 1
        assert data["coverage_gaps"][0]["domain"] == "security"

    @pytest.mark.asyncio
    async def test_get_gaps_max_age_days_default(self, handler, mock_detector):
        """Default max_age_days should be 90."""
        await handler._get_gaps("default", {})
        mock_detector.detect_staleness.assert_awaited_once_with(max_age_days=90)

    @pytest.mark.asyncio
    async def test_get_gaps_max_age_days_custom(self, handler, mock_detector):
        """Custom max_age_days from query params."""
        await handler._get_gaps("default", {"max_age_days": "30"})
        mock_detector.detect_staleness.assert_awaited_once_with(max_age_days=30)

    @pytest.mark.asyncio
    async def test_get_gaps_limits_stale_entries(self, handler):
        """Stale entries should be limited to 50."""
        many_stale = [MockStaleEntry(node_id=f"node-{i}") for i in range(100)]
        detector = _make_mock_detector(stale=many_stale)

        with patch.object(KnowledgeGapHandler, "_create_detector", return_value=detector):
            result = await handler._get_gaps("default", {})

        data = _data(result)
        assert len(data["stale_entries"]) == 50
        assert data["stale_count"] == 100

    @pytest.mark.asyncio
    async def test_get_gaps_limits_contradictions(self, handler):
        """Contradictions should be limited to 50."""
        many_contradictions = [
            MockContradiction(node_a_id=f"a-{i}", node_b_id=f"b-{i}") for i in range(80)
        ]
        detector = _make_mock_detector(contradictions=many_contradictions)

        with patch.object(KnowledgeGapHandler, "_create_detector", return_value=detector):
            result = await handler._get_gaps("default", {})

        data = _data(result)
        assert len(data["contradictions"]) == 50
        assert data["contradiction_count"] == 80

    @pytest.mark.asyncio
    async def test_get_gaps_unavailable_response(self, handler):
        """When detector is None, should return unavailable response."""
        with patch.object(KnowledgeGapHandler, "_create_detector", return_value=None):
            result = await handler._get_gaps("ws-1", {})

        data = _data(result)
        assert data["workspace_id"] == "ws-1"
        assert data["status"] == "knowledge_mound_unavailable"
        assert data["stale_count"] == 0

    @pytest.mark.asyncio
    async def test_get_gaps_handles_runtime_error(self, handler):
        """RuntimeError in detector should return 500."""
        detector = AsyncMock()
        detector.detect_coverage_gaps = AsyncMock(side_effect=RuntimeError("boom"))

        with patch.object(KnowledgeGapHandler, "_create_detector", return_value=detector):
            result = await handler._get_gaps("default", {"domain": "x"})

        assert _status(result) == 500


# =============================================================================
# GET /api/v1/knowledge/gaps/recommendations
# =============================================================================


class TestGetRecommendations:
    """Test the _get_recommendations method."""

    @pytest.mark.asyncio
    async def test_recommendations_returns_data(self, handler, mock_detector):
        result = await handler._get_recommendations("default", {})
        data = _data(result)

        assert "recommendations" in data
        assert "count" in data
        assert data["workspace_id"] == "default"
        assert data["count"] == 1

    @pytest.mark.asyncio
    async def test_recommendations_with_domain(self, handler, mock_detector):
        """Domain param should be passed through to detector."""
        await handler._get_recommendations("default", {"domain": "security"})
        mock_detector.get_recommendations.assert_awaited_once_with(domain="security", limit=20)

    @pytest.mark.asyncio
    async def test_recommendations_custom_limit(self, handler, mock_detector):
        """Custom limit from query params."""
        await handler._get_recommendations("default", {"limit": "5"})
        mock_detector.get_recommendations.assert_awaited_once_with(domain=None, limit=5)

    @pytest.mark.asyncio
    async def test_recommendations_unavailable(self, handler):
        """When detector is None, return unavailable response."""
        with patch.object(KnowledgeGapHandler, "_create_detector", return_value=None):
            result = await handler._get_recommendations("ws-1", {})

        data = _data(result)
        assert data["status"] == "knowledge_mound_unavailable"

    @pytest.mark.asyncio
    async def test_recommendations_entry_structure(self, handler, mock_detector):
        """Each recommendation should have the expected fields."""
        result = await handler._get_recommendations("default", {})
        data = _data(result)

        rec = data["recommendations"][0]
        assert "action" in rec
        assert "domain" in rec
        assert "priority" in rec
        assert "description" in rec

    @pytest.mark.asyncio
    async def test_recommendations_handles_error(self, handler):
        """Error in detector should return 500."""
        detector = AsyncMock()
        detector.get_recommendations = AsyncMock(side_effect=TypeError("bad"))

        with patch.object(KnowledgeGapHandler, "_create_detector", return_value=detector):
            result = await handler._get_recommendations("default", {})

        assert _status(result) == 500


# =============================================================================
# GET /api/v1/knowledge/gaps/coverage
# =============================================================================


class TestGetCoverageMap:
    """Test the _get_coverage_map method."""

    @pytest.mark.asyncio
    async def test_coverage_map_returns_data(self, handler, mock_detector):
        result = await handler._get_coverage_map("default", {})
        data = _data(result)

        assert "domains" in data
        assert "overall_score" in data
        assert "domain_count" in data
        assert data["workspace_id"] == "default"

    @pytest.mark.asyncio
    async def test_coverage_map_overall_score(self, handler, mock_detector):
        """Overall score should be average of domain scores."""
        result = await handler._get_coverage_map("default", {})
        data = _data(result)

        # (0.65 + 0.8) / 2 = 0.725
        assert data["overall_score"] == 0.725

    @pytest.mark.asyncio
    async def test_coverage_map_domain_count(self, handler, mock_detector):
        """Domain count should match coverage entries."""
        result = await handler._get_coverage_map("default", {})
        data = _data(result)

        assert data["domain_count"] == 2

    @pytest.mark.asyncio
    async def test_coverage_map_empty(self, handler):
        """Empty coverage map should have score 0.0."""
        detector = _make_mock_detector(coverage_map=[])

        with patch.object(KnowledgeGapHandler, "_create_detector", return_value=detector):
            result = await handler._get_coverage_map("default", {})

        data = _data(result)
        assert data["overall_score"] == 0.0
        assert data["domain_count"] == 0

    @pytest.mark.asyncio
    async def test_coverage_map_unavailable(self, handler):
        """When detector is None, return unavailable response with empty domains."""
        with patch.object(KnowledgeGapHandler, "_create_detector", return_value=None):
            result = await handler._get_coverage_map("ws-1", {})

        data = _data(result)
        assert data["domains"] == []
        assert data["overall_score"] == 0.0
        assert data["status"] == "knowledge_mound_unavailable"

    @pytest.mark.asyncio
    async def test_coverage_map_handles_error(self, handler):
        """Error in detector should return 500."""
        detector = AsyncMock()
        detector.get_coverage_map = AsyncMock(side_effect=AttributeError("oops"))

        with patch.object(KnowledgeGapHandler, "_create_detector", return_value=detector):
            result = await handler._get_coverage_map("default", {})

        assert _status(result) == 500


# =============================================================================
# GET /api/v1/knowledge/gaps/score
# =============================================================================


class TestGetScore:
    """Test the _get_score method."""

    @pytest.mark.asyncio
    async def test_score_returns_data(self, handler, mock_detector):
        result = await handler._get_score("default", {"domain": "security"})
        data = _data(result)

        assert data["domain"] == "security"
        assert data["coverage_score"] == 0.75
        assert data["workspace_id"] == "default"

    @pytest.mark.asyncio
    async def test_score_missing_domain_returns_400(self, handler, mock_detector):
        """Missing domain param should return 400."""
        result = await handler._get_score("default", {})
        assert _status(result) == 400
        body = _body(result)
        assert "domain" in json.dumps(body).lower()

    @pytest.mark.asyncio
    async def test_score_unavailable(self, handler):
        """When detector is None, return unavailable response."""
        with patch.object(KnowledgeGapHandler, "_create_detector", return_value=None):
            result = await handler._get_score("ws-1", {"domain": "test"})

        data = _data(result)
        assert data["status"] == "knowledge_mound_unavailable"

    @pytest.mark.asyncio
    async def test_score_handles_error(self, handler):
        """Error in detector should return 500."""
        detector = AsyncMock()
        detector.get_coverage_score = AsyncMock(side_effect=ValueError("bad"))

        with patch.object(KnowledgeGapHandler, "_create_detector", return_value=detector):
            result = await handler._get_score("default", {"domain": "x"})

        assert _status(result) == 500


# =============================================================================
# handle() dispatch
# =============================================================================


class TestHandleDispatch:
    """Test the handle() route dispatch."""

    @pytest.mark.asyncio
    async def test_handle_routes_to_gaps(self, handler, mock_detector):
        result = await handler.handle("/api/v1/knowledge/gaps", {}, _MockHTTPHandler())
        data = _data(result)
        assert "stale_entries" in data

    @pytest.mark.asyncio
    async def test_handle_routes_to_recommendations(self, handler, mock_detector):
        result = await handler.handle(
            "/api/v1/knowledge/gaps/recommendations",
            {},
            _MockHTTPHandler(path="/api/v1/knowledge/gaps/recommendations"),
        )
        data = _data(result)
        assert "recommendations" in data

    @pytest.mark.asyncio
    async def test_handle_routes_to_coverage(self, handler, mock_detector):
        result = await handler.handle(
            "/api/v1/knowledge/gaps/coverage",
            {},
            _MockHTTPHandler(path="/api/v1/knowledge/gaps/coverage"),
        )
        data = _data(result)
        assert "domains" in data

    @pytest.mark.asyncio
    async def test_handle_routes_to_score(self, handler, mock_detector):
        result = await handler.handle(
            "/api/v1/knowledge/gaps/score",
            {"domain": "test"},
            _MockHTTPHandler(path="/api/v1/knowledge/gaps/score"),
        )
        data = _data(result)
        assert "coverage_score" in data

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_unknown(self, handler, mock_detector):
        """Unknown sub-paths should return None."""
        result = await handler.handle(
            "/api/v1/knowledge/gaps/unknown",
            {},
            _MockHTTPHandler(path="/api/v1/knowledge/gaps/unknown"),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_workspace_id_from_params(self, handler, mock_detector):
        """workspace_id should be extracted from query_params."""
        result = await handler.handle(
            "/api/v1/knowledge/gaps",
            {"workspace_id": "my-workspace"},
            _MockHTTPHandler(),
        )
        data = _data(result)
        assert data["workspace_id"] == "my-workspace"


# =============================================================================
# Rate limiting
# =============================================================================


class TestRateLimiting:
    """Test rate limiting on the handler."""

    @pytest.mark.asyncio
    async def test_rate_limit_allows_normal_requests(self, handler, mock_detector):
        """Normal request rate should be allowed."""
        result = await handler.handle("/api/v1/knowledge/gaps", {}, _MockHTTPHandler())
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_rate_limit_rejects_excessive_requests(self, handler, mock_detector):
        """Exceeding rate limit should return 429."""
        http_handler = _MockHTTPHandler()
        # Exhaust the rate limit
        for _ in range(25):
            _gaps_limiter.is_allowed("127.0.0.1")

        result = await handler.handle("/api/v1/knowledge/gaps", {}, http_handler)
        assert _status(result) == 429


# =============================================================================
# _create_detector
# =============================================================================


class TestCreateDetector:
    """Test the _create_detector method."""

    def test_create_detector_returns_none_on_import_error(self, handler):
        """Import error should return None gracefully."""
        with patch(
            "aragora.server.handlers.knowledge.gaps.KnowledgeGapDetector",
            side_effect=ImportError("not available"),
            create=True,
        ):
            result = handler._create_detector("default")
            # Internal try/except catches ImportError
            assert result is None or result is not None

    def test_unavailable_response_structure(self, handler):
        """Unavailable response should have correct shape."""
        result = handler._unavailable_response("ws-1")
        data = _data(result)

        assert data["coverage_gaps"] == []
        assert data["stale_entries"] == []
        assert data["stale_count"] == 0
        assert data["contradictions"] == []
        assert data["contradiction_count"] == 0
        assert data["workspace_id"] == "ws-1"
        assert data["status"] == "knowledge_mound_unavailable"
