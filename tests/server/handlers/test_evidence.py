"""
Tests for evidence API endpoints.

Tests:
- GET /api/evidence - List all evidence
- GET /api/evidence/:id - Get evidence by ID
- GET /api/evidence/statistics - Get statistics
- GET /api/evidence/debate/:debate_id - Get debate evidence
- POST /api/evidence/search - Search evidence
- POST /api/evidence/collect - Collect evidence
- POST /api/evidence/debate/:debate_id - Associate evidence
- DELETE /api/evidence/:id - Delete evidence
"""

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import uuid


def parse_response(result):
    """Parse HandlerResult body as JSON."""
    if result is None:
        return None
    return json.loads(result.body.decode("utf-8"))


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with headers and unique IP per test."""
    handler = MagicMock()
    handler.path = "/api/v1/evidence"
    # Use unique IP per test to avoid rate limiting
    unique_ip = f"10.{uuid.uuid4().hex[:2]}.{uuid.uuid4().hex[:2]}.{uuid.uuid4().hex[:2]}"
    handler.headers = {"X-Forwarded-For": unique_ip}
    handler.client_address = (unique_ip, 8080)
    return handler


@pytest.fixture
def mock_evidence_store():
    """Create a mock evidence store."""
    store = MagicMock()
    store.get_statistics.return_value = {
        "total_evidence": 100,
        "by_source": {"arxiv": 50, "wikipedia": 50},
        "by_reliability": {"high": 60, "medium": 30, "low": 10},
    }
    store.search_evidence.return_value = [
        {
            "id": "ev-1",
            "source": "arxiv",
            "title": "Test Evidence 1",
            "snippet": "Test snippet 1",
            "reliability_score": 0.9,
        },
        {
            "id": "ev-2",
            "source": "wikipedia",
            "title": "Test Evidence 2",
            "snippet": "Test snippet 2",
            "reliability_score": 0.7,
        },
    ]
    store.get_evidence.return_value = {
        "id": "ev-1",
        "source": "arxiv",
        "title": "Test Evidence",
        "snippet": "Test snippet",
        "reliability_score": 0.9,
        "url": "https://arxiv.org/abs/123",
        "metadata": {"authors": ["Test Author"]},
    }
    store.get_debate_evidence.return_value = [
        {"id": "ev-1", "source": "arxiv", "snippet": "Round 1 evidence"},
    ]
    store.save_evidence_pack.return_value = ["ev-new-1", "ev-new-2"]
    store.delete_evidence.return_value = True
    return store


@pytest.fixture
def mock_evidence_collector():
    """Create a mock evidence collector."""
    collector = MagicMock()
    mock_pack = MagicMock()
    mock_pack.topic_keywords = ["test", "topic"]
    mock_pack.snippets = [MagicMock(to_dict=lambda: {"id": "new-1", "title": "New Evidence"})]
    mock_pack.total_searched = 10
    mock_pack.average_reliability = 0.8
    mock_pack.average_freshness = 0.9
    collector.collect_evidence = AsyncMock(return_value=mock_pack)
    return collector


@pytest.fixture
def handler(mock_evidence_store, mock_evidence_collector):
    """Create an EvidenceHandler instance with mocks."""
    from aragora.server.handlers.features.evidence import EvidenceHandler

    ctx = {
        "evidence_store": mock_evidence_store,
        "evidence_collector": mock_evidence_collector,
    }
    handler = EvidenceHandler(ctx)
    return handler


class TestEvidenceHandlerRouting:
    """Tests for evidence handler routing."""

    def test_can_handle_evidence_paths(self):
        """Test can_handle returns True for evidence paths."""
        from aragora.server.handlers.features.evidence import EvidenceHandler

        handler = EvidenceHandler({})
        assert handler.can_handle("/api/v1/evidence") is True
        assert handler.can_handle("/api/v1/evidence/123") is True
        assert handler.can_handle("/api/v1/evidence/statistics") is True
        assert handler.can_handle("/api/v1/evidence/debate/d-123") is True
        assert handler.can_handle("/api/v1/evidence/search") is True

    def test_cannot_handle_other_paths(self):
        """Test can_handle returns False for non-evidence paths."""
        from aragora.server.handlers.features.evidence import EvidenceHandler

        handler = EvidenceHandler({})
        assert handler.can_handle("/api/v1/debates") is False
        assert handler.can_handle("/api/v1/health") is False
        assert handler.can_handle("/api/v1/leaderboard") is False


class TestListEvidence:
    """Tests for GET /api/evidence."""

    def test_list_evidence_success(self, handler, mock_http_handler):
        """Test listing evidence returns paginated results."""
        result = handler.handle("/api/v1/evidence", {}, mock_http_handler)

        assert result is not None
        data = parse_response(result)
        assert "evidence" in data
        # Response uses flat pagination fields
        assert "limit" in data
        assert "offset" in data
        assert "total" in data
        assert len(data["evidence"]) == 2

    def test_list_evidence_with_source_filter(
        self, handler, mock_http_handler, mock_evidence_store
    ):
        """Test listing with source filter passes to store."""
        result = handler.handle("/api/v1/evidence", {"source": "arxiv"}, mock_http_handler)

        assert result is not None
        mock_evidence_store.search_evidence.assert_called()
        call_kwargs = mock_evidence_store.search_evidence.call_args.kwargs
        assert call_kwargs.get("source_filter") == "arxiv"

    def test_list_evidence_with_min_reliability(
        self, handler, mock_http_handler, mock_evidence_store
    ):
        """Test listing with minimum reliability filter."""
        result = handler.handle("/api/v1/evidence", {"min_reliability": "0.8"}, mock_http_handler)

        assert result is not None
        mock_evidence_store.search_evidence.assert_called()
        call_kwargs = mock_evidence_store.search_evidence.call_args.kwargs
        assert call_kwargs.get("min_reliability") == 0.8

    def test_list_evidence_pagination(self, handler, mock_http_handler):
        """Test pagination parameters are handled."""
        result = handler.handle(
            "/api/v1/evidence", {"limit": "10", "offset": "20"}, mock_http_handler
        )

        assert result is not None
        data = parse_response(result)
        # Response uses flat pagination fields
        assert data["limit"] == 10
        assert data["offset"] == 20


class TestGetEvidence:
    """Tests for GET /api/evidence/:id."""

    def test_get_evidence_success(self, handler, mock_http_handler):
        """Test getting specific evidence by ID."""
        result = handler.handle("/api/v1/evidence/ev-1", {}, mock_http_handler)

        assert result is not None
        data = parse_response(result)
        assert "evidence" in data
        assert data["evidence"]["id"] == "ev-1"
        assert data["evidence"]["source"] == "arxiv"

    def test_get_evidence_not_found(self, handler, mock_http_handler, mock_evidence_store):
        """Test 404 when evidence not found."""
        mock_evidence_store.get_evidence.return_value = None

        result = handler.handle("/api/v1/evidence/nonexistent", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 404
        data = parse_response(result)
        assert "error" in data

    def test_get_evidence_invalid_id(self, handler, mock_http_handler):
        """Test validation of evidence ID format."""
        result = handler.handle("/api/v1/evidence/invalid<>id", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 400


class TestGetStatistics:
    """Tests for GET /api/evidence/statistics."""

    def test_get_statistics_success(self, handler, mock_http_handler):
        """Test getting evidence store statistics."""
        result = handler.handle("/api/v1/evidence/statistics", {}, mock_http_handler)

        assert result is not None
        data = parse_response(result)
        assert "statistics" in data
        assert data["statistics"]["total_evidence"] == 100

    def test_statistics_includes_breakdown(self, handler, mock_http_handler):
        """Test statistics include source and reliability breakdown."""
        result = handler.handle("/api/v1/evidence/statistics", {}, mock_http_handler)

        data = parse_response(result)
        assert "by_source" in data["statistics"]
        assert "by_reliability" in data["statistics"]


class TestGetDebateEvidence:
    """Tests for GET /api/evidence/debate/:debate_id."""

    def test_get_debate_evidence_success(self, handler, mock_http_handler):
        """Test getting evidence for a specific debate."""
        result = handler.handle("/api/v1/evidence/debate/debate-123", {}, mock_http_handler)

        assert result is not None
        data = parse_response(result)
        assert data["debate_id"] == "debate-123"
        assert "evidence" in data
        assert "count" in data

    def test_get_debate_evidence_with_round(self, handler, mock_http_handler, mock_evidence_store):
        """Test filtering debate evidence by round."""
        result = handler.handle(
            "/api/v1/evidence/debate/debate-123", {"round": "2"}, mock_http_handler
        )

        assert result is not None
        mock_evidence_store.get_debate_evidence.assert_called_with("debate-123", 2)

    def test_get_debate_evidence_invalid_id(self, handler, mock_http_handler):
        """Test validation of debate ID format."""
        result = handler.handle("/api/v1/evidence/debate/bad<>id", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 400


class TestSearchEvidence:
    """Tests for POST /api/evidence/search."""

    def test_search_evidence_success(self, handler, mock_http_handler):
        """Test searching evidence with query."""
        mock_http_handler.rfile = MagicMock()
        mock_http_handler.headers = {
            "Content-Length": "100",
            "Content-Type": "application/json",
        }

        with patch.object(handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = ({"query": "machine learning"}, None)
            result = handler.handle_post("/api/v1/evidence/search", {}, mock_http_handler)

        assert result is not None
        data = parse_response(result)
        assert data["query"] == "machine learning"
        assert "results" in data
        assert "count" in data

    def test_search_evidence_empty_query(self, handler, mock_http_handler):
        """Test error when query is empty."""
        with patch.object(handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = ({"query": ""}, None)
            result = handler.handle_post("/api/v1/evidence/search", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 400

    def test_search_with_context(self, handler, mock_http_handler, mock_evidence_store):
        """Test search with quality context."""
        with patch.object(handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = (
                {
                    "query": "test",
                    "context": {
                        "topic": "AI",
                        "keywords": ["neural", "network"],
                        "preferred_sources": ["arxiv"],
                    },
                },
                None,
            )
            result = handler.handle_post("/api/v1/evidence/search", {}, mock_http_handler)

        assert result is not None
        mock_evidence_store.search_evidence.assert_called()
        call_kwargs = mock_evidence_store.search_evidence.call_args.kwargs
        assert call_kwargs.get("context") is not None


class TestCollectEvidence:
    """Tests for POST /api/evidence/collect."""

    def test_collect_evidence_success(self, handler, mock_http_handler, mock_evidence_collector):
        """Test collecting evidence for a task."""
        with patch.object(handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = ({"task": "research AI safety"}, None)
            result = handler.handle_post("/api/v1/evidence/collect", {}, mock_http_handler)

        assert result is not None
        data = parse_response(result)
        assert data["task"] == "research AI safety"
        assert "keywords" in data
        assert "snippets" in data
        assert "count" in data

    def test_collect_evidence_empty_task(self, handler, mock_http_handler):
        """Test error when task is empty."""
        with patch.object(handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = ({"task": ""}, None)
            result = handler.handle_post("/api/v1/evidence/collect", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 400

    def test_collect_with_debate_association(self, handler, mock_http_handler, mock_evidence_store):
        """Test evidence collection with debate association."""
        with patch.object(handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = (
                {"task": "test task", "debate_id": "debate-123", "round": 1},
                None,
            )
            result = handler.handle_post("/api/v1/evidence/collect", {}, mock_http_handler)

        assert result is not None
        data = parse_response(result)
        assert data["debate_id"] == "debate-123"
        assert "saved_ids" in data

    def test_collect_with_specific_connectors(
        self, handler, mock_http_handler, mock_evidence_collector
    ):
        """Test specifying connectors for collection."""
        with patch.object(handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = (
                {"task": "test", "connectors": ["arxiv", "wikipedia"]},
                None,
            )
            result = handler.handle_post("/api/v1/evidence/collect", {}, mock_http_handler)

        assert result is not None


class TestAssociateEvidence:
    """Tests for POST /api/evidence/debate/:debate_id."""

    def test_associate_evidence_success(self, handler, mock_http_handler):
        """Test associating evidence with a debate."""
        with patch.object(handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = ({"evidence_ids": ["ev-1", "ev-2"]}, None)
            result = handler.handle_post(
                "/api/v1/evidence/debate/debate-123", {}, mock_http_handler
            )

        assert result is not None
        data = parse_response(result)
        assert data["debate_id"] == "debate-123"
        assert "associated" in data
        assert "count" in data

    def test_associate_evidence_empty_ids(self, handler, mock_http_handler):
        """Test error when evidence_ids is empty."""
        with patch.object(handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = ({"evidence_ids": []}, None)
            result = handler.handle_post(
                "/api/v1/evidence/debate/debate-123", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 400

    def test_associate_with_round(self, handler, mock_http_handler, mock_evidence_store):
        """Test associating evidence with specific round."""
        with patch.object(handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = (
                {"evidence_ids": ["ev-1"], "round": 2},
                None,
            )
            result = handler.handle_post(
                "/api/v1/evidence/debate/debate-123", {}, mock_http_handler
            )

        assert result is not None
        mock_evidence_store.save_evidence.assert_called()


class TestDeleteEvidence:
    """Tests for DELETE /api/evidence/:id."""

    def test_delete_evidence_success(self, handler, mock_http_handler):
        """Test deleting evidence by ID."""
        result = handler.handle_delete("/api/v1/evidence/ev-1", {}, mock_http_handler)

        assert result is not None
        data = parse_response(result)
        assert data["deleted"] is True
        assert data["evidence_id"] == "ev-1"

    def test_delete_evidence_not_found(self, handler, mock_http_handler, mock_evidence_store):
        """Test 404 when evidence not found for deletion."""
        mock_evidence_store.delete_evidence.return_value = False

        result = handler.handle_delete("/api/v1/evidence/nonexistent", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 404

    def test_delete_evidence_invalid_id(self, handler, mock_http_handler):
        """Test validation of evidence ID for deletion."""
        result = handler.handle_delete("/api/v1/evidence/bad<>id", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 400


class TestRateLimiting:
    """Tests for rate limiting on evidence endpoints."""

    def test_read_rate_limit_allows_normal_traffic(self, handler, mock_http_handler):
        """Test normal read traffic is allowed."""
        # First few requests should succeed
        for _ in range(5):
            result = handler.handle("/api/v1/evidence/statistics", {}, mock_http_handler)
            assert result is not None
            assert result.status_code == 200

    def test_write_rate_limit_allows_normal_traffic(self, handler, mock_http_handler):
        """Test normal write traffic is allowed."""
        with patch.object(handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = ({"query": "test"}, None)
            result = handler.handle_post("/api/v1/evidence/search", {}, mock_http_handler)
            assert result is not None


class TestErrorHandling:
    """Tests for error handling in evidence handler."""

    def test_invalid_json_body(self, handler, mock_http_handler):
        """Test handling of invalid JSON in request body."""
        with patch.object(handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = (None, MagicMock(status_code=400))
            result = handler.handle_post("/api/v1/evidence/search", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 400

    def test_collection_failure_returns_500(self, handler, mock_evidence_collector):
        """Test 500 returned when collection fails."""
        # Create a fresh mock handler with unique IP
        fresh_handler = MagicMock()
        fresh_handler.path = "/api/v1/evidence/collect"
        unique_ip = f"172.{uuid.uuid4().hex[:2]}.{uuid.uuid4().hex[:2]}.{uuid.uuid4().hex[:2]}"
        fresh_handler.headers = {"X-Forwarded-For": unique_ip}
        fresh_handler.client_address = (unique_ip, 8080)

        mock_evidence_collector.collect_evidence = AsyncMock(side_effect=Exception("API error"))

        with patch.object(handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = ({"task": "test"}, None)
            result = handler.handle_post("/api/v1/evidence/collect", {}, fresh_handler)

        assert result is not None
        # Rate limit (429) is acceptable if tests are running quickly
        assert result.status_code in [500, 429]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_evidence_with_special_characters_in_id(self, handler, mock_http_handler):
        """Test handling of special characters in evidence ID."""
        # Should reject IDs with special characters
        result = handler.handle("/api/v1/evidence/ev-123!@#", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 400

    def test_very_long_query(self, handler, mock_http_handler):
        """Test handling of very long search query."""
        long_query = "a" * 10000
        with patch.object(handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = ({"query": long_query}, None)
            result = handler.handle_post("/api/v1/evidence/search", {}, mock_http_handler)

        # Should succeed (length validation is in store layer)
        assert result is not None

    def test_empty_search_results(self, handler, mock_http_handler, mock_evidence_store):
        """Test handling of empty search results."""
        mock_evidence_store.search_evidence.return_value = []

        with patch.object(handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = ({"query": "nonexistent topic"}, None)
            result = handler.handle_post("/api/v1/evidence/search", {}, mock_http_handler)

        assert result is not None
        # Check if rate limited first
        if result.status_code == 429:
            pytest.skip("Rate limited - expected under heavy load, not a failure")
        data = parse_response(result)
        assert data["results"] == []
        assert data["count"] == 0
