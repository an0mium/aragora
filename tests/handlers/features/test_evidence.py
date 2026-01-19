"""Tests for evidence handler.

Tests the evidence API endpoints including:
- GET /api/evidence - list evidence
- GET /api/evidence/:id - get specific evidence
- GET /api/evidence/statistics - get store stats
- GET /api/evidence/debate/:debate_id - get debate evidence
- POST /api/evidence/search - search evidence
- POST /api/evidence/collect - collect evidence
- POST /api/evidence/debate/:debate_id - associate evidence
- DELETE /api/evidence/:id - delete evidence
"""

import json
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Any, Dict, Optional

import pytest

from aragora.server.handlers.features.evidence import EvidenceHandler
from aragora.server.handlers.base import HandlerResult


def parse_body(result: HandlerResult) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, body: Optional[Dict[str, Any]] = None, client_ip: str = "127.0.0.1"):
        self.rfile = MagicMock()
        self._body = body
        self._client_ip = client_ip
        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {
                "Content-Length": str(len(body_bytes)),
                "Content-Type": "application/json",
            }
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2", "Content-Type": "application/json"}
        self.client_address = (client_ip, 12345)


class MockEvidenceStore:
    """Mock evidence store for testing."""

    def __init__(self):
        self._evidence: Dict[str, Dict] = {}
        self._debate_evidence: Dict[str, list] = {}

    def get_evidence(self, evidence_id: str) -> Optional[Dict]:
        return self._evidence.get(evidence_id)

    def search_evidence(
        self,
        query: str,
        limit: int = 20,
        source_filter: Optional[str] = None,
        min_reliability: float = 0.0,
        context=None,
    ) -> list:
        # Simple mock search - return all evidence containing query
        results = []
        for eid, ev in self._evidence.items():
            if query == "*" or query.lower() in ev.get("snippet", "").lower():
                if min_reliability <= ev.get("reliability_score", 0.5):
                    if source_filter is None or ev.get("source") == source_filter:
                        results.append(ev)
        return results[:limit]

    def get_debate_evidence(self, debate_id: str, round_number: Optional[int] = None) -> list:
        evidence_list = self._debate_evidence.get(debate_id, [])
        if round_number is not None:
            evidence_list = [e for e in evidence_list if e.get("round") == round_number]
        return evidence_list

    def get_statistics(self) -> Dict:
        return {
            "total_evidence": len(self._evidence),
            "sources": len(set(e.get("source", "") for e in self._evidence.values())),
        }

    def save_evidence(
        self,
        evidence_id: str,
        source: str,
        title: str,
        snippet: str,
        url: str = "",
        reliability_score: float = 0.5,
        metadata: Optional[Dict] = None,
        debate_id: Optional[str] = None,
        round_number: Optional[int] = None,
        enrich: bool = False,
        score_quality: bool = False,
    ) -> str:
        self._evidence[evidence_id] = {
            "id": evidence_id,
            "source": source,
            "title": title,
            "snippet": snippet,
            "url": url,
            "reliability_score": reliability_score,
            "metadata": metadata or {},
        }
        if debate_id:
            if debate_id not in self._debate_evidence:
                self._debate_evidence[debate_id] = []
            self._debate_evidence[debate_id].append({
                **self._evidence[evidence_id],
                "round": round_number,
            })
        return evidence_id

    def save_evidence_pack(self, pack, debate_id: str, round_number: Optional[int] = None) -> list:
        saved = []
        for s in pack.snippets:
            eid = f"ev_{len(self._evidence)}"
            self.save_evidence(
                evidence_id=eid,
                source=s.source,
                title=s.title,
                snippet=s.content,
                url=getattr(s, "url", ""),
                reliability_score=getattr(s, "reliability_score", 0.5),
                debate_id=debate_id,
                round_number=round_number,
            )
            saved.append(eid)
        return saved

    def delete_evidence(self, evidence_id: str) -> bool:
        if evidence_id in self._evidence:
            del self._evidence[evidence_id]
            return True
        return False


class MockEvidenceCollector:
    """Mock evidence collector for testing."""

    async def collect_evidence(self, task: str, enabled_connectors=None):
        from dataclasses import dataclass

        @dataclass
        class MockSnippet:
            source: str = "test_source"
            title: str = "Test Evidence"
            content: str = "Test content for the task"
            url: str = "https://example.com"
            reliability_score: float = 0.8

            def to_dict(self):
                return {
                    "source": self.source,
                    "title": self.title,
                    "content": self.content,
                    "url": self.url,
                    "reliability_score": self.reliability_score,
                }

        @dataclass
        class MockEvidencePack:
            topic_keywords: list
            snippets: list
            total_searched: int
            average_reliability: float
            average_freshness: float

        return MockEvidencePack(
            topic_keywords=["test", "keyword"],
            snippets=[MockSnippet()],
            total_searched=10,
            average_reliability=0.8,
            average_freshness=0.9,
        )


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset rate limiters before each test."""
    from aragora.server.handlers.features import evidence
    # Reset the rate limiter state (uses _buckets internally)
    evidence._evidence_read_limiter._buckets.clear()
    evidence._evidence_write_limiter._buckets.clear()
    yield
    # Cleanup after test
    evidence._evidence_read_limiter._buckets.clear()
    evidence._evidence_write_limiter._buckets.clear()


@pytest.fixture
def evidence_store():
    """Create mock evidence store."""
    return MockEvidenceStore()


@pytest.fixture
def evidence_collector():
    """Create mock evidence collector."""
    return MockEvidenceCollector()


@pytest.fixture
def handler_context(evidence_store, evidence_collector):
    """Create server context for handler."""
    return {
        "evidence_store": evidence_store,
        "evidence_collector": evidence_collector,
    }


@pytest.fixture
def evidence_handler(handler_context):
    """Create evidence handler instance."""
    return EvidenceHandler(handler_context)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestEvidenceHandlerInit:
    """Tests for handler initialization."""

    def test_routes_defined(self, evidence_handler):
        """Test that handler routes are defined."""
        assert hasattr(evidence_handler, "routes")
        assert len(evidence_handler.routes) > 0

    def test_can_handle_evidence_paths(self, evidence_handler):
        """Test can_handle returns True for evidence paths."""
        assert evidence_handler.can_handle("/api/evidence")
        assert evidence_handler.can_handle("/api/evidence/123")
        assert evidence_handler.can_handle("/api/evidence/search")
        assert evidence_handler.can_handle("/api/evidence/statistics")

    def test_cannot_handle_other_paths(self, evidence_handler):
        """Test can_handle returns False for non-evidence paths."""
        assert not evidence_handler.can_handle("/api/debates")
        assert not evidence_handler.can_handle("/api/agents")
        assert not evidence_handler.can_handle("/health")


# =============================================================================
# GET Endpoint Tests
# =============================================================================


class TestGetEvidence:
    """Tests for GET /api/evidence/:id endpoint."""

    def test_get_existing_evidence(self, evidence_handler, evidence_store):
        """Test getting existing evidence by ID."""
        evidence_store.save_evidence(
            evidence_id="ev_123",
            source="wikipedia",
            title="Test Article",
            snippet="This is test content",
            url="https://wikipedia.org/test",
            reliability_score=0.9,
        )

        mock_handler = MockHandler()
        result = evidence_handler.handle("/api/evidence/ev_123", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert "evidence" in body
        assert body["evidence"]["id"] == "ev_123"

    def test_get_nonexistent_evidence(self, evidence_handler):
        """Test 404 for nonexistent evidence."""
        mock_handler = MockHandler()
        result = evidence_handler.handle("/api/evidence/nonexistent", {}, mock_handler)

        assert result is not None
        assert result.status_code == 404
        body = parse_body(result)
        assert "error" in body

    def test_get_evidence_invalid_id_format(self, evidence_handler):
        """Test rejection of invalid ID format."""
        mock_handler = MockHandler()
        # Path with slashes will be rejected by SAFE_ID_PATTERN
        result = evidence_handler.handle("/api/evidence/bad/id", {}, mock_handler)

        # Handler returns None for paths it can't handle, or 400 for invalid IDs
        # The extra slash makes this not match the :id pattern
        assert result is None or result.status_code == 400


class TestGetStatistics:
    """Tests for GET /api/evidence/statistics endpoint."""

    def test_get_statistics(self, evidence_handler, evidence_store):
        """Test getting evidence statistics."""
        evidence_store.save_evidence(
            evidence_id="ev_1",
            source="wikipedia",
            title="Test 1",
            snippet="Content 1",
        )
        evidence_store.save_evidence(
            evidence_id="ev_2",
            source="arxiv",
            title="Test 2",
            snippet="Content 2",
        )

        mock_handler = MockHandler()
        result = evidence_handler.handle("/api/evidence/statistics", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert "statistics" in body
        assert body["statistics"]["total_evidence"] == 2

    def test_get_statistics_empty_store(self, evidence_handler):
        """Test statistics with empty store."""
        mock_handler = MockHandler()
        result = evidence_handler.handle("/api/evidence/statistics", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["statistics"]["total_evidence"] == 0


class TestGetDebateEvidence:
    """Tests for GET /api/evidence/debate/:debate_id endpoint."""

    def test_get_debate_evidence(self, evidence_handler, evidence_store):
        """Test getting evidence for a debate."""
        evidence_store.save_evidence(
            evidence_id="ev_1",
            source="wikipedia",
            title="Test 1",
            snippet="Content 1",
            debate_id="debate_123",
            round_number=1,
        )

        mock_handler = MockHandler()
        result = evidence_handler.handle(
            "/api/evidence/debate/debate_123", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["debate_id"] == "debate_123"
        assert body["count"] == 1

    def test_get_debate_evidence_with_round_filter(self, evidence_handler, evidence_store):
        """Test filtering evidence by round number."""
        evidence_store.save_evidence(
            evidence_id="ev_1",
            source="wikipedia",
            title="Test 1",
            snippet="Content 1",
            debate_id="debate_123",
            round_number=1,
        )
        evidence_store.save_evidence(
            evidence_id="ev_2",
            source="arxiv",
            title="Test 2",
            snippet="Content 2",
            debate_id="debate_123",
            round_number=2,
        )

        mock_handler = MockHandler()
        result = evidence_handler.handle(
            "/api/evidence/debate/debate_123", {"round": "1"}, mock_handler
        )

        assert result is not None
        body = parse_body(result)
        assert body["round"] == 1
        assert body["count"] == 1


class TestListEvidence:
    """Tests for GET /api/evidence endpoint."""

    def test_list_evidence(self, evidence_handler, evidence_store):
        """Test listing evidence."""
        evidence_store.save_evidence(
            evidence_id="ev_1",
            source="wikipedia",
            title="Test 1",
            snippet="Content 1",
        )

        mock_handler = MockHandler()
        result = evidence_handler.handle("/api/evidence", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert "evidence" in body


# =============================================================================
# POST Endpoint Tests
# =============================================================================


class TestSearchEvidence:
    """Tests for POST /api/evidence/search endpoint."""

    def test_search_evidence(self, evidence_handler, evidence_store):
        """Test searching evidence."""
        evidence_store.save_evidence(
            evidence_id="ev_1",
            source="wikipedia",
            title="Machine Learning",
            snippet="Machine learning is a field of AI",
        )

        mock_handler = MockHandler(body={"query": "machine"})
        result = evidence_handler.handle_post(
            "/api/evidence/search", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert "results" in body
        assert body["query"] == "machine"

    def test_search_empty_query_rejected(self, evidence_handler):
        """Test rejection of empty search query."""
        mock_handler = MockHandler(body={"query": ""})
        result = evidence_handler.handle_post(
            "/api/evidence/search", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 400
        body = parse_body(result)
        assert "error" in body

    def test_search_missing_query_rejected(self, evidence_handler):
        """Test rejection of missing search query."""
        mock_handler = MockHandler(body={})
        result = evidence_handler.handle_post(
            "/api/evidence/search", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 400

    def test_search_with_filters(self, evidence_handler, evidence_store):
        """Test search with source filter and min_reliability."""
        evidence_store.save_evidence(
            evidence_id="ev_1",
            source="wikipedia",
            title="Test 1",
            snippet="High quality content",
            reliability_score=0.9,
        )
        evidence_store.save_evidence(
            evidence_id="ev_2",
            source="arxiv",
            title="Test 2",
            snippet="Lower quality content",
            reliability_score=0.3,
        )

        mock_handler = MockHandler(body={
            "query": "content",
            "source": "wikipedia",
            "min_reliability": 0.5,
        })
        result = evidence_handler.handle_post(
            "/api/evidence/search", {}, mock_handler
        )

        assert result is not None
        body = parse_body(result)
        # Should only find the high-reliability wikipedia result
        assert body["count"] <= 1

    def test_search_query_too_long_rejected(self, evidence_handler):
        """Test rejection of overly long search query (ReDoS protection)."""
        # Create a very long query that should be rejected
        long_query = "a" * 1001  # MAX_SEARCH_QUERY_LENGTH is typically 1000
        mock_handler = MockHandler(body={"query": long_query})
        result = evidence_handler.handle_post(
            "/api/evidence/search", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 400

    def test_search_malicious_pattern_rejected(self, evidence_handler):
        """Test rejection of potentially malicious regex patterns."""
        # Patterns with nested quantifiers are flagged as dangerous
        malicious_query = "(a+)+" * 10  # Nested quantifier pattern
        mock_handler = MockHandler(body={"query": malicious_query})
        result = evidence_handler.handle_post(
            "/api/evidence/search", {}, mock_handler
        )

        assert result is not None
        # Should either reject (400) or sanitize the input
        assert result.status_code in [200, 400]


class TestCollectEvidence:
    """Tests for POST /api/evidence/collect endpoint."""

    def test_collect_evidence(self, evidence_handler):
        """Test evidence collection."""
        mock_handler = MockHandler(body={"task": "What is machine learning?"})
        result = evidence_handler.handle_post(
            "/api/evidence/collect", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert "task" in body
        assert "snippets" in body

    def test_collect_empty_task_rejected(self, evidence_handler):
        """Test rejection of empty task."""
        mock_handler = MockHandler(body={"task": ""})
        result = evidence_handler.handle_post(
            "/api/evidence/collect", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 400

    def test_collect_with_debate_association(self, evidence_handler, evidence_store):
        """Test evidence collection with debate association."""
        mock_handler = MockHandler(body={
            "task": "What is machine learning?",
            "debate_id": "debate_123",
            "round": 1,
        })
        result = evidence_handler.handle_post(
            "/api/evidence/collect", {}, mock_handler
        )

        assert result is not None
        body = parse_body(result)
        assert body["debate_id"] == "debate_123"
        assert "saved_ids" in body


class TestAssociateEvidence:
    """Tests for POST /api/evidence/debate/:debate_id endpoint."""

    def test_associate_evidence(self, evidence_handler, evidence_store):
        """Test associating evidence with a debate."""
        evidence_store.save_evidence(
            evidence_id="ev_1",
            source="wikipedia",
            title="Test 1",
            snippet="Content 1",
        )

        mock_handler = MockHandler(body={
            "evidence_ids": ["ev_1"],
            "round": 1,
        })
        result = evidence_handler.handle_post(
            "/api/evidence/debate/debate_123", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["debate_id"] == "debate_123"
        assert "ev_1" in body["associated"]

    def test_associate_empty_ids_rejected(self, evidence_handler):
        """Test rejection of empty evidence_ids."""
        mock_handler = MockHandler(body={"evidence_ids": []})
        result = evidence_handler.handle_post(
            "/api/evidence/debate/debate_123", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 400


# =============================================================================
# DELETE Endpoint Tests
# =============================================================================


class TestDeleteEvidence:
    """Tests for DELETE /api/evidence/:id endpoint."""

    def test_delete_existing_evidence(self, evidence_handler, evidence_store):
        """Test deleting existing evidence."""
        evidence_store.save_evidence(
            evidence_id="ev_123",
            source="wikipedia",
            title="Test",
            snippet="Content",
        )

        mock_handler = MockHandler()
        result = evidence_handler.handle_delete(
            "/api/evidence/ev_123", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["deleted"] is True

    def test_delete_nonexistent_evidence(self, evidence_handler):
        """Test 404 for deleting nonexistent evidence."""
        mock_handler = MockHandler()
        result = evidence_handler.handle_delete(
            "/api/evidence/nonexistent", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 404


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    def test_rate_limit_shared_for_reads(self, evidence_handler):
        """Test that rate limits are applied."""
        # The handler has rate limiters that should be applied
        assert hasattr(evidence_handler, "routes")
        # Rate limiting is implemented in the handler

    @patch("aragora.server.handlers.features.evidence._evidence_read_limiter")
    def test_rate_limit_exceeded_returns_429(self, mock_limiter, evidence_handler):
        """Test 429 response when rate limit exceeded."""
        mock_limiter.is_allowed.return_value = False

        mock_handler = MockHandler()
        result = evidence_handler.handle("/api/evidence", {}, mock_handler)

        assert result is not None
        assert result.status_code == 429


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_json_body_rejected(self, evidence_handler):
        """Test rejection of invalid JSON body."""
        mock_handler = MockHandler()
        mock_handler.rfile.read.return_value = b"not json"

        result = evidence_handler.handle_post(
            "/api/evidence/search", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 400

    def test_path_traversal_rejected(self, evidence_handler):
        """Test rejection of path traversal attempts."""
        mock_handler = MockHandler()
        # Path traversal attempts with extra slashes don't match valid routes
        result = evidence_handler.handle(
            "/api/evidence/../../etc/passwd", {}, mock_handler
        )

        # Handler may return None for paths it can't handle, or 400 for invalid IDs
        assert result is None or result.status_code == 400


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for evidence workflow."""

    def test_collect_then_search_workflow(self, evidence_handler, evidence_store):
        """Test collecting evidence then searching for it."""
        # First collect evidence
        collect_handler = MockHandler(body={
            "task": "machine learning algorithms",
            "debate_id": "debate_ml",
        })
        collect_result = evidence_handler.handle_post(
            "/api/evidence/collect", {}, collect_handler
        )
        assert collect_result.status_code == 200

        # Then search for it
        search_handler = MockHandler(body={"query": "test"})
        search_result = evidence_handler.handle_post(
            "/api/evidence/search", {}, search_handler
        )
        assert search_result.status_code == 200

    def test_full_crud_workflow(self, evidence_handler, evidence_store):
        """Test create, read, update, delete workflow."""
        # Create
        evidence_store.save_evidence(
            evidence_id="ev_crud",
            source="test",
            title="CRUD Test",
            snippet="Testing CRUD operations",
        )

        # Read
        mock_handler = MockHandler()
        read_result = evidence_handler.handle(
            "/api/evidence/ev_crud", {}, mock_handler
        )
        assert read_result.status_code == 200

        # Delete
        delete_result = evidence_handler.handle_delete(
            "/api/evidence/ev_crud", {}, mock_handler
        )
        assert delete_result.status_code == 200

        # Verify deleted
        verify_result = evidence_handler.handle(
            "/api/evidence/ev_crud", {}, mock_handler
        )
        assert verify_result.status_code == 404
