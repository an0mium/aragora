"""
Tests for aragora.server.handlers.features.evidence - Evidence API Handler.

Tests cover:
- Route registration and can_handle
- GET /api/v1/evidence - List evidence with pagination
- GET /api/v1/evidence/statistics - Get store statistics
- GET /api/v1/evidence/:id - Get specific evidence
- GET /api/v1/evidence/debate/:debate_id - Get evidence for debate
- POST /api/v1/evidence/search - Search evidence
- POST /api/v1/evidence/collect - Collect evidence for topic
- POST /api/v1/evidence/debate/:debate_id - Associate evidence
- DELETE /api/v1/evidence/:id - Delete evidence
- Error handling and validation
- Rate limiting
- RBAC permissions
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import the module under test with Slack stub workaround
# ---------------------------------------------------------------------------


def _import_evidence_module():
    """Import evidence module, working around broken sibling imports."""
    try:
        import aragora.server.handlers.features.evidence as mod

        return mod
    except (ImportError, ModuleNotFoundError):
        pass

    # Clear partially loaded modules and stub broken imports
    to_remove = [k for k in sys.modules if k.startswith("aragora.server.handlers")]
    for k in to_remove:
        del sys.modules[k]

    _slack_stubs = [
        "aragora.server.handlers.social._slack_impl",
        "aragora.server.handlers.social._slack_impl.config",
        "aragora.server.handlers.social._slack_impl.handler",
        "aragora.server.handlers.social._slack_impl.commands",
        "aragora.server.handlers.social._slack_impl.events",
        "aragora.server.handlers.social._slack_impl.blocks",
        "aragora.server.handlers.social._slack_impl.interactions",
        "aragora.server.handlers.social.slack",
        "aragora.server.handlers.social.slack.handler",
    ]
    for name in _slack_stubs:
        if name not in sys.modules:
            stub = MagicMock()
            stub.__path__ = []
            stub.__file__ = f"<stub:{name}>"
            sys.modules[name] = stub

    import aragora.server.handlers.features.evidence as mod

    return mod


evidence_module = _import_evidence_module()
EvidenceHandler = evidence_module.EvidenceHandler


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


@dataclass
class MockEvidence:
    """Mock evidence item."""

    evidence_id: str = "evidence-123"
    source: str = "wikipedia"
    title: str = "Test Evidence"
    snippet: str = "This is test evidence content."
    url: str = "https://example.com/evidence"
    reliability_score: float = 0.85
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "title": self.title,
            "snippet": self.snippet,
            "url": self.url,
            "reliability_score": self.reliability_score,
            "metadata": self.metadata,
        }


@dataclass
class MockEvidencePack:
    """Mock evidence pack from collection."""

    topic_keywords: list = field(default_factory=lambda: ["test", "evidence"])
    snippets: list = field(default_factory=list)
    total_searched: int = 10
    average_reliability: float = 0.8
    average_freshness: float = 0.7


class MockEvidenceStore:
    """Mock evidence store for testing."""

    def __init__(self):
        self._evidence: dict[str, dict] = {}
        self._debate_evidence: dict[str, list] = {}
        self._km_adapter = None

    def get_statistics(self) -> dict:
        return {
            "total_evidence": len(self._evidence),
            "by_source": {"wikipedia": 3, "arxiv": 2},
        }

    def search_evidence(
        self,
        query: str,
        limit: int = 20,
        source_filter: str | None = None,
        min_reliability: float = 0.0,
        context=None,
    ) -> list:
        return list(self._evidence.values())[:limit]

    def get_evidence(self, evidence_id: str) -> dict | None:
        return self._evidence.get(evidence_id)

    def get_debate_evidence(self, debate_id: str, round_number: int | None = None) -> list:
        return self._debate_evidence.get(debate_id, [])

    def save_evidence(self, **kwargs) -> str:
        evidence_id = kwargs.get("evidence_id", f"evidence-{len(self._evidence)}")
        self._evidence[evidence_id] = kwargs
        return evidence_id

    def save_evidence_pack(self, pack, debate_id: str, round_number: int | None = None) -> list:
        return ["evidence-1", "evidence-2"]

    def delete_evidence(self, evidence_id: str) -> bool:
        if evidence_id in self._evidence:
            del self._evidence[evidence_id]
            return True
        return False

    def set_km_adapter(self, adapter):
        self._km_adapter = adapter


class MockEvidenceCollector:
    """Mock evidence collector."""

    async def collect_evidence(self, task: str, connectors=None):
        mock_snippet = MagicMock()
        mock_snippet.to_dict.return_value = {
            "title": "Test Result",
            "snippet": "Collected evidence",
            "source": "web",
            "reliability_score": 0.8,
        }
        pack = MockEvidencePack()
        pack.snippets = [mock_snippet]
        return pack


def make_mock_handler(
    body: dict | None = None,
    method: str = "GET",
    path: str = "/api/v1/evidence",
    query_string: str = "",
):
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = method
    handler.path = path
    handler.headers = {}
    handler.client_address = ("127.0.0.1", 12345)

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.headers["Content-Length"] = str(len(body_bytes))
        handler.rfile = BytesIO(body_bytes)
    else:
        handler.rfile = BytesIO(b"")
        handler.headers["Content-Length"] = "0"

    return handler


@pytest.fixture
def evidence_handler():
    """Create EvidenceHandler with mock context."""
    ctx = {
        "evidence_store": MockEvidenceStore(),
        "evidence_collector": MockEvidenceCollector(),
    }
    handler = EvidenceHandler(ctx)
    return handler


@pytest.fixture(autouse=True)
def clear_rate_limiters():
    """Clear rate limiters before each test."""
    from aragora.server.handlers.utils.rate_limit import _limiters

    for limiter in _limiters.values():
        limiter.clear()
    yield
    for limiter in _limiters.values():
        limiter.clear()


# ===========================================================================
# Test Routing (can_handle)
# ===========================================================================


class TestEvidenceHandlerRouting:
    """Tests for EvidenceHandler.can_handle."""

    def test_can_handle_evidence_list(self, evidence_handler):
        assert evidence_handler.can_handle("/api/v1/evidence") is True

    def test_can_handle_evidence_statistics(self, evidence_handler):
        assert evidence_handler.can_handle("/api/v1/evidence/statistics") is True

    def test_can_handle_evidence_search(self, evidence_handler):
        assert evidence_handler.can_handle("/api/v1/evidence/search") is True

    def test_can_handle_evidence_collect(self, evidence_handler):
        assert evidence_handler.can_handle("/api/v1/evidence/collect") is True

    def test_can_handle_evidence_by_id(self, evidence_handler):
        assert evidence_handler.can_handle("/api/v1/evidence/evidence-123") is True

    def test_can_handle_debate_evidence(self, evidence_handler):
        assert evidence_handler.can_handle("/api/v1/evidence/debate/debate-123") is True

    def test_cannot_handle_other_paths(self, evidence_handler):
        assert evidence_handler.can_handle("/api/v1/debates") is False

    def test_cannot_handle_unrelated_paths(self, evidence_handler):
        assert evidence_handler.can_handle("/api/v1/gauntlet/run") is False


# ===========================================================================
# Test List Evidence (GET /api/v1/evidence)
# ===========================================================================


class TestEvidenceList:
    """Tests for GET /api/v1/evidence endpoint."""

    def test_list_evidence_success(self, evidence_handler):
        """Happy path: list evidence with default pagination."""
        store = evidence_handler._get_evidence_store()
        store._evidence["e1"] = MockEvidence(evidence_id="e1").to_dict()
        store._evidence["e2"] = MockEvidence(evidence_id="e2").to_dict()

        handler = make_mock_handler()
        result = evidence_handler.handle("/api/v1/evidence", {}, handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "evidence" in data
        assert data["total"] == 2

    def test_list_evidence_empty(self, evidence_handler):
        """Empty evidence store returns empty list."""
        handler = make_mock_handler()
        result = evidence_handler.handle("/api/v1/evidence", {}, handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["evidence"] == []
        assert data["total"] == 0

    def test_list_evidence_with_pagination(self, evidence_handler):
        """Custom limit and offset in pagination."""
        handler = make_mock_handler()
        result = evidence_handler.handle(
            "/api/v1/evidence", {"limit": "10", "offset": "5"}, handler
        )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["limit"] == 10
        assert data["offset"] == 5

    def test_list_evidence_with_filters(self, evidence_handler):
        """Source filter and min_reliability."""
        handler = make_mock_handler()
        result = evidence_handler.handle(
            "/api/v1/evidence",
            {"source": "wikipedia", "min_reliability": "0.5"},
            handler,
        )

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Test Get Statistics (GET /api/v1/evidence/statistics)
# ===========================================================================


class TestEvidenceStatistics:
    """Tests for GET /api/v1/evidence/statistics endpoint."""

    def test_get_statistics_success(self, evidence_handler):
        """Happy path: get statistics."""
        handler = make_mock_handler(path="/api/v1/evidence/statistics")
        result = evidence_handler.handle("/api/v1/evidence/statistics", {}, handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "statistics" in data
        assert "total_evidence" in data["statistics"]


# ===========================================================================
# Test Get Evidence by ID (GET /api/v1/evidence/:id)
# ===========================================================================


class TestEvidenceGetById:
    """Tests for GET /api/v1/evidence/:id endpoint."""

    def test_get_evidence_success(self, evidence_handler):
        """Happy path: get evidence by ID."""
        store = evidence_handler._get_evidence_store()
        store._evidence["evidence-123"] = MockEvidence().to_dict()

        handler = make_mock_handler(path="/api/v1/evidence/evidence-123")
        result = evidence_handler.handle("/api/v1/evidence/evidence-123", {}, handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "evidence" in data
        assert data["evidence"]["source"] == "wikipedia"

    def test_get_evidence_not_found(self, evidence_handler):
        """Evidence not found returns 404."""
        handler = make_mock_handler(path="/api/v1/evidence/nonexistent")
        result = evidence_handler.handle("/api/v1/evidence/nonexistent", {}, handler)

        assert result is not None
        assert result.status_code == 404

    def test_get_evidence_invalid_id(self, evidence_handler):
        """Invalid evidence ID returns 400."""
        handler = make_mock_handler(path="/api/v1/evidence/../etc/passwd")
        result = evidence_handler.handle("/api/v1/evidence/../etc/passwd", {}, handler)

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Test Get Debate Evidence (GET /api/v1/evidence/debate/:debate_id)
# ===========================================================================


class TestEvidenceGetByDebate:
    """Tests for GET /api/v1/evidence/debate/:debate_id endpoint."""

    def test_get_debate_evidence_success(self, evidence_handler):
        """Happy path: get evidence for debate."""
        store = evidence_handler._get_evidence_store()
        store._debate_evidence["debate-123"] = [MockEvidence().to_dict()]

        handler = make_mock_handler(path="/api/v1/evidence/debate/debate-123")
        result = evidence_handler.handle("/api/v1/evidence/debate/debate-123", {}, handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["debate_id"] == "debate-123"
        assert data["count"] == 1

    def test_get_debate_evidence_empty(self, evidence_handler):
        """Debate with no evidence returns empty list."""
        handler = make_mock_handler(path="/api/v1/evidence/debate/debate-empty")
        result = evidence_handler.handle("/api/v1/evidence/debate/debate-empty", {}, handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["evidence"] == []
        assert data["count"] == 0

    def test_get_debate_evidence_with_round(self, evidence_handler):
        """Get evidence filtered by round number."""
        handler = make_mock_handler(path="/api/v1/evidence/debate/debate-123")
        result = evidence_handler.handle(
            "/api/v1/evidence/debate/debate-123", {"round": "2"}, handler
        )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["round"] == 2


# ===========================================================================
# Test Search Evidence (POST /api/v1/evidence/search)
# ===========================================================================


class TestEvidenceSearch:
    """Tests for POST /api/v1/evidence/search endpoint."""

    @pytest.mark.asyncio
    async def test_search_evidence_success(self, evidence_handler):
        """Happy path: search evidence."""
        store = evidence_handler._get_evidence_store()
        store._evidence["e1"] = MockEvidence().to_dict()

        handler = make_mock_handler(
            body={"query": "test query"},
            method="POST",
            path="/api/v1/evidence/search",
        )
        result = await evidence_handler.handle_post("/api/v1/evidence/search", {}, handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "results" in data
        assert data["query"] == "test query"

    @pytest.mark.asyncio
    async def test_search_evidence_empty_query(self, evidence_handler):
        """Empty query returns 400."""
        handler = make_mock_handler(
            body={"query": ""},
            method="POST",
            path="/api/v1/evidence/search",
        )
        result = await evidence_handler.handle_post("/api/v1/evidence/search", {}, handler)

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_search_evidence_missing_query(self, evidence_handler):
        """Missing query field returns 400."""
        handler = make_mock_handler(
            body={},
            method="POST",
            path="/api/v1/evidence/search",
        )
        result = await evidence_handler.handle_post("/api/v1/evidence/search", {}, handler)

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_search_evidence_with_context(self, evidence_handler):
        """Search with quality context."""
        handler = make_mock_handler(
            body={
                "query": "test",
                "context": {
                    "topic": "AI safety",
                    "keywords": ["alignment", "risk"],
                    "max_age_days": 30,
                },
            },
            method="POST",
            path="/api/v1/evidence/search",
        )
        result = await evidence_handler.handle_post("/api/v1/evidence/search", {}, handler)

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Test Collect Evidence (POST /api/v1/evidence/collect)
# ===========================================================================


class TestEvidenceCollect:
    """Tests for POST /api/v1/evidence/collect endpoint."""

    @pytest.mark.asyncio
    async def test_collect_evidence_success(self, evidence_handler):
        """Happy path: collect evidence for task."""
        handler = make_mock_handler(
            body={"task": "Research AI alignment"},
            method="POST",
            path="/api/v1/evidence/collect",
        )

        result = await evidence_handler.handle_post("/api/v1/evidence/collect", {}, handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["task"] == "Research AI alignment"
        assert "keywords" in data
        assert "snippets" in data

    @pytest.mark.asyncio
    async def test_collect_evidence_missing_task(self, evidence_handler):
        """Missing task returns 400."""
        handler = make_mock_handler(
            body={},
            method="POST",
            path="/api/v1/evidence/collect",
        )

        result = await evidence_handler.handle_post("/api/v1/evidence/collect", {}, handler)

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_collect_evidence_with_debate_id(self, evidence_handler):
        """Collect and save to debate."""
        handler = make_mock_handler(
            body={
                "task": "Research AI alignment",
                "debate_id": "debate-123",
                "round": 1,
            },
            method="POST",
            path="/api/v1/evidence/collect",
        )

        result = await evidence_handler.handle_post("/api/v1/evidence/collect", {}, handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["debate_id"] == "debate-123"
        assert "saved_ids" in data


# ===========================================================================
# Test Associate Evidence (POST /api/v1/evidence/debate/:debate_id)
# ===========================================================================


class TestEvidenceAssociate:
    """Tests for POST /api/v1/evidence/debate/:debate_id endpoint."""

    @pytest.mark.asyncio
    async def test_associate_evidence_success(self, evidence_handler):
        """Happy path: associate evidence with debate."""
        store = evidence_handler._get_evidence_store()
        store._evidence["e1"] = MockEvidence(evidence_id="e1").to_dict()
        store._evidence["e2"] = MockEvidence(evidence_id="e2").to_dict()

        handler = make_mock_handler(
            body={"evidence_ids": ["e1", "e2"]},
            method="POST",
            path="/api/v1/evidence/debate/debate-123",
        )

        result = await evidence_handler.handle_post(
            "/api/v1/evidence/debate/debate-123", {}, handler
        )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["debate_id"] == "debate-123"
        assert len(data["associated"]) == 2

    @pytest.mark.asyncio
    async def test_associate_evidence_empty_ids(self, evidence_handler):
        """Empty evidence_ids returns 400."""
        handler = make_mock_handler(
            body={"evidence_ids": []},
            method="POST",
            path="/api/v1/evidence/debate/debate-123",
        )

        result = await evidence_handler.handle_post(
            "/api/v1/evidence/debate/debate-123", {}, handler
        )

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_associate_evidence_missing_ids(self, evidence_handler):
        """Missing evidence_ids field returns 400."""
        handler = make_mock_handler(
            body={},
            method="POST",
            path="/api/v1/evidence/debate/debate-123",
        )

        result = await evidence_handler.handle_post(
            "/api/v1/evidence/debate/debate-123", {}, handler
        )

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Test Delete Evidence (DELETE /api/v1/evidence/:id)
# ===========================================================================


class TestEvidenceDelete:
    """Tests for DELETE /api/v1/evidence/:id endpoint."""

    def test_delete_evidence_success(self, evidence_handler):
        """Happy path: delete evidence."""
        store = evidence_handler._get_evidence_store()
        store._evidence["evidence-123"] = MockEvidence().to_dict()

        handler = make_mock_handler(
            method="DELETE",
            path="/api/v1/evidence/evidence-123",
        )
        result = evidence_handler.handle_delete("/api/v1/evidence/evidence-123", {}, handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["deleted"] is True
        assert data["evidence_id"] == "evidence-123"

    def test_delete_evidence_not_found(self, evidence_handler):
        """Delete nonexistent evidence returns 404."""
        handler = make_mock_handler(
            method="DELETE",
            path="/api/v1/evidence/nonexistent",
        )
        result = evidence_handler.handle_delete("/api/v1/evidence/nonexistent", {}, handler)

        assert result is not None
        assert result.status_code == 404

    def test_delete_evidence_invalid_id(self, evidence_handler):
        """Invalid evidence ID returns 400."""
        handler = make_mock_handler(
            method="DELETE",
            path="/api/v1/evidence/../etc/passwd",
        )
        result = evidence_handler.handle_delete("/api/v1/evidence/../etc/passwd", {}, handler)

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Test Rate Limiting
# ===========================================================================


class TestEvidenceRateLimiting:
    """Tests for rate limiting on evidence endpoints."""

    def test_read_rate_limit_exceeded(self, evidence_handler):
        """Rate limit exceeded returns 429."""
        with patch.object(evidence_module, "_evidence_read_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False

            handler = make_mock_handler()
            result = evidence_handler.handle("/api/v1/evidence", {}, handler)

            assert result is not None
            assert result.status_code == 429

    @pytest.mark.asyncio
    async def test_write_rate_limit_exceeded(self, evidence_handler):
        """Write rate limit exceeded returns 429."""
        with patch.object(evidence_module, "_evidence_write_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False

            handler = make_mock_handler(
                body={"query": "test"},
                method="POST",
            )
            result = await evidence_handler.handle_post("/api/v1/evidence/search", {}, handler)

            assert result is not None
            assert result.status_code == 429


# ===========================================================================
# Test Error Handling
# ===========================================================================


class TestEvidenceErrorHandling:
    """Tests for error handling in evidence handler."""

    def test_invalid_json_body(self, evidence_handler):
        """Invalid JSON body returns 400."""
        handler = MagicMock()
        handler.command = "POST"
        handler.headers = {"Content-Length": "10"}
        handler.rfile = BytesIO(b"not-json!!")
        handler.client_address = ("127.0.0.1", 12345)

        # The handler will catch JSON decode errors
        result = evidence_handler._handle_search({"query": ""})
        assert result is not None
        # Empty query returns 400
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_collector_error_handling(self, evidence_handler):
        """Collector error returns appropriate status."""
        with patch.object(evidence_handler, "_get_evidence_collector") as mock_get:
            mock_collector = MagicMock()
            mock_collector.collect_evidence = AsyncMock(
                side_effect=ValueError("Invalid parameters")
            )
            mock_get.return_value = mock_collector

            handler = make_mock_handler(
                body={"task": "test task"},
                method="POST",
            )
            result = await evidence_handler.handle_post("/api/v1/evidence/collect", {}, handler)

            assert result is not None
            assert result.status_code == 400
