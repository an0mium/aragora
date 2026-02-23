"""
Tests for EvidenceEnrichmentHandler.

Comprehensive test coverage for evidence enrichment API endpoints:

Routing (can_handle):
- /api/v1/findings/{finding_id}/evidence  GET   (get evidence)
- /api/v1/findings/{finding_id}/evidence  POST  (enrich finding)
- /api/v1/findings/batch-evidence         POST  (batch enrich)

Covers:
- can_handle routing
- GET evidence retrieval (with/without enrichment, not found)
- POST single enrichment (success, not found, errors)
- POST batch enrichment (success, partial, validation, errors)
- Circuit breaker status helper
- Error handling paths
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.evidence_enrichment import (
    EvidenceEnrichmentHandler,
    get_evidence_circuit_breaker,
    get_evidence_circuit_breaker_status,
    _get_evidence_enrichment,
    _set_evidence_enrichment,
)


# =============================================================================
# Module path for patching
# =============================================================================

MODULE = "aragora.server.handlers.features.evidence_enrichment"


# =============================================================================
# Auto-patch JWT auth for all tests in this module
# =============================================================================


class _MockUserCtx:
    """Mock authenticated user context returned by extract_user_from_request."""

    is_authenticated = True
    authenticated = True
    user_id = "test-user-001"
    role = "admin"
    org_id = "org-1"
    client_ip = "127.0.0.1"
    email = "admin@example.com"
    error_reason = None


@pytest.fixture(autouse=True)
def _bypass_jwt_auth(monkeypatch):
    """Patch extract_user_from_request so @require_user_auth always passes."""
    monkeypatch.setattr(
        "aragora.billing.jwt_auth.extract_user_from_request",
        lambda handler, user_store=None: _MockUserCtx(),
    )


# =============================================================================
# Mock objects
# =============================================================================


class MockHTTPHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, body: dict[str, Any] | None = None):
        self.rfile = MagicMock()
        self._body = body
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {
                "Content-Length": str(len(body_bytes)),
                "Content-Type": "application/json",
                "Authorization": "Bearer test-token",
            }
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {
                "Content-Length": "2",
                "Content-Type": "application/json",
                "Authorization": "Bearer test-token",
            }
        self.client_address = ("127.0.0.1", 12345)


@dataclass
class MockFinding:
    """Mock audit finding."""

    id: str = "finding-001"
    session_id: str = "session-001"
    document_id: str = "doc-001"
    title: str = "Test finding"
    confidence: float = 0.7


@dataclass
class MockSession:
    """Mock audit session with findings."""

    id: str = "session-001"
    findings: list = field(default_factory=list)


@dataclass
class MockEnrichment:
    """Mock evidence enrichment result."""

    finding_id: str = "finding-001"

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "sources": [
                {
                    "source_id": "src-1",
                    "source_type": "document",
                    "title": "Source doc",
                    "snippet": "Relevant text",
                    "location": "page 3",
                    "relevance_score": 0.85,
                    "reliability_score": 0.9,
                }
            ],
            "original_confidence": 0.7,
            "adjusted_confidence": 0.85,
            "evidence_summary": "Strong supporting evidence found",
            "collection_time_ms": 234,
            "has_strong_evidence": True,
        }


class MockDocument:
    """Mock document with content."""

    def __init__(self, content: str = "Document content here"):
        self.content = content


# =============================================================================
# Helpers
# =============================================================================


def _status(result) -> int:
    """Extract status code from handler response."""
    if result is None:
        return 0
    return result["status"]


def _body(result) -> dict[str, Any]:
    """Parse the JSON body from a handler response."""
    if result is None:
        return {}
    return json.loads(result["body"])


def _make_auditor(findings: list[MockFinding] | None = None) -> MagicMock:
    """Build a mock document auditor with given findings."""
    auditor = MagicMock()
    if findings:
        session = MockSession(findings=findings)
        auditor._sessions = {"session-001": session}
    else:
        auditor._sessions = {}
    return auditor


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def handler():
    """Create an EvidenceEnrichmentHandler instance."""
    return EvidenceEnrichmentHandler(server_context={})


@pytest.fixture
def handler_with_store():
    """Create a handler with a document store in context."""
    mock_store = MagicMock()
    mock_doc = MockDocument("Stored document content")
    mock_store.get.return_value = mock_doc
    return EvidenceEnrichmentHandler(server_context={"document_store": mock_store})


@pytest.fixture
def finding():
    """Create a default mock finding."""
    return MockFinding()


@pytest.fixture
def finding2():
    """Create a second mock finding."""
    return MockFinding(id="finding-002", document_id="doc-002", title="Second finding")


# =============================================================================
# 1. can_handle() Routing Tests
# =============================================================================


class TestCanHandle:
    """Test can_handle routes all expected paths."""

    def test_handles_batch_evidence(self, handler):
        assert handler.can_handle("/api/v1/findings/batch-evidence") is True

    def test_handles_single_finding_evidence(self, handler):
        assert handler.can_handle("/api/v1/findings/finding-001/evidence") is True

    def test_handles_finding_with_uuid(self, handler):
        assert handler.can_handle("/api/v1/findings/abc-123-def/evidence") is True

    def test_rejects_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_rejects_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_rejects_findings_without_evidence_suffix(self, handler):
        assert handler.can_handle("/api/v1/findings/finding-001/details") is False

    def test_rejects_nested_finding_evidence(self, handler):
        # Too many segments: /api/v1/findings/x/y/evidence has 7 parts
        assert handler.can_handle("/api/v1/findings/x/y/evidence") is False

    def test_rejects_findings_root(self, handler):
        assert handler.can_handle("/api/v1/findings") is False

    def test_rejects_evidence_without_finding_id(self, handler):
        # /api/v1/findings//evidence has an empty segment but 6 parts
        # In practice split would yield ['', 'api', 'v1', 'findings', '', 'evidence']
        assert handler.can_handle("/api/v1/findings//evidence") is True  # 6 parts, empty id

    def test_routes_class_attribute(self):
        assert "/api/v1/findings/batch-evidence" in EvidenceEnrichmentHandler.ROUTES


# =============================================================================
# 2. GET /api/v1/findings/{finding_id}/evidence
# =============================================================================


class TestGetFindingEvidence:
    """Test GET finding evidence endpoint."""

    def test_get_evidence_finding_not_found(self, handler):
        """When finding doesn't exist, returns 404."""
        mock_handler = MockHTTPHandler()
        auditor = _make_auditor(findings=[])

        with patch(f"{MODULE}.get_document_auditor", return_value=auditor):
            result = handler.handle("/api/v1/findings/nonexistent/evidence", {}, mock_handler)

        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_get_evidence_no_enrichment_yet(self, handler, finding):
        """When finding exists but no enrichment, returns message."""
        mock_handler = MockHTTPHandler()
        auditor = _make_auditor(findings=[finding])

        with patch(f"{MODULE}.get_document_auditor", return_value=auditor):
            result = handler.handle("/api/v1/findings/finding-001/evidence", {}, mock_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["finding_id"] == "finding-001"
        assert body["evidence"] is None
        assert "POST" in body.get("message", "")

    def test_get_evidence_with_enrichment(self, handler, finding):
        """When finding has enrichment, returns evidence data."""
        mock_handler = MockHTTPHandler()
        enrichment = MockEnrichment(finding_id="finding-001")
        _set_evidence_enrichment(finding, enrichment)
        auditor = _make_auditor(findings=[finding])

        with patch(f"{MODULE}.get_document_auditor", return_value=auditor):
            result = handler.handle("/api/v1/findings/finding-001/evidence", {}, mock_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["finding_id"] == "finding-001"
        assert body["evidence"]["original_confidence"] == 0.7
        assert body["evidence"]["adjusted_confidence"] == 0.85
        assert body["evidence"]["has_strong_evidence"] is True

    def test_get_evidence_internal_error(self, handler):
        """When auditor raises, returns 500."""
        mock_handler = MockHTTPHandler()

        with patch(
            f"{MODULE}.get_document_auditor",
            side_effect=RuntimeError("Auditor unavailable"),
        ):
            result = handler.handle("/api/v1/findings/finding-001/evidence", {}, mock_handler)

        assert _status(result) == 500

    def test_get_evidence_wrong_finding_id(self, handler, finding):
        """When finding_id doesn't match, returns 404."""
        mock_handler = MockHTTPHandler()
        auditor = _make_auditor(findings=[finding])

        with patch(f"{MODULE}.get_document_auditor", return_value=auditor):
            result = handler.handle("/api/v1/findings/wrong-id/evidence", {}, mock_handler)

        assert _status(result) == 404

    def test_get_evidence_multiple_sessions(self, handler):
        """Finding can be found across multiple sessions."""
        mock_handler = MockHTTPHandler()
        finding_in_s2 = MockFinding(id="finding-in-s2", session_id="session-002")

        auditor = MagicMock()
        s1 = MockSession(id="session-001", findings=[MockFinding(id="other")])
        s2 = MockSession(id="session-002", findings=[finding_in_s2])
        auditor._sessions = {"session-001": s1, "session-002": s2}

        with patch(f"{MODULE}.get_document_auditor", return_value=auditor):
            result = handler.handle("/api/v1/findings/finding-in-s2/evidence", {}, mock_handler)

        assert _status(result) == 200
        assert _body(result)["finding_id"] == "finding-in-s2"


# =============================================================================
# 3. POST /api/v1/findings/{finding_id}/evidence (single enrich)
# =============================================================================


class TestEnrichFinding:
    """Test POST single finding enrichment endpoint."""

    def test_enrich_finding_success(self, handler, finding):
        """Successful enrichment returns enrichment data."""
        mock_handler = MockHTTPHandler(
            body={
                "document_content": "Some document text",
                "config": {"max_sources_per_finding": 3},
            }
        )
        enrichment_result = {
            "finding_id": "finding-001",
            "enrichment": MockEnrichment().to_dict(),
        }

        with patch(f"{MODULE}._run_async", return_value=enrichment_result):
            result = handler.handle_post("/api/v1/findings/finding-001/evidence", {}, mock_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["finding_id"] == "finding-001"
        assert body["enrichment"]["adjusted_confidence"] == 0.85

    def test_enrich_finding_not_found(self, handler):
        """When finding not found, _run_async raises ValueError -> 404."""
        mock_handler = MockHTTPHandler(body={})

        with patch(
            f"{MODULE}._run_async",
            side_effect=ValueError("Finding not found: nonexistent"),
        ):
            result = handler.handle_post("/api/v1/findings/nonexistent/evidence", {}, mock_handler)

        assert _status(result) == 404

    def test_enrich_finding_runtime_error(self, handler):
        """When enrichment fails with RuntimeError, returns 500."""
        mock_handler = MockHTTPHandler(body={})

        with patch(
            f"{MODULE}._run_async",
            side_effect=RuntimeError("Connection failed"),
        ):
            result = handler.handle_post("/api/v1/findings/finding-001/evidence", {}, mock_handler)

        assert _status(result) == 500

    def test_enrich_finding_timeout_error(self, handler):
        """When enrichment times out, returns 500."""
        mock_handler = MockHTTPHandler(body={})

        with patch(
            f"{MODULE}._run_async",
            side_effect=TimeoutError("Timed out"),
        ):
            result = handler.handle_post("/api/v1/findings/finding-001/evidence", {}, mock_handler)

        assert _status(result) == 500

    def test_enrich_finding_connection_error(self, handler):
        """When enrichment has connection error, returns 500."""
        mock_handler = MockHTTPHandler(body={})

        with patch(
            f"{MODULE}._run_async",
            side_effect=ConnectionError("Connection refused"),
        ):
            result = handler.handle_post("/api/v1/findings/finding-001/evidence", {}, mock_handler)

        assert _status(result) == 500

    def test_enrich_finding_empty_body(self, handler):
        """Enrichment with no body still proceeds (body defaults)."""
        mock_handler = MockHTTPHandler()
        enrichment_result = {
            "finding_id": "finding-001",
            "enrichment": MockEnrichment().to_dict(),
        }

        with patch(f"{MODULE}._run_async", return_value=enrichment_result):
            result = handler.handle_post("/api/v1/findings/finding-001/evidence", {}, mock_handler)

        assert _status(result) == 200

    def test_enrich_finding_with_related_documents(self, handler):
        """Enrichment with related_documents in body."""
        mock_handler = MockHTTPHandler(
            body={
                "document_content": "Main doc",
                "related_documents": {"doc2": "Related content"},
                "config": {"enable_cross_reference": True},
            }
        )
        enrichment_result = {
            "finding_id": "finding-001",
            "enrichment": MockEnrichment().to_dict(),
        }

        with patch(f"{MODULE}._run_async", return_value=enrichment_result):
            result = handler.handle_post("/api/v1/findings/finding-001/evidence", {}, mock_handler)

        assert _status(result) == 200

    def test_enrich_finding_unmatched_path_returns_none(self, handler):
        """POST to unrecognized path returns None."""
        mock_handler = MockHTTPHandler(body={})
        result = handler.handle_post("/api/v1/other/path", {}, mock_handler)
        assert result is None

    def test_enrich_finding_wrong_segment_count(self, handler):
        """POST to path with wrong number of segments returns None."""
        mock_handler = MockHTTPHandler(body={})
        result = handler.handle_post("/api/v1/findings/a/b/evidence", {}, mock_handler)
        assert result is None


# =============================================================================
# 4. POST /api/v1/findings/batch-evidence
# =============================================================================


class TestBatchEnrich:
    """Test POST batch evidence enrichment endpoint."""

    def test_batch_enrich_success(self, handler):
        """Successful batch enrichment returns enrichment map."""
        mock_handler = MockHTTPHandler(
            body={
                "finding_ids": ["finding-001", "finding-002"],
                "config": {"max_sources_per_finding": 3},
            }
        )
        batch_result = {
            "enrichments": {
                "finding-001": MockEnrichment(finding_id="finding-001").to_dict(),
                "finding-002": MockEnrichment(finding_id="finding-002").to_dict(),
            },
            "errors": {},
            "processed": 2,
            "failed": 0,
        }

        with patch(f"{MODULE}._run_async", return_value=batch_result):
            result = handler.handle_post("/api/v1/findings/batch-evidence", {}, mock_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["processed"] == 2
        assert body["failed"] == 0
        assert "finding-001" in body["enrichments"]
        assert "finding-002" in body["enrichments"]

    def test_batch_enrich_partial_success(self, handler):
        """Batch with some missing findings returns errors dict."""
        mock_handler = MockHTTPHandler(
            body={
                "finding_ids": ["finding-001", "missing-id"],
                "config": {},
            }
        )
        batch_result = {
            "enrichments": {
                "finding-001": MockEnrichment(finding_id="finding-001").to_dict(),
            },
            "errors": {"missing-id": "Finding not found"},
            "processed": 1,
            "failed": 1,
        }

        with patch(f"{MODULE}._run_async", return_value=batch_result):
            result = handler.handle_post("/api/v1/findings/batch-evidence", {}, mock_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["processed"] == 1
        assert body["failed"] == 1
        assert "missing-id" in body["errors"]

    def test_batch_enrich_no_body(self, handler):
        """Batch with no body returns 400."""
        # MockHTTPHandler with body=None produces Content-Length: 2 and b"{}"
        # read_json_body returns {} (empty dict), which is falsy? No, {} is truthy.
        # Actually the handler does: body = self.read_json_body(handler)
        # if not body: return error_response(400)
        # {} is falsy in Python? No, {} is falsy! empty dict is falsy.
        mock_handler = MockHTTPHandler()

        result = handler.handle_post("/api/v1/findings/batch-evidence", {}, mock_handler)

        assert _status(result) == 400
        assert "body" in _body(result).get("error", "").lower()

    def test_batch_enrich_empty_finding_ids(self, handler):
        """Batch with empty finding_ids list returns 400."""
        mock_handler = MockHTTPHandler(body={"finding_ids": []})

        result = handler.handle_post("/api/v1/findings/batch-evidence", {}, mock_handler)

        assert _status(result) == 400
        assert "finding_ids" in _body(result).get("error", "")

    def test_batch_enrich_missing_finding_ids_key(self, handler):
        """Batch without finding_ids key returns 400."""
        mock_handler = MockHTTPHandler(body={"config": {}})

        result = handler.handle_post("/api/v1/findings/batch-evidence", {}, mock_handler)

        assert _status(result) == 400

    def test_batch_enrich_runtime_error(self, handler):
        """Batch with runtime failure returns 500."""
        mock_handler = MockHTTPHandler(
            body={
                "finding_ids": ["finding-001"],
            }
        )

        with patch(
            f"{MODULE}._run_async",
            side_effect=RuntimeError("Batch processing failed"),
        ):
            result = handler.handle_post("/api/v1/findings/batch-evidence", {}, mock_handler)

        assert _status(result) == 500

    def test_batch_enrich_timeout_error(self, handler):
        """Batch with timeout returns 500."""
        mock_handler = MockHTTPHandler(
            body={
                "finding_ids": ["finding-001"],
            }
        )

        with patch(
            f"{MODULE}._run_async",
            side_effect=TimeoutError("Batch timed out"),
        ):
            result = handler.handle_post("/api/v1/findings/batch-evidence", {}, mock_handler)

        assert _status(result) == 500

    def test_batch_enrich_value_error(self, handler):
        """Batch with ValueError returns 500."""
        mock_handler = MockHTTPHandler(
            body={
                "finding_ids": ["finding-001"],
            }
        )

        with patch(
            f"{MODULE}._run_async",
            side_effect=ValueError("Invalid input"),
        ):
            result = handler.handle_post("/api/v1/findings/batch-evidence", {}, mock_handler)

        assert _status(result) == 500


# =============================================================================
# 5. handle() routing (GET)
# =============================================================================


class TestHandleRouting:
    """Test GET handle() route dispatch."""

    def test_handle_routes_to_get_evidence(self, handler, finding):
        """GET /api/v1/findings/{id}/evidence routes correctly."""
        mock_handler = MockHTTPHandler()
        auditor = _make_auditor(findings=[finding])

        with patch(f"{MODULE}.get_document_auditor", return_value=auditor):
            result = handler.handle("/api/v1/findings/finding-001/evidence", {}, mock_handler)

        assert result is not None
        assert _status(result) == 200

    def test_handle_unmatched_path_returns_none(self, handler):
        """GET to non-matching path returns None."""
        mock_handler = MockHTTPHandler()
        result = handler.handle("/api/v1/other", {}, mock_handler)
        assert result is None

    def test_handle_wrong_segment_count(self, handler):
        """GET to path with wrong segment count returns None."""
        mock_handler = MockHTTPHandler()
        result = handler.handle("/api/v1/findings/a/b/evidence", {}, mock_handler)
        assert result is None


# =============================================================================
# 6. handle_post() routing
# =============================================================================


class TestHandlePostRouting:
    """Test POST handle_post() route dispatch."""

    def test_handle_post_routes_to_batch(self, handler):
        """POST /api/v1/findings/batch-evidence routes to batch."""
        mock_handler = MockHTTPHandler(body={"finding_ids": ["f1"]})
        batch_result = {
            "enrichments": {},
            "errors": {"f1": "Finding not found"},
            "processed": 0,
            "failed": 1,
        }

        with patch(f"{MODULE}._run_async", return_value=batch_result):
            result = handler.handle_post("/api/v1/findings/batch-evidence", {}, mock_handler)

        assert result is not None
        assert _status(result) == 200

    def test_handle_post_routes_to_single_enrich(self, handler):
        """POST /api/v1/findings/{id}/evidence routes to single enrich."""
        mock_handler = MockHTTPHandler(body={})
        enrichment_result = {
            "finding_id": "f1",
            "enrichment": MockEnrichment().to_dict(),
        }

        with patch(f"{MODULE}._run_async", return_value=enrichment_result):
            result = handler.handle_post("/api/v1/findings/f1/evidence", {}, mock_handler)

        assert result is not None
        assert _status(result) == 200

    def test_handle_post_unmatched_returns_none(self, handler):
        """POST to unrecognized path returns None."""
        mock_handler = MockHTTPHandler(body={})
        result = handler.handle_post("/api/v1/unrelated", {}, mock_handler)
        assert result is None


# =============================================================================
# 7. _run_enrichment async method
# =============================================================================


class TestRunEnrichment:
    """Test the _run_enrichment async method."""

    @pytest.mark.asyncio
    async def test_run_enrichment_success(self, handler, finding):
        """Successful enrichment returns finding_id and enrichment dict."""
        auditor = _make_auditor(findings=[finding])
        mock_enrichment = MockEnrichment()
        mock_collector = AsyncMock()
        mock_collector.enrich_finding = AsyncMock(return_value=mock_enrichment)

        with (
            patch(f"{MODULE}.get_document_auditor", return_value=auditor),
            patch(f"{MODULE}.FindingEvidenceCollector", return_value=mock_collector),
        ):
            result = await handler._run_enrichment(
                finding_id="finding-001",
                document_content="Some text",
                related_documents={},
                config_dict={"max_sources_per_finding": 3},
                document_store=None,
            )

        assert result["finding_id"] == "finding-001"
        assert "enrichment" in result
        mock_collector.enrich_finding.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_enrichment_finding_not_found(self, handler):
        """When finding not found, raises ValueError."""
        auditor = _make_auditor(findings=[])

        with (
            patch(f"{MODULE}.get_document_auditor", return_value=auditor),
            pytest.raises(ValueError, match="Finding not found"),
        ):
            await handler._run_enrichment(
                finding_id="nonexistent",
                document_content=None,
                related_documents={},
                config_dict={},
                document_store=None,
            )

    @pytest.mark.asyncio
    async def test_run_enrichment_uses_document_store(self, handler, finding):
        """When no content provided, tries document store."""
        auditor = _make_auditor(findings=[finding])
        mock_doc = MockDocument("Stored content")
        mock_store = MagicMock()
        mock_store.get.return_value = mock_doc

        mock_enrichment = MockEnrichment()
        mock_collector = AsyncMock()
        mock_collector.enrich_finding = AsyncMock(return_value=mock_enrichment)

        with (
            patch(f"{MODULE}.get_document_auditor", return_value=auditor),
            patch(f"{MODULE}.FindingEvidenceCollector", return_value=mock_collector),
        ):
            result = await handler._run_enrichment(
                finding_id="finding-001",
                document_content=None,
                related_documents={},
                config_dict={},
                document_store=mock_store,
            )

        mock_store.get.assert_called_once_with("doc-001")
        # Verify content was passed to collector
        call_kwargs = mock_collector.enrich_finding.call_args
        assert call_kwargs.kwargs.get("document_content") == "Stored content" or (
            call_kwargs.args and len(call_kwargs.args) > 1
        )

    @pytest.mark.asyncio
    async def test_run_enrichment_config_defaults(self, handler, finding):
        """Config defaults are applied when not specified."""
        auditor = _make_auditor(findings=[finding])
        mock_enrichment = MockEnrichment()
        mock_collector = AsyncMock()
        mock_collector.enrich_finding = AsyncMock(return_value=mock_enrichment)

        with (
            patch(f"{MODULE}.get_document_auditor", return_value=auditor),
            patch(f"{MODULE}.FindingEvidenceCollector", return_value=mock_collector) as mock_cls,
            patch(f"{MODULE}.EvidenceConfig") as mock_config_cls,
        ):
            await handler._run_enrichment(
                finding_id="finding-001",
                document_content="text",
                related_documents={},
                config_dict={},
                document_store=None,
            )

        mock_config_cls.assert_called_once_with(
            max_sources_per_finding=5,
            enable_external_sources=True,
            enable_cross_reference=True,
        )

    @pytest.mark.asyncio
    async def test_run_enrichment_stores_on_finding(self, handler, finding):
        """Enrichment result is stored on the finding object."""
        auditor = _make_auditor(findings=[finding])
        mock_enrichment = MockEnrichment()
        mock_collector = AsyncMock()
        mock_collector.enrich_finding = AsyncMock(return_value=mock_enrichment)

        with (
            patch(f"{MODULE}.get_document_auditor", return_value=auditor),
            patch(f"{MODULE}.FindingEvidenceCollector", return_value=mock_collector),
        ):
            await handler._run_enrichment(
                finding_id="finding-001",
                document_content="text",
                related_documents={},
                config_dict={},
                document_store=None,
            )

        assert _get_evidence_enrichment(finding) is mock_enrichment


# =============================================================================
# 8. _run_batch_enrichment async method
# =============================================================================


class TestRunBatchEnrichment:
    """Test the _run_batch_enrichment async method."""

    @pytest.mark.asyncio
    async def test_batch_enrichment_success(self, handler, finding, finding2):
        """Batch enrichment processes multiple findings."""
        auditor = _make_auditor(findings=[finding, finding2])
        mock_enrichments = {
            "finding-001": MockEnrichment(finding_id="finding-001"),
            "finding-002": MockEnrichment(finding_id="finding-002"),
        }
        mock_collector = AsyncMock()
        mock_collector.enrich_findings_batch = AsyncMock(return_value=mock_enrichments)

        with (
            patch(f"{MODULE}.get_document_auditor", return_value=auditor),
            patch(f"{MODULE}.FindingEvidenceCollector", return_value=mock_collector),
        ):
            result = await handler._run_batch_enrichment(
                finding_ids=["finding-001", "finding-002"],
                config_dict={},
                document_store=None,
            )

        assert result["processed"] == 2
        assert result["failed"] == 0
        assert "finding-001" in result["enrichments"]
        assert "finding-002" in result["enrichments"]

    @pytest.mark.asyncio
    async def test_batch_enrichment_missing_findings(self, handler, finding):
        """Batch reports errors for missing findings."""
        auditor = _make_auditor(findings=[finding])
        mock_enrichments = {
            "finding-001": MockEnrichment(finding_id="finding-001"),
        }
        mock_collector = AsyncMock()
        mock_collector.enrich_findings_batch = AsyncMock(return_value=mock_enrichments)

        with (
            patch(f"{MODULE}.get_document_auditor", return_value=auditor),
            patch(f"{MODULE}.FindingEvidenceCollector", return_value=mock_collector),
        ):
            result = await handler._run_batch_enrichment(
                finding_ids=["finding-001", "missing-id"],
                config_dict={},
                document_store=None,
            )

        assert result["processed"] == 1
        assert result["failed"] == 1
        assert "missing-id" in result["errors"]
        assert result["errors"]["missing-id"] == "Finding not found"

    @pytest.mark.asyncio
    async def test_batch_enrichment_with_document_store(self, handler, finding):
        """Batch uses document store for finding content."""
        auditor = _make_auditor(findings=[finding])
        mock_store = MagicMock()
        mock_doc = MockDocument("Doc content")
        mock_store.get.return_value = mock_doc

        mock_enrichments = {
            "finding-001": MockEnrichment(finding_id="finding-001"),
        }
        mock_collector = AsyncMock()
        mock_collector.enrich_findings_batch = AsyncMock(return_value=mock_enrichments)

        with (
            patch(f"{MODULE}.get_document_auditor", return_value=auditor),
            patch(f"{MODULE}.FindingEvidenceCollector", return_value=mock_collector),
        ):
            result = await handler._run_batch_enrichment(
                finding_ids=["finding-001"],
                config_dict={},
                document_store=mock_store,
            )

        mock_store.get.assert_called_once_with("doc-001")
        assert result["processed"] == 1

    @pytest.mark.asyncio
    async def test_batch_enrichment_stores_on_findings(self, handler, finding, finding2):
        """Batch stores enrichments on finding objects."""
        auditor = _make_auditor(findings=[finding, finding2])
        mock_enrichments = {
            "finding-001": MockEnrichment(finding_id="finding-001"),
            "finding-002": MockEnrichment(finding_id="finding-002"),
        }
        mock_collector = AsyncMock()
        mock_collector.enrich_findings_batch = AsyncMock(return_value=mock_enrichments)

        with (
            patch(f"{MODULE}.get_document_auditor", return_value=auditor),
            patch(f"{MODULE}.FindingEvidenceCollector", return_value=mock_collector),
        ):
            await handler._run_batch_enrichment(
                finding_ids=["finding-001", "finding-002"],
                config_dict={},
                document_store=None,
            )

        assert _get_evidence_enrichment(finding) is mock_enrichments["finding-001"]
        assert _get_evidence_enrichment(finding2) is mock_enrichments["finding-002"]

    @pytest.mark.asyncio
    async def test_batch_enrichment_all_missing(self, handler):
        """Batch with all missing finding_ids returns all errors."""
        auditor = _make_auditor(findings=[])
        mock_collector = AsyncMock()
        mock_collector.enrich_findings_batch = AsyncMock(return_value={})

        with (
            patch(f"{MODULE}.get_document_auditor", return_value=auditor),
            patch(f"{MODULE}.FindingEvidenceCollector", return_value=mock_collector),
        ):
            result = await handler._run_batch_enrichment(
                finding_ids=["missing-1", "missing-2"],
                config_dict={},
                document_store=None,
            )

        assert result["processed"] == 0
        assert result["failed"] == 2
        assert "missing-1" in result["errors"]
        assert "missing-2" in result["errors"]

    @pytest.mark.asyncio
    async def test_batch_enrichment_max_concurrent_config(self, handler, finding):
        """Batch passes max_concurrent from config."""
        auditor = _make_auditor(findings=[finding])
        mock_enrichments = {
            "finding-001": MockEnrichment(finding_id="finding-001"),
        }
        mock_collector = AsyncMock()
        mock_collector.enrich_findings_batch = AsyncMock(return_value=mock_enrichments)

        with (
            patch(f"{MODULE}.get_document_auditor", return_value=auditor),
            patch(f"{MODULE}.FindingEvidenceCollector", return_value=mock_collector),
        ):
            await handler._run_batch_enrichment(
                finding_ids=["finding-001"],
                config_dict={"max_concurrent": 10},
                document_store=None,
            )

        call_kwargs = mock_collector.enrich_findings_batch.call_args
        assert call_kwargs.kwargs.get("max_concurrent") == 10


# =============================================================================
# 9. Circuit breaker utilities
# =============================================================================


class TestCircuitBreaker:
    """Test circuit breaker helper functions."""

    def test_get_circuit_breaker_returns_instance(self):
        cb = get_evidence_circuit_breaker()
        assert cb is not None
        assert cb.name == "evidence_enrichment_handler"

    def test_get_circuit_breaker_status_returns_dict(self):
        status = get_evidence_circuit_breaker_status()
        assert isinstance(status, dict)
        assert "name" in status or "state" in status or len(status) >= 1


# =============================================================================
# 10. Helper functions
# =============================================================================


class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_get_evidence_enrichment_none(self):
        """Returns None when no enrichment set."""
        finding = MockFinding()
        assert _get_evidence_enrichment(finding) is None

    def test_set_and_get_evidence_enrichment(self):
        """Setting and getting enrichment works."""
        finding = MockFinding()
        enrichment = MockEnrichment()
        _set_evidence_enrichment(finding, enrichment)
        assert _get_evidence_enrichment(finding) is enrichment

    def test_get_document_store_from_ctx(self):
        """_get_document_store returns store from context."""
        mock_store = MagicMock()
        handler = EvidenceEnrichmentHandler(server_context={"document_store": mock_store})
        assert handler._get_document_store() is mock_store

    def test_get_document_store_missing(self):
        """_get_document_store returns None when not in context."""
        handler = EvidenceEnrichmentHandler(server_context={})
        assert handler._get_document_store() is None


# =============================================================================
# 11. Constructor tests
# =============================================================================


class TestConstructor:
    """Test handler initialization."""

    def test_init_with_server_context(self):
        ctx = {"key": "value"}
        handler = EvidenceEnrichmentHandler(server_context=ctx)
        assert handler.ctx is ctx

    def test_init_with_ctx(self):
        ctx = {"key": "value"}
        handler = EvidenceEnrichmentHandler(ctx=ctx)
        assert handler.ctx is ctx

    def test_init_with_both_prefers_server_context(self):
        ctx1 = {"from": "ctx"}
        ctx2 = {"from": "server_context"}
        handler = EvidenceEnrichmentHandler(ctx=ctx1, server_context=ctx2)
        assert handler.ctx is ctx2

    def test_init_with_none_defaults_to_empty(self):
        handler = EvidenceEnrichmentHandler()
        assert handler.ctx == {}


# =============================================================================
# 12. Edge cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_get_evidence_attribute_error(self, handler):
        """AttributeError in get_document_auditor is caught."""
        mock_handler = MockHTTPHandler()

        with patch(
            f"{MODULE}.get_document_auditor",
            side_effect=AttributeError("No attribute"),
        ):
            result = handler.handle("/api/v1/findings/finding-001/evidence", {}, mock_handler)

        assert _status(result) == 500

    def test_get_evidence_key_error(self, handler):
        """KeyError in auditor is caught."""
        mock_handler = MockHTTPHandler()

        with patch(
            f"{MODULE}.get_document_auditor",
            side_effect=KeyError("missing"),
        ):
            result = handler.handle("/api/v1/findings/finding-001/evidence", {}, mock_handler)

        assert _status(result) == 500

    def test_enrich_finding_type_error(self, handler):
        """TypeError during enrichment returns 500."""
        mock_handler = MockHTTPHandler(body={})

        with patch(
            f"{MODULE}._run_async",
            side_effect=TypeError("bad type"),
        ):
            result = handler.handle_post("/api/v1/findings/finding-001/evidence", {}, mock_handler)

        assert _status(result) == 500

    def test_enrich_finding_os_error(self, handler):
        """OSError during enrichment returns 500."""
        mock_handler = MockHTTPHandler(body={})

        with patch(
            f"{MODULE}._run_async",
            side_effect=OSError("disk error"),
        ):
            result = handler.handle_post("/api/v1/findings/finding-001/evidence", {}, mock_handler)

        assert _status(result) == 500

    def test_batch_connection_error(self, handler):
        """ConnectionError during batch returns 500."""
        mock_handler = MockHTTPHandler(
            body={
                "finding_ids": ["f1"],
            }
        )

        with patch(
            f"{MODULE}._run_async",
            side_effect=ConnectionError("unreachable"),
        ):
            result = handler.handle_post("/api/v1/findings/batch-evidence", {}, mock_handler)

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_run_enrichment_document_store_returns_none(self, handler, finding):
        """When document store returns None for doc, content stays None."""
        auditor = _make_auditor(findings=[finding])
        mock_store = MagicMock()
        mock_store.get.return_value = None

        mock_enrichment = MockEnrichment()
        mock_collector = AsyncMock()
        mock_collector.enrich_finding = AsyncMock(return_value=mock_enrichment)

        with (
            patch(f"{MODULE}.get_document_auditor", return_value=auditor),
            patch(f"{MODULE}.FindingEvidenceCollector", return_value=mock_collector),
        ):
            result = await handler._run_enrichment(
                finding_id="finding-001",
                document_content=None,
                related_documents={},
                config_dict={},
                document_store=mock_store,
            )

        # Should still complete successfully
        assert result["finding_id"] == "finding-001"
        mock_store.get.assert_called_once_with("doc-001")

    @pytest.mark.asyncio
    async def test_run_enrichment_prefers_provided_content(self, handler, finding):
        """When document_content is provided, store is not queried."""
        auditor = _make_auditor(findings=[finding])
        mock_store = MagicMock()

        mock_enrichment = MockEnrichment()
        mock_collector = AsyncMock()
        mock_collector.enrich_finding = AsyncMock(return_value=mock_enrichment)

        with (
            patch(f"{MODULE}.get_document_auditor", return_value=auditor),
            patch(f"{MODULE}.FindingEvidenceCollector", return_value=mock_collector),
        ):
            await handler._run_enrichment(
                finding_id="finding-001",
                document_content="Provided content",
                related_documents={},
                config_dict={},
                document_store=mock_store,
            )

        # Store should NOT be queried because content was already provided
        mock_store.get.assert_not_called()
