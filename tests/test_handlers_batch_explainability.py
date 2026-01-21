"""
Tests for batch explainability endpoints in ExplainabilityHandler.

Tests cover:
- POST /api/v1/explainability/batch - Create batch explanation job
- GET  /api/v1/explainability/batch/:id/status - Get batch job status
- GET  /api/v1/explainability/batch/:id/results - Get batch job results
- POST /api/v1/explainability/compare - Compare explanations across debates
"""

import json
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock

from aragora.server.handlers.explainability import (
    ExplainabilityHandler,
    BatchStatus,
    BatchJob,
    BatchDebateResult,
    _batch_jobs_memory as _batch_jobs,
)
from aragora.server.handlers.base import HandlerResult


def parse_handler_result(result: HandlerResult) -> tuple[dict, int]:
    """Helper to parse HandlerResult into (body_dict, status_code)."""
    body_str = result.body.decode("utf-8") if isinstance(result.body, bytes) else result.body
    try:
        body_dict = json.loads(body_str)
    except (json.JSONDecodeError, TypeError):
        body_dict = {"raw": body_str}
    return body_dict, result.status_code


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def handler():
    """Create a fresh handler instance for each test."""
    return ExplainabilityHandler(server_context={})


@pytest.fixture
def mock_post_request():
    """Create a mock POST request handler."""
    request = Mock()
    request.headers = {"Content-Type": "application/json", "Content-Length": "100"}
    request.command = "POST"
    request.client_address = ("127.0.0.1", 12345)
    return request


@pytest.fixture
def mock_get_request():
    """Create a mock GET request handler."""
    request = Mock()
    request.headers = {"Content-Type": "application/json"}
    request.command = "GET"
    request.client_address = ("127.0.0.1", 12345)
    return request


@pytest.fixture(autouse=True)
def clear_batch_jobs():
    """Clear batch jobs before each test."""
    _batch_jobs.clear()
    yield
    _batch_jobs.clear()


# ============================================================================
# Test BatchStatus Enum
# ============================================================================


class TestBatchStatus:
    """Test BatchStatus enum."""

    def test_status_values(self):
        assert BatchStatus.PENDING.value == "pending"
        assert BatchStatus.PROCESSING.value == "processing"
        assert BatchStatus.COMPLETED.value == "completed"
        assert BatchStatus.PARTIAL.value == "partial"
        assert BatchStatus.FAILED.value == "failed"


# ============================================================================
# Test BatchDebateResult
# ============================================================================


class TestBatchDebateResult:
    """Test BatchDebateResult dataclass."""

    def test_to_dict_success(self):
        result = BatchDebateResult(
            debate_id="debate-123",
            status="success",
            explanation={"confidence": 0.85},
            processing_time_ms=125.5,
        )
        data = result.to_dict()

        assert data["debate_id"] == "debate-123"
        assert data["status"] == "success"
        assert data["explanation"] == {"confidence": 0.85}
        assert data["processing_time_ms"] == 125.5
        assert "error" not in data

    def test_to_dict_error(self):
        result = BatchDebateResult(
            debate_id="debate-456",
            status="error",
            error="Debate not found",
            processing_time_ms=10.2,
        )
        data = result.to_dict()

        assert data["debate_id"] == "debate-456"
        assert data["status"] == "error"
        assert data["error"] == "Debate not found"
        assert "explanation" not in data


# ============================================================================
# Test BatchJob
# ============================================================================


class TestBatchJob:
    """Test BatchJob dataclass."""

    def test_to_dict(self):
        job = BatchJob(
            batch_id="batch-123",
            debate_ids=["d1", "d2", "d3"],
            status=BatchStatus.PROCESSING,
            processed_count=1,
            results=[BatchDebateResult(debate_id="d1", status="success", processing_time_ms=100)],
        )
        data = job.to_dict()

        assert data["batch_id"] == "batch-123"
        assert data["status"] == "processing"
        assert data["total_debates"] == 3
        assert data["processed_count"] == 1
        assert data["success_count"] == 1
        assert data["error_count"] == 0
        assert data["progress_pct"] == 33.3

    def test_progress_calculation(self):
        job = BatchJob(
            batch_id="batch-456",
            debate_ids=["d1", "d2", "d3", "d4"],
            processed_count=2,
        )
        data = job.to_dict()
        assert data["progress_pct"] == 50.0

    def test_empty_debates_progress(self):
        job = BatchJob(
            batch_id="batch-789",
            debate_ids=[],
            processed_count=0,
        )
        data = job.to_dict()
        assert data["progress_pct"] == 0


# ============================================================================
# Test can_handle
# ============================================================================


class TestCanHandle:
    """Test can_handle for batch endpoints."""

    def test_batch_create_post(self, handler):
        assert handler.can_handle("/api/v1/explainability/batch", "POST") is True

    def test_batch_create_get_rejected(self, handler):
        assert handler.can_handle("/api/v1/explainability/batch", "GET") is False

    def test_batch_status_get(self, handler):
        assert handler.can_handle("/api/v1/explainability/batch/batch-123/status", "GET") is True

    def test_batch_results_get(self, handler):
        assert handler.can_handle("/api/v1/explainability/batch/batch-123/results", "GET") is True

    def test_compare_post(self, handler):
        assert handler.can_handle("/api/v1/explainability/compare", "POST") is True

    def test_compare_get_rejected(self, handler):
        assert handler.can_handle("/api/v1/explainability/compare", "GET") is False


# ============================================================================
# Test Create Batch Job
# ============================================================================


class TestCreateBatchJob:
    """Test POST /api/v1/explainability/batch endpoint."""

    def test_create_batch_valid(self, handler, mock_post_request):
        body = {
            "debate_ids": ["debate-1", "debate-2", "debate-3"],
            "options": {
                "include_evidence": True,
                "format": "summary",
            },
        }
        mock_post_request.rfile = Mock()
        mock_post_request.rfile.read = Mock(return_value=json.dumps(body).encode())
        mock_post_request.headers["Content-Length"] = len(json.dumps(body))

        result = handler.handle("/api/v1/explainability/batch", {}, mock_post_request)
        response_body, status = parse_handler_result(result)

        assert status == 202
        assert "batch_id" in response_body
        assert response_body["status"] == "pending"
        assert response_body["total_debates"] == 3
        assert "status_url" in response_body
        assert "results_url" in response_body

    def test_create_batch_empty_debate_ids(self, handler, mock_post_request):
        body = {"debate_ids": []}
        mock_post_request.rfile = Mock()
        mock_post_request.rfile.read = Mock(return_value=json.dumps(body).encode())
        mock_post_request.headers["Content-Length"] = len(json.dumps(body))

        result = handler.handle("/api/v1/explainability/batch", {}, mock_post_request)
        response_body, status = parse_handler_result(result)

        assert status == 400
        assert "error" in response_body

    def test_create_batch_missing_debate_ids(self, handler, mock_post_request):
        body = {"options": {"format": "full"}}
        mock_post_request.rfile = Mock()
        mock_post_request.rfile.read = Mock(return_value=json.dumps(body).encode())
        mock_post_request.headers["Content-Length"] = len(json.dumps(body))

        result = handler.handle("/api/v1/explainability/batch", {}, mock_post_request)
        response_body, status = parse_handler_result(result)

        assert status == 400
        assert "error" in response_body

    def test_create_batch_invalid_json(self, handler, mock_post_request):
        mock_post_request.rfile = Mock()
        mock_post_request.rfile.read = Mock(return_value=b"not valid json")
        mock_post_request.headers["Content-Length"] = 14

        result = handler.handle("/api/v1/explainability/batch", {}, mock_post_request)
        response_body, status = parse_handler_result(result)

        assert status == 400
        assert "error" in response_body


# ============================================================================
# Test Get Batch Status
# ============================================================================


class TestGetBatchStatus:
    """Test GET /api/v1/explainability/batch/:id/status endpoint."""

    def test_get_status_pending(self, handler, mock_get_request):
        # Create a job directly
        job = BatchJob(
            batch_id="batch-test-123",
            debate_ids=["d1", "d2"],
            status=BatchStatus.PENDING,
        )
        _batch_jobs["batch-test-123"] = job

        result = handler.handle(
            "/api/v1/explainability/batch/batch-test-123/status", {}, mock_get_request
        )
        response_body, status = parse_handler_result(result)

        assert status == 200
        assert response_body["batch_id"] == "batch-test-123"
        assert response_body["status"] == "pending"
        assert response_body["total_debates"] == 2

    def test_get_status_processing(self, handler, mock_get_request):
        job = BatchJob(
            batch_id="batch-test-456",
            debate_ids=["d1", "d2", "d3"],
            status=BatchStatus.PROCESSING,
            processed_count=1,
            started_at=time.time(),
        )
        _batch_jobs["batch-test-456"] = job

        result = handler.handle(
            "/api/v1/explainability/batch/batch-test-456/status", {}, mock_get_request
        )
        response_body, status = parse_handler_result(result)

        assert status == 200
        assert response_body["status"] == "processing"
        assert response_body["processed_count"] == 1
        assert response_body["progress_pct"] == 33.3

    def test_get_status_not_found(self, handler, mock_get_request):
        result = handler.handle(
            "/api/v1/explainability/batch/nonexistent/status", {}, mock_get_request
        )
        response_body, status = parse_handler_result(result)

        assert status == 404
        assert "error" in response_body


# ============================================================================
# Test Get Batch Results
# ============================================================================


class TestGetBatchResults:
    """Test GET /api/v1/explainability/batch/:id/results endpoint."""

    def test_get_results_completed(self, handler, mock_get_request):
        job = BatchJob(
            batch_id="batch-results-123",
            debate_ids=["d1", "d2"],
            status=BatchStatus.COMPLETED,
            processed_count=2,
            completed_at=time.time(),
            results=[
                BatchDebateResult(
                    debate_id="d1",
                    status="success",
                    explanation={"confidence": 0.9},
                    processing_time_ms=100,
                ),
                BatchDebateResult(
                    debate_id="d2",
                    status="success",
                    explanation={"confidence": 0.8},
                    processing_time_ms=150,
                ),
            ],
        )
        _batch_jobs["batch-results-123"] = job

        result = handler.handle(
            "/api/v1/explainability/batch/batch-results-123/results", {}, mock_get_request
        )
        response_body, status = parse_handler_result(result)

        assert status == 200
        assert response_body["status"] == "completed"
        assert len(response_body["results"]) == 2
        assert response_body["results"][0]["debate_id"] == "d1"
        assert "pagination" in response_body

    def test_get_results_pending(self, handler, mock_get_request):
        job = BatchJob(
            batch_id="batch-pending-123",
            debate_ids=["d1", "d2"],
            status=BatchStatus.PENDING,
        )
        _batch_jobs["batch-pending-123"] = job

        result = handler.handle(
            "/api/v1/explainability/batch/batch-pending-123/results", {}, mock_get_request
        )
        response_body, status = parse_handler_result(result)

        assert status == 202
        assert "error" in response_body or "message" in response_body

    def test_get_results_partial_allowed(self, handler, mock_get_request):
        job = BatchJob(
            batch_id="batch-partial-123",
            debate_ids=["d1", "d2", "d3"],
            status=BatchStatus.PROCESSING,
            processed_count=1,
            results=[
                BatchDebateResult(
                    debate_id="d1",
                    status="success",
                    explanation={"confidence": 0.9},
                    processing_time_ms=100,
                ),
            ],
        )
        _batch_jobs["batch-partial-123"] = job

        result = handler.handle(
            "/api/v1/explainability/batch/batch-partial-123/results",
            {"include_partial": "true"},
            mock_get_request,
        )
        response_body, status = parse_handler_result(result)

        assert status == 200
        assert len(response_body["results"]) == 1

    def test_get_results_pagination(self, handler, mock_get_request):
        results = [
            BatchDebateResult(
                debate_id=f"d{i}", status="success", explanation={}, processing_time_ms=100
            )
            for i in range(10)
        ]
        job = BatchJob(
            batch_id="batch-paginated",
            debate_ids=[f"d{i}" for i in range(10)],
            status=BatchStatus.COMPLETED,
            processed_count=10,
            results=results,
        )
        _batch_jobs["batch-paginated"] = job

        result = handler.handle(
            "/api/v1/explainability/batch/batch-paginated/results",
            {"limit": "3", "offset": "2"},
            mock_get_request,
        )
        response_body, status = parse_handler_result(result)

        assert status == 200
        assert len(response_body["results"]) == 3
        assert response_body["pagination"]["offset"] == 2
        assert response_body["pagination"]["limit"] == 3
        assert response_body["pagination"]["has_more"] is True

    def test_get_results_not_found(self, handler, mock_get_request):
        result = handler.handle(
            "/api/v1/explainability/batch/nonexistent/results", {}, mock_get_request
        )
        response_body, status = parse_handler_result(result)

        assert status == 404
        assert "error" in response_body


# ============================================================================
# Test Compare Explanations
# ============================================================================


class TestCompareExplanations:
    """Test POST /api/v1/explainability/compare endpoint."""

    def test_compare_insufficient_debates(self, handler, mock_post_request):
        body = {"debate_ids": ["debate-1"]}  # Need at least 2
        mock_post_request.rfile = Mock()
        mock_post_request.rfile.read = Mock(return_value=json.dumps(body).encode())
        mock_post_request.headers["Content-Length"] = len(json.dumps(body))

        result = handler.handle("/api/v1/explainability/compare", {}, mock_post_request)
        response_body, status = parse_handler_result(result)

        assert status == 400
        assert "error" in response_body

    def test_compare_too_many_debates(self, handler, mock_post_request):
        body = {"debate_ids": [f"debate-{i}" for i in range(15)]}  # Max is 10
        mock_post_request.rfile = Mock()
        mock_post_request.rfile.read = Mock(return_value=json.dumps(body).encode())
        mock_post_request.headers["Content-Length"] = len(json.dumps(body))

        result = handler.handle("/api/v1/explainability/compare", {}, mock_post_request)
        response_body, status = parse_handler_result(result)

        assert status == 400
        assert "error" in response_body

    def test_compare_invalid_json(self, handler, mock_post_request):
        mock_post_request.rfile = Mock()
        mock_post_request.rfile.read = Mock(return_value=b"invalid json")
        mock_post_request.headers["Content-Length"] = 12

        result = handler.handle("/api/v1/explainability/compare", {}, mock_post_request)
        response_body, status = parse_handler_result(result)

        assert status == 400
        assert "error" in response_body
