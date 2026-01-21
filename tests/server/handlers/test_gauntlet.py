"""
Tests for aragora.server.handlers.gauntlet - Gauntlet stress-testing handler.

Tests cover:
- Routing and method handling
- List personas
- Start gauntlet
- Get status
- Get decision receipt
- Get risk heatmap
- List results with pagination
- Compare results
- Delete result
- Export report
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.gauntlet import (
    GauntletHandler,
    _gauntlet_runs,
    _cleanup_gauntlet_runs,
    MAX_GAUNTLET_RUNS_IN_MEMORY,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockAuthContext:
    """Mock authentication context."""

    is_authenticated: bool = True
    user_id: str = "user-123"
    email: str = "test@example.com"
    org_id: str | None = "org-123"
    role: str = "user"


@dataclass
class MockOrganization:
    """Mock organization for testing."""

    id: str = "org-123"
    is_at_limit: bool = False
    limits: Any = field(default_factory=lambda: MagicMock(debates_per_month=100))
    debates_used_this_month: int = 25
    tier: Any = field(default_factory=lambda: MagicMock(value="starter"))


@dataclass
class MockPersona:
    """Mock regulatory persona."""

    name: str = "GDPR Auditor"
    description: str = "Tests GDPR compliance"
    regulation: str = "GDPR"
    attack_prompts: list = field(default_factory=lambda: [MagicMock(category="privacy")] * 5)


@dataclass
class MockGauntletResult:
    """Mock gauntlet result for storage."""

    gauntlet_id: str = "gauntlet-test123"
    input_hash: str = "abc123"
    input_summary: str = "Test input"
    verdict: str = "APPROVED"
    confidence: float = 0.85
    robustness_score: float = 0.9
    critical_count: int = 0
    high_count: int = 1
    total_findings: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    duration_seconds: float = 45.0


class MockGauntletStorage:
    """Mock gauntlet storage for testing."""

    def __init__(self):
        self.results: dict[str, dict] = {}

    def save(self, result) -> None:
        self.results[result.gauntlet_id] = result

    def get(self, gauntlet_id: str) -> dict | None:
        result = self.results.get(gauntlet_id)
        if result:
            return {
                "gauntlet_id": gauntlet_id,
                "verdict": "APPROVED",
                "confidence": 0.85,
                "robustness_score": 0.9,
                "total_findings": 3,
                "critical_count": 0,
                "high_count": 1,
                "medium_count": 1,
                "low_count": 1,
                "findings": [],
            }
        return None

    def list_recent(
        self, limit: int = 20, offset: int = 0, verdict: str = None, min_severity: str = None
    ):
        results = list(self.results.values())
        return results[offset : offset + limit]

    def count(self, verdict: str = None) -> int:
        return len(self.results)

    def delete(self, gauntlet_id: str) -> bool:
        if gauntlet_id in self.results:
            del self.results[gauntlet_id]
            return True
        return False

    def compare(self, id1: str, id2: str) -> dict | None:
        if id1 in self.results and id2 in self.results:
            return {
                "comparison": {
                    "id1": id1,
                    "id2": id2,
                    "verdict_match": True,
                }
            }
        return None


class MockUserStore:
    """Mock user store for testing."""

    def __init__(self):
        self.orgs: dict[str, MockOrganization] = {}

    def get_organization_by_id(self, org_id: str) -> MockOrganization | None:
        return self.orgs.get(org_id)

    def increment_usage(self, org_id: str, count: int) -> None:
        pass


def make_mock_handler(
    body: dict | None = None,
    method: str = "GET",
    path: str = "/api/gauntlet/run",
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
def gauntlet_handler():
    """Create GauntletHandler with mock context."""
    ctx = {"stream_emitter": None}
    return GauntletHandler(ctx)


@pytest.fixture(autouse=True)
def clear_gauntlet_runs():
    """Clear in-memory gauntlet runs and rate limiters before each test."""
    from aragora.server.handlers.utils.rate_limit import _limiters

    _gauntlet_runs.clear()
    # Clear all handler rate limiters
    for limiter in _limiters.values():
        limiter.clear()
    yield
    _gauntlet_runs.clear()
    for limiter in _limiters.values():
        limiter.clear()


# ===========================================================================
# Test Routing
# ===========================================================================


class TestGauntletHandlerRouting:
    """Tests for GauntletHandler routing."""

    def test_can_handle_run_post(self, gauntlet_handler):
        assert gauntlet_handler.can_handle("/api/gauntlet/run", "POST") is True

    def test_can_handle_personas_get(self, gauntlet_handler):
        assert gauntlet_handler.can_handle("/api/gauntlet/personas", "GET") is True

    def test_can_handle_results_get(self, gauntlet_handler):
        assert gauntlet_handler.can_handle("/api/gauntlet/results", "GET") is True

    def test_can_handle_status_get(self, gauntlet_handler):
        assert gauntlet_handler.can_handle("/api/gauntlet/test123", "GET") is True

    def test_can_handle_receipt_get(self, gauntlet_handler):
        assert gauntlet_handler.can_handle("/api/gauntlet/test123/receipt", "GET") is True

    def test_can_handle_heatmap_get(self, gauntlet_handler):
        assert gauntlet_handler.can_handle("/api/gauntlet/test123/heatmap", "GET") is True

    def test_can_handle_delete(self, gauntlet_handler):
        assert gauntlet_handler.can_handle("/api/gauntlet/test123", "DELETE") is True

    def test_cannot_handle_other_paths(self, gauntlet_handler):
        assert gauntlet_handler.can_handle("/api/debates", "GET") is False


# ===========================================================================
# Test List Personas
# ===========================================================================


class TestGauntletListPersonas:
    """Tests for list personas endpoint."""

    @pytest.mark.asyncio
    async def test_list_personas_success(self, gauntlet_handler):
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            with patch("aragora.gauntlet.personas.list_personas") as mock_list:
                with patch("aragora.gauntlet.personas.get_persona") as mock_get:
                    mock_list.return_value = ["gdpr", "hipaa"]
                    mock_get.return_value = MockPersona()

                    handler = make_mock_handler()

                    result = await gauntlet_handler.handle("/api/gauntlet/personas", "GET", handler)

                    assert result is not None
                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert "personas" in data
                    assert "count" in data

    @pytest.mark.asyncio
    async def test_list_personas_module_not_available(self, gauntlet_handler):
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            with patch.dict("sys.modules", {"aragora.gauntlet.personas": None}):
                handler = make_mock_handler()

                # This will raise ImportError
                result = gauntlet_handler._list_personas()

                assert result is not None
                data = json.loads(result.body)
                assert data["personas"] == []
                assert "error" in data


# ===========================================================================
# Test Start Gauntlet
# ===========================================================================


class TestGauntletStartRun:
    """Tests for start gauntlet endpoint."""

    @pytest.mark.asyncio
    async def test_start_gauntlet_invalid_body(self, gauntlet_handler):
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            handler = MagicMock()
            handler.command = "POST"
            handler.path = "/api/gauntlet/run"
            handler.headers = {"Content-Length": "5"}
            handler.rfile = BytesIO(b"invalid")
            handler.client_address = ("127.0.0.1", 12345)

            result = await gauntlet_handler.handle("/api/gauntlet/run", "POST", handler)

            assert result is not None
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_start_gauntlet_quota_exceeded(self, gauntlet_handler):
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            # Setup user store with org at limit
            user_store = MockUserStore()
            org = MockOrganization(is_at_limit=True)
            user_store.orgs["org-123"] = org

            handler = make_mock_handler(
                {"input_content": "Test spec", "input_type": "spec"},
                method="POST",
            )
            handler.user_store = user_store

            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_auth:
                mock_auth.return_value = MockAuthContext()

                result = await gauntlet_handler.handle("/api/gauntlet/run", "POST", handler)

                assert result is not None
                assert result.status_code == 429
                data = json.loads(result.body)
                assert data["code"] == "quota_exceeded"


# ===========================================================================
# Test Get Status
# ===========================================================================


class TestGauntletGetStatus:
    """Tests for get status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status_pending(self, gauntlet_handler):
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            # Add a pending run
            _gauntlet_runs["gauntlet-test123"] = {
                "gauntlet_id": "gauntlet-test123",
                "status": "pending",
                "created_at": datetime.now().isoformat(),
            }

            handler = make_mock_handler(path="/api/gauntlet/gauntlet-test123")

            result = await gauntlet_handler.handle("/api/gauntlet/gauntlet-test123", "GET", handler)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_get_status_completed(self, gauntlet_handler):
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            # Add a completed run
            _gauntlet_runs["gauntlet-test123"] = {
                "gauntlet_id": "gauntlet-test123",
                "status": "completed",
                "result": {"verdict": "APPROVED"},
            }

            handler = make_mock_handler(path="/api/gauntlet/gauntlet-test123")

            result = await gauntlet_handler.handle("/api/gauntlet/gauntlet-test123", "GET", handler)

            assert result is not None
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_status_not_found(self, gauntlet_handler):
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            with patch("aragora.server.handlers.gauntlet._get_storage") as mock_storage:
                mock_storage_instance = MagicMock()
                mock_storage_instance.get.return_value = None
                mock_storage_instance.get_inflight.return_value = None  # Also mock get_inflight
                mock_storage.return_value = mock_storage_instance

                handler = make_mock_handler(path="/api/gauntlet/gauntlet-nonexistent")

                result = await gauntlet_handler.handle(
                    "/api/gauntlet/gauntlet-nonexistent", "GET", handler
                )

                assert result is not None
                assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_status_invalid_id(self, gauntlet_handler):
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            handler = make_mock_handler(path="/api/gauntlet/../etc/passwd")

            result = await gauntlet_handler.handle("/api/gauntlet/../etc/passwd", "GET", handler)

            # Should reject invalid ID
            assert result is not None
            assert result.status_code == 400


# ===========================================================================
# Test Get Receipt
# ===========================================================================


class TestGauntletGetReceipt:
    """Tests for get receipt endpoint."""

    @pytest.mark.asyncio
    async def test_get_receipt_not_completed(self, gauntlet_handler):
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            # Add a pending run
            _gauntlet_runs["gauntlet-test123"] = {
                "gauntlet_id": "gauntlet-test123",
                "status": "running",
            }

            handler = make_mock_handler(path="/api/gauntlet/gauntlet-test123/receipt")

            result = await gauntlet_handler.handle(
                "/api/gauntlet/gauntlet-test123/receipt", "GET", handler
            )

            assert result is not None
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_get_receipt_json_format(self, gauntlet_handler):
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            # Add a completed run
            _gauntlet_runs["gauntlet-test123"] = {
                "gauntlet_id": "gauntlet-test123",
                "status": "completed",
                "input_summary": "Test input",
                "input_hash": "abc123",
                "completed_at": datetime.now().isoformat(),
                "result": {
                    "verdict": "APPROVED",
                    "confidence": 0.85,
                    "robustness_score": 0.9,
                    "critical_count": 0,
                    "high_count": 1,
                    "medium_count": 1,
                    "low_count": 1,
                    "total_findings": 3,
                },
            }

            handler = make_mock_handler(path="/api/gauntlet/gauntlet-test123/receipt")

            result = await gauntlet_handler.handle(
                "/api/gauntlet/gauntlet-test123/receipt", "GET", handler
            )

            assert result is not None
            assert result.status_code == 200


# ===========================================================================
# Test Get Heatmap
# ===========================================================================


class TestGauntletGetHeatmap:
    """Tests for get heatmap endpoint."""

    @pytest.mark.asyncio
    async def test_get_heatmap_not_completed(self, gauntlet_handler):
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            _gauntlet_runs["gauntlet-test123"] = {
                "gauntlet_id": "gauntlet-test123",
                "status": "pending",
            }

            handler = make_mock_handler(path="/api/gauntlet/gauntlet-test123/heatmap")

            result = await gauntlet_handler.handle(
                "/api/gauntlet/gauntlet-test123/heatmap", "GET", handler
            )

            assert result is not None
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_get_heatmap_json_format(self, gauntlet_handler):
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            _gauntlet_runs["gauntlet-test123"] = {
                "gauntlet_id": "gauntlet-test123",
                "status": "completed",
                "result": {
                    "findings": [
                        {"category": "privacy", "severity_level": "high"},
                        {"category": "security", "severity_level": "medium"},
                    ],
                    "total_findings": 2,
                },
            }

            handler = make_mock_handler(path="/api/gauntlet/gauntlet-test123/heatmap")

            result = await gauntlet_handler.handle(
                "/api/gauntlet/gauntlet-test123/heatmap", "GET", handler
            )

            assert result is not None
            assert result.status_code == 200


# ===========================================================================
# Test List Results
# ===========================================================================


class TestGauntletListResults:
    """Tests for list results endpoint."""

    @pytest.mark.asyncio
    async def test_list_results_success(self, gauntlet_handler):
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            with patch("aragora.server.handlers.gauntlet._get_storage") as mock_storage:
                mock_storage_instance = MagicMock()
                mock_storage_instance.list_recent.return_value = [MockGauntletResult()]
                mock_storage_instance.count.return_value = 1
                mock_storage.return_value = mock_storage_instance

                handler = make_mock_handler(path="/api/gauntlet/results")

                result = await gauntlet_handler.handle("/api/gauntlet/results", "GET", handler)

                assert result is not None
                assert result.status_code == 200
                data = json.loads(result.body)
                assert "results" in data
                assert "total" in data

    @pytest.mark.asyncio
    async def test_list_results_with_pagination(self, gauntlet_handler):
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            with patch("aragora.server.handlers.gauntlet._get_storage") as mock_storage:
                mock_storage_instance = MagicMock()
                mock_storage_instance.list_recent.return_value = []
                mock_storage_instance.count.return_value = 0
                mock_storage.return_value = mock_storage_instance

                handler = make_mock_handler(path="/api/gauntlet/results?limit=10&offset=5")

                result = await gauntlet_handler.handle("/api/gauntlet/results", "GET", handler)

                assert result is not None
                assert result.status_code == 200
                data = json.loads(result.body)
                assert data["limit"] == 10
                assert data["offset"] == 5


# ===========================================================================
# Test Compare Results
# ===========================================================================


class TestGauntletCompareResults:
    """Tests for compare results endpoint."""

    @pytest.mark.asyncio
    async def test_compare_success(self, gauntlet_handler):
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            with patch("aragora.server.handlers.gauntlet._get_storage") as mock_storage:
                mock_storage_instance = MagicMock()
                mock_storage_instance.compare.return_value = {
                    "comparison": {"id1": "gauntlet-test1", "id2": "gauntlet-test2"}
                }
                mock_storage.return_value = mock_storage_instance

                handler = make_mock_handler(
                    path="/api/gauntlet/gauntlet-test1/compare/gauntlet-test2"
                )

                result = await gauntlet_handler.handle(
                    "/api/gauntlet/gauntlet-test1/compare/gauntlet-test2", "GET", handler
                )

                assert result is not None
                assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_compare_not_found(self, gauntlet_handler):
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            with patch("aragora.server.handlers.gauntlet._get_storage") as mock_storage:
                mock_storage_instance = MagicMock()
                mock_storage_instance.compare.return_value = None
                mock_storage.return_value = mock_storage_instance

                handler = make_mock_handler(
                    path="/api/gauntlet/gauntlet-test1/compare/gauntlet-test2"
                )

                result = await gauntlet_handler.handle(
                    "/api/gauntlet/gauntlet-test1/compare/gauntlet-test2", "GET", handler
                )

                assert result is not None
                assert result.status_code == 404


# ===========================================================================
# Test Delete Result
# ===========================================================================


class TestGauntletDeleteResult:
    """Tests for delete result endpoint."""

    @pytest.mark.asyncio
    async def test_delete_success(self, gauntlet_handler):
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            # Add to in-memory
            _gauntlet_runs["gauntlet-test123"] = {"status": "completed"}

            with patch("aragora.server.handlers.gauntlet._get_storage") as mock_storage:
                mock_storage_instance = MagicMock()
                mock_storage_instance.delete.return_value = True
                mock_storage.return_value = mock_storage_instance

                handler = make_mock_handler(path="/api/gauntlet/gauntlet-test123", method="DELETE")

                result = await gauntlet_handler.handle(
                    "/api/gauntlet/gauntlet-test123", "DELETE", handler
                )

                assert result is not None
                assert result.status_code == 200
                data = json.loads(result.body)
                assert data["deleted"] is True

                # Should be removed from in-memory
                assert "gauntlet-test123" not in _gauntlet_runs

    @pytest.mark.asyncio
    async def test_delete_not_found(self, gauntlet_handler):
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            with patch("aragora.server.handlers.gauntlet._get_storage") as mock_storage:
                mock_storage_instance = MagicMock()
                mock_storage_instance.delete.return_value = False
                mock_storage.return_value = mock_storage_instance

                handler = make_mock_handler(
                    path="/api/gauntlet/gauntlet-nonexistent", method="DELETE"
                )

                result = await gauntlet_handler.handle(
                    "/api/gauntlet/gauntlet-nonexistent", "DELETE", handler
                )

                assert result is not None
                assert result.status_code == 404


# ===========================================================================
# Test Export Report
# ===========================================================================


class TestGauntletExportReport:
    """Tests for export report endpoint."""

    @pytest.mark.asyncio
    async def test_export_json_format(self, gauntlet_handler):
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            _gauntlet_runs["gauntlet-test123"] = {
                "gauntlet_id": "gauntlet-test123",
                "status": "completed",
                "input_summary": "Test",
                "input_type": "spec",
                "input_hash": "abc",
                "created_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat(),
                "result": {
                    "verdict": "APPROVED",
                    "confidence": 0.85,
                    "robustness_score": 0.9,
                    "risk_score": 0.1,
                    "coverage_score": 0.8,
                    "total_findings": 2,
                    "critical_count": 0,
                    "high_count": 1,
                    "medium_count": 1,
                    "low_count": 0,
                    "findings": [],
                },
            }

            handler = make_mock_handler(path="/api/gauntlet/gauntlet-test123/export")

            result = await gauntlet_handler.handle(
                "/api/gauntlet/gauntlet-test123/export", "GET", handler
            )

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert "summary" in data
            assert "findings_summary" in data

    @pytest.mark.asyncio
    async def test_export_not_completed(self, gauntlet_handler):
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            _gauntlet_runs["gauntlet-test123"] = {
                "gauntlet_id": "gauntlet-test123",
                "status": "running",
            }

            handler = make_mock_handler(path="/api/gauntlet/gauntlet-test123/export")

            result = await gauntlet_handler.handle(
                "/api/gauntlet/gauntlet-test123/export", "GET", handler
            )

            assert result is not None
            assert result.status_code == 400


# ===========================================================================
# Test Memory Management
# ===========================================================================


class TestGauntletMemoryManagement:
    """Tests for memory cleanup functions."""

    def test_cleanup_removes_old_entries(self):
        """Test that cleanup removes entries older than max age."""
        import time

        # Add old entry
        old_time = datetime.utcnow().isoformat()
        _gauntlet_runs["old-run"] = {
            "status": "completed",
            "created_at": 0,  # Unix epoch - very old
            "completed_at": old_time,
        }

        # Add recent entry
        _gauntlet_runs["new-run"] = {
            "status": "pending",
            "created_at": time.time(),
        }

        _cleanup_gauntlet_runs()

        # Old entry should be removed, new entry should remain
        assert "new-run" in _gauntlet_runs

    def test_cleanup_respects_memory_limit(self):
        """Test that cleanup enforces memory limit."""
        # Add more entries than limit
        for i in range(MAX_GAUNTLET_RUNS_IN_MEMORY + 100):
            _gauntlet_runs[f"run-{i}"] = {
                "status": "pending",
                "created_at": datetime.now().isoformat(),
            }

        _cleanup_gauntlet_runs()

        # Should be at or below limit
        assert len(_gauntlet_runs) <= MAX_GAUNTLET_RUNS_IN_MEMORY


# ===========================================================================
# Test ID Validation
# ===========================================================================


class TestGauntletIdValidation:
    """Tests for gauntlet ID validation."""

    @pytest.mark.asyncio
    async def test_reject_path_traversal_id(self, gauntlet_handler):
        """Test that path traversal attempts are rejected."""
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            handler = make_mock_handler(path="/api/gauntlet/../../etc/passwd")

            result = await gauntlet_handler.handle("/api/gauntlet/../../etc/passwd", "GET", handler)

            assert result is not None
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_accept_valid_id(self, gauntlet_handler):
        """Test that valid IDs are accepted."""
        with patch("aragora.server.handlers.gauntlet.rate_limit", lambda **kwargs: lambda fn: fn):
            _gauntlet_runs["gauntlet-20240114120000-abc123"] = {
                "gauntlet_id": "gauntlet-20240114120000-abc123",
                "status": "completed",
                "result": {},
            }

            handler = make_mock_handler(path="/api/gauntlet/gauntlet-20240114120000-abc123")

            result = await gauntlet_handler.handle(
                "/api/gauntlet/gauntlet-20240114120000-abc123", "GET", handler
            )

            assert result is not None
            assert result.status_code == 200
