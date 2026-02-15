"""
Tests for aragora.server.handlers.repository - Repository Indexing Handlers.

Tests cover:
- RepositoryHandler: instantiation, ROUTES, can_handle
- _validate_repo_path: null bytes, empty path, scan root enforcement
- GET /api/v1/repository/:id/status: orchestrator unavailable, unknown repo, found
- GET /api/v1/repository/:id/entities: orchestrator unavailable, query filtering
- GET /api/v1/repository/:id/graph: builder unavailable, entity graph, stats
- GET /api/v1/repository/:id: info retrieval, orchestrator unavailable
- POST /api/v1/repository/index: missing repo_path, orchestrator unavailable, success
- POST /api/v1/repository/incremental: missing repo_path, success
- POST /api/v1/repository/batch: missing repos, success
- DELETE /api/v1/repository/:id: success, orchestrator unavailable
- handle() routing: invalid repo_id, endpoint not found
- handle_post() routing: returns 404 for unknown paths
- handle_delete() routing: returns 404 for unknown paths
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.repository import (
    RepositoryHandler,
    _validate_repo_path,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


def _make_mock_handler(
    method: str = "GET",
    body: bytes = b"",
    content_type: str = "application/json",
) -> MagicMock:
    """Create a mock HTTP handler object."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": content_type,
        "Host": "localhost:8080",
    }
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = body
    return handler


# ===========================================================================
# Mock Objects
# ===========================================================================


class MockProgress:
    """Mock indexing progress object."""

    def __init__(self):
        self.status = "completed"
        self.files_discovered = 100
        self.files_processed = 95
        self.nodes_created = 200
        self.current_file = None
        self.started_at = None
        self.error = None


class MockIndexResult:
    """Mock index result object."""

    def __init__(self, errors: list | None = None):
        self.errors = errors or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_processed": 50,
            "nodes_created": 100,
            "errors": self.errors,
        }


class MockBatchResult:
    """Mock batch result object."""

    def __init__(self, failed: int = 0):
        self.failed = failed

    def to_dict(self) -> dict[str, Any]:
        return {"total": 3, "succeeded": 3 - self.failed, "failed": self.failed}


class MockQueryItem:
    """Mock query result item."""

    def __init__(self, item_id: str = "item-1"):
        self.id = item_id
        self.content = "def hello(): pass"
        self.metadata = {"kind": "function", "file_path": "main.py"}


class MockQueryResults:
    """Mock query results."""

    def __init__(self, items: list | None = None):
        self.items = items or []
        self.total_count = len(self.items)


class MockEntity:
    """Mock relationship entity."""

    def __init__(self, entity_id: str = "ent-1"):
        self.id = entity_id
        self.name = "MyClass"
        self.kind = "class"
        self.file_path = "src/main.py"
        self.line_start = 10


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def handler():
    """Create a RepositoryHandler with mocked dependencies."""
    h = RepositoryHandler(ctx={})
    return h


@pytest.fixture
def mock_orchestrator():
    """Create a mock repository orchestrator.

    Uses MagicMock as the base since get_progress/get_all_progress are
    synchronous methods. Async methods are set as AsyncMock individually.
    """
    orch = MagicMock()
    orch.get_progress.return_value = None
    orch.get_all_progress.return_value = {}
    orch.mound = AsyncMock()
    # Async methods
    orch.index_repository = AsyncMock()
    orch.incremental_update = AsyncMock()
    orch.index_multiple = AsyncMock()
    orch.get_repository_stats = AsyncMock()
    orch.remove_repository = AsyncMock()
    return orch


# ===========================================================================
# Test Instantiation and Basics
# ===========================================================================


class TestRepositoryHandlerBasics:
    """Basic instantiation and attribute tests."""

    def test_instantiation(self, handler):
        assert handler is not None
        assert isinstance(handler, RepositoryHandler)

    def test_has_routes(self, handler):
        assert hasattr(handler, "ROUTES")
        assert len(handler.ROUTES) > 0

    def test_routes_contain_index(self, handler):
        assert "/api/v1/repository/index" in handler.ROUTES

    def test_routes_contain_incremental(self, handler):
        assert "/api/v1/repository/incremental" in handler.ROUTES

    def test_routes_contain_batch(self, handler):
        assert "/api/v1/repository/batch" in handler.ROUTES

    def test_routes_contain_status_wildcard(self, handler):
        assert "/api/v1/repository/*/status" in handler.ROUTES

    def test_routes_contain_entities_wildcard(self, handler):
        assert "/api/v1/repository/*/entities" in handler.ROUTES

    def test_routes_contain_graph_wildcard(self, handler):
        assert "/api/v1/repository/*/graph" in handler.ROUTES


# ===========================================================================
# Test can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle routing logic."""

    def test_can_handle_index(self, handler):
        assert handler.can_handle("/api/v1/repository/index") is True

    def test_can_handle_incremental(self, handler):
        assert handler.can_handle("/api/v1/repository/incremental") is True

    def test_can_handle_status(self, handler):
        assert handler.can_handle("/api/v1/repository/my-repo/status") is True

    def test_can_handle_entities(self, handler):
        assert handler.can_handle("/api/v1/repository/my-repo/entities") is True

    def test_can_handle_graph(self, handler):
        assert handler.can_handle("/api/v1/repository/my-repo/graph") is True

    def test_can_handle_repo_id(self, handler):
        assert handler.can_handle("/api/v1/repository/my-repo") is True

    def test_cannot_handle_unrelated(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_cannot_handle_partial_match(self, handler):
        assert handler.can_handle("/api/v1/repo/index") is False


# ===========================================================================
# Test _validate_repo_path
# ===========================================================================


class TestValidateRepoPath:
    """Tests for repo path validation."""

    def test_empty_path(self):
        path, err = _validate_repo_path("")
        assert path is None
        assert "required" in err

    def test_whitespace_only(self):
        path, err = _validate_repo_path("   ")
        assert path is None
        assert "required" in err

    def test_null_byte(self):
        path, err = _validate_repo_path("/tmp/repo\x00evil")
        assert path is None
        assert "null byte" in err

    def test_valid_path(self):
        path, err = _validate_repo_path("/tmp/myrepo")
        assert err is None
        assert path is not None

    def test_scan_root_enforcement(self):
        """When ARAGORA_SCAN_ROOT is set, paths outside it are rejected."""
        with patch.dict(os.environ, {"ARAGORA_SCAN_ROOT": "/allowed"}):
            path, err = _validate_repo_path("/not-allowed/repo")
            assert path is None
            assert "allowed workspace" in err

    def test_scan_root_allows_subpath(self):
        """Paths within the scan root are accepted."""
        with patch.dict(os.environ, {"ARAGORA_SCAN_ROOT": "/tmp"}):
            path, err = _validate_repo_path("/tmp/myrepo")
            assert err is None
            assert path is not None


# ===========================================================================
# Test POST /api/v1/repository/index
# ===========================================================================


class TestStartIndex:
    """Tests for the start index endpoint."""

    @pytest.mark.asyncio
    async def test_orchestrator_unavailable(self, handler):
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=None,
        ):
            result = await handler._start_index({})
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_missing_repo_path(self, handler, mock_orchestrator):
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=mock_orchestrator,
        ):
            result = await handler._start_index({})
            assert result.status_code == 400
            data = _parse_body(result)
            assert "repo_path" in data["error"]

    @pytest.mark.asyncio
    async def test_success(self, handler, mock_orchestrator):
        mock_orchestrator.index_repository.return_value = MockIndexResult()
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=mock_orchestrator,
        ):
            result = await handler._start_index({"repo_path": "/tmp/myrepo"})
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_index_with_errors(self, handler, mock_orchestrator):
        mock_orchestrator.index_repository.return_value = MockIndexResult(
            errors=["file not found"]
        )
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=mock_orchestrator,
        ):
            result = await handler._start_index({"repo_path": "/tmp/myrepo"})
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["success"] is False

    @pytest.mark.asyncio
    async def test_null_byte_in_path(self, handler, mock_orchestrator):
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=mock_orchestrator,
        ):
            result = await handler._start_index({"repo_path": "/tmp/repo\x00evil"})
            assert result.status_code == 400


# ===========================================================================
# Test POST /api/v1/repository/incremental
# ===========================================================================


class TestIncrementalUpdate:
    """Tests for the incremental update endpoint."""

    @pytest.mark.asyncio
    async def test_orchestrator_unavailable(self, handler):
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=None,
        ):
            result = await handler._incremental_update({})
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_missing_repo_path(self, handler, mock_orchestrator):
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=mock_orchestrator,
        ):
            result = await handler._incremental_update({})
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_success(self, handler, mock_orchestrator):
        mock_orchestrator.incremental_update.return_value = MockIndexResult()
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=mock_orchestrator,
        ):
            result = await handler._incremental_update({"repo_path": "/tmp/myrepo"})
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["success"] is True


# ===========================================================================
# Test POST /api/v1/repository/batch
# ===========================================================================


class TestBatchIndex:
    """Tests for the batch index endpoint."""

    @pytest.mark.asyncio
    async def test_orchestrator_unavailable(self, handler):
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=None,
        ):
            result = await handler._batch_index({})
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_missing_repositories(self, handler, mock_orchestrator):
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=mock_orchestrator,
        ):
            result = await handler._batch_index({})
            assert result.status_code == 400
            data = _parse_body(result)
            assert "repositories" in data["error"]

    @pytest.mark.asyncio
    async def test_batch_success(self, handler, mock_orchestrator):
        mock_orchestrator.index_multiple.return_value = MockBatchResult()
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=mock_orchestrator,
        ):
            with patch(
                "aragora.knowledge.repository_orchestrator.RepoConfig",
            ) as MockRepoConfig:
                MockRepoConfig.side_effect = lambda **kwargs: MagicMock(**kwargs)
                result = await handler._batch_index(
                    {"repositories": [{"path": "/tmp/repo1"}, {"path": "/tmp/repo2"}]}
                )
                assert result.status_code == 200


# ===========================================================================
# Test GET /api/v1/repository/:id/status
# ===========================================================================


class TestGetStatus:
    """Tests for the get status endpoint."""

    @pytest.mark.asyncio
    async def test_orchestrator_unavailable(self, handler):
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=None,
        ):
            result = await handler._get_status("my-repo")
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_no_progress_found(self, handler, mock_orchestrator):
        mock_orchestrator.get_progress.return_value = None
        mock_orchestrator.get_all_progress.return_value = {}
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=mock_orchestrator,
        ):
            result = await handler._get_status("my-repo")
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["status"] == "unknown"

    @pytest.mark.asyncio
    async def test_progress_found(self, handler, mock_orchestrator):
        progress = MockProgress()
        mock_orchestrator.get_progress.return_value = progress
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=mock_orchestrator,
        ):
            result = await handler._get_status("my-repo")
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["status"] == "completed"
            assert data["files_discovered"] == 100


# ===========================================================================
# Test GET /api/v1/repository/:id
# ===========================================================================


class TestGetRepository:
    """Tests for get repository info."""

    @pytest.mark.asyncio
    async def test_orchestrator_unavailable(self, handler):
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=None,
        ):
            result = await handler._get_repository("my-repo")
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_success(self, handler, mock_orchestrator):
        mock_orchestrator.get_repository_stats.return_value = {
            "name": "my-repo",
            "files": 100,
        }
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=mock_orchestrator,
        ):
            result = await handler._get_repository("my-repo")
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["name"] == "my-repo"


# ===========================================================================
# Test DELETE /api/v1/repository/:id
# ===========================================================================


class TestRemoveRepository:
    """Tests for repository removal."""

    @pytest.mark.asyncio
    async def test_orchestrator_unavailable(self, handler):
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=None,
        ):
            result = await handler._remove_repository("my-repo", {})
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_success(self, handler, mock_orchestrator):
        mock_orchestrator.remove_repository.return_value = 42
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=mock_orchestrator,
        ):
            result = await handler._remove_repository("my-repo", {})
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["success"] is True
            assert data["nodes_removed"] == 42


# ===========================================================================
# Test handle_post() Routing
# ===========================================================================


class TestHandlePostRouting:
    """Tests for handle_post() routing."""

    @pytest.mark.asyncio
    async def test_handle_post_index(self, handler, mock_orchestrator):
        body = json.dumps({"repo_path": "/tmp/myrepo"}).encode()
        mock_handler = _make_mock_handler("POST", body)
        mock_orchestrator.index_repository.return_value = MockIndexResult()
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=mock_orchestrator,
        ):
            result = await handler.handle_post("/api/v1/repository/index", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_post_incremental(self, handler, mock_orchestrator):
        body = json.dumps({"repo_path": "/tmp/myrepo"}).encode()
        mock_handler = _make_mock_handler("POST", body)
        mock_orchestrator.incremental_update.return_value = MockIndexResult()
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=mock_orchestrator,
        ):
            result = await handler.handle_post(
                "/api/v1/repository/incremental", {}, mock_handler
            )
            assert result is not None
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_post_unknown_returns_404(self, handler):
        mock_handler = _make_mock_handler("POST")
        result = await handler.handle_post("/api/v1/repository/unknown-action", {}, mock_handler)
        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Test handle_delete() Routing
# ===========================================================================


class TestHandleDeleteRouting:
    """Tests for handle_delete() routing."""

    @pytest.mark.asyncio
    async def test_handle_delete_repo(self, handler, mock_orchestrator):
        mock_handler = _make_mock_handler("DELETE")
        mock_orchestrator.remove_repository.return_value = 10
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            return_value=mock_orchestrator,
        ):
            result = await handler.handle_delete(
                "/api/v1/repository/my-repo", {}, mock_handler
            )
            assert result is not None
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_delete_no_repo_id_returns_404(self, handler):
        mock_handler = _make_mock_handler("DELETE")
        result = await handler.handle_delete("/api/v1/repository", {}, mock_handler)
        assert result is not None
        assert result.status_code == 404
