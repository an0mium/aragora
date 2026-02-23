"""Tests for repository handler.

Covers:
- Path traversal validation (_validate_repo_path)
- Route matching (can_handle)
- GET endpoints (status, entities, graph, repository info)
- POST endpoints (index, incremental, batch)
- DELETE endpoint (remove repository)
- Orchestrator unavailability (503)
- Invalid repo_id rejection
- Error handling and response codes
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


def _body(result) -> dict:
    """Parse HandlerResult.body bytes into dict."""
    return json.loads(result.body)


def _make_repo_handler(body: dict[str, Any] | None = None) -> MagicMock:
    """Create a mock HTTP handler with proper headers."""
    mock = MagicMock()
    if body:
        body_bytes = json.dumps(body).encode()
    else:
        body_bytes = b"{}"
    _headers = {"Content-Length": str(len(body_bytes))}
    mock.rfile.read.return_value = body_bytes
    mock.headers = MagicMock()
    mock.headers.get = lambda k, d=0: _headers.get(k, str(d))
    return mock


# ============================================================================
# Path Traversal Validation
# ============================================================================


class TestValidateRepoPath:
    """Test path traversal prevention."""

    def test_empty_path_rejected(self):
        resolved, err = _validate_repo_path("")
        assert resolved is None
        assert "required" in err

    def test_whitespace_path_rejected(self):
        resolved, err = _validate_repo_path("   ")
        assert resolved is None
        assert "required" in err

    def test_null_byte_rejected(self):
        resolved, err = _validate_repo_path("/tmp/repo\x00/evil")
        assert resolved is None
        assert "null byte" in err

    def test_valid_path_resolves(self):
        resolved, err = _validate_repo_path("/tmp")
        assert err is None
        assert resolved is not None
        assert "\x00" not in resolved

    def test_scan_root_allows_child(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", "/tmp")
        resolved, err = _validate_repo_path("/tmp/my_repo")
        assert err is None
        assert resolved is not None

    def test_scan_root_rejects_outside(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", "/tmp/allowed")
        resolved, err = _validate_repo_path("/etc/passwd")
        assert resolved is None
        assert "allowed workspace" in err

    def test_scan_root_exact_match_allowed(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", "/tmp")
        resolved, err = _validate_repo_path("/tmp")
        assert err is None

    def test_scan_root_prefix_attack_prevented(self, monkeypatch):
        """Ensure /tmp_evil doesn't match scan root /tmp."""
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", "/tmp")
        # /tmp_evil would match startswith("/tmp") without trailing sep
        resolved, err = _validate_repo_path("/tmp_evil")
        assert resolved is None
        assert "allowed workspace" in err

    def test_no_scan_root_allows_any(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_SCAN_ROOT", raising=False)
        resolved, err = _validate_repo_path("/usr/local/repos")
        assert err is None

    def test_root_scan_root_allows_all(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", "/")
        resolved, err = _validate_repo_path("/any/path/here")
        assert err is None


# ============================================================================
# Route Matching
# ============================================================================


class TestRepositoryCanHandle:
    """Test route matching."""

    def setup_method(self):
        self.handler = RepositoryHandler()

    def test_index_route(self):
        assert self.handler.can_handle("/api/v1/repository/index") is True

    def test_incremental_route(self):
        assert self.handler.can_handle("/api/v1/repository/incremental") is True

    def test_batch_route(self):
        assert self.handler.can_handle("/api/v1/repository/batch") is True

    def test_status_route(self):
        assert self.handler.can_handle("/api/v1/repository/abc123/status") is True

    def test_entities_route(self):
        assert self.handler.can_handle("/api/v1/repository/abc123/entities") is True

    def test_graph_route(self):
        assert self.handler.can_handle("/api/v1/repository/abc123/graph") is True

    def test_repo_info_route(self):
        assert self.handler.can_handle("/api/v1/repository/abc123") is True

    def test_unrelated_route(self):
        assert self.handler.can_handle("/api/v1/debates") is False

    def test_partial_prefix_no_match(self):
        assert self.handler.can_handle("/api/v1/repository") is False


# ============================================================================
# GET Endpoints
# ============================================================================


class TestRepositoryGet:
    """Test GET request handling."""

    def setup_method(self):
        self.handler = RepositoryHandler()

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.repository._get_orchestrator",
        new_callable=AsyncMock,
        return_value=None,
    )
    async def test_status_orchestrator_unavailable(self, mock_orch):
        result = await self.handler._get_status("repo1")
        assert result.status_code == 503

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.repository._get_orchestrator", new_callable=AsyncMock)
    async def test_status_no_progress(self, mock_orch):
        orch = MagicMock()
        orch.get_progress.return_value = None
        orch.get_all_progress.return_value = {}
        mock_orch.return_value = orch

        result = await self.handler._get_status("repo1")
        assert result.status_code == 200
        assert _body(result)["status"] == "unknown"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.repository._get_orchestrator", new_callable=AsyncMock)
    async def test_status_with_progress(self, mock_orch):
        progress = MagicMock()
        progress.status = "indexing"
        progress.files_discovered = 100
        progress.files_processed = 50
        progress.nodes_created = 200
        progress.current_file = "src/main.py"
        progress.started_at = None
        progress.error = None

        orch = MagicMock()
        orch.get_progress.return_value = progress
        mock_orch.return_value = orch

        result = await self.handler._get_status("repo1")
        assert result.status_code == 200
        body = _body(result)
        assert body["status"] == "indexing"
        assert body["files_discovered"] == 100
        assert body["files_processed"] == 50

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.repository._get_orchestrator",
        new_callable=AsyncMock,
        return_value=None,
    )
    async def test_entities_orchestrator_unavailable(self, mock_orch):
        result = await self.handler._get_entities("repo1", {})
        assert result.status_code == 503

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.repository._get_orchestrator", new_callable=AsyncMock)
    async def test_entities_returns_list(self, mock_orch):
        item = MagicMock()
        item.id = "entity1"
        item.content = "function hello()"
        item.metadata = {"kind": "function"}

        query_result = MagicMock()
        query_result.items = [item]
        query_result.total_count = 1

        orch = MagicMock()
        orch.mound = MagicMock()
        orch.mound.query = AsyncMock(return_value=query_result)
        mock_orch.return_value = orch

        result = await self.handler._get_entities("repo1", {"kind": "function"})
        assert result.status_code == 200
        body = _body(result)
        assert len(body["entities"]) == 1
        assert body["entities"][0]["id"] == "entity1"

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.repository._get_relationship_builder",
        new_callable=AsyncMock,
        return_value=None,
    )
    async def test_graph_builder_unavailable(self, mock_builder):
        result = await self.handler._get_graph("repo1", {})
        assert result.status_code == 503

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.repository._get_relationship_builder", new_callable=AsyncMock)
    async def test_graph_statistics(self, mock_builder):
        builder = MagicMock()
        builder.get_statistics.return_value = {"nodes": 100, "edges": 50}
        mock_builder.return_value = builder

        result = await self.handler._get_graph("repo1", {})
        assert result.status_code == 200
        assert _body(result)["statistics"]["nodes"] == 100

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.repository._get_relationship_builder", new_callable=AsyncMock)
    async def test_graph_entity_dependencies(self, mock_builder):
        entity = MagicMock()
        entity.id = "e1"
        entity.name = "main"
        entity.kind = "function"
        entity.file_path = "src/main.py"
        entity.line_start = 10

        builder = MagicMock()
        builder.find_dependencies = AsyncMock(return_value=[entity])
        mock_builder.return_value = builder

        result = await self.handler._get_graph(
            "repo1",
            {"entity_id": "e1", "direction": "dependencies", "depth": "3"},
        )
        assert result.status_code == 200
        assert len(_body(result)["entities"]) == 1

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.repository._get_orchestrator",
        new_callable=AsyncMock,
        return_value=None,
    )
    async def test_get_repository_orchestrator_unavailable(self, mock_orch):
        result = await self.handler._get_repository("repo1")
        assert result.status_code == 503

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.repository._get_orchestrator", new_callable=AsyncMock)
    async def test_get_repository_success(self, mock_orch):
        orch = MagicMock()
        orch.get_repository_stats = AsyncMock(return_value={"name": "repo1", "files": 100})
        mock_orch.return_value = orch

        result = await self.handler._get_repository("repo1")
        assert result.status_code == 200
        assert _body(result)["name"] == "repo1"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.repository._get_orchestrator", new_callable=AsyncMock)
    async def test_get_repository_error(self, mock_orch):
        orch = MagicMock()
        orch.get_repository_stats = AsyncMock(side_effect=RuntimeError("db error"))
        mock_orch.return_value = orch

        result = await self.handler._get_repository("repo1")
        assert result.status_code == 500


# ============================================================================
# GET Dispatcher
# ============================================================================


class TestRepositoryHandleGet:
    """Test handle() GET dispatcher.

    Note: The handle() method uses path.split("/") indexing.
    We test the internal methods directly for coverage and use
    dispatcher tests mainly for routing validation.
    """

    def setup_method(self):
        self.handler = RepositoryHandler()

    @pytest.mark.asyncio
    async def test_dispatch_not_found(self):
        result = await self.handler.handle("/api/v1/repository/", {})
        assert result is not None
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_action_routes_skip_validation(self):
        """Verify index/incremental/batch don't trigger repo_id validation on GET."""
        result = await self.handler.handle("/api/v1/repository/index", {})
        # These are action routes, not repo IDs - should reach 404 for GET
        assert result.status_code == 404


# ============================================================================
# POST Endpoints
# ============================================================================


class TestRepositoryPost:
    """Test POST request handling."""

    def setup_method(self):
        self.handler = RepositoryHandler()

    def _make_handler(self, body: dict[str, Any] | None = None) -> MagicMock:
        mock = MagicMock()
        if body:
            body_bytes = json.dumps(body).encode()
            mock.rfile.read.return_value = body_bytes
            mock.headers = {"Content-Length": str(len(body_bytes))}
            mock.headers.get = lambda k, d=0: str(len(body_bytes)) if k == "Content-Length" else d
        else:
            mock.rfile.read.return_value = b"{}"
            mock.headers = {"Content-Length": "2"}
            mock.headers.get = lambda k, d=0: "2" if k == "Content-Length" else d
        return mock

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.repository._get_orchestrator",
        new_callable=AsyncMock,
        return_value=None,
    )
    async def test_index_orchestrator_unavailable(self, mock_orch):
        result = await self.handler._start_index({})
        assert result.status_code == 503

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.repository._get_orchestrator", new_callable=AsyncMock)
    async def test_index_missing_repo_path(self, mock_orch):
        mock_orch.return_value = MagicMock()
        result = await self.handler._start_index({})
        assert result.status_code == 400
        assert "repo_path" in _body(result)["error"]

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.repository._get_orchestrator", new_callable=AsyncMock)
    async def test_index_null_byte_in_path(self, mock_orch):
        mock_orch.return_value = MagicMock()
        result = await self.handler._start_index({"repo_path": "/tmp/repo\x00evil"})
        assert result.status_code == 400
        assert "null byte" in _body(result)["error"]

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.repository._get_orchestrator", new_callable=AsyncMock)
    async def test_index_path_outside_scan_root(self, mock_orch, monkeypatch):
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", "/tmp/allowed")
        mock_orch.return_value = MagicMock()
        result = await self.handler._start_index({"repo_path": "/etc/passwd"})
        assert result.status_code == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.repository._get_orchestrator", new_callable=AsyncMock)
    async def test_index_success(self, mock_orch):
        index_result = MagicMock()
        index_result.errors = []
        index_result.to_dict.return_value = {"files_indexed": 100}

        orch = MagicMock()
        orch.index_repository = AsyncMock(return_value=index_result)
        mock_orch.return_value = orch

        result = await self.handler._start_index(
            {"repo_path": "/tmp/my_repo", "workspace_id": "ws1"}
        )
        assert result.status_code == 200
        assert _body(result)["success"] is True

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.repository._get_orchestrator", new_callable=AsyncMock)
    async def test_incremental_missing_path(self, mock_orch):
        mock_orch.return_value = MagicMock()
        result = await self.handler._incremental_update({})
        assert result.status_code == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.repository._get_orchestrator", new_callable=AsyncMock)
    async def test_incremental_success(self, mock_orch):
        update_result = MagicMock()
        update_result.errors = []
        update_result.to_dict.return_value = {"files_updated": 5}

        orch = MagicMock()
        orch.incremental_update = AsyncMock(return_value=update_result)
        mock_orch.return_value = orch

        result = await self.handler._incremental_update({"repo_path": "/tmp/repo"})
        assert result.status_code == 200
        assert _body(result)["success"] is True

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.repository._get_orchestrator", new_callable=AsyncMock)
    async def test_batch_empty_repos(self, mock_orch):
        mock_orch.return_value = MagicMock()
        result = await self.handler._batch_index({"repositories": []})
        assert result.status_code == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.repository._get_orchestrator", new_callable=AsyncMock)
    async def test_batch_missing_repos(self, mock_orch):
        mock_orch.return_value = MagicMock()
        result = await self.handler._batch_index({})
        assert result.status_code == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.repository._get_orchestrator", new_callable=AsyncMock)
    async def test_batch_invalid_path_in_repo(self, mock_orch, monkeypatch):
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", "/tmp/allowed")
        mock_orch.return_value = MagicMock()

        result = await self.handler._batch_index(
            {
                "repositories": [
                    {"path": "/etc/passwd", "workspace_id": "default"},
                ]
            }
        )
        assert result.status_code == 400
        assert "Invalid path" in _body(result)["error"]


# ============================================================================
# POST Dispatcher
# ============================================================================


class TestRepositoryHandlePost:
    """Test handle_post() dispatcher."""

    def setup_method(self):
        self.handler = RepositoryHandler()

    @pytest.mark.asyncio
    async def test_dispatch_index(self):
        mock_handler = _make_repo_handler({"repo_path": "/tmp"})

        with patch.object(self.handler, "_start_index", new_callable=AsyncMock) as mock_start:
            mock_start.return_value = MagicMock(status_code=200)
            result = await self.handler.handle_post("/api/v1/repository/index", {}, mock_handler)
            mock_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_incremental(self):
        mock_handler = _make_repo_handler()

        with patch.object(self.handler, "_incremental_update", new_callable=AsyncMock) as mock_inc:
            mock_inc.return_value = MagicMock(status_code=200)
            result = await self.handler.handle_post(
                "/api/v1/repository/incremental", {}, mock_handler
            )
            mock_inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_batch(self):
        mock_handler = _make_repo_handler()

        with patch.object(self.handler, "_batch_index", new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = MagicMock(status_code=200)
            result = await self.handler.handle_post("/api/v1/repository/batch", {}, mock_handler)
            mock_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_unknown_post_404(self):
        mock_handler = _make_repo_handler()

        result = await self.handler.handle_post(
            "/api/v1/repository/unknown_action", {}, mock_handler
        )
        assert result.status_code == 404


# ============================================================================
# DELETE Endpoint
# ============================================================================


class TestRepositoryDelete:
    """Test DELETE request handling."""

    def setup_method(self):
        self.handler = RepositoryHandler()

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.repository._get_orchestrator",
        new_callable=AsyncMock,
        return_value=None,
    )
    async def test_remove_orchestrator_unavailable(self, mock_orch):
        result = await self.handler._remove_repository("repo1", {})
        assert result.status_code == 503

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.repository._get_orchestrator", new_callable=AsyncMock)
    async def test_remove_success(self, mock_orch):
        orch = MagicMock()
        orch.remove_repository = AsyncMock(return_value=42)
        mock_orch.return_value = orch

        result = await self.handler._remove_repository("repo1", {"workspace_id": "ws1"})
        assert result.status_code == 200
        body = _body(result)
        assert body["success"] is True
        assert body["nodes_removed"] == 42

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.repository._get_orchestrator", new_callable=AsyncMock)
    async def test_remove_error(self, mock_orch):
        orch = MagicMock()
        orch.remove_repository = AsyncMock(side_effect=RuntimeError("db error"))
        mock_orch.return_value = orch

        result = await self.handler._remove_repository("repo1", {})
        assert result.status_code == 500


# ============================================================================
# DELETE Dispatcher
# ============================================================================


class TestRepositoryHandleDelete:
    """Test handle_delete() dispatcher."""

    def setup_method(self):
        self.handler = RepositoryHandler()

    @pytest.mark.asyncio
    async def test_delete_short_path_rejected(self):
        mock_handler = _make_repo_handler()
        result = await self.handler.handle_delete("/api/v1/repository/", {}, mock_handler)
        # Short path without repo_id returns 400 or 404
        assert result.status_code in (400, 404)

    @pytest.mark.asyncio
    async def test_delete_dispatches_to_remove(self):
        mock_handler = _make_repo_handler()

        with patch.object(
            self.handler, "_remove_repository", new_callable=AsyncMock
        ) as mock_remove:
            mock_remove.return_value = MagicMock(status_code=200)
            # DELETE /api/v1/repository/repo1 → parts = ['', 'api', 'v1', 'repository', 'repo1']
            result = await self.handler.handle_delete("/api/v1/repository/repo1", {}, mock_handler)
            mock_remove.assert_called_once()
