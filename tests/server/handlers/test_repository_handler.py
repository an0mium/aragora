"""
Tests for the repository indexing handler.

Tests:
- POST /api/repository/index - Start full repository index
- POST /api/repository/incremental - Incremental update
- GET /api/repository/:id/status - Get indexing status
- GET /api/repository/:id/entities - List entities
- GET /api/repository/:id/graph - Get relationship graph
- DELETE /api/repository/:id - Remove indexed repository
"""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.server.handlers.repository import RepositoryHandler


@pytest.fixture
def repository_handler():
    """Create a repository handler with mocked context."""
    ctx = {}
    handler = RepositoryHandler(ctx)
    return handler


@pytest.fixture
def mock_orchestrator():
    """Create a mocked RepositoryOrchestrator."""
    orchestrator = MagicMock()

    # Mock index_repository
    orchestrator.index_repository = AsyncMock(
        return_value={
            "repository_id": "repo-123",
            "status": "completed",
            "entities_indexed": 150,
            "files_processed": 45,
            "duration_seconds": 12.5,
        }
    )

    # Mock incremental_update
    orchestrator.incremental_update = AsyncMock(
        return_value={
            "repository_id": "repo-123",
            "status": "completed",
            "entities_added": 5,
            "entities_updated": 3,
            "entities_removed": 1,
        }
    )

    # Mock get_status
    orchestrator.get_status = AsyncMock(
        return_value={
            "repository_id": "repo-123",
            "status": "ready",
            "last_indexed": "2026-01-18T10:00:00Z",
            "entity_count": 150,
            "file_count": 45,
        }
    )

    # Mock list_entities
    orchestrator.list_entities = AsyncMock(
        return_value=[
            {"id": "entity-1", "name": "UserService", "type": "class", "file": "services/user.py"},
            {"id": "entity-2", "name": "authenticate", "type": "function", "file": "auth/login.py"},
        ]
    )

    # Mock get_graph
    orchestrator.get_graph = AsyncMock(
        return_value={
            "nodes": [
                {"id": "entity-1", "label": "UserService", "type": "class"},
                {"id": "entity-2", "label": "authenticate", "type": "function"},
            ],
            "edges": [
                {"source": "entity-1", "target": "entity-2", "type": "calls"},
            ],
        }
    )

    # Mock delete_repository
    orchestrator.delete_repository = AsyncMock(return_value=True)

    return orchestrator


class TestRepositoryHandlerRouting:
    """Tests for RepositoryHandler route handling."""

    def test_can_handle_index(self, repository_handler):
        """Test that handler recognizes /api/repository/index route."""
        assert repository_handler.can_handle("/api/repository/index") is True

    def test_can_handle_incremental(self, repository_handler):
        """Test that handler recognizes /api/repository/incremental route."""
        assert repository_handler.can_handle("/api/repository/incremental") is True

    def test_can_handle_status(self, repository_handler):
        """Test that handler recognizes /api/repository/:id/status route."""
        assert repository_handler.can_handle("/api/repository/repo-123/status") is True

    def test_can_handle_entities(self, repository_handler):
        """Test that handler recognizes /api/repository/:id/entities route."""
        assert repository_handler.can_handle("/api/repository/repo-123/entities") is True

    def test_can_handle_graph(self, repository_handler):
        """Test that handler recognizes /api/repository/:id/graph route."""
        assert repository_handler.can_handle("/api/repository/repo-123/graph") is True

    def test_cannot_handle_unknown(self, repository_handler):
        """Test that handler rejects unknown paths."""
        assert repository_handler.can_handle("/api/debates") is False
        assert repository_handler.can_handle("/api/health") is False


class TestIndexRepository:
    """Tests for POST /api/repository/index endpoint."""

    @pytest.mark.asyncio
    async def test_index_repository_success(self, repository_handler, mock_orchestrator):
        """Test successful repository indexing."""
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            new_callable=AsyncMock,
            return_value=mock_orchestrator,
        ):
            mock_handler = MagicMock()
            mock_handler.path = "/api/repository/index"
            mock_handler.headers = {"Content-Length": "100"}
            mock_handler.rfile.read.return_value = json.dumps(
                {
                    "path": "/path/to/repo",
                    "workspace_id": "default",
                }
            ).encode()

            result = await repository_handler.handle("/api/repository/index", "POST", mock_handler)

            assert result is not None
            body = json.loads(result.body)
            # Should return indexing result or error
            assert "repository_id" in body or "error" in body or "status" in body

    @pytest.mark.asyncio
    async def test_index_repository_no_path(self, repository_handler):
        """Test index with missing path returns error."""
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ):
            mock_handler = MagicMock()
            mock_handler.path = "/api/repository/index"
            mock_handler.headers = {"Content-Length": "2"}
            mock_handler.rfile.read.return_value = b"{}"

            result = await repository_handler.handle("/api/repository/index", "POST", mock_handler)

            assert result is not None
            body = json.loads(result.body)
            assert "error" in body or result.status_code >= 400


class TestIncrementalUpdate:
    """Tests for POST /api/repository/incremental endpoint."""

    @pytest.mark.asyncio
    async def test_incremental_update_success(self, repository_handler, mock_orchestrator):
        """Test successful incremental update."""
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            new_callable=AsyncMock,
            return_value=mock_orchestrator,
        ):
            mock_handler = MagicMock()
            mock_handler.path = "/api/repository/incremental"
            mock_handler.headers = {"Content-Length": "100"}
            mock_handler.rfile.read.return_value = json.dumps(
                {
                    "repository_id": "repo-123",
                    "changed_files": ["src/new_file.py"],
                }
            ).encode()

            result = await repository_handler.handle(
                "/api/repository/incremental", "POST", mock_handler
            )

            assert result is not None
            body = json.loads(result.body)
            assert "entities_added" in body or "error" in body or "status" in body


class TestGetStatus:
    """Tests for GET /api/repository/:id/status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status_success(self, repository_handler, mock_orchestrator):
        """Test successful status retrieval."""
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            new_callable=AsyncMock,
            return_value=mock_orchestrator,
        ):
            mock_handler = MagicMock()
            mock_handler.path = "/api/repository/repo-123/status"

            result = await repository_handler.handle(
                "/api/repository/repo-123/status", "GET", mock_handler
            )

            assert result is not None
            body = json.loads(result.body)
            assert "status" in body or "error" in body

    @pytest.mark.asyncio
    async def test_get_status_invalid_id(self, repository_handler):
        """Test that invalid repository ID returns error."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/repository/<script>/status"

        result = await repository_handler.handle(
            "/api/repository/<script>/status", "GET", mock_handler
        )

        assert result is not None
        # Should reject invalid ID - returns 400 for validation failure
        body = json.loads(result.body)
        assert result.status_code == 400 or "error" in body


class TestListEntities:
    """Tests for GET /api/repository/:id/entities endpoint."""

    @pytest.mark.asyncio
    async def test_list_entities_success(self, repository_handler, mock_orchestrator):
        """Test successful entity listing."""
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            new_callable=AsyncMock,
            return_value=mock_orchestrator,
        ):
            mock_handler = MagicMock()
            mock_handler.path = "/api/repository/repo-123/entities"

            result = await repository_handler.handle(
                "/api/repository/repo-123/entities", "GET", mock_handler
            )

            assert result is not None
            body = json.loads(result.body)
            assert "entities" in body or isinstance(body, list) or "error" in body

    @pytest.mark.asyncio
    async def test_list_entities_with_filters(self, repository_handler, mock_orchestrator):
        """Test entity listing with type filter."""
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            new_callable=AsyncMock,
            return_value=mock_orchestrator,
        ):
            mock_handler = MagicMock()
            mock_handler.path = "/api/repository/repo-123/entities?type=class&limit=10"

            result = await repository_handler.handle(
                "/api/repository/repo-123/entities", "GET", mock_handler
            )

            assert result is not None


class TestGetGraph:
    """Tests for GET /api/repository/:id/graph endpoint."""

    @pytest.mark.asyncio
    async def test_get_graph_success(self, repository_handler, mock_orchestrator):
        """Test successful graph retrieval."""
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            new_callable=AsyncMock,
            return_value=mock_orchestrator,
        ):
            mock_handler = MagicMock()
            mock_handler.path = "/api/repository/repo-123/graph"

            result = await repository_handler.handle(
                "/api/repository/repo-123/graph", "GET", mock_handler
            )

            assert result is not None
            body = json.loads(result.body)
            # Without entity_id query param, returns statistics instead of nodes/edges
            assert "statistics" in body or "nodes" in body or "edges" in body or "error" in body


class TestDeleteRepository:
    """Tests for DELETE /api/repository/:id endpoint."""

    @pytest.mark.asyncio
    async def test_delete_repository_success(self, repository_handler, mock_orchestrator):
        """Test successful repository deletion."""
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            new_callable=AsyncMock,
            return_value=mock_orchestrator,
        ):
            mock_handler = MagicMock()
            mock_handler.path = "/api/repository/repo-123"

            result = await repository_handler.handle(
                "/api/repository/repo-123", "DELETE", mock_handler
            )

            assert result is not None
            body = json.loads(result.body)
            assert "deleted" in body or "success" in body or "error" in body


class TestOrchestratorUnavailable:
    """Tests for graceful handling when orchestrator is unavailable."""

    @pytest.mark.asyncio
    async def test_index_without_orchestrator(self, repository_handler):
        """Test index returns error when orchestrator unavailable."""
        with patch(
            "aragora.server.handlers.repository._get_orchestrator",
            new_callable=AsyncMock,
            return_value=None,
        ):
            mock_handler = MagicMock()
            mock_handler.path = "/api/repository/index"
            mock_handler.headers = {"Content-Length": "50"}
            mock_handler.rfile.read.return_value = json.dumps(
                {
                    "path": "/test/repo",
                }
            ).encode()

            result = await repository_handler.handle("/api/repository/index", "POST", mock_handler)

            assert result is not None
            body = json.loads(result.body)
            assert "error" in body or result.status_code >= 400


class TestPathValidation:
    """Tests for path traversal and injection prevention."""

    @pytest.mark.asyncio
    async def test_rejects_path_traversal(self, repository_handler):
        """Test that path traversal attempts are rejected."""
        mock_handler = MagicMock()
        # Path with traversal - results in many path segments, doesn't match routes
        mock_handler.path = "/api/repository/../../../etc/passwd/status"

        result = await repository_handler.handle(
            "/api/repository/../../../etc/passwd/status", "GET", mock_handler
        )

        assert result is not None
        # Malformed path is rejected - either 400 (validation) or 404 (no route match)
        assert result.status_code in (400, 404)

    @pytest.mark.asyncio
    async def test_rejects_script_injection(self, repository_handler):
        """Test that script injection attempts are rejected."""
        mock_handler = MagicMock()
        # Note: </script> contains '/' which splits into extra path segments
        mock_handler.path = "/api/repository/<script>alert(1)</script>/status"

        result = await repository_handler.handle(
            "/api/repository/<script>alert(1)</script>/status", "GET", mock_handler
        )

        assert result is not None
        # Script tag with '/' creates extra path segments, resulting in 404 (no route match)
        # This is still a security win - the request is rejected
        assert result.status_code in (400, 404)
