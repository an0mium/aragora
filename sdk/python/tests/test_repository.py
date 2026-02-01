"""Tests for Repository SDK namespace."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock client."""
    return MagicMock()


class TestRepositoryAPI:
    """Test synchronous RepositoryAPI."""

    def test_init(self, mock_client: MagicMock) -> None:
        """Test API initialization."""
        from aragora.namespaces.repository import RepositoryAPI

        api = RepositoryAPI(mock_client)
        assert api._client is mock_client

    def test_index(self, mock_client: MagicMock) -> None:
        """Test index calls correct endpoint."""
        from aragora.namespaces.repository import RepositoryAPI

        mock_client.request.return_value = {
            "index_id": "idx_123",
            "status": "pending",
            "started_at": "2024-01-01T00:00:00Z",
        }

        api = RepositoryAPI(mock_client)
        result = api.index(repository_url="https://github.com/org/repo")

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/repository/index")
        assert call_args[1]["json"]["repository_url"] == "https://github.com/org/repo"
        assert result["index_id"] == "idx_123"

    def test_index_with_all_options(self, mock_client: MagicMock) -> None:
        """Test index with all options."""
        from aragora.namespaces.repository import RepositoryAPI

        mock_client.request.return_value = {"index_id": "idx_456"}

        api = RepositoryAPI(mock_client)
        api.index(
            repository_url="https://github.com/org/repo",
            local_path="/path/to/repo",
            branch="develop",
            include_patterns=["**/*.py", "**/*.ts"],
            exclude_patterns=["**/node_modules/**", "**/venv/**"],
            max_file_size_kb=500,
            extract_entities=True,
            build_graph=True,
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["repository_url"] == "https://github.com/org/repo"
        assert json_body["local_path"] == "/path/to/repo"
        assert json_body["branch"] == "develop"
        assert json_body["include_patterns"] == ["**/*.py", "**/*.ts"]
        assert json_body["exclude_patterns"] == ["**/node_modules/**", "**/venv/**"]
        assert json_body["max_file_size_kb"] == 500
        assert json_body["extract_entities"] is True
        assert json_body["build_graph"] is True

    def test_incremental_index(self, mock_client: MagicMock) -> None:
        """Test incremental_index calls correct endpoint."""
        from aragora.namespaces.repository import RepositoryAPI

        mock_client.request.return_value = {
            "index_id": "idx_123",
            "status": "pending",
        }

        api = RepositoryAPI(mock_client)
        api.incremental_index(
            index_id="idx_123",
            changed_files=["src/main.py", "src/utils.py"],
        )

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/repository/incremental")
        assert call_args[1]["json"]["index_id"] == "idx_123"
        assert call_args[1]["json"]["changed_files"] == ["src/main.py", "src/utils.py"]

    def test_incremental_index_with_since_commit(self, mock_client: MagicMock) -> None:
        """Test incremental_index with since_commit parameter."""
        from aragora.namespaces.repository import RepositoryAPI

        mock_client.request.return_value = {"index_id": "idx_123"}

        api = RepositoryAPI(mock_client)
        api.incremental_index(
            index_id="idx_123",
            since_commit="abc123def456",
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["since_commit"] == "abc123def456"

    def test_batch_index(self, mock_client: MagicMock) -> None:
        """Test batch_index calls correct endpoint."""
        from aragora.namespaces.repository import RepositoryAPI

        mock_client.request.return_value = {
            "batch_id": "batch_789",
            "index_ids": ["idx_1", "idx_2"],
            "status": "pending",
        }

        api = RepositoryAPI(mock_client)
        result = api.batch_index(
            repositories=[
                {"repository_url": "https://github.com/org/repo1"},
                {"repository_url": "https://github.com/org/repo2"},
            ]
        )

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/repository/batch")
        assert len(call_args[1]["json"]["repositories"]) == 2
        assert result["batch_id"] == "batch_789"

    def test_batch_index_with_options(self, mock_client: MagicMock) -> None:
        """Test batch_index with parallel and max_concurrent options."""
        from aragora.namespaces.repository import RepositoryAPI

        mock_client.request.return_value = {"batch_id": "batch_abc"}

        api = RepositoryAPI(mock_client)
        api.batch_index(
            repositories=[{"repository_url": "https://github.com/org/repo"}],
            parallel=False,
            max_concurrent=2,
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["parallel"] is False
        assert json_body["max_concurrent"] == 2

    def test_get_status(self, mock_client: MagicMock) -> None:
        """Test get_status calls correct endpoint."""
        from aragora.namespaces.repository import RepositoryAPI

        mock_client.request.return_value = {
            "index_id": "idx_123",
            "status": "completed",
            "progress_percent": 100,
            "files_indexed": 150,
            "files_total": 150,
            "entities_extracted": 500,
            "relationships_found": 200,
        }

        api = RepositoryAPI(mock_client)
        result = api.get_status("idx_123")

        mock_client.request.assert_called_once_with("GET", "/api/v1/repository/idx_123/status")
        assert result["status"] == "completed"
        assert result["progress_percent"] == 100

    def test_list_entities(self, mock_client: MagicMock) -> None:
        """Test list_entities calls correct endpoint."""
        from aragora.namespaces.repository import RepositoryAPI

        mock_client.request.return_value = {
            "entities": [
                {"id": "ent_1", "type": "class", "name": "UserService"},
                {"id": "ent_2", "type": "function", "name": "process_data"},
            ],
            "total": 2,
        }

        api = RepositoryAPI(mock_client)
        result = api.list_entities("idx_123")

        mock_client.request.assert_called_once_with(
            "GET", "/api/v1/repository/idx_123/entities", params={}
        )
        assert len(result["entities"]) == 2

    def test_list_entities_with_filters(self, mock_client: MagicMock) -> None:
        """Test list_entities with all filter options."""
        from aragora.namespaces.repository import RepositoryAPI

        mock_client.request.return_value = {"entities": [], "total": 0}

        api = RepositoryAPI(mock_client)
        api.list_entities(
            index_id="idx_123",
            type="class",
            language="python",
            file_pattern="src/**/*.py",
            name_pattern="*Service",
            visibility="public",
            limit=50,
            offset=10,
        )

        mock_client.request.assert_called_once_with(
            "GET",
            "/api/v1/repository/idx_123/entities",
            params={
                "type": "class",
                "language": "python",
                "file_pattern": "src/**/*.py",
                "name_pattern": "*Service",
                "visibility": "public",
                "limit": 50,
                "offset": 10,
            },
        )

    def test_get_graph(self, mock_client: MagicMock) -> None:
        """Test get_graph calls correct endpoint."""
        from aragora.namespaces.repository import RepositoryAPI

        mock_client.request.return_value = {
            "nodes": [
                {"id": "ent_1", "type": "class", "name": "A"},
                {"id": "ent_2", "type": "class", "name": "B"},
            ],
            "edges": [{"source": "ent_1", "target": "ent_2", "type": "imports"}],
            "statistics": {"total_nodes": 2, "total_edges": 1},
        }

        api = RepositoryAPI(mock_client)
        result = api.get_graph("idx_123")

        mock_client.request.assert_called_once_with("GET", "/api/v1/repository/idx_123/graph")
        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1

    def test_get_entity(self, mock_client: MagicMock) -> None:
        """Test get_entity calls correct endpoint."""
        from aragora.namespaces.repository import RepositoryAPI

        mock_client.request.return_value = {
            "id": "ent_123",
            "type": "class",
            "name": "UserRepository",
            "qualified_name": "app.repositories.UserRepository",
            "file_path": "app/repositories.py",
            "line_start": 10,
            "line_end": 50,
            "language": "python",
            "docstring": "Repository for user data access.",
            "visibility": "public",
        }

        api = RepositoryAPI(mock_client)
        result = api.get_entity("idx_123", "ent_123")

        mock_client.request.assert_called_once_with(
            "GET", "/api/v1/repository/idx_123/entities/ent_123"
        )
        assert result["name"] == "UserRepository"
        assert result["type"] == "class"


@pytest.fixture
def mock_async_client() -> MagicMock:
    """Create a mock async client."""
    from unittest.mock import AsyncMock

    client = MagicMock()
    client.request = AsyncMock()
    return client


class TestAsyncRepositoryAPI:
    """Test asynchronous AsyncRepositoryAPI."""

    @pytest.mark.asyncio
    async def test_init(self, mock_async_client: MagicMock) -> None:
        """Test API initialization."""
        from aragora.namespaces.repository import AsyncRepositoryAPI

        api = AsyncRepositoryAPI(mock_async_client)
        assert api._client is mock_async_client

    @pytest.mark.asyncio
    async def test_index(self, mock_async_client: MagicMock) -> None:
        """Test index calls correct endpoint."""
        from aragora.namespaces.repository import AsyncRepositoryAPI

        mock_async_client.request.return_value = {
            "index_id": "idx_async_123",
            "status": "pending",
        }

        api = AsyncRepositoryAPI(mock_async_client)
        result = await api.index(
            repository_url="https://github.com/org/async-repo",
            branch="main",
            extract_entities=True,
        )

        mock_async_client.request.assert_called_once()
        call_args = mock_async_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/repository/index")
        json_body = call_args[1]["json"]
        assert json_body["repository_url"] == "https://github.com/org/async-repo"
        assert json_body["branch"] == "main"
        assert json_body["extract_entities"] is True
        assert result["index_id"] == "idx_async_123"

    @pytest.mark.asyncio
    async def test_incremental_index(self, mock_async_client: MagicMock) -> None:
        """Test incremental_index calls correct endpoint."""
        from aragora.namespaces.repository import AsyncRepositoryAPI

        mock_async_client.request.return_value = {"index_id": "idx_456"}

        api = AsyncRepositoryAPI(mock_async_client)
        await api.incremental_index(
            index_id="idx_456",
            changed_files=["file1.py"],
            since_commit="def789",
        )

        mock_async_client.request.assert_called_once()
        call_args = mock_async_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/repository/incremental")
        json_body = call_args[1]["json"]
        assert json_body["index_id"] == "idx_456"
        assert json_body["since_commit"] == "def789"

    @pytest.mark.asyncio
    async def test_batch_index(self, mock_async_client: MagicMock) -> None:
        """Test batch_index calls correct endpoint."""
        from aragora.namespaces.repository import AsyncRepositoryAPI

        mock_async_client.request.return_value = {
            "batch_id": "batch_async",
            "index_ids": ["idx_a", "idx_b"],
        }

        api = AsyncRepositoryAPI(mock_async_client)
        result = await api.batch_index(
            repositories=[
                {"repository_url": "https://github.com/org/repo-a"},
                {"local_path": "/local/repo-b"},
            ],
            parallel=True,
            max_concurrent=4,
        )

        mock_async_client.request.assert_called_once()
        call_args = mock_async_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/repository/batch")
        json_body = call_args[1]["json"]
        assert json_body["parallel"] is True
        assert json_body["max_concurrent"] == 4
        assert result["batch_id"] == "batch_async"

    @pytest.mark.asyncio
    async def test_get_status(self, mock_async_client: MagicMock) -> None:
        """Test get_status calls correct endpoint."""
        from aragora.namespaces.repository import AsyncRepositoryAPI

        mock_async_client.request.return_value = {
            "index_id": "idx_789",
            "status": "in_progress",
            "progress_percent": 50,
        }

        api = AsyncRepositoryAPI(mock_async_client)
        result = await api.get_status("idx_789")

        mock_async_client.request.assert_called_once_with(
            "GET", "/api/v1/repository/idx_789/status"
        )
        assert result["status"] == "in_progress"
        assert result["progress_percent"] == 50

    @pytest.mark.asyncio
    async def test_list_entities(self, mock_async_client: MagicMock) -> None:
        """Test list_entities calls correct endpoint."""
        from aragora.namespaces.repository import AsyncRepositoryAPI

        mock_async_client.request.return_value = {
            "entities": [{"id": "e1", "type": "function"}],
            "total": 1,
        }

        api = AsyncRepositoryAPI(mock_async_client)
        result = await api.list_entities(
            index_id="idx_ent",
            type="function",
            language="typescript",
            limit=100,
        )

        mock_async_client.request.assert_called_once_with(
            "GET",
            "/api/v1/repository/idx_ent/entities",
            params={"type": "function", "language": "typescript", "limit": 100},
        )
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_get_graph(self, mock_async_client: MagicMock) -> None:
        """Test get_graph calls correct endpoint."""
        from aragora.namespaces.repository import AsyncRepositoryAPI

        mock_async_client.request.return_value = {
            "nodes": [],
            "edges": [],
            "statistics": {},
        }

        api = AsyncRepositoryAPI(mock_async_client)
        result = await api.get_graph("idx_graph")

        mock_async_client.request.assert_called_once_with(
            "GET", "/api/v1/repository/idx_graph/graph"
        )
        assert "nodes" in result
        assert "edges" in result

    @pytest.mark.asyncio
    async def test_get_entity(self, mock_async_client: MagicMock) -> None:
        """Test get_entity calls correct endpoint."""
        from aragora.namespaces.repository import AsyncRepositoryAPI

        mock_async_client.request.return_value = {
            "id": "ent_abc",
            "type": "method",
            "name": "process",
            "language": "go",
        }

        api = AsyncRepositoryAPI(mock_async_client)
        result = await api.get_entity("idx_go", "ent_abc")

        mock_async_client.request.assert_called_once_with(
            "GET", "/api/v1/repository/idx_go/entities/ent_abc"
        )
        assert result["name"] == "process"
        assert result["language"] == "go"
