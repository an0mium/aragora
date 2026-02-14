"""Tests for Repository namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestRepositoryIndexing:
    """Tests for repository indexing operations."""

    def test_index_with_url(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"index_id": "idx_001", "status": "started"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.repository.index(
                repository_url="https://github.com/org/repo",
                branch="main",
                extract_entities=True,
                build_graph=True,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/repository/index",
                json={
                    "repository_url": "https://github.com/org/repo",
                    "branch": "main",
                    "extract_entities": True,
                    "build_graph": True,
                },
            )
            assert result["index_id"] == "idx_001"
            assert result["status"] == "started"
            client.close()

    def test_index_with_local_path_and_patterns(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"index_id": "idx_002", "status": "started"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.repository.index(
                local_path="/home/user/project",
                include_patterns=["*.py", "*.ts"],
                exclude_patterns=["node_modules/**"],
                max_file_size_kb=500,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/repository/index",
                json={
                    "local_path": "/home/user/project",
                    "include_patterns": ["*.py", "*.ts"],
                    "exclude_patterns": ["node_modules/**"],
                    "max_file_size_kb": 500,
                },
            )
            client.close()

    def test_incremental_index(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"index_id": "idx_001", "status": "started"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.repository.incremental_index(
                index_id="idx_001",
                changed_files=["src/main.py", "src/utils.py"],
                since_commit="abc123",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/repository/incremental",
                json={
                    "index_id": "idx_001",
                    "changed_files": ["src/main.py", "src/utils.py"],
                    "since_commit": "abc123",
                },
            )
            assert result["index_id"] == "idx_001"
            client.close()

    def test_batch_index(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            repos = [
                {"repository_url": "https://github.com/org/repo1", "branch": "main"},
                {"repository_url": "https://github.com/org/repo2", "branch": "develop"},
            ]
            mock_request.return_value = {
                "batch_id": "batch_001",
                "index_ids": ["idx_001", "idx_002"],
                "status": "started",
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.repository.batch_index(
                repositories=repos,
                parallel=True,
                max_concurrent=4,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/repository/batch",
                json={
                    "repositories": repos,
                    "parallel": True,
                    "max_concurrent": 4,
                },
            )
            assert result["batch_id"] == "batch_001"
            assert len(result["index_ids"]) == 2
            client.close()

class TestRepositoryStatusAndQueries:
    """Tests for status checking and entity querying."""

    def test_get_status(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "index_id": "idx_001",
                "status": "completed",
                "progress_percent": 100,
                "files_indexed": 250,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.repository.get_status("idx_001")
            mock_request.assert_called_once_with("GET", "/api/v1/repository/idx_001/status")
            assert result["status"] == "completed"
            assert result["progress_percent"] == 100
            client.close()

    def test_list_entities_with_filters(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "entities": [{"id": "e1", "type": "class", "name": "Arena"}],
                "total": 1,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.repository.list_entities(
                index_id="idx_001",
                type="class",
                language="python",
                visibility="public",
                limit=10,
                offset=0,
            )
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/repository/idx_001/entities",
                params={
                    "type": "class",
                    "language": "python",
                    "visibility": "public",
                    "limit": 10,
                    "offset": 0,
                },
            )
            assert result["total"] == 1
            assert result["entities"][0]["name"] == "Arena"
            client.close()

    def test_get_graph(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "nodes": [{"id": "n1"}],
                "edges": [{"source": "n1", "target": "n2"}],
                "statistics": {"total_nodes": 1, "total_edges": 1},
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.repository.get_graph("idx_001")
            mock_request.assert_called_once_with("GET", "/api/v1/repository/idx_001/graph")
            assert len(result["nodes"]) == 1
            assert len(result["edges"]) == 1
            client.close()

class TestAsyncRepository:
    """Tests for async repository methods."""

    @pytest.mark.asyncio
    async def test_async_index(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"index_id": "idx_001", "status": "started"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.repository.index(
                repository_url="https://github.com/org/repo",
                branch="main",
                extract_entities=True,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/repository/index",
                json={
                    "repository_url": "https://github.com/org/repo",
                    "branch": "main",
                    "extract_entities": True,
                },
            )
            assert result["index_id"] == "idx_001"
            await client.close()

    @pytest.mark.asyncio
    async def test_async_get_status(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "index_id": "idx_001",
                "status": "in_progress",
                "progress_percent": 45,
            }
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.repository.get_status("idx_001")
            mock_request.assert_called_once_with("GET", "/api/v1/repository/idx_001/status")
            assert result["progress_percent"] == 45
            await client.close()

    @pytest.mark.asyncio
    async def test_async_list_entities(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"entities": [], "total": 0}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.repository.list_entities(
                index_id="idx_001",
                type="function",
                language="python",
                limit=20,
            )
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/repository/idx_001/entities",
                params={"type": "function", "language": "python", "limit": 20},
            )
            assert result["total"] == 0
            await client.close()

    @pytest.mark.asyncio
    async def test_async_batch_index(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            repos = [{"repository_url": "https://github.com/org/repo", "branch": "main"}]
            mock_request.return_value = {
                "batch_id": "batch_002",
                "index_ids": ["idx_010"],
                "status": "started",
            }
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.repository.batch_index(repositories=repos, parallel=False)
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/repository/batch",
                json={"repositories": repos, "parallel": False},
            )
            assert result["batch_id"] == "batch_002"
            await client.close()
