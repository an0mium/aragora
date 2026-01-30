"""
Comprehensive tests for Repository Orchestrator.

Tests cover:
- RepoConfig, IndexResult, BatchResult, OrchestratorConfig, IndexProgress dataclasses
- RepositoryOrchestrator class methods
- Repository indexing operations
- Batch processing
- Progress tracking
- Error handling and edge cases
"""

import asyncio
import pytest
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from aragora.knowledge.repository_orchestrator import (
    RepoConfig,
    IndexResult,
    BatchResult,
    OrchestratorConfig,
    IndexProgress,
    RepositoryOrchestrator,
)
from aragora.connectors.repository_crawler import (
    CrawlConfig,
    CrawlResult,
    CrawledFile,
    FileSymbol,
    FileDependency,
    FileType,
)


# =============================================================================
# RepoConfig Tests
# =============================================================================


class TestRepoConfig:
    """Tests for RepoConfig dataclass."""

    def test_init_minimal(self):
        """Should initialize with required fields only."""
        config = RepoConfig(path="/path/to/repo", workspace_id="ws_test")
        assert config.path == "/path/to/repo"
        assert config.workspace_id == "ws_test"
        assert config.name is None
        assert config.crawl_config is None
        assert config.priority == 0
        assert config.metadata == {}

    def test_init_with_all_fields(self):
        """Should initialize with all fields provided."""
        crawl_config = CrawlConfig(max_files=100)
        config = RepoConfig(
            path="/path/to/repo",
            workspace_id="ws_test",
            name="custom-name",
            crawl_config=crawl_config,
            priority=10,
            metadata={"key": "value"},
        )
        assert config.name == "custom-name"
        assert config.crawl_config is crawl_config
        assert config.priority == 10
        assert config.metadata == {"key": "value"}

    def test_default_metadata_is_independent(self):
        """Should have independent default metadata per instance."""
        config1 = RepoConfig(path="/path1", workspace_id="ws1")
        config2 = RepoConfig(path="/path2", workspace_id="ws2")
        config1.metadata["key"] = "value"
        assert "key" not in config2.metadata


# =============================================================================
# IndexResult Tests
# =============================================================================


class TestIndexResult:
    """Tests for IndexResult dataclass."""

    def test_init_with_defaults(self):
        """Should initialize with default values."""
        result = IndexResult(
            repository_name="test-repo",
            repository_path="/path/to/repo",
            workspace_id="ws_test",
            files_indexed=10,
            nodes_created=20,
            relationships_created=5,
            symbols_indexed=15,
            errors=[],
            warnings=[],
            duration_ms=1000.5,
        )
        assert result.repository_name == "test-repo"
        assert result.files_indexed == 10
        assert result.git_info is None
        assert result.started_at is not None
        assert result.completed_at is None

    def test_init_with_all_fields(self):
        """Should initialize with all fields provided."""
        started = datetime.now(timezone.utc)
        completed = datetime.now(timezone.utc)
        result = IndexResult(
            repository_name="test-repo",
            repository_path="/path/to/repo",
            workspace_id="ws_test",
            files_indexed=10,
            nodes_created=20,
            relationships_created=5,
            symbols_indexed=15,
            errors=["error1"],
            warnings=["warning1"],
            duration_ms=1000.5,
            git_info={"branch": "main"},
            started_at=started,
            completed_at=completed,
        )
        assert result.errors == ["error1"]
        assert result.warnings == ["warning1"]
        assert result.git_info == {"branch": "main"}
        assert result.started_at == started
        assert result.completed_at == completed

    def test_to_dict_without_completed(self):
        """Should convert to dict without completed_at."""
        result = IndexResult(
            repository_name="test-repo",
            repository_path="/path/to/repo",
            workspace_id="ws_test",
            files_indexed=10,
            nodes_created=20,
            relationships_created=5,
            symbols_indexed=15,
            errors=["e1", "e2"],
            warnings=["w1"],
            duration_ms=1500.0,
        )
        d = result.to_dict()
        assert d["repository_name"] == "test-repo"
        assert d["repository_path"] == "/path/to/repo"
        assert d["workspace_id"] == "ws_test"
        assert d["files_indexed"] == 10
        assert d["nodes_created"] == 20
        assert d["relationships_created"] == 5
        assert d["symbols_indexed"] == 15
        assert d["error_count"] == 2
        assert d["warning_count"] == 1
        assert d["duration_ms"] == 1500.0
        assert d["completed_at"] is None

    def test_to_dict_with_completed(self):
        """Should convert to dict with completed_at."""
        completed = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = IndexResult(
            repository_name="test-repo",
            repository_path="/path/to/repo",
            workspace_id="ws_test",
            files_indexed=5,
            nodes_created=10,
            relationships_created=3,
            symbols_indexed=7,
            errors=[],
            warnings=[],
            duration_ms=500.0,
            completed_at=completed,
        )
        d = result.to_dict()
        assert d["completed_at"] == "2025-01-15T12:00:00+00:00"


# =============================================================================
# BatchResult Tests
# =============================================================================


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_init(self):
        """Should initialize with all fields."""
        repo_results = [
            IndexResult(
                repository_name="repo1",
                repository_path="/path1",
                workspace_id="ws1",
                files_indexed=10,
                nodes_created=20,
                relationships_created=5,
                symbols_indexed=15,
                errors=[],
                warnings=[],
                duration_ms=1000.0,
            ),
            IndexResult(
                repository_name="repo2",
                repository_path="/path2",
                workspace_id="ws2",
                files_indexed=5,
                nodes_created=10,
                relationships_created=2,
                symbols_indexed=8,
                errors=["error"],
                warnings=[],
                duration_ms=500.0,
            ),
        ]
        batch = BatchResult(
            repositories=repo_results,
            total_files=15,
            total_nodes=30,
            total_relationships=7,
            total_duration_ms=1500.0,
            successful=1,
            failed=1,
        )
        assert len(batch.repositories) == 2
        assert batch.total_files == 15
        assert batch.successful == 1
        assert batch.failed == 1

    def test_to_dict(self):
        """Should convert to dict including all repos."""
        repo_result = IndexResult(
            repository_name="repo1",
            repository_path="/path1",
            workspace_id="ws1",
            files_indexed=10,
            nodes_created=20,
            relationships_created=5,
            symbols_indexed=15,
            errors=[],
            warnings=[],
            duration_ms=1000.0,
        )
        batch = BatchResult(
            repositories=[repo_result],
            total_files=10,
            total_nodes=20,
            total_relationships=5,
            total_duration_ms=1000.0,
            successful=1,
            failed=0,
        )
        d = batch.to_dict()
        assert len(d["repositories"]) == 1
        assert d["total_files"] == 10
        assert d["total_nodes"] == 20
        assert d["total_relationships"] == 5
        assert d["total_duration_ms"] == 1000.0
        assert d["successful"] == 1
        assert d["failed"] == 0


# =============================================================================
# OrchestratorConfig Tests
# =============================================================================


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig dataclass."""

    def test_defaults(self):
        """Should have correct default values."""
        config = OrchestratorConfig()
        assert config.max_concurrent_repos == 3
        assert config.max_concurrent_files_per_repo == 20
        assert config.store_file_content is True
        assert config.store_symbols is True
        assert config.store_dependencies is True
        assert config.store_chunks is True
        assert config.checkpoint_enabled is True
        assert config.checkpoint_interval_files == 100

    def test_custom_values(self):
        """Should accept custom values."""
        config = OrchestratorConfig(
            max_concurrent_repos=5,
            max_concurrent_files_per_repo=50,
            store_file_content=False,
            store_symbols=False,
            store_dependencies=False,
            store_chunks=False,
            checkpoint_enabled=False,
            checkpoint_interval_files=50,
        )
        assert config.max_concurrent_repos == 5
        assert config.max_concurrent_files_per_repo == 50
        assert config.store_file_content is False
        assert config.store_symbols is False
        assert config.store_dependencies is False
        assert config.store_chunks is False
        assert config.checkpoint_enabled is False
        assert config.checkpoint_interval_files == 50


# =============================================================================
# IndexProgress Tests
# =============================================================================


class TestIndexProgress:
    """Tests for IndexProgress dataclass."""

    def test_init_minimal(self):
        """Should initialize with required fields."""
        progress = IndexProgress(
            repository_path="/path/to/repo",
            workspace_id="ws_test",
            status="pending",
        )
        assert progress.repository_path == "/path/to/repo"
        assert progress.workspace_id == "ws_test"
        assert progress.status == "pending"
        assert progress.files_discovered == 0
        assert progress.files_processed == 0
        assert progress.nodes_created == 0
        assert progress.current_file is None
        assert progress.started_at is None
        assert progress.last_updated is None
        assert progress.error is None

    def test_init_with_all_fields(self):
        """Should initialize with all fields."""
        started = datetime.now(timezone.utc)
        updated = datetime.now(timezone.utc)
        progress = IndexProgress(
            repository_path="/path/to/repo",
            workspace_id="ws_test",
            status="indexing",
            files_discovered=100,
            files_processed=50,
            nodes_created=75,
            current_file="src/main.py",
            started_at=started,
            last_updated=updated,
            error=None,
        )
        assert progress.files_discovered == 100
        assert progress.files_processed == 50
        assert progress.nodes_created == 75
        assert progress.current_file == "src/main.py"
        assert progress.started_at == started
        assert progress.last_updated == updated

    def test_status_values(self):
        """Should accept valid status values."""
        for status in ["pending", "crawling", "indexing", "completed", "failed"]:
            progress = IndexProgress(
                repository_path="/path",
                workspace_id="ws",
                status=status,
            )
            assert progress.status == status


# =============================================================================
# RepositoryOrchestrator Tests
# =============================================================================


class TestRepositoryOrchestratorInit:
    """Tests for RepositoryOrchestrator initialization."""

    def test_init_with_mound_only(self):
        """Should initialize with just a mound."""
        mock_mound = MagicMock()
        orchestrator = RepositoryOrchestrator(mound=mock_mound)
        assert orchestrator.mound is mock_mound
        assert orchestrator.config is not None
        assert isinstance(orchestrator.config, OrchestratorConfig)

    def test_init_with_custom_config(self):
        """Should initialize with custom config."""
        mock_mound = MagicMock()
        config = OrchestratorConfig(max_concurrent_repos=10)
        orchestrator = RepositoryOrchestrator(mound=mock_mound, config=config)
        assert orchestrator.config is config
        assert orchestrator.config.max_concurrent_repos == 10

    def test_init_creates_crawler(self):
        """Should create a RepositoryCrawler instance."""
        mock_mound = MagicMock()
        orchestrator = RepositoryOrchestrator(mound=mock_mound)
        assert orchestrator._crawler is not None

    def test_init_empty_progress_tracking(self):
        """Should initialize with empty progress tracking."""
        mock_mound = MagicMock()
        orchestrator = RepositoryOrchestrator(mound=mock_mound)
        assert orchestrator._progress == {}
        assert orchestrator._active_tasks == {}


class TestRepositoryOrchestratorIndexRepository:
    """Tests for index_repository method."""

    @pytest.fixture
    def mock_mound(self):
        """Create a mock KnowledgeMound."""
        mound = MagicMock()
        mound.add = AsyncMock(return_value=MagicMock(node_id="kn_123"))
        mound.query = AsyncMock(return_value=MagicMock(items=[], total_count=0))
        mound.delete = AsyncMock()
        return mound

    @pytest.fixture
    def mock_crawl_result(self):
        """Create a mock CrawlResult."""
        crawled_file = CrawledFile(
            path="/repo/src/main.py",
            relative_path="src/main.py",
            file_type=FileType.PYTHON,
            content="def main(): pass",
            content_hash="abc123",
            size_bytes=100,
            line_count=1,
            symbols=[
                FileSymbol(
                    name="main",
                    kind="function",
                    line_start=1,
                    line_end=1,
                    signature="def main()",
                    docstring="Main function",
                )
            ],
            dependencies=[
                FileDependency(
                    source="src/main.py",
                    target="os",
                    kind="import",
                    line=1,
                )
            ],
            chunks=[
                {
                    "index": 0,
                    "content": "def main(): pass",
                    "start_line": 1,
                    "end_line": 1,
                }
            ],
        )
        return CrawlResult(
            repository_path="/repo",
            repository_name="test-repo",
            files=[crawled_file],
            total_files=1,
            total_lines=1,
            total_bytes=100,
            file_type_counts={"python": 1},
            symbol_counts={"function": 1},
            dependency_graph={"src/main.py": ["os"]},
            crawl_duration_ms=100.0,
            errors=[],
            warnings=[],
            git_info={"branch": "main"},
        )

    @pytest.mark.asyncio
    async def test_index_repository_success(self, mock_mound, mock_crawl_result):
        """Should successfully index a repository."""
        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        with patch.object(orchestrator._crawler, "crawl", new_callable=AsyncMock) as mock_crawl:
            mock_crawl.return_value = mock_crawl_result

            result = await orchestrator.index_repository(
                repo_path="/repo",
                workspace_id="ws_test",
            )

            assert result.repository_name == "test-repo"
            assert result.files_indexed == 1
            assert result.nodes_created > 0
            assert len(result.errors) == 0
            assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_index_repository_tracks_progress(self, mock_mound, mock_crawl_result):
        """Should track progress during indexing."""
        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        with patch.object(orchestrator._crawler, "crawl", new_callable=AsyncMock) as mock_crawl:
            mock_crawl.return_value = mock_crawl_result

            await orchestrator.index_repository(
                repo_path="/repo",
                workspace_id="ws_test",
            )

            progress = orchestrator.get_progress("/repo")
            assert progress is not None
            assert progress.status == "completed"
            assert progress.files_discovered == 1

    @pytest.mark.asyncio
    async def test_index_repository_with_custom_config(self, mock_mound, mock_crawl_result):
        """Should use custom crawl config."""
        orchestrator = RepositoryOrchestrator(mound=mock_mound)
        crawl_config = CrawlConfig(max_files=50)

        with patch.object(orchestrator._crawler, "crawl", new_callable=AsyncMock) as mock_crawl:
            mock_crawl.return_value = mock_crawl_result

            await orchestrator.index_repository(
                repo_path="/repo",
                workspace_id="ws_test",
                crawl_config=crawl_config,
            )

            assert orchestrator._crawler._config is crawl_config

    @pytest.mark.asyncio
    async def test_index_repository_handles_crawl_error(self, mock_mound):
        """Should handle crawl errors gracefully."""
        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        with patch.object(orchestrator._crawler, "crawl", new_callable=AsyncMock) as mock_crawl:
            mock_crawl.side_effect = RuntimeError("Crawl failed")

            result = await orchestrator.index_repository(
                repo_path="/repo",
                workspace_id="ws_test",
            )

            assert result.files_indexed == 0
            assert len(result.errors) == 1
            assert "Crawl failed" in result.errors[0]

            progress = orchestrator.get_progress("/repo")
            assert progress.status == "failed"
            assert progress.error is not None

    @pytest.mark.asyncio
    async def test_index_repository_handles_file_indexing_error(
        self, mock_mound, mock_crawl_result
    ):
        """Should continue indexing when individual mound.add calls fail."""
        orchestrator = RepositoryOrchestrator(mound=mock_mound)
        mock_mound.add = AsyncMock(side_effect=RuntimeError("Add failed"))

        with patch.object(orchestrator._crawler, "crawl", new_callable=AsyncMock) as mock_crawl:
            mock_crawl.return_value = mock_crawl_result

            result = await orchestrator.index_repository(
                repo_path="/repo",
                workspace_id="ws_test",
            )

            # Should still complete successfully (errors are caught internally)
            assert result.completed_at is not None
            # Individual add failures don't bubble up as errors
            # (they're logged at debug level) - the process continues
            assert result.files_indexed == 1  # File was processed
            assert result.nodes_created == 0  # But no nodes were created due to add failures

    @pytest.mark.asyncio
    async def test_index_repository_handles_index_file_exception(
        self, mock_mound, mock_crawl_result
    ):
        """Should capture errors when _index_file raises exception."""
        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        with patch.object(orchestrator._crawler, "crawl", new_callable=AsyncMock) as mock_crawl:
            mock_crawl.return_value = mock_crawl_result

            # Patch _index_file to raise an exception
            with patch.object(orchestrator, "_index_file", new_callable=AsyncMock) as mock_index:
                mock_index.side_effect = RuntimeError("Index file failed")

                result = await orchestrator.index_repository(
                    repo_path="/repo",
                    workspace_id="ws_test",
                )

                # Should still complete, but with errors
                assert result.completed_at is not None
                assert len(result.errors) > 0
                assert "Index file failed" in result.errors[0]


class TestRepositoryOrchestratorIncrementalUpdate:
    """Tests for incremental_update method."""

    @pytest.fixture
    def mock_mound(self):
        """Create a mock KnowledgeMound."""
        mound = MagicMock()
        mound.add = AsyncMock(return_value=MagicMock(node_id="kn_123"))
        return mound

    @pytest.mark.asyncio
    async def test_incremental_update_calls_index_repository(self, mock_mound):
        """Should call index_repository with incremental=True."""
        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        with patch.object(orchestrator, "index_repository", new_callable=AsyncMock) as mock_index:
            mock_index.return_value = MagicMock()

            await orchestrator.incremental_update(
                repo_path="/repo",
                workspace_id="ws_test",
            )

            mock_index.assert_called_once_with("/repo", "ws_test", incremental=True)


class TestRepositoryOrchestratorIndexMultiple:
    """Tests for index_multiple method."""

    @pytest.fixture
    def mock_mound(self):
        """Create a mock KnowledgeMound."""
        mound = MagicMock()
        mound.add = AsyncMock(return_value=MagicMock(node_id="kn_123"))
        return mound

    @pytest.mark.asyncio
    async def test_index_multiple_repos(self, mock_mound):
        """Should index multiple repositories."""
        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        repos = [
            RepoConfig(path="/repo1", workspace_id="ws1"),
            RepoConfig(path="/repo2", workspace_id="ws2"),
        ]

        mock_result = IndexResult(
            repository_name="test-repo",
            repository_path="/repo",
            workspace_id="ws_test",
            files_indexed=5,
            nodes_created=10,
            relationships_created=2,
            symbols_indexed=7,
            errors=[],
            warnings=[],
            duration_ms=500.0,
        )

        with patch.object(orchestrator, "index_repository", new_callable=AsyncMock) as mock_index:
            mock_index.return_value = mock_result

            batch_result = await orchestrator.index_multiple(repos)

            assert len(batch_result.repositories) == 2
            assert batch_result.total_files == 10
            assert batch_result.successful == 2
            assert batch_result.failed == 0

    @pytest.mark.asyncio
    async def test_index_multiple_respects_priority(self, mock_mound):
        """Should process repos by priority (highest first)."""
        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        repos = [
            RepoConfig(path="/repo1", workspace_id="ws1", priority=1),
            RepoConfig(path="/repo2", workspace_id="ws2", priority=10),
            RepoConfig(path="/repo3", workspace_id="ws3", priority=5),
        ]

        call_order = []

        async def track_calls(repo_path, workspace_id, **kwargs):
            call_order.append(repo_path)
            return IndexResult(
                repository_name=Path(repo_path).name,
                repository_path=repo_path,
                workspace_id=workspace_id,
                files_indexed=1,
                nodes_created=1,
                relationships_created=0,
                symbols_indexed=0,
                errors=[],
                warnings=[],
                duration_ms=100.0,
            )

        with patch.object(orchestrator, "index_repository", side_effect=track_calls):
            await orchestrator.index_multiple(repos)

            # First call should be highest priority
            assert call_order[0] == "/repo2"

    @pytest.mark.asyncio
    async def test_index_multiple_handles_failures(self, mock_mound):
        """Should handle individual repo failures."""
        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        repos = [
            RepoConfig(path="/repo1", workspace_id="ws1"),
            RepoConfig(path="/repo2", workspace_id="ws2"),
        ]

        results = [
            IndexResult(
                repository_name="repo1",
                repository_path="/repo1",
                workspace_id="ws1",
                files_indexed=5,
                nodes_created=10,
                relationships_created=2,
                symbols_indexed=7,
                errors=[],
                warnings=[],
                duration_ms=500.0,
            ),
            IndexResult(
                repository_name="repo2",
                repository_path="/repo2",
                workspace_id="ws2",
                files_indexed=0,
                nodes_created=0,
                relationships_created=0,
                symbols_indexed=0,
                errors=["Failed to crawl"],
                warnings=[],
                duration_ms=100.0,
            ),
        ]

        with patch.object(orchestrator, "index_repository", new_callable=AsyncMock) as mock_index:
            mock_index.side_effect = results

            batch_result = await orchestrator.index_multiple(repos)

            assert batch_result.successful == 1
            assert batch_result.failed == 1

    @pytest.mark.asyncio
    async def test_index_multiple_respects_concurrency(self, mock_mound):
        """Should respect max_concurrent_repos setting."""
        config = OrchestratorConfig(max_concurrent_repos=1)
        orchestrator = RepositoryOrchestrator(mound=mock_mound, config=config)

        repos = [
            RepoConfig(path="/repo1", workspace_id="ws1"),
            RepoConfig(path="/repo2", workspace_id="ws2"),
        ]

        concurrent_count = 0
        max_concurrent = 0

        async def track_concurrent(repo_path, workspace_id, **kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)  # Small delay to test concurrency
            concurrent_count -= 1
            return IndexResult(
                repository_name=Path(repo_path).name,
                repository_path=repo_path,
                workspace_id=workspace_id,
                files_indexed=1,
                nodes_created=1,
                relationships_created=0,
                symbols_indexed=0,
                errors=[],
                warnings=[],
                duration_ms=100.0,
            )

        with patch.object(orchestrator, "index_repository", side_effect=track_concurrent):
            await orchestrator.index_multiple(repos)

            # With semaphore of 1, max concurrent should be 1
            assert max_concurrent == 1


class TestRepositoryOrchestratorProgress:
    """Tests for progress tracking methods."""

    def test_get_progress_returns_none_for_unknown(self):
        """Should return None for unknown repo path."""
        mock_mound = MagicMock()
        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        progress = orchestrator.get_progress("/unknown/repo")
        assert progress is None

    def test_get_all_progress_empty(self):
        """Should return empty dict when no progress tracked."""
        mock_mound = MagicMock()
        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        all_progress = orchestrator.get_all_progress()
        assert all_progress == {}

    @pytest.mark.asyncio
    async def test_get_all_progress_after_indexing(self):
        """Should return all tracked progress."""
        mock_mound = MagicMock()
        mock_mound.add = AsyncMock(return_value=MagicMock(node_id="kn_123"))
        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        mock_crawl_result = CrawlResult(
            repository_path="/repo",
            repository_name="test-repo",
            files=[],
            total_files=0,
            total_lines=0,
            total_bytes=0,
            file_type_counts={},
            symbol_counts={},
            dependency_graph={},
            crawl_duration_ms=100.0,
            errors=[],
            warnings=[],
        )

        with patch.object(orchestrator._crawler, "crawl", new_callable=AsyncMock) as mock_crawl:
            mock_crawl.return_value = mock_crawl_result

            await orchestrator.index_repository("/repo1", "ws1")
            await orchestrator.index_repository("/repo2", "ws2")

            all_progress = orchestrator.get_all_progress()
            assert "/repo1" in all_progress
            assert "/repo2" in all_progress


class TestRepositoryOrchestratorIndexFile:
    """Tests for _index_file internal method."""

    @pytest.fixture
    def mock_mound(self):
        """Create a mock KnowledgeMound."""
        mound = MagicMock()
        mound.add = AsyncMock(return_value=MagicMock(node_id="kn_123"))
        return mound

    @pytest.mark.asyncio
    async def test_index_file_with_chunks(self, mock_mound):
        """Should index file chunks."""
        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        crawled_file = CrawledFile(
            path="/repo/main.py",
            relative_path="main.py",
            file_type=FileType.PYTHON,
            content="print('hello')",
            content_hash="abc123",
            size_bytes=15,
            line_count=1,
            chunks=[{"index": 0, "content": "print('hello')", "start_line": 1, "end_line": 1}],
        )

        nodes, rels, syms = await orchestrator._index_file(crawled_file, "test-repo", "ws_test")

        assert nodes >= 1
        mock_mound.add.assert_called()

    @pytest.mark.asyncio
    async def test_index_file_with_symbols(self, mock_mound):
        """Should index file symbols with docstrings."""
        config = OrchestratorConfig(store_symbols=True, store_chunks=False)
        orchestrator = RepositoryOrchestrator(mound=mock_mound, config=config)

        crawled_file = CrawledFile(
            path="/repo/main.py",
            relative_path="main.py",
            file_type=FileType.PYTHON,
            content="def main(): pass",
            content_hash="abc123",
            size_bytes=20,
            line_count=1,
            symbols=[
                FileSymbol(
                    name="main",
                    kind="function",
                    line_start=1,
                    line_end=1,
                    signature="def main()",
                    docstring="Main entry point",
                )
            ],
        )

        nodes, rels, syms = await orchestrator._index_file(crawled_file, "test-repo", "ws_test")

        assert syms == 1
        assert nodes >= 1

    @pytest.mark.asyncio
    async def test_index_file_with_dependencies(self, mock_mound):
        """Should index file dependencies."""
        config = OrchestratorConfig(
            store_dependencies=True, store_chunks=False, store_symbols=False
        )
        orchestrator = RepositoryOrchestrator(mound=mock_mound, config=config)

        crawled_file = CrawledFile(
            path="/repo/main.py",
            relative_path="main.py",
            file_type=FileType.PYTHON,
            content="import os",
            content_hash="abc123",
            size_bytes=10,
            line_count=1,
            dependencies=[FileDependency(source="main.py", target="os", kind="import", line=1)],
        )

        nodes, rels, syms = await orchestrator._index_file(crawled_file, "test-repo", "ws_test")

        assert rels >= 1

    @pytest.mark.asyncio
    async def test_index_file_skips_disabled_options(self, mock_mound):
        """Should skip indexing when options are disabled."""
        config = OrchestratorConfig(
            store_chunks=False, store_symbols=False, store_dependencies=False
        )
        orchestrator = RepositoryOrchestrator(mound=mock_mound, config=config)

        crawled_file = CrawledFile(
            path="/repo/main.py",
            relative_path="main.py",
            file_type=FileType.PYTHON,
            content="import os",
            content_hash="abc123",
            size_bytes=10,
            line_count=1,
            chunks=[{"index": 0, "content": "import os", "start_line": 1, "end_line": 1}],
            symbols=[FileSymbol(name="test", kind="function", line_start=1, line_end=1)],
            dependencies=[FileDependency(source="main.py", target="os", kind="import", line=1)],
        )

        nodes, rels, syms = await orchestrator._index_file(crawled_file, "test-repo", "ws_test")

        assert nodes == 0
        assert rels == 0
        mock_mound.add.assert_not_called()


class TestRepositoryOrchestratorIndexDependencyGraph:
    """Tests for _index_dependency_graph internal method."""

    @pytest.fixture
    def mock_mound(self):
        """Create a mock KnowledgeMound."""
        mound = MagicMock()
        mound.add = AsyncMock(return_value=MagicMock(node_id="kn_123"))
        return mound

    @pytest.mark.asyncio
    async def test_index_dependency_graph(self, mock_mound):
        """Should index dependency graph relationships."""
        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        dependency_graph = {
            "main.py": ["utils.py", "config.py"],
            "utils.py": ["helpers.py"],
        }

        rels = await orchestrator._index_dependency_graph(dependency_graph, "test-repo", "ws_test")

        assert rels == 3  # 2 + 1 relationships
        assert mock_mound.add.call_count == 3

    @pytest.mark.asyncio
    async def test_index_dependency_graph_empty(self, mock_mound):
        """Should handle empty dependency graph."""
        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        rels = await orchestrator._index_dependency_graph({}, "test-repo", "ws_test")

        assert rels == 0
        mock_mound.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_index_dependency_graph_handles_errors(self, mock_mound):
        """Should continue on individual edge errors."""
        mock_mound.add = AsyncMock(side_effect=RuntimeError("Add failed"))
        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        dependency_graph = {"main.py": ["utils.py"]}

        rels = await orchestrator._index_dependency_graph(dependency_graph, "test-repo", "ws_test")

        # Should return 0 since all additions failed
        assert rels == 0


class TestRepositoryOrchestratorFormatSymbol:
    """Tests for _format_symbol_content internal method."""

    def test_format_symbol_basic(self):
        """Should format basic symbol."""
        mock_mound = MagicMock()
        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        symbol = FileSymbol(
            name="my_func",
            kind="function",
            line_start=10,
            line_end=20,
        )

        content = orchestrator._format_symbol_content(symbol, "main.py")

        assert "function my_func" in content
        assert "main.py:10" in content

    def test_format_symbol_with_parent(self):
        """Should include parent in formatted content."""
        mock_mound = MagicMock()
        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        symbol = FileSymbol(
            name="my_method",
            kind="method",
            line_start=15,
            line_end=25,
            parent="MyClass",
        )

        content = orchestrator._format_symbol_content(symbol, "main.py")

        assert "in MyClass" in content

    def test_format_symbol_with_signature(self):
        """Should include signature in formatted content."""
        mock_mound = MagicMock()
        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        symbol = FileSymbol(
            name="my_func",
            kind="function",
            line_start=10,
            line_end=20,
            signature="def my_func(arg1: str, arg2: int) -> bool",
        )

        content = orchestrator._format_symbol_content(symbol, "main.py")

        assert "Signature:" in content
        assert "def my_func(arg1: str, arg2: int) -> bool" in content

    def test_format_symbol_with_docstring(self):
        """Should include docstring in formatted content."""
        mock_mound = MagicMock()
        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        symbol = FileSymbol(
            name="my_func",
            kind="function",
            line_start=10,
            line_end=20,
            docstring="This is a docstring explaining the function.",
        )

        content = orchestrator._format_symbol_content(symbol, "main.py")

        assert "This is a docstring explaining the function." in content


class TestRepositoryOrchestratorRemoveRepository:
    """Tests for remove_repository method."""

    @pytest.fixture
    def mock_mound(self):
        """Create a mock KnowledgeMound."""
        mound = MagicMock()
        mound.query = AsyncMock()
        mound.delete = AsyncMock()
        return mound

    @pytest.mark.asyncio
    async def test_remove_repository_success(self, mock_mound):
        """Should remove all repository nodes."""
        mock_item1 = MagicMock()
        mock_item1.id = "kn_1"
        mock_item2 = MagicMock()
        mock_item2.id = "kn_2"

        mock_mound.query.return_value = MagicMock(items=[mock_item1, mock_item2])

        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        removed = await orchestrator.remove_repository("test-repo", "ws_test")

        assert removed == 2
        assert mock_mound.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_remove_repository_no_items(self, mock_mound):
        """Should handle repository with no items."""
        mock_mound.query.return_value = MagicMock(items=[])

        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        removed = await orchestrator.remove_repository("test-repo", "ws_test")

        assert removed == 0
        mock_mound.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_remove_repository_handles_error(self, mock_mound):
        """Should handle query errors gracefully."""
        mock_mound.query.side_effect = RuntimeError("Query failed")

        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        removed = await orchestrator.remove_repository("test-repo", "ws_test")

        assert removed == 0

    @pytest.mark.asyncio
    async def test_remove_repository_without_delete_method(self, mock_mound):
        """Should handle mound without delete method."""
        delattr(mock_mound, "delete")
        mock_item = MagicMock()
        mock_item.id = "kn_1"
        mock_mound.query.return_value = MagicMock(items=[mock_item])

        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        removed = await orchestrator.remove_repository("test-repo", "ws_test")

        # Should not crash, but also not delete anything
        assert removed == 0


class TestRepositoryOrchestratorGetStats:
    """Tests for get_repository_stats method."""

    @pytest.fixture
    def mock_mound(self):
        """Create a mock KnowledgeMound."""
        mound = MagicMock()
        mound.query = AsyncMock()
        return mound

    @pytest.mark.asyncio
    async def test_get_stats_indexed_repo(self, mock_mound):
        """Should return stats for indexed repository."""
        mock_mound.query.return_value = MagicMock(total_count=100)

        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        stats = await orchestrator.get_repository_stats("test-repo", "ws_test")

        assert stats["repository_name"] == "test-repo"
        assert stats["workspace_id"] == "ws_test"
        assert stats["indexed"] is True
        assert stats["total_nodes"] == 100

    @pytest.mark.asyncio
    async def test_get_stats_unindexed_repo(self, mock_mound):
        """Should return stats for unindexed repository."""
        mock_mound.query.return_value = MagicMock(total_count=0)

        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        stats = await orchestrator.get_repository_stats("test-repo", "ws_test")

        assert stats["indexed"] is False
        assert stats["total_nodes"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_handles_error(self, mock_mound):
        """Should handle query errors gracefully."""
        mock_mound.query.side_effect = RuntimeError("Query failed")

        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        stats = await orchestrator.get_repository_stats("test-repo", "ws_test")

        assert stats["repository_name"] == "test-repo"
        assert stats["indexed"] is False
        assert "error" in stats


# =============================================================================
# Integration Tests
# =============================================================================


class TestRepositoryOrchestratorIntegration:
    """Integration tests for RepositoryOrchestrator."""

    def test_all_classes_importable(self):
        """Should be able to import all public classes."""
        from aragora.knowledge.repository_orchestrator import (
            RepoConfig,
            IndexResult,
            BatchResult,
            OrchestratorConfig,
            IndexProgress,
            RepositoryOrchestrator,
        )

        assert RepoConfig is not None
        assert IndexResult is not None
        assert BatchResult is not None
        assert OrchestratorConfig is not None
        assert IndexProgress is not None
        assert RepositoryOrchestrator is not None

    def test_orchestrator_methods_exist(self):
        """Should have all expected public methods."""
        mock_mound = MagicMock()
        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        # Public methods
        assert hasattr(orchestrator, "index_repository")
        assert hasattr(orchestrator, "incremental_update")
        assert hasattr(orchestrator, "index_multiple")
        assert hasattr(orchestrator, "get_progress")
        assert hasattr(orchestrator, "get_all_progress")
        assert hasattr(orchestrator, "remove_repository")
        assert hasattr(orchestrator, "get_repository_stats")

        # Internal methods
        assert hasattr(orchestrator, "_index_file")
        assert hasattr(orchestrator, "_index_dependency_graph")
        assert hasattr(orchestrator, "_format_symbol_content")

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete indexing workflow."""
        mock_mound = MagicMock()
        mock_mound.add = AsyncMock(return_value=MagicMock(node_id="kn_123"))
        mock_mound.query = AsyncMock(return_value=MagicMock(items=[], total_count=5))

        orchestrator = RepositoryOrchestrator(mound=mock_mound)

        # Create mock crawl result
        crawled_file = CrawledFile(
            path="/repo/main.py",
            relative_path="main.py",
            file_type=FileType.PYTHON,
            content="def main(): pass",
            content_hash="abc123",
            size_bytes=20,
            line_count=1,
            symbols=[
                FileSymbol(
                    name="main",
                    kind="function",
                    line_start=1,
                    line_end=1,
                    docstring="Main function",
                )
            ],
            chunks=[{"index": 0, "content": "def main(): pass", "start_line": 1, "end_line": 1}],
        )

        mock_crawl_result = CrawlResult(
            repository_path="/repo",
            repository_name="test-repo",
            files=[crawled_file],
            total_files=1,
            total_lines=1,
            total_bytes=20,
            file_type_counts={"python": 1},
            symbol_counts={"function": 1},
            dependency_graph={},
            crawl_duration_ms=100.0,
            errors=[],
            warnings=[],
        )

        with patch.object(orchestrator._crawler, "crawl", new_callable=AsyncMock) as mock_crawl:
            mock_crawl.return_value = mock_crawl_result

            # Index repository
            result = await orchestrator.index_repository("/repo", "ws_test")

            assert result.files_indexed == 1
            assert result.repository_name == "test-repo"

            # Check progress
            progress = orchestrator.get_progress("/repo")
            assert progress.status == "completed"

            # Get stats
            stats = await orchestrator.get_repository_stats("test-repo", "ws_test")
            assert stats["indexed"] is True
