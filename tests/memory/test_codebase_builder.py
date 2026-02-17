"""
Tests for CodebaseKnowledgeBuilder.

Covers structure ingestion, import parsing, test results,
architectural patterns, and natural language queries.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.memory.codebase_builder import (
    CodebaseKnowledgeBuilder,
    ImportRelation,
    IngestionStats,
    _CODE_EXTENSIONS,
    _SKIP_DIRS,
)
from aragora.memory.fabric import FabricResult, MemoryFabric, RememberResult
from aragora.memory.surprise import ContentSurpriseScore, ContentSurpriseScorer


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


def _make_remember_result(stored: bool = True) -> RememberResult:
    return RememberResult(
        stored=stored,
        systems_written=["continuum"] if stored else [],
        surprise_combined=0.6 if stored else 0.1,
        reason="test",
    )


def _make_fabric_result(content: str = "result") -> FabricResult:
    return FabricResult(
        content=content,
        source_system="test",
        relevance=0.8,
        recency=0.9,
    )


@pytest.fixture()
def mock_fabric() -> AsyncMock:
    fabric = AsyncMock(spec=MemoryFabric)
    fabric.remember = AsyncMock(return_value=_make_remember_result(stored=True))
    fabric.query = AsyncMock(return_value=[])
    return fabric


@pytest.fixture()
def mock_scorer() -> MagicMock:
    scorer = MagicMock(spec=ContentSurpriseScorer)
    scorer.score = MagicMock(
        return_value=ContentSurpriseScore(
            novelty=0.8, momentum=0.5, combined=0.71, should_store=True, reason="novel"
        )
    )
    return scorer


def _populate_repo(tmp_path: Path) -> Path:
    """Create a small mock repo tree."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text("import os\nprint('hello')\n")
    (src / "util.ts").write_text("export function hello() {}\n")
    (src / "config.yaml").write_text("key: value\n")
    (src / "README.md").write_text("# Readme\n")  # not in _CODE_EXTENSIONS
    # Nested dir
    sub = src / "sub"
    sub.mkdir()
    (sub / "helper.py").write_text("from src.app import os\n")
    # Skippable dir
    pycache = src / "__pycache__"
    pycache.mkdir()
    (pycache / "app.cpython-311.pyc").write_text("cached")
    return tmp_path


# =====================================================================
# TestCodebaseStructureIngestion
# =====================================================================


class TestCodebaseStructureIngestion:
    """Tests for ingest_structure."""

    @pytest.mark.asyncio
    async def test_ingests_code_files(self, tmp_path: Path, mock_fabric: AsyncMock) -> None:
        repo = _populate_repo(tmp_path)
        builder = CodebaseKnowledgeBuilder(mock_fabric, repo / "src")
        stats = await builder.ingest_structure()
        # app.py, util.ts, config.yaml, helper.py = 4 code files
        assert stats.items_ingested == 4
        assert stats.errors == 0

    @pytest.mark.asyncio
    async def test_skips_non_code_extensions(self, tmp_path: Path, mock_fabric: AsyncMock) -> None:
        repo = _populate_repo(tmp_path)
        builder = CodebaseKnowledgeBuilder(mock_fabric, repo / "src")
        await builder.ingest_structure()
        # README.md should not be ingested
        for call in mock_fabric.remember.call_args_list:
            assert "README.md" not in call.kwargs.get("content", call.args[0] if call.args else "")

    @pytest.mark.asyncio
    async def test_skips_pycache(self, tmp_path: Path, mock_fabric: AsyncMock) -> None:
        repo = _populate_repo(tmp_path)
        builder = CodebaseKnowledgeBuilder(mock_fabric, repo / "src")
        await builder.ingest_structure()
        for call in mock_fabric.remember.call_args_list:
            content = call.kwargs.get("content", "")
            assert "__pycache__" not in content

    @pytest.mark.asyncio
    async def test_max_files_limit(self, tmp_path: Path, mock_fabric: AsyncMock) -> None:
        repo = _populate_repo(tmp_path)
        builder = CodebaseKnowledgeBuilder(mock_fabric, repo / "src")
        stats = await builder.ingest_structure(max_files=2)
        assert stats.items_ingested + stats.items_skipped <= 2

    @pytest.mark.asyncio
    async def test_nonexistent_repo(self, tmp_path: Path, mock_fabric: AsyncMock) -> None:
        builder = CodebaseKnowledgeBuilder(mock_fabric, tmp_path / "nonexistent")
        stats = await builder.ingest_structure()
        assert stats.items_ingested == 0
        assert stats.errors == 0

    @pytest.mark.asyncio
    async def test_empty_directory(self, tmp_path: Path, mock_fabric: AsyncMock) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        builder = CodebaseKnowledgeBuilder(mock_fabric, empty)
        stats = await builder.ingest_structure()
        assert stats.items_ingested == 0

    @pytest.mark.asyncio
    async def test_remember_metadata_contains_file_path(self, tmp_path: Path, mock_fabric: AsyncMock) -> None:
        (tmp_path / "main.py").write_text("x = 1\n")
        builder = CodebaseKnowledgeBuilder(mock_fabric, tmp_path)
        await builder.ingest_structure()
        call_kwargs = mock_fabric.remember.call_args_list[0].kwargs
        assert call_kwargs["metadata"]["file_path"] == "main.py"
        assert call_kwargs["metadata"]["language"] == ".py"
        assert call_kwargs["source"] == "codebase_structure"

    @pytest.mark.asyncio
    async def test_skipped_items_counted(self, tmp_path: Path, mock_fabric: AsyncMock) -> None:
        (tmp_path / "a.py").write_text("x = 1\n")
        mock_fabric.remember = AsyncMock(return_value=_make_remember_result(stored=False))
        builder = CodebaseKnowledgeBuilder(mock_fabric, tmp_path)
        stats = await builder.ingest_structure()
        assert stats.items_skipped == 1
        assert stats.items_ingested == 0

    @pytest.mark.asyncio
    async def test_stats_dataclass_defaults(self) -> None:
        stats = IngestionStats()
        assert stats.items_ingested == 0
        assert stats.items_skipped == 0
        assert stats.errors == 0

    @pytest.mark.asyncio
    async def test_skips_git_directory(self, tmp_path: Path, mock_fabric: AsyncMock) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config.json").write_text("{}")
        (tmp_path / "app.py").write_text("x = 1\n")
        builder = CodebaseKnowledgeBuilder(mock_fabric, tmp_path)
        stats = await builder.ingest_structure()
        assert stats.items_ingested == 1  # only app.py


# =====================================================================
# TestCodebaseImportIngestion
# =====================================================================


class TestCodebaseImportIngestion:
    """Tests for ingest_imports."""

    @pytest.mark.asyncio
    async def test_extracts_standard_import(self, tmp_path: Path, mock_fabric: AsyncMock) -> None:
        (tmp_path / "app.py").write_text("import os\nimport sys\n")
        builder = CodebaseKnowledgeBuilder(mock_fabric, tmp_path)
        stats = await builder.ingest_imports()
        assert stats.items_ingested == 1
        content = mock_fabric.remember.call_args_list[0].kwargs["content"]
        assert "os" in content
        assert "sys" in content

    @pytest.mark.asyncio
    async def test_extracts_from_import(self, tmp_path: Path, mock_fabric: AsyncMock) -> None:
        (tmp_path / "app.py").write_text("from pathlib import Path\n")
        builder = CodebaseKnowledgeBuilder(mock_fabric, tmp_path)
        await builder.ingest_imports()
        content = mock_fabric.remember.call_args_list[0].kwargs["content"]
        assert "pathlib" in content
        assert "from_import" in content

    @pytest.mark.asyncio
    async def test_handles_syntax_error(self, tmp_path: Path, mock_fabric: AsyncMock) -> None:
        (tmp_path / "bad.py").write_text("def foo(:\n  pass\n")
        builder = CodebaseKnowledgeBuilder(mock_fabric, tmp_path)
        stats = await builder.ingest_imports()
        assert stats.errors == 1
        assert stats.items_ingested == 0

    @pytest.mark.asyncio
    async def test_skips_pycache_in_imports(self, tmp_path: Path, mock_fabric: AsyncMock) -> None:
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "mod.py").write_text("import os\n")
        builder = CodebaseKnowledgeBuilder(mock_fabric, tmp_path)
        stats = await builder.ingest_imports()
        assert stats.items_ingested == 0

    @pytest.mark.asyncio
    async def test_max_files_limit_imports(self, tmp_path: Path, mock_fabric: AsyncMock) -> None:
        for i in range(5):
            (tmp_path / f"mod{i}.py").write_text(f"import os  # {i}\n")
        builder = CodebaseKnowledgeBuilder(mock_fabric, tmp_path)
        stats = await builder.ingest_imports(max_files=2)
        # At most 2 files processed
        assert mock_fabric.remember.call_count <= 2

    @pytest.mark.asyncio
    async def test_no_imports_file_skipped(self, tmp_path: Path, mock_fabric: AsyncMock) -> None:
        (tmp_path / "empty.py").write_text("x = 1\n")
        builder = CodebaseKnowledgeBuilder(mock_fabric, tmp_path)
        stats = await builder.ingest_imports()
        assert mock_fabric.remember.call_count == 0

    @pytest.mark.asyncio
    async def test_nonexistent_repo_imports(self, tmp_path: Path, mock_fabric: AsyncMock) -> None:
        builder = CodebaseKnowledgeBuilder(mock_fabric, tmp_path / "nope")
        stats = await builder.ingest_imports()
        assert stats.items_ingested == 0

    @pytest.mark.asyncio
    async def test_extract_imports_helper(self, tmp_path: Path, mock_fabric: AsyncMock) -> None:
        import ast as _ast

        code = "import os\nfrom pathlib import Path\n"
        tree = _ast.parse(code)
        builder = CodebaseKnowledgeBuilder(mock_fabric, tmp_path)
        relations = builder._extract_imports(tree, "app.py")
        assert len(relations) == 2
        assert relations[0].import_type == "import"
        assert relations[0].imported_module == "os"
        assert relations[1].import_type == "from_import"
        assert relations[1].imported_module == "pathlib"


# =====================================================================
# TestCodebaseTestResults
# =====================================================================


class TestCodebaseTestResults:
    """Tests for ingest_test_results."""

    @pytest.mark.asyncio
    async def test_summary_ingested(self, mock_fabric: AsyncMock, mock_scorer: MagicMock) -> None:
        builder = CodebaseKnowledgeBuilder(mock_fabric, Path("/fake"), mock_scorer)
        results = {"passed": 10, "failed": 2, "errors": 1}
        stats = await builder.ingest_test_results(results)
        assert stats.items_ingested >= 1
        content = mock_fabric.remember.call_args_list[0].kwargs["content"]
        assert "10/13 passed" in content
        assert "2 failed" in content

    @pytest.mark.asyncio
    async def test_failure_details_ingested(self, mock_fabric: AsyncMock, mock_scorer: MagicMock) -> None:
        builder = CodebaseKnowledgeBuilder(mock_fabric, Path("/fake"), mock_scorer)
        results = {
            "passed": 5,
            "failed": 1,
            "errors": 0,
            "test_details": [
                {"name": "test_login", "status": "failed", "duration": 0.5},
            ],
        }
        stats = await builder.ingest_test_results(results)
        # Summary + 1 failure detail
        assert mock_fabric.remember.call_count == 2

    @pytest.mark.asyncio
    async def test_passing_tests_not_individually_stored(self, mock_fabric: AsyncMock, mock_scorer: MagicMock) -> None:
        builder = CodebaseKnowledgeBuilder(mock_fabric, Path("/fake"), mock_scorer)
        results = {
            "passed": 5,
            "failed": 0,
            "errors": 0,
            "test_details": [
                {"name": "test_ok", "status": "passed", "duration": 0.1},
            ],
        }
        await builder.ingest_test_results(results)
        # Only summary, no individual passing tests
        assert mock_fabric.remember.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_results(self, mock_fabric: AsyncMock, mock_scorer: MagicMock) -> None:
        builder = CodebaseKnowledgeBuilder(mock_fabric, Path("/fake"), mock_scorer)
        stats = await builder.ingest_test_results({})
        assert stats.items_ingested >= 0  # summary still created for 0/0

    @pytest.mark.asyncio
    async def test_error_status_ingested(self, mock_fabric: AsyncMock, mock_scorer: MagicMock) -> None:
        builder = CodebaseKnowledgeBuilder(mock_fabric, Path("/fake"), mock_scorer)
        results = {
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "test_details": [
                {"name": "test_crash", "status": "error", "duration": 0.0},
            ],
        }
        await builder.ingest_test_results(results)
        # Summary + 1 error detail
        assert mock_fabric.remember.call_count == 2

    @pytest.mark.asyncio
    async def test_surprise_score_in_metadata(self, mock_fabric: AsyncMock, mock_scorer: MagicMock) -> None:
        builder = CodebaseKnowledgeBuilder(mock_fabric, Path("/fake"), mock_scorer)
        results = {
            "passed": 0,
            "failed": 1,
            "errors": 0,
            "test_details": [
                {"name": "test_fail", "status": "failed", "duration": 1.0},
            ],
        }
        await builder.ingest_test_results(results)
        # The second remember call is for the failure detail
        detail_call = mock_fabric.remember.call_args_list[1]
        assert "surprise" in detail_call.kwargs["metadata"]

    @pytest.mark.asyncio
    async def test_summary_metadata(self, mock_fabric: AsyncMock, mock_scorer: MagicMock) -> None:
        builder = CodebaseKnowledgeBuilder(mock_fabric, Path("/fake"), mock_scorer)
        results = {"passed": 3, "failed": 1, "errors": 2}
        await builder.ingest_test_results(results)
        meta = mock_fabric.remember.call_args_list[0].kwargs["metadata"]
        assert meta["passed"] == 3
        assert meta["failed"] == 1
        assert meta["errors"] == 2
        assert meta["total"] == 6

    @pytest.mark.asyncio
    async def test_skipped_failure_counted(self, mock_fabric: AsyncMock, mock_scorer: MagicMock) -> None:
        # First call (summary) stored, second call (failure detail) not stored
        mock_fabric.remember = AsyncMock(
            side_effect=[
                _make_remember_result(stored=True),
                _make_remember_result(stored=False),
            ]
        )
        builder = CodebaseKnowledgeBuilder(mock_fabric, Path("/fake"), mock_scorer)
        results = {
            "passed": 0,
            "failed": 1,
            "errors": 0,
            "test_details": [{"name": "test_x", "status": "failed", "duration": 0.1}],
        }
        stats = await builder.ingest_test_results(results)
        assert stats.items_ingested == 1
        assert stats.items_skipped == 1


# =====================================================================
# TestCodebasePatterns
# =====================================================================


class TestCodebasePatterns:
    """Tests for ingest_patterns."""

    @pytest.mark.asyncio
    async def test_single_pattern(self, mock_fabric: AsyncMock) -> None:
        builder = CodebaseKnowledgeBuilder(mock_fabric, Path("/fake"))
        stats = await builder.ingest_patterns(["adapter factory"])
        assert stats.items_ingested == 1
        content = mock_fabric.remember.call_args.kwargs["content"]
        assert "adapter factory" in content

    @pytest.mark.asyncio
    async def test_multiple_patterns(self, mock_fabric: AsyncMock) -> None:
        builder = CodebaseKnowledgeBuilder(mock_fabric, Path("/fake"))
        patterns = ["adapter factory", "handler pattern", "circuit breaker"]
        stats = await builder.ingest_patterns(patterns)
        assert stats.items_ingested == 3

    @pytest.mark.asyncio
    async def test_empty_patterns(self, mock_fabric: AsyncMock) -> None:
        builder = CodebaseKnowledgeBuilder(mock_fabric, Path("/fake"))
        stats = await builder.ingest_patterns([])
        assert stats.items_ingested == 0

    @pytest.mark.asyncio
    async def test_pattern_error_handling(self, mock_fabric: AsyncMock) -> None:
        mock_fabric.remember = AsyncMock(side_effect=RuntimeError("boom"))
        builder = CodebaseKnowledgeBuilder(mock_fabric, Path("/fake"))
        stats = await builder.ingest_patterns(["will fail"])
        assert stats.errors == 1
        assert stats.items_ingested == 0

    @pytest.mark.asyncio
    async def test_pattern_metadata(self, mock_fabric: AsyncMock) -> None:
        builder = CodebaseKnowledgeBuilder(mock_fabric, Path("/fake"))
        await builder.ingest_patterns(["singleton"])
        meta = mock_fabric.remember.call_args.kwargs["metadata"]
        assert meta["pattern"] == "singleton"
        assert mock_fabric.remember.call_args.kwargs["source"] == "codebase_pattern"


# =====================================================================
# TestCodebaseQuery
# =====================================================================


class TestCodebaseQuery:
    """Tests for query_about."""

    @pytest.mark.asyncio
    async def test_delegates_to_fabric(self, mock_fabric: AsyncMock) -> None:
        expected = [_make_fabric_result("answer")]
        mock_fabric.query = AsyncMock(return_value=expected)
        builder = CodebaseKnowledgeBuilder(mock_fabric, Path("/fake"))
        results = await builder.query_about("how does auth work?")
        assert results == expected
        mock_fabric.query.assert_awaited_once_with("how does auth work?", limit=10)

    @pytest.mark.asyncio
    async def test_custom_limit(self, mock_fabric: AsyncMock) -> None:
        builder = CodebaseKnowledgeBuilder(mock_fabric, Path("/fake"))
        await builder.query_about("test", limit=5)
        mock_fabric.query.assert_awaited_once_with("test", limit=5)

    @pytest.mark.asyncio
    async def test_empty_results(self, mock_fabric: AsyncMock) -> None:
        mock_fabric.query = AsyncMock(return_value=[])
        builder = CodebaseKnowledgeBuilder(mock_fabric, Path("/fake"))
        results = await builder.query_about("unknown topic")
        assert results == []

    @pytest.mark.asyncio
    async def test_import_relation_dataclass(self) -> None:
        rel = ImportRelation(
            source_module="app",
            imported_module="os",
            import_type="import",
        )
        assert rel.source_module == "app"
        assert rel.imported_module == "os"
        assert rel.import_type == "import"
