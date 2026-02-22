"""Tests for CodebaseIndexer — searchable codebase structure in Knowledge Mound."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.codebase_indexer import (
    CodebaseIndexer,
    IndexStats,
    ModuleInfo,
)


# ---------------------------------------------------------------------------
# TestModuleInfo
# ---------------------------------------------------------------------------


class TestModuleInfo:
    """Test ModuleInfo dataclass and serialization."""

    def test_to_km_entry_fields(self):
        info = ModuleInfo(
            path="aragora/debate/consensus.py",
            docstring="Consensus detection logic.",
            classes=["ConsensusEngine", "VoteAggregator"],
            functions=["detect_consensus"],
            imports_from=["aragora.debate.convergence"],
            line_count=200,
        )
        entry = info.to_km_entry()

        assert entry["type"] == "module"
        assert entry["path"] == "aragora/debate/consensus.py"
        assert entry["docstring"] == "Consensus detection logic."
        assert entry["classes"] == ["ConsensusEngine", "VoteAggregator"]
        assert entry["functions"] == ["detect_consensus"]
        assert entry["imports_from"] == ["aragora.debate.convergence"]
        assert entry["line_count"] == 200

    def test_searchable_text_includes_all_parts(self):
        info = ModuleInfo(
            path="aragora/billing/cost_tracker.py",
            docstring="Tracks API costs.",
            classes=["CostTracker"],
            functions=["get_daily_cost"],
            imports_from=[],
            line_count=50,
        )
        text = info.to_km_entry()["searchable_text"]

        assert "aragora/billing/cost_tracker.py" in text
        assert "Tracks API costs." in text
        assert "CostTracker" in text
        assert "get_daily_cost" in text

    def test_empty_module_info(self):
        info = ModuleInfo(
            path="aragora/__init__.py",
            docstring="",
            classes=[],
            functions=[],
            imports_from=[],
            line_count=1,
        )
        entry = info.to_km_entry()
        assert entry["classes"] == []
        assert entry["functions"] == []
        assert entry["searchable_text"].startswith("aragora/__init__.py")


# ---------------------------------------------------------------------------
# TestIndexStats
# ---------------------------------------------------------------------------


class TestIndexStats:
    """Test IndexStats dataclass and serialization."""

    def test_defaults(self):
        stats = IndexStats()
        assert stats.modules_indexed == 0
        assert stats.classes_found == 0
        assert stats.functions_found == 0
        assert stats.test_files_found == 0
        assert stats.total_lines == 0
        assert stats.errors == []

    def test_to_dict(self):
        stats = IndexStats(
            modules_indexed=10,
            classes_found=25,
            functions_found=40,
            test_files_found=8,
            total_lines=5000,
            errors=["file1.py: SyntaxError", "file2.py: UnicodeDecodeError"],
        )
        d = stats.to_dict()

        assert d["modules_indexed"] == 10
        assert d["classes_found"] == 25
        assert d["functions_found"] == 40
        assert d["test_files_found"] == 8
        assert d["total_lines"] == 5000
        assert d["errors_count"] == 2

    def test_to_dict_no_errors(self):
        stats = IndexStats(modules_indexed=5)
        d = stats.to_dict()
        assert d["errors_count"] == 0


# ---------------------------------------------------------------------------
# TestAnalyzeModule
# ---------------------------------------------------------------------------


class TestAnalyzeModule:
    """Test _analyze_module on real and synthetic files."""

    def test_analyze_real_module(self):
        """Analyze a real file from the aragora codebase."""
        repo = Path(__file__).resolve().parent.parent.parent
        indexer = CodebaseIndexer(repo_path=repo)
        target = repo / "aragora" / "nomic" / "outcome_tracker.py"

        if not target.exists():
            pytest.skip("outcome_tracker.py not found")

        info = indexer._analyze_module(target)
        assert info is not None
        assert info.path == "aragora/nomic/outcome_tracker.py"
        assert info.line_count > 0
        # The file has a docstring starting with "Nomic Loop Outcome Tracker"
        assert "Outcome Tracker" in info.docstring or "outcome" in info.docstring.lower()
        # Should find at least one class
        assert len(info.classes) > 0

    def test_extracts_docstring(self, tmp_path: Path):
        source = textwrap.dedent('''\
            """Module-level docstring here."""

            class Foo:
                pass
        ''')
        py = tmp_path / "mod.py"
        py.write_text(source)

        indexer = CodebaseIndexer(repo_path=tmp_path)
        info = indexer._analyze_module(py)

        assert info is not None
        assert info.docstring == "Module-level docstring here."

    def test_truncates_long_docstring(self, tmp_path: Path):
        long_doc = "A" * 300
        source = f'"""{long_doc}"""\n'
        py = tmp_path / "mod.py"
        py.write_text(source)

        indexer = CodebaseIndexer(repo_path=tmp_path)
        info = indexer._analyze_module(py)

        assert info is not None
        assert len(info.docstring) <= 200

    def test_extracts_classes_and_functions(self, tmp_path: Path):
        source = textwrap.dedent('''\
            """Mod."""

            class Alpha:
                def method(self):
                    pass

            class Beta:
                pass

            def top_level_func():
                pass

            async def async_top_level():
                pass
        ''')
        py = tmp_path / "mod.py"
        py.write_text(source)

        indexer = CodebaseIndexer(repo_path=tmp_path)
        info = indexer._analyze_module(py)

        assert info is not None
        assert "Alpha" in info.classes
        assert "Beta" in info.classes
        assert "top_level_func" in info.functions
        assert "async_top_level" in info.functions
        # Methods should NOT appear in functions
        assert "method" not in info.functions

    def test_extracts_aragora_imports(self, tmp_path: Path):
        source = textwrap.dedent('''\
            from aragora.debate.consensus import ConsensusEngine
            from aragora.memory.coordinator import MemoryCoordinator
            import os
        ''')
        py = tmp_path / "mod.py"
        py.write_text(source)

        indexer = CodebaseIndexer(repo_path=tmp_path)
        info = indexer._analyze_module(py)

        assert info is not None
        assert "aragora.debate.consensus" in info.imports_from
        assert "aragora.memory.coordinator" in info.imports_from
        # os is not an aragora import
        assert len(info.imports_from) == 2

    def test_deduplicates_imports(self, tmp_path: Path):
        source = textwrap.dedent('''\
            from aragora.debate.consensus import ConsensusEngine
            from aragora.debate.consensus import VoteAggregator
        ''')
        py = tmp_path / "mod.py"
        py.write_text(source)

        indexer = CodebaseIndexer(repo_path=tmp_path)
        info = indexer._analyze_module(py)

        assert info is not None
        assert info.imports_from.count("aragora.debate.consensus") == 1

    def test_handles_syntax_error(self, tmp_path: Path):
        py = tmp_path / "bad.py"
        py.write_text("def broken(:\n")

        indexer = CodebaseIndexer(repo_path=tmp_path)
        info = indexer._analyze_module(py)

        assert info is None

    def test_handles_empty_file(self, tmp_path: Path):
        py = tmp_path / "empty.py"
        py.write_text("")

        indexer = CodebaseIndexer(repo_path=tmp_path)
        info = indexer._analyze_module(py)

        assert info is not None
        assert info.docstring == ""
        assert info.classes == []
        assert info.functions == []
        assert info.line_count == 0

    def test_handles_unreadable_file(self, tmp_path: Path):
        py = tmp_path / "gone.py"
        # File does not exist at all
        indexer = CodebaseIndexer(repo_path=tmp_path)
        info = indexer._analyze_module(py)

        assert info is None

    def test_line_count_accurate(self, tmp_path: Path):
        source = "line1\nline2\nline3\n"
        py = tmp_path / "mod.py"
        py.write_text(source)

        indexer = CodebaseIndexer(repo_path=tmp_path)
        info = indexer._analyze_module(py)

        assert info is not None
        assert info.line_count == 3


# ---------------------------------------------------------------------------
# TestIndex
# ---------------------------------------------------------------------------


class TestIndex:
    """Test the full index() scan."""

    @pytest.mark.asyncio
    async def test_index_finds_modules(self, tmp_path: Path):
        """Index finds .py files in source directories."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "__init__.py").write_text('"""Package."""\n')
        (src / "alpha.py").write_text(
            '"""Alpha module."""\n\nclass AlphaClass:\n    pass\n'
        )
        (src / "beta.py").write_text(
            '"""Beta module."""\n\ndef beta_func():\n    pass\n'
        )

        indexer = CodebaseIndexer(
            repo_path=tmp_path,
            source_dirs=["src"],
            test_dirs=[],
        )
        stats = await indexer.index()

        assert stats.modules_indexed == 3  # __init__, alpha, beta
        assert stats.classes_found >= 1
        assert stats.functions_found >= 1

    @pytest.mark.asyncio
    async def test_index_respects_max_modules(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        for i in range(20):
            (src / f"mod{i:02d}.py").write_text(f'"""Module {i}."""\n')

        indexer = CodebaseIndexer(
            repo_path=tmp_path,
            source_dirs=["src"],
            test_dirs=[],
            max_modules=5,
        )
        stats = await indexer.index()

        assert stats.modules_indexed == 5

    @pytest.mark.asyncio
    async def test_index_finds_test_files(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "core.py").write_text('"""Core."""\n')

        tdir = tmp_path / "tests"
        tdir.mkdir()
        (tdir / "test_core.py").write_text(
            "from aragora.core import something\n"
        )
        (tdir / "test_utils.py").write_text(
            "from aragora.utils import helper\n"
        )

        indexer = CodebaseIndexer(
            repo_path=tmp_path,
            source_dirs=["src"],
            test_dirs=["tests"],
        )
        stats = await indexer.index()

        assert stats.test_files_found == 2

    @pytest.mark.asyncio
    async def test_index_skips_private_modules(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "__init__.py").write_text('"""Init."""\n')
        (src / "_private.py").write_text('"""Private."""\n')
        (src / "public.py").write_text('"""Public."""\n')

        indexer = CodebaseIndexer(
            repo_path=tmp_path,
            source_dirs=["src"],
            test_dirs=[],
        )
        stats = await indexer.index()

        # __init__ and public only; _private is skipped
        assert stats.modules_indexed == 2
        paths = [m.path for m in indexer._modules]
        assert any("public.py" in p for p in paths)
        assert not any("_private.py" in p for p in paths)

    @pytest.mark.asyncio
    async def test_index_skips_missing_source_dir(self, tmp_path: Path):
        indexer = CodebaseIndexer(
            repo_path=tmp_path,
            source_dirs=["nonexistent"],
            test_dirs=[],
        )
        stats = await indexer.index()
        assert stats.modules_indexed == 0

    @pytest.mark.asyncio
    async def test_index_records_syntax_errors(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "good.py").write_text('"""Good."""\n')
        (src / "bad.py").write_text("def broken(:\n")

        indexer = CodebaseIndexer(
            repo_path=tmp_path,
            source_dirs=["src"],
            test_dirs=[],
        )
        stats = await indexer.index()

        # good.py indexed; bad.py returns None from _analyze_module (not an
        # exception that bubbles up because SyntaxError is caught internally)
        assert stats.modules_indexed == 1


# ---------------------------------------------------------------------------
# TestQuery
# ---------------------------------------------------------------------------


class TestQuery:
    """Test the query() search interface."""

    @pytest.fixture
    def populated_indexer(self) -> CodebaseIndexer:
        indexer = CodebaseIndexer()
        indexer._modules = [
            ModuleInfo(
                path="aragora/debate/consensus.py",
                docstring="Consensus detection for multi-agent debates.",
                classes=["ConsensusEngine"],
                functions=["detect_consensus"],
                imports_from=["aragora.debate.convergence"],
                line_count=300,
            ),
            ModuleInfo(
                path="aragora/billing/cost_tracker.py",
                docstring="Tracks API call costs per tenant.",
                classes=["CostTracker"],
                functions=["get_daily_cost"],
                imports_from=[],
                line_count=150,
            ),
            ModuleInfo(
                path="aragora/memory/coordinator.py",
                docstring="Coordinates memory writes across tiers.",
                classes=["MemoryCoordinator"],
                functions=["coordinate_write"],
                imports_from=["aragora.memory.continuum"],
                line_count=250,
            ),
        ]
        return indexer

    @pytest.mark.asyncio
    async def test_query_finds_by_keyword(self, populated_indexer: CodebaseIndexer):
        results = await populated_indexer.query("consensus")
        assert len(results) > 0
        assert results[0].path == "aragora/debate/consensus.py"

    @pytest.mark.asyncio
    async def test_query_ranks_path_matches_higher(self, populated_indexer: CodebaseIndexer):
        results = await populated_indexer.query("billing")
        assert len(results) > 0
        assert results[0].path == "aragora/billing/cost_tracker.py"

    @pytest.mark.asyncio
    async def test_query_returns_empty_for_no_match(self, populated_indexer: CodebaseIndexer):
        results = await populated_indexer.query("zzz_nonexistent_zzz")
        assert results == []

    @pytest.mark.asyncio
    async def test_query_respects_limit(self, populated_indexer: CodebaseIndexer):
        # "aragora" appears in all paths, so all 3 modules match
        results = await populated_indexer.query("aragora", limit=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_multi_word(self, populated_indexer: CodebaseIndexer):
        results = await populated_indexer.query("memory coordinator")
        assert len(results) > 0
        assert results[0].path == "aragora/memory/coordinator.py"


# ---------------------------------------------------------------------------
# TestTestMap
# ---------------------------------------------------------------------------


class TestTestMap:
    """Test get_tests_for_module() and the test→source mapping."""

    @pytest.mark.asyncio
    async def test_get_tests_for_module(self, tmp_path: Path):
        src = tmp_path / "aragora"
        src.mkdir()
        (src / "core.py").write_text('"""Core."""\nclass CoreClass:\n    pass\n')

        tdir = tmp_path / "tests"
        tdir.mkdir()
        (tdir / "test_core.py").write_text(
            "from aragora.core import CoreClass\n"
        )

        indexer = CodebaseIndexer(
            repo_path=tmp_path,
            source_dirs=["aragora"],
            test_dirs=["tests"],
        )
        await indexer.index()

        tests = indexer.get_tests_for_module("aragora/core.py")
        assert len(tests) == 1
        assert "tests/test_core.py" in tests[0]

    @pytest.mark.asyncio
    async def test_get_tests_returns_empty_for_untested(self, tmp_path: Path):
        src = tmp_path / "aragora"
        src.mkdir()
        (src / "orphan.py").write_text('"""Orphan."""\n')

        indexer = CodebaseIndexer(
            repo_path=tmp_path,
            source_dirs=["aragora"],
            test_dirs=["tests"],
        )
        await indexer.index()

        tests = indexer.get_tests_for_module("aragora/orphan.py")
        assert tests == []

    @pytest.mark.asyncio
    async def test_multiple_tests_for_one_module(self, tmp_path: Path):
        src = tmp_path / "aragora"
        src.mkdir()
        (src / "shared.py").write_text('"""Shared."""\n')

        tdir = tmp_path / "tests"
        tdir.mkdir()
        (tdir / "test_shared_a.py").write_text("from aragora.shared import x\n")
        (tdir / "test_shared_b.py").write_text("from aragora.shared import y\n")

        indexer = CodebaseIndexer(
            repo_path=tmp_path,
            source_dirs=["aragora"],
            test_dirs=["tests"],
        )
        await indexer.index()

        tests = indexer.get_tests_for_module("aragora/shared.py")
        assert len(tests) == 2


# ---------------------------------------------------------------------------
# TestPersistToKM
# ---------------------------------------------------------------------------


class TestPersistToKM:
    """Test Knowledge Mound persistence with mocked adapter."""

    @pytest.mark.asyncio
    async def test_persist_calls_adapter(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "mod.py").write_text('"""Mod."""\n')

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.NomicCycleAdapter"
        ) as MockAdapter:
            mock_instance = MagicMock()
            mock_instance.ingest_cycle_outcome = AsyncMock()
            MockAdapter.return_value = mock_instance

            indexer = CodebaseIndexer(
                repo_path=tmp_path,
                source_dirs=["src"],
                test_dirs=[],
            )
            await indexer.index()

            mock_instance.ingest_cycle_outcome.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_graceful_on_import_error(self, tmp_path: Path):
        """When KM adapter is unavailable, index still succeeds."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "mod.py").write_text('"""Mod."""\n')

        with patch.dict(
            "sys.modules",
            {"aragora.knowledge.mound.adapters.nomic_cycle_adapter": None},
        ):
            indexer = CodebaseIndexer(
                repo_path=tmp_path,
                source_dirs=["src"],
                test_dirs=[],
            )
            stats = await indexer.index()

            # Index still completed
            assert stats.modules_indexed == 1

    @pytest.mark.asyncio
    async def test_persist_graceful_on_runtime_error(self, tmp_path: Path):
        """When KM adapter raises RuntimeError, index still succeeds."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "mod.py").write_text('"""Mod."""\n')

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.NomicCycleAdapter"
        ) as MockAdapter:
            mock_instance = MagicMock()
            mock_instance.ingest_cycle_outcome = AsyncMock(
                side_effect=RuntimeError("KM down")
            )
            MockAdapter.return_value = mock_instance

            indexer = CodebaseIndexer(
                repo_path=tmp_path,
                source_dirs=["src"],
                test_dirs=[],
            )
            stats = await indexer.index()

            assert stats.modules_indexed == 1


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end test: index a small part of codebase, then query."""

    @pytest.mark.asyncio
    async def test_index_and_query_consensus(self, tmp_path: Path):
        """Build a mini codebase and query for 'consensus'."""
        src = tmp_path / "aragora" / "debate"
        src.mkdir(parents=True)

        (src / "__init__.py").write_text('"""Debate package."""\n')
        (src / "consensus.py").write_text(textwrap.dedent('''\
            """Consensus detection for multi-agent debates.

            Implements majority voting and convergence-based consensus.
            """

            from aragora.debate.convergence import is_converged

            class ConsensusEngine:
                """Detects when agents reach consensus."""

                def detect(self, votes):
                    pass

            class VoteAggregator:
                pass

            def calculate_majority(votes):
                return max(votes, key=votes.count)
        '''))
        (src / "convergence.py").write_text(textwrap.dedent('''\
            """Semantic convergence detection."""

            def is_converged(positions):
                return False
        '''))

        tdir = tmp_path / "tests"
        tdir.mkdir()
        (tdir / "test_consensus.py").write_text(
            "from aragora.debate.consensus import ConsensusEngine\n"
            "\ndef test_detect():\n    pass\n"
        )

        indexer = CodebaseIndexer(
            repo_path=tmp_path,
            source_dirs=["aragora"],
            test_dirs=["tests"],
        )
        stats = await indexer.index()

        assert stats.modules_indexed >= 3
        assert stats.test_files_found == 1

        # Query for consensus
        results = await indexer.query("consensus")
        assert len(results) > 0
        top = results[0]
        assert "consensus" in top.path.lower()
        assert "ConsensusEngine" in top.classes

        # Query for convergence
        results = await indexer.query("convergence")
        assert len(results) > 0
        assert any("convergence" in r.path for r in results)

        # Test map should link consensus.py to its test
        tests = indexer.get_tests_for_module("aragora/debate/consensus.py")
        assert len(tests) == 1
        assert "test_consensus" in tests[0]

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_index_real_nomic_directory(self):
        """Smoke test: index the real aragora/nomic/ directory (capped)."""
        repo = Path(__file__).resolve().parent.parent.parent
        nomic_dir = repo / "aragora" / "nomic"

        if not nomic_dir.exists():
            pytest.skip("aragora/nomic not found")

        indexer = CodebaseIndexer(
            repo_path=repo,
            source_dirs=["aragora/nomic"],
            test_dirs=[],
            max_modules=5,
        )
        stats = await indexer.index()

        assert stats.modules_indexed > 0
        assert stats.classes_found > 0
        assert stats.total_lines > 0

        # Should find something if we search for "outcome"
        results = await indexer.query("outcome tracker")
        assert len(results) > 0
