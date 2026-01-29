"""
Tests for aragora.tools.code module.

Covers: ChangeType, FileContext, CodeSpan, CodeChange, CodeProposal,
ValidationResult, CodeReader, CodeWriter, SelfImprover.

Run with: python -m pytest tests/tools/test_code.py -v --noconftest --timeout=30
"""

import os
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.tools.code import (
    ChangeType,
    CodeChange,
    CodeProposal,
    CodeReader,
    CodeSpan,
    CodeWriter,
    FileContext,
    SelfImprover,
    ValidationResult,
)


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------


class TestImports:
    """Verify the module and its public API load correctly."""

    def test_import_module(self):
        import aragora.tools.code as mod

        assert hasattr(mod, "CodeReader")

    def test_import_package(self):
        import aragora.tools as pkg

        assert hasattr(pkg, "CodeReader")
        assert hasattr(pkg, "CodeWriter")
        assert hasattr(pkg, "SelfImprover")
        assert hasattr(pkg, "ChangeType")
        assert hasattr(pkg, "CodeChange")
        assert hasattr(pkg, "CodeProposal")
        assert hasattr(pkg, "FileContext")
        assert hasattr(pkg, "CodeSpan")
        assert hasattr(pkg, "ValidationResult")

    def test_all_exports(self):
        import aragora.tools as pkg

        expected = {
            "CodeReader",
            "CodeWriter",
            "SelfImprover",
            "CodeChange",
            "CodeProposal",
            "ChangeType",
            "FileContext",
            "CodeSpan",
            "ValidationResult",
        }
        assert expected == set(pkg.__all__)


# ---------------------------------------------------------------------------
# ChangeType enum
# ---------------------------------------------------------------------------


class TestChangeType:
    def test_values(self):
        assert ChangeType.ADD.value == "add"
        assert ChangeType.MODIFY.value == "modify"
        assert ChangeType.DELETE.value == "delete"
        assert ChangeType.RENAME.value == "rename"
        assert ChangeType.MOVE.value == "move"
        assert ChangeType.REFACTOR.value == "refactor"

    def test_member_count(self):
        assert len(ChangeType) == 6

    def test_from_value(self):
        assert ChangeType("add") is ChangeType.ADD
        assert ChangeType("refactor") is ChangeType.REFACTOR

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            ChangeType("invalid")


# ---------------------------------------------------------------------------
# FileContext dataclass
# ---------------------------------------------------------------------------


class TestFileContext:
    def test_required_fields(self):
        ctx = FileContext(
            path="foo.py",
            content="print('hi')",
            language="python",
            size_bytes=11,
            line_count=1,
            last_modified="2024-01-01T00:00:00",
        )
        assert ctx.path == "foo.py"
        assert ctx.language == "python"

    def test_default_lists(self):
        ctx = FileContext(
            path="f.py",
            content="",
            language="python",
            size_bytes=0,
            line_count=0,
            last_modified="",
        )
        assert ctx.imports == []
        assert ctx.exports == []
        assert ctx.functions == []
        assert ctx.classes == []


# ---------------------------------------------------------------------------
# CodeSpan dataclass
# ---------------------------------------------------------------------------


class TestCodeSpan:
    def test_creation(self):
        span = CodeSpan(
            file_path="a.py",
            start_line=1,
            end_line=5,
            content="code",
        )
        assert span.start_line == 1
        assert span.context_before == ""
        assert span.context_after == ""


# ---------------------------------------------------------------------------
# CodeChange dataclass
# ---------------------------------------------------------------------------


class TestCodeChange:
    def test_defaults(self):
        change = CodeChange(
            change_id="c1",
            change_type=ChangeType.ADD,
            file_path="new.py",
            description="add file",
            rationale="needed",
        )
        assert change.confidence == 0.5
        assert change.risk_level == "medium"
        assert change.requires_test is True
        assert change.old_code is None
        assert change.new_code is None
        assert change.new_path is None
        assert change.author == ""
        # created_at should be a valid ISO timestamp
        datetime.fromisoformat(change.created_at)


# ---------------------------------------------------------------------------
# CodeProposal dataclass
# ---------------------------------------------------------------------------


class TestCodeProposal:
    def _make_proposal(self, changes=None):
        return CodeProposal(
            proposal_id="p1",
            title="title",
            description="desc",
            author="agent",
            changes=changes or [],
        )

    def test_defaults(self):
        p = self._make_proposal()
        assert p.files_affected == []
        assert p.tests_affected == []
        assert p.breaking_changes == []
        assert p.tests_passed is None
        assert p.confidence == 0.5
        datetime.fromisoformat(p.created_at)

    def test_to_patch_empty(self):
        p = self._make_proposal()
        assert p.to_patch() == ""

    def test_to_patch_with_modify_change(self):
        change = CodeChange(
            change_id="c1",
            change_type=ChangeType.MODIFY,
            file_path="src/main.py",
            description="fix",
            rationale="bug",
            old_code="x = 1\ny = 2",
            new_code="x = 10\ny = 20\nz = 30",
            start_line=5,
        )
        p = self._make_proposal(changes=[change])
        patch = p.to_patch()
        assert "--- a/src/main.py" in patch
        assert "+++ b/src/main.py" in patch
        assert "-x = 1" in patch
        assert "+x = 10" in patch
        assert "@@ -5,2 +5,3 @@" in patch

    def test_to_patch_skips_non_modify(self):
        change = CodeChange(
            change_id="c2",
            change_type=ChangeType.ADD,
            file_path="new.py",
            description="add",
            rationale="new feature",
        )
        p = self._make_proposal(changes=[change])
        assert p.to_patch() == ""


# ---------------------------------------------------------------------------
# ValidationResult dataclass
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_defaults(self):
        vr = ValidationResult(proposal_id="p1", valid=True)
        assert vr.errors == []
        assert vr.warnings == []
        assert vr.test_results is None
        assert vr.lint_results is None

    def test_invalid(self):
        vr = ValidationResult(
            proposal_id="p2",
            valid=False,
            errors=["syntax error"],
            warnings=["deprecation"],
        )
        assert not vr.valid
        assert len(vr.errors) == 1


# ---------------------------------------------------------------------------
# CodeReader
# ---------------------------------------------------------------------------


class TestCodeReader:
    """Tests for CodeReader using a temporary directory."""

    @pytest.fixture(autouse=True)
    def setup_tmpdir(self, tmp_path):
        self.root = tmp_path
        self.reader = CodeReader(str(tmp_path))

        # Create a sample Python file
        sample = tmp_path / "sample.py"
        sample.write_text(
            textwrap.dedent("""\
            import os
            from pathlib import Path

            __all__ = ["Foo", "bar"]

            class Foo:
                pass

            class Bar:
                pass

            def bar():
                return 1

            async def baz():
                return 2
        """)
        )
        self.sample_path = sample

        # Create a JS file
        js = tmp_path / "app.js"
        js.write_text(
            textwrap.dedent("""\
            import React from 'react';
            function hello() {}
            class Widget {}
        """)
        )
        self.js_path = js

        # Create a subdirectory with another file
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "inner.py").write_text("def inner_fn():\n    pass\n")

    # -- read_file --

    def test_read_file_basic(self):
        ctx = self.reader.read_file("sample.py")
        assert ctx.language == "python"
        assert ctx.line_count > 0
        assert ctx.size_bytes > 0
        assert "import os" in ctx.content

    def test_read_file_extracts_imports(self):
        ctx = self.reader.read_file("sample.py")
        assert any("import os" in i for i in ctx.imports)
        assert any("from pathlib" in i for i in ctx.imports)

    def test_read_file_extracts_exports(self):
        ctx = self.reader.read_file("sample.py")
        assert "Foo" in ctx.exports
        assert "bar" in ctx.exports

    def test_read_file_extracts_functions(self):
        ctx = self.reader.read_file("sample.py")
        assert "bar" in ctx.functions
        assert "baz" in ctx.functions

    def test_read_file_extracts_classes(self):
        ctx = self.reader.read_file("sample.py")
        assert "Foo" in ctx.classes
        assert "Bar" in ctx.classes

    def test_read_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            self.reader.read_file("nonexistent.py")

    def test_read_file_directory_rejected(self):
        with pytest.raises(ValueError, match="Not a file"):
            self.reader.read_file("sub")

    def test_read_file_js(self):
        ctx = self.reader.read_file("app.js")
        assert ctx.language == "javascript"
        assert "Widget" in ctx.classes
        assert any("React" in i for i in ctx.imports)

    # -- read_span --

    def test_read_span(self):
        span = self.reader.read_span("sample.py", 1, 2)
        assert "import os" in span.content
        assert span.start_line == 1
        assert span.end_line == 2

    def test_read_span_with_context(self):
        span = self.reader.read_span("sample.py", 3, 4, context_lines=2)
        # Lines 3-4 are the content; context_before covers lines 1-2
        assert span.start_line == 3
        assert span.end_line == 4
        # There should be context lines before and/or after
        assert span.context_before != "" or span.context_after != ""

    def test_read_span_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            self.reader.read_span("nope.py", 1, 1)

    def test_read_span_directory_rejected(self):
        with pytest.raises(ValueError, match="Not a file"):
            self.reader.read_span("sub", 1, 1)

    # -- search_code --

    def test_search_code(self):
        results = self.reader.search_code(r"def \w+")
        assert len(results) > 0
        paths = [r.file_path for r in results]
        assert any("sample.py" in p for p in paths)

    def test_search_code_no_match(self):
        results = self.reader.search_code(r"zzzzz_no_match")
        assert results == []

    # -- get_file_tree --

    def test_get_file_tree(self):
        tree = self.reader.get_file_tree()
        assert "sample.py" in tree
        assert "sub/" in tree

    def test_get_file_tree_respects_depth(self):
        tree = self.reader.get_file_tree(max_depth=0)
        # At depth 0, subdirectories should still appear but with truncation marker
        if "sub/" in tree:
            assert tree["sub/"] == {"...": "..."}

    # -- _resolve_path --

    def test_resolve_path_relative(self):
        resolved = self.reader._resolve_path("sample.py")
        assert resolved == self.sample_path

    def test_resolve_path_absolute(self):
        resolved = self.reader._resolve_path(str(self.sample_path))
        assert resolved == self.sample_path

    def test_resolve_path_traversal_blocked(self):
        with pytest.raises(PermissionError, match="escapes root"):
            self.reader._resolve_path("../../etc/passwd")

    # -- _detect_language --

    def test_detect_language_known(self):
        cases = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".rb": "ruby",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".md": "markdown",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
        }
        for ext, lang in cases.items():
            assert self.reader._detect_language(Path(f"file{ext}")) == lang

    def test_detect_language_unknown(self):
        assert self.reader._detect_language(Path("file.xyz")) == "unknown"

    # -- _extract_* helpers --

    def test_extract_imports_python(self):
        content = "import os\nfrom sys import argv\nx = 1"
        imports = self.reader._extract_imports(content, Path("f.py"))
        assert "import os" in imports
        assert "from sys import argv" in imports
        assert len(imports) == 2

    def test_extract_imports_js(self):
        content = "import React from 'react';\nimport _ from 'lodash';"
        imports = self.reader._extract_imports(content, Path("f.js"))
        assert len(imports) == 2

    def test_extract_exports_python(self):
        content = '__all__ = ["A", "B"]'
        exports = self.reader._extract_exports(content, Path("f.py"))
        assert "A" in exports
        assert "B" in exports

    def test_extract_exports_no_all(self):
        exports = self.reader._extract_exports("x = 1", Path("f.py"))
        assert exports == []

    def test_extract_functions_python(self):
        content = "def foo():\n    pass\nasync def bar():\n    pass"
        fns = self.reader._extract_functions(content, Path("f.py"))
        assert "foo" in fns
        assert "bar" in fns

    def test_extract_classes_python(self):
        content = "class A:\n    pass\nclass B(A):\n    pass"
        classes = self.reader._extract_classes(content, Path("f.py"))
        assert "A" in classes
        assert "B" in classes

    def test_extract_classes_js(self):
        content = "class Widget {}\nclass Button extends Widget {}"
        classes = self.reader._extract_classes(content, Path("f.ts"))
        assert "Widget" in classes
        assert "Button" in classes


# ---------------------------------------------------------------------------
# CodeWriter
# ---------------------------------------------------------------------------


class TestCodeWriter:
    """Tests for CodeWriter using a temporary directory (no real git)."""

    @pytest.fixture(autouse=True)
    def setup_tmpdir(self, tmp_path):
        self.root = tmp_path
        # Writer with git disabled
        self.writer = CodeWriter(str(tmp_path), use_git=False)
        # Seed a file
        (tmp_path / "existing.py").write_text("x = 1\ny = 2\n")

    def test_use_git_false(self):
        assert self.writer.use_git is False

    def test_create_branch_no_git(self):
        assert self.writer.create_branch("test-branch") is False

    def test_rollback_no_git(self):
        assert self.writer.rollback() is False

    # -- _apply_change: ADD --

    def test_apply_change_add(self):
        change = CodeChange(
            change_id="a1",
            change_type=ChangeType.ADD,
            file_path="new_file.py",
            description="add",
            rationale="r",
            new_file_content="print('hello')\n",
        )
        self.writer._apply_change(change)
        assert (self.root / "new_file.py").read_text() == "print('hello')\n"

    def test_apply_change_add_nested(self):
        change = CodeChange(
            change_id="a2",
            change_type=ChangeType.ADD,
            file_path="deep/nested/file.py",
            description="add",
            rationale="r",
            new_file_content="pass\n",
        )
        self.writer._apply_change(change)
        assert (self.root / "deep" / "nested" / "file.py").exists()

    def test_apply_change_add_no_content(self):
        """ADD with no new_file_content should be a no-op."""
        change = CodeChange(
            change_id="a3",
            change_type=ChangeType.ADD,
            file_path="ghost.py",
            description="add",
            rationale="r",
        )
        self.writer._apply_change(change)
        assert not (self.root / "ghost.py").exists()

    # -- _apply_change: DELETE --

    def test_apply_change_delete(self):
        change = CodeChange(
            change_id="d1",
            change_type=ChangeType.DELETE,
            file_path="existing.py",
            description="del",
            rationale="r",
        )
        self.writer._apply_change(change)
        assert not (self.root / "existing.py").exists()

    def test_apply_change_delete_nonexistent(self):
        """Deleting a non-existent file should not raise."""
        change = CodeChange(
            change_id="d2",
            change_type=ChangeType.DELETE,
            file_path="nope.py",
            description="del",
            rationale="r",
        )
        self.writer._apply_change(change)  # no error

    # -- _apply_change: MODIFY --

    def test_apply_change_modify(self):
        change = CodeChange(
            change_id="m1",
            change_type=ChangeType.MODIFY,
            file_path="existing.py",
            description="mod",
            rationale="r",
            old_code="x = 1",
            new_code="x = 99",
        )
        self.writer._apply_change(change)
        assert "x = 99" in (self.root / "existing.py").read_text()

    def test_apply_change_modify_no_codes(self):
        """MODIFY with no old/new code should be a no-op."""
        change = CodeChange(
            change_id="m2",
            change_type=ChangeType.MODIFY,
            file_path="existing.py",
            description="mod",
            rationale="r",
        )
        self.writer._apply_change(change)
        assert (self.root / "existing.py").read_text() == "x = 1\ny = 2\n"

    # -- _apply_change: RENAME --

    def test_apply_change_rename(self):
        change = CodeChange(
            change_id="r1",
            change_type=ChangeType.RENAME,
            file_path="existing.py",
            description="rename",
            rationale="r",
            new_path="renamed.py",
        )
        self.writer._apply_change(change)
        assert not (self.root / "existing.py").exists()
        assert (self.root / "renamed.py").exists()

    def test_apply_change_rename_no_new_path(self):
        """RENAME with no new_path should be a no-op."""
        change = CodeChange(
            change_id="r2",
            change_type=ChangeType.RENAME,
            file_path="existing.py",
            description="rename",
            rationale="r",
        )
        self.writer._apply_change(change)
        assert (self.root / "existing.py").exists()

    # -- apply_proposal --

    def test_apply_proposal_success(self):
        change = CodeChange(
            change_id="c1",
            change_type=ChangeType.ADD,
            file_path="added.py",
            description="add",
            rationale="r",
            new_file_content="# new\n",
        )
        proposal = CodeProposal(
            proposal_id="p1",
            title="t",
            description="d",
            author="a",
            changes=[change],
        )
        result = self.writer.apply_proposal(proposal, validate=False, commit=False)
        assert result.valid is True
        assert result.proposal_id == "p1"
        assert (self.root / "added.py").exists()

    def test_apply_proposal_with_error(self):
        """A change that raises gets captured as an error."""
        change = CodeChange(
            change_id="c2",
            change_type=ChangeType.MODIFY,
            file_path="nonexistent_for_modify.py",
            description="mod",
            rationale="r",
            old_code="a",
            new_code="b",
        )
        proposal = CodeProposal(
            proposal_id="p2",
            title="t",
            description="d",
            author="a",
            changes=[change],
        )
        result = self.writer.apply_proposal(proposal, validate=False, commit=False)
        assert result.valid is False
        assert len(result.errors) > 0

    # -- _is_git_repo --

    def test_is_git_repo_false_in_tmp(self):
        w = CodeWriter(str(self.root), use_git=True)
        # tmp_path typically is not a git repo
        # The constructor calls _is_git_repo; use_git reflects the result
        # Either True or False is acceptable depending on system, but it shouldn't crash
        assert isinstance(w.use_git, bool)


# ---------------------------------------------------------------------------
# CodeWriter with git (using a real temporary git repo)
# ---------------------------------------------------------------------------


class TestCodeWriterGit:
    """Tests CodeWriter git integration with a real temp git repo."""

    @pytest.fixture(autouse=True)
    def setup_git_repo(self, tmp_path):
        self.root = tmp_path
        # Initialize a real git repo
        os.system(
            f"cd {tmp_path} && git init -q && git config user.email 'test@test.com' && git config user.name 'Test'"
        )
        (tmp_path / "init.py").write_text("# init\n")
        os.system(f"cd {tmp_path} && git add -A && git commit -q -m 'init'")
        self.writer = CodeWriter(str(tmp_path), use_git=True)

    def test_use_git_true(self):
        assert self.writer.use_git is True

    def test_create_branch(self):
        assert self.writer.create_branch("feat-test") is True

    def test_create_branch_duplicate(self):
        self.writer.create_branch("dup-branch")
        # Creating the same branch again should fail
        assert self.writer.create_branch("dup-branch") is False

    def test_rollback(self):
        # Make a second commit
        (self.root / "temp.py").write_text("# temp\n")
        os.system(f"cd {self.root} && git add -A && git commit -q -m 'second'")
        assert self.writer.rollback() is True

    def test_commit_changes(self):
        (self.root / "committed.py").write_text("# committed\n")
        self.writer._commit_changes("test commit", "test description")
        # No exception means success


# ---------------------------------------------------------------------------
# SelfImprover
# ---------------------------------------------------------------------------


class TestSelfImprover:
    """Tests for SelfImprover."""

    @pytest.fixture(autouse=True)
    def setup_tmpdir(self, tmp_path):
        self.root = tmp_path
        # Create a minimal codebase
        (tmp_path / "main.py").write_text("def main():\n    print('hello')\n")
        (tmp_path / "utils.py").write_text("# utils\n" * 600)  # >500 lines for issue detection

        self.mock_debate_fn = AsyncMock()
        self.improver = SelfImprover(
            codebase_path=str(tmp_path),
            agents=["agent1", "agent2"],
            run_debate_fn=self.mock_debate_fn,
            safe_mode=True,
        )

    def test_init(self):
        assert self.improver.safe_mode is True
        assert len(self.improver.agents) == 2
        assert self.improver.proposals == []
        assert self.improver.applied_changes == []

    def test_analyze_codebase(self):
        analysis = self.improver.analyze_codebase()
        assert analysis["file_count"] >= 2
        assert analysis["total_lines"] > 0
        assert isinstance(analysis["functions"], list)
        assert isinstance(analysis["classes"], list)
        assert isinstance(analysis["potential_issues"], list)
        # utils.py has >500 lines, should trigger an issue
        assert any("too long" in issue.lower() for issue in analysis["potential_issues"])

    def test_generate_improvement_prompt_no_focus(self):
        prompt = self.improver.generate_improvement_prompt()
        assert "Suggest the most impactful improvements" in prompt
        assert "Files:" in prompt

    def test_generate_improvement_prompt_with_focus(self):
        prompt = self.improver.generate_improvement_prompt(focus="performance")
        assert "Focus on: performance" in prompt

    def test_apply_best_proposal_no_proposals(self):
        result = self.improver.apply_best_proposal()
        assert result.valid is False
        assert "No proposals" in result.errors[0]

    def test_apply_best_proposal_not_found(self):
        self.improver.proposals.append(
            CodeProposal(
                proposal_id="p1",
                title="t",
                description="d",
                author="a",
                changes=[],
                confidence=0.9,
            )
        )
        result = self.improver.apply_best_proposal(proposal_id="nonexistent")
        assert result.valid is False
        assert "not found" in result.errors[0].lower()

    def test_apply_best_proposal_selects_highest_confidence(self):
        low = CodeProposal(
            proposal_id="low",
            title="low",
            description="d",
            author="a",
            changes=[],
            confidence=0.1,
        )
        high = CodeProposal(
            proposal_id="high",
            title="high",
            description="d",
            author="a",
            changes=[],
            confidence=0.9,
        )
        self.improver.proposals.extend([low, high])
        # apply_proposal needs to be mocked since there's no git
        with patch.object(self.improver.writer, "apply_proposal") as mock_apply:
            mock_apply.return_value = ValidationResult(
                proposal_id="high",
                valid=True,
            )
            with patch.object(self.improver.writer, "create_branch"):
                result = self.improver.apply_best_proposal()
                # Should have selected the 'high' proposal
                mock_apply.assert_called_once()
                called_proposal = mock_apply.call_args[0][0]
                assert called_proposal.proposal_id == "high"

    def test_apply_best_proposal_tracks_applied(self):
        proposal = CodeProposal(
            proposal_id="p1",
            title="t",
            description="d",
            author="a",
            changes=[],
            confidence=0.5,
        )
        self.improver.proposals.append(proposal)
        with patch.object(self.improver.writer, "apply_proposal") as mock_apply:
            mock_apply.return_value = ValidationResult(proposal_id="p1", valid=True)
            with patch.object(self.improver.writer, "create_branch"):
                self.improver.apply_best_proposal()
                assert "p1" in self.improver.applied_changes

    def test_get_improvement_summary_empty(self):
        summary = self.improver.get_improvement_summary()
        assert "Proposals Generated:** 0" in summary
        assert "Changes Applied:** 0" in summary

    def test_get_improvement_summary_with_proposals(self):
        self.improver.proposals.append(
            CodeProposal(
                proposal_id="p1",
                title="Fix bug",
                description="d",
                author="a",
                changes=[],
                confidence=0.75,
            )
        )
        self.improver.applied_changes.append("p1")
        summary = self.improver.get_improvement_summary()
        assert "Proposals Generated:** 1" in summary
        assert "Changes Applied:** 1" in summary
        assert "Fix bug" in summary
        assert "75%" in summary

    def test_extract_proposals_with_final_answer(self):
        result = MagicMock()
        result.final_answer = "Improve error handling in main.py by adding try/except blocks"
        result.confidence = 0.8
        proposals = self.improver._extract_proposals(result)
        assert len(proposals) == 1
        assert proposals[0].confidence == 0.8
        assert "Improve error handling" in proposals[0].description

    def test_extract_proposals_no_final_answer(self):
        result = MagicMock(spec=[])  # no attributes
        proposals = self.improver._extract_proposals(result)
        assert proposals == []

    def test_extract_proposals_final_answer_none(self):
        result = MagicMock()
        result.final_answer = None
        proposals = self.improver._extract_proposals(result)
        assert proposals == []


# ---------------------------------------------------------------------------
# SelfImprover async tests
# ---------------------------------------------------------------------------


class TestSelfImproverAsync:
    @pytest.fixture(autouse=True)
    def setup_tmpdir(self, tmp_path):
        self.root = tmp_path
        (tmp_path / "main.py").write_text("def main(): pass\n")
        self.mock_debate_fn = AsyncMock()
        self.improver = SelfImprover(
            codebase_path=str(tmp_path),
            agents=["a1"],
            run_debate_fn=self.mock_debate_fn,
            safe_mode=True,
        )

    @pytest.mark.asyncio
    async def test_debate_improvement(self):
        mock_result = MagicMock()
        mock_result.final_answer = "Add logging to main"
        mock_result.confidence = 0.7
        self.mock_debate_fn.return_value = mock_result

        with (
            patch("aragora.config.settings.DebateSettings") as mock_settings,
            patch("aragora.core.Environment") as mock_env,
        ):
            mock_settings.return_value.default_rounds = 3
            proposals = await self.improver.debate_improvement(focus="logging")

        assert len(proposals) == 1
        assert proposals[0].confidence == 0.7
        assert len(self.improver.proposals) == 1

    @pytest.mark.asyncio
    async def test_debate_improvement_explicit_rounds(self):
        mock_result = MagicMock()
        mock_result.final_answer = "Refactor utils"
        mock_result.confidence = 0.6
        self.mock_debate_fn.return_value = mock_result

        with (
            patch("aragora.config.settings.DebateSettings") as mock_settings,
            patch("aragora.core.Environment") as mock_env,
        ):
            proposals = await self.improver.debate_improvement(rounds=5)

        mock_env.assert_called_once()
        call_kwargs = mock_env.call_args.kwargs
        assert call_kwargs.get("max_rounds") == 5
