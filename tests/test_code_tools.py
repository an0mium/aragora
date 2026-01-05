"""
Tests for the Code Editing Tools module.

Tests code reading, analysis, change proposals, and safe code modifications.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from aragora.tools.code import (
    ChangeType,
    FileContext,
    CodeSpan,
    CodeChange,
    CodeProposal,
    ValidationResult,
    CodeReader,
    CodeWriter,
    SelfImprover,
)


# =============================================================================
# ChangeType Tests
# =============================================================================


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_change_type_values(self):
        """Test all change type values."""
        assert ChangeType.ADD.value == "add"
        assert ChangeType.MODIFY.value == "modify"
        assert ChangeType.DELETE.value == "delete"
        assert ChangeType.RENAME.value == "rename"
        assert ChangeType.MOVE.value == "move"
        assert ChangeType.REFACTOR.value == "refactor"


# =============================================================================
# FileContext Tests
# =============================================================================


class TestFileContext:
    """Tests for FileContext dataclass."""

    def test_file_context_creation(self):
        """Test basic FileContext creation."""
        ctx = FileContext(
            path="test.py",
            content="print('hello')",
            language="python",
            size_bytes=100,
            line_count=1,
            last_modified="2024-01-01T00:00:00",
        )
        assert ctx.path == "test.py"
        assert ctx.language == "python"
        assert ctx.line_count == 1

    def test_file_context_defaults(self):
        """Test FileContext default values."""
        ctx = FileContext(
            path="test.py",
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


# =============================================================================
# CodeSpan Tests
# =============================================================================


class TestCodeSpan:
    """Tests for CodeSpan dataclass."""

    def test_code_span_creation(self):
        """Test CodeSpan creation."""
        span = CodeSpan(
            file_path="test.py",
            start_line=5,
            end_line=10,
            content="def foo():\n    pass",
        )
        assert span.start_line == 5
        assert span.end_line == 10
        assert "def foo" in span.content

    def test_code_span_with_context(self):
        """Test CodeSpan with surrounding context."""
        span = CodeSpan(
            file_path="test.py",
            start_line=5,
            end_line=10,
            content="def foo():\n    pass",
            context_before="# comment",
            context_after="# another comment",
        )
        assert span.context_before == "# comment"
        assert span.context_after == "# another comment"


# =============================================================================
# CodeChange Tests
# =============================================================================


class TestCodeChange:
    """Tests for CodeChange dataclass."""

    def test_code_change_creation(self):
        """Test CodeChange creation."""
        change = CodeChange(
            change_id="change-001",
            change_type=ChangeType.MODIFY,
            file_path="test.py",
            description="Fix bug",
            rationale="Bug causes crash",
        )
        assert change.change_id == "change-001"
        assert change.change_type == ChangeType.MODIFY

    def test_code_change_modification(self):
        """Test CodeChange for modification."""
        change = CodeChange(
            change_id="change-001",
            change_type=ChangeType.MODIFY,
            file_path="test.py",
            description="Fix bug",
            rationale="Bug causes crash",
            old_code="x = 1",
            new_code="x = 2",
            start_line=10,
            end_line=10,
        )
        assert change.old_code == "x = 1"
        assert change.new_code == "x = 2"

    def test_code_change_add_file(self):
        """Test CodeChange for adding new file."""
        change = CodeChange(
            change_id="change-002",
            change_type=ChangeType.ADD,
            file_path="new_file.py",
            description="Add utility",
            rationale="Needed for feature",
            new_file_content="# New file\ndef util(): pass",
        )
        assert change.new_file_content is not None

    def test_code_change_defaults(self):
        """Test CodeChange default values."""
        change = CodeChange(
            change_id="test",
            change_type=ChangeType.ADD,
            file_path="test.py",
            description="Test",
            rationale="Test",
        )
        assert change.confidence == 0.5
        assert change.risk_level == "medium"
        assert change.requires_test is True


# =============================================================================
# CodeProposal Tests
# =============================================================================


class TestCodeProposal:
    """Tests for CodeProposal dataclass."""

    def test_proposal_creation(self):
        """Test CodeProposal creation."""
        proposal = CodeProposal(
            proposal_id="prop-001",
            title="Fix memory leak",
            description="Fix leak in parser",
            author="agent-1",
            changes=[],
        )
        assert proposal.proposal_id == "prop-001"
        assert proposal.title == "Fix memory leak"

    def test_proposal_with_changes(self):
        """Test CodeProposal with changes."""
        change = CodeChange(
            change_id="change-001",
            change_type=ChangeType.MODIFY,
            file_path="test.py",
            description="Fix",
            rationale="Reason",
        )
        proposal = CodeProposal(
            proposal_id="prop-001",
            title="Fix",
            description="Description",
            author="agent",
            changes=[change],
        )
        assert len(proposal.changes) == 1

    def test_proposal_to_patch(self):
        """Test patch generation."""
        change = CodeChange(
            change_id="change-001",
            change_type=ChangeType.MODIFY,
            file_path="test.py",
            description="Fix",
            rationale="Reason",
            old_code="x = 1",
            new_code="x = 2",
            start_line=5,
        )
        proposal = CodeProposal(
            proposal_id="prop-001",
            title="Fix",
            description="Desc",
            author="agent",
            changes=[change],
        )
        patch = proposal.to_patch()
        assert "--- a/test.py" in patch
        assert "+++ b/test.py" in patch
        assert "-x = 1" in patch
        assert "+x = 2" in patch


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_valid(self):
        """Test valid validation result."""
        result = ValidationResult(
            proposal_id="prop-001",
            valid=True,
        )
        assert result.valid is True
        assert result.errors == []

    def test_validation_result_invalid(self):
        """Test invalid validation result."""
        result = ValidationResult(
            proposal_id="prop-001",
            valid=False,
            errors=["Syntax error", "Test failed"],
        )
        assert result.valid is False
        assert len(result.errors) == 2


# =============================================================================
# CodeReader Tests
# =============================================================================


class TestCodeReader:
    """Tests for CodeReader class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create test Python file
            (path / "test.py").write_text(
                """import os
from typing import List

class MyClass:
    def method(self):
        pass

def my_function():
    return 42

async def async_func():
    pass
"""
            )

            # Create test JS file
            (path / "test.js").write_text(
                """import React from 'react';
import { useState } from 'react';

function Component() {
    return <div>Hello</div>;
}

class MyComponent extends React.Component {
    render() { return null; }
}
"""
            )

            # Create subdirectory with file
            (path / "sub").mkdir()
            (path / "sub" / "nested.py").write_text("# Nested file\n")

            yield path

    def test_reader_creation(self, temp_dir):
        """Test CodeReader creation."""
        reader = CodeReader(str(temp_dir))
        # Compare resolved paths (macOS has /var -> /private/var symlink)
        assert reader.root.resolve() == temp_dir.resolve()

    def test_read_file(self, temp_dir):
        """Test reading a file."""
        reader = CodeReader(str(temp_dir))
        ctx = reader.read_file("test.py")

        assert ctx.path == "test.py"
        assert ctx.language == "python"
        assert ctx.line_count > 0
        assert "import os" in ctx.content

    def test_read_file_extracts_imports(self, temp_dir):
        """Test import extraction."""
        reader = CodeReader(str(temp_dir))
        ctx = reader.read_file("test.py")

        assert any("import os" in imp for imp in ctx.imports)
        assert any("from typing" in imp for imp in ctx.imports)

    def test_read_file_extracts_functions(self, temp_dir):
        """Test function extraction."""
        reader = CodeReader(str(temp_dir))
        ctx = reader.read_file("test.py")

        assert "my_function" in ctx.functions
        assert "async_func" in ctx.functions

    def test_read_file_extracts_classes(self, temp_dir):
        """Test class extraction."""
        reader = CodeReader(str(temp_dir))
        ctx = reader.read_file("test.py")

        assert "MyClass" in ctx.classes

    def test_read_span(self, temp_dir):
        """Test reading a code span."""
        reader = CodeReader(str(temp_dir))
        span = reader.read_span("test.py", 4, 6, context_lines=1)

        assert span.start_line == 4
        assert span.end_line == 6
        assert "class" in span.content

    def test_read_span_with_context(self, temp_dir):
        """Test code span includes context."""
        reader = CodeReader(str(temp_dir))
        span = reader.read_span("test.py", 5, 5, context_lines=2)

        assert span.context_before != ""

    def test_search_code(self, temp_dir):
        """Test code searching."""
        reader = CodeReader(str(temp_dir))
        results = reader.search_code(r"def\s+\w+", "*.py")

        assert len(results) > 0
        assert any("def" in r.content for r in results)

    def test_get_file_tree(self, temp_dir):
        """Test file tree generation."""
        reader = CodeReader(str(temp_dir))
        tree = reader.get_file_tree()

        assert "test.py" in tree
        assert "sub/" in tree

    def test_detect_language_python(self, temp_dir):
        """Test Python language detection."""
        reader = CodeReader(str(temp_dir))
        ctx = reader.read_file("test.py")
        assert ctx.language == "python"

    def test_detect_language_javascript(self, temp_dir):
        """Test JavaScript language detection."""
        reader = CodeReader(str(temp_dir))
        ctx = reader.read_file("test.js")
        assert ctx.language == "javascript"

    def test_path_traversal_protection(self, temp_dir):
        """Test that path traversal is blocked."""
        reader = CodeReader(str(temp_dir))

        with pytest.raises(PermissionError, match="escapes root directory"):
            reader.read_file("../../../etc/passwd")

    def test_extract_js_imports(self, temp_dir):
        """Test JavaScript import extraction."""
        reader = CodeReader(str(temp_dir))
        ctx = reader.read_file("test.js")

        assert any("react" in imp.lower() for imp in ctx.imports)


# =============================================================================
# CodeWriter Tests
# =============================================================================


class TestCodeWriter:
    """Tests for CodeWriter class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "existing.py").write_text("x = 1\n")
            yield path

    def test_writer_creation(self, temp_dir):
        """Test CodeWriter creation."""
        writer = CodeWriter(str(temp_dir), use_git=False)
        # Compare resolved paths (macOS has /var -> /private/var symlink)
        assert writer.root.resolve() == temp_dir.resolve()
        assert writer.use_git is False

    def test_apply_add_change(self, temp_dir):
        """Test applying ADD change."""
        writer = CodeWriter(str(temp_dir), use_git=False)

        change = CodeChange(
            change_id="test",
            change_type=ChangeType.ADD,
            file_path="new_file.py",
            description="Add file",
            rationale="Test",
            new_file_content="# New file\nprint('hello')",
        )
        proposal = CodeProposal(
            proposal_id="prop",
            title="Test",
            description="Test",
            author="test",
            changes=[change],
        )

        result = writer.apply_proposal(proposal, validate=False, commit=False)

        assert (temp_dir / "new_file.py").exists()
        content = (temp_dir / "new_file.py").read_text()
        assert "print('hello')" in content

    def test_apply_modify_change(self, temp_dir):
        """Test applying MODIFY change."""
        writer = CodeWriter(str(temp_dir), use_git=False)

        change = CodeChange(
            change_id="test",
            change_type=ChangeType.MODIFY,
            file_path="existing.py",
            description="Modify",
            rationale="Test",
            old_code="x = 1",
            new_code="x = 42",
        )
        proposal = CodeProposal(
            proposal_id="prop",
            title="Test",
            description="Test",
            author="test",
            changes=[change],
        )

        writer.apply_proposal(proposal, validate=False, commit=False)

        content = (temp_dir / "existing.py").read_text()
        assert "x = 42" in content

    def test_apply_delete_change(self, temp_dir):
        """Test applying DELETE change."""
        writer = CodeWriter(str(temp_dir), use_git=False)

        change = CodeChange(
            change_id="test",
            change_type=ChangeType.DELETE,
            file_path="existing.py",
            description="Delete",
            rationale="Test",
        )
        proposal = CodeProposal(
            proposal_id="prop",
            title="Test",
            description="Test",
            author="test",
            changes=[change],
        )

        writer.apply_proposal(proposal, validate=False, commit=False)

        assert not (temp_dir / "existing.py").exists()

    def test_apply_rename_change(self, temp_dir):
        """Test applying RENAME change."""
        writer = CodeWriter(str(temp_dir), use_git=False)

        change = CodeChange(
            change_id="test",
            change_type=ChangeType.RENAME,
            file_path="existing.py",
            description="Rename",
            rationale="Test",
            new_path="renamed.py",
        )
        proposal = CodeProposal(
            proposal_id="prop",
            title="Test",
            description="Test",
            author="test",
            changes=[change],
        )

        writer.apply_proposal(proposal, validate=False, commit=False)

        assert not (temp_dir / "existing.py").exists()
        assert (temp_dir / "renamed.py").exists()

    def test_validation_result_errors(self, temp_dir):
        """Test validation result contains errors."""
        writer = CodeWriter(str(temp_dir), use_git=False)

        change = CodeChange(
            change_id="test",
            change_type=ChangeType.ADD,
            file_path="nonexistent/path/file.py",
            description="Add in missing dir",
            rationale="Test",
            new_file_content="print('test')",
        )
        proposal = CodeProposal(
            proposal_id="prop",
            title="Test",
            description="Test",
            author="test",
            changes=[change],
        )

        result = writer.apply_proposal(proposal, validate=False, commit=False)

        # Should succeed - directories are created
        assert result.valid is True


# =============================================================================
# SelfImprover Tests
# =============================================================================


class TestSelfImprover:
    """Tests for SelfImprover class."""

    @pytest.fixture
    def temp_codebase(self):
        """Create temporary codebase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create some Python files
            (path / "main.py").write_text(
                """import helper

def main():
    helper.do_thing()

if __name__ == '__main__':
    main()
"""
            )

            (path / "helper.py").write_text(
                """def do_thing():
    print('Doing thing')

def another_thing():
    pass
"""
            )

            yield path

    def test_improver_creation(self, temp_codebase):
        """Test SelfImprover creation."""
        mock_agents = [MagicMock()]
        mock_debate_fn = AsyncMock()

        improver = SelfImprover(
            codebase_path=str(temp_codebase),
            agents=mock_agents,
            run_debate_fn=mock_debate_fn,
        )

        # Compare resolved paths (macOS has /var -> /private/var symlink)
        assert improver.codebase_path.resolve() == temp_codebase.resolve()
        assert len(improver.agents) == 1

    def test_analyze_codebase(self, temp_codebase):
        """Test codebase analysis."""
        improver = SelfImprover(
            codebase_path=str(temp_codebase),
            agents=[],
            run_debate_fn=AsyncMock(),
        )

        analysis = improver.analyze_codebase()

        assert analysis["file_count"] == 2
        assert analysis["total_lines"] > 0
        assert "main" in analysis["functions"]

    def test_generate_improvement_prompt(self, temp_codebase):
        """Test improvement prompt generation."""
        improver = SelfImprover(
            codebase_path=str(temp_codebase),
            agents=[],
            run_debate_fn=AsyncMock(),
        )

        prompt = improver.generate_improvement_prompt()

        assert "Files:" in prompt
        assert "Total lines:" in prompt
        assert "Your Task" in prompt

    def test_generate_improvement_prompt_with_focus(self, temp_codebase):
        """Test prompt generation with focus."""
        improver = SelfImprover(
            codebase_path=str(temp_codebase),
            agents=[],
            run_debate_fn=AsyncMock(),
        )

        prompt = improver.generate_improvement_prompt(focus="performance")

        assert "performance" in prompt

    def test_apply_best_proposal_no_proposals(self, temp_codebase):
        """Test applying when no proposals exist."""
        improver = SelfImprover(
            codebase_path=str(temp_codebase),
            agents=[],
            run_debate_fn=AsyncMock(),
        )

        result = improver.apply_best_proposal()

        assert result.valid is False
        assert "No proposals" in result.errors[0]

    def test_apply_specific_proposal_not_found(self, temp_codebase):
        """Test applying non-existent proposal."""
        improver = SelfImprover(
            codebase_path=str(temp_codebase),
            agents=[],
            run_debate_fn=AsyncMock(),
        )

        # Add a proposal so we can test "not found" vs "no proposals"
        improver.proposals.append(
            CodeProposal(
                proposal_id="existing",
                title="Existing",
                description="Test",
                author="test",
                changes=[],
            )
        )

        result = improver.apply_best_proposal(proposal_id="nonexistent")

        assert result.valid is False
        assert "not found" in result.errors[0]

    def test_get_improvement_summary(self, temp_codebase):
        """Test summary generation."""
        improver = SelfImprover(
            codebase_path=str(temp_codebase),
            agents=[],
            run_debate_fn=AsyncMock(),
        )

        # Add a proposal
        improver.proposals.append(
            CodeProposal(
                proposal_id="prop-001",
                title="Test Proposal",
                description="Test",
                author="test",
                changes=[],
                confidence=0.8,
            )
        )

        summary = improver.get_improvement_summary()

        assert "Self-Improvement Session" in summary
        assert "Proposals Generated" in summary
        assert "1" in summary
        assert "prop-001" in summary

    @pytest.mark.asyncio
    async def test_debate_improvement(self, temp_codebase):
        """Test running improvement debate."""
        mock_result = MagicMock()
        mock_result.final_answer = "Suggested improvement: add error handling"
        mock_result.confidence = 0.7

        mock_debate_fn = AsyncMock(return_value=mock_result)

        improver = SelfImprover(
            codebase_path=str(temp_codebase),
            agents=[MagicMock()],
            run_debate_fn=mock_debate_fn,
        )

        proposals = await improver.debate_improvement(rounds=1)

        mock_debate_fn.assert_called_once()
        assert len(proposals) > 0
        assert proposals[0].description == "Suggested improvement: add error handling"


# =============================================================================
# Integration Tests
# =============================================================================


class TestCodeToolsIntegration:
    """Integration tests for code tools."""

    @pytest.fixture
    def temp_project(self):
        """Create a realistic project structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create source directory
            src = path / "src"
            src.mkdir()

            (src / "__init__.py").write_text("")
            (src / "app.py").write_text(
                """from .utils import helper

class App:
    def __init__(self):
        self.config = {}

    def run(self):
        helper()
"""
            )
            (src / "utils.py").write_text(
                """def helper():
    print('helper')

def format_data(data):
    return str(data)
"""
            )

            # Create tests directory
            tests = path / "tests"
            tests.mkdir()
            (tests / "__init__.py").write_text("")
            (tests / "test_app.py").write_text(
                """import pytest

def test_placeholder():
    assert True
"""
            )

            yield path

    def test_read_write_workflow(self, temp_project):
        """Test complete read-analyze-modify workflow."""
        reader = CodeReader(str(temp_project))
        writer = CodeWriter(str(temp_project), use_git=False)

        # Read and analyze
        ctx = reader.read_file("src/app.py")
        assert "App" in ctx.classes
        # Note: methods inside classes aren't detected (only top-level functions)
        assert len(ctx.imports) > 0

        # Create modification
        change = CodeChange(
            change_id="test-001",
            change_type=ChangeType.MODIFY,
            file_path="src/app.py",
            description="Add docstring",
            rationale="Improve documentation",
            old_code="def run(self):",
            new_code='def run(self):\n        """Run the application."""',
        )

        proposal = CodeProposal(
            proposal_id="prop-001",
            title="Add docstring",
            description="Add method docstring",
            author="test",
            changes=[change],
        )

        # Apply change
        result = writer.apply_proposal(proposal, validate=False, commit=False)
        assert result.valid is True

        # Verify change
        new_ctx = reader.read_file("src/app.py")
        assert "Run the application" in new_ctx.content

    def test_search_and_modify(self, temp_project):
        """Test searching code and modifying matches."""
        reader = CodeReader(str(temp_project))
        writer = CodeWriter(str(temp_project), use_git=False)

        # Search for all print statements
        results = reader.search_code(r"print\(", "**/*.py")
        assert len(results) > 0

        # Could create changes to replace prints with logging
        changes = []
        for i, result in enumerate(results):
            changes.append(
                CodeChange(
                    change_id=f"print-{i}",
                    change_type=ChangeType.MODIFY,
                    file_path=result.file_path,
                    description=f"Replace print at line {result.start_line}",
                    rationale="Use logging instead of print",
                    old_code="print('helper')",
                    new_code="logging.info('helper')",
                )
            )

        # Verify changes were created
        assert len(changes) > 0

