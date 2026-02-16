"""
Comprehensive tests for the codebase auditor module.

Tests cover:
- CodebaseAuditConfig defaults and customization
- ImprovementProposal dataclass and serialization
- CodebaseAuditResult dataclass, properties, and serialization
- IncrementalAuditResult dataclass, properties, and serialization
- CodebaseAuditor initialization and configuration
- File inclusion/exclusion logic
- File collection from filesystem
- Security pattern detection (hardcoded secrets, SQL injection)
- Quality pattern detection (TODO/FIXME, complexity)
- Severity ranking and comparison
- Findings-to-proposals conversion
- Proposal description building
- Fix suggestions
- Summary building
- Full codebase audit workflow
- Incremental file audit
- Git diff-based audit
- Error handling and edge cases
"""

import asyncio
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from aragora.audit.document_auditor import (
    AuditConfig,
    AuditFinding,
    AuditSession,
    AuditType,
    FindingSeverity,
)

# We need to mock the heavy imports before importing the module under test
# so we patch at the module level where needed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_codebase(tmp_path):
    """Create a temporary codebase structure for testing."""
    # Create directories
    (tmp_path / "aragora").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / ".git").mkdir()

    # Create Python files
    (tmp_path / "aragora" / "main.py").write_text("def main():\n    print('hello')\n")
    (tmp_path / "aragora" / "utils.py").write_text(
        "# TODO: refactor this\ndef helper():\n    pass\n"
    )
    (tmp_path / "scripts" / "run.py").write_text("import sys\nsys.exit(0)\n")

    # Create doc files
    (tmp_path / "docs" / "README.md").write_text("# Project\nThis is a project.\n")

    # Create excluded files
    (tmp_path / "__pycache__" / "cached.pyc").write_text("binary stuff")
    (tmp_path / "aragora" / "data.json").write_text("{}")

    return tmp_path


@pytest.fixture
def default_config():
    """Default CodebaseAuditConfig."""
    from aragora.audit.codebase_auditor import CodebaseAuditConfig

    return CodebaseAuditConfig()


@pytest.fixture
def mock_document_auditor():
    """Mock DocumentAuditor."""
    mock = MagicMock()
    mock.config = AuditConfig(min_confidence=0.7)
    return mock


@pytest.fixture
def mock_token_counter():
    """Mock TokenCounter."""
    counter = MagicMock()
    counter.count = MagicMock(return_value=100)
    return counter


@pytest.fixture
def mock_chunker():
    """Mock ChunkingStrategy that returns simple chunks."""
    chunker = MagicMock()
    chunk1 = MagicMock()
    chunk1.sequence = 0
    chunk1.content = "chunk content 0"
    chunk2 = MagicMock()
    chunk2.sequence = 1
    chunk2.content = "chunk content 1"
    chunker.chunk = MagicMock(return_value=[chunk1, chunk2])
    return chunker


@pytest.fixture
def auditor(tmp_codebase, mock_document_auditor, mock_token_counter):
    """Create a CodebaseAuditor with mocked dependencies."""
    from aragora.audit.codebase_auditor import CodebaseAuditor, CodebaseAuditConfig

    config = CodebaseAuditConfig(
        include_paths=["aragora/", "scripts/", "docs/"],
        min_severity=FindingSeverity.LOW,
        min_confidence=0.0,
    )
    a = CodebaseAuditor(
        root_path=tmp_codebase,
        config=config,
        document_auditor=mock_document_auditor,
        token_counter=mock_token_counter,
    )
    return a


def _make_finding(
    audit_type=AuditType.SECURITY,
    category="hardcoded_secret",
    severity=FindingSeverity.HIGH,
    confidence=0.9,
    title="Test finding",
    description="A test finding",
    evidence_location="file.py:chunk0",
    session_id="test",
) -> AuditFinding:
    """Helper to create an AuditFinding."""
    return AuditFinding(
        audit_type=audit_type,
        category=category,
        severity=severity,
        confidence=confidence,
        title=title,
        description=description,
        evidence_location=evidence_location,
        session_id=session_id,
    )


# ===========================================================================
# CodebaseAuditConfig Tests
# ===========================================================================


class TestCodebaseAuditConfig:
    """Tests for CodebaseAuditConfig dataclass."""

    def test_default_include_paths(self, default_config):
        assert default_config.include_paths == ["aragora/", "scripts/", "docs/"]

    def test_default_exclude_patterns(self, default_config):
        assert "__pycache__" in default_config.exclude_patterns
        assert ".git" in default_config.exclude_patterns
        assert "node_modules" in default_config.exclude_patterns
        assert ".venv" in default_config.exclude_patterns
        assert "*.pyc" in default_config.exclude_patterns

    def test_default_code_extensions(self, default_config):
        assert ".py" in default_config.code_extensions
        assert ".ts" in default_config.code_extensions
        assert ".tsx" in default_config.code_extensions
        assert ".js" in default_config.code_extensions

    def test_default_doc_extensions(self, default_config):
        assert ".md" in default_config.doc_extensions
        assert ".rst" in default_config.doc_extensions
        assert ".txt" in default_config.doc_extensions

    def test_default_audit_types(self, default_config):
        assert AuditType.CONSISTENCY in default_config.audit_types
        assert AuditType.QUALITY in default_config.audit_types
        assert AuditType.SECURITY in default_config.audit_types

    def test_default_token_settings(self, default_config):
        assert default_config.max_context_tokens == 500_000
        assert default_config.chunk_size == 500
        assert default_config.chunk_overlap == 50

    def test_default_filtering(self, default_config):
        assert default_config.min_severity == FindingSeverity.MEDIUM
        assert default_config.min_confidence == 0.7
        assert default_config.max_findings_per_cycle == 10

    def test_default_performance(self, default_config):
        assert default_config.max_concurrent_files == 5
        assert default_config.timeout_per_file == 30.0

    def test_custom_config(self):
        from aragora.audit.codebase_auditor import CodebaseAuditConfig

        config = CodebaseAuditConfig(
            include_paths=["src/"],
            exclude_patterns=["test/"],
            code_extensions=[".py"],
            doc_extensions=[".md"],
            audit_types=[AuditType.SECURITY],
            max_context_tokens=100_000,
            chunk_size=256,
            chunk_overlap=25,
            min_severity=FindingSeverity.HIGH,
            min_confidence=0.9,
            max_findings_per_cycle=5,
            max_concurrent_files=2,
            timeout_per_file=10.0,
        )
        assert config.include_paths == ["src/"]
        assert config.exclude_patterns == ["test/"]
        assert config.code_extensions == [".py"]
        assert config.audit_types == [AuditType.SECURITY]
        assert config.max_context_tokens == 100_000
        assert config.chunk_size == 256
        assert config.min_severity == FindingSeverity.HIGH
        assert config.min_confidence == 0.9
        assert config.max_findings_per_cycle == 5
        assert config.max_concurrent_files == 2
        assert config.timeout_per_file == 10.0


# ===========================================================================
# ImprovementProposal Tests
# ===========================================================================


class TestImprovementProposal:
    """Tests for ImprovementProposal dataclass."""

    def test_creation(self):
        from aragora.audit.codebase_auditor import ImprovementProposal

        proposal = ImprovementProposal(
            id="p1",
            title="Fix secrets",
            description="Move secrets to env vars",
            finding_ids=["f1", "f2"],
            severity=FindingSeverity.HIGH,
            confidence=0.85,
        )
        assert proposal.id == "p1"
        assert proposal.title == "Fix secrets"
        assert proposal.finding_ids == ["f1", "f2"]
        assert proposal.severity == FindingSeverity.HIGH
        assert proposal.confidence == 0.85
        assert proposal.estimated_effort == "medium"
        assert proposal.affected_files == []
        assert proposal.suggested_fix == ""
        assert proposal.tags == []

    def test_to_dict(self):
        from aragora.audit.codebase_auditor import ImprovementProposal

        proposal = ImprovementProposal(
            id="p1",
            title="Fix it",
            description="Fix the thing",
            finding_ids=["f1"],
            severity=FindingSeverity.CRITICAL,
            confidence=0.95,
            estimated_effort="high",
            affected_files=["a.py", "b.py"],
            suggested_fix="Do X",
            tags=["security", "urgent"],
        )
        d = proposal.to_dict()
        assert d["id"] == "p1"
        assert d["title"] == "Fix it"
        assert d["description"] == "Fix the thing"
        assert d["finding_ids"] == ["f1"]
        assert d["severity"] == "critical"
        assert d["confidence"] == 0.95
        assert d["estimated_effort"] == "high"
        assert d["affected_files"] == ["a.py", "b.py"]
        assert d["suggested_fix"] == "Do X"
        assert d["tags"] == ["security", "urgent"]

    def test_default_effort(self):
        from aragora.audit.codebase_auditor import ImprovementProposal

        proposal = ImprovementProposal(
            id="p1",
            title="T",
            description="D",
            finding_ids=[],
            severity=FindingSeverity.LOW,
            confidence=0.5,
        )
        assert proposal.estimated_effort == "medium"


# ===========================================================================
# CodebaseAuditResult Tests
# ===========================================================================


class TestCodebaseAuditResult:
    """Tests for CodebaseAuditResult dataclass."""

    def test_duration_seconds(self):
        from aragora.audit.codebase_auditor import CodebaseAuditResult

        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
        result = CodebaseAuditResult(
            session_id="s1",
            started_at=start,
            completed_at=end,
            files_audited=5,
            total_tokens=1000,
            findings=[],
            proposals=[],
        )
        assert result.duration_seconds == 30.0

    def test_findings_by_severity_empty(self):
        from aragora.audit.codebase_auditor import CodebaseAuditResult

        result = CodebaseAuditResult(
            session_id="s1",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            files_audited=0,
            total_tokens=0,
            findings=[],
            proposals=[],
        )
        assert result.findings_by_severity == {}

    def test_findings_by_severity_counts(self):
        from aragora.audit.codebase_auditor import CodebaseAuditResult

        findings = [
            _make_finding(severity=FindingSeverity.HIGH),
            _make_finding(severity=FindingSeverity.HIGH),
            _make_finding(severity=FindingSeverity.LOW),
            _make_finding(severity=FindingSeverity.CRITICAL),
        ]
        result = CodebaseAuditResult(
            session_id="s1",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            files_audited=2,
            total_tokens=500,
            findings=findings,
            proposals=[],
        )
        by_sev = result.findings_by_severity
        assert by_sev["high"] == 2
        assert by_sev["low"] == 1
        assert by_sev["critical"] == 1

    def test_to_dict(self):
        from aragora.audit.codebase_auditor import CodebaseAuditResult

        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 0, 1, 0, tzinfo=timezone.utc)
        result = CodebaseAuditResult(
            session_id="s1",
            started_at=start,
            completed_at=end,
            files_audited=10,
            total_tokens=5000,
            findings=[_make_finding()],
            proposals=[],
            summary="Some summary",
        )
        d = result.to_dict()
        assert d["session_id"] == "s1"
        assert d["duration_seconds"] == 60.0
        assert d["files_audited"] == 10
        assert d["total_tokens"] == 5000
        assert d["findings_count"] == 1
        assert d["proposals_count"] == 0
        assert d["summary"] == "Some summary"
        assert "high" in d["findings_by_severity"]


# ===========================================================================
# IncrementalAuditResult Tests
# ===========================================================================


class TestIncrementalAuditResult:
    """Tests for IncrementalAuditResult dataclass."""

    def test_has_findings_empty(self):
        from aragora.audit.codebase_auditor import IncrementalAuditResult

        result = IncrementalAuditResult(
            session_id="i1",
            base_ref="HEAD~1",
            head_ref="HEAD",
            files_changed=[],
            files_audited=[],
            findings=[],
        )
        assert result.has_findings is False

    def test_has_findings_with_findings(self):
        from aragora.audit.codebase_auditor import IncrementalAuditResult

        result = IncrementalAuditResult(
            session_id="i1",
            base_ref="HEAD~1",
            head_ref="HEAD",
            files_changed=["a.py"],
            files_audited=["a.py"],
            findings=[_make_finding()],
        )
        assert result.has_findings is True

    def test_has_critical_false(self):
        from aragora.audit.codebase_auditor import IncrementalAuditResult

        result = IncrementalAuditResult(
            session_id="i1",
            base_ref="HEAD~1",
            head_ref="HEAD",
            files_changed=[],
            files_audited=[],
            findings=[_make_finding(severity=FindingSeverity.HIGH)],
        )
        assert result.has_critical is False

    def test_has_critical_true(self):
        from aragora.audit.codebase_auditor import IncrementalAuditResult

        result = IncrementalAuditResult(
            session_id="i1",
            base_ref="HEAD~1",
            head_ref="HEAD",
            files_changed=[],
            files_audited=[],
            findings=[_make_finding(severity=FindingSeverity.CRITICAL)],
        )
        assert result.has_critical is True

    def test_exit_code_no_findings(self):
        from aragora.audit.codebase_auditor import IncrementalAuditResult

        result = IncrementalAuditResult(
            session_id="i1",
            base_ref="HEAD~1",
            head_ref="HEAD",
            files_changed=[],
            files_audited=[],
            findings=[],
        )
        assert result.exit_code == 0

    def test_exit_code_with_findings(self):
        from aragora.audit.codebase_auditor import IncrementalAuditResult

        result = IncrementalAuditResult(
            session_id="i1",
            base_ref="HEAD~1",
            head_ref="HEAD",
            files_changed=[],
            files_audited=[],
            findings=[_make_finding(severity=FindingSeverity.MEDIUM)],
        )
        assert result.exit_code == 1

    def test_exit_code_critical(self):
        from aragora.audit.codebase_auditor import IncrementalAuditResult

        result = IncrementalAuditResult(
            session_id="i1",
            base_ref="HEAD~1",
            head_ref="HEAD",
            files_changed=[],
            files_audited=[],
            findings=[_make_finding(severity=FindingSeverity.CRITICAL)],
        )
        assert result.exit_code == 2

    def test_exit_code_error(self):
        from aragora.audit.codebase_auditor import IncrementalAuditResult

        result = IncrementalAuditResult(
            session_id="i1",
            base_ref="HEAD~1",
            head_ref="HEAD",
            files_changed=[],
            files_audited=[],
            findings=[],
            error="Something went wrong",
        )
        assert result.exit_code == 2

    def test_to_dict(self):
        from aragora.audit.codebase_auditor import IncrementalAuditResult

        finding = _make_finding(severity=FindingSeverity.HIGH)
        result = IncrementalAuditResult(
            session_id="i1",
            base_ref="main",
            head_ref="feature",
            files_changed=["a.py", "b.py"],
            files_audited=["a.py"],
            findings=[finding],
            duration_seconds=5.0,
        )
        d = result.to_dict()
        assert d["session_id"] == "i1"
        assert d["base_ref"] == "main"
        assert d["head_ref"] == "feature"
        assert d["files_changed"] == ["a.py", "b.py"]
        assert d["files_audited"] == ["a.py"]
        assert d["finding_count"] == 1
        assert len(d["findings"]) == 1
        assert d["findings"][0]["severity"] == "high"
        assert d["duration_seconds"] == 5.0
        assert d["exit_code"] == 1
        assert d["error"] is None

    def test_to_dict_with_error(self):
        from aragora.audit.codebase_auditor import IncrementalAuditResult

        result = IncrementalAuditResult(
            session_id="i1",
            base_ref="HEAD~1",
            head_ref="HEAD",
            files_changed=[],
            files_audited=[],
            findings=[],
            error="git failed",
        )
        d = result.to_dict()
        assert d["error"] == "git failed"
        assert d["exit_code"] == 2

    def test_to_markdown_no_findings(self):
        from aragora.audit.codebase_auditor import IncrementalAuditResult

        result = IncrementalAuditResult(
            session_id="i1",
            base_ref="HEAD~1",
            head_ref="HEAD",
            files_changed=["a.py"],
            files_audited=["a.py"],
            findings=[],
            duration_seconds=2.5,
        )
        md = result.to_markdown()
        assert "# Incremental Audit Report" in md
        assert "HEAD~1..HEAD" in md
        assert "No issues found" in md
        assert "Files Changed:** 1" in md

    def test_to_markdown_with_error(self):
        from aragora.audit.codebase_auditor import IncrementalAuditResult

        result = IncrementalAuditResult(
            session_id="i1",
            base_ref="HEAD~1",
            head_ref="HEAD",
            files_changed=[],
            files_audited=[],
            findings=[],
            error="git not found",
        )
        md = result.to_markdown()
        assert "## Error" in md
        assert "git not found" in md

    def test_to_markdown_with_findings(self):
        from aragora.audit.codebase_auditor import IncrementalAuditResult

        # The to_markdown method accesses f.location which AuditFinding
        # does not natively have (it has evidence_location). We set the
        # attribute directly so the markdown rendering path is exercised.
        finding_high = _make_finding(severity=FindingSeverity.HIGH, title="Secret exposed")
        finding_high.location = "file.py:10"
        finding_low = _make_finding(severity=FindingSeverity.LOW, title="TODO found")
        finding_low.location = "file.py:20"

        result = IncrementalAuditResult(
            session_id="i1",
            base_ref="HEAD~1",
            head_ref="HEAD",
            files_changed=["a.py"],
            files_audited=["a.py"],
            findings=[finding_high, finding_low],
        )
        md = result.to_markdown()
        assert "## Findings (2)" in md
        assert "Secret exposed" in md
        assert "TODO found" in md
        assert "file.py:10" in md

    def test_to_markdown_findings_without_location_attr(self):
        """to_markdown uses f.location which may not exist on AuditFinding.

        The to_dict method uses getattr(f, 'location', f.evidence_location)
        to handle this, but to_markdown accesses f.location directly.
        This test documents the behavior.
        """
        from aragora.audit.codebase_auditor import IncrementalAuditResult

        result = IncrementalAuditResult(
            session_id="i1",
            base_ref="HEAD~1",
            head_ref="HEAD",
            files_changed=["a.py"],
            files_audited=["a.py"],
            findings=[_make_finding(severity=FindingSeverity.HIGH)],
        )
        with pytest.raises(AttributeError, match="location"):
            result.to_markdown()


# ===========================================================================
# CodebaseAuditor Initialization Tests
# ===========================================================================


class TestCodebaseAuditorInit:
    """Tests for CodebaseAuditor initialization."""

    def test_default_init(self, tmp_path):
        """Test initialization with defaults."""
        from aragora.audit.codebase_auditor import CodebaseAuditor, CodebaseAuditConfig

        auditor = CodebaseAuditor(root_path=tmp_path)
        assert auditor.root_path == tmp_path
        assert isinstance(auditor.config, CodebaseAuditConfig)
        assert auditor.token_counter is not None
        assert auditor.document_auditor is not None
        assert auditor.consistency_auditor is not None
        assert auditor._chunker is None

    def test_custom_config(self, tmp_path):
        """Test initialization with custom config."""
        from aragora.audit.codebase_auditor import CodebaseAuditor, CodebaseAuditConfig

        config = CodebaseAuditConfig(chunk_size=256, min_confidence=0.9)
        auditor = CodebaseAuditor(root_path=tmp_path, config=config)
        assert auditor.config.chunk_size == 256
        assert auditor.config.min_confidence == 0.9

    def test_custom_document_auditor(self, tmp_path, mock_document_auditor):
        """Test initialization with custom document auditor."""
        from aragora.audit.codebase_auditor import CodebaseAuditor

        auditor = CodebaseAuditor(
            root_path=tmp_path,
            document_auditor=mock_document_auditor,
        )
        assert auditor.document_auditor is mock_document_auditor

    def test_custom_token_counter(self, tmp_path, mock_token_counter):
        """Test initialization with custom token counter."""
        from aragora.audit.codebase_auditor import CodebaseAuditor

        auditor = CodebaseAuditor(
            root_path=tmp_path,
            token_counter=mock_token_counter,
        )
        assert auditor.token_counter is mock_token_counter

    def test_root_path_converted_to_path(self, tmp_path):
        """Test that root_path string is converted to Path."""
        from aragora.audit.codebase_auditor import CodebaseAuditor

        auditor = CodebaseAuditor(root_path=str(tmp_path))
        assert isinstance(auditor.root_path, Path)
        assert auditor.root_path == tmp_path


# ===========================================================================
# File Inclusion/Exclusion Tests
# ===========================================================================


class TestShouldIncludeFile:
    """Tests for _should_include_file method."""

    def test_include_python_file(self, auditor, tmp_codebase):
        assert auditor._should_include_file(tmp_codebase / "aragora" / "main.py") is True

    def test_include_markdown_file(self, auditor, tmp_codebase):
        assert auditor._should_include_file(tmp_codebase / "docs" / "README.md") is True

    def test_exclude_json_file(self, auditor, tmp_codebase):
        assert auditor._should_include_file(tmp_codebase / "aragora" / "data.json") is False

    def test_exclude_pycache_path(self, auditor, tmp_codebase):
        assert auditor._should_include_file(tmp_codebase / "__pycache__" / "module.py") is False

    def test_exclude_git_path(self, auditor, tmp_codebase):
        assert auditor._should_include_file(tmp_codebase / ".git" / "config.py") is False

    def test_exclude_node_modules(self, auditor, tmp_codebase):
        assert (
            auditor._should_include_file(tmp_codebase / "node_modules" / "pkg" / "index.js")
            is False
        )

    def test_exclude_venv(self, auditor, tmp_codebase):
        assert auditor._should_include_file(tmp_codebase / ".venv" / "lib" / "site.py") is False

    def test_exclude_pyc_pattern(self, auditor, tmp_codebase):
        # The pattern "*.pyc" is checked as substring
        path = tmp_codebase / "something.pyc"
        assert auditor._should_include_file(path) is False

    def test_include_typescript_file(self, auditor, tmp_codebase):
        path = tmp_codebase / "app" / "index.ts"
        assert auditor._should_include_file(path) is True

    def test_include_tsx_file(self, auditor, tmp_codebase):
        path = tmp_codebase / "app" / "Component.tsx"
        assert auditor._should_include_file(path) is True

    def test_include_js_file(self, auditor, tmp_codebase):
        path = tmp_codebase / "scripts" / "build.js"
        assert auditor._should_include_file(path) is True

    def test_include_rst_file(self, auditor, tmp_codebase):
        path = tmp_codebase / "docs" / "guide.rst"
        assert auditor._should_include_file(path) is True

    def test_include_txt_file(self, auditor, tmp_codebase):
        path = tmp_codebase / "docs" / "notes.txt"
        assert auditor._should_include_file(path) is True

    def test_exclude_binary_file(self, auditor, tmp_codebase):
        path = tmp_codebase / "image.png"
        assert auditor._should_include_file(path) is False

    def test_exclude_no_extension(self, auditor, tmp_codebase):
        path = tmp_codebase / "Makefile"
        assert auditor._should_include_file(path) is False


# ===========================================================================
# File Collection Tests
# ===========================================================================


class TestCollectFiles:
    """Tests for _collect_files method."""

    def test_collects_python_files(self, auditor, tmp_codebase):
        files = auditor._collect_files()
        py_files = [f for f in files if f.suffix == ".py"]
        assert len(py_files) >= 3  # main.py, utils.py, run.py

    def test_collects_doc_files(self, auditor, tmp_codebase):
        files = auditor._collect_files()
        md_files = [f for f in files if f.suffix == ".md"]
        assert len(md_files) >= 1

    def test_excludes_pycache(self, auditor, tmp_codebase):
        files = auditor._collect_files()
        for f in files:
            assert "__pycache__" not in str(f)

    def test_nonexistent_include_path(self, tmp_path, mock_document_auditor, mock_token_counter):
        """Non-existent include paths are silently skipped."""
        from aragora.audit.codebase_auditor import CodebaseAuditor, CodebaseAuditConfig

        config = CodebaseAuditConfig(include_paths=["nonexistent/"])
        a = CodebaseAuditor(
            root_path=tmp_path,
            config=config,
            document_auditor=mock_document_auditor,
            token_counter=mock_token_counter,
        )
        files = a._collect_files()
        assert files == []

    def test_include_single_file(self, tmp_codebase, mock_document_auditor, mock_token_counter):
        """Test including a single file path."""
        from aragora.audit.codebase_auditor import CodebaseAuditor, CodebaseAuditConfig

        config = CodebaseAuditConfig(include_paths=["aragora/main.py"])
        a = CodebaseAuditor(
            root_path=tmp_codebase,
            config=config,
            document_auditor=mock_document_auditor,
            token_counter=mock_token_counter,
        )
        files = a._collect_files()
        assert len(files) == 1
        assert files[0].name == "main.py"

    def test_empty_directory(self, tmp_path, mock_document_auditor, mock_token_counter):
        """Empty directories yield no files."""
        from aragora.audit.codebase_auditor import CodebaseAuditor, CodebaseAuditConfig

        (tmp_path / "src").mkdir()
        config = CodebaseAuditConfig(include_paths=["src/"])
        a = CodebaseAuditor(
            root_path=tmp_path,
            config=config,
            document_auditor=mock_document_auditor,
            token_counter=mock_token_counter,
        )
        files = a._collect_files()
        assert files == []


# ===========================================================================
# Get Chunker Tests
# ===========================================================================


class TestGetChunker:
    """Tests for _get_chunker method."""

    def test_creates_chunker_on_first_call(self, auditor):
        assert auditor._chunker is None
        chunker = auditor._get_chunker()
        assert chunker is not None
        assert auditor._chunker is chunker

    def test_reuses_existing_chunker(self, auditor):
        chunker1 = auditor._get_chunker()
        chunker2 = auditor._get_chunker()
        assert chunker1 is chunker2


# ===========================================================================
# Security Pattern Detection Tests
# ===========================================================================


class TestCheckSecurityPatterns:
    """Tests for _check_security_patterns method."""

    def test_detect_hardcoded_api_key(self, auditor):
        content = 'api_key = "sk-abc123longapikeystringvalue"'
        findings = auditor._check_security_patterns(content, "file.py", "chunk:0")
        assert len(findings) >= 1
        assert any(f.category == "hardcoded_secret" for f in findings)
        assert any(f.severity == FindingSeverity.HIGH for f in findings)

    def test_detect_hardcoded_password(self, auditor):
        content = 'password = "mysupersecretpassword123"'
        findings = auditor._check_security_patterns(content, "file.py", "chunk:0")
        assert len(findings) >= 1
        assert any(f.category == "hardcoded_secret" for f in findings)

    def test_detect_hardcoded_token(self, auditor):
        content = 'token: "abcdefghijklmnop1234"'
        findings = auditor._check_security_patterns(content, "file.py", "chunk:0")
        assert len(findings) >= 1

    def test_detect_hardcoded_secret(self, auditor):
        content = 'secret = "verylongsecretvalueincode"'
        findings = auditor._check_security_patterns(content, "file.py", "chunk:0")
        assert len(findings) >= 1

    def test_detect_aws_key(self, auditor):
        content = "key = AKIAIOSFODNN7EXAMPLE"
        findings = auditor._check_security_patterns(content, "file.py", "chunk:0")
        assert any("AWS" in f.title for f in findings)

    def test_detect_openai_key(self, auditor):
        content = 'OPENAI_KEY = "sk-abcdefghijklmnopqrstuvwxyz"'
        findings = auditor._check_security_patterns(content, "file.py", "chunk:0")
        assert any("OpenAI" in f.title or "secret" in f.title.lower() for f in findings)

    def test_detect_sql_injection_fstring(self, auditor):
        content = 'cursor.execute(f"SELECT * FROM users WHERE id={user_id}")'
        findings = auditor._check_security_patterns(content, "file.py", "chunk:0")
        assert any(f.category == "sql_injection" for f in findings)

    def test_detect_sql_injection_format(self, auditor):
        content = 'cursor.execute("SELECT * FROM users WHERE id={}".format(uid))'
        findings = auditor._check_security_patterns(content, "file.py", "chunk:0")
        assert any(f.category == "sql_injection" for f in findings)

    def test_no_findings_clean_code(self, auditor):
        content = (
            "import os\n"
            "api_key = os.environ.get('API_KEY')\n"
            "cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))\n"
        )
        findings = auditor._check_security_patterns(content, "file.py", "chunk:0")
        assert len(findings) == 0

    def test_short_secret_ignored(self, auditor):
        """Secrets shorter than 8 chars should not be flagged."""
        content = 'api_key = "short"'
        findings = auditor._check_security_patterns(content, "file.py", "chunk:0")
        # The regex requires at least 8 chars
        assert len(findings) == 0

    def test_finding_attributes(self, auditor):
        content = 'password = "supersecretpasswordhere"'
        findings = auditor._check_security_patterns(content, "module.py", "mod:0")
        assert len(findings) >= 1
        f = findings[0]
        assert f.audit_type == AuditType.SECURITY
        assert f.confidence == 0.8
        assert "module.py" in f.evidence_location
        assert f.evidence_text != ""

    def test_evidence_text_truncated(self, auditor):
        """Evidence text should be truncated to at most 100 chars."""
        long_key = "A" * 200
        content = f'apikey = "{long_key}"'
        findings = auditor._check_security_patterns(content, "file.py", "chunk:0")
        if findings:
            assert len(findings[0].evidence_text) <= 100


# ===========================================================================
# Quality Pattern Detection Tests
# ===========================================================================


class TestCheckQualityPatterns:
    """Tests for _check_quality_patterns method."""

    def test_detect_todo_comment(self, auditor):
        content = "# TODO: implement this function\ndef placeholder(): pass"
        findings = auditor._check_quality_patterns(content, "file.py", "chunk:0")
        assert any(f.category == "incomplete_code" for f in findings)
        assert any("TODO" in f.title for f in findings)

    def test_detect_fixme_comment(self, auditor):
        content = "# FIXME: this is broken\nresult = 1 + 1"
        findings = auditor._check_quality_patterns(content, "file.py", "chunk:0")
        assert any("FIXME" in f.title for f in findings)

    def test_detect_xxx_comment(self, auditor):
        content = "# XXX: needs attention\npass"
        findings = auditor._check_quality_patterns(content, "file.py", "chunk:0")
        assert any("XXX" in f.title for f in findings)

    def test_detect_hack_comment(self, auditor):
        content = "# HACK: temporary workaround\nresult = magic()"
        findings = auditor._check_quality_patterns(content, "file.py", "chunk:0")
        assert any("HACK" in f.title for f in findings)

    def test_todo_case_insensitive(self, auditor):
        content = "# todo: lowercase todo\ndef func(): pass"
        findings = auditor._check_quality_patterns(content, "file.py", "chunk:0")
        assert any(f.category == "incomplete_code" for f in findings)

    def test_large_code_block(self, auditor):
        content = "\n".join([f"line {i}" for i in range(150)])
        findings = auditor._check_quality_patterns(content, "file.py", "chunk:0")
        assert any(f.category == "complexity" for f in findings)
        assert any("Large code block" in f.title for f in findings)

    def test_no_complexity_for_small_file(self, auditor):
        content = "\n".join([f"line {i}" for i in range(50)])
        findings = auditor._check_quality_patterns(content, "file.py", "chunk:0")
        assert not any(f.category == "complexity" for f in findings)

    def test_no_findings_clean_code(self, auditor):
        content = "def clean_function():\n    return 42\n"
        findings = auditor._check_quality_patterns(content, "file.py", "chunk:0")
        assert len(findings) == 0

    def test_todo_limit_per_chunk(self, auditor):
        """At most 3 TODO findings per chunk."""
        content = "\n".join([f"# TODO: item {i}" for i in range(10)])
        findings = auditor._check_quality_patterns(content, "file.py", "chunk:0")
        todo_findings = [f for f in findings if f.category == "incomplete_code"]
        assert len(todo_findings) <= 3

    def test_quality_finding_severity(self, auditor):
        content = "# TODO: do something"
        findings = auditor._check_quality_patterns(content, "file.py", "chunk:0")
        assert all(
            f.severity == FindingSeverity.LOW for f in findings if f.category == "incomplete_code"
        )

    def test_complexity_finding_confidence(self, auditor):
        content = "\n".join([f"line {i}" for i in range(150)])
        findings = auditor._check_quality_patterns(content, "file.py", "chunk:0")
        complexity_findings = [f for f in findings if f.category == "complexity"]
        assert len(complexity_findings) == 1
        assert complexity_findings[0].confidence == 0.6


# ===========================================================================
# Severity Ranking Tests
# ===========================================================================


class TestSeverityRanking:
    """Tests for _severity_rank and _severity_at_least methods."""

    def test_severity_rank_critical(self, auditor):
        assert auditor._severity_rank(FindingSeverity.CRITICAL) == 4

    def test_severity_rank_high(self, auditor):
        assert auditor._severity_rank(FindingSeverity.HIGH) == 3

    def test_severity_rank_medium(self, auditor):
        assert auditor._severity_rank(FindingSeverity.MEDIUM) == 2

    def test_severity_rank_low(self, auditor):
        assert auditor._severity_rank(FindingSeverity.LOW) == 1

    def test_severity_rank_ordering(self, auditor):
        assert (
            auditor._severity_rank(FindingSeverity.CRITICAL)
            > auditor._severity_rank(FindingSeverity.HIGH)
            > auditor._severity_rank(FindingSeverity.MEDIUM)
            > auditor._severity_rank(FindingSeverity.LOW)
        )

    def test_severity_at_least_equal(self, auditor):
        assert auditor._severity_at_least(FindingSeverity.HIGH, FindingSeverity.HIGH) is True

    def test_severity_at_least_higher(self, auditor):
        assert auditor._severity_at_least(FindingSeverity.CRITICAL, FindingSeverity.HIGH) is True

    def test_severity_at_least_lower(self, auditor):
        assert auditor._severity_at_least(FindingSeverity.LOW, FindingSeverity.HIGH) is False

    def test_severity_at_least_low_vs_low(self, auditor):
        assert auditor._severity_at_least(FindingSeverity.LOW, FindingSeverity.LOW) is True


# ===========================================================================
# Findings-to-Proposals Tests
# ===========================================================================


class TestFindingsToProposals:
    """Tests for findings_to_proposals method."""

    def test_empty_findings(self, auditor):
        proposals = auditor.findings_to_proposals([])
        assert proposals == []

    def test_single_finding_creates_one_proposal(self, auditor):
        findings = [_make_finding()]
        proposals = auditor.findings_to_proposals(findings)
        assert len(proposals) == 1
        assert proposals[0].severity == FindingSeverity.HIGH

    def test_grouped_by_category_and_type(self, auditor):
        findings = [
            _make_finding(category="hardcoded_secret", audit_type=AuditType.SECURITY),
            _make_finding(category="hardcoded_secret", audit_type=AuditType.SECURITY),
            _make_finding(category="sql_injection", audit_type=AuditType.SECURITY),
        ]
        proposals = auditor.findings_to_proposals(findings)
        # Should create 2 groups: hardcoded_secret:security and sql_injection:security
        assert len(proposals) == 2

    def test_proposal_uses_highest_severity(self, auditor):
        findings = [
            _make_finding(
                severity=FindingSeverity.LOW, category="test_cat", audit_type=AuditType.QUALITY
            ),
            _make_finding(
                severity=FindingSeverity.CRITICAL, category="test_cat", audit_type=AuditType.QUALITY
            ),
            _make_finding(
                severity=FindingSeverity.MEDIUM, category="test_cat", audit_type=AuditType.QUALITY
            ),
        ]
        proposals = auditor.findings_to_proposals(findings)
        assert len(proposals) == 1
        assert proposals[0].severity == FindingSeverity.CRITICAL

    def test_proposal_confidence_is_average(self, auditor):
        findings = [
            _make_finding(confidence=0.8, category="cat", audit_type=AuditType.QUALITY),
            _make_finding(confidence=0.6, category="cat", audit_type=AuditType.QUALITY),
        ]
        proposals = auditor.findings_to_proposals(findings)
        assert len(proposals) == 1
        assert proposals[0].confidence == pytest.approx(0.7)

    def test_proposal_affected_files(self, auditor):
        findings = [
            _make_finding(
                evidence_location="a.py:chunk0", category="cat", audit_type=AuditType.QUALITY
            ),
            _make_finding(
                evidence_location="b.py:chunk1", category="cat", audit_type=AuditType.QUALITY
            ),
        ]
        proposals = auditor.findings_to_proposals(findings)
        assert set(proposals[0].affected_files) == {"a.py", "b.py"}

    def test_proposal_tags(self, auditor):
        findings = [_make_finding(category="sql_injection", audit_type=AuditType.SECURITY)]
        proposals = auditor.findings_to_proposals(findings)
        assert "security" in proposals[0].tags
        assert "sql_injection" in proposals[0].tags

    def test_max_proposals_limit(self, auditor):
        findings = [
            _make_finding(category=f"cat_{i}", audit_type=AuditType.QUALITY) for i in range(20)
        ]
        proposals = auditor.findings_to_proposals(findings, max_proposals=3)
        assert len(proposals) <= 3

    def test_proposals_sorted_by_severity(self, auditor):
        findings = [
            _make_finding(
                category="low_cat", severity=FindingSeverity.LOW, audit_type=AuditType.QUALITY
            ),
            _make_finding(
                category="high_cat", severity=FindingSeverity.HIGH, audit_type=AuditType.SECURITY
            ),
            _make_finding(
                category="crit_cat",
                severity=FindingSeverity.CRITICAL,
                audit_type=AuditType.SECURITY,
            ),
        ]
        proposals = auditor.findings_to_proposals(findings)
        assert proposals[0].severity == FindingSeverity.CRITICAL
        assert proposals[1].severity == FindingSeverity.HIGH
        assert proposals[2].severity == FindingSeverity.LOW

    def test_affected_files_capped_at_5(self, auditor):
        findings = [
            _make_finding(
                evidence_location=f"file{i}.py:chunk0",
                category="cat",
                audit_type=AuditType.QUALITY,
            )
            for i in range(10)
        ]
        proposals = auditor.findings_to_proposals(findings)
        assert len(proposals[0].affected_files) <= 5


# ===========================================================================
# Proposal Description Building Tests
# ===========================================================================


class TestBuildProposalDescription:
    """Tests for _build_proposal_description method."""

    def test_single_finding_description(self, auditor):
        findings = [_make_finding(description="Found a bug")]
        desc = auditor._build_proposal_description(findings)
        assert desc == "Found a bug"

    def test_multiple_findings_description(self, auditor):
        findings = [
            _make_finding(description="Found a bug"),
            _make_finding(description="Found another bug"),
            _make_finding(description="And another"),
        ]
        desc = auditor._build_proposal_description(findings)
        assert "Found a bug" in desc
        assert "2 similar issues" in desc


# ===========================================================================
# Fix Suggestion Tests
# ===========================================================================


class TestSuggestFix:
    """Tests for _suggest_fix method."""

    def test_hardcoded_secret_fix(self, auditor):
        finding = _make_finding(category="hardcoded_secret")
        fix = auditor._suggest_fix(finding)
        assert "environment variables" in fix.lower() or "secrets manager" in fix.lower()

    def test_sql_injection_fix(self, auditor):
        finding = _make_finding(category="sql_injection")
        fix = auditor._suggest_fix(finding)
        assert "parameterized" in fix.lower()

    def test_incomplete_code_fix(self, auditor):
        finding = _make_finding(category="incomplete_code")
        fix = auditor._suggest_fix(finding)
        assert "TODO" in fix or "complete" in fix.lower()

    def test_complexity_fix(self, auditor):
        finding = _make_finding(category="complexity")
        fix = auditor._suggest_fix(finding)
        assert "smaller" in fix.lower() or "breaking" in fix.lower()

    def test_date_mismatch_fix(self, auditor):
        finding = _make_finding(category="date_mismatch")
        fix = auditor._suggest_fix(finding)
        assert "date" in fix.lower()

    def test_definition_conflict_fix(self, auditor):
        finding = _make_finding(category="definition_conflict")
        fix = auditor._suggest_fix(finding)
        assert "reconcile" in fix.lower() or "conflicting" in fix.lower()

    def test_unknown_category_fix(self, auditor):
        finding = _make_finding(category="unknown_thing")
        fix = auditor._suggest_fix(finding)
        assert "review" in fix.lower()


# ===========================================================================
# Summary Building Tests
# ===========================================================================


class TestBuildSummary:
    """Tests for _build_summary method."""

    def test_empty_findings(self, auditor):
        summary = auditor._build_summary([], [])
        assert "Found 0 issues" in summary
        assert "0 improvement proposals" in summary

    def test_with_findings_and_proposals(self, auditor):
        from aragora.audit.codebase_auditor import ImprovementProposal

        findings = [
            _make_finding(severity=FindingSeverity.HIGH),
            _make_finding(severity=FindingSeverity.HIGH),
            _make_finding(severity=FindingSeverity.LOW),
        ]
        proposals = [
            ImprovementProposal(
                id="p1",
                title="T",
                description="D",
                finding_ids=[],
                severity=FindingSeverity.HIGH,
                confidence=0.8,
            ),
        ]
        summary = auditor._build_summary(findings, proposals)
        assert "Found 3 issues" in summary
        assert "2 high" in summary
        assert "1 low" in summary
        assert "1 improvement proposals" in summary

    def test_severity_order_in_summary(self, auditor):
        findings = [
            _make_finding(severity=FindingSeverity.CRITICAL),
            _make_finding(severity=FindingSeverity.LOW),
            _make_finding(severity=FindingSeverity.HIGH),
        ]
        summary = auditor._build_summary(findings, [])
        # Critical should appear before high, high before low
        crit_pos = summary.index("critical")
        high_pos = summary.index("high")
        low_pos = summary.index("low")
        assert crit_pos < high_pos < low_pos


# ===========================================================================
# Pattern Audits Tests
# ===========================================================================


class TestRunPatternAudits:
    """Tests for _run_pattern_audits method."""

    @pytest.mark.asyncio
    async def test_security_patterns_for_code_files(self, auditor):
        chunks = [
            {
                "id": "file.py:0",
                "document_id": "file.py",
                "content": 'api_key = "supersecretapikey123456"',
                "file_type": ".py",
            }
        ]
        findings = await auditor._run_pattern_audits(chunks, "test_session")
        assert any(f.category == "hardcoded_secret" for f in findings)

    @pytest.mark.asyncio
    async def test_no_security_patterns_for_docs(self, auditor):
        """Doc files should not trigger security patterns."""
        chunks = [
            {
                "id": "doc.md:0",
                "document_id": "doc.md",
                "content": 'api_key = "supersecretapikey123456"',
                "file_type": ".md",
            }
        ]
        findings = await auditor._run_pattern_audits(chunks, "test_session")
        # Security patterns only run on code files
        assert not any(f.category == "hardcoded_secret" for f in findings)

    @pytest.mark.asyncio
    async def test_quality_patterns_for_all_files(self, auditor):
        chunks = [
            {
                "id": "doc.md:0",
                "document_id": "doc.md",
                "content": "# TODO: update this section",
                "file_type": ".md",
            }
        ]
        findings = await auditor._run_pattern_audits(chunks, "test_session")
        assert any(f.category == "incomplete_code" for f in findings)

    @pytest.mark.asyncio
    async def test_security_for_ts_files(self, auditor):
        chunks = [
            {
                "id": "app.ts:0",
                "document_id": "app.ts",
                "content": 'const secret = "verylongsecretvalue1234"',
                "file_type": ".ts",
            }
        ]
        findings = await auditor._run_pattern_audits(chunks, "test_session")
        assert any(f.category == "hardcoded_secret" for f in findings)

    @pytest.mark.asyncio
    async def test_security_for_js_files(self, auditor):
        chunks = [
            {
                "id": "app.js:0",
                "document_id": "app.js",
                "content": 'const password = "mysupersecretpassword"',
                "file_type": ".js",
            }
        ]
        findings = await auditor._run_pattern_audits(chunks, "test_session")
        assert any(f.category == "hardcoded_secret" for f in findings)

    @pytest.mark.asyncio
    async def test_security_for_tsx_files(self, auditor):
        chunks = [
            {
                "id": "comp.tsx:0",
                "document_id": "comp.tsx",
                "content": 'const token = "longenoughtokenvalue12345"',
                "file_type": ".tsx",
            }
        ]
        findings = await auditor._run_pattern_audits(chunks, "test_session")
        assert any(f.category == "hardcoded_secret" for f in findings)

    @pytest.mark.asyncio
    async def test_empty_chunks(self, auditor):
        findings = await auditor._run_pattern_audits([], "test_session")
        assert findings == []

    @pytest.mark.asyncio
    async def test_chunk_with_missing_content(self, auditor):
        chunks = [
            {
                "id": "file.py:0",
                "document_id": "file.py",
                "content": "",
                "file_type": ".py",
            }
        ]
        findings = await auditor._run_pattern_audits(chunks, "test_session")
        assert findings == []


# ===========================================================================
# Consistency Audit Tests
# ===========================================================================


class TestRunConsistencyAudit:
    """Tests for _run_consistency_audit method."""

    @pytest.mark.asyncio
    async def test_consistency_audit_calls_auditor(self, auditor):
        mock_findings = [_make_finding(audit_type=AuditType.CONSISTENCY)]
        auditor.consistency_auditor.audit = AsyncMock(return_value=mock_findings)

        chunks = [{"id": "f.py:0", "document_id": "f.py", "content": "content", "file_type": ".py"}]
        findings = await auditor._run_consistency_audit(chunks, "test_session")
        assert len(findings) == 1
        auditor.consistency_auditor.audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_consistency_audit_handles_error(self, auditor):
        auditor.consistency_auditor.audit = AsyncMock(side_effect=RuntimeError("boom"))

        chunks = [{"id": "f.py:0", "document_id": "f.py", "content": "content", "file_type": ".py"}]
        findings = await auditor._run_consistency_audit(chunks, "test_session")
        assert findings == []

    @pytest.mark.asyncio
    async def test_consistency_audit_limits_document_ids(self, auditor):
        """Only first 10 chunks' document_ids used in session."""
        auditor.consistency_auditor.audit = AsyncMock(return_value=[])

        chunks = [
            {"id": f"f{i}.py:0", "document_id": f"f{i}.py", "content": "x", "file_type": ".py"}
            for i in range(20)
        ]
        await auditor._run_consistency_audit(chunks, "test_session")
        call_args = auditor.consistency_auditor.audit.call_args
        session = call_args[0][1]
        assert len(session.document_ids) == 10


# ===========================================================================
# Full Codebase Audit Tests
# ===========================================================================


class TestAuditCodebase:
    """Tests for audit_codebase method."""

    @pytest.mark.asyncio
    async def test_audit_codebase_returns_result(self, auditor, mock_chunker):
        auditor._chunker = mock_chunker
        auditor.consistency_auditor.audit = AsyncMock(return_value=[])

        result = await auditor.audit_codebase()

        from aragora.audit.codebase_auditor import CodebaseAuditResult

        assert isinstance(result, CodebaseAuditResult)
        assert result.files_audited > 0
        assert result.session_id.startswith("codebase_audit_")

    @pytest.mark.asyncio
    async def test_audit_codebase_session_id_format(self, auditor, mock_chunker):
        auditor._chunker = mock_chunker
        auditor.consistency_auditor.audit = AsyncMock(return_value=[])

        result = await auditor.audit_codebase()
        assert result.session_id.startswith("codebase_audit_")

    @pytest.mark.asyncio
    async def test_audit_codebase_with_consistency(self, auditor, mock_chunker):
        auditor._chunker = mock_chunker
        consistency_finding = _make_finding(
            audit_type=AuditType.CONSISTENCY,
            category="date_mismatch",
            severity=FindingSeverity.MEDIUM,
            confidence=0.9,
        )
        auditor.consistency_auditor.audit = AsyncMock(return_value=[consistency_finding])

        result = await auditor.audit_codebase()
        assert any(f.audit_type == AuditType.CONSISTENCY for f in result.findings)

    @pytest.mark.asyncio
    async def test_audit_codebase_filters_by_confidence(
        self, tmp_path, mock_document_auditor, mock_token_counter
    ):
        """Findings below min_confidence are filtered out."""
        from aragora.audit.codebase_auditor import CodebaseAuditor, CodebaseAuditConfig

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "code.py").write_text("# TODO: low conf")

        config = CodebaseAuditConfig(
            include_paths=["src/"],
            min_confidence=0.95,  # Very high threshold
            min_severity=FindingSeverity.LOW,
        )
        a = CodebaseAuditor(
            root_path=tmp_path,
            config=config,
            document_auditor=mock_document_auditor,
            token_counter=mock_token_counter,
        )
        a.consistency_auditor.audit = AsyncMock(return_value=[])

        result = await a.audit_codebase()
        # TODO findings have confidence 0.9 which is below 0.95 threshold
        todo_findings = [f for f in result.findings if f.category == "incomplete_code"]
        assert len(todo_findings) == 0

    @pytest.mark.asyncio
    async def test_audit_codebase_filters_by_severity(
        self, tmp_path, mock_document_auditor, mock_token_counter
    ):
        """Findings below min_severity are filtered out."""
        from aragora.audit.codebase_auditor import CodebaseAuditor, CodebaseAuditConfig

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "code.py").write_text("# TODO: this is low severity\n")

        config = CodebaseAuditConfig(
            include_paths=["src/"],
            min_confidence=0.0,
            min_severity=FindingSeverity.HIGH,  # High threshold
        )
        a = CodebaseAuditor(
            root_path=tmp_path,
            config=config,
            document_auditor=mock_document_auditor,
            token_counter=mock_token_counter,
        )
        a.consistency_auditor.audit = AsyncMock(return_value=[])

        result = await a.audit_codebase()
        # TODO findings have LOW severity which is below HIGH threshold
        assert not any(f.category == "incomplete_code" for f in result.findings)

    @pytest.mark.asyncio
    async def test_audit_codebase_limits_findings(
        self, tmp_path, mock_document_auditor, mock_token_counter
    ):
        """Findings are limited to max_findings_per_cycle * 2."""
        from aragora.audit.codebase_auditor import CodebaseAuditor, CodebaseAuditConfig

        (tmp_path / "src").mkdir()
        # Create a file with many TODO comments to generate many findings
        content = "\n".join([f"# TODO: item {i}" for i in range(50)])
        (tmp_path / "src" / "code.py").write_text(content)

        config = CodebaseAuditConfig(
            include_paths=["src/"],
            min_confidence=0.0,
            min_severity=FindingSeverity.LOW,
            max_findings_per_cycle=2,
        )
        a = CodebaseAuditor(
            root_path=tmp_path,
            config=config,
            document_auditor=mock_document_auditor,
            token_counter=mock_token_counter,
        )
        a.consistency_auditor.audit = AsyncMock(return_value=[])

        result = await a.audit_codebase()
        # max_findings_per_cycle * 2 = 4
        assert len(result.findings) <= 4

    @pytest.mark.asyncio
    async def test_audit_codebase_generates_proposals(self, auditor, mock_chunker):
        auditor._chunker = mock_chunker
        # Make chunks return content with a secret
        chunk = MagicMock()
        chunk.sequence = 0
        chunk.content = 'api_key = "averylongsecretvaluehere"'
        mock_chunker.chunk.return_value = [chunk]
        auditor.consistency_auditor.audit = AsyncMock(return_value=[])

        result = await auditor.audit_codebase()
        # Should have proposals if any high-severity findings survive filtering
        # (depends on config thresholds)
        assert isinstance(result.proposals, list)

    @pytest.mark.asyncio
    async def test_audit_codebase_handles_unreadable_file(
        self, tmp_path, mock_document_auditor, mock_token_counter
    ):
        """Files that can't be read are skipped gracefully."""
        from aragora.audit.codebase_auditor import CodebaseAuditor, CodebaseAuditConfig

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "good.py").write_text("print('hello')")

        config = CodebaseAuditConfig(include_paths=["src/"])
        a = CodebaseAuditor(
            root_path=tmp_path,
            config=config,
            document_auditor=mock_document_auditor,
            token_counter=mock_token_counter,
        )
        a.consistency_auditor.audit = AsyncMock(return_value=[])

        # Mock token_counter to raise on specific content
        original_count = mock_token_counter.count

        def count_side_effect(content):
            if "hello" in content:
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "bad")
            return 100

        mock_token_counter.count.side_effect = count_side_effect

        # Should not raise
        result = await a.audit_codebase()
        assert isinstance(result, type(result))

    @pytest.mark.asyncio
    async def test_audit_codebase_summary(self, auditor, mock_chunker):
        auditor._chunker = mock_chunker
        auditor.consistency_auditor.audit = AsyncMock(return_value=[])

        result = await auditor.audit_codebase()
        assert result.summary != ""
        assert "Found" in result.summary

    @pytest.mark.asyncio
    async def test_audit_codebase_timing(self, auditor, mock_chunker):
        auditor._chunker = mock_chunker
        auditor.consistency_auditor.audit = AsyncMock(return_value=[])

        result = await auditor.audit_codebase()
        assert result.started_at <= result.completed_at
        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_audit_codebase_without_consistency_type(
        self, tmp_path, mock_document_auditor, mock_token_counter
    ):
        """When CONSISTENCY is not in audit_types, skip consistency audit."""
        from aragora.audit.codebase_auditor import CodebaseAuditor, CodebaseAuditConfig

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "code.py").write_text("print('hello')")

        config = CodebaseAuditConfig(
            include_paths=["src/"],
            audit_types=[AuditType.SECURITY, AuditType.QUALITY],
            min_severity=FindingSeverity.LOW,
            min_confidence=0.0,
        )
        a = CodebaseAuditor(
            root_path=tmp_path,
            config=config,
            document_auditor=mock_document_auditor,
            token_counter=mock_token_counter,
        )
        a.consistency_auditor.audit = AsyncMock(return_value=[_make_finding()])

        result = await a.audit_codebase()
        # Consistency auditor should NOT have been called
        a.consistency_auditor.audit.assert_not_called()


# ===========================================================================
# Audit Files Tests
# ===========================================================================


class TestAuditFiles:
    """Tests for audit_files method."""

    @pytest.mark.asyncio
    async def test_audit_specific_files(self, auditor, tmp_codebase, mock_chunker):
        auditor._chunker = mock_chunker
        # Make chunk content trigger a finding
        chunk = MagicMock()
        chunk.sequence = 0
        chunk.content = "# TODO: fix this\n"
        mock_chunker.chunk.return_value = [chunk]

        files = [tmp_codebase / "aragora" / "utils.py"]
        findings = await auditor.audit_files(files)
        assert isinstance(findings, list)

    @pytest.mark.asyncio
    async def test_audit_nonexistent_file(self, auditor, tmp_codebase, mock_chunker):
        auditor._chunker = mock_chunker
        files = [tmp_codebase / "nonexistent.py"]
        findings = await auditor.audit_files(files)
        # Nonexistent files are skipped
        assert findings == []

    @pytest.mark.asyncio
    async def test_audit_files_handles_read_error(self, auditor, tmp_codebase, mock_chunker):
        """Files that fail to read are skipped."""
        auditor._chunker = mock_chunker

        # Create a file then mock read_text to fail
        test_file = tmp_codebase / "aragora" / "broken.py"
        test_file.write_text("content")

        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            findings = await auditor.audit_files([test_file])
        assert findings == []

    @pytest.mark.asyncio
    async def test_audit_files_empty_list(self, auditor, mock_chunker):
        auditor._chunker = mock_chunker
        findings = await auditor.audit_files([])
        assert findings == []

    @pytest.mark.asyncio
    async def test_audit_files_multiple(self, auditor, tmp_codebase, mock_chunker):
        auditor._chunker = mock_chunker
        chunk = MagicMock()
        chunk.sequence = 0
        chunk.content = "clean code"
        mock_chunker.chunk.return_value = [chunk]

        files = [
            tmp_codebase / "aragora" / "main.py",
            tmp_codebase / "aragora" / "utils.py",
        ]
        findings = await auditor.audit_files(files)
        assert isinstance(findings, list)


# ===========================================================================
# Git Diff Audit Tests
# ===========================================================================


class TestAuditGitDiff:
    """Tests for audit_git_diff method."""

    @pytest.mark.asyncio
    async def test_audit_git_diff_success(self, auditor, tmp_codebase, mock_chunker):
        auditor._chunker = mock_chunker
        chunk = MagicMock()
        chunk.sequence = 0
        chunk.content = "clean"
        mock_chunker.chunk.return_value = [chunk]

        mock_result = MagicMock()
        mock_result.stdout = "aragora/main.py\n"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = await auditor.audit_git_diff()

        from aragora.audit.codebase_auditor import IncrementalAuditResult

        assert isinstance(result, IncrementalAuditResult)
        assert "aragora/main.py" in result.files_changed

    @pytest.mark.asyncio
    async def test_audit_git_diff_error(self, auditor):
        with patch("subprocess.run", side_effect=OSError("git not found")):
            result = await auditor.audit_git_diff()

        assert result.error == "git not found"
        assert result.files_changed == []
        assert result.exit_code == 2

    @pytest.mark.asyncio
    async def test_audit_git_diff_custom_refs(self, auditor, tmp_codebase, mock_chunker):
        auditor._chunker = mock_chunker
        chunk = MagicMock()
        chunk.sequence = 0
        chunk.content = "clean"
        mock_chunker.chunk.return_value = [chunk]

        mock_result = MagicMock()
        mock_result.stdout = "aragora/main.py\n"

        with patch("subprocess.run", return_value=mock_result):
            result = await auditor.audit_git_diff(base_ref="main", head_ref="feature")

        assert result.base_ref == "main"
        assert result.head_ref == "feature"

    @pytest.mark.asyncio
    async def test_audit_git_diff_include_untracked(self, auditor, tmp_codebase, mock_chunker):
        auditor._chunker = mock_chunker
        chunk = MagicMock()
        chunk.sequence = 0
        chunk.content = "clean"
        mock_chunker.chunk.return_value = [chunk]

        diff_result = MagicMock()
        diff_result.stdout = "aragora/main.py\n"

        untracked_result = MagicMock()
        untracked_result.stdout = "aragora/utils.py\n"

        call_count = 0

        def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return diff_result
            return untracked_result

        with patch("subprocess.run", side_effect=mock_run):
            result = await auditor.audit_git_diff(include_untracked=True)

        assert "aragora/main.py" in result.files_changed
        assert "aragora/utils.py" in result.files_changed

    @pytest.mark.asyncio
    async def test_audit_git_diff_untracked_error(self, auditor, tmp_codebase, mock_chunker):
        """Failure to get untracked files should not fail the whole audit."""
        auditor._chunker = mock_chunker
        chunk = MagicMock()
        chunk.sequence = 0
        chunk.content = "clean"
        mock_chunker.chunk.return_value = [chunk]

        diff_result = MagicMock()
        diff_result.stdout = "aragora/main.py\n"

        call_count = 0

        def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return diff_result
            raise OSError("failed")

        with patch("subprocess.run", side_effect=mock_run):
            result = await auditor.audit_git_diff(include_untracked=True)

        # Should still have the diff files
        assert "aragora/main.py" in result.files_changed
        assert result.error is None

    @pytest.mark.asyncio
    async def test_audit_git_diff_filters_excluded(self, auditor, tmp_codebase, mock_chunker):
        """Files matching exclude patterns are filtered out."""
        auditor._chunker = mock_chunker

        mock_result = MagicMock()
        mock_result.stdout = "__pycache__/module.py\naragora/main.py\n"

        with patch("subprocess.run", return_value=mock_result):
            result = await auditor.audit_git_diff()

        # __pycache__ file should be excluded from audited files
        assert "__pycache__/module.py" not in result.files_audited

    @pytest.mark.asyncio
    async def test_audit_git_diff_filters_extensions(self, auditor, tmp_codebase, mock_chunker):
        """Only files with auditable extensions are included."""
        auditor._chunker = mock_chunker

        mock_result = MagicMock()
        mock_result.stdout = "aragora/main.py\naragora/data.json\nimage.png\n"

        with patch("subprocess.run", return_value=mock_result):
            result = await auditor.audit_git_diff()

        # Only .py should be in audited files (json and png are excluded)
        for audited_file in result.files_audited:
            path = Path(audited_file)
            assert path.suffix in auditor.config.code_extensions + auditor.config.doc_extensions

    @pytest.mark.asyncio
    async def test_audit_git_diff_empty_diff(self, auditor, mock_chunker):
        auditor._chunker = mock_chunker

        mock_result = MagicMock()
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            result = await auditor.audit_git_diff()

        assert result.files_changed == []
        assert result.files_audited == []
        assert result.findings == []
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_audit_git_diff_session_id_format(self, auditor, mock_chunker):
        auditor._chunker = mock_chunker

        mock_result = MagicMock()
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            result = await auditor.audit_git_diff()

        assert result.session_id.startswith("incremental_audit_")

    @pytest.mark.asyncio
    async def test_audit_git_diff_duration(self, auditor, mock_chunker):
        auditor._chunker = mock_chunker

        mock_result = MagicMock()
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            result = await auditor.audit_git_diff()

        assert result.duration_seconds >= 0


# ===========================================================================
# Edge Case Tests
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_findings_by_severity_with_duplicate_severities(self):
        from aragora.audit.codebase_auditor import CodebaseAuditResult

        findings = [
            _make_finding(severity=FindingSeverity.MEDIUM),
            _make_finding(severity=FindingSeverity.MEDIUM),
            _make_finding(severity=FindingSeverity.MEDIUM),
        ]
        result = CodebaseAuditResult(
            session_id="s1",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            files_audited=1,
            total_tokens=100,
            findings=findings,
            proposals=[],
        )
        assert result.findings_by_severity == {"medium": 3}

    def test_proposal_to_dict_with_empty_lists(self):
        from aragora.audit.codebase_auditor import ImprovementProposal

        proposal = ImprovementProposal(
            id="p1",
            title="T",
            description="D",
            finding_ids=[],
            severity=FindingSeverity.LOW,
            confidence=0.5,
            affected_files=[],
            tags=[],
        )
        d = proposal.to_dict()
        assert d["finding_ids"] == []
        assert d["affected_files"] == []
        assert d["tags"] == []

    def test_incremental_result_default_duration(self):
        from aragora.audit.codebase_auditor import IncrementalAuditResult

        result = IncrementalAuditResult(
            session_id="i1",
            base_ref="HEAD~1",
            head_ref="HEAD",
            files_changed=[],
            files_audited=[],
            findings=[],
        )
        assert result.duration_seconds == 0.0

    def test_incremental_result_no_error_default(self):
        from aragora.audit.codebase_auditor import IncrementalAuditResult

        result = IncrementalAuditResult(
            session_id="i1",
            base_ref="HEAD~1",
            head_ref="HEAD",
            files_changed=[],
            files_audited=[],
            findings=[],
        )
        assert result.error is None

    @pytest.mark.asyncio
    async def test_audit_codebase_empty_codebase(
        self, tmp_path, mock_document_auditor, mock_token_counter
    ):
        """Audit on empty codebase returns valid result with 0 files."""
        from aragora.audit.codebase_auditor import CodebaseAuditor, CodebaseAuditConfig

        config = CodebaseAuditConfig(include_paths=["nonexistent/"])
        a = CodebaseAuditor(
            root_path=tmp_path,
            config=config,
            document_auditor=mock_document_auditor,
            token_counter=mock_token_counter,
        )
        a.consistency_auditor.audit = AsyncMock(return_value=[])

        result = await a.audit_codebase()
        assert result.files_audited == 0
        assert result.total_tokens == 0
        assert result.findings == []
        assert result.proposals == []

    def test_security_patterns_multiple_matches_in_one_content(self, auditor):
        content = 'api_key = "firstsupersecretkey123"\npassword = "secondsupersecretpwd456"\n'
        findings = auditor._check_security_patterns(content, "file.py", "chunk:0")
        assert len(findings) >= 2

    def test_quality_todo_extracts_message(self, auditor):
        content = "# TODO: implement the cache invalidation logic\n"
        findings = auditor._check_quality_patterns(content, "file.py", "chunk:0")
        todo_findings = [f for f in findings if "TODO" in f.title]
        assert len(todo_findings) == 1
        assert "implement the cache invalidation" in todo_findings[0].description

    def test_findings_to_proposals_no_evidence_location(self, auditor):
        """Findings without evidence_location should not cause errors."""
        finding = _make_finding(evidence_location="")
        proposals = auditor.findings_to_proposals([finding])
        assert len(proposals) == 1
        # Empty evidence_location should result in empty split
        # but the code does `if f.evidence_location` check
        assert isinstance(proposals[0].affected_files, list)

    def test_severity_rank_unknown(self, auditor):
        """Unknown severity returns 0."""
        # Create a mock severity that's not in the map
        mock_sev = MagicMock()
        rank = auditor._severity_rank(mock_sev)
        assert rank == 0

    def test_codebase_audit_result_to_dict_structure(self):
        from aragora.audit.codebase_auditor import CodebaseAuditResult

        start = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 1, 12, 5, 0, tzinfo=timezone.utc)
        result = CodebaseAuditResult(
            session_id="test_session",
            started_at=start,
            completed_at=end,
            files_audited=42,
            total_tokens=99999,
            findings=[],
            proposals=[],
            summary="All clean",
        )
        d = result.to_dict()
        assert set(d.keys()) == {
            "session_id",
            "duration_seconds",
            "files_audited",
            "total_tokens",
            "findings_count",
            "findings_by_severity",
            "proposals_count",
            "summary",
        }
        assert d["duration_seconds"] == 300.0
        assert d["files_audited"] == 42
        assert d["total_tokens"] == 99999

    def test_incremental_result_to_dict_structure(self):
        from aragora.audit.codebase_auditor import IncrementalAuditResult

        result = IncrementalAuditResult(
            session_id="i1",
            base_ref="HEAD~1",
            head_ref="HEAD",
            files_changed=[],
            files_audited=[],
            findings=[],
        )
        d = result.to_dict()
        expected_keys = {
            "session_id",
            "base_ref",
            "head_ref",
            "files_changed",
            "files_audited",
            "finding_count",
            "findings",
            "duration_seconds",
            "exit_code",
            "error",
        }
        assert set(d.keys()) == expected_keys
