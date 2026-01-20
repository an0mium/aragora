"""Tests for harnesses module.

Tests the code analysis harness system including:
- CodeAnalysisHarness: abstract base class
- HarnessConfig: configuration
- HarnessResult: analysis results
- AnalysisFinding: individual findings
- ClaudeCodeHarness: Claude Code integration
"""

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.harnesses.base import (
    AnalysisFinding,
    AnalysisType,
    CodeAnalysisHarness,
    HarnessConfig,
    HarnessConfigError,
    HarnessError,
    HarnessResult,
    HarnessTimeoutError,
    SessionContext,
    SessionResult,
)
from aragora.harnesses.claude_code import (
    ClaudeCodeConfig,
    ClaudeCodeHarness,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def temp_repo(temp_dir):
    """Create a temporary repository with files."""
    # Create some test files
    (temp_dir / "main.py").write_text("print('hello')\n")
    (temp_dir / "test.py").write_text("def test_func():\n    pass\n")
    (temp_dir / "data.json").write_text('{"key": "value"}\n')

    # Create subdirectory
    sub = temp_dir / "src"
    sub.mkdir()
    (sub / "module.py").write_text("class MyClass:\n    pass\n")

    return temp_dir


@pytest.fixture
def harness_config():
    """Create a HarnessConfig for testing."""
    return HarnessConfig(
        timeout_seconds=30,
        max_retries=1,
        max_file_size_mb=5,
        max_files=50,
        include_patterns=["**/*.py", "**/*.json"],
        exclude_patterns=["**/__pycache__/**", "**/node_modules/**"],
    )


@pytest.fixture
def claude_config():
    """Create a ClaudeCodeConfig for testing."""
    return ClaudeCodeConfig(
        timeout_seconds=60,
        claude_code_path="claude",
        model="claude-sonnet-4-20250514",
    )


# =============================================================================
# AnalysisType Tests
# =============================================================================


class TestAnalysisType:
    """Test AnalysisType enum."""

    def test_all_types_exist(self):
        """Test all analysis types are defined."""
        assert AnalysisType.SECURITY.value == "security"
        assert AnalysisType.QUALITY.value == "quality"
        assert AnalysisType.PERFORMANCE.value == "performance"
        assert AnalysisType.ARCHITECTURE.value == "architecture"
        assert AnalysisType.DEPENDENCIES.value == "dependencies"
        assert AnalysisType.DOCUMENTATION.value == "documentation"
        assert AnalysisType.TESTING.value == "testing"
        assert AnalysisType.GENERAL.value == "general"


# =============================================================================
# HarnessError Tests
# =============================================================================


class TestHarnessErrors:
    """Test harness error classes."""

    def test_harness_error(self):
        """Test base HarnessError."""
        error = HarnessError("Test error", harness="test", details={"key": "value"})

        assert str(error) == "Test error"
        assert error.harness == "test"
        assert error.details == {"key": "value"}

    def test_harness_timeout_error(self):
        """Test HarnessTimeoutError."""
        error = HarnessTimeoutError("Timed out", "claude")

        assert isinstance(error, HarnessError)
        assert error.harness == "claude"

    def test_harness_config_error(self):
        """Test HarnessConfigError."""
        error = HarnessConfigError("Invalid config", "codex")

        assert isinstance(error, HarnessError)
        assert error.harness == "codex"


# =============================================================================
# HarnessConfig Tests
# =============================================================================


class TestHarnessConfig:
    """Test HarnessConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HarnessConfig()

        assert config.timeout_seconds == 300
        assert config.max_retries == 2
        assert config.verbose is False
        assert config.stream_output is True
        assert config.max_file_size_mb == 10
        assert config.max_files == 1000

    def test_custom_config(self, harness_config):
        """Test custom configuration."""
        assert harness_config.timeout_seconds == 30
        assert harness_config.max_retries == 1
        assert "**/*.py" in harness_config.include_patterns

    def test_default_exclude_patterns(self):
        """Test default exclude patterns."""
        config = HarnessConfig()

        assert "**/.git/**" in config.exclude_patterns
        assert "**/node_modules/**" in config.exclude_patterns
        assert "**/__pycache__/**" in config.exclude_patterns


# =============================================================================
# AnalysisFinding Tests
# =============================================================================


class TestAnalysisFinding:
    """Test AnalysisFinding dataclass."""

    def test_create_finding(self):
        """Test creating an analysis finding."""
        finding = AnalysisFinding(
            id="finding-001",
            title="SQL Injection Vulnerability",
            description="User input not sanitized",
            severity="high",
            confidence=0.9,
            category="security",
            file_path="src/db.py",
            line_start=42,
            line_end=45,
            code_snippet="query = f'SELECT * FROM users WHERE id={user_id}'",
            recommendation="Use parameterized queries",
        )

        assert finding.id == "finding-001"
        assert finding.severity == "high"
        assert finding.confidence == 0.9
        assert finding.line_start == 42

    def test_finding_defaults(self):
        """Test finding default values."""
        finding = AnalysisFinding(
            id="f1",
            title="Title",
            description="Desc",
            severity="medium",
            confidence=0.5,
            category="quality",
            file_path="test.py",
        )

        assert finding.line_start is None
        assert finding.code_snippet == ""
        assert finding.recommendation == ""
        assert finding.references == []
        assert finding.metadata == {}


# =============================================================================
# HarnessResult Tests
# =============================================================================


class TestHarnessResult:
    """Test HarnessResult dataclass."""

    def test_create_result(self):
        """Test creating a harness result."""
        finding = AnalysisFinding(
            id="f1",
            title="Test",
            description="Test finding",
            severity="medium",
            confidence=0.8,
            category="quality",
            file_path="test.py",
        )

        result = HarnessResult(
            harness="test-harness",
            analysis_type=AnalysisType.QUALITY,
            success=True,
            findings=[finding],
            files_analyzed=10,
            lines_analyzed=500,
        )

        assert result.harness == "test-harness"
        assert result.success is True
        assert len(result.findings) == 1
        assert result.files_analyzed == 10

    def test_result_severity_counts(self):
        """Test automatic severity counting."""
        findings = [
            AnalysisFinding(
                id="1",
                title="T",
                description="D",
                severity="high",
                confidence=0.9,
                category="c",
                file_path="f",
            ),
            AnalysisFinding(
                id="2",
                title="T",
                description="D",
                severity="high",
                confidence=0.8,
                category="c",
                file_path="f",
            ),
            AnalysisFinding(
                id="3",
                title="T",
                description="D",
                severity="medium",
                confidence=0.7,
                category="c",
                file_path="f",
            ),
            AnalysisFinding(
                id="4",
                title="T",
                description="D",
                severity="low",
                confidence=0.6,
                category="c",
                file_path="f",
            ),
        ]

        result = HarnessResult(
            harness="test",
            analysis_type=AnalysisType.SECURITY,
            success=True,
            findings=findings,
        )

        assert result.findings_by_severity["high"] == 2
        assert result.findings_by_severity["medium"] == 1
        assert result.findings_by_severity["low"] == 1

    def test_result_duration_calculation(self):
        """Test duration calculation."""
        started = datetime.utcnow()
        completed = started + timedelta(seconds=30)

        result = HarnessResult(
            harness="test",
            analysis_type=AnalysisType.GENERAL,
            success=True,
            findings=[],
            started_at=started,
            completed_at=completed,
        )

        assert abs(result.duration_seconds - 30) < 1

    def test_result_error_case(self):
        """Test result with error."""
        result = HarnessResult(
            harness="test",
            analysis_type=AnalysisType.SECURITY,
            success=False,
            findings=[],
            error_message="Analysis failed",
        )

        assert result.success is False
        assert result.error_message == "Analysis failed"


# =============================================================================
# SessionContext and SessionResult Tests
# =============================================================================


class TestSessionTypes:
    """Test session-related types."""

    def test_session_context(self, temp_repo):
        """Test SessionContext creation."""
        context = SessionContext(
            session_id="session-001",
            repo_path=temp_repo,
            files_in_context=["main.py", "test.py"],
        )

        assert context.session_id == "session-001"
        assert context.repo_path == temp_repo
        assert len(context.files_in_context) == 2
        assert context.conversation_history == []

    def test_session_result(self):
        """Test SessionResult creation."""
        result = SessionResult(
            session_id="session-001",
            response="Here is my analysis...",
            suggestions=["Consider adding tests", "Add documentation"],
            continue_conversation=True,
        )

        assert result.session_id == "session-001"
        assert len(result.suggestions) == 2
        assert result.continue_conversation is True


# =============================================================================
# CodeAnalysisHarness Base Class Tests
# =============================================================================


class TestCodeAnalysisHarnessBase:
    """Test CodeAnalysisHarness abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            CodeAnalysisHarness()

    def test_validate_path_exists(self, temp_repo):
        """Test path validation for existing path."""

        # Create a concrete implementation for testing
        class TestHarness(CodeAnalysisHarness):
            @property
            def name(self):
                return "test"

            @property
            def supported_analysis_types(self):
                return [AnalysisType.GENERAL]

            async def analyze_repository(
                self, repo_path, analysis_type=AnalysisType.GENERAL, prompt=None, options=None
            ):
                return HarnessResult(
                    harness="test", analysis_type=analysis_type, success=True, findings=[]
                )

            async def analyze_files(
                self, files, analysis_type=AnalysisType.GENERAL, prompt=None, options=None
            ):
                return HarnessResult(
                    harness="test", analysis_type=analysis_type, success=True, findings=[]
                )

        harness = TestHarness()
        # Should not raise
        harness._validate_path(temp_repo)

    def test_validate_path_not_exists(self):
        """Test path validation for non-existing path."""

        class TestHarness(CodeAnalysisHarness):
            @property
            def name(self):
                return "test"

            @property
            def supported_analysis_types(self):
                return [AnalysisType.GENERAL]

            async def analyze_repository(
                self, repo_path, analysis_type=AnalysisType.GENERAL, prompt=None, options=None
            ):
                return HarnessResult(
                    harness="test", analysis_type=analysis_type, success=True, findings=[]
                )

            async def analyze_files(
                self, files, analysis_type=AnalysisType.GENERAL, prompt=None, options=None
            ):
                return HarnessResult(
                    harness="test", analysis_type=analysis_type, success=True, findings=[]
                )

        harness = TestHarness()

        with pytest.raises(HarnessConfigError):
            harness._validate_path(Path("/nonexistent/path"))

    def test_should_include_file(self, harness_config, temp_repo):
        """Test file inclusion logic."""

        class TestHarness(CodeAnalysisHarness):
            @property
            def name(self):
                return "test"

            @property
            def supported_analysis_types(self):
                return [AnalysisType.GENERAL]

            async def analyze_repository(
                self, repo_path, analysis_type=AnalysisType.GENERAL, prompt=None, options=None
            ):
                return HarnessResult(
                    harness="test", analysis_type=analysis_type, success=True, findings=[]
                )

            async def analyze_files(
                self, files, analysis_type=AnalysisType.GENERAL, prompt=None, options=None
            ):
                return HarnessResult(
                    harness="test", analysis_type=analysis_type, success=True, findings=[]
                )

        harness = TestHarness(harness_config)

        # Python file should be included
        assert harness._should_include_file(temp_repo / "main.py")

        # JSON file should be included (in include_patterns)
        assert harness._should_include_file(temp_repo / "data.json")

    def test_interactive_session_not_implemented(self, harness_config):
        """Test that interactive session raises NotImplementedError by default."""

        class TestHarness(CodeAnalysisHarness):
            @property
            def name(self):
                return "test"

            @property
            def supported_analysis_types(self):
                return [AnalysisType.GENERAL]

            async def analyze_repository(
                self, repo_path, analysis_type=AnalysisType.GENERAL, prompt=None, options=None
            ):
                return HarnessResult(
                    harness="test", analysis_type=analysis_type, success=True, findings=[]
                )

            async def analyze_files(
                self, files, analysis_type=AnalysisType.GENERAL, prompt=None, options=None
            ):
                return HarnessResult(
                    harness="test", analysis_type=analysis_type, success=True, findings=[]
                )

        harness = TestHarness(harness_config)
        context = SessionContext(session_id="s1", repo_path=Path("."))

        with pytest.raises(NotImplementedError):
            asyncio.run(harness.start_interactive_session(context))

        with pytest.raises(NotImplementedError):
            asyncio.run(harness.continue_session(context, "test input"))


# =============================================================================
# ClaudeCodeConfig Tests
# =============================================================================


class TestClaudeCodeConfig:
    """Test ClaudeCodeConfig."""

    def test_default_config(self):
        """Test default ClaudeCode configuration."""
        config = ClaudeCodeConfig()

        assert config.claude_code_path == "claude"
        assert "claude" in config.model
        assert config.max_thinking_tokens == 10000
        assert config.parse_structured_output is True

    def test_analysis_prompts(self):
        """Test analysis prompts are defined."""
        config = ClaudeCodeConfig()

        assert AnalysisType.SECURITY.value in config.analysis_prompts
        assert AnalysisType.QUALITY.value in config.analysis_prompts
        assert AnalysisType.ARCHITECTURE.value in config.analysis_prompts
        assert AnalysisType.GENERAL.value in config.analysis_prompts

        # Security prompt should mention vulnerabilities
        assert "vulnerab" in config.analysis_prompts[AnalysisType.SECURITY.value].lower()


# =============================================================================
# ClaudeCodeHarness Tests
# =============================================================================


class TestClaudeCodeHarness:
    """Test ClaudeCodeHarness."""

    def test_harness_properties(self, claude_config):
        """Test harness properties."""
        harness = ClaudeCodeHarness(claude_config)

        assert harness.name == "claude-code"
        assert AnalysisType.SECURITY in harness.supported_analysis_types
        assert AnalysisType.GENERAL in harness.supported_analysis_types

    @pytest.mark.asyncio
    async def test_initialize_cli_not_found(self, claude_config):
        """Test initialization when CLI not found."""
        harness = ClaudeCodeHarness(claude_config)

        with patch("shutil.which", return_value=None):
            result = await harness.initialize()
            assert result is False

    @pytest.mark.asyncio
    async def test_initialize_success(self, claude_config):
        """Test successful initialization."""
        harness = ClaudeCodeHarness(claude_config)

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = AsyncMock()
                mock_proc.communicate.return_value = (b"claude 1.0.0", b"")
                mock_proc.returncode = 0
                mock_exec.return_value = mock_proc

                result = await harness.initialize()
                assert result is True
                assert harness._initialized is True

    @pytest.mark.asyncio
    async def test_analyze_repository(self, claude_config, temp_repo):
        """Test repository analysis."""
        harness = ClaudeCodeHarness(claude_config)

        mock_output = """Here is my analysis:
[
    {
        "id": "finding-1",
        "title": "Potential Issue",
        "description": "Found something",
        "severity": "medium",
        "confidence": 0.75,
        "category": "quality",
        "file_path": "main.py"
    }
]
"""

        with patch.object(harness, "_run_claude_code", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (mock_output, "")

            result = await harness.analyze_repository(
                temp_repo,
                analysis_type=AnalysisType.QUALITY,
            )

            assert result.success is True
            assert len(result.findings) == 1
            assert result.findings[0].title == "Potential Issue"

    @pytest.mark.asyncio
    async def test_analyze_repository_timeout(self, claude_config, temp_repo):
        """Test repository analysis timeout."""
        harness = ClaudeCodeHarness(claude_config)

        with patch.object(harness, "_run_claude_code", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = HarnessTimeoutError("Timed out", "claude-code")

            result = await harness.analyze_repository(temp_repo)

            assert result.success is False
            assert "timed out" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_analyze_files(self, claude_config, temp_repo):
        """Test file analysis."""
        harness = ClaudeCodeHarness(claude_config)

        with patch.object(harness, "analyze_repository", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = HarnessResult(
                harness="claude-code",
                analysis_type=AnalysisType.SECURITY,
                success=True,
                findings=[],
            )

            files = [temp_repo / "main.py", temp_repo / "test.py"]
            result = await harness.analyze_files(files, AnalysisType.SECURITY)

            assert result.success is True

    @pytest.mark.asyncio
    async def test_analyze_files_empty(self, claude_config):
        """Test analyzing empty file list."""
        harness = ClaudeCodeHarness(claude_config)

        result = await harness.analyze_files([])

        assert result.success is False
        assert "no files" in result.error_message.lower()

    def test_parse_findings_json(self, claude_config):
        """Test parsing JSON findings from output."""
        harness = ClaudeCodeHarness(claude_config)

        output = """Some text here...
[
    {"id": "1", "title": "Finding 1", "description": "Desc", "severity": "high", "confidence": 0.9, "category": "security", "file_path": "test.py"},
    {"id": "2", "title": "Finding 2", "description": "Desc", "severity": "low", "confidence": 0.6, "category": "quality", "file_path": "other.py"}
]
More text..."""

        findings = harness._parse_findings(output, AnalysisType.SECURITY)

        assert len(findings) == 2
        assert findings[0].title == "Finding 1"
        assert findings[0].severity == "high"
        assert findings[1].title == "Finding 2"

    def test_parse_findings_no_json(self, claude_config):
        """Test parsing when no JSON found."""
        harness = ClaudeCodeHarness(claude_config)

        output = "This is just plain text analysis with no structured data."

        findings = harness._parse_findings(output, AnalysisType.GENERAL)

        # Should return empty list or attempt text parsing
        assert isinstance(findings, list)

    def test_collect_files(self, claude_config, temp_repo):
        """Test file collection."""
        harness = ClaudeCodeHarness(claude_config)

        files = harness._collect_files(temp_repo)

        # Should find the Python and JSON files
        file_names = [f.name for f in files]
        assert "main.py" in file_names
        assert "test.py" in file_names

    def test_collect_files_respects_limits(self, claude_config, temp_dir):
        """Test file collection respects max_files limit."""
        claude_config.max_files = 2
        harness = ClaudeCodeHarness(claude_config)

        # Create more files than limit
        for i in range(5):
            (temp_dir / f"file{i}.py").write_text(f"# file {i}")

        files = harness._collect_files(temp_dir)

        assert len(files) <= 2

    @pytest.mark.asyncio
    async def test_interactive_session(self, claude_config, temp_repo):
        """Test interactive session."""
        harness = ClaudeCodeHarness(claude_config)

        context = SessionContext(
            session_id="test-session",
            repo_path=temp_repo,
        )

        with patch.object(harness, "_run_claude_code", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ("Initial response", "")

            result = await harness.start_interactive_session(context)

            assert result.session_id == "test-session"
            assert result.response == "Initial response"
            assert result.continue_conversation is True

    @pytest.mark.asyncio
    async def test_continue_session(self, claude_config, temp_repo):
        """Test continuing interactive session."""
        harness = ClaudeCodeHarness(claude_config)

        context = SessionContext(
            session_id="test-session",
            repo_path=temp_repo,
        )

        # Start session first
        harness._sessions["test-session"] = context

        with patch.object(harness, "_run_claude_code", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ("Follow-up response", "")

            result = await harness.continue_session(context, "What about security?")

            assert result.response == "Follow-up response"
            # Should have updated conversation history
            assert len(context.conversation_history) == 2

    @pytest.mark.asyncio
    async def test_continue_session_not_found(self, claude_config, temp_repo):
        """Test continuing non-existent session."""
        harness = ClaudeCodeHarness(claude_config)

        context = SessionContext(
            session_id="nonexistent",
            repo_path=temp_repo,
        )

        with pytest.raises(HarnessError, match="Session not found"):
            await harness.continue_session(context, "test")

    @pytest.mark.asyncio
    async def test_end_session(self, claude_config, temp_repo):
        """Test ending session."""
        harness = ClaudeCodeHarness(claude_config)

        context = SessionContext(
            session_id="test-session",
            repo_path=temp_repo,
        )
        harness._sessions["test-session"] = context

        await harness.end_session(context)

        assert "test-session" not in harness._sessions


# =============================================================================
# Integration Tests
# =============================================================================


class TestHarnessIntegration:
    """Integration tests for harness system."""

    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, claude_config, temp_repo):
        """Test complete analysis workflow."""
        harness = ClaudeCodeHarness(claude_config)

        mock_output = """Analysis complete.
[
    {"id": "1", "title": "Missing docstring", "description": "Function lacks documentation",
     "severity": "low", "confidence": 0.8, "category": "documentation", "file_path": "main.py",
     "line_start": 1, "recommendation": "Add a docstring"}
]
"""

        with patch.object(harness, "_run_claude_code", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (mock_output, "")

            # Run analysis
            result = await harness.analyze_repository(
                temp_repo,
                analysis_type=AnalysisType.DOCUMENTATION,
            )

            assert result.success is True
            assert result.harness == "claude-code"
            assert result.analysis_type == AnalysisType.DOCUMENTATION
            assert len(result.findings) == 1

            finding = result.findings[0]
            assert finding.title == "Missing docstring"
            assert finding.recommendation == "Add a docstring"
            assert finding.line_start == 1

    def test_config_inheritance(self):
        """Test ClaudeCodeConfig inherits from HarnessConfig."""
        config = ClaudeCodeConfig(
            timeout_seconds=120,
            max_files=500,
            claude_code_path="/custom/path",
        )

        assert config.timeout_seconds == 120
        assert config.max_files == 500
        assert config.claude_code_path == "/custom/path"
        # Should also have base class defaults
        assert config.capture_stderr is True
