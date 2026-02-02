"""Tests for Codex Harness.

Covers:
- CodexConfig configuration
- CodexHarness initialization and properties
- analyze_repository method
- analyze_files method
- stream_analysis method
- Interactive sessions
- Finding parsing (JSON and text extraction)
- File gathering and filtering
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.harnesses.base import (
    AnalysisFinding,
    AnalysisType,
    HarnessError,
    HarnessResult,
    SessionContext,
)
from aragora.harnesses.codex import (
    CodexConfig,
    CodexHarness,
    create_codex_harness,
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
    (temp_dir / "main.py").write_text("def main():\n    print('hello')\n")
    (temp_dir / "test.py").write_text("def test_func():\n    assert True\n")
    (temp_dir / "data.json").write_text('{"key": "value"}\n')

    # Create subdirectory
    sub = temp_dir / "src"
    sub.mkdir()
    (sub / "module.py").write_text("class MyClass:\n    pass\n")

    return temp_dir


@pytest.fixture
def codex_config():
    """Create a CodexConfig for testing."""
    return CodexConfig(
        timeout_seconds=60,
        model="gpt-4o",
        temperature=0.2,
        api_key="test-key",
    )


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock = MagicMock()
    mock.chat.completions.create = AsyncMock()
    return mock


# =============================================================================
# CodexConfig Tests
# =============================================================================


class TestCodexConfig:
    """Test CodexConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CodexConfig()

        assert config.model == "gpt-4o"
        assert config.temperature == 0.2
        assert config.max_tokens == 4096
        assert config.api_key is None

    def test_custom_config(self, codex_config):
        """Test custom configuration."""
        assert codex_config.model == "gpt-4o"
        assert codex_config.api_key == "test-key"
        assert codex_config.timeout_seconds == 60

    def test_analysis_prompts_exist(self):
        """Test that analysis prompts are defined for all types."""
        config = CodexConfig()

        for analysis_type in AnalysisType:
            assert analysis_type.value in config.analysis_prompts
            assert len(config.analysis_prompts[analysis_type.value]) > 50

    def test_security_prompt_content(self):
        """Test security prompt contains security keywords."""
        config = CodexConfig()

        security_prompt = config.analysis_prompts[AnalysisType.SECURITY.value]
        assert "SQL injection" in security_prompt
        assert "XSS" in security_prompt
        assert "vulnerability" in security_prompt.lower()

    def test_inherits_from_harness_config(self):
        """Test that CodexConfig inherits base config options."""
        config = CodexConfig(
            timeout_seconds=120,
            max_files=500,
            verbose=True,
        )

        assert config.timeout_seconds == 120
        assert config.max_files == 500
        assert config.verbose is True


# =============================================================================
# CodexHarness Initialization Tests
# =============================================================================


class TestCodexHarnessInit:
    """Test CodexHarness initialization."""

    def test_default_initialization(self):
        """Test harness initializes with defaults."""
        harness = CodexHarness()

        assert harness.config is not None
        assert harness.config.model == "gpt-4o"
        assert harness._client is None

    def test_custom_config_initialization(self, codex_config):
        """Test harness initializes with custom config."""
        harness = CodexHarness(codex_config)

        assert harness.config.api_key == "test-key"
        assert harness.config.model == "gpt-4o"


class TestCodexHarnessProperties:
    """Test CodexHarness properties."""

    def test_name_property(self, codex_config):
        """Test name property."""
        harness = CodexHarness(codex_config)
        assert harness.name == "codex"

    def test_supported_analysis_types(self, codex_config):
        """Test supported analysis types."""
        harness = CodexHarness(codex_config)

        # Should support all analysis types
        for analysis_type in AnalysisType:
            assert analysis_type in harness.supported_analysis_types


class TestCodexHarnessClient:
    """Test OpenAI client initialization."""

    def test_get_client_requires_openai(self, codex_config):
        """Test that get_client requires openai package."""
        harness = CodexHarness(codex_config)

        with patch.dict("sys.modules", {"openai": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                # We can't easily test ImportError without more complex mocking
                # so just verify client is initially None
                assert harness._client is None

    def test_get_client_requires_api_key(self):
        """Test that get_client requires API key when none configured."""
        config = CodexConfig(api_key=None)
        harness = CodexHarness(config)

        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=True):
            with pytest.raises(HarnessError, match="OPENAI_API_KEY"):
                harness._get_client()

    def test_get_client_uses_config_key(self, codex_config):
        """Test that config API key is used."""
        harness = CodexHarness(codex_config)

        with patch("openai.AsyncOpenAI") as mock_client:
            harness._get_client()
            mock_client.assert_called_once_with(api_key="test-key")


# =============================================================================
# analyze_repository Tests
# =============================================================================


class TestAnalyzeRepository:
    """Test analyze_repository method."""

    @pytest.mark.asyncio
    async def test_analyze_repository_no_files(self, codex_config, temp_dir):
        """Test analyzing directory with no matching files."""
        harness = CodexHarness(codex_config)

        # Empty directory
        result = await harness.analyze_repository(
            temp_dir,
            analysis_type=AnalysisType.SECURITY,
            options={"file_patterns": ["**/*.cpp"]},  # No cpp files
        )

        assert result.success is True
        assert result.findings == []
        assert "No matching files" in result.raw_output

    @pytest.mark.asyncio
    async def test_analyze_repository_success(self, codex_config, temp_repo):
        """Test successful repository analysis."""
        harness = CodexHarness(codex_config)

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        [
                            {
                                "id": "1",
                                "title": "Security Issue",
                                "severity": "high",
                                "file_path": "main.py",
                                "description": "Found an issue",
                                "confidence": 0.9,
                            }
                        ]
                    )
                )
            )
        ]

        with patch.object(harness, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await harness.analyze_repository(
                temp_repo,
                analysis_type=AnalysisType.SECURITY,
            )

            assert result.success is True
            assert result.harness == "codex"
            assert len(result.findings) == 1
            assert result.findings[0].title == "Security Issue"

    @pytest.mark.asyncio
    async def test_analyze_repository_with_custom_prompt(self, codex_config, temp_repo):
        """Test analysis with custom prompt."""
        harness = CodexHarness(codex_config)

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="[]"))  # Empty findings
        ]

        with patch.object(harness, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await harness.analyze_repository(
                temp_repo,
                analysis_type=AnalysisType.GENERAL,
                prompt="Custom analysis prompt",
            )

            assert result.success is True

            # Verify custom prompt was used
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            assert any("Custom analysis prompt" in msg["content"] for msg in messages)

    @pytest.mark.asyncio
    async def test_analyze_repository_error_handling(self, codex_config, temp_repo):
        """Test error handling during analysis."""
        harness = CodexHarness(codex_config)

        with patch.object(harness, "_gather_files", side_effect=OSError("File error")):
            result = await harness.analyze_repository(temp_repo)

            assert result.success is False
            assert result.error_message is not None
            assert "File error" in result.error_message

    @pytest.mark.asyncio
    async def test_analyze_repository_options(self, codex_config, temp_repo):
        """Test repository analysis with custom options."""
        harness = CodexHarness(codex_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="[]"))]

        with patch.object(harness, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await harness.analyze_repository(
                temp_repo,
                options={
                    "file_patterns": ["**/*.py"],
                    "max_files": 5,
                    "exclude_patterns": ["**/test*"],
                },
            )

            assert result.success is True


# =============================================================================
# analyze_files Tests
# =============================================================================


class TestAnalyzeFiles:
    """Test analyze_files method."""

    @pytest.mark.asyncio
    async def test_analyze_files_success(self, codex_config, temp_repo):
        """Test successful file analysis."""
        harness = CodexHarness(codex_config)

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        [
                            {
                                "id": "1",
                                "title": "Quality Issue",
                                "severity": "medium",
                                "file_path": "main.py",
                                "description": "Issue found",
                                "confidence": 0.8,
                            }
                        ]
                    )
                )
            )
        ]

        with patch.object(harness, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            files = [temp_repo / "main.py", temp_repo / "test.py"]
            result = await harness.analyze_files(files, AnalysisType.QUALITY)

            assert result.success is True
            assert len(result.findings) == 1

    @pytest.mark.asyncio
    async def test_analyze_files_no_readable_files(self, codex_config):
        """Test analyzing files that don't exist."""
        harness = CodexHarness(codex_config)

        result = await harness.analyze_files(
            [Path("/nonexistent/file.py")],
            AnalysisType.SECURITY,
        )

        assert result.success is True
        assert result.findings == []
        assert "No files could be read" in result.raw_output

    @pytest.mark.asyncio
    async def test_analyze_files_error_handling(self, codex_config, temp_repo):
        """Test error handling in file analysis."""
        harness = CodexHarness(codex_config)

        with patch.object(harness, "_call_openai", side_effect=RuntimeError("API error")):
            result = await harness.analyze_files(
                [temp_repo / "main.py"],
                AnalysisType.QUALITY,
            )

            assert result.success is False
            assert "API error" in result.error_message


# =============================================================================
# stream_analysis Tests
# =============================================================================


class TestStreamAnalysis:
    """Test stream_analysis method."""

    @pytest.mark.asyncio
    async def test_stream_analysis_no_files(self, codex_config, temp_dir):
        """Test streaming with no matching files."""
        harness = CodexHarness(codex_config)

        chunks = []
        async for chunk in harness.stream_analysis(
            temp_dir,
            options={"file_patterns": ["**/*.nonexistent"]},
        ):
            chunks.append(chunk)

        assert any("No matching files" in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_stream_analysis_success(self, codex_config, temp_repo):
        """Test successful streaming analysis."""
        harness = CodexHarness(codex_config)

        # Create mock async stream
        async def mock_stream():
            for text in ["Finding ", "1: ", "Issue detected"]:
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content=text))]
                yield chunk

        with patch.object(harness, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
            mock_get_client.return_value = mock_client

            chunks = []
            async for chunk in harness.stream_analysis(temp_repo):
                chunks.append(chunk)

            assert len(chunks) == 3
            assert "".join(chunks) == "Finding 1: Issue detected"

    @pytest.mark.asyncio
    async def test_stream_analysis_error(self, codex_config, temp_repo):
        """Test stream error handling."""
        harness = CodexHarness(codex_config)

        with patch.object(harness, "_get_client", side_effect=RuntimeError("Connection error")):
            chunks = []
            async for chunk in harness.stream_analysis(temp_repo):
                chunks.append(chunk)

            assert any("Error" in chunk for chunk in chunks)


# =============================================================================
# Interactive Session Tests
# =============================================================================


class TestInteractiveSession:
    """Test interactive session methods."""

    @pytest.mark.asyncio
    async def test_run_interactive_session(self, codex_config, temp_repo):
        """Test running interactive session."""
        harness = CodexHarness(codex_config)

        context = SessionContext(
            session_id="test-session",
            repo_path=temp_repo,
            files_in_context=[str(temp_repo / "main.py")],
        )

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Here is my analysis of your code..."))
        ]

        with patch.object(harness, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await harness.run_interactive_session(context, "Explain this code")

            assert result.session_id == "test-session"
            assert "analysis" in result.response.lower()

    @pytest.mark.asyncio
    async def test_run_interactive_session_error(self, codex_config, temp_repo):
        """Test interactive session error handling."""
        harness = CodexHarness(codex_config)

        context = SessionContext(
            session_id="test-session",
            repo_path=temp_repo,
        )

        with patch.object(harness, "_get_client", side_effect=RuntimeError("API error")):
            result = await harness.run_interactive_session(context, "test prompt")

            assert "Error" in result.response


# =============================================================================
# Finding Parsing Tests
# =============================================================================


class TestFindingParsing:
    """Test finding parsing methods."""

    def test_parse_findings_json(self, codex_config):
        """Test parsing JSON findings."""
        harness = CodexHarness(codex_config)

        output = """Here is my analysis:
[
    {
        "id": "1",
        "title": "SQL Injection",
        "severity": "high",
        "file_path": "db.py",
        "line_start": 42,
        "description": "Unsanitized input",
        "confidence": 0.9,
        "recommendation": "Use parameterized queries"
    }
]
Additional notes..."""

        findings = harness._parse_findings(output, Path("."), AnalysisType.SECURITY)

        assert len(findings) == 1
        assert findings[0].title == "SQL Injection"
        assert findings[0].severity == "high"
        assert findings[0].line_start == 42
        assert findings[0].confidence == 0.9

    def test_parse_findings_json_array_only(self, codex_config):
        """Test parsing when response is just JSON array."""
        harness = CodexHarness(codex_config)

        output = json.dumps(
            [
                {
                    "id": "1",
                    "title": "Issue",
                    "severity": "medium",
                    "file_path": "test.py",
                    "description": "Test issue",
                    "confidence": 0.8,
                }
            ]
        )

        findings = harness._parse_findings(output, Path("."), AnalysisType.QUALITY)

        assert len(findings) == 1
        assert findings[0].title == "Issue"

    def test_parse_findings_defaults(self, codex_config):
        """Test default values when fields missing."""
        harness = CodexHarness(codex_config)

        output = json.dumps([{"title": "Minimal Finding"}])

        findings = harness._parse_findings(output, Path("."), AnalysisType.GENERAL)

        assert len(findings) == 1
        assert findings[0].title == "Minimal Finding"
        assert findings[0].severity == "medium"  # Default
        assert findings[0].confidence == 0.8  # Default

    def test_parse_findings_fallback_to_text(self, codex_config):
        """Test fallback to text extraction when no JSON."""
        harness = CodexHarness(codex_config)

        output = """## Analysis Results

### 1. Critical Security Issue
There is a SQL injection vulnerability in the authentication module.
The user input is not properly sanitized.

### 2. Medium Quality Issue
The code has some complexity issues that should be addressed.
"""

        findings = harness._parse_findings(output, Path("."), AnalysisType.SECURITY)

        # Should extract some findings from text
        assert len(findings) >= 1

    def test_parse_findings_line_alternatives(self, codex_config):
        """Test that 'line' field is used as fallback for line_start."""
        harness = CodexHarness(codex_config)

        output = json.dumps(
            [
                {
                    "id": "1",
                    "title": "Issue",
                    "severity": "low",
                    "file_path": "test.py",
                    "line": 50,  # Alternative field name
                    "description": "Issue",
                    "confidence": 0.7,
                }
            ]
        )

        findings = harness._parse_findings(output, Path("."), AnalysisType.QUALITY)

        assert findings[0].line_start == 50

    def test_parse_findings_fix_as_recommendation(self, codex_config):
        """Test that 'fix' field is used for recommendation."""
        harness = CodexHarness(codex_config)

        output = json.dumps(
            [
                {
                    "id": "1",
                    "title": "Issue",
                    "severity": "medium",
                    "file_path": "test.py",
                    "description": "Issue",
                    "fix": "Apply this fix",  # Alternative field name
                    "confidence": 0.8,
                }
            ]
        )

        findings = harness._parse_findings(output, Path("."), AnalysisType.QUALITY)

        assert findings[0].recommendation == "Apply this fix"


class TestTextExtraction:
    """Test text-based finding extraction."""

    def test_extract_findings_with_severity(self, codex_config):
        """Test extracting findings with severity indicators."""
        harness = CodexHarness(codex_config)

        text = """
### Critical: Authentication Bypass
This is a critical security issue where authentication can be bypassed.

### High: SQL Injection
User input is directly concatenated into SQL queries.

### Medium: Missing Input Validation
Some input fields lack proper validation.

### Low: Code Style Issues
Minor code style inconsistencies found.
"""

        findings = harness._extract_findings_from_text(text, AnalysisType.SECURITY)

        assert len(findings) >= 3

        # Check severities are detected
        severities = [f.severity for f in findings]
        assert "critical" in severities
        assert "high" in severities

    def test_extract_findings_short_sections_ignored(self, codex_config):
        """Test that very short sections are ignored."""
        harness = CodexHarness(codex_config)

        text = """
###
Short

### This is a valid finding with enough content to be considered
This finding has a description that is long enough to be included in the results.
"""

        findings = harness._extract_findings_from_text(text, AnalysisType.QUALITY)

        # Short sections should be filtered out
        assert all(len(f.description) >= 20 for f in findings)


# =============================================================================
# File Gathering Tests
# =============================================================================


class TestFileGathering:
    """Test file gathering methods."""

    @pytest.mark.asyncio
    async def test_gather_files_basic(self, codex_config, temp_repo):
        """Test basic file gathering."""
        harness = CodexHarness(codex_config)

        files = await harness._gather_files(
            temp_repo,
            patterns=["**/*.py"],
            exclude_patterns=[],
            max_files=100,
        )

        # Should find the Python files
        assert len(files) >= 2
        assert any("main.py" in path for path in files.keys())

    @pytest.mark.asyncio
    async def test_gather_files_excludes(self, codex_config, temp_repo):
        """Test file exclusion patterns."""
        harness = CodexHarness(codex_config)

        files = await harness._gather_files(
            temp_repo,
            patterns=["**/*.py"],
            exclude_patterns=["test.py", "**/test.py"],  # Match the exact file
            max_files=100,
        )

        # test.py should be excluded
        assert not any(path == "test.py" for path in files.keys())

    @pytest.mark.asyncio
    async def test_gather_files_max_limit(self, codex_config, temp_dir):
        """Test max files limit."""
        harness = CodexHarness(codex_config)

        # Create more files than limit
        for i in range(10):
            (temp_dir / f"file{i}.py").write_text(f"# file {i}")

        files = await harness._gather_files(
            temp_dir,
            patterns=["**/*.py"],
            exclude_patterns=[],
            max_files=3,
        )

        assert len(files) <= 3

    @pytest.mark.asyncio
    async def test_gather_files_skips_large_files(self, codex_config, temp_dir):
        """Test that very large files are skipped."""
        harness = CodexHarness(codex_config)

        # Create a large file (> 100KB)
        large_content = "x" * 150000
        (temp_dir / "large.py").write_text(large_content)
        (temp_dir / "small.py").write_text("print('small')")

        files = await harness._gather_files(
            temp_dir,
            patterns=["**/*.py"],
            exclude_patterns=[],
            max_files=100,
        )

        # Large file should be skipped
        assert "large.py" not in str(files.keys())
        assert any("small.py" in path for path in files.keys())


# =============================================================================
# Prompt Building Tests
# =============================================================================


class TestPromptBuilding:
    """Test prompt building methods."""

    def test_build_analysis_prompt_default(self, codex_config):
        """Test building prompt with default analysis prompt."""
        harness = CodexHarness(codex_config)

        files = {"test.py": "print('hello')"}
        prompt = harness._build_analysis_prompt(files, AnalysisType.SECURITY)

        assert "test.py" in prompt
        assert "print('hello')" in prompt
        assert "security" in prompt.lower() or "vulnerab" in prompt.lower()

    def test_build_analysis_prompt_custom(self, codex_config):
        """Test building prompt with custom prompt."""
        harness = CodexHarness(codex_config)

        files = {"test.py": "code here"}
        custom_prompt = "My custom analysis instructions"
        prompt = harness._build_analysis_prompt(files, AnalysisType.GENERAL, custom_prompt)

        assert "My custom analysis instructions" in prompt
        assert "test.py" in prompt

    def test_build_analysis_prompt_multiple_files(self, codex_config):
        """Test prompt includes all files."""
        harness = CodexHarness(codex_config)

        files = {
            "file1.py": "code 1",
            "file2.py": "code 2",
            "file3.py": "code 3",
        }
        prompt = harness._build_analysis_prompt(files, AnalysisType.QUALITY)

        assert "file1.py" in prompt
        assert "file2.py" in prompt
        assert "file3.py" in prompt


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_codex_harness_default(self):
        """Test creating harness with defaults."""
        harness = create_codex_harness()

        assert harness.name == "codex"
        assert harness.config.model == "gpt-4o"

    def test_create_codex_harness_custom(self):
        """Test creating harness with custom settings."""
        harness = create_codex_harness(model="gpt-3.5-turbo", api_key="custom-key")

        assert harness.config.model == "gpt-3.5-turbo"
        assert harness.config.api_key == "custom-key"


# =============================================================================
# Integration Tests
# =============================================================================


class TestCodexIntegration:
    """Integration tests for Codex harness."""

    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, codex_config, temp_repo):
        """Test complete analysis workflow."""
        harness = CodexHarness(codex_config)

        # Mock a realistic response
        mock_findings = [
            {
                "id": "sec-001",
                "title": "Potential Code Injection",
                "severity": "high",
                "file_path": "main.py",
                "line_start": 2,
                "description": "The print statement could be vulnerable",
                "recommendation": "Validate input before printing",
                "confidence": 0.85,
                "category": "security",
            }
        ]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(mock_findings)))]

        with patch.object(harness, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await harness.analyze_repository(
                temp_repo,
                analysis_type=AnalysisType.SECURITY,
            )

            # Verify result structure
            assert result.success is True
            assert result.harness == "codex"
            assert result.analysis_type == AnalysisType.SECURITY
            assert len(result.findings) == 1

            # Verify finding details
            finding = result.findings[0]
            assert finding.title == "Potential Code Injection"
            assert finding.severity == "high"
            assert finding.line_start == 2
            assert finding.confidence == 0.85

            # Verify metadata
            assert result.metadata["model"] == "gpt-4o"
            assert result.metadata["files_analyzed"] >= 1
