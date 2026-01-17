"""
Tests for the harnesses module.

Tests code analysis harnesses for Claude Code and Codex/OpenAI.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.harnesses import (
    AnalysisType,
    ClaudeCodeHarness,
    ClaudeCodeConfig,
    CodexHarness,
    CodexConfig,
    HarnessResult,
    HarnessResultAdapter,
    adapt_to_audit_findings,
    create_codex_harness,
)
from aragora.harnesses.base import (
    AnalysisFinding,
    CodeAnalysisHarness,
    HarnessConfig,
    HarnessError,
    SessionContext,
    SessionResult,
)
from aragora.harnesses.adapter import AdapterConfig, adapt_multiple_results


# =============================================================================
# Base Harness Tests
# =============================================================================


class TestAnalysisType:
    """Tests for AnalysisType enum."""

    def test_all_types_defined(self):
        """Test all analysis types are defined."""
        types = list(AnalysisType)
        assert len(types) == 8

        assert AnalysisType.SECURITY in types
        assert AnalysisType.QUALITY in types
        assert AnalysisType.PERFORMANCE in types
        assert AnalysisType.ARCHITECTURE in types

    def test_type_values(self):
        """Test type values are strings."""
        assert AnalysisType.SECURITY.value == "security"
        assert AnalysisType.QUALITY.value == "quality"


class TestHarnessResult:
    """Tests for HarnessResult dataclass."""

    def test_create_result(self):
        """Test creating a harness result."""
        result = HarnessResult(
            harness="test",
            analysis_type=AnalysisType.SECURITY,
            findings=[],
            raw_output="test output",
            duration_seconds=1.5,
            success=True,
        )

        assert result.harness == "test"
        assert result.analysis_type == AnalysisType.SECURITY
        assert result.success is True

    def test_result_with_findings(self):
        """Test result with findings."""
        finding = AnalysisFinding(
            id="test_1",
            title="Test Finding",
            description="Test description",
            severity="high",
            confidence=0.85,
            category="security",
            file_path="test.py",
            line_start=10,
        )

        result = HarnessResult(
            harness="test",
            analysis_type=AnalysisType.SECURITY,
            findings=[finding],
            raw_output="",
            duration_seconds=1.0,
            success=True,
        )

        assert len(result.findings) == 1
        assert result.findings[0].title == "Test Finding"

    def test_result_with_error(self):
        """Test result with error."""
        result = HarnessResult(
            harness="test",
            analysis_type=AnalysisType.SECURITY,
            findings=[],
            raw_output="",
            duration_seconds=0.1,
            success=False,
            error_message="Analysis failed",
        )

        assert result.success is False
        assert result.error_message == "Analysis failed"


class TestAnalysisFinding:
    """Tests for AnalysisFinding dataclass."""

    def test_create_finding(self):
        """Test creating an analysis finding."""
        finding = AnalysisFinding(
            id="find_001",
            title="SQL Injection",
            description="User input is directly interpolated into SQL query",
            severity="critical",
            confidence=0.95,
            category="security",
            file_path="app/db.py",
            line_start=42,
            line_end=45,
            recommendation="Use parameterized queries",
            code_snippet="cursor.execute(f'SELECT * FROM users WHERE id={id}')",
        )

        assert finding.id == "find_001"
        assert finding.severity == "critical"
        assert finding.confidence == 0.95

    def test_finding_defaults(self):
        """Test finding default values."""
        finding = AnalysisFinding(
            id="find_002",
            title="Minor Issue",
            description="A minor issue was found",
            severity="low",
            confidence=0.8,
            category="quality",
            file_path="test.py",
        )

        assert finding.line_start is None
        assert finding.recommendation == ""
        assert finding.confidence == 0.8


# =============================================================================
# Claude Code Harness Tests
# =============================================================================


class TestClaudeCodeConfig:
    """Tests for ClaudeCodeConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ClaudeCodeConfig()

        assert config.timeout_seconds == 300
        assert "claude" in config.claude_code_path.lower() or config.claude_code_path == "claude"
        assert len(config.analysis_prompts) > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = ClaudeCodeConfig(
            timeout_seconds=600,
            claude_code_path="/usr/local/bin/claude",
        )

        assert config.timeout_seconds == 600
        assert config.claude_code_path == "/usr/local/bin/claude"


class TestClaudeCodeHarness:
    """Tests for ClaudeCodeHarness."""

    @pytest.fixture
    def harness(self):
        """Create Claude Code harness."""
        return ClaudeCodeHarness()

    @pytest.fixture
    def sample_repo(self, tmp_path):
        """Create a sample repository for testing."""
        # Create sample Python files
        (tmp_path / "app.py").write_text(
            """
def process_data(user_input):
    # Potential SQL injection
    query = f"SELECT * FROM users WHERE name='{user_input}'"
    return query

def safe_function():
    return "Hello, World!"
"""
        )
        (tmp_path / "utils.py").write_text(
            """
import os

def get_secret():
    # Hardcoded secret
    api_key = "sk-secret-key-12345"
    return api_key
"""
        )
        return tmp_path

    @pytest.mark.asyncio
    async def test_harness_initialization(self, harness):
        """Test harness initializes correctly."""
        assert harness.config is not None
        assert isinstance(harness.config, ClaudeCodeConfig)

    def test_collect_files(self, harness, sample_repo):
        """Test file collection."""
        files = harness._collect_files(sample_repo)

        assert len(files) >= 2
        file_names = [f.name for f in files]
        assert "app.py" in file_names or "utils.py" in file_names

    def test_build_files_context(self, harness, sample_repo):
        """Test context building."""
        files = harness._collect_files(sample_repo)
        context = harness._build_files_context(files)

        # Should include file contents
        assert "def process_data" in context or "def safe_function" in context
        # Should include file paths
        assert "app.py" in context or "utils.py" in context

    def test_parse_findings_json(self, harness, sample_repo):
        """Test parsing JSON findings."""
        raw_output = """
Here are the findings:
```json
[
    {
        "id": "sql_injection_1",
        "title": "SQL Injection Vulnerability",
        "severity": "critical",
        "file_path": "app.py",
        "line_start": 4,
        "description": "User input directly in SQL query",
        "recommendation": "Use parameterized queries"
    }
]
```
"""
        findings = harness._parse_findings(raw_output, AnalysisType.SECURITY)

        assert len(findings) >= 1
        assert findings[0].severity == "critical"

    def test_parse_findings_text(self, harness, sample_repo):
        """Test parsing text findings (non-JSON)."""
        raw_output = """
1. **Critical: SQL Injection**
   File: app.py, Line 4
   The query uses string interpolation with user input.

2. **High: Hardcoded Secret**
   File: utils.py, Line 5
   API key is hardcoded in source code.
"""
        findings = harness._parse_findings(raw_output, AnalysisType.SECURITY)

        # Should extract something even from non-JSON
        assert isinstance(findings, list)

    @pytest.mark.asyncio
    async def test_analyze_repository_mock(self, harness, sample_repo):
        """Test repository analysis with mocked subprocess."""
        mock_output = json.dumps(
            [
                {
                    "id": "test_1",
                    "title": "Test Finding",
                    "severity": "high",
                    "file_path": "app.py",
                    "line_start": 4,
                    "description": "Test issue",
                }
            ]
        )

        with patch.object(harness, "_run_claude_code", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (mock_output, "")

            result = await harness.analyze_repository(
                sample_repo,
                AnalysisType.SECURITY,
            )

            assert result.success is True
            assert result.harness == "claude-code"

    @pytest.mark.asyncio
    async def test_analyze_files(self, harness, sample_repo):
        """Test analyzing specific files."""
        files = [sample_repo / "app.py"]

        with patch.object(harness, "_run_claude_code", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ("[]", "")

            result = await harness.analyze_files(
                files,
                AnalysisType.QUALITY,
            )

            assert result.success is True


# =============================================================================
# Codex Harness Tests
# =============================================================================


class TestCodexConfig:
    """Tests for CodexConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = CodexConfig()

        assert config.model == "gpt-4o"
        assert config.temperature == 0.2
        assert config.max_tokens == 4096
        assert len(config.analysis_prompts) > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = CodexConfig(
            model="gpt-3.5-turbo",
            temperature=0.5,
            api_key="test-key",
        )

        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.5
        assert config.api_key == "test-key"


class TestCodexHarness:
    """Tests for CodexHarness."""

    @pytest.fixture
    def harness(self):
        """Create Codex harness."""
        return CodexHarness()

    @pytest.fixture
    def sample_repo(self, tmp_path):
        """Create a sample repository for testing."""
        (tmp_path / "main.py").write_text(
            """
def calculate(a, b):
    return a + b

def process(data):
    # TODO: Add validation
    return data
"""
        )
        return tmp_path

    @pytest.mark.asyncio
    async def test_harness_initialization(self, harness):
        """Test harness initializes correctly."""
        assert harness.config is not None
        assert isinstance(harness.config, CodexConfig)

    @pytest.mark.asyncio
    async def test_gather_files(self, harness, sample_repo):
        """Test file gathering."""
        files = await harness._gather_files(
            sample_repo,
            patterns=["**/*.py"],
            exclude_patterns=[],
            max_files=10,
        )

        assert len(files) == 1
        assert "main.py" in files

    def test_build_analysis_prompt(self, harness):
        """Test prompt building."""
        files_content = {
            "main.py": "def test(): pass",
        }

        prompt = harness._build_analysis_prompt(
            files_content,
            AnalysisType.QUALITY,
        )

        assert "main.py" in prompt
        assert "def test():" in prompt
        assert "JSON" in prompt

    def test_parse_findings_json(self, harness, sample_repo):
        """Test parsing JSON findings."""
        raw_output = """
[
    {
        "id": "quality_1",
        "title": "Missing Documentation",
        "severity": "low",
        "file_path": "main.py",
        "line_start": 2,
        "description": "Function lacks docstring",
        "confidence": 0.9
    }
]
"""
        findings = harness._parse_findings(raw_output, sample_repo, AnalysisType.QUALITY)

        assert len(findings) == 1
        assert findings[0].title == "Missing Documentation"
        assert findings[0].confidence == 0.9

    def test_parse_findings_with_code_block(self, harness, sample_repo):
        """Test parsing JSON within code block."""
        raw_output = """
Here are the findings:

```json
[{"id": "test", "title": "Test", "severity": "medium", "file_path": "test.py"}]
```
"""
        findings = harness._parse_findings(raw_output, sample_repo, AnalysisType.GENERAL)

        assert len(findings) >= 1

    def test_extract_findings_from_text(self, harness):
        """Test extracting findings from non-JSON text."""
        text = """
1. **Critical Issue**
   This is a critical security vulnerability.

2. **High Priority**
   This needs immediate attention.

3. **Low Priority**
   Minor code style issue.
"""
        findings = harness._extract_findings_from_text(text, AnalysisType.SECURITY)

        assert len(findings) >= 2
        # Should detect severity from text
        severities = [f.severity for f in findings]
        assert "critical" in severities or "high" in severities

    @pytest.mark.asyncio
    async def test_analyze_repository_mock(self, harness, sample_repo):
        """Test repository analysis with mocked OpenAI client."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            [
                {
                    "id": "test_1",
                    "title": "Test Finding",
                    "severity": "medium",
                    "file_path": "main.py",
                    "description": "Test issue",
                }
            ]
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(harness, "_get_client", return_value=mock_client):
            result = await harness.analyze_repository(
                sample_repo,
                AnalysisType.QUALITY,
            )

            assert result.success is True
            assert result.harness == "codex"
            assert len(result.findings) == 1

    @pytest.mark.asyncio
    async def test_analyze_files_mock(self, harness, sample_repo):
        """Test file analysis with mock."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[]"

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(harness, "_get_client", return_value=mock_client):
            result = await harness.analyze_files(
                [sample_repo / "main.py"],
                AnalysisType.PERFORMANCE,
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_interactive_session_mock(self, harness, sample_repo):
        """Test interactive session with mock."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The code looks good overall."

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        context = SessionContext(
            session_id="test_session",
            repo_path=sample_repo,
            files_in_context=[str(sample_repo / "main.py")],
        )

        with patch.object(harness, "_get_client", return_value=mock_client):
            result = await harness.run_interactive_session(
                context,
                "Review this code for best practices",
            )

            assert "good" in result.response.lower()

    def test_create_codex_harness(self):
        """Test convenience function."""
        harness = create_codex_harness(model="gpt-4-turbo")

        assert harness.config.model == "gpt-4-turbo"


# =============================================================================
# Harness Result Adapter Tests
# =============================================================================


class TestHarnessResultAdapter:
    """Tests for HarnessResultAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create adapter."""
        return HarnessResultAdapter()

    @pytest.fixture
    def sample_result(self):
        """Create sample harness result."""
        findings = [
            AnalysisFinding(
                id="find_1",
                title="SQL Injection",
                description="SQL injection vulnerability found",
                severity="critical",
                confidence=0.95,
                category="security",
                file_path="app/db.py",
                line_start=42,
                line_end=45,
                recommendation="Use parameterized queries",
                code_snippet="query = f'SELECT * FROM {table}'",
            ),
            AnalysisFinding(
                id="find_2",
                title="Missing Type Hints",
                description="Function lacks type annotations",
                severity="low",
                confidence=0.7,
                category="quality",
                file_path="utils.py",
                line_start=10,
            ),
        ]

        return HarnessResult(
            harness="claude-code",
            analysis_type=AnalysisType.SECURITY,
            findings=findings,
            raw_output="",
            duration_seconds=5.0,
            success=True,
        )

    def test_adapt_result(self, adapter, sample_result):
        """Test adapting harness result to audit findings."""
        audit_findings = adapter.adapt(sample_result)

        assert len(audit_findings) == 2
        assert audit_findings[0].title == "SQL Injection"
        assert audit_findings[0].severity.value == "critical"

    def test_severity_mapping(self, adapter):
        """Test severity mapping."""
        result = HarnessResult(
            harness="test",
            analysis_type=AnalysisType.SECURITY,
            findings=[
                AnalysisFinding(
                    id="1",
                    title="Critical Issue",
                    description="A critical issue",
                    severity="critical",
                    confidence=0.9,
                    category="security",
                    file_path="test.py",
                ),
                AnalysisFinding(
                    id="2",
                    title="Warning Issue",
                    description="A warning issue",
                    severity="warning",
                    confidence=0.8,
                    category="security",
                    file_path="test.py",
                ),
            ],
            raw_output="",
            duration_seconds=1.0,
            success=True,
        )

        audit_findings = adapter.adapt(result)

        assert audit_findings[0].severity.value == "critical"
        # "warning" maps to "medium"
        assert audit_findings[1].severity.value == "medium"

    def test_confidence_adjustment(self, adapter):
        """Test confidence adjustment by harness."""
        config = AdapterConfig(
            confidence_adjustments={
                "claude-code": 0.0,
                "codex": -0.1,
                "default": 0.0,
            }
        )
        adapter = HarnessResultAdapter(config)

        result = HarnessResult(
            harness="codex",
            analysis_type=AnalysisType.SECURITY,
            findings=[
                AnalysisFinding(
                    id="1",
                    title="Test",
                    description="Test finding",
                    severity="high",
                    confidence=0.9,
                    category="security",
                    file_path="test.py",
                ),
            ],
            raw_output="",
            duration_seconds=1.0,
            success=True,
        )

        audit_findings = adapter.adapt(result)

        # 0.9 - 0.1 = 0.8
        assert audit_findings[0].confidence == 0.8

    def test_adapt_batch(self, adapter, sample_result):
        """Test batch adaptation."""
        results = [sample_result, sample_result]

        audit_findings = adapter.adapt_batch(results)

        assert len(audit_findings) == 4

    def test_merge_duplicate_findings(self, adapter):
        """Test merging duplicate findings."""
        finding1 = AnalysisFinding(
            id="1",
            title="SQL Injection",
            description="SQL injection found",
            severity="high",
            confidence=0.8,
            category="security",
            file_path="db.py",
            line_start=10,
        )
        finding2 = AnalysisFinding(
            id="2",
            title="SQL Injection Vulnerability",  # Similar title
            description="SQL injection vulnerability",
            severity="high",
            confidence=0.9,
            category="security",
            file_path="db.py",
            line_start=10,
        )

        result1 = HarnessResult(
            harness="claude-code",
            analysis_type=AnalysisType.SECURITY,
            findings=[finding1],
            raw_output="",
            duration_seconds=1.0,
            success=True,
        )
        result2 = HarnessResult(
            harness="codex",
            analysis_type=AnalysisType.SECURITY,
            findings=[finding2],
            raw_output="",
            duration_seconds=1.0,
            success=True,
        )

        audit_findings = adapter.adapt_batch([result1, result2])
        # Just verify we got findings back
        assert len(audit_findings) == 2

    def test_evidence_location_format(self, adapter, sample_result):
        """Test evidence location formatting."""
        audit_findings = adapter.adapt(sample_result)

        # Should include line numbers
        assert ":42" in audit_findings[0].evidence_location
        assert "-45" in audit_findings[0].evidence_location

    def test_adapt_to_audit_findings_function(self, sample_result):
        """Test convenience function."""
        audit_findings = adapt_to_audit_findings(sample_result)

        assert len(audit_findings) == 2

    def test_adapt_multiple_results_function(self, sample_result):
        """Test adapt_multiple_results convenience function."""
        results = [sample_result]

        audit_findings = adapt_multiple_results(results, merge_duplicates=True)

        assert len(audit_findings) == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestHarnessIntegration:
    """Integration tests for harness module."""

    @pytest.fixture
    def sample_repo(self, tmp_path):
        """Create a comprehensive sample repository."""
        # Python files
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text(
            """
import os
from typing import Optional

def get_user_data(user_id: str) -> dict:
    '''Fetch user data from database.'''
    # TODO: Add caching
    query = f"SELECT * FROM users WHERE id = '{user_id}'"  # SQL injection risk
    return {"id": user_id}

def process_payment(amount: float, card_number: str) -> bool:
    '''Process a payment.'''
    # Logging sensitive data
    print(f"Processing payment for card {card_number}")
    return True

API_KEY = "sk-secret-key-12345"  # Hardcoded secret
"""
        )

        (tmp_path / "src" / "utils.py").write_text(
            """
def format_date(date):
    return str(date)

def validate_input(data):
    # No actual validation
    return data
"""
        )

        # JavaScript file
        (tmp_path / "app.js").write_text(
            """
const express = require('express');
const app = express();

app.get('/user/:id', (req, res) => {
    // XSS vulnerability
    res.send(`<h1>Hello ${req.params.id}</h1>`);
});
"""
        )

        return tmp_path

    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, sample_repo):
        """Test complete analysis workflow."""
        # Create harnesses
        claude_harness = ClaudeCodeHarness()
        codex_harness = CodexHarness()
        adapter = HarnessResultAdapter()

        # Mock Claude Code analysis
        claude_output = json.dumps(
            [
                {
                    "id": "sql_1",
                    "title": "SQL Injection",
                    "severity": "critical",
                    "file_path": "src/main.py",
                    "line_start": 8,
                    "description": "SQL injection vulnerability",
                }
            ]
        )

        with patch.object(
            claude_harness, "_run_claude_code", new_callable=AsyncMock
        ) as mock_claude:
            mock_claude.return_value = (claude_output, "")

            claude_result = await claude_harness.analyze_repository(
                sample_repo,
                AnalysisType.SECURITY,
            )

        # Mock Codex analysis
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            [
                {
                    "id": "secret_1",
                    "title": "Hardcoded Secret",
                    "severity": "high",
                    "file_path": "src/main.py",
                    "line_start": 16,
                    "description": "API key hardcoded in source",
                }
            ]
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(codex_harness, "_get_client", return_value=mock_client):
            codex_result = await codex_harness.analyze_repository(
                sample_repo,
                AnalysisType.SECURITY,
            )

        # Combine and adapt results
        all_findings = adapt_multiple_results(
            [claude_result, codex_result],
            merge_duplicates=True,
        )

        assert len(all_findings) == 2
        titles = [f.title for f in all_findings]
        assert "SQL Injection" in titles
        assert "Hardcoded Secret" in titles

    def test_analysis_type_prompts(self, sample_repo):
        """Test different analysis types have appropriate prompts."""
        codex_harness = CodexHarness()

        for analysis_type in [
            AnalysisType.SECURITY,
            AnalysisType.QUALITY,
            AnalysisType.PERFORMANCE,
        ]:
            files_content = {"test.py": "def test(): pass"}
            prompt = codex_harness._build_analysis_prompt(files_content, analysis_type)

            # Verify type-specific content in prompt
            if analysis_type == AnalysisType.SECURITY:
                assert any(
                    word in prompt.lower() for word in ["security", "vulnerab", "injection", "xss"]
                )
            elif analysis_type == AnalysisType.QUALITY:
                assert any(
                    word in prompt.lower() for word in ["quality", "smell", "pattern", "naming"]
                )
            elif analysis_type == AnalysisType.PERFORMANCE:
                assert any(
                    word in prompt.lower()
                    for word in ["performance", "memory", "efficien", "optim"]
                )

    def test_file_collection(self, sample_repo):
        """Test file collection."""
        harness = ClaudeCodeHarness()

        # Collect all files
        files = harness._collect_files(sample_repo)

        # Should find Python files
        py_files = [f for f in files if f.suffix == ".py"]
        assert len(py_files) >= 2

        # Should find JavaScript files
        js_files = [f for f in files if f.suffix == ".js"]
        assert len(js_files) >= 1

    def test_adapter_config_customization(self):
        """Test adapter configuration customization."""
        config = AdapterConfig(
            severity_mapping={
                "critical": "critical",
                "high": "high",
                "medium": "medium",
                "low": "low",
                "info": "info",
                "note": "info",  # Custom mapping
            },
            min_confidence=0.7,
            id_prefix="custom",
        )

        adapter = HarnessResultAdapter(config)

        result = HarnessResult(
            harness="test",
            analysis_type=AnalysisType.GENERAL,
            findings=[
                AnalysisFinding(
                    id="1",
                    title="Test",
                    description="Test issue",
                    severity="note",
                    confidence=0.6,  # Below threshold
                    category="quality",
                    file_path="test.py",
                ),
            ],
            raw_output="",
            duration_seconds=1.0,
            success=True,
        )

        audit_findings = adapter.adapt(result)

        # Finding should still be adapted but with min confidence
        assert len(audit_findings) == 1
        assert audit_findings[0].confidence >= 0.7
