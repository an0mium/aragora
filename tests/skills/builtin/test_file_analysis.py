"""
Tests for aragora.skills.builtin.file_analysis module.

Covers:
- FileAnalysisSkill manifest and initialization
- Basic file metrics
- Code analysis (functions, classes, complexity)
- Language detection
- Top words extraction
- Structure extraction
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.skills.base import SkillCapability, SkillContext
from aragora.skills.builtin.file_analysis import (
    FileAnalysisSkill,
    FileMetrics,
    LANGUAGE_MAP,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def skill() -> FileAnalysisSkill:
    """Create a file analysis skill for testing."""
    return FileAnalysisSkill()


@pytest.fixture
def context() -> SkillContext:
    """Create a context for testing."""
    return SkillContext(user_id="user123", permissions=["files:read"])


@pytest.fixture
def sample_python_code() -> str:
    """Sample Python code for testing."""
    return '''
"""Module docstring."""

import os
import sys

# This is a comment
class MyClass:
    """Class docstring."""

    def __init__(self, value):
        self.value = value

    def get_value(self):
        """Return the value."""
        return self.value


def helper_function(x, y):
    """Add two numbers."""
    if x > 0:
        return x + y
    else:
        return y


def another_function():
    """Do something."""
    for i in range(10):
        print(i)
'''


@pytest.fixture
def sample_javascript_code() -> str:
    """Sample JavaScript code for testing."""
    return """
// Module comment
import { Component } from 'react';

class MyComponent extends Component {
    constructor(props) {
        super(props);
        this.state = {};
    }

    render() {
        return <div>Hello</div>;
    }
}

function helperFunction(x, y) {
    if (x > 0) {
        return x + y;
    }
    return y;
}

const arrowFunc = (a) => a * 2;
"""


# =============================================================================
# FileMetrics Tests
# =============================================================================


class TestFileMetrics:
    """Tests for FileMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating file metrics."""
        metrics = FileMetrics(
            file_path="/path/to/file.py",
            file_name="file.py",
            extension=".py",
            size_bytes=1024,
            line_count=50,
            word_count=200,
            char_count=2000,
        )

        assert metrics.file_name == "file.py"
        assert metrics.size_bytes == 1024
        assert metrics.line_count == 50

    def test_metrics_defaults(self):
        """Test metrics default values."""
        metrics = FileMetrics(
            file_path="",
            file_name="test.txt",
            extension=".txt",
            size_bytes=100,
            line_count=10,
            word_count=50,
            char_count=500,
        )

        assert metrics.language is None
        assert metrics.blank_lines == 0
        assert metrics.function_count == 0
        assert metrics.class_count == 0

    def test_to_dict(self):
        """Test converting metrics to dict."""
        metrics = FileMetrics(
            file_path="/test/file.py",
            file_name="file.py",
            extension=".py",
            size_bytes=2048,
            line_count=100,
            word_count=500,
            char_count=5000,
            language="python",
        )

        data = metrics.to_dict()

        assert data["file_name"] == "file.py"
        assert data["language"] == "python"
        assert "size_readable" in data

    def test_format_size(self):
        """Test size formatting."""
        metrics = FileMetrics(
            file_path="",
            file_name="test",
            extension="",
            size_bytes=0,
            line_count=0,
            word_count=0,
            char_count=0,
        )

        assert "B" in metrics._format_size(500)
        assert "KB" in metrics._format_size(2048)
        assert "MB" in metrics._format_size(2 * 1024 * 1024)


# =============================================================================
# FileAnalysisSkill Manifest Tests
# =============================================================================


class TestFileAnalysisSkillManifest:
    """Tests for FileAnalysisSkill manifest."""

    def test_manifest_name(self, skill: FileAnalysisSkill):
        """Test manifest name."""
        assert skill.manifest.name == "file_analysis"

    def test_manifest_version(self, skill: FileAnalysisSkill):
        """Test manifest version."""
        assert skill.manifest.version == "1.0.0"

    def test_manifest_capabilities(self, skill: FileAnalysisSkill):
        """Test manifest capabilities."""
        caps = skill.manifest.capabilities
        assert SkillCapability.READ_LOCAL in caps

    def test_manifest_input_schema(self, skill: FileAnalysisSkill):
        """Test manifest input schema."""
        schema = skill.manifest.input_schema

        assert "content" in schema
        assert "file_path" in schema
        assert "file_name" in schema
        assert "analyze_code" in schema

    def test_manifest_requires_permission(self, skill: FileAnalysisSkill):
        """Test manifest requires file read permission."""
        assert "files:read" in skill.manifest.required_permissions


# =============================================================================
# Language Detection Tests
# =============================================================================


class TestLanguageDetection:
    """Tests for language detection via extension."""

    def test_python_detection(self):
        """Test Python detection."""
        assert LANGUAGE_MAP.get(".py") == "python"

    def test_javascript_detection(self):
        """Test JavaScript detection."""
        assert LANGUAGE_MAP.get(".js") == "javascript"

    def test_typescript_detection(self):
        """Test TypeScript detection."""
        assert LANGUAGE_MAP.get(".ts") == "typescript"

    def test_java_detection(self):
        """Test Java detection."""
        assert LANGUAGE_MAP.get(".java") == "java"

    def test_go_detection(self):
        """Test Go detection."""
        assert LANGUAGE_MAP.get(".go") == "go"

    def test_rust_detection(self):
        """Test Rust detection."""
        assert LANGUAGE_MAP.get(".rs") == "rust"

    def test_yaml_detection(self):
        """Test YAML detection."""
        assert LANGUAGE_MAP.get(".yaml") == "yaml"
        assert LANGUAGE_MAP.get(".yml") == "yaml"


# =============================================================================
# Code Analysis Tests
# =============================================================================


class TestCodeAnalysis:
    """Tests for code analysis."""

    def test_analyze_python_functions(self, skill: FileAnalysisSkill, sample_python_code: str):
        """Test Python function counting."""
        lines = sample_python_code.split("\n")
        metrics = FileMetrics(
            file_path="",
            file_name="test.py",
            extension=".py",
            size_bytes=len(sample_python_code),
            line_count=len(lines),
            word_count=len(sample_python_code.split()),
            char_count=len(sample_python_code),
        )

        skill._analyze_code(sample_python_code, lines, metrics, "python")

        # Should find: __init__, get_value, helper_function, another_function
        assert metrics.function_count >= 3

    def test_analyze_python_classes(self, skill: FileAnalysisSkill, sample_python_code: str):
        """Test Python class counting."""
        lines = sample_python_code.split("\n")
        metrics = FileMetrics(
            file_path="",
            file_name="test.py",
            extension=".py",
            size_bytes=len(sample_python_code),
            line_count=len(lines),
            word_count=len(sample_python_code.split()),
            char_count=len(sample_python_code),
        )

        skill._analyze_code(sample_python_code, lines, metrics, "python")

        assert metrics.class_count >= 1

    def test_analyze_python_imports(self, skill: FileAnalysisSkill, sample_python_code: str):
        """Test Python import counting."""
        lines = sample_python_code.split("\n")
        metrics = FileMetrics(
            file_path="",
            file_name="test.py",
            extension=".py",
            size_bytes=len(sample_python_code),
            line_count=len(lines),
            word_count=len(sample_python_code.split()),
            char_count=len(sample_python_code),
        )

        skill._analyze_code(sample_python_code, lines, metrics, "python")

        assert metrics.import_count >= 2  # os and sys

    def test_analyze_line_types(self, skill: FileAnalysisSkill, sample_python_code: str):
        """Test line type counting (blank, comment, code)."""
        lines = sample_python_code.split("\n")
        metrics = FileMetrics(
            file_path="",
            file_name="test.py",
            extension=".py",
            size_bytes=len(sample_python_code),
            line_count=len(lines),
            word_count=len(sample_python_code.split()),
            char_count=len(sample_python_code),
        )

        skill._analyze_code(sample_python_code, lines, metrics, "python")

        assert metrics.blank_lines > 0
        assert metrics.comment_lines >= 1
        assert metrics.code_lines > 0
        # Total should equal line count
        total = metrics.blank_lines + metrics.comment_lines + metrics.code_lines
        assert total == len(lines)

    def test_analyze_javascript(self, skill: FileAnalysisSkill, sample_javascript_code: str):
        """Test JavaScript analysis."""
        lines = sample_javascript_code.split("\n")
        metrics = FileMetrics(
            file_path="",
            file_name="test.js",
            extension=".js",
            size_bytes=len(sample_javascript_code),
            line_count=len(lines),
            word_count=len(sample_javascript_code.split()),
            char_count=len(sample_javascript_code),
        )

        skill._analyze_code(sample_javascript_code, lines, metrics, "javascript")

        assert metrics.class_count >= 1  # MyComponent
        assert metrics.function_count >= 1


# =============================================================================
# Top Words Tests
# =============================================================================


class TestTopWords:
    """Tests for top words extraction."""

    def test_get_top_words_basic(self, skill: FileAnalysisSkill):
        """Test basic top words extraction."""
        text = "word word word test test example"
        result = skill._get_top_words(text)

        assert len(result) > 0
        # word should be most common
        assert result[0][0] == "word"
        assert result[0][1] == 3

    def test_get_top_words_filters_stop_words(self, skill: FileAnalysisSkill):
        """Test that stop words are filtered."""
        text = "the quick brown fox jumps over the lazy dog"
        result = skill._get_top_words(text)

        # Should not contain "the"
        words = [w for w, _ in result]
        assert "the" not in words

    def test_get_top_words_filters_short_words(self, skill: FileAnalysisSkill):
        """Test that short words are filtered."""
        text = "a b c longer words here"
        result = skill._get_top_words(text)

        # Should not contain single characters
        words = [w for w, _ in result]
        assert "a" not in words
        assert "b" not in words


# =============================================================================
# Structure Extraction Tests
# =============================================================================


class TestStructureExtraction:
    """Tests for structure extraction."""

    def test_extract_python_structure(self, skill: FileAnalysisSkill, sample_python_code: str):
        """Test Python structure extraction."""
        structure = skill._extract_structure(sample_python_code, "python")

        assert "functions" in structure
        assert "classes" in structure
        assert len(structure["functions"]) >= 2
        assert len(structure["classes"]) >= 1

    def test_extract_javascript_structure(
        self, skill: FileAnalysisSkill, sample_javascript_code: str
    ):
        """Test JavaScript structure extraction."""
        structure = skill._extract_structure(sample_javascript_code, "javascript")

        assert "functions" in structure
        assert "classes" in structure

    def test_extract_markdown_structure(self, skill: FileAnalysisSkill):
        """Test Markdown structure extraction."""
        markdown = """
# Heading 1
Some content.

## Heading 2
More content.

### Heading 3
Even more content.
"""
        structure = skill._extract_structure(markdown, None)

        assert "sections" in structure
        assert len(structure["sections"]) == 3


# =============================================================================
# Full Execution Tests
# =============================================================================


class TestFileAnalysisExecution:
    """Tests for full skill execution."""

    @pytest.mark.asyncio
    async def test_execute_missing_input(self, skill: FileAnalysisSkill, context: SkillContext):
        """Test execution fails without content or file_path."""
        result = await skill.execute({}, context)

        assert result.success is False
        assert (
            "content" in result.error_message.lower() or "file_path" in result.error_message.lower()
        )

    @pytest.mark.asyncio
    async def test_execute_with_content(
        self, skill: FileAnalysisSkill, context: SkillContext, sample_python_code: str
    ):
        """Test execution with content input."""
        result = await skill.execute(
            {"content": sample_python_code, "file_name": "test.py"},
            context,
        )

        assert result.success is True
        assert "metrics" in result.data
        metrics = result.data["metrics"]
        assert metrics["language"] == "python"
        assert metrics["function_count"] >= 2

    @pytest.mark.asyncio
    async def test_execute_without_code_analysis(
        self, skill: FileAnalysisSkill, context: SkillContext, sample_python_code: str
    ):
        """Test execution without code analysis."""
        result = await skill.execute(
            {
                "content": sample_python_code,
                "file_name": "test.py",
                "analyze_code": False,
            },
            context,
        )

        assert result.success is True
        # Code metrics should not be analyzed
        metrics = result.data["metrics"]
        assert metrics["function_count"] == 0

    @pytest.mark.asyncio
    async def test_execute_with_top_words(
        self, skill: FileAnalysisSkill, context: SkillContext, sample_python_code: str
    ):
        """Test execution with top words."""
        result = await skill.execute(
            {
                "content": sample_python_code,
                "file_name": "test.py",
                "include_top_words": True,
            },
            context,
        )

        assert result.success is True
        metrics = result.data["metrics"]
        assert "top_words" in metrics
        assert len(metrics["top_words"]) > 0

    @pytest.mark.asyncio
    async def test_execute_with_structure(
        self, skill: FileAnalysisSkill, context: SkillContext, sample_python_code: str
    ):
        """Test execution with structure extraction."""
        result = await skill.execute(
            {
                "content": sample_python_code,
                "file_name": "test.py",
                "extract_structure": True,
            },
            context,
        )

        assert result.success is True
        assert "structure" in result.data
        structure = result.data["structure"]
        assert "functions" in structure
        assert "classes" in structure

    @pytest.mark.asyncio
    async def test_execute_plain_text(self, skill: FileAnalysisSkill, context: SkillContext):
        """Test execution with plain text file."""
        text = "This is a plain text file with some words and content."
        result = await skill.execute(
            {"content": text, "file_name": "readme.txt"},
            context,
        )

        assert result.success is True
        metrics = result.data["metrics"]
        assert metrics["language"] is None  # .txt not recognized as code
        assert metrics["word_count"] > 0


# =============================================================================
# SKILLS Registration Tests
# =============================================================================


class TestSkillsRegistration:
    """Tests for SKILLS module-level list."""

    def test_skills_list_exists(self):
        """Test SKILLS list exists in module."""
        from aragora.skills.builtin import file_analysis

        assert hasattr(file_analysis, "SKILLS")

    def test_skills_list_contains_skill(self):
        """Test SKILLS list contains FileAnalysisSkill."""
        from aragora.skills.builtin.file_analysis import SKILLS

        assert len(SKILLS) == 1
        assert isinstance(SKILLS[0], FileAnalysisSkill)
