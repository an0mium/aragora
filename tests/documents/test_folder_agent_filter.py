"""
Tests for agent-based file filtering module.

Tests cover:
- FilterDecision dataclass creation
- AgentFileFilter construction with default and custom config
- _get_file_preview for text and non-text files
- _format_file_for_prompt formatting
- _build_prompt generation
- _parse_response JSON parsing (valid and invalid)
- filter_batch with empty files, empty prompt, batch processing
- filter_files convenience method
- get_agent_filter factory function
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.documents.folder.agent_filter import (
    AgentFileFilter,
    FilterDecision,
    get_agent_filter,
)
from aragora.documents.folder.config import ExcludedFile, ExclusionReason, FileInfo


class TestFilterDecision:
    """Tests for FilterDecision dataclass."""

    def test_creation_minimal(self):
        file_info = FileInfo(
            path="test.py",
            absolute_path="/tmp/test.py",
            size_bytes=100,
            extension=".py",
        )
        decision = FilterDecision(
            file=file_info,
            include=True,
            reason="Relevant file",
        )
        assert decision.file == file_info
        assert decision.include is True
        assert decision.reason == "Relevant file"
        assert decision.confidence == 1.0  # default

    def test_creation_with_confidence(self):
        file_info = FileInfo(
            path="test.py",
            absolute_path="/tmp/test.py",
            size_bytes=100,
            extension=".py",
        )
        decision = FilterDecision(
            file=file_info,
            include=False,
            reason="Not relevant",
            confidence=0.75,
        )
        assert decision.confidence == 0.75

    def test_exclude_decision(self):
        file_info = FileInfo(
            path="test.txt",
            absolute_path="/tmp/test.txt",
            size_bytes=50,
            extension=".txt",
        )
        decision = FilterDecision(
            file=file_info,
            include=False,
            reason="File type not needed",
        )
        assert decision.include is False


class TestAgentFileFilterConstruction:
    """Tests for AgentFileFilter construction."""

    def test_default_config(self):
        filter = AgentFileFilter()
        assert filter.model == "gemini-3.1-pro-preview"
        assert filter.batch_size == AgentFileFilter.DEFAULT_BATCH_SIZE
        assert filter.include_previews is True
        assert filter.max_preview_size == AgentFileFilter.MAX_PREVIEW_SIZE
        assert filter._client is None

    def test_custom_config(self):
        filter = AgentFileFilter(
            model="claude-3-sonnet",
            batch_size=25,
            include_previews=False,
            max_preview_size=200,
        )
        assert filter.model == "claude-3-sonnet"
        assert filter.batch_size == 25
        assert filter.include_previews is False
        assert filter.max_preview_size == 200


class TestAgentFileFilterGetFilePreview:
    """Tests for _get_file_preview method."""

    def test_preview_disabled(self, tmp_path: Path):
        filter = AgentFileFilter(include_previews=False)
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        file_info = FileInfo(
            path="test.py",
            absolute_path=str(test_file),
            size_bytes=15,
            extension=".py",
        )
        assert filter._get_file_preview(file_info) is None

    def test_preview_non_text_extension(self, tmp_path: Path):
        filter = AgentFileFilter(include_previews=True)
        file_info = FileInfo(
            path="image.png",
            absolute_path="/tmp/image.png",
            size_bytes=1000,
            extension=".png",
        )
        assert filter._get_file_preview(file_info) is None

    def test_preview_text_file(self, tmp_path: Path):
        filter = AgentFileFilter(include_previews=True, max_preview_size=100)
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    print('world')")

        file_info = FileInfo(
            path="test.py",
            absolute_path=str(test_file),
            size_bytes=30,
            extension=".py",
        )
        preview = filter._get_file_preview(file_info)
        assert preview is not None
        assert "def hello" in preview

    def test_preview_truncation(self, tmp_path: Path):
        filter = AgentFileFilter(include_previews=True, max_preview_size=10)
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a much longer text that exceeds the limit")

        file_info = FileInfo(
            path="test.txt",
            absolute_path=str(test_file),
            size_bytes=50,
            extension=".txt",
        )
        preview = filter._get_file_preview(file_info)
        assert preview is not None
        assert len(preview) == 13  # 10 chars + "..."
        assert preview.endswith("...")

    def test_preview_nonexistent_file(self):
        filter = AgentFileFilter(include_previews=True)
        file_info = FileInfo(
            path="nonexistent.py",
            absolute_path="/nonexistent/path/test.py",
            size_bytes=0,
            extension=".py",
        )
        assert filter._get_file_preview(file_info) is None

    def test_preview_various_text_extensions(self, tmp_path: Path):
        filter = AgentFileFilter(include_previews=True)

        for ext in [".txt", ".md", ".py", ".js", ".json", ".yaml", ".yml", ".csv"]:
            test_file = tmp_path / f"test{ext}"
            test_file.write_text("content")
            file_info = FileInfo(
                path=f"test{ext}",
                absolute_path=str(test_file),
                size_bytes=7,
                extension=ext,
            )
            preview = filter._get_file_preview(file_info)
            assert preview == "content", f"Failed for extension {ext}"


class TestAgentFileFilterFormatFileForPrompt:
    """Tests for _format_file_for_prompt method."""

    def test_basic_formatting(self):
        filter = AgentFileFilter(include_previews=False)
        file_info = FileInfo(
            path="src/main.py",
            absolute_path="/project/src/main.py",
            size_bytes=1024,
            extension=".py",
        )
        formatted = filter._format_file_for_prompt(file_info, 0)

        assert "1. src/main.py" in formatted
        assert "Extension: .py" in formatted
        assert "Size: 1024 bytes" in formatted
        assert "Preview:" not in formatted

    def test_formatting_with_preview(self, tmp_path: Path):
        filter = AgentFileFilter(include_previews=True, max_preview_size=50)
        test_file = tmp_path / "test.py"
        test_file.write_text("def main():\n    pass")

        file_info = FileInfo(
            path="test.py",
            absolute_path=str(test_file),
            size_bytes=20,
            extension=".py",
        )
        formatted = filter._format_file_for_prompt(file_info, 5)

        assert "6. test.py" in formatted  # index + 1
        assert "Preview:" in formatted


class TestAgentFileFilterBuildPrompt:
    """Tests for _build_prompt method."""

    def test_prompt_structure(self):
        filter = AgentFileFilter(include_previews=False)
        files = [
            FileInfo(path="a.py", absolute_path="/a.py", size_bytes=100, extension=".py"),
            FileInfo(path="b.txt", absolute_path="/b.txt", size_bytes=200, extension=".txt"),
        ]
        prompt = filter._build_prompt(files, "Only Python files")

        assert "USER CRITERIA:" in prompt
        assert "Only Python files" in prompt
        assert "FILES TO EVALUATE:" in prompt
        assert "1. a.py" in prompt
        assert "2. b.txt" in prompt
        assert "JSON" in prompt
        assert "decisions" in prompt


class TestAgentFileFilterParseResponse:
    """Tests for _parse_response method."""

    def test_valid_json_response(self):
        filter = AgentFileFilter()
        files = [
            FileInfo(path="a.py", absolute_path="/a.py", size_bytes=100, extension=".py"),
            FileInfo(path="b.txt", absolute_path="/b.txt", size_bytes=200, extension=".txt"),
        ]
        response = json.dumps(
            {
                "decisions": [
                    {"index": 1, "include": True, "reason": "Python file needed"},
                    {"index": 2, "include": False, "reason": "Text file not needed"},
                ]
            }
        )

        decisions = filter._parse_response(response, files)

        assert len(decisions) == 2
        assert decisions[0].include is True
        assert decisions[0].reason == "Python file needed"
        assert decisions[1].include is False
        assert decisions[1].reason == "Text file not needed"

    def test_json_with_markdown_code_block(self):
        filter = AgentFileFilter()
        files = [
            FileInfo(path="a.py", absolute_path="/a.py", size_bytes=100, extension=".py"),
        ]
        response = """```json
{
  "decisions": [
    {"index": 1, "include": true, "reason": "Included"}
  ]
}
```"""

        decisions = filter._parse_response(response, files)
        assert len(decisions) == 1
        assert decisions[0].include is True

    def test_json_with_generic_code_block(self):
        filter = AgentFileFilter()
        files = [
            FileInfo(path="a.py", absolute_path="/a.py", size_bytes=100, extension=".py"),
        ]
        response = """Here's my analysis:
```
{
  "decisions": [
    {"index": 1, "include": false, "reason": "Excluded"}
  ]
}
```"""

        decisions = filter._parse_response(response, files)
        assert len(decisions) == 1
        assert decisions[0].include is False

    def test_missing_decision_defaults_to_include(self):
        filter = AgentFileFilter()
        files = [
            FileInfo(path="a.py", absolute_path="/a.py", size_bytes=100, extension=".py"),
            FileInfo(path="b.txt", absolute_path="/b.txt", size_bytes=200, extension=".txt"),
        ]
        response = json.dumps(
            {"decisions": [{"index": 1, "include": True, "reason": "Found"}]}  # Missing index 2
        )

        decisions = filter._parse_response(response, files)
        assert len(decisions) == 2
        assert decisions[0].include is True
        assert decisions[1].include is True  # Default
        assert decisions[1].confidence == 0.5

    def test_invalid_json_defaults_to_include_all(self):
        filter = AgentFileFilter()
        files = [
            FileInfo(path="a.py", absolute_path="/a.py", size_bytes=100, extension=".py"),
        ]
        response = "This is not valid JSON"

        decisions = filter._parse_response(response, files)
        assert len(decisions) == 1
        assert decisions[0].include is True
        assert decisions[0].confidence == 0.5
        assert "Parse error" in decisions[0].reason

    def test_missing_decisions_key(self):
        filter = AgentFileFilter()
        files = [
            FileInfo(path="a.py", absolute_path="/a.py", size_bytes=100, extension=".py"),
        ]
        response = json.dumps({"results": []})  # Wrong key

        decisions = filter._parse_response(response, files)
        assert len(decisions) == 1
        assert decisions[0].include is True
        assert "Parse error" in decisions[0].reason

    def test_decision_with_confidence(self):
        filter = AgentFileFilter()
        files = [
            FileInfo(path="a.py", absolute_path="/a.py", size_bytes=100, extension=".py"),
        ]
        response = json.dumps(
            {"decisions": [{"index": 1, "include": True, "reason": "Sure", "confidence": 0.9}]}
        )

        decisions = filter._parse_response(response, files)
        assert decisions[0].confidence == 0.9


class TestAgentFileFilterFilterBatch:
    """Tests for filter_batch method."""

    @pytest.mark.asyncio
    async def test_empty_files(self):
        filter = AgentFileFilter()
        decisions = await filter.filter_batch([], "Any criteria")
        assert decisions == []

    @pytest.mark.asyncio
    async def test_empty_prompt(self):
        files = [
            FileInfo(path="a.py", absolute_path="/a.py", size_bytes=100, extension=".py"),
            FileInfo(path="b.txt", absolute_path="/b.txt", size_bytes=200, extension=".txt"),
        ]
        filter = AgentFileFilter()
        decisions = await filter.filter_batch(files, "")

        assert len(decisions) == 2
        assert all(d.include is True for d in decisions)
        assert all("No filter criteria" in d.reason for d in decisions)

    @pytest.mark.asyncio
    async def test_whitespace_only_prompt(self):
        files = [
            FileInfo(path="a.py", absolute_path="/a.py", size_bytes=100, extension=".py"),
        ]
        filter = AgentFileFilter()
        decisions = await filter.filter_batch(files, "   \n\t  ")

        assert len(decisions) == 1
        assert decisions[0].include is True

    @pytest.mark.asyncio
    async def test_batch_with_mocked_llm(self):
        files = [
            FileInfo(path="a.py", absolute_path="/a.py", size_bytes=100, extension=".py"),
            FileInfo(path="b.txt", absolute_path="/b.txt", size_bytes=200, extension=".txt"),
        ]

        filter = AgentFileFilter(batch_size=10)

        mock_response = json.dumps(
            {
                "decisions": [
                    {"index": 1, "include": True, "reason": "Keep"},
                    {"index": 2, "include": False, "reason": "Remove"},
                ]
            }
        )

        with patch.object(filter, "_call_llm", return_value=mock_response) as mock_call:
            decisions = await filter.filter_batch(files, "Only Python")

        mock_call.assert_called_once()
        assert len(decisions) == 2
        assert decisions[0].include is True
        assert decisions[1].include is False

    @pytest.mark.asyncio
    async def test_batch_processing_multiple_batches(self):
        # Create more files than batch size
        files = [
            FileInfo(
                path=f"file{i}.py", absolute_path=f"/file{i}.py", size_bytes=100, extension=".py"
            )
            for i in range(5)
        ]

        filter = AgentFileFilter(batch_size=2)
        call_count = 0

        async def mock_call_llm(prompt):
            nonlocal call_count
            call_count += 1
            # Parse prompt to count files
            file_count = prompt.count("Extension: .py")
            decisions = [
                {"index": i + 1, "include": True, "reason": "OK"} for i in range(file_count)
            ]
            return json.dumps({"decisions": decisions})

        with patch.object(filter, "_call_llm", side_effect=mock_call_llm):
            decisions = await filter.filter_batch(files, "All Python")

        # Should make 3 calls: 2+2+1 files
        assert call_count == 3
        assert len(decisions) == 5
        assert all(d.include is True for d in decisions)

    @pytest.mark.asyncio
    async def test_batch_llm_error_defaults_to_include(self):
        files = [
            FileInfo(path="a.py", absolute_path="/a.py", size_bytes=100, extension=".py"),
        ]

        filter = AgentFileFilter()

        with patch.object(filter, "_call_llm", side_effect=ConnectionError("LLM Error")):
            decisions = await filter.filter_batch(files, "Test")

        assert len(decisions) == 1
        assert decisions[0].include is True
        assert decisions[0].confidence == 0.0
        assert "Filter error" in decisions[0].reason


class TestAgentFileFilterFilterFiles:
    """Tests for filter_files convenience method."""

    @pytest.mark.asyncio
    async def test_filter_files_splits_results(self):
        files = [
            FileInfo(path="a.py", absolute_path="/a.py", size_bytes=100, extension=".py"),
            FileInfo(path="b.txt", absolute_path="/b.txt", size_bytes=200, extension=".txt"),
            FileInfo(path="c.md", absolute_path="/c.md", size_bytes=150, extension=".md"),
        ]

        filter = AgentFileFilter()

        mock_response = json.dumps(
            {
                "decisions": [
                    {"index": 1, "include": True, "reason": "Keep Python"},
                    {"index": 2, "include": False, "reason": "Skip text"},
                    {"index": 3, "include": True, "reason": "Keep docs"},
                ]
            }
        )

        with patch.object(filter, "_call_llm", return_value=mock_response):
            included, excluded = await filter.filter_files(files, "Code and docs only")

        assert len(included) == 2
        assert len(excluded) == 1
        assert included[0].path == "a.py"
        assert included[1].path == "c.md"
        assert excluded[0].path == "b.txt"
        assert excluded[0].reason == ExclusionReason.AGENT
        assert excluded[0].details == "Skip text"


class TestGetAgentFilter:
    """Tests for get_agent_filter factory function."""

    def test_default_config(self):
        filter = get_agent_filter()
        assert isinstance(filter, AgentFileFilter)
        assert filter.model == "gemini-2.0-flash"
        assert filter.batch_size == AgentFileFilter.DEFAULT_BATCH_SIZE

    def test_custom_config(self):
        filter = get_agent_filter(model="claude-3-opus", batch_size=10)
        assert filter.model == "claude-3-opus"
        assert filter.batch_size == 10


class TestAgentFileFilterGetClient:
    """Tests for _get_client method."""

    @pytest.mark.asyncio
    async def test_unsupported_model_raises(self):
        filter = AgentFileFilter(model="unsupported-model")
        with pytest.raises(ValueError, match="Unsupported model"):
            await filter._get_client()

    @pytest.mark.asyncio
    async def test_gemini_missing_api_key(self):
        filter = AgentFileFilter(model="gemini-pro")

        # Mock the google.generativeai import
        mock_genai = MagicMock()
        with patch.dict("os.environ", {}, clear=True):
            with patch.dict(
                "sys.modules", {"google.generativeai": mock_genai, "google": MagicMock()}
            ):
                with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                    await filter._get_client()

    @pytest.mark.asyncio
    async def test_claude_missing_api_key(self):
        filter = AgentFileFilter(model="claude-3-sonnet")

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                await filter._get_client()

    @pytest.mark.asyncio
    async def test_gpt_missing_api_key(self):
        filter = AgentFileFilter(model="gpt-4")

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                await filter._get_client()

    @pytest.mark.asyncio
    async def test_client_reused(self):
        filter = AgentFileFilter(model="gemini-pro")
        mock_client = MagicMock()
        filter._client = mock_client

        client = await filter._get_client()
        assert client is mock_client


class TestAgentFileFilterCallLLM:
    """Tests for _call_llm method."""

    @pytest.mark.asyncio
    async def test_unsupported_model_in_call(self):
        filter = AgentFileFilter(model="unknown-model")
        filter._client = MagicMock()  # Bypass _get_client

        with pytest.raises(ValueError, match="Unsupported model"):
            await filter._call_llm("Test prompt")
