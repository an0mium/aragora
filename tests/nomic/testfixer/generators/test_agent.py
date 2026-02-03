"""Tests for AgentCodeGenerator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.testfixer.generators.agent import (
    AgentCodeGenerator,
    AgentGeneratorConfig,
    _CONFIDENCE_RE,
    _CRITIQUE_RE,
    _FILE_RE,
    _RATIONALE_RE,
    _VERDICT_RE,
)
from aragora.nomic.testfixer.runner import TestFailure
from aragora.nomic.testfixer.analyzer import FailureAnalysis, FailureCategory, FixTarget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_failure() -> TestFailure:
    return TestFailure(
        test_name="test_example",
        test_file="tests/test_foo.py",
        error_type="AssertionError",
        error_message="expected 1 got 2",
        stack_trace="Traceback ...",
    )


def _make_analysis(failure: TestFailure | None = None) -> FailureAnalysis:
    return FailureAnalysis(
        failure=failure or _make_failure(),
        category=FailureCategory.TEST_ASSERTION,
        confidence=0.8,
        fix_target=FixTarget.TEST_FILE,
        root_cause="Wrong assertion",
        root_cause_file="tests/test_foo.py",
    )


def _build_generator(agent_mock: AsyncMock | None = None) -> AgentCodeGenerator:
    """Build an AgentCodeGenerator with a mocked agent."""
    with patch("aragora.nomic.testfixer.generators.agent.create_agent") as mock_create:
        mock_agent = agent_mock or AsyncMock()
        mock_create.return_value = mock_agent
        config = AgentGeneratorConfig(
            agent_type="claude",
            model="test-model",
            max_context_chars=1000,
            max_output_chars=5000,
        )
        gen = AgentCodeGenerator(config)
    return gen


# ---------------------------------------------------------------------------
# _truncate tests
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_text_unchanged(self) -> None:
        gen = _build_generator()
        text = "short text"
        assert gen._truncate(text, 100) == text

    def test_long_text_truncated_with_marker(self) -> None:
        gen = _build_generator()
        text = "a" * 2000
        result = gen._truncate(text, 500)
        assert len(result) < len(text)
        assert "chars truncated" in result
        # Verify the truncation message includes a count
        assert "1500 chars truncated" in result


# ---------------------------------------------------------------------------
# _extract_tag tests
# ---------------------------------------------------------------------------


class TestExtractTag:
    def test_tag_found(self) -> None:
        gen = _build_generator()
        text = "before <file>content here</file> after"
        result = gen._extract_tag(_FILE_RE, text)
        assert result == "content here"

    def test_tag_not_found_returns_none(self) -> None:
        gen = _build_generator()
        text = "no tags here"
        result = gen._extract_tag(_FILE_RE, text)
        assert result is None

    def test_tag_with_whitespace_is_stripped(self) -> None:
        gen = _build_generator()
        text = "<rationale>  some rationale  </rationale>"
        result = gen._extract_tag(_RATIONALE_RE, text)
        assert result == "some rationale"

    def test_confidence_tag(self) -> None:
        gen = _build_generator()
        text = "<confidence>0.85</confidence>"
        result = gen._extract_tag(_CONFIDENCE_RE, text)
        assert result == "0.85"

    def test_verdict_tag(self) -> None:
        gen = _build_generator()
        text = "<verdict>pass</verdict>"
        result = gen._extract_tag(_VERDICT_RE, text)
        assert result == "pass"

    def test_critique_tag(self) -> None:
        gen = _build_generator()
        text = "<critique>looks good</critique>"
        result = gen._extract_tag(_CRITIQUE_RE, text)
        assert result == "looks good"


# ---------------------------------------------------------------------------
# _clean_file_output tests
# ---------------------------------------------------------------------------


class TestCleanFileOutput:
    def test_with_file_tags(self) -> None:
        gen = _build_generator()
        text = "preamble\n<file>\ndef foo():\n    pass\n</file>\npostscript"
        result = gen._clean_file_output(text)
        assert result == "def foo():\n    pass"

    def test_with_code_fences(self) -> None:
        gen = _build_generator()
        text = "Here is the fix:\n```python\ndef bar():\n    return 1\n```\nDone."
        result = gen._clean_file_output(text)
        assert result == "def bar():\n    return 1"

    def test_plain_text(self) -> None:
        gen = _build_generator()
        text = "  just plain content  "
        result = gen._clean_file_output(text)
        assert result == "just plain content"

    def test_file_tags_take_precedence_over_fences(self) -> None:
        gen = _build_generator()
        text = "<file>inside tags</file>\n```python\ninside fences\n```"
        result = gen._clean_file_output(text)
        assert result == "inside tags"


# ---------------------------------------------------------------------------
# _parse_confidence tests
# ---------------------------------------------------------------------------


class TestParseConfidence:
    def test_with_confidence_tag(self) -> None:
        gen = _build_generator()
        text = "response <confidence>0.8</confidence> end"
        result = gen._parse_confidence(text, default=0.5)
        assert result == 0.8

    def test_missing_tag_returns_default(self) -> None:
        gen = _build_generator()
        text = "no confidence tag here"
        result = gen._parse_confidence(text, default=0.42)
        assert result == 0.42

    def test_invalid_value_returns_default(self) -> None:
        gen = _build_generator()
        text = "<confidence>not_a_number</confidence>"
        result = gen._parse_confidence(text, default=0.3)
        assert result == 0.3

    def test_clamps_above_one(self) -> None:
        gen = _build_generator()
        text = "<confidence>5.0</confidence>"
        result = gen._parse_confidence(text, default=0.5)
        assert result == 1.0

    def test_clamps_below_zero(self) -> None:
        gen = _build_generator()
        text = "<confidence>-0.5</confidence>"
        result = gen._parse_confidence(text, default=0.5)
        assert result == 0.0


# ---------------------------------------------------------------------------
# generate_fix tests
# ---------------------------------------------------------------------------


class TestGenerateFix:
    @pytest.mark.asyncio
    async def test_extracts_from_tagged_response(self) -> None:
        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(
            return_value=(
                "<file>def fixed():\n    return 42</file>\n"
                "<rationale>Fixed the return value</rationale>\n"
                "<confidence>0.9</confidence>"
            )
        )
        gen = _build_generator(mock_agent)

        analysis = _make_analysis()
        content, rationale, confidence = await gen.generate_fix(
            analysis, "def broken(): pass", "tests/test_foo.py"
        )

        assert content == "def fixed():\n    return 42"
        assert rationale == "Fixed the return value"
        assert confidence == 0.9

    @pytest.mark.asyncio
    async def test_fallback_when_no_file_content(self) -> None:
        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(
            return_value="<rationale>Could not determine fix</rationale>"
        )
        gen = _build_generator(mock_agent)

        analysis = _make_analysis()
        content, rationale, confidence = await gen.generate_fix(
            analysis, "original content", "tests/test_foo.py"
        )

        # When _clean_file_output returns empty-ish, the method falls back
        # to the original content with capped confidence
        assert "Could not determine fix" in rationale


# ---------------------------------------------------------------------------
# critique_fix tests
# ---------------------------------------------------------------------------


class TestCritiqueFix:
    @pytest.mark.asyncio
    async def test_pass_verdict(self) -> None:
        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(
            return_value=(
                "<verdict>pass</verdict>\n"
                "<critique>The fix looks correct and addresses the root cause.</critique>"
            )
        )
        gen = _build_generator(mock_agent)

        analysis = _make_analysis()
        critique, is_ok = await gen.critique_fix(analysis, "original", "fixed", "rationale")

        assert is_ok is True
        assert "correct" in critique

    @pytest.mark.asyncio
    async def test_fail_verdict(self) -> None:
        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(
            return_value=(
                "<verdict>fail</verdict>\n<critique>The fix introduces a regression.</critique>"
            )
        )
        gen = _build_generator(mock_agent)

        analysis = _make_analysis()
        critique, is_ok = await gen.critique_fix(analysis, "original", "bad fix", "rationale")

        assert is_ok is False
        assert "regression" in critique

    @pytest.mark.asyncio
    async def test_missing_verdict_defaults_to_fail(self) -> None:
        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(
            return_value="<critique>I am not sure about this fix.</critique>"
        )
        gen = _build_generator(mock_agent)

        analysis = _make_analysis()
        _, is_ok = await gen.critique_fix(analysis, "original", "proposed", "rationale")

        assert is_ok is False


# ---------------------------------------------------------------------------
# synthesize_fixes tests
# ---------------------------------------------------------------------------


class TestSynthesizeFixes:
    @pytest.mark.asyncio
    async def test_synthesis(self) -> None:
        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(
            return_value=(
                "<file>def synthesized():\n    return 99</file>\n"
                "<rationale>Combined best parts</rationale>\n"
                "<confidence>0.85</confidence>"
            )
        )
        gen = _build_generator(mock_agent)

        analysis = _make_analysis()
        proposals = [
            ("def a(): pass", "approach a", 0.6),
            ("def b(): pass", "approach b", 0.7),
        ]
        critiques = ["a is ok", "b has issues"]

        content, rationale, confidence = await gen.synthesize_fixes(analysis, proposals, critiques)

        assert content == "def synthesized():\n    return 99"
        assert rationale == "Combined best parts"
        assert confidence == 0.85

    @pytest.mark.asyncio
    async def test_empty_proposals_fallback(self) -> None:
        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(return_value="<rationale>Nothing to work with</rationale>")
        gen = _build_generator(mock_agent)

        analysis = _make_analysis()
        content, rationale, confidence = await gen.synthesize_fixes(analysis, [], [])

        # With no proposals, the response text becomes the content (no <file> tag),
        # and confidence uses default=0.6 since no <confidence> tag present
        assert confidence <= 0.6
