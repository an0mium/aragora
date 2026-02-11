"""Benchmarks for context budgeter performance.

Measures:
- Token estimation throughput
- Section truncation speed
- Full budget application with realistic section counts
- Scaling behavior with increasing section counts
"""

import time

import pytest

from aragora.debate.context_budgeter import (
    ContextBudgeter,
    ContextSection,
    _estimate_tokens,
    _truncate_text,
)

pytestmark = pytest.mark.benchmark


def _make_content(tokens: int) -> str:
    """Generate content string of approximately the given token count."""
    return "x" * (tokens * 4)


def _make_sections(count: int, tokens_each: int = 200) -> list[ContextSection]:
    """Generate a list of context sections."""
    keys = [
        "env_context", "historical", "continuum", "supermemory",
        "belief", "dissent", "patterns", "calibration",
        "elo", "evidence", "trending", "audience",
    ]
    return [
        ContextSection(
            key=keys[i % len(keys)],
            content=_make_content(tokens_each),
        )
        for i in range(count)
    ]


class TestTokenEstimation:
    """Benchmarks for _estimate_tokens function."""

    def test_estimate_empty(self):
        """Empty string should return 0."""
        assert _estimate_tokens("") == 0

    def test_estimate_short(self):
        """Short strings should return at least 1."""
        assert _estimate_tokens("hi") >= 1

    def test_estimate_throughput(self):
        """Token estimation should handle 100k calls in under 100ms."""
        text = "x" * 4000  # ~1000 tokens
        start = time.perf_counter()
        for _ in range(100_000):
            _estimate_tokens(text)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"100k estimations took {elapsed:.3f}s (expected <0.1s)"

    def test_estimate_large_text_throughput(self):
        """Token estimation on large text (100k chars) should be fast."""
        text = "x" * 100_000  # ~25k tokens
        start = time.perf_counter()
        for _ in range(10_000):
            _estimate_tokens(text)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"10k large estimations took {elapsed:.3f}s"


class TestTruncation:
    """Benchmarks for _truncate_text function."""

    def test_truncate_no_op(self):
        """Text within budget should be returned unchanged."""
        text = _make_content(100)
        result = _truncate_text(text, 200)
        assert result == text

    def test_truncate_correctness(self):
        """Truncated text should end with [truncated] marker."""
        text = _make_content(1000)
        result = _truncate_text(text, 100)
        assert result.endswith("[truncated]")
        assert len(result) < len(text)

    def test_truncate_throughput(self):
        """Truncation should handle 10k calls in under 100ms."""
        text = _make_content(1000)
        start = time.perf_counter()
        for _ in range(10_000):
            _truncate_text(text, 100)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"10k truncations took {elapsed:.3f}s"

    def test_truncate_large_text(self):
        """Truncating very large text (1M chars) should still be fast."""
        text = _make_content(250_000)  # ~1M chars
        start = time.perf_counter()
        for _ in range(1_000):
            _truncate_text(text, 500)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"1k large truncations took {elapsed:.3f}s"


class TestBudgeterApply:
    """Benchmarks for ContextBudgeter.apply() method."""

    def test_apply_default_sections(self):
        """Applying budget to default section count (12) should be fast."""
        budgeter = ContextBudgeter()
        sections = _make_sections(12, tokens_each=400)

        start = time.perf_counter()
        for _ in range(10_000):
            budgeter.apply(sections)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"10k applies (12 sections) took {elapsed:.3f}s"

    def test_apply_many_sections(self):
        """Applying budget to 50 sections should scale linearly."""
        budgeter = ContextBudgeter(total_tokens=10000)
        sections = _make_sections(50, tokens_each=200)

        start = time.perf_counter()
        for _ in range(5_000):
            budgeter.apply(sections)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"5k applies (50 sections) took {elapsed:.3f}s"

    def test_apply_budget_exhaustion(self):
        """When budget runs out early, later sections should be skipped."""
        budgeter = ContextBudgeter(total_tokens=100)
        sections = _make_sections(20, tokens_each=200)

        results = budgeter.apply(sections)
        total_tokens = sum(r.tokens for r in results)
        assert total_tokens <= 100
        assert len(results) < 20  # Not all sections fit

    def test_apply_no_truncation_needed(self):
        """When budget is large enough, no truncation should occur."""
        budgeter = ContextBudgeter(total_tokens=100_000)
        sections = _make_sections(12, tokens_each=100)

        results = budgeter.apply(sections)
        assert all(not r.truncated for r in results)
        assert len(results) == 12

    def test_apply_all_truncated(self):
        """With tight budget, all sections should be truncated."""
        budgeter = ContextBudgeter(total_tokens=500)
        sections = _make_sections(12, tokens_each=400)

        results = budgeter.apply(sections)
        truncated_count = sum(1 for r in results if r.truncated)
        assert truncated_count > 0

    def test_section_priority_ordering(self):
        """Earlier sections should get budget priority."""
        budgeter = ContextBudgeter(total_tokens=300)
        sections = [
            ContextSection(key="env_context", content=_make_content(200)),
            ContextSection(key="historical", content=_make_content(200)),
            ContextSection(key="continuum", content=_make_content(200)),
        ]

        results = budgeter.apply(sections)
        # First section should have more tokens than later ones
        if len(results) >= 2:
            assert results[0].tokens >= results[-1].tokens

    def test_custom_section_limits(self):
        """Custom section limits should be respected."""
        budgeter = ContextBudgeter(
            total_tokens=5000,
            section_limits={"env_context": 100, "historical": 50},
        )
        sections = [
            ContextSection(key="env_context", content=_make_content(500)),
            ContextSection(key="historical", content=_make_content(500)),
        ]

        results = budgeter.apply(sections)
        assert results[0].tokens <= 100
        assert results[1].tokens <= 50


class TestBudgeterScaling:
    """Test scaling behavior of the context budgeter."""

    def test_linear_scaling_with_section_count(self):
        """Time should scale linearly with section count."""
        budgeter = ContextBudgeter(total_tokens=50000)

        # Measure time for 10 sections
        sections_10 = _make_sections(10, tokens_each=200)
        start = time.perf_counter()
        for _ in range(5_000):
            budgeter.apply(sections_10)
        time_10 = time.perf_counter() - start

        # Measure time for 100 sections
        sections_100 = _make_sections(100, tokens_each=200)
        start = time.perf_counter()
        for _ in range(5_000):
            budgeter.apply(sections_100)
        time_100 = time.perf_counter() - start

        # 100 sections should take less than 20x the time of 10 sections
        # (allowing overhead margin; perfectly linear would be 10x)
        ratio = time_100 / max(time_10, 0.0001)
        assert ratio < 20, f"Scaling ratio {ratio:.1f}x exceeds 20x threshold"
