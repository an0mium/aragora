"""Tests for ContextBudgeter."""

from aragora.debate.context_budgeter import ContextBudgeter, ContextSection


def test_budgeter_truncates_section_by_limit():
    budgeter = ContextBudgeter(total_tokens=100, section_limits={"alpha": 5})
    content = "a" * 200  # ~50 tokens
    sections = [ContextSection("alpha", content)]
    result = budgeter.apply(sections)
    assert result
    section = result[0]
    assert section.truncated is True
    assert section.tokens <= 5
    assert section.content.endswith("...[truncated]")


def test_budgeter_respects_total_budget_order():
    budgeter = ContextBudgeter(total_tokens=8, section_limits={})
    sections = [
        ContextSection("first", "x" * 40),  # ~10 tokens
        ContextSection("second", "y" * 40),  # ~10 tokens
    ]
    result = budgeter.apply(sections)
    assert result
    assert result[0].tokens <= 8
    # Second section should be omitted due to total budget exhaustion
    assert len(result) == 1


def test_budgeter_preserves_leading_whitespace():
    budgeter = ContextBudgeter(total_tokens=5, section_limits={"alpha": 5})
    content = "\n\n## Header\n" + ("z" * 200)
    sections = [ContextSection("alpha", content)]
    result = budgeter.apply(sections)
    assert result
    assert result[0].content.startswith("\n\n")


# ---------------------------------------------------------------------------
# Codebase section (Piece 1)
# ---------------------------------------------------------------------------


def test_codebase_in_default_section_limits():
    """Verify 'codebase' section is in DEFAULT_SECTION_LIMITS."""
    from aragora.debate.context_budgeter import DEFAULT_SECTION_LIMITS

    assert "codebase" in DEFAULT_SECTION_LIMITS
    assert DEFAULT_SECTION_LIMITS["codebase"] == 500


def test_codebase_section_limit_applied():
    """Verify budgeter respects the codebase section limit."""
    budgeter = ContextBudgeter()
    long_content = "x" * 10000  # Way over 500 tokens
    sections = [ContextSection("codebase", long_content)]
    result = budgeter.apply(sections)
    assert result
    assert result[0].truncated is True
    assert result[0].tokens <= 500


def test_codebase_section_env_override():
    """Verify env var can override codebase section limit."""
    import os

    from aragora.debate.context_budgeter import _parse_section_map

    # Test parsing the override format
    parsed = _parse_section_map("codebase=1000")
    assert parsed == {"codebase": 1000}
