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
    assert result[0].content.startswith("\n\n## Header")
