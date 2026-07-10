"""Focused tests for the built-in PR reviewer skill."""

from unittest.mock import AsyncMock

import pytest

from aragora.skills.base import SkillContext
from aragora.skills.builtin.pr_reviewer import PRReviewerSkill


@pytest.mark.asyncio
async def test_empty_structured_findings_are_successful(monkeypatch: pytest.MonkeyPatch) -> None:
    skill = PRReviewerSkill(post_comment=False)
    monkeypatch.setattr(skill, "_run_review", AsyncMock(return_value=({}, None)))

    result = await skill.execute({"diff": "diff --git a/a.py b/a.py"}, SkillContext())

    assert result.success is True
    assert result.data is not None
    assert result.data["findings"] == {}


@pytest.mark.asyncio
async def test_missing_structured_result_is_protocol_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = PRReviewerSkill(post_comment=False)
    monkeypatch.setattr(skill, "_run_review", AsyncMock(return_value=(None, None)))

    result = await skill.execute({"diff": "diff --git a/a.py b/a.py"}, SkillContext())

    assert result.success is False
    assert result.error_code == "REVIEW_FAILED"
    assert result.error_message is not None
    assert "no structured result" in result.error_message
