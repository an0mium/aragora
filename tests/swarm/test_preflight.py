from __future__ import annotations

import pytest

from aragora.swarm.preflight import _normalize_agent, _work_order


def test_work_order_normalizes_codex_model_family() -> None:
    work_order = _work_order("gpt-4.1-codex")

    assert work_order["target_agent"] == "codex"


def test_work_order_normalizes_claude_model_family() -> None:
    work_order = _work_order("claude-opus-4-6")

    assert work_order["target_agent"] == "claude"


def test_normalize_agent_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="Unsupported preflight agent"):
        _normalize_agent("gemini-cli")
