"""B3: opt-in OpenRouter Fusion participation in self-improvement planning.

Locks in the contract for MetaPlanner._maybe_add_fusion:
- default OFF (no behavior change),
- gated on the enable_fusion flag AND a positive per-debate budget,
- never duplicates fusion if already selected,
- fail-open: any error leaves the agent list untouched (planning never breaks).
"""

from __future__ import annotations

import pytest

from aragora.nomic.meta_planner import MetaPlanner, MetaPlannerConfig


def _planner(**cfg) -> MetaPlanner:
    return MetaPlanner(MetaPlannerConfig(**cfg))


def test_default_off_does_not_add_fusion() -> None:
    planner = _planner()  # enable_fusion defaults False
    assert planner._maybe_add_fusion(["claude", "deepseek"]) == ["claude", "deepseek"]


def test_enabled_with_budget_adds_fusion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "aragora.nomic.meta_planner.get_flag",
        lambda name, default=None: 50.0 if "budget" in name else default,
    )
    planner = _planner(enable_fusion=True)
    out = planner._maybe_add_fusion(["claude", "deepseek"])
    assert "fusion" in out
    # Existing participants are preserved, fusion is appended (not replacing).
    assert out[:2] == ["claude", "deepseek"]


def test_zero_budget_blocks_fusion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "aragora.nomic.meta_planner.get_flag",
        lambda name, default=None: 0.0 if "budget" in name else default,
    )
    planner = _planner(enable_fusion=True)
    assert "fusion" not in planner._maybe_add_fusion(["claude", "deepseek"])


def test_no_duplicate_when_already_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "aragora.nomic.meta_planner.get_flag",
        lambda name, default=None: 50.0 if "budget" in name else default,
    )
    planner = _planner(enable_fusion=True)
    out = planner._maybe_add_fusion(["claude", "fusion"])
    assert out.count("fusion") == 1


def test_fail_open_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a, **_k):
        raise RuntimeError("flag store down")

    monkeypatch.setattr("aragora.nomic.meta_planner.get_flag", _boom)
    planner = _planner(enable_fusion=True)
    # Must not raise, must return the original list unchanged.
    assert planner._maybe_add_fusion(["claude"]) == ["claude"]
