"""Tests for aragora.swarm.harness_fallback (Round 30f phase 2)."""

from __future__ import annotations

import pytest

from aragora.swarm.harness_fallback import (
    FallbackLadder,
    FallbackResolution,
    default_implementation_ladder,
    default_review_ladder,
)
from aragora.swarm.harness_health import HarnessHealthRegistry


class TestFallbackLadderConstruction:
    def test_simple_construction(self) -> None:
        ladder = FallbackLadder(name="test", steps=("a", "b", "c"))
        assert ladder.name == "test"
        assert ladder.steps == ("a", "b", "c")

    def test_rejects_empty_steps(self) -> None:
        with pytest.raises(ValueError, match="at least one step"):
            FallbackLadder(name="test", steps=())

    def test_rejects_duplicate_steps(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            FallbackLadder(name="test", steps=("a", "b", "a"))

    def test_rejects_empty_string_step(self) -> None:
        with pytest.raises(ValueError, match="non-empty strings"):
            FallbackLadder(name="test", steps=("a", ""))

    def test_from_steps_builder(self) -> None:
        ladder = FallbackLadder.from_steps("review", ["claude-code", "codex"])
        assert ladder.steps == ("claude-code", "codex")


class TestFallbackResolution:
    def test_first_step_is_chosen_when_available(self) -> None:
        reg = HarnessHealthRegistry()
        ladder = FallbackLadder(name="t", steps=("a", "b", "c"))
        result = ladder.next_available(registry=reg)
        assert result.chosen == "a"
        assert result.skipped == ()
        assert result.reasons == {}
        assert result.is_resolved() is True

    def test_skips_pinned_first_step(self) -> None:
        reg = HarnessHealthRegistry()
        reg.record_failure("a", reason="api key", status_code=401)
        ladder = FallbackLadder(name="t", steps=("a", "b", "c"))
        result = ladder.next_available(registry=reg)
        assert result.chosen == "b"
        assert result.skipped == ("a",)
        assert "auth" in result.reasons["a"]

    def test_returns_none_when_all_pinned(self) -> None:
        reg = HarnessHealthRegistry()
        reg.record_failure("a", reason="auth", status_code=401)
        reg.record_failure("b", reason="auth", status_code=401)
        reg.record_failure("c", reason="auth", status_code=401)
        ladder = FallbackLadder(name="t", steps=("a", "b", "c"))
        result = ladder.next_available(registry=reg)
        assert result.chosen is None
        assert result.skipped == ("a", "b", "c")
        assert result.is_resolved() is False

    def test_uses_singleton_when_no_registry_passed(self, monkeypatch) -> None:
        # next_available without registry should consult the global singleton.
        from aragora.swarm import harness_health as hh

        hh.reset_harness_health_registry()
        ladder = FallbackLadder(name="t", steps=("a", "b"))
        result = ladder.next_available()
        # Singleton is fresh => "a" available
        assert result.chosen == "a"


class TestDefaultLadders:
    def test_default_implementation_order(self) -> None:
        ladder = default_implementation_ladder()
        assert ladder.name == "implementation"
        # Round 30g: aider removed — no real AiderHarness ships in
        # aragora.harnesses, so the ladder must only list harnesses
        # that can actually run.
        assert ladder.steps == ("claude-code", "codex")

    def test_default_review_order(self) -> None:
        ladder = default_review_ladder()
        assert ladder.name == "review"
        assert ladder.steps == ("claude-code", "codex")

    def test_implementation_falls_back_when_claude_pinned(self) -> None:
        reg = HarnessHealthRegistry()
        reg.record_failure("claude-code", reason="auth", status_code=401)
        ladder = default_implementation_ladder()
        result = ladder.next_available(registry=reg)
        assert result.chosen == "codex"

    def test_review_returns_none_when_both_pinned(self) -> None:
        reg = HarnessHealthRegistry()
        reg.record_failure("claude-code", reason="auth", status_code=401)
        reg.record_failure("codex", reason="auth", status_code=401)
        ladder = default_review_ladder()
        result = ladder.next_available(registry=reg)
        assert result.chosen is None
        assert result.skipped == ("claude-code", "codex")
