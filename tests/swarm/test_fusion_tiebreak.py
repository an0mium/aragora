"""Tests for the Fusion merge-quorum tie-breaker (aragora.swarm.fusion_tiebreak).

Pure decision + comment composition; the Fusion call is injected, so no network
or model dependency. Verifies the hard constraint: the tie-breaker is advisory
and explicitly NON-counting (Fusion is a blend, never a quorum family).
"""

from __future__ import annotations

import pytest

from aragora.config.feature_flags import FeatureFlagRegistry
from aragora.swarm import fusion_tiebreak as ft


def test_should_run_only_on_genuine_split_when_enabled() -> None:
    # Split: one PASS, one changes-requested, no quorum, flag on -> run.
    assert ft.should_run_tiebreaker(
        supportive_families=["claude"],
        dissenting_families=["grok"],
        has_supportive_quorum=False,
        flag_enabled=True,
    )


def test_should_not_run_when_flag_off() -> None:
    assert not ft.should_run_tiebreaker(
        supportive_families=["claude"],
        dissenting_families=["grok"],
        has_supportive_quorum=False,
        flag_enabled=False,
    )


def test_should_not_run_when_quorum_met() -> None:
    assert not ft.should_run_tiebreaker(
        supportive_families=["claude", "grok"],
        dissenting_families=[],
        has_supportive_quorum=True,
        flag_enabled=True,
    )


def test_should_not_run_on_unanimous_fail() -> None:
    # No supportive family => not a split; the dissent is real and must be fixed,
    # not overridden by a tie-breaker.
    assert not ft.should_run_tiebreaker(
        supportive_families=[],
        dissenting_families=["claude", "grok"],
        has_supportive_quorum=False,
        flag_enabled=True,
    )


def test_run_tiebreaker_composes_disclosed_noncounting_comment() -> None:
    out = ft.run_tiebreaker(
        supportive_families=["claude"],
        dissenting_families=["grok"],
        has_supportive_quorum=False,
        flag_enabled=True,
        head_sha="abcdef1234567890",
        pr=8444,
        fusion_review=lambda: "Verdict: lean-pass\nThe dissent is a non-blocking nit.",
    )
    assert out.ran is True
    assert out.is_tie_breaker is True
    assert out.comment is not None
    # Disclosure + the hard non-counting constraint must be explicit.
    assert ft.TIEBREAKER_HEADING in out.comment
    assert "NOT count as an independent quorum family" in out.comment
    assert "advisory" in out.comment.lower()
    assert "abcdef123456" in out.comment  # head short-sha
    assert "#8444" in out.comment


@pytest.mark.parametrize("head_sha", ["", "   ", "abc123"])
def test_run_tiebreaker_treats_empty_or_short_head_sha_as_unknown(head_sha: str) -> None:
    out = ft.run_tiebreaker(
        supportive_families=["claude"],
        dissenting_families=["grok"],
        has_supportive_quorum=False,
        flag_enabled=True,
        head_sha=head_sha,
        pr=8444,
        fusion_review=lambda: "Verdict: lean-pass",
    )

    assert out.ran is True
    assert out.comment is not None
    assert "head unknown" in out.comment
    assert "head abc123" not in out.comment
    assert "head ." not in out.comment


def test_run_tiebreaker_keeps_none_head_sha_fail_loud() -> None:
    with pytest.raises(TypeError, match="head_sha"):
        ft.run_tiebreaker(
            supportive_families=["claude"],
            dissenting_families=["grok"],
            has_supportive_quorum=False,
            flag_enabled=True,
            head_sha=None,  # type: ignore[arg-type]
            pr=8444,
            fusion_review=lambda: "Verdict: lean-pass",
        )


def test_run_tiebreaker_noop_without_runner() -> None:
    # Fusion not runnable here (no key/slug) -> graceful no-op, never blocks.
    out = ft.run_tiebreaker(
        supportive_families=["claude"],
        dissenting_families=["grok"],
        has_supportive_quorum=False,
        flag_enabled=True,
        head_sha="deadbeef",
        pr=1,
        fusion_review=None,
    )
    assert out.ran is False
    assert out.is_tie_breaker is False
    assert "not runnable" in out.reason


def test_run_tiebreaker_noop_on_empty_fusion_output() -> None:
    out = ft.run_tiebreaker(
        supportive_families=["claude"],
        dissenting_families=["grok"],
        has_supportive_quorum=False,
        flag_enabled=True,
        head_sha="deadbeef",
        pr=1,
        fusion_review=lambda: "   ",
    )
    assert out.ran is False
    assert out.is_tie_breaker is False
    assert "empty" in out.reason


def test_run_tiebreaker_noop_when_fusion_runner_raises() -> None:
    def raises() -> str:
        raise RuntimeError("fusion unavailable")

    out = ft.run_tiebreaker(
        supportive_families=["claude"],
        dissenting_families=["grok"],
        has_supportive_quorum=False,
        flag_enabled=True,
        head_sha="deadbeef",
        pr=1,
        fusion_review=raises,
    )

    assert out.ran is False
    assert out.comment is None
    assert out.is_tie_breaker is False
    assert out.reason == "fusion review raised RuntimeError"
    assert "fusion unavailable" not in out.reason


def test_run_tiebreaker_noop_when_no_split() -> None:
    # Quorum already met -> nothing to break, even with a runner present.
    called = []
    out = ft.run_tiebreaker(
        supportive_families=["claude", "grok"],
        dissenting_families=[],
        has_supportive_quorum=True,
        flag_enabled=True,
        head_sha="deadbeef",
        pr=1,
        fusion_review=lambda: called.append(1) or "should not run",
    )
    assert out.ran is False
    assert out.is_tie_breaker is False
    assert called == []  # runner must NOT be invoked when there's no tie


def test_tiebreak_flag_registered_default_off(monkeypatch) -> None:
    monkeypatch.delenv("ARAGORA_ENABLE_FUSION_QUORUM_TIEBREAK", raising=False)
    reg = FeatureFlagRegistry()
    assert reg.is_enabled("enable_fusion_quorum_tiebreak") is False
