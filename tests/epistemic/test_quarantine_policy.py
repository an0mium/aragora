"""Tests for DIC-21 Fail-Closed Quarantine Policy (aragora.epistemic.quarantine_policy)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from aragora.epistemic.quarantine_policy import (
    DEFAULT_POLICIES,
    EscalationMap,
    QuarantinePolicy,
    apply_quarantine_policy,
    quarantine_policy_enabled,
)


@dataclass
class _R:
    kind: str
    detail: str = ""
    claim_id: str = ""
    crux_id: str = ""


@dataclass
class _S:
    code_unit_id: str = "u"
    integrity_score: float = 0.9
    recommended_action: str = "report_only"
    reasons: list[_R] = field(default_factory=list)


class TestHappyPath:
    def test_healthy_report_only(self) -> None:
        d = apply_quarantine_policy(_S())
        assert d.policy_action == "report_only" and not d.fail_closed

    def test_report_only_empty_provenance(self) -> None:
        assert apply_quarantine_policy(_S()).provenance_hash == ""

    def test_non_report_only_sha256_provenance(self) -> None:
        assert (
            len(apply_quarantine_policy(_S(recommended_action="repair_required")).provenance_hash)
            == 64
        )

    def test_provenance_deterministic(self) -> None:
        s = _S(integrity_score=0.75, recommended_action="repair_required")
        assert (
            apply_quarantine_policy(s).provenance_hash == apply_quarantine_policy(s).provenance_hash
        )

    def test_score_and_uid_copied(self) -> None:
        d = apply_quarantine_policy(_S(code_unit_id="foo", integrity_score=0.72))
        assert d.code_unit_id == "foo" and d.integrity_score == pytest.approx(0.72)


class TestFailClosed:
    def test_below_threshold(self) -> None:
        d = apply_quarantine_policy(
            _S(integrity_score=0.3), policy=QuarantinePolicy(fail_closed_threshold=0.4)
        )
        assert d.policy_action == "fail_closed" and d.fail_closed

    def test_at_threshold_safe(self) -> None:
        d = apply_quarantine_policy(
            _S(integrity_score=0.4), policy=QuarantinePolicy(fail_closed_threshold=0.4)
        )
        assert not d.fail_closed

    def test_fail_closed_overrides_escalation(self) -> None:
        policy = QuarantinePolicy(fail_closed_threshold=0.5)
        d = apply_quarantine_policy(
            _S(integrity_score=0.2, recommended_action="repair_required"), policy=policy
        )
        assert d.policy_action == "fail_closed"

    def test_fail_closed_has_provenance(self) -> None:
        assert apply_quarantine_policy(_S(integrity_score=0.1)).provenance_hash != ""


class TestLiveSwap:
    def test_blocked_by_default(self) -> None:
        assert apply_quarantine_policy(_S(), request_live_swap=True).live_swap_blocked

    def test_permitted_when_in_allowlist(self) -> None:
        policy = QuarantinePolicy(live_swap_allowlist=frozenset({"u"}), fail_closed_threshold=0.0)
        d = apply_quarantine_policy(_S(code_unit_id="u"), policy=policy, request_live_swap=True)
        assert not d.live_swap_blocked

    def test_blocked_swap_escalates_to_quarantine(self) -> None:
        policy = QuarantinePolicy(fail_closed_threshold=0.0)
        d = apply_quarantine_policy(_S(integrity_score=0.9), policy=policy, request_live_swap=True)
        assert d.policy_action in {"quarantine", "repair_required", "fail_closed"}


class TestDefaults:
    def test_all_classes_present(self) -> None:
        for c in ("live_dispatch", "report_surface", "demo", "pure_policy", "default"):
            assert c in DEFAULT_POLICIES

    def test_all_allowlists_empty(self) -> None:
        assert all(len(p.live_swap_allowlist) == 0 for p in DEFAULT_POLICIES.values())

    def test_class_lookup(self) -> None:
        d = apply_quarantine_policy(
            _S(recommended_action="repair_required"), code_unit_class="report_surface"
        )
        assert d.policy_action == "degrade"

    def test_unknown_class_falls_back(self) -> None:
        assert apply_quarantine_policy(_S(), code_unit_class="nope").policy_action == "report_only"


class TestFlagGate:
    def test_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_QUARANTINE_POLICY_ENABLED", raising=False)
        assert not quarantine_policy_enabled()

    @pytest.mark.parametrize("v", ["1", "true", "yes", "on"])
    def test_truthy(self, monkeypatch: pytest.MonkeyPatch, v: str) -> None:
        monkeypatch.setenv("ARAGORA_QUARANTINE_POLICY_ENABLED", v)
        assert quarantine_policy_enabled()
