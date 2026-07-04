"""Synthetic world-state event tests for DIC-20 / #6031.

Verifies that external events (CVE drops, API changes, dependency
version bumps) correctly translate into ClaimResult decay and
propagate through evaluate_unit, satisfying the DIC-20 acceptance
criterion for synthetic world-event invalidation tests.
"""
from __future__ import annotations

import pytest

from aragora.epistemic.claim_verifier import ClaimStatus
from aragora.epistemic.decay_monitor import evaluate_unit
from aragora.epistemic.proof_unit import DecayPolicy, FallbackPolicy, ProofCarryingCodeUnit
from aragora.epistemic.world_event import (
    WorldEventKind,
    WorldStateEvent,
    claims_affected_by_event,
    world_event_to_claim_results,
    world_events_enabled,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(claims: list[str]) -> ProofCarryingCodeUnit:
    return ProofCarryingCodeUnit(
        code_unit_id="unit.test",
        symbol="tests.fake.fn",
        source_path="tests/fake.py",
        owner="test",
        decision_receipts=["receipt-001"],
        claims=claims,
        assumptions=["External dependency is stable."],
        verifiers=[],
        freshness_sla_hours=24,
        decay_policy=DecayPolicy(),
        fallback_policy=FallbackPolicy(),
        linked_crux_ids=[],
    )


def _cve(scope: list[str] | None = None, eid: str = "CVE-2024-9999") -> WorldStateEvent:
    return WorldStateEvent(
        event_id=eid,
        kind=WorldEventKind.CVE,
        description="Critical issue in libfoo ≤ 3.2.1",
        affected_scope=scope if scope is not None else ["libfoo"],
        timestamp="2024-10-01T12:00:00Z",
    )


def _api_event() -> WorldStateEvent:
    return WorldStateEvent(
        event_id="api.openai.v1-sunset",
        kind=WorldEventKind.API_CHANGE,
        description="OpenAI v1 completions sunsetted",
        affected_scope=["openai.completions"],
    )


def _dep_event() -> WorldStateEvent:
    return WorldStateEvent(
        event_id="dep.pydantic.v3",
        kind=WorldEventKind.DEPENDENCY_BUMP,
        description="Pydantic v3 breaks model.dict()",
        affected_scope=["pydantic."],
    )


# ---------------------------------------------------------------------------
# claims_affected_by_event
# ---------------------------------------------------------------------------


class TestClaimsAffectedByEvent:
    def test_cve_matches_by_substring(self):
        unit = _unit(["env.libfoo.version_pinned", "env.curl.ok"])
        affected = claims_affected_by_event(unit, _cve(["libfoo"]))
        assert "env.libfoo.version_pinned" in affected
        assert "env.curl.ok" not in affected

    def test_api_change_matches_by_prefix(self):
        unit = _unit(["openai.completions.v1.reachable", "anthropic.api.reachable"])
        affected = claims_affected_by_event(unit, _api_event())
        assert "openai.completions.v1.reachable" in affected
        assert "anthropic.api.reachable" not in affected

    def test_dependency_bump_matches_dot_prefix(self):
        unit = _unit(["pydantic.model.dict_supported", "sqlalchemy.ok"])
        affected = claims_affected_by_event(unit, _dep_event())
        assert "pydantic.model.dict_supported" in affected
        assert "sqlalchemy.ok" not in affected

    def test_empty_scope_matches_nothing(self):
        event = WorldStateEvent(
            event_id="noop", kind=WorldEventKind.CORPUS_REVISION,
            description="no scope", affected_scope=[],
        )
        assert claims_affected_by_event(_unit(["claim.a"]), event) == frozenset()

    def test_multiple_patterns_match_union(self):
        unit = _unit(["libfoo.pinned", "curl.version", "libz.ok"])
        event = WorldStateEvent(
            event_id="multi", kind=WorldEventKind.CVE,
            description="Multiple vulns", affected_scope=["libfoo", "curl"],
        )
        affected = claims_affected_by_event(unit, event)
        assert "libfoo.pinned" in affected and "curl.version" in affected
        assert "libz.ok" not in affected


# ---------------------------------------------------------------------------
# world_event_to_claim_results
# ---------------------------------------------------------------------------


class TestWorldEventToClaimResults:
    def test_affected_claim_gets_stale_status(self):
        unit = _unit(["libfoo.version_pinned", "other.claim"])
        results = world_event_to_claim_results(unit, _cve(["libfoo"]), require_enabled=False)
        assert results["libfoo.version_pinned"].status == ClaimStatus.STALE
        assert "other.claim" not in results

    def test_message_contains_event_id_and_kind(self):
        unit = _unit(["libfoo.pinned"])
        results = world_event_to_claim_results(unit, _cve(["libfoo"], "CVE-2024-9999"), require_enabled=False)
        msg = results["libfoo.pinned"].message
        assert "CVE-2024-9999" in msg and "cve" in msg

    def test_empty_scope_returns_empty_dict(self):
        event = WorldStateEvent(event_id="noop", kind=WorldEventKind.CORPUS_REVISION, description="x")
        assert world_event_to_claim_results(_unit(["c"]), event, require_enabled=False) == {}

    def test_flag_off_raises_by_default(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_WORLD_EVENTS_ENABLED", raising=False)
        with pytest.raises(RuntimeError, match="ARAGORA_WORLD_EVENTS_ENABLED"):
            world_event_to_claim_results(_unit(["libfoo.p"]), _cve(["libfoo"]))

    def test_require_enabled_false_bypasses_flag(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_WORLD_EVENTS_ENABLED", raising=False)
        results = world_event_to_claim_results(_unit(["libfoo.p"]), _cve(["libfoo"]), require_enabled=False)
        assert "libfoo.p" in results

    def test_flag_on_allows_normal_call(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_WORLD_EVENTS_ENABLED", "1")
        results = world_event_to_claim_results(_unit(["libfoo.p"]), _cve(["libfoo"]))
        assert "libfoo.p" in results

    def test_world_events_enabled_truth_values(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_WORLD_EVENTS_ENABLED", "true")
        assert world_events_enabled() is True
        monkeypatch.delenv("ARAGORA_WORLD_EVENTS_ENABLED")
        assert world_events_enabled() is False


# ---------------------------------------------------------------------------
# Integration: world event → evaluate_unit (DIC-20 end-to-end)
# ---------------------------------------------------------------------------


class TestWorldEventDecayIntegration:
    def test_cve_lowers_integrity_score(self):
        unit = _unit(["libfoo.version_pinned", "libfoo.no_known_vulns"])
        results = world_event_to_claim_results(unit, _cve(["libfoo"]), require_enabled=False)
        signal = evaluate_unit(unit, claim_results=results)
        assert signal.integrity_score < 1.0
        assert any(r.kind == "stale_evidence" for r in signal.reasons)

    def test_api_change_marks_specific_claim_stale(self):
        unit = _unit(["openai.completions.v1.reachable", "openai.embeddings.ok"])
        results = world_event_to_claim_results(unit, _api_event(), require_enabled=False)
        stale = [r for r in evaluate_unit(unit, claim_results=results).reasons if r.kind == "stale_evidence"]
        assert any(r.claim_id == "openai.completions.v1.reachable" for r in stale)

    def test_dependency_bump_propagates_decay(self):
        unit = _unit(["pydantic.model.dict_supported"])
        results = world_event_to_claim_results(unit, _dep_event(), require_enabled=False)
        assert any(r.kind == "stale_evidence" for r in evaluate_unit(unit, claim_results=results).reasons)

    def test_unrelated_event_leaves_score_intact(self):
        unit = _unit(["anthropic.api.stable", "project.build.passing"])
        results = world_event_to_claim_results(unit, _cve(["libfoo"]), require_enabled=False)
        assert evaluate_unit(unit, claim_results=results).integrity_score == 1.0
