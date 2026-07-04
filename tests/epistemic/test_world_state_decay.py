"""Synthetic world-state event tests for DIC-20 / #6031.

Verifies that external events (CVE drops, API changes, dependency
version bumps) correctly translate into ClaimResult decay and
propagate through evaluate_unit, satisfying the DIC-20 acceptance
criterion for synthetic world-event invalidation tests.
"""

from __future__ import annotations

import inspect
import os

import pytest

import aragora.epistemic.world_event as _world_event_module
from aragora.epistemic.claim_verifier import ClaimStatus
from aragora.epistemic.decay_monitor import evaluate_unit
from aragora.epistemic.proof_unit import DecayPolicy, FallbackPolicy, ProofCarryingCodeUnit
from aragora.epistemic.world_event import (
    WorldEventKind,
    WorldStateEvent,
    _world_event_to_claim_results_unchecked,
    claims_affected_by_event,
    enable_world_events,
    reset_world_events,
    world_event_to_claim_results,
    world_events_enabled,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_world_event_override() -> pytest.IterableFixture:  # type: ignore[type-arg]
    reset_world_events()
    yield
    reset_world_events()


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
    def test_scope_matching_ignores_overbroad_patterns(self):
        unit = _unit(
            [
                "openai.completions.v1.reachable",
                "anthropic.api.reachable",
                "project.build.passing",
            ]
        )
        event = WorldStateEvent(
            event_id="too-broad",
            kind=WorldEventKind.API_CHANGE,
            description="broad event",
            affected_scope=[".", "v1", "api"],
        )

        assert claims_affected_by_event(unit, event) == frozenset()

    def test_scope_matching_uses_claim_id_boundaries(self):
        unit = _unit(
            [
                "env.libfoo.version_pinned",
                "libfoo.pinned",
                "libfoobar.pinned",
            ]
        )

        affected = claims_affected_by_event(unit, _cve(["libfoo"]))

        assert affected == frozenset({"env.libfoo.version_pinned", "libfoo.pinned"})

    def test_scope_matching_does_not_match_inside_segments(self):
        unit = _unit(
            [
                "runtime.py.version",
                "env.py.version",
                "studio.python.version",
            ]
        )
        event = WorldStateEvent(
            event_id="py-runtime",
            kind=WorldEventKind.DEPENDENCY_BUMP,
            description="Python runtime patch",
            affected_scope=["py"],
        )

        affected = claims_affected_by_event(unit, event)

        assert affected == frozenset({"runtime.py.version", "env.py.version"})

    def test_generic_short_scope_tokens_match_nothing(self):
        unit = _unit(
            [
                "env.jwt.version_pinned",
                "transport.ssl.enabled",
                "api.openai.available",
                "language.go.mod_current",
                "site.com.reachable",
            ]
        )
        event = WorldStateEvent(
            event_id="generic-short-scopes",
            kind=WorldEventKind.DEPENDENCY_BUMP,
            description="Noisy external feed with generic short scopes",
            affected_scope=["jwt", "ssl", "api", "go", "com"],
        )

        assert claims_affected_by_event(unit, event) == frozenset()

    def test_version_like_scope_tokens_match_nothing(self):
        unit = _unit(
            [
                "openai.completions.v1.reachable",
                "runtime.v1beta.compatible",
                "runtime.v2rc1.compatible",
            ]
        )
        event = WorldStateEvent(
            event_id="version-like-scopes",
            kind=WorldEventKind.API_CHANGE,
            description="Version-like scopes are too broad for claim invalidation",
            affected_scope=["v1", "v1beta", "v2rc1"],
        )

        assert claims_affected_by_event(unit, event) == frozenset()

    def test_scope_matching_is_case_insensitive(self):
        unit = _unit(
            [
                "libfoo.version_pinned",
                "env.libfoo.pinned",
                "openai.completions.v1.reachable",
                "anthropic.api.reachable",
            ]
        )
        event = WorldStateEvent(
            event_id="mixed-case-scope",
            kind=WorldEventKind.API_CHANGE,
            description="Externally supplied scopes can vary in case",
            affected_scope=["LibFoo", "OpenAI.Completions"],
        )

        affected = claims_affected_by_event(unit, event)

        assert affected == frozenset(
            {
                "libfoo.version_pinned",
                "env.libfoo.pinned",
                "openai.completions.v1.reachable",
            }
        )

    def test_cve_matches_by_segment_boundary(self):
        unit = _unit(["env.libfoo.version_pinned", "env.curl.ok"])
        affected = claims_affected_by_event(unit, _cve(["libfoo"]))
        assert "env.libfoo.version_pinned" in affected
        assert "env.curl.ok" not in affected

    def test_short_scope_tokens_match_claim_id_boundaries(self):
        unit = _unit(
            [
                "aws.iam.policy.current",
                "build.npm.lockfile.current",
                "language.golang.mod_current",
                "cargo.good",
                "project.api.available",
            ]
        )
        event = WorldStateEvent(
            event_id="short-scopes",
            kind=WorldEventKind.DEPENDENCY_BUMP,
            description="short but specific dependency/API scopes",
            affected_scope=["aws.iam", "npm.lockfile", "golang", "api"],
        )

        assert claims_affected_by_event(unit, event) == frozenset(
            {
                "aws.iam.policy.current",
                "build.npm.lockfile.current",
                "language.golang.mod_current",
            }
        )

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
            event_id="noop",
            kind=WorldEventKind.CORPUS_REVISION,
            description="no scope",
            affected_scope=[],
        )
        assert claims_affected_by_event(_unit(["claim.a"]), event) == frozenset()

    def test_multiple_patterns_match_union(self):
        unit = _unit(["libfoo.pinned", "curl.version", "libz.ok"])
        event = WorldStateEvent(
            event_id="multi",
            kind=WorldEventKind.CVE,
            description="Multiple vulns",
            affected_scope=["libfoo", "curl"],
        )
        affected = claims_affected_by_event(unit, event)
        assert "libfoo.pinned" in affected and "curl.version" in affected
        assert "libz.ok" not in affected


# ---------------------------------------------------------------------------
# world_event_to_claim_results
# ---------------------------------------------------------------------------


class TestWorldEventToClaimResults:
    def test_string_kind_normalizes_to_world_event_kind(self):
        event = WorldStateEvent(
            event_id="CVE-2024-9999",
            kind="cve",
            description="Critical issue in libfoo",
            affected_scope=["libfoo"],
        )

        assert event.kind is WorldEventKind.CVE
        assert event.to_dict()["kind"] == "cve"

    def test_affected_scope_is_copied_to_immutable_tuple(self):
        scope = ["libfoo"]
        event = WorldStateEvent(
            event_id="CVE-2024-9999",
            kind="cve",
            description="Critical issue in libfoo",
            affected_scope=scope,
        )

        scope.append("curl")

        assert event.affected_scope == ("libfoo",)
        assert event.to_dict()["affected_scope"] == ["libfoo"]

    def test_invalid_string_kind_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported world event kind"):
            WorldStateEvent(event_id="bad", kind="not-a-kind", description="bad")

    def test_public_translation_has_no_flag_bypass_parameter(self):
        assert "require_enabled" not in inspect.signature(world_event_to_claim_results).parameters

    def test_affected_claim_gets_stale_status(self):
        unit = _unit(["libfoo.version_pinned", "other.claim"])
        results = _world_event_to_claim_results_unchecked(unit, _cve(["libfoo"]))
        assert results["libfoo.version_pinned"].status == ClaimStatus.STALE
        assert results["libfoo.version_pinned"].detail["source"] == "world_event"
        assert "other.claim" not in results

    def test_message_contains_event_id_and_kind(self):
        unit = _unit(["libfoo.pinned"])
        results = _world_event_to_claim_results_unchecked(unit, _cve(["libfoo"], "CVE-2024-9999"))
        msg = results["libfoo.pinned"].message
        assert "CVE-2024-9999" in msg and "cve" in msg

    def test_empty_scope_returns_empty_dict(self):
        event = WorldStateEvent(
            event_id="noop", kind=WorldEventKind.CORPUS_REVISION, description="x"
        )
        assert _world_event_to_claim_results_unchecked(_unit(["c"]), event) == {}

    def test_flag_off_raises_by_default(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_WORLD_EVENTS_ENABLED", raising=False)
        with pytest.raises(RuntimeError, match="ARAGORA_WORLD_EVENTS_ENABLED"):
            world_event_to_claim_results(_unit(["libfoo.p"]), _cve(["libfoo"]))

    def test_internal_unchecked_helper_bypasses_flag_for_tests(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_WORLD_EVENTS_ENABLED", raising=False)
        results = _world_event_to_claim_results_unchecked(_unit(["libfoo.p"]), _cve(["libfoo"]))
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

    def test_enable_helper_sets_override_without_mutating_environ(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_WORLD_EVENTS_ENABLED", raising=False)
        before = dict(os.environ)

        enable_world_events()

        assert world_events_enabled() is True
        assert _world_event_module._world_events_enabled_override is True  # noqa: SLF001
        assert os.environ == before

    def test_reset_helper_clears_override(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_WORLD_EVENTS_ENABLED", raising=False)
        enable_world_events()
        reset_world_events()

        assert _world_event_module._world_events_enabled_override is None  # noqa: SLF001
        assert world_events_enabled() is False


# ---------------------------------------------------------------------------
# Integration: world event → evaluate_unit (DIC-20 end-to-end)
# ---------------------------------------------------------------------------


class TestWorldEventDecayIntegration:
    def test_cve_lowers_integrity_score(self):
        unit = _unit(["libfoo.version_pinned", "libfoo.no_known_vulns"])
        enable_world_events()
        results = _world_event_to_claim_results_unchecked(unit, _cve(["libfoo"]))
        signal = evaluate_unit(unit, claim_results=results)
        assert signal.integrity_score < 1.0
        assert any(r.kind == "stale_evidence" for r in signal.reasons)

    def test_api_change_marks_specific_claim_stale(self):
        unit = _unit(["openai.completions.v1.reachable", "openai.embeddings.ok"])
        enable_world_events()
        results = _world_event_to_claim_results_unchecked(unit, _api_event())
        stale = [
            r
            for r in evaluate_unit(unit, claim_results=results).reasons
            if r.kind == "stale_evidence"
        ]
        assert any(r.claim_id == "openai.completions.v1.reachable" for r in stale)

    def test_dependency_bump_propagates_decay(self):
        unit = _unit(["pydantic.model.dict_supported"])
        enable_world_events()
        results = _world_event_to_claim_results_unchecked(unit, _dep_event())
        assert any(
            r.kind == "stale_evidence" for r in evaluate_unit(unit, claim_results=results).reasons
        )

    def test_unchecked_world_event_results_fail_closed_when_disabled(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_WORLD_EVENTS_ENABLED", raising=False)
        unit = _unit(["libfoo.version_pinned"])
        results = _world_event_to_claim_results_unchecked(unit, _cve(["libfoo"]))

        with pytest.raises(RuntimeError, match="world-event claim result"):
            evaluate_unit(unit, claim_results=results)

    def test_enabled_public_translation_propagates_decay(self):
        unit = _unit(["libfoo.version_pinned", "other.claim"])
        enable_world_events()

        results = world_event_to_claim_results(unit, _cve(["libfoo"]))
        signal = evaluate_unit(unit, claim_results=results)

        assert "libfoo.version_pinned" in results
        assert signal.integrity_score < 1.0
        assert any(r.kind == "stale_evidence" for r in signal.reasons)

    def test_unrelated_event_leaves_score_intact(self):
        unit = _unit(["anthropic.api.stable", "project.build.passing"])
        enable_world_events()
        results = _world_event_to_claim_results_unchecked(unit, _cve(["libfoo"]))
        assert evaluate_unit(unit, claim_results=results).integrity_score == 1.0
