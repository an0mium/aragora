"""Tests for the domain-free cross-subscriber registry + layered bootstrap (P4a E1).

Covers the enabler surface introduced by the EventBus/Job-Queue registry
inversion (docs/architecture/P4A_EVENTS_QUEUE_INVERSION.md §4.1, §4.4):

- ``register_subscriber`` / ``register_factory`` / ``get_registered_subscribers``
  / ``reset_registry`` on ``aragora.events.cross_subscribers``.
- The domain-free ``bootstrap()`` that wires registered subscribers into the
  manager (idempotent).
- The layered composition roots: the domain-subset
  ``aragora.debate.event_subscribers.bootstrap_debate_event_subscribers`` and the
  interface superset ``aragora.server.startup.event_subscribers.bootstrap_event_subscribers``.
- The registration-completeness safeguard: the superset bootstrap must register
  the full pre-inversion subscriber set (parity, no silent drops).
- The registry surface must carry ZERO domain imports.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aragora.events.cross_subscribers import (
    bootstrap,
    get_cross_subscriber_manager,
    get_registered_subscribers,
    register_factory,
    register_subscriber,
    registered_subscriber_names,
    reset_cross_subscriber_manager,
    reset_registry,
)
from aragora.events.types import StreamEventType

# Pre-inversion registered-subscriber set, captured on origin/main @708d116ed2
# (before E2 begins). The superset bootstrap must keep parity with this set; any
# silent drop as handlers relocate in E2-E7 fails the batch (§4.4 safeguard).
GOLDEN_SUBSCRIBER_NAMES = frozenset(
    {
        "agent_birth_to_control_plane",
        "agent_death_to_control_plane",
        "agent_evolution_to_control_plane",
        "agent_message_to_rhetorical",
        "alert_escalated_to_workflow_brake",
        "approval_to_km_reinforcement",
        "belief_to_mound",
        "budget_alert_to_team_selection",
        "calibration_to_agent",
        "consensus_to_learning",
        "consensus_to_mound",
        "culture_to_debate",
        "debate_end_to_cost_tracking",
        "debate_end_to_explainability",
        "debate_outcome_to_knowledge",
        "elo_to_debate",
        "elo_to_mound",
        "evidence_to_insight",
        "flip_to_mound",
        "gauntlet_to_notification",
        "insight_to_mound",
        "km_validation_feedback",
        "knowledge_to_memory",
        "memory_to_mound",
        "memory_to_rlm",
        "meta_learning_to_team_selection",
        "mound_to_belief",
        "mound_to_culture",
        "mound_to_memory",
        "mound_to_memory_retrieval",
        "mound_to_provenance",
        "mound_to_rlm",
        "mound_to_team_selection",
        "mound_to_trickster",
        "provenance_to_mound",
        "risk_warning_to_health",
        "rlm_to_mound",
        "staleness_to_debate",
        "tier_demotion_to_revalidation",
        "tier_promotion_to_knowledge",
        "vote_to_belief",
        "webhook_agent_elo_updated",
        "webhook_calibration_update",
        "webhook_evidence_found",
        "webhook_knowledge_indexed",
        "webhook_knowledge_queried",
        "webhook_memory_retrieved",
        "webhook_memory_stored",
        "webhook_mound_updated",
        "workflow_complete_to_supermemory",
        "workflow_failed_to_supermemory",
    }
)

# Handlers relocated OUT of the manager's built-in set into their coupled home
# modules by the P4a EventBus inversion (E2-E7). The bare manager no longer
# registers these; each self-registers via its home module plus a bootstrap. The
# superset bootstrap must still yield full GOLDEN parity (see the parity test).
# E2a: knowledge_mound reactions -> aragora/knowledge/event_subscribers.py.
# E2b: validation/consensus/provenance KM reactions -> the same knowledge home.
# E2c: the knowledge-coupled reactions embedded in the basic/culture/strategic
# mixins -> the same knowledge home; this closes out events->knowledge.
RELOCATED_SUBSCRIBER_NAMES = frozenset(
    {
        "memory_to_mound",
        "mound_to_memory_retrieval",
        "belief_to_mound",
        "mound_to_belief",
        "rlm_to_mound",
        "mound_to_rlm",
        "elo_to_mound",
        "mound_to_team_selection",
        "insight_to_mound",
        "flip_to_mound",
        "mound_to_trickster",
        "provenance_to_mound",
        "mound_to_provenance",
        "consensus_to_mound",
        "km_validation_feedback",
        "mound_to_culture",
        "debate_outcome_to_knowledge",
        "workflow_complete_to_supermemory",
        "workflow_failed_to_supermemory",
        "tier_demotion_to_revalidation",
        "tier_promotion_to_knowledge",
        "approval_to_km_reinforcement",
    }
)


class _FakeSubscriber:
    """Minimal subscriber that wires one handler into the manager on register."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.register_calls = 0

    def register(self, manager: object) -> None:
        self.register_calls += 1
        manager.register(self.name, StreamEventType.DEBATE_START, lambda event: None)


@pytest.fixture(autouse=True)
def _clean_registry_and_manager():
    """Isolate each test: empty registry + fresh manager before and after."""
    reset_registry()
    reset_cross_subscriber_manager()
    yield
    reset_registry()
    reset_cross_subscriber_manager()


def test_register_subscriber_and_get():
    sub = _FakeSubscriber("test_fake_subscriber")
    register_subscriber("test_fake_subscriber", sub)

    registered = get_registered_subscribers()
    assert registered["test_fake_subscriber"] is sub
    assert "test_fake_subscriber" in registered_subscriber_names()


def test_register_subscriber_is_keyed_idempotent():
    first = _FakeSubscriber("dup")
    second = _FakeSubscriber("dup")
    register_subscriber("dup", first)
    register_subscriber("dup", second)

    registered = get_registered_subscribers()
    assert registered["dup"] is second
    assert registered_subscriber_names().count("dup") == 1


def test_register_factory_materializes_once():
    calls = {"n": 0}

    def factory() -> _FakeSubscriber:
        calls["n"] += 1
        return _FakeSubscriber("from_factory")

    register_factory("from_factory", factory)

    first = get_registered_subscribers()["from_factory"]
    second = get_registered_subscribers()["from_factory"]
    assert first is second
    assert calls["n"] == 1


def test_reset_registry_clears():
    register_subscriber("gone", _FakeSubscriber("gone"))
    assert "gone" in get_registered_subscribers()

    reset_registry()
    assert get_registered_subscribers() == {}
    assert registered_subscriber_names() == []


def test_bootstrap_applies_registered_subscriber():
    sub = _FakeSubscriber("wired_via_bootstrap")
    register_subscriber("wired_via_bootstrap", sub)

    manager = bootstrap()

    assert sub.register_calls == 1
    assert "wired_via_bootstrap" in manager.get_stats()


def test_bootstrap_is_idempotent():
    sub = _FakeSubscriber("only_once")
    register_subscriber("only_once", sub)

    bootstrap()
    manager = bootstrap()
    bootstrap()

    # register() runs at most once per manager instance even across repeated
    # bootstrap calls (registration is keyed and application is tracked).
    assert sub.register_calls == 1
    assert "only_once" in manager.get_stats()


def test_get_cross_subscriber_manager_path_unchanged():
    """Acceptance #3: the public accessor stays at its historical import path."""
    import importlib

    module = importlib.import_module("aragora.events.cross_subscribers")
    assert hasattr(module, "get_cross_subscriber_manager")
    assert module.get_cross_subscriber_manager is get_cross_subscriber_manager

    manager1 = get_cross_subscriber_manager()
    manager2 = get_cross_subscriber_manager()
    assert manager1 is manager2


def test_builtin_handlers_still_registered_via_manager():
    """The bare manager registers every NON-relocated built-in handler.

    After the P4a inversion (E2+), relocated reactions self-register via their home
    module plus a bootstrap, so constructing the manager alone must (a) still
    register every built-in that has NOT been relocated and (b) NOT register any
    relocated one (proving the reaction truly left infrastructure ``events``).
    Full parity is covered by ``test_superset_bootstrap_completeness_parity``.
    """
    manager = get_cross_subscriber_manager()
    registered = set(manager.get_stats())

    still_builtin = GOLDEN_SUBSCRIBER_NAMES - RELOCATED_SUBSCRIBER_NAMES
    missing = still_builtin - registered
    assert not missing, f"non-relocated builtin subscribers dropped: {sorted(missing)}"

    leaked = RELOCATED_SUBSCRIBER_NAMES & registered
    assert not leaked, f"relocated handlers still built into the bare manager: {sorted(leaked)}"


def test_direct_manager_does_not_implicitly_apply_relocated_home_subscribers():
    """Direct construction must not depend on prior home-module import order."""
    from aragora.events.cross_subscribers import CrossSubscriberManager
    from aragora.knowledge import event_subscribers as knowledge_home

    knowledge_home.register()

    manager = CrossSubscriberManager()
    registered = set(manager.get_stats())

    leaked = RELOCATED_SUBSCRIBER_NAMES & registered
    assert not leaked, (
        "direct manager construction implicitly applied relocated handlers; "
        f"use an explicit bootstrap instead: {sorted(leaked)}"
    )


def test_domain_bootstrap_fails_closed_when_knowledge_home_registration_is_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    """A missing knowledge-home registration must fail instead of silently dropping KM."""
    from aragora.debate.event_subscribers import bootstrap_debate_event_subscribers
    from aragora.events.cross_subscribers import reset_cross_subscriber_manager, reset_registry
    from aragora.knowledge import event_subscribers as knowledge_home

    reset_registry()
    reset_cross_subscriber_manager()
    monkeypatch.setattr(knowledge_home, "register", lambda: None)

    with pytest.raises(RuntimeError, match="Knowledge event subscriber bootstrap incomplete"):
        bootstrap_debate_event_subscribers()


def test_domain_subset_bootstrap_returns_manager():
    from aragora.debate.event_subscribers import bootstrap_debate_event_subscribers

    manager = bootstrap_debate_event_subscribers()
    assert manager is get_cross_subscriber_manager()


def test_superset_bootstrap_completeness_parity():
    """Registration-completeness safeguard (§4.4): superset bootstrap must
    register the FULL pre-inversion subscriber set with no silent drops."""
    from aragora.server.startup.event_subscribers import bootstrap_event_subscribers

    manager = bootstrap_event_subscribers()
    registered = set(manager.get_stats())

    missing = GOLDEN_SUBSCRIBER_NAMES - registered
    assert not missing, f"superset bootstrap dropped subscribers: {sorted(missing)}"


def test_relocated_reactions_registered_via_home_bootstrap():
    """E2+: relocated reactions self-register through their domain home module.

    The domain-subset bootstrap imports the domain home modules; every relocated
    name must then be wired into the manager (parity for the relocated slice, so a
    silently dropped home import is caught here and not only in the superset test).
    """
    from aragora.debate.event_subscribers import bootstrap_debate_event_subscribers

    manager = bootstrap_debate_event_subscribers()
    registered = set(manager.get_stats())

    missing = RELOCATED_SUBSCRIBER_NAMES - registered
    assert not missing, f"home bootstrap failed to wire relocated reactions: {sorted(missing)}"


def test_registry_module_has_zero_domain_imports():
    """The registry surface must not import any domain-layer package (§4.1).

    Static source guard complementing the grimp full-layer re-check: the registry
    module carries zero domain imports (eager OR lazy).
    """
    import aragora.events.cross_subscribers.registry as registry_module

    source = Path(registry_module.__file__).read_text(encoding="utf-8")
    domain_packages = (
        "debate",
        "agents",
        "memory",
        "knowledge",
        "ranking",
        "reasoning",
        "evidence",
        "evaluation",
        "explainability",
        "learning",
        "ml",
    )
    offenders = [pkg for pkg in domain_packages if f"aragora.{pkg}" in source]
    assert not offenders, f"registry.py imports domain packages: {offenders}"
