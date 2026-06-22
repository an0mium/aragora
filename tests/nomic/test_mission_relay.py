"""Tests for the relay-with-timeout decision core (pure, deterministic)."""

from __future__ import annotations

import pytest

from aragora.nomic.mission_relay import (
    RelayAction,
    RelayContext,
    RelayDecision,
    RelayPolicy,
    decide_relay_action,
)


def test_healthy_item_continues() -> None:
    d = decide_relay_action(RelayPolicy(), RelayContext(item_id="a"))
    assert d.action is RelayAction.CONTINUE
    assert d.notify is False


def test_needs_human_parks_not_stops() -> None:
    # The core behavior: a blocked item PARKS (mission continues others), never STOPS.
    d = decide_relay_action(
        RelayPolicy(relay_channel="slack"), RelayContext(item_id="a", needs_human=True)
    )
    assert d.action is RelayAction.PARK_AND_NOTIFY
    assert d.notify is True


def test_park_notify_false_when_channel_none() -> None:
    d = decide_relay_action(
        RelayPolicy(relay_channel="none"), RelayContext(item_id="a", needs_human=True)
    )
    assert d.action is RelayAction.PARK_AND_NOTIFY
    assert d.notify is False  # parked, but nowhere to notify


def test_failures_threshold_parks() -> None:
    pol = RelayPolicy(max_item_failures=3, relay_channel="email")
    assert (
        decide_relay_action(pol, RelayContext(item_id="a", consecutive_failures=2)).action
        is RelayAction.CONTINUE
    )
    d = decide_relay_action(pol, RelayContext(item_id="a", consecutive_failures=3))
    assert d.action is RelayAction.PARK_AND_NOTIFY and d.notify is True


def test_item_timeout_parks() -> None:
    pol = RelayPolicy(item_timeout_seconds=120.0)
    assert (
        decide_relay_action(pol, RelayContext(item_id="a", elapsed_seconds=119.0)).action
        is RelayAction.CONTINUE
    )
    assert (
        decide_relay_action(pol, RelayContext(item_id="a", elapsed_seconds=120.0)).action
        is RelayAction.PARK_AND_NOTIFY
    )


def test_no_timeout_when_unset() -> None:
    d = decide_relay_action(
        RelayPolicy(item_timeout_seconds=None), RelayContext(item_id="a", elapsed_seconds=1e9)
    )
    assert d.action is RelayAction.CONTINUE


def test_budget_exhausted_stops_mission() -> None:
    d = decide_relay_action(RelayPolicy(), RelayContext(item_id="a", budget_exhausted=True))
    assert d.action is RelayAction.STOP_MISSION


def test_max_hours_exceeded_stops_mission() -> None:
    d = decide_relay_action(RelayPolicy(), RelayContext(item_id="a", max_hours_exceeded=True))
    assert d.action is RelayAction.STOP_MISSION


def test_global_limit_precedes_item_block() -> None:
    # Budget exhaustion wins even if the item also needs a human.
    d = decide_relay_action(
        RelayPolicy(), RelayContext(item_id="a", needs_human=True, budget_exhausted=True)
    )
    assert d.action is RelayAction.STOP_MISSION


def test_no_action_ever_merges_or_settles() -> None:
    # Invariant: the action vocabulary is exactly {continue, park, stop} — nothing
    # that could merge/settle/mark-ready. Guards against future scope creep.
    assert {a.value for a in RelayAction} == {"continue", "park_and_notify", "stop_mission"}


def test_policy_validation() -> None:
    with pytest.raises(ValueError, match="max_item_failures"):
        RelayPolicy(max_item_failures=0)
    with pytest.raises(ValueError, match="item_timeout_seconds"):
        RelayPolicy(item_timeout_seconds=0.0)
    with pytest.raises(ValueError, match="relay_channel"):
        RelayPolicy(relay_channel="fax")


def test_decision_to_dict() -> None:
    d: RelayDecision = decide_relay_action(RelayPolicy(), RelayContext(item_id="a"))
    assert d.to_dict() == {"action": "continue", "reason": "item healthy", "notify": False}


def test_deterministic() -> None:
    pol, ctx = RelayPolicy(relay_channel="slack"), RelayContext(item_id="a", needs_human=True)
    assert decide_relay_action(pol, ctx) == decide_relay_action(pol, ctx)
