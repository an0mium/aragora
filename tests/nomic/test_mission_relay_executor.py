"""Tests for the relay executor (park + notify) and MissionStore.set_item_status."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aragora.nomic.mission import MissionSpec, MissionStore, WorkItem, WorkItemStatus
from aragora.nomic.mission_relay import RelayAction, RelayContext, RelayPolicy
from aragora.nomic.mission_relay_executor import MissionRelay, RelayOutcome


def _store(tmp_path: Path) -> MissionStore:
    return MissionStore(state_dir=tmp_path)


def _seed(store: MissionStore, mission_id: str = "m1") -> None:
    spec = MissionSpec(goal="ship it", mission_id=mission_id)
    items = [
        WorkItem(item_id="i1", description="first"),
        WorkItem(item_id="i2", description="second"),
    ]
    store.save_mission(spec, items)


class FakeNotifier:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def notify(self, notification, channels=None):
        self.calls.append((notification, channels))
        return []


# --- MissionStore.set_item_status ------------------------------------------


def test_set_item_status_parks_one_item_leaves_others(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store)
    assert store.set_item_status("m1", "i1", WorkItemStatus.PARKED) is True
    _, items = store.load_mission("m1")
    by_id = {i.item_id: i.status for i in items}
    assert by_id["i1"] is WorkItemStatus.PARKED
    assert by_id["i2"] is WorkItemStatus.PENDING  # untouched


def test_set_item_status_missing_mission_returns_false(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.set_item_status("nope", "i1", WorkItemStatus.PARKED) is False


def test_set_item_status_missing_item_returns_false(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store)
    assert store.set_item_status("m1", "ghost", WorkItemStatus.PARKED) is False


# --- MissionRelay.evaluate (pure delegation) -------------------------------


def test_evaluate_delegates_to_core(tmp_path: Path) -> None:
    relay = MissionRelay(RelayPolicy(), _store(tmp_path))
    assert relay.evaluate(RelayContext(item_id="i1")).action is RelayAction.CONTINUE
    needs_human = RelayContext(item_id="i1", needs_human=True)
    assert relay.evaluate(needs_human).action is RelayAction.PARK_AND_NOTIFY


# --- MissionRelay.apply ----------------------------------------------------


def test_apply_park_marks_parked_and_notifies(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store)
    notifier = FakeNotifier()
    relay = MissionRelay(RelayPolicy(relay_channel="slack"), store, notifier=notifier)
    decision = relay.evaluate(RelayContext(item_id="i1", needs_human=True))
    outcome = asyncio.run(relay.apply("m1", "i1", decision))

    assert isinstance(outcome, RelayOutcome)
    assert outcome.action is RelayAction.PARK_AND_NOTIFY
    assert outcome.parked is True
    assert outcome.notified is True
    assert outcome.stopped is False
    # Persisted.
    _, items = store.load_mission("m1")
    assert {i.item_id: i.status for i in items}["i1"] is WorkItemStatus.PARKED
    # Notified once, on the slack channel.
    assert len(notifier.calls) == 1
    _, channels = notifier.calls[0]
    assert [c.value for c in channels] == ["slack"]


def test_apply_stop_notifies_but_does_not_park(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store)
    notifier = FakeNotifier()
    relay = MissionRelay(RelayPolicy(relay_channel="email"), store, notifier=notifier)
    decision = relay.evaluate(RelayContext(item_id="i1", budget_exhausted=True))
    outcome = asyncio.run(relay.apply("m1", "i1", decision))

    assert outcome.action is RelayAction.STOP_MISSION
    assert outcome.stopped is True
    assert outcome.parked is False
    assert outcome.notified is True
    # The triggering item is NOT parked (global limit, not an item fault).
    _, items = store.load_mission("m1")
    assert {i.item_id: i.status for i in items}["i1"] is WorkItemStatus.PENDING
    assert len(notifier.calls) == 1


def test_apply_continue_is_noop(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store)
    notifier = FakeNotifier()
    relay = MissionRelay(RelayPolicy(relay_channel="slack"), store, notifier=notifier)
    decision = relay.evaluate(RelayContext(item_id="i1"))  # healthy
    outcome = asyncio.run(relay.apply("m1", "i1", decision))

    assert outcome.action is RelayAction.CONTINUE
    assert (outcome.parked, outcome.stopped, outcome.notified) == (False, False, False)
    assert notifier.calls == []
    _, items = store.load_mission("m1")
    assert {i.item_id: i.status for i in items}["i1"] is WorkItemStatus.PENDING


def test_no_notify_when_channel_none(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store)
    notifier = FakeNotifier()
    # relay_channel="none" -> core sets notify=False -> executor must not notify.
    relay = MissionRelay(RelayPolicy(relay_channel="none"), store, notifier=notifier)
    decision = relay.evaluate(RelayContext(item_id="i1", needs_human=True))
    outcome = asyncio.run(relay.apply("m1", "i1", decision))
    assert outcome.parked is True  # still parks
    assert outcome.notified is False
    assert notifier.calls == []


def test_no_crash_when_notifier_absent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store)
    relay = MissionRelay(RelayPolicy(relay_channel="slack"), store, notifier=None)
    decision = relay.evaluate(RelayContext(item_id="i1", needs_human=True))
    outcome = asyncio.run(relay.apply("m1", "i1", decision))
    assert outcome.parked is True
    assert outcome.notified is False


def test_park_failure_does_not_send_misleading_notify(tmp_path: Path) -> None:
    """If the item/mission vanished, a PARK that didn't take must not notify."""
    store = _store(tmp_path)
    _seed(store)
    notifier = FakeNotifier()
    relay = MissionRelay(RelayPolicy(relay_channel="slack"), store, notifier=notifier)
    decision = relay.evaluate(RelayContext(item_id="ghost", needs_human=True))
    outcome = asyncio.run(relay.apply("m1", "ghost", decision))  # item not in mission
    assert outcome.parked is False
    assert outcome.notified is False
    assert notifier.calls == []  # no "parked" alert about an item we didn't park


def test_evaluate_and_apply_end_to_end(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store)
    relay = MissionRelay(
        RelayPolicy(max_item_failures=2, relay_channel="slack"), store, notifier=FakeNotifier()
    )
    ctx = RelayContext(item_id="i2", consecutive_failures=2)  # hits the failure cap -> park
    outcome = asyncio.run(relay.evaluate_and_apply("m1", "i2", ctx))
    assert outcome.action is RelayAction.PARK_AND_NOTIFY
    assert outcome.parked is True
    _, items = store.load_mission("m1")
    assert {i.item_id: i.status for i in items}["i2"] is WorkItemStatus.PARKED


def test_invariant_executor_never_merges_or_settles() -> None:
    """The relay can park + notify only — never merge/settle/mark-ready (gate stays sole authority)."""
    import aragora.nomic.mission_relay_executor as mod

    src = Path(mod.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("settle", "merge", "mark_ready", "--admin", "gh pr merge"):
        # allow the word inside the module docstring's invariant statement, but not as a call.
        assert f"{forbidden}(" not in src, f"executor must not invoke {forbidden}"
