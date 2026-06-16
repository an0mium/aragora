"""Tests for the native mission contracts (pure data + adapter)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from aragora.config.feature_flags import FeatureFlagRegistry
from aragora.nomic.mission import (
    MissionSpec,
    MissionTransport,
    WorkItem,
    WorkItemStatus,
    work_items_from_subtasks,
)


def _spec(**kw: object) -> MissionSpec:
    base: dict[str, object] = {"goal": "improve X", "mission_id": "m1"}
    base.update(kw)
    return MissionSpec(**base)  # type: ignore[arg-type]


def test_minimal_spec_defaults() -> None:
    s = _spec()
    assert s.transport is MissionTransport.CLI  # subscriptions by default
    assert s.relay == "none"
    assert s.auto_settle_max_tier == 2
    assert s.budget_usd is None and s.max_hours is None


def test_spec_frozen() -> None:
    s = _spec()
    with pytest.raises(Exception):
        s.goal = "other"  # type: ignore[misc]


def test_empty_goal_and_id_rejected() -> None:
    with pytest.raises(ValueError, match="goal"):
        _spec(goal="   ")
    with pytest.raises(ValueError, match="mission_id"):
        _spec(mission_id="")


@pytest.mark.parametrize("bad", [-1.0])
def test_negative_budget_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="budget_usd"):
        _spec(budget_usd=bad)
    assert _spec(budget_usd=0.0).budget_usd == 0.0


def test_nonpositive_max_hours_rejected() -> None:
    with pytest.raises(ValueError, match="max_hours"):
        _spec(max_hours=0.0)
    assert _spec(max_hours=7.0).max_hours == 7.0


def test_invalid_relay_rejected() -> None:
    with pytest.raises(ValueError, match="relay"):
        _spec(relay="carrier-pigeon")
    for ok in ("none", "slack", "email"):
        assert _spec(relay=ok).relay == ok


@pytest.mark.parametrize("bad", [-1, 5])
def test_auto_settle_tier_range(bad: int) -> None:
    with pytest.raises(ValueError, match="auto_settle_max_tier"):
        _spec(auto_settle_max_tier=bad)


def test_spec_dict_roundtrip() -> None:
    s = _spec(
        acceptance_criteria=("tests green", "wheel builds"),
        budget_usd=12.5,
        max_hours=6.0,
        transport=MissionTransport.API,
        relay="slack",
        auto_settle_max_tier=1,
    )
    restored = MissionSpec.from_dict(s.to_dict())
    assert restored == s


def test_from_dict_tolerates_missing_optionals() -> None:
    s = MissionSpec.from_dict({"goal": "g", "mission_id": "m"})
    assert s.transport is MissionTransport.CLI and s.auto_settle_max_tier == 2


@dataclass
class _FakeSubTask:
    id: str
    title: str = ""
    description: str = ""
    estimated_complexity: str = "low"
    file_scope: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()


def test_work_items_from_subtasks_maps_fields_and_preserves_order() -> None:
    subs = [
        _FakeSubTask(
            id="a",
            description="do A",
            estimated_complexity="high",
            file_scope=("x.py",),
            dependencies=(),
        ),
        _FakeSubTask(id="b", title="B title", estimated_complexity="medium", dependencies=("a",)),
    ]
    items = work_items_from_subtasks(subs)
    assert [i.item_id for i in items] == ["a", "b"]
    assert items[0].description == "do A" and items[0].complexity == "high"
    assert items[0].file_scope == ("x.py",)
    assert items[1].description == "B title"  # falls back to title
    assert items[1].dependencies == ("a",)
    assert all(i.status is WorkItemStatus.PENDING for i in items)


def test_work_items_skips_idless_without_raising() -> None:
    items = work_items_from_subtasks([_FakeSubTask(id=""), _FakeSubTask(id="ok")])
    assert [i.item_id for i in items] == ["ok"]


def test_work_item_to_dict_shape() -> None:
    d = WorkItem(item_id="a", description="x").to_dict()
    assert set(d) == {
        "item_id",
        "description",
        "status",
        "complexity",
        "file_scope",
        "dependencies",
    }
    assert d["status"] == "pending"


def test_flag_registered_default_off(monkeypatch) -> None:
    monkeypatch.delenv("ARAGORA_ENABLE_NATIVE_MISSION", raising=False)
    reg = FeatureFlagRegistry()
    assert reg.is_enabled("enable_native_mission") is False
