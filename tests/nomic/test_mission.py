"""Tests for the native mission contracts (pure data + adapter)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from aragora.config.feature_flags import FeatureFlagRegistry
from aragora.nomic.mission import (
    MissionSpec,
    MissionTransport,
    WorkItem,
    WorkItemStatus,
    work_items_from_subtasks,
    MissionStore,
    NativeMissionRunner,
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


def test_work_item_from_dict_roundtrip() -> None:
    wi = WorkItem(
        item_id="wi-1",
        description="Write tests",
        status=WorkItemStatus.RUNNING,
        complexity="high",
        file_scope=("a.py", "b.py"),
        dependencies=("wi-0",),
    )
    restored = WorkItem.from_dict(wi.to_dict())
    assert restored == wi


def test_work_item_from_dict_defaults() -> None:
    wi = WorkItem.from_dict({"item_id": "wi-default", "description": "desc"})
    assert wi.status is WorkItemStatus.PENDING
    assert wi.complexity == "low"
    assert wi.file_scope == ()
    assert wi.dependencies == ()


def test_mission_store_lifecycle(tmp_path: Path) -> None:
    store = MissionStore(state_dir=tmp_path)
    spec = _spec(mission_id="mission-test")
    items = [
        WorkItem(item_id="sub-1", description="Subtask 1"),
        WorkItem(item_id="sub-2", description="Subtask 2"),
    ]

    # Save
    p = store.save_mission(spec, items)
    assert p.exists()
    assert p.name == "mission-test.json"

    # List
    assert store.list_missions() == ["mission-test"]

    # Load
    loaded = store.load_mission("mission-test")
    assert loaded is not None
    loaded_spec, loaded_items = loaded
    assert loaded_spec == spec
    assert len(loaded_items) == 2
    assert loaded_items[0].item_id == "sub-1"
    assert loaded_items[1].item_id == "sub-2"


def test_mission_store_path_sanitization(tmp_path: Path) -> None:
    store = MissionStore(state_dir=tmp_path)
    spec = _spec(mission_id="my-cool_mission-123")
    p = store.save_mission(spec, [])
    assert p.name == "my-cool_mission-123.json"

    with pytest.raises(ValueError, match="Invalid mission ID"):
        store.path_for("   ")

    with pytest.raises(ValueError, match="Invalid mission ID"):
        store.path_for("../../etc/passwd")


def test_runner_raises_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("ARAGORA_ENABLE_NATIVE_MISSION", "false")
    runner = NativeMissionRunner()
    spec = _spec()
    with pytest.raises(RuntimeError, match="Native mission orchestrator is disabled"):
        import asyncio

        asyncio.run(runner.ingest_mission(spec))


def test_runner_ingest_mission(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ARAGORA_ENABLE_NATIVE_MISSION", "true")

    mock_orch = MagicMock()
    mock_orch.decompose_goal = AsyncMock()

    dummy_subtasks = [
        _FakeSubTask(id="sub-1", title="Task 1", estimated_complexity="medium"),
        _FakeSubTask(id="sub-2", title="Task 2", estimated_complexity="low"),
    ]
    from aragora.nomic.task_decomposer import TaskDecomposition

    dummy_decomp = TaskDecomposition(
        original_task="Improve codebase",
        complexity_score=50,
        complexity_level="medium",
        should_decompose=True,
        subtasks=dummy_subtasks,
    )
    mock_orch.decompose_goal.return_value = dummy_decomp

    store = MissionStore(state_dir=tmp_path)
    runner = NativeMissionRunner(orchestrator=mock_orch, store=store)

    spec = _spec(goal="Improve codebase", mission_id="runner-test-mission")

    import asyncio

    items = asyncio.run(runner.ingest_mission(spec))

    mock_orch.decompose_goal.assert_called_once_with("Improve codebase", tracks=None)

    assert len(items) == 2
    assert items[0].item_id == "sub-1"
    assert items[0].complexity == "medium"
    assert items[1].item_id == "sub-2"
    assert items[1].complexity == "low"

    loaded = store.load_mission("runner-test-mission")
    assert loaded is not None
    loaded_spec, loaded_items = loaded
    assert loaded_spec == spec
    assert len(loaded_items) == 2
    assert loaded_items[0].item_id == "sub-1"


def test_runner_lazy_orchestrator(monkeypatch) -> None:
    runner = NativeMissionRunner()
    from aragora.nomic.autonomous_orchestrator import AutonomousOrchestrator

    assert isinstance(runner.orchestrator, AutonomousOrchestrator)
