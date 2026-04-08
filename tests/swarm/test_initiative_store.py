from __future__ import annotations

from aragora.swarm.initiative_models import (
    InitiativeCheckpoint,
    InitiativeMilestone,
    InitiativeRecord,
    InitiativeSlice,
)
from aragora.swarm.initiative_store import InitiativeStore


def _record(initiative_id: str, *, updated_at: str) -> InitiativeRecord:
    return InitiativeRecord(
        initiative_id=initiative_id,
        title=f"Title {initiative_id}",
        goal="Track roadmap work as a durable initiative.",
        rationale="Founders need a persistent planning artifact.",
        slices=[
            InitiativeSlice(
                slice_id=f"{initiative_id}-slice-1",
                title="Registry",
                description="Persist initiatives locally.",
                validations=["python3 -m pytest tests/swarm/test_initiative_store.py -q"],
            )
        ],
        dependencies=["decision-plan-core"],
        validations=["python3 -m pytest tests/swarm/test_initiative_store.py -q"],
        feature_flag_name="initiative_registry",
        milestones=[
            InitiativeMilestone(
                milestone_id=f"{initiative_id}-milestone-1",
                title="Persistence landed",
                slice_ids=[f"{initiative_id}-slice-1"],
                checkpoint_ids=[f"{initiative_id}-checkpoint-1"],
            )
        ],
        checkpoints=[
            InitiativeCheckpoint(
                checkpoint_id=f"{initiative_id}-checkpoint-1",
                title="Registry validated",
                dependencies=[f"{initiative_id}-slice-1"],
                validations=["python3 -m pytest tests/swarm/test_initiative_store.py -q"],
            )
        ],
        updated_at=updated_at,
        created_at=updated_at,
    )


def test_initiative_store_round_trips_json_payload(tmp_path) -> None:
    store = InitiativeStore(state_dir=tmp_path)
    record = _record("initiative-registry", updated_at="2026-04-07T12:00:00+00:00")

    saved_path = store.save(record)
    loaded = store.get("initiative-registry")

    assert saved_path.exists()
    assert loaded is not None
    assert loaded.initiative_id == "initiative-registry"
    assert loaded.feature_flag_name == "initiative_registry"
    assert loaded.slices[0].title == "Registry"
    assert loaded.checkpoints[0].dependencies == ["initiative-registry-slice-1"]


def test_initiative_store_lists_newest_first(tmp_path) -> None:
    store = InitiativeStore(state_dir=tmp_path)
    older = _record("older", updated_at="2026-04-07T10:00:00+00:00")
    newer = _record("newer", updated_at="2026-04-07T11:00:00+00:00")

    store.save(older)
    store.save(newer)

    items = store.list()

    assert [item.initiative_id for item in items] == ["newer", "older"]
