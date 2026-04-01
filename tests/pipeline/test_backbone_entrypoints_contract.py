"""Contract tests for RunLedger entrypoint coverage across DecisionPlan surfaces."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.pipeline.backbone_entrypoints_inventory import (
    ENTRYPOINT_INVENTORY,
    INTERNAL_BACKBONE_HELPERS,
    discover_backbone_entrypoints,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GREEN_WIRING_MODES = frozenset({"canonical_queue", "manual_run", "manual_seed"})
_GREEN_SIGNALS = frozenset(
    {
        "ensure_decision_plan_backbone_run",
        "execute_decision_plan_with_backbone",
        "queue_plan_execution",
        "run_ledger_create",
    }
)


def _inventory_map() -> dict[str, tuple[str, ...]]:
    return {entry.identifier: entry.signals for entry in ENTRYPOINT_INVENTORY}


def test_backbone_entrypoint_inventory_has_unique_identifiers() -> None:
    identifiers = [entry.identifier for entry in ENTRYPOINT_INVENTORY]
    assert len(identifiers) == len(set(identifiers))


def test_backbone_entrypoint_inventory_matches_repo_scan() -> None:
    discovered = discover_backbone_entrypoints(_REPO_ROOT)
    missing_internal_helpers = sorted(set(INTERNAL_BACKBONE_HELPERS) - set(discovered))
    assert not missing_internal_helpers

    relevant = {
        identifier: signals
        for identifier, signals in discovered.items()
        if identifier not in INTERNAL_BACKBONE_HELPERS
    }
    expected = _inventory_map()

    missing = sorted(set(expected) - set(relevant))
    unexpected = sorted(set(relevant) - set(expected))

    assert not missing
    assert not unexpected

    mismatched = {
        identifier: {"expected": expected[identifier], "discovered": relevant[identifier]}
        for identifier in expected
        if relevant[identifier] != expected[identifier]
    }
    assert not mismatched


@pytest.mark.parametrize("entry", ENTRYPOINT_INVENTORY, ids=lambda entry: entry.identifier)
def test_backbone_entrypoint_labels_match_signals(entry) -> None:
    signals = set(entry.signals)

    if entry.coverage == "green":
        assert entry.wiring_mode in _GREEN_WIRING_MODES
    elif entry.coverage == "reuse_only":
        assert entry.wiring_mode == "execution_bridge_only"
    else:
        assert entry.wiring_mode in {"direct_execute", "legacy_create"}

    if entry.wiring_mode == "canonical_queue":
        assert signals & {"execute_decision_plan_with_backbone", "queue_plan_execution"}
    elif entry.wiring_mode == "manual_seed":
        assert signals & {"ensure_decision_plan_backbone_run", "run_ledger_create"}
    elif entry.wiring_mode == "manual_run":
        assert "run_ledger_create" in signals
    elif entry.wiring_mode == "execution_bridge_only":
        assert signals == {"bridge_schedule_execution"}
        assert signals.isdisjoint(_GREEN_SIGNALS)
    elif entry.wiring_mode == "legacy_create":
        assert signals & {"decision_plan_ctor", "decision_plan_factory"}
        assert signals & {"plan_store_create", "store_plan"}
        assert signals.isdisjoint(_GREEN_SIGNALS)
        assert "bridge_schedule_execution" not in signals
    else:
        assert "plan_executor_execute" in signals
        assert signals.isdisjoint(_GREEN_SIGNALS)
        assert "bridge_schedule_execution" not in signals
