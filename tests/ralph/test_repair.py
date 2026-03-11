"""Tests for ralph repair task generation."""

from __future__ import annotations

import pytest

from aragora.ralph.classifier import BlockerKind
from aragora.ralph.repair import RepairTask, generate_repair_task


class TestGenerateRepairTask:
    def test_reviewer_missing_diff(self) -> None:
        task = generate_repair_task(BlockerKind.REVIEWER_MISSING_DIFF)
        assert task is not None
        assert "diff" in task.title.lower() or "reviewer" in task.title.lower()
        assert "aragora/swarm/campaign.py" in task.allowed_paths
        assert task.blocker_kind == "reviewer_missing_diff_context"

    def test_scope_false_positive(self) -> None:
        task = generate_repair_task(BlockerKind.SCOPE_FALSE_POSITIVE)
        assert task is not None
        assert "scope" in task.title.lower()
        assert "aragora/swarm/worker_launcher.py" in task.allowed_paths

    def test_worker_clean_exit(self) -> None:
        task = generate_repair_task(BlockerKind.WORKER_CLEAN_EXIT_NO_EFFECT)
        assert task is not None
        assert "worker" in task.title.lower() or "deliverable" in task.title.lower()

    def test_manifest_collision(self) -> None:
        task = generate_repair_task(BlockerKind.MANIFEST_IDENTIFIER_COLLISION)
        assert task is not None
        assert "collision" in task.title.lower() or "identifier" in task.title.lower()

    def test_runtime_timeout(self) -> None:
        task = generate_repair_task(BlockerKind.RUNTIME_TIMEOUT_CONFIG)
        assert task is not None
        assert "time" in task.title.lower() or "timeout" in task.title.lower()

    def test_receipt_gap(self) -> None:
        task = generate_repair_task(BlockerKind.RECEIPT_EMISSION_GAP)
        assert task is not None
        assert "receipt" in task.title.lower()

    def test_escalation_kinds_return_none(self) -> None:
        assert generate_repair_task(BlockerKind.BUDGET_EXHAUSTION) is None
        assert generate_repair_task(BlockerKind.INFRA_FAILURE) is None
        assert generate_repair_task(BlockerKind.UNKNOWN) is None

    def test_affected_project_ids(self) -> None:
        task = generate_repair_task(
            BlockerKind.REVIEWER_MISSING_DIFF,
            affected_project_ids=["p1", "p2"],
        )
        assert task is not None
        assert task.affected_project_ids == ["p1", "p2"]

    def test_to_dict_round_trip(self) -> None:
        task = generate_repair_task(BlockerKind.REVIEWER_MISSING_DIFF)
        assert task is not None
        d = task.to_dict()
        assert isinstance(d, dict)
        assert d["blocker_kind"] == "reviewer_missing_diff_context"
        assert isinstance(d["allowed_paths"], list)
        assert isinstance(d["required_tests"], list)

    def test_all_deterministic_kinds_have_templates(self) -> None:
        for kind in BlockerKind:
            if kind.is_deterministic:
                task = generate_repair_task(kind)
                assert task is not None, f"Missing template for {kind.value}"
