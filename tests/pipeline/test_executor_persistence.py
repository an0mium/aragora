"""Tests for executor persistence — verifying delegation to PlanStore backend.

Ensures that store_plan/get_plan/list_plans delegate to the persistent
PlanStore when available, and fall back to in-memory dicts when PlanStore
initialization fails.
"""

from __future__ import annotations

import importlib
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.pipeline.decision_plan.core import (
    ApprovalMode,
    BudgetAllocation,
    DecisionPlan,
    PlanStatus,
)


def _make_plan(
    plan_id: str = "dp-test-001", status: PlanStatus = PlanStatus.CREATED
) -> DecisionPlan:
    """Create a minimal DecisionPlan for testing."""
    return DecisionPlan(
        id=plan_id,
        debate_id="debate-abc",
        task="Test task",
        status=status,
        approval_mode=ApprovalMode.RISK_BASED,
        budget=BudgetAllocation(limit_usd=10.0),
        created_at=datetime.now(),
    )


@pytest.fixture(autouse=True)
def _reset_executor_module():
    """Reset executor module-level state between tests."""
    import aragora.pipeline.executor as executor_mod

    # Save original state
    orig_backing = executor_mod._backing_store
    orig_failed = executor_mod._backing_store_init_failed
    orig_fallback = executor_mod._plan_store_fallback.copy()
    orig_outcomes = executor_mod._plan_outcomes_fallback.copy()

    yield

    # Restore original state
    executor_mod._backing_store = orig_backing
    executor_mod._backing_store_init_failed = orig_failed
    executor_mod._plan_store_fallback.clear()
    executor_mod._plan_store_fallback.update(orig_fallback)
    executor_mod._plan_outcomes_fallback.clear()
    executor_mod._plan_outcomes_fallback.update(orig_outcomes)


class TestPersistentBackendDelegation:
    """Tests that executor functions delegate to PlanStore."""

    def test_store_plan_delegates_to_plan_store(self, tmp_path: Path):
        """store_plan should delegate to the persistent PlanStore backend."""
        from aragora.pipeline.plan_store import PlanStore
        import aragora.pipeline.executor as executor_mod

        db_path = str(tmp_path / "test_plans.db")
        store = PlanStore(db_path=db_path)
        executor_mod._backing_store = store
        executor_mod._backing_store_init_failed = False

        plan = _make_plan("dp-persist-001")
        executor_mod.store_plan(plan)

        # Verify it was written to the persistent store
        retrieved = store.get("dp-persist-001")
        assert retrieved is not None
        assert retrieved.id == "dp-persist-001"
        assert retrieved.task == "Test task"

    def test_get_plan_delegates_to_plan_store(self, tmp_path: Path):
        """get_plan should retrieve from the persistent PlanStore backend."""
        from aragora.pipeline.plan_store import PlanStore
        import aragora.pipeline.executor as executor_mod

        db_path = str(tmp_path / "test_plans.db")
        store = PlanStore(db_path=db_path)
        executor_mod._backing_store = store
        executor_mod._backing_store_init_failed = False

        plan = _make_plan("dp-persist-002")
        store.create(plan)

        retrieved = executor_mod.get_plan("dp-persist-002")
        assert retrieved is not None
        assert retrieved.id == "dp-persist-002"

    def test_list_plans_delegates_to_plan_store(self, tmp_path: Path):
        """list_plans should delegate to PlanStore.list()."""
        from aragora.pipeline.plan_store import PlanStore
        import aragora.pipeline.executor as executor_mod

        db_path = str(tmp_path / "test_plans.db")
        store = PlanStore(db_path=db_path)
        executor_mod._backing_store = store
        executor_mod._backing_store_init_failed = False

        plan1 = _make_plan("dp-list-001", status=PlanStatus.CREATED)
        plan2 = _make_plan("dp-list-002", status=PlanStatus.APPROVED)
        store.create(plan1)
        store.create(plan2)

        # List all
        all_plans = executor_mod.list_plans()
        assert len(all_plans) == 2

        # Filter by status
        approved_plans = executor_mod.list_plans(status=PlanStatus.APPROVED)
        assert len(approved_plans) == 1
        assert approved_plans[0].id == "dp-list-002"

    def test_get_plan_returns_none_for_missing(self, tmp_path: Path):
        """get_plan returns None when plan not found in persistent store."""
        from aragora.pipeline.plan_store import PlanStore
        import aragora.pipeline.executor as executor_mod

        db_path = str(tmp_path / "test_plans.db")
        store = PlanStore(db_path=db_path)
        executor_mod._backing_store = store
        executor_mod._backing_store_init_failed = False

        assert executor_mod.get_plan("nonexistent") is None


class TestInMemoryFallback:
    """Tests that in-memory fallback works when PlanStore is unavailable."""

    def test_fallback_when_init_fails(self):
        """When PlanStore init fails, executor falls back to in-memory."""
        import aragora.pipeline.executor as executor_mod

        executor_mod._backing_store = None
        executor_mod._backing_store_init_failed = True
        executor_mod._plan_store_fallback.clear()

        plan = _make_plan("dp-fallback-001")
        executor_mod.store_plan(plan)

        retrieved = executor_mod.get_plan("dp-fallback-001")
        assert retrieved is not None
        assert retrieved.id == "dp-fallback-001"

    def test_fallback_list_plans(self):
        """list_plans works with in-memory fallback."""
        import aragora.pipeline.executor as executor_mod

        executor_mod._backing_store = None
        executor_mod._backing_store_init_failed = True
        executor_mod._plan_store_fallback.clear()

        plan1 = _make_plan("dp-fb-001", status=PlanStatus.CREATED)
        plan2 = _make_plan("dp-fb-002", status=PlanStatus.APPROVED)
        executor_mod.store_plan(plan1)
        executor_mod.store_plan(plan2)

        all_plans = executor_mod.list_plans()
        assert len(all_plans) == 2

        approved = executor_mod.list_plans(status=PlanStatus.APPROVED)
        assert len(approved) == 1
        assert approved[0].id == "dp-fb-002"

    def test_fallback_eviction(self):
        """In-memory fallback evicts completed plans when at capacity."""
        import aragora.pipeline.executor as executor_mod

        executor_mod._backing_store = None
        executor_mod._backing_store_init_failed = True
        executor_mod._plan_store_fallback.clear()

        # Fill to MAX_PLANS with completed plans
        for i in range(executor_mod._MAX_PLANS):
            p = _make_plan(f"dp-evict-{i:04d}", status=PlanStatus.COMPLETED)
            executor_mod._plan_store_fallback[p.id] = p

        assert len(executor_mod._plan_store_fallback) == executor_mod._MAX_PLANS

        # Store one more — should evict one completed plan
        new_plan = _make_plan("dp-evict-new", status=PlanStatus.CREATED)
        executor_mod.store_plan(new_plan)

        assert len(executor_mod._plan_store_fallback) <= executor_mod._MAX_PLANS
        assert executor_mod.get_plan("dp-evict-new") is not None

    def test_fallback_on_persistent_store_error(self, tmp_path: Path):
        """If persistent store raises on store_plan, falls back to in-memory."""
        import aragora.pipeline.executor as executor_mod

        mock_store = MagicMock()
        mock_store.create.side_effect = RuntimeError("DB locked")
        mock_store.get.side_effect = RuntimeError("DB locked")
        mock_store.list.side_effect = RuntimeError("DB locked")

        executor_mod._backing_store = mock_store
        executor_mod._backing_store_init_failed = False
        executor_mod._plan_store_fallback.clear()

        plan = _make_plan("dp-err-001")
        executor_mod.store_plan(plan)

        # Should fall through to in-memory
        assert "dp-err-001" in executor_mod._plan_store_fallback

    def test_get_outcome_still_works(self):
        """get_outcome uses in-memory store (outcomes not in PlanStore)."""
        import aragora.pipeline.executor as executor_mod

        executor_mod._plan_outcomes_fallback.clear()
        assert executor_mod.get_outcome("nonexistent") is None


class TestLazySingletonInit:
    """Tests for the lazy singleton initialization of the backing store."""

    def test_get_backing_store_initializes_once(self, tmp_path: Path):
        """_get_backing_store creates a PlanStore on first call."""
        import aragora.pipeline.executor as executor_mod

        executor_mod._backing_store = None
        executor_mod._backing_store_init_failed = False

        with patch.dict(os.environ, {"ARAGORA_FORCE_PERSISTENT_PLAN_STORE": "1"}):
            with patch("aragora.pipeline.plan_store.PlanStore") as MockStore:
                mock_instance = MagicMock()
                MockStore.return_value = mock_instance

                store1 = executor_mod._get_backing_store()
                store2 = executor_mod._get_backing_store()

                # Should create only once
                MockStore.assert_called_once()
                assert store1 is mock_instance
                assert store2 is mock_instance

    def test_get_backing_store_remembers_failure(self):
        """_get_backing_store does not retry after init failure."""
        import aragora.pipeline.executor as executor_mod

        executor_mod._backing_store = None
        executor_mod._backing_store_init_failed = False

        with patch.dict(os.environ, {"ARAGORA_FORCE_PERSISTENT_PLAN_STORE": "1"}):
            with patch(
                "aragora.pipeline.executor._try_init_backing_store",
                side_effect=RuntimeError("Cannot create DB"),
            ):
                store1 = executor_mod._get_backing_store()
                assert store1 is None
                assert executor_mod._backing_store_init_failed is True

                # Second call should not retry
                store2 = executor_mod._get_backing_store()
                assert store2 is None


class TestStoreOutcomeFallback:
    """Tests for outcome storage which always uses in-memory."""

    def test_store_and_get_outcome(self):
        """store_outcome and get_outcome work via in-memory dict."""
        import aragora.pipeline.executor as executor_mod
        from aragora.pipeline.decision_plan import PlanOutcome

        executor_mod._plan_outcomes_fallback.clear()

        outcome = PlanOutcome(
            plan_id="dp-outcome-001",
            debate_id="debate-001",
            task="Test task",
            success=True,
        )
        executor_mod.store_outcome(outcome)

        retrieved = executor_mod.get_outcome("dp-outcome-001")
        assert retrieved is not None
        assert retrieved.success is True
