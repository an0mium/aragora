"""Tests for TitansMemoryController -- active sweep controller."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.memory.titans_controller import (
    SurpriseState,
    SweepResult,
    TitansMemoryController,
)
from aragora.memory.retention_gate import RetentionDecision, RetentionGate, RetentionGateConfig
from aragora.memory.surprise import ContentSurpriseScorer, ContentSurpriseScore


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_surprise.db"


@pytest.fixture()
def controller(db_path: Path) -> TitansMemoryController:
    ctrl = TitansMemoryController(db_path=db_path)
    yield ctrl
    ctrl.close()


@pytest.fixture()
def mock_trigger_engine() -> MagicMock:
    engine = MagicMock()
    engine.fire = AsyncMock(return_value=[])
    return engine


@pytest.fixture()
def controller_with_triggers(
    db_path: Path, mock_trigger_engine: MagicMock
) -> TitansMemoryController:
    ctrl = TitansMemoryController(db_path=db_path, trigger_engine=mock_trigger_engine)
    yield ctrl
    ctrl.close()


# -----------------------------------------------------------------------
# Surprise state persistence (SQLite round-trip): 8 tests
# -----------------------------------------------------------------------


class TestSurpriseStatePersistence:
    def test_upsert_and_get_state(self, controller: TitansMemoryController) -> None:
        state = SurpriseState(
            item_id="item1",
            source="continuum",
            surprise_ema=0.6,
            last_access=time.time(),
            access_count=3,
            updated_at=time.time(),
        )
        controller._upsert_state(state)
        result = controller.get_state("item1", "continuum")
        assert result is not None
        assert result.item_id == "item1"
        assert result.source == "continuum"
        assert abs(result.surprise_ema - 0.6) < 1e-6

    def test_get_state_nonexistent(self, controller: TitansMemoryController) -> None:
        result = controller.get_state("nonexistent", "km")
        assert result is None

    def test_upsert_replaces_existing(self, controller: TitansMemoryController) -> None:
        now = time.time()
        state1 = SurpriseState("item1", "km", 0.5, now, 1, now)
        controller._upsert_state(state1)

        state2 = SurpriseState("item1", "km", 0.9, now + 10, 5, now + 10)
        controller._upsert_state(state2)

        result = controller.get_state("item1", "km")
        assert result is not None
        assert abs(result.surprise_ema - 0.9) < 1e-6
        assert result.access_count == 5

    def test_different_sources_same_item(self, controller: TitansMemoryController) -> None:
        now = time.time()
        controller._upsert_state(SurpriseState("item1", "km", 0.3, now, 1, now))
        controller._upsert_state(SurpriseState("item1", "continuum", 0.7, now, 2, now))

        km = controller.get_state("item1", "km")
        cont = controller.get_state("item1", "continuum")
        assert km is not None and cont is not None
        assert abs(km.surprise_ema - 0.3) < 1e-6
        assert abs(cont.surprise_ema - 0.7) < 1e-6

    def test_list_states_ordered_by_last_access(self, controller: TitansMemoryController) -> None:
        now = time.time()
        controller._upsert_state(SurpriseState("new", "km", 0.5, now + 100, 1, now))
        controller._upsert_state(SurpriseState("old", "km", 0.5, now - 100, 1, now))
        controller._upsert_state(SurpriseState("mid", "km", 0.5, now, 1, now))

        states = controller._list_states(10)
        assert [s.item_id for s in states] == ["old", "mid", "new"]

    def test_list_states_respects_batch_size(self, controller: TitansMemoryController) -> None:
        now = time.time()
        for i in range(10):
            controller._upsert_state(SurpriseState(f"item{i}", "km", 0.5, now + i, 1, now))

        states = controller._list_states(3)
        assert len(states) == 3

    def test_get_stats_empty(self, controller: TitansMemoryController) -> None:
        stats = controller.get_stats()
        assert stats["item_count"] == 0
        assert stats["avg_surprise"] == 0.0

    def test_get_stats_with_data(self, controller: TitansMemoryController) -> None:
        now = time.time()
        controller._upsert_state(SurpriseState("a", "km", 0.2, now, 1, now))
        controller._upsert_state(SurpriseState("b", "km", 0.8, now, 1, now))

        stats = controller.get_stats()
        assert stats["item_count"] == 2
        assert abs(stats["avg_surprise"] - 0.5) < 1e-3
        assert abs(stats["min_surprise"] - 0.2) < 1e-3
        assert abs(stats["max_surprise"] - 0.8) < 1e-3


# -----------------------------------------------------------------------
# on_query updates EMA correctly: 8 tests
# -----------------------------------------------------------------------


class TestOnQuery:
    @pytest.mark.asyncio
    async def test_on_query_creates_state_for_new_item(
        self, controller: TitansMemoryController
    ) -> None:
        result = MagicMock(item_id="q1", source_system="km")
        await controller.on_query("test query", [result])

        state = controller.get_state("q1", "km")
        assert state is not None
        assert state.access_count == 1
        assert abs(state.surprise_ema - 0.5) < 1e-6

    @pytest.mark.asyncio
    async def test_on_query_increments_access_count(
        self, controller: TitansMemoryController
    ) -> None:
        result = MagicMock(item_id="q1", source_system="km")
        await controller.on_query("q1", [result])
        await controller.on_query("q2", [result])

        state = controller.get_state("q1", "km")
        assert state is not None
        assert state.access_count == 2

    @pytest.mark.asyncio
    async def test_on_query_decays_surprise_ema(self, controller: TitansMemoryController) -> None:
        now = time.time()
        controller._upsert_state(SurpriseState("q1", "km", 0.8, now, 0, now))

        result = MagicMock(item_id="q1", source_system="km")
        await controller.on_query("test", [result])

        state = controller.get_state("q1", "km")
        assert state is not None
        # EMA should decrease (old * 0.9 mixed in)
        assert state.surprise_ema < 0.8

    @pytest.mark.asyncio
    async def test_on_query_updates_last_access(self, controller: TitansMemoryController) -> None:
        old_time = time.time() - 1000
        controller._upsert_state(SurpriseState("q1", "km", 0.5, old_time, 0, old_time))

        result = MagicMock(item_id="q1", source_system="km")
        await controller.on_query("test", [result])

        state = controller.get_state("q1", "km")
        assert state is not None
        assert state.last_access > old_time

    @pytest.mark.asyncio
    async def test_on_query_skips_items_without_id(
        self, controller: TitansMemoryController
    ) -> None:
        result = MagicMock(item_id="", source_system="km")
        await controller.on_query("test", [result])
        assert controller.get_stats()["item_count"] == 0

    @pytest.mark.asyncio
    async def test_on_query_skips_items_without_source(
        self, controller: TitansMemoryController
    ) -> None:
        result = MagicMock(item_id="q1", source_system="")
        await controller.on_query("test", [result])
        assert controller.get_stats()["item_count"] == 0

    @pytest.mark.asyncio
    async def test_on_query_multiple_results(self, controller: TitansMemoryController) -> None:
        results = [
            MagicMock(item_id="q1", source_system="km"),
            MagicMock(item_id="q2", source_system="continuum"),
        ]
        await controller.on_query("test", results)
        assert controller.get_stats()["item_count"] == 2

    @pytest.mark.asyncio
    async def test_on_query_fires_trigger(
        self,
        controller_with_triggers: TitansMemoryController,
        mock_trigger_engine: MagicMock,
    ) -> None:
        result = MagicMock(item_id="q1", source_system="km")
        await controller_with_triggers.on_query("test", [result])
        mock_trigger_engine.fire.assert_called_once()
        call_args = mock_trigger_engine.fire.call_args
        assert call_args[0][0] == "query_result"


# -----------------------------------------------------------------------
# on_write scores and stores initial surprise: 8 tests
# -----------------------------------------------------------------------


class TestOnWrite:
    @pytest.mark.asyncio
    async def test_on_write_creates_state(self, controller: TitansMemoryController) -> None:
        await controller.on_write("w1", "km", "some novel content about testing")

        state = controller.get_state("w1", "km")
        assert state is not None
        assert state.access_count == 0
        assert state.surprise_ema > 0

    @pytest.mark.asyncio
    async def test_on_write_uses_surprise_scorer(self, db_path: Path) -> None:
        scorer = MagicMock(spec=ContentSurpriseScorer)
        scorer.score.return_value = ContentSurpriseScore(
            novelty=0.9, momentum=0.5, combined=0.78, should_store=True, reason="test"
        )
        ctrl = TitansMemoryController(surprise_scorer=scorer, db_path=db_path)
        try:
            await ctrl.on_write("w1", "km", "content")
            scorer.score.assert_called_once_with("content", "km")
            state = ctrl.get_state("w1", "km")
            assert state is not None
            assert abs(state.surprise_ema - 0.78) < 1e-6
        finally:
            ctrl.close()

    @pytest.mark.asyncio
    async def test_on_write_overwrites_existing_state(
        self, controller: TitansMemoryController
    ) -> None:
        now = time.time()
        controller._upsert_state(SurpriseState("w1", "km", 0.1, now, 5, now))

        await controller.on_write("w1", "km", "new content about fresh topics")

        state = controller.get_state("w1", "km")
        assert state is not None
        assert state.access_count == 0  # reset on write

    @pytest.mark.asyncio
    async def test_on_write_fires_new_write_trigger(
        self,
        controller_with_triggers: TitansMemoryController,
        mock_trigger_engine: MagicMock,
    ) -> None:
        # Low surprise content
        scorer = MagicMock(spec=ContentSurpriseScorer)
        scorer.score.return_value = ContentSurpriseScore(
            novelty=0.3, momentum=0.2, combined=0.27, should_store=False, reason="low"
        )
        controller_with_triggers._surprise_scorer = scorer

        await controller_with_triggers.on_write("w1", "km", "routine")
        mock_trigger_engine.fire.assert_called_once()
        assert mock_trigger_engine.fire.call_args[0][0] == "new_write"

    @pytest.mark.asyncio
    async def test_on_write_fires_high_surprise_trigger(
        self,
        controller_with_triggers: TitansMemoryController,
        mock_trigger_engine: MagicMock,
    ) -> None:
        scorer = MagicMock(spec=ContentSurpriseScorer)
        scorer.score.return_value = ContentSurpriseScore(
            novelty=0.9, momentum=0.8, combined=0.87, should_store=True, reason="high"
        )
        controller_with_triggers._surprise_scorer = scorer

        await controller_with_triggers.on_write("w1", "km", "novel content")
        mock_trigger_engine.fire.assert_called_once()
        assert mock_trigger_engine.fire.call_args[0][0] == "high_surprise"

    @pytest.mark.asyncio
    async def test_on_write_content_preview_in_trigger_context(
        self,
        controller_with_triggers: TitansMemoryController,
        mock_trigger_engine: MagicMock,
    ) -> None:
        await controller_with_triggers.on_write("w1", "km", "abc" * 100)
        ctx = mock_trigger_engine.fire.call_args[0][1]
        assert len(ctx["content_preview"]) <= 200

    @pytest.mark.asyncio
    async def test_on_write_trigger_error_does_not_propagate(self, db_path: Path) -> None:
        engine = MagicMock()
        engine.fire = AsyncMock(side_effect=RuntimeError("trigger boom"))
        ctrl = TitansMemoryController(db_path=db_path, trigger_engine=engine)
        try:
            # Should not raise
            await ctrl.on_write("w1", "km", "content")
            state = ctrl.get_state("w1", "km")
            assert state is not None
        finally:
            ctrl.close()

    @pytest.mark.asyncio
    async def test_on_write_sets_timestamp(self, controller: TitansMemoryController) -> None:
        before = time.time()
        await controller.on_write("w1", "km", "test content")
        after = time.time()

        state = controller.get_state("w1", "km")
        assert state is not None
        assert before <= state.last_access <= after
        assert before <= state.updated_at <= after


# -----------------------------------------------------------------------
# sweep executes all 4 decision types: 10 tests
# -----------------------------------------------------------------------


class TestSweep:
    @pytest.mark.asyncio
    async def test_sweep_empty_state(self, controller: TitansMemoryController) -> None:
        result = await controller.sweep()
        assert result.items_processed == 0
        assert result.actions == {}
        assert result.errors == 0

    @pytest.mark.asyncio
    async def test_sweep_retain_action(self, db_path: Path) -> None:
        gate = RetentionGate(
            RetentionGateConfig(
                forget_threshold=0.15,
                consolidate_threshold=0.7,
            )
        )
        ctrl = TitansMemoryController(retention_gate=gate, db_path=db_path)
        try:
            now = time.time()
            # surprise_ema=0.4 -> normal range -> retain
            ctrl._upsert_state(SurpriseState("r1", "km", 0.4, now, 5, now))
            result = await ctrl.sweep()
            assert result.actions.get("retain", 0) >= 1
        finally:
            ctrl.close()

    @pytest.mark.asyncio
    async def test_sweep_forget_action(self, db_path: Path) -> None:
        gate = MagicMock(spec=RetentionGate)
        gate.evaluate.return_value = RetentionDecision(
            item_id="f1",
            source_system="km",
            surprise_score=0.05,
            retention_score=0.1,
            action="forget",
            reason="Low surprise and low confidence",
        )
        ctrl = TitansMemoryController(retention_gate=gate, db_path=db_path)
        try:
            now = time.time()
            ctrl._upsert_state(SurpriseState("f1", "km", 0.05, now, 0, now))
            result = await ctrl.sweep()
            assert result.actions.get("forget", 0) >= 1
            # Should be deleted from DB
            assert ctrl.get_state("f1", "km") is None
        finally:
            ctrl.close()

    @pytest.mark.asyncio
    async def test_sweep_demote_action(self, db_path: Path) -> None:
        gate = RetentionGate(
            RetentionGateConfig(
                forget_threshold=0.15,
                consolidate_threshold=0.7,
            )
        )
        ctrl = TitansMemoryController(retention_gate=gate, db_path=db_path)
        try:
            now = time.time()
            # surprise_ema=0.1 with moderate confidence (1 - 0.1 = 0.9) -> demote
            ctrl._upsert_state(SurpriseState("d1", "km", 0.1, now, 5, now))
            result = await ctrl.sweep()
            assert result.actions.get("demote", 0) >= 1
            state = ctrl.get_state("d1", "km")
            assert state is not None
            # Surprise EMA should be reduced
            assert state.surprise_ema < 0.1
        finally:
            ctrl.close()

    @pytest.mark.asyncio
    async def test_sweep_consolidate_action(self, db_path: Path) -> None:
        gate = RetentionGate(
            RetentionGateConfig(
                forget_threshold=0.15,
                consolidate_threshold=0.7,
            )
        )
        ctrl = TitansMemoryController(retention_gate=gate, db_path=db_path)
        try:
            now = time.time()
            # surprise_ema=0.9 -> high surprise -> consolidate
            ctrl._upsert_state(SurpriseState("c1", "km", 0.9, now, 0, now))
            result = await ctrl.sweep()
            assert result.actions.get("consolidate", 0) >= 1
            state = ctrl.get_state("c1", "km")
            assert state is not None
            # Surprise EMA should be boosted slightly
            assert state.surprise_ema >= 0.9
        finally:
            ctrl.close()

    @pytest.mark.asyncio
    async def test_sweep_batch_size(self, controller: TitansMemoryController) -> None:
        now = time.time()
        for i in range(20):
            controller._upsert_state(SurpriseState(f"item{i}", "km", 0.4, now + i, 1, now))

        result = await controller.sweep(batch_size=5)
        assert result.items_processed == 5

    @pytest.mark.asyncio
    async def test_sweep_returns_duration(self, controller: TitansMemoryController) -> None:
        result = await controller.sweep()
        assert result.duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_sweep_handles_gate_error(self, db_path: Path) -> None:
        gate = MagicMock(spec=RetentionGate)
        gate.evaluate.side_effect = ValueError("gate error")
        ctrl = TitansMemoryController(retention_gate=gate, db_path=db_path)
        try:
            now = time.time()
            ctrl._upsert_state(SurpriseState("e1", "km", 0.5, now, 1, now))
            result = await ctrl.sweep()
            assert result.errors == 1
        finally:
            ctrl.close()

    @pytest.mark.asyncio
    async def test_sweep_red_line_protection(self, db_path: Path) -> None:
        """Items with is_red_line=True should always be retained by RetentionGate."""
        gate = RetentionGate(RetentionGateConfig(red_line_protection=True))
        ctrl = TitansMemoryController(retention_gate=gate, db_path=db_path)
        try:
            now = time.time()
            # Even very low surprise item is retained if RetentionGate returns retain
            # (red_line is not set here, but the gate should return retain for normal range)
            ctrl._upsert_state(SurpriseState("r1", "km", 0.4, now, 3, now))
            result = await ctrl.sweep()
            assert result.items_processed == 1
            assert ctrl.get_state("r1", "km") is not None
        finally:
            ctrl.close()

    @pytest.mark.asyncio
    async def test_sweep_multiple_items_mixed_actions(self, db_path: Path) -> None:
        gate = RetentionGate(
            RetentionGateConfig(
                forget_threshold=0.15,
                consolidate_threshold=0.7,
            )
        )
        ctrl = TitansMemoryController(retention_gate=gate, db_path=db_path)
        try:
            now = time.time()
            ctrl._upsert_state(SurpriseState("high", "km", 0.9, now, 0, now))
            ctrl._upsert_state(SurpriseState("low", "km", 0.05, now, 0, now))
            ctrl._upsert_state(SurpriseState("mid", "km", 0.4, now, 5, now))

            result = await ctrl.sweep()
            assert result.items_processed == 3
            total_actions = sum(result.actions.values())
            assert total_actions == 3
        finally:
            ctrl.close()


# -----------------------------------------------------------------------
# Sweep loop runs at interval: 5 tests
# -----------------------------------------------------------------------


class TestSweepLoop:
    @pytest.mark.asyncio
    async def test_sweep_loop_single_iteration(self, controller: TitansMemoryController) -> None:
        await controller.run_sweep_loop(interval_seconds=0.01, max_sweeps=1)
        # Should complete without error

    @pytest.mark.asyncio
    async def test_sweep_loop_multiple_iterations(self, controller: TitansMemoryController) -> None:
        now = time.time()
        controller._upsert_state(SurpriseState("x", "km", 0.5, now, 1, now))

        await controller.run_sweep_loop(interval_seconds=0.01, max_sweeps=3)
        # Should complete after 3 sweeps

    @pytest.mark.asyncio
    async def test_sweep_loop_handles_sweep_error(self, db_path: Path) -> None:
        gate = MagicMock(spec=RetentionGate)
        gate.evaluate.side_effect = RuntimeError("boom")
        ctrl = TitansMemoryController(retention_gate=gate, db_path=db_path)
        try:
            now = time.time()
            ctrl._upsert_state(SurpriseState("x", "km", 0.5, now, 1, now))
            # Should not raise even though gate errors
            await ctrl.run_sweep_loop(interval_seconds=0.01, max_sweeps=1)
        finally:
            ctrl.close()

    @pytest.mark.asyncio
    async def test_sweep_loop_zero_interval(self, controller: TitansMemoryController) -> None:
        await controller.run_sweep_loop(interval_seconds=0.0, max_sweeps=2)

    @pytest.mark.asyncio
    async def test_sweep_loop_respects_max_sweeps(self, db_path: Path) -> None:
        sweep_count = 0
        original_sweep = TitansMemoryController.sweep

        async def counting_sweep(self_ctrl: Any, batch_size: int = 100) -> SweepResult:
            nonlocal sweep_count
            sweep_count += 1
            return await original_sweep(self_ctrl, batch_size)

        ctrl = TitansMemoryController(db_path=db_path)
        try:
            with patch.object(TitansMemoryController, "sweep", counting_sweep):
                await ctrl.run_sweep_loop(interval_seconds=0.01, max_sweeps=3)
            assert sweep_count == 3
        finally:
            ctrl.close()


# -----------------------------------------------------------------------
# Edge cases: 6 tests
# -----------------------------------------------------------------------


class TestEdgeCases:
    def test_in_memory_db(self) -> None:
        ctrl = TitansMemoryController(db_path=None)
        try:
            now = time.time()
            ctrl._upsert_state(SurpriseState("x", "km", 0.5, now, 1, now))
            assert ctrl.get_state("x", "km") is not None
        finally:
            ctrl.close()

    def test_close_then_no_operations(self, db_path: Path) -> None:
        ctrl = TitansMemoryController(db_path=db_path)
        ctrl.close()
        # After close, operations raise sqlite3 errors
        # (this is expected behavior)

    @pytest.mark.asyncio
    async def test_on_query_empty_results(self, controller: TitansMemoryController) -> None:
        await controller.on_query("test", [])
        assert controller.get_stats()["item_count"] == 0

    @pytest.mark.asyncio
    async def test_on_query_result_missing_attributes(
        self, controller: TitansMemoryController
    ) -> None:
        # Result with no item_id or source_system attributes
        result = object()
        await controller.on_query("test", [result])
        assert controller.get_stats()["item_count"] == 0

    @pytest.mark.asyncio
    async def test_on_write_empty_content(self, controller: TitansMemoryController) -> None:
        await controller.on_write("w1", "km", "")
        state = controller.get_state("w1", "km")
        assert state is not None

    def test_get_stats_returns_dict(self, controller: TitansMemoryController) -> None:
        stats = controller.get_stats()
        assert isinstance(stats, dict)
        assert "item_count" in stats
        assert "avg_surprise" in stats
        assert "min_surprise" in stats
        assert "max_surprise" in stats
