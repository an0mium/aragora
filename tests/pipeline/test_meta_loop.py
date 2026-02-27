"""Tests for Meta-Loop Trigger."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from aragora.pipeline.meta_loop import (
    MetaLoopConfig,
    MetaLoopResult,
    MetaLoopTarget,
    MetaLoopTrigger,
)


@pytest.fixture
def trigger():
    return MetaLoopTrigger()


class TestMetaLoopConfig:
    def test_defaults(self):
        cfg = MetaLoopConfig()
        assert cfg.cooldown_cycles == 10
        assert cfg.require_human_approval is True

    def test_eligibility(self):
        cfg = MetaLoopConfig(cycle_count=0)
        assert cfg.is_eligible


class TestMetaLoopTarget:
    def test_basic(self):
        target = MetaLoopTarget(
            module="aragora/interrogation", description="Improve question quality", priority=0.8
        )
        assert target.priority == 0.8
        assert target.risk == "low"


class TestShouldTrigger:
    def test_before_cooldown(self, trigger):
        assert not trigger.should_trigger()

    def test_after_cooldown(self):
        trigger = MetaLoopTrigger(config=MetaLoopConfig(cooldown_cycles=2))
        trigger.increment_cycle()
        trigger.increment_cycle()
        assert trigger.should_trigger()

    def test_high_quality_skips(self):
        trigger = MetaLoopTrigger(
            config=MetaLoopConfig(cooldown_cycles=0), quality_scorer=lambda: 0.95
        )
        assert not trigger.should_trigger()


class TestIdentifyTargets:
    def test_default_targets(self, trigger):
        targets = trigger.identify_targets()
        assert len(targets) >= 1
        assert targets[0].module == "aragora/interrogation"

    def test_from_outcomes(self, trigger):
        outcomes = [
            {"execution_succeeded": False, "failed_module": "aragora/debate"},
            {"execution_succeeded": False, "failed_module": "aragora/debate"},
            {"execution_succeeded": True},
        ]
        targets = trigger.identify_targets(pipeline_outcomes=outcomes)
        assert any(t.module == "aragora/debate" for t in targets)


class TestExecute:
    def test_approval_mode_skips_all(self, trigger):
        targets = [MetaLoopTarget(module="mod", description="test", priority=0.5)]
        result = trigger.execute(targets)
        assert not result.approved
        assert len(result.targets_skipped) == 1
        assert len(result.targets_executed) == 0

    def test_autonomous_with_executor(self):
        trigger = MetaLoopTrigger(config=MetaLoopConfig(require_human_approval=False))
        targets = [
            MetaLoopTarget(module="mod", description="test", priority=0.8, estimated_files=2)
        ]
        result = trigger.execute(targets, executor=lambda t: True)
        assert result.approved
        assert len(result.targets_executed) == 1
        assert result.total_files_changed == 2

    def test_high_risk_skipped(self):
        trigger = MetaLoopTrigger(config=MetaLoopConfig(require_human_approval=False))
        targets = [MetaLoopTarget(module="mod", description="risky", priority=0.9, risk="high")]
        result = trigger.execute(targets, executor=lambda t: True)
        assert len(result.targets_skipped) == 1
        assert len(result.targets_executed) == 0

    def test_budget_limit(self):
        trigger = MetaLoopTrigger(
            config=MetaLoopConfig(require_human_approval=False, max_files_changed=3)
        )
        targets = [
            MetaLoopTarget(module="a", description="small", priority=0.9, estimated_files=2),
            MetaLoopTarget(module="b", description="big", priority=0.8, estimated_files=5),
        ]
        result = trigger.execute(targets, executor=lambda t: True)
        assert len(result.targets_executed) == 1
        assert len(result.targets_skipped) == 1

    def test_quality_tracking(self):
        quality = [0.6]

        def scorer():
            return quality[0]

        trigger = MetaLoopTrigger(
            config=MetaLoopConfig(require_human_approval=False), quality_scorer=scorer
        )
        targets = [MetaLoopTarget(module="mod", description="improve", priority=0.8)]

        def executor(t):
            quality[0] = 0.8
            return True

        result = trigger.execute(targets, executor=executor)
        assert result.quality_before == 0.6
        assert result.quality_after == 0.8
        assert result.improved
        assert result.quality_delta == pytest.approx(0.2)

    def test_km_integration(self):
        km = MagicMock()
        trigger = MetaLoopTrigger(
            config=MetaLoopConfig(require_human_approval=False), knowledge_mound=km
        )
        trigger.execute(
            [MetaLoopTarget(module="mod", description="test", priority=0.5)],
            executor=lambda t: True,
        )
        km.ingest.assert_called_once()
        call_data = km.ingest.call_args[0][0]
        assert call_data["type"] == "meta_loop_result"


class TestMetaLoopResult:
    def test_quality_delta(self):
        r = MetaLoopResult(quality_before=0.5, quality_after=0.8)
        assert r.quality_delta == pytest.approx(0.3)
        assert r.improved

    def test_no_improvement(self):
        r = MetaLoopResult(quality_before=0.8, quality_after=0.7)
        assert not r.improved


class TestHistory:
    def test_history(self):
        trigger = MetaLoopTrigger(config=MetaLoopConfig(require_human_approval=False))
        for _ in range(3):
            trigger.execute(
                [MetaLoopTarget(module="mod", description="t", priority=0.5)],
                executor=lambda t: True,
            )
        assert len(trigger.get_history()) == 3
        assert len(trigger.get_history(limit=2)) == 2
