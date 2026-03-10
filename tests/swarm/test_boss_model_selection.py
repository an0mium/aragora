"""Tests for boss/worker model selection plumbing (issue #907).

Verifies that:
1. BossLoopConfig carries boss_model/worker_model with correct defaults
2. BossLoopResult serializes model selection into JSON output
3. CLI args flow through to BossLoopConfig
4. worker_model propagates as default_target_agent through commander/supervisor
5. Supervisor start_run applies default_target_agent to work orders
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestBossLoopConfigModelDefaults:
    """BossLoopConfig has correct defaults and accepts overrides."""

    def test_default_models_are_codex(self) -> None:
        from aragora.swarm.boss_loop import BossLoopConfig

        config = BossLoopConfig()
        assert config.boss_model == "codex"
        assert config.worker_model == "codex"

    def test_custom_models(self) -> None:
        from aragora.swarm.boss_loop import BossLoopConfig

        config = BossLoopConfig(boss_model="claude", worker_model="claude")
        assert config.boss_model == "claude"
        assert config.worker_model == "claude"

    def test_mixed_models(self) -> None:
        from aragora.swarm.boss_loop import BossLoopConfig

        config = BossLoopConfig(boss_model="claude", worker_model="codex")
        assert config.boss_model == "claude"
        assert config.worker_model == "codex"


class TestBossLoopResultModelSerialization:
    """BossLoopResult includes model selection in JSON output."""

    def test_default_models_in_dict(self) -> None:
        from aragora.swarm.boss_loop import BossLoopResult

        result = BossLoopResult(
            run_id="test-123",
            iterations_completed=1,
            total_elapsed_seconds=10.0,
            stop_reason="max_iterations",
            issues_attempted=[],
            issues_completed=[],
            issues_failed=[],
            iteration_statuses=[],
            needs_human_reasons=[],
            next_actions=[],
        )
        d = result.to_dict()
        assert d["boss_model"] == "codex"
        assert d["worker_model"] == "codex"

    def test_custom_models_in_dict(self) -> None:
        from aragora.swarm.boss_loop import BossLoopResult

        result = BossLoopResult(
            run_id="test-456",
            iterations_completed=1,
            total_elapsed_seconds=10.0,
            stop_reason="max_iterations",
            issues_attempted=[],
            issues_completed=[],
            issues_failed=[],
            iteration_statuses=[],
            needs_human_reasons=[],
            next_actions=[],
            boss_model="claude",
            worker_model="gemini",
        )
        d = result.to_dict()
        assert d["boss_model"] == "claude"
        assert d["worker_model"] == "gemini"

    def test_model_fields_present_in_json(self) -> None:
        from aragora.swarm.boss_loop import BossLoopResult

        result = BossLoopResult(
            run_id="test-789",
            iterations_completed=0,
            total_elapsed_seconds=0.0,
            stop_reason="manual_stop",
            issues_attempted=[],
            issues_completed=[],
            issues_failed=[],
            iteration_statuses=[],
            needs_human_reasons=[],
            next_actions=[],
            boss_model="claude",
            worker_model="codex",
        )
        payload = json.dumps(result.to_dict())
        parsed = json.loads(payload)
        assert parsed["boss_model"] == "claude"
        assert parsed["worker_model"] == "codex"


class TestBossLoopPropagatesModelToResult:
    """BossLoop.run() copies model config into the final result."""

    @pytest.mark.asyncio
    async def test_models_propagated_to_result(self) -> None:
        from aragora.swarm.boss_loop import BossLoop, BossLoopConfig

        config = BossLoopConfig(
            max_iterations=1,
            boss_model="claude",
            worker_model="gemini",
        )
        loop = BossLoop(
            config=config,
            freshness_checker=lambda **kw: MagicMock(
                fresh=False,
                blocked_reason="test_block",
                to_dict=lambda: {"fresh": False, "blocked_reason": "test_block"},
            ),
        )
        result = await loop.run()
        assert result.boss_model == "claude"
        assert result.worker_model == "gemini"
        d = result.to_dict()
        assert d["boss_model"] == "claude"
        assert d["worker_model"] == "gemini"


class TestSupervisorDefaultTargetAgent:
    """Supervisor.start_run applies default_target_agent to work orders."""

    def test_default_target_agent_overrides_work_orders(self) -> None:
        from aragora.swarm.supervisor import SwarmSupervisor

        supervisor = SwarmSupervisor.__new__(SwarmSupervisor)

        # Create a minimal mock spec
        spec = MagicMock()
        spec.refined_goal = "Test goal"
        spec.raw_goal = "Test goal"
        spec.acceptance_criteria = []
        spec.constraints = []
        spec.file_scope_hints = []
        spec.work_orders = []
        spec.to_dict.return_value = {}

        # Mock the internal methods
        mock_work_order = MagicMock()
        mock_work_order.to_dict.return_value = {
            "work_order_id": "wo-1",
            "target_agent": "codex",
            "title": "test",
        }
        supervisor._build_supervised_work_orders = MagicMock(return_value=[mock_work_order])

        # Mock the store
        store = MagicMock()
        created_record: dict[str, Any] = {}

        def capture_create(**kwargs: Any) -> dict[str, Any]:
            created_record.update(kwargs)
            created_record["run_id"] = "run-test"
            created_record["created_at"] = "2026-01-01T00:00:00"
            created_record["updated_at"] = "2026-01-01T00:00:00"
            return created_record

        store.create_supervisor_run.side_effect = capture_create
        store.get_supervisor_run.return_value = None
        supervisor.store = store
        supervisor.refresh_run = MagicMock(return_value=MagicMock())

        # Call with default_target_agent="claude"
        supervisor.start_run(
            spec=spec,
            default_target_agent="claude",
            refresh_scaling=False,
        )

        # Verify the work orders passed to create_supervisor_run had target_agent overridden
        call_kwargs = store.create_supervisor_run.call_args
        work_orders = call_kwargs.kwargs.get("work_orders") or call_kwargs[1].get("work_orders", [])
        assert len(work_orders) == 1
        assert work_orders[0]["target_agent"] == "claude"

    def test_no_default_preserves_original(self) -> None:
        from aragora.swarm.supervisor import SwarmSupervisor

        supervisor = SwarmSupervisor.__new__(SwarmSupervisor)

        spec = MagicMock()
        spec.refined_goal = "Test goal"
        spec.raw_goal = "Test goal"
        spec.to_dict.return_value = {}

        mock_work_order = MagicMock()
        mock_work_order.to_dict.return_value = {
            "work_order_id": "wo-1",
            "target_agent": "codex",
            "title": "test",
        }
        supervisor._build_supervised_work_orders = MagicMock(return_value=[mock_work_order])

        store = MagicMock()

        def capture_create(**kwargs: Any) -> dict[str, Any]:
            record = dict(kwargs)
            record["run_id"] = "run-test"
            record["created_at"] = "2026-01-01T00:00:00"
            record["updated_at"] = "2026-01-01T00:00:00"
            return record

        store.create_supervisor_run.side_effect = capture_create
        supervisor.store = store
        supervisor.refresh_run = MagicMock(return_value=MagicMock())

        # Call without default_target_agent
        supervisor.start_run(
            spec=spec,
            refresh_scaling=False,
        )

        call_kwargs = store.create_supervisor_run.call_args
        work_orders = call_kwargs.kwargs.get("work_orders") or call_kwargs[1].get("work_orders", [])
        assert len(work_orders) == 1
        assert work_orders[0]["target_agent"] == "codex"  # unchanged


class TestDispatchPassesWorkerModel:
    """BossLoop._dispatch_issue passes worker_model through to commander."""

    @pytest.mark.asyncio
    async def test_non_default_worker_model_passed_to_commander(self) -> None:
        from aragora.swarm.boss_loop import (
            BossLoop,
            BossLoopConfig,
            GitHubIssue,
            RunnerFreshnessResult,
        )

        config = BossLoopConfig(worker_model="claude")
        loop = BossLoop(config=config)

        issue = GitHubIssue(
            number=100,
            title="Test issue",
            body="Test body",
            labels=[],
            url="https://github.com/test/test/issues/100",
            state="OPEN",
            created_at="2026-01-01T00:00:00",
        )
        freshness = RunnerFreshnessResult(
            fresh=True,
            runner_ids=["r1"],
            checked_at="2026-01-01T00:00:00",
        )

        captured_kwargs: dict[str, Any] = {}

        async def mock_run_supervised(spec_arg: Any, **kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            run = MagicMock()
            run.to_dict.return_value = {
                "status": "completed",
                "work_orders": [
                    {
                        "status": "completed",
                        "pr_url": "https://github.com/test/test/pull/1",
                        "work_order_id": "wo-1",
                        "worker_outcome": "completed",
                    }
                ],
            }
            return run

        with (
            patch("aragora.swarm.spec.SwarmSpec") as mock_spec_cls,
            patch("aragora.swarm.commander.SwarmCommander") as mock_commander_cls,
            patch("aragora.swarm.config.SwarmCommanderConfig"),
            patch("aragora.swarm.supervisor.SwarmApprovalPolicy"),
        ):
            mock_spec = MagicMock()
            mock_spec.is_dispatch_bounded.return_value = True
            mock_spec_cls.from_direct_goal.return_value = mock_spec

            mock_commander = MagicMock()
            mock_commander.run_supervised_from_spec = mock_run_supervised
            mock_commander_cls.return_value = mock_commander

            result = await loop._dispatch_issue(issue, freshness)

        assert result["status"] == "completed"
        assert captured_kwargs.get("default_target_agent") == "claude"

    @pytest.mark.asyncio
    async def test_default_codex_does_not_pass_override(self) -> None:
        from aragora.swarm.boss_loop import (
            BossLoop,
            BossLoopConfig,
            GitHubIssue,
            RunnerFreshnessResult,
        )

        config = BossLoopConfig(worker_model="codex")  # default
        loop = BossLoop(config=config)

        issue = GitHubIssue(
            number=101,
            title="Test issue",
            body="Test body",
            labels=[],
            url="https://github.com/test/test/issues/101",
            state="OPEN",
            created_at="2026-01-01T00:00:00",
        )
        freshness = RunnerFreshnessResult(
            fresh=True,
            runner_ids=["r1"],
            checked_at="2026-01-01T00:00:00",
        )

        captured_kwargs: dict[str, Any] = {}

        async def mock_run_supervised(spec_arg: Any, **kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            run = MagicMock()
            run.to_dict.return_value = {
                "status": "completed",
                "work_orders": [
                    {
                        "status": "completed",
                        "pr_url": "https://github.com/test/test/pull/2",
                        "work_order_id": "wo-1",
                        "worker_outcome": "completed",
                    }
                ],
            }
            return run

        with (
            patch("aragora.swarm.spec.SwarmSpec") as mock_spec_cls,
            patch("aragora.swarm.commander.SwarmCommander") as mock_commander_cls,
            patch("aragora.swarm.config.SwarmCommanderConfig"),
            patch("aragora.swarm.supervisor.SwarmApprovalPolicy"),
        ):
            mock_spec = MagicMock()
            mock_spec.is_dispatch_bounded.return_value = True
            mock_spec_cls.from_direct_goal.return_value = mock_spec

            mock_commander = MagicMock()
            mock_commander.run_supervised_from_spec = mock_run_supervised
            mock_commander_cls.return_value = mock_commander

            result = await loop._dispatch_issue(issue, freshness)

        assert result["status"] == "completed"
        # default codex should NOT override (sends None)
        assert captured_kwargs.get("default_target_agent") is None


class TestCLIArgsWiring:
    """CLI parser produces boss_model/worker_model args that flow to config."""

    def test_parser_accepts_model_args(self) -> None:
        from aragora.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["swarm", "boss-loop", "--boss-model", "claude", "--worker-model", "gemini"]
        )
        assert args.boss_model == "claude"
        assert args.worker_model == "gemini"

    def test_parser_defaults_to_none(self) -> None:
        from aragora.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["swarm", "boss-loop"])
        assert getattr(args, "boss_model", None) is None
        assert getattr(args, "worker_model", None) is None


class TestBossAgentWiring:
    """boss_model flows through to supervisor_agents and reviewer_agent on work orders."""

    def test_boss_agent_sets_planner_and_judge(self) -> None:
        from aragora.swarm.supervisor import SwarmSupervisor

        supervisor = SwarmSupervisor.__new__(SwarmSupervisor)

        spec = MagicMock()
        spec.refined_goal = "Test goal"
        spec.raw_goal = "Test goal"
        spec.to_dict.return_value = {}

        mock_work_order = MagicMock()
        mock_work_order.to_dict.return_value = {
            "work_order_id": "wo-1",
            "target_agent": "codex",
            "reviewer_agent": "claude",
            "title": "test",
        }
        supervisor._build_supervised_work_orders = MagicMock(return_value=[mock_work_order])

        store = MagicMock()
        created_record: dict[str, Any] = {}

        def capture_create(**kwargs: Any) -> dict[str, Any]:
            created_record.update(kwargs)
            created_record["run_id"] = "run-test"
            created_record["created_at"] = "2026-01-01T00:00:00"
            created_record["updated_at"] = "2026-01-01T00:00:00"
            return created_record

        store.create_supervisor_run.side_effect = capture_create
        supervisor.store = store
        supervisor.refresh_run = MagicMock(return_value=MagicMock())

        supervisor.start_run(
            spec=spec,
            boss_agent="claude",
            refresh_scaling=False,
        )

        call_kwargs = store.create_supervisor_run.call_args
        # supervisor_agents should reflect the boss_agent
        supervisor_agents = call_kwargs.kwargs.get("supervisor_agents") or call_kwargs[1].get(
            "supervisor_agents", {}
        )
        assert supervisor_agents["planner"] == "claude"
        assert supervisor_agents["judge"] == "codex"  # opposite of planner

        # reviewer_agent on work orders should be overridden to boss_agent
        work_orders = call_kwargs.kwargs.get("work_orders") or call_kwargs[1].get("work_orders", [])
        assert work_orders[0]["reviewer_agent"] == "claude"

    def test_no_boss_agent_preserves_defaults(self) -> None:
        from aragora.swarm.supervisor import SwarmSupervisor

        supervisor = SwarmSupervisor.__new__(SwarmSupervisor)

        spec = MagicMock()
        spec.refined_goal = "Test goal"
        spec.raw_goal = "Test goal"
        spec.to_dict.return_value = {}

        mock_work_order = MagicMock()
        mock_work_order.to_dict.return_value = {
            "work_order_id": "wo-1",
            "target_agent": "codex",
            "reviewer_agent": "claude",
            "title": "test",
        }
        supervisor._build_supervised_work_orders = MagicMock(return_value=[mock_work_order])

        store = MagicMock()

        def capture_create(**kwargs: Any) -> dict[str, Any]:
            record = dict(kwargs)
            record["run_id"] = "run-test"
            record["created_at"] = "2026-01-01T00:00:00"
            record["updated_at"] = "2026-01-01T00:00:00"
            return record

        store.create_supervisor_run.side_effect = capture_create
        supervisor.store = store
        supervisor.refresh_run = MagicMock(return_value=MagicMock())

        supervisor.start_run(
            spec=spec,
            refresh_scaling=False,
        )

        call_kwargs = store.create_supervisor_run.call_args
        supervisor_agents = call_kwargs.kwargs.get("supervisor_agents") or call_kwargs[1].get(
            "supervisor_agents", {}
        )
        assert supervisor_agents["planner"] == "codex"  # default
        assert supervisor_agents["judge"] == "claude"  # default

        work_orders = call_kwargs.kwargs.get("work_orders") or call_kwargs[1].get("work_orders", [])
        assert work_orders[0]["reviewer_agent"] == "claude"  # unchanged

    @pytest.mark.asyncio
    async def test_dispatch_passes_boss_agent_to_commander(self) -> None:
        from aragora.swarm.boss_loop import (
            BossLoop,
            BossLoopConfig,
            GitHubIssue,
            RunnerFreshnessResult,
        )

        config = BossLoopConfig(boss_model="claude", worker_model="codex")
        loop = BossLoop(config=config)

        issue = GitHubIssue(
            number=200,
            title="Test boss model",
            body="Body",
            labels=[],
            url="https://github.com/test/test/issues/200",
            state="OPEN",
            created_at="2026-01-01T00:00:00",
        )
        freshness = RunnerFreshnessResult(
            fresh=True,
            runner_ids=["r1"],
            checked_at="2026-01-01T00:00:00",
        )

        captured_kwargs: dict[str, Any] = {}

        async def mock_run_supervised(spec_arg: Any, **kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            run = MagicMock()
            run.to_dict.return_value = {
                "status": "completed",
                "work_orders": [
                    {
                        "status": "completed",
                        "pr_url": "https://github.com/test/test/pull/3",
                        "work_order_id": "wo-1",
                        "worker_outcome": "completed",
                    }
                ],
            }
            return run

        with (
            patch("aragora.swarm.spec.SwarmSpec") as mock_spec_cls,
            patch("aragora.swarm.commander.SwarmCommander") as mock_commander_cls,
            patch("aragora.swarm.config.SwarmCommanderConfig"),
            patch("aragora.swarm.supervisor.SwarmApprovalPolicy"),
        ):
            mock_spec = MagicMock()
            mock_spec.is_dispatch_bounded.return_value = True
            mock_spec_cls.from_direct_goal.return_value = mock_spec

            mock_commander = MagicMock()
            mock_commander.run_supervised_from_spec = mock_run_supervised
            mock_commander_cls.return_value = mock_commander

            result = await loop._dispatch_issue(issue, freshness)

        assert result["status"] == "completed"
        assert captured_kwargs.get("boss_agent") == "claude"
        # worker_model is codex (default), so default_target_agent should be None
        assert captured_kwargs.get("default_target_agent") is None
