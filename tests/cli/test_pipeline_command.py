"""Tests for ``aragora pipeline`` command behavior and parser wiring."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aragora.cli.commands.pipeline import (
    _cmd_pipeline_self_improve,
    _extract_pipeline_objectives,
)


class _FakeTaskDecomposer:
    def __init__(self, config=None) -> None:
        self.config = config

    def analyze(self, _goal: str):
        return SimpleNamespace(
            complexity_score=6,
            complexity_level="medium",
            subtasks=[
                SimpleNamespace(
                    title="Improve test reliability",
                    estimated_complexity="medium",
                    file_scope=["tests/test_example.py"],
                )
            ],
        )


class _FakeMetaPlanner:
    def __init__(self, _config) -> None:
        pass

    async def prioritize_work(self, objective: str):
        track = SimpleNamespace(value="core")
        return [
            SimpleNamespace(
                priority=1,
                track=track,
                description=f"Improve: {objective}",
                estimated_impact="high",
                rationale="Top priority",
            )
        ]


class _FakeIdeaToExecutionPipeline:
    def from_ideas(self, _ideas):
        goals = [
            SimpleNamespace(
                title="Bridge pipeline to self-improve",
                description="Connect idea output to autonomous self-improvement execution",
                priority="critical",
                confidence=0.81,
            ),
            SimpleNamespace(
                title="Polish docs",
                description="Improve docs around CLI usage",
                priority="low",
                confidence=0.95,
            ),
        ]
        return SimpleNamespace(
            pipeline_id="pipe-123",
            goal_graph=SimpleNamespace(goals=goals),
            stage_results=[
                SimpleNamespace(stage_name="ideas", status="completed", duration=0.1),
                SimpleNamespace(stage_name="goals", status="completed", duration=0.2),
            ],
            provenance=["p1", "p2"],
            duration=0.5,
        )


def _fake_module_payload() -> dict[str, object]:
    return {
        "aragora.nomic.task_decomposer": SimpleNamespace(
            DecomposerConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            TaskDecomposer=_FakeTaskDecomposer,
        ),
        "aragora.nomic.meta_planner": SimpleNamespace(
            MetaPlannerConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            MetaPlanner=_FakeMetaPlanner,
        ),
        "aragora.pipeline.idea_to_execution": SimpleNamespace(
            PipelineConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            IdeaToExecutionPipeline=_FakeIdeaToExecutionPipeline,
        ),
    }


class TestPipelineParser:
    def test_pipeline_self_improve_parser_accepts_handoff_flags(self):
        from aragora.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "pipeline",
                "self-improve",
                "Improve reliability",
                "--execute",
                "--max-goals",
                "3",
                "--quick-mode",
                "--max-parallel",
                "2",
            ]
        )

        assert args.command == "pipeline"
        assert args.pipeline_action == "self-improve"
        assert args.execute is True
        assert args.max_goals == 3
        assert args.quick_mode is True
        assert args.max_parallel == 2


class TestPipelineObjectiveExtraction:
    def test_extract_pipeline_objectives_ranks_by_priority_then_confidence(self):
        goals = [
            SimpleNamespace(
                title="Low",
                description="minor impact objective",
                priority="low",
                confidence=1.0,
            ),
            SimpleNamespace(
                title="High",
                description="priority objective A",
                priority="high",
                confidence=0.2,
            ),
            SimpleNamespace(
                title="Critical",
                description="urgent reliability objective",
                priority="critical",
                confidence=0.1,
            ),
            SimpleNamespace(
                title="High2",
                description="priority objective B",
                priority="high",
                confidence=0.9,
            ),
        ]
        result = SimpleNamespace(goal_graph=SimpleNamespace(goals=goals))

        objectives = _extract_pipeline_objectives(result, max_goals=3)

        assert objectives[0].startswith("Critical")
        assert objectives[1].startswith("High2")
        assert objectives[2].startswith("High")

    def test_extract_pipeline_objectives_avoids_duplicate_title_description(self):
        goals = [
            SimpleNamespace(
                title="Achieve: Improve arbitration confidence scoring",
                description="Improve arbitration confidence scoring",
                priority="high",
                confidence=0.8,
            )
        ]
        result = SimpleNamespace(goal_graph=SimpleNamespace(goals=goals))

        objectives = _extract_pipeline_objectives(result, max_goals=1)

        assert objectives == ["Improve arbitration confidence scoring"]


class TestPipelineSelfImproveCommand:
    def _args(self, **overrides):
        base = {
            "goal": "Make Aragora more useful",
            "dry_run": True,
            "require_approval": False,
            "budget_limit": None,
            "execute": False,
            "max_goals": 1,
            "quick_mode": False,
            "max_parallel": 4,
        }
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_self_improve_planning_only_does_not_call_handoff(self, capsys):
        args = self._args(execute=False)

        with (
            patch.dict("sys.modules", _fake_module_payload()),
            patch("aragora.cli.commands.pipeline._run_self_improve_handoff") as mock_handoff,
        ):
            _cmd_pipeline_self_improve(args)

        out = capsys.readouterr().out
        assert "Handoff not executed" in out
        mock_handoff.assert_not_called()

    def test_self_improve_execute_calls_handoff_with_ranked_objective(self):
        args = self._args(execute=True, max_goals=1)

        with (
            patch.dict("sys.modules", _fake_module_payload()),
            patch("aragora.cli.commands.pipeline._run_self_improve_handoff") as mock_handoff,
        ):
            _cmd_pipeline_self_improve(args)

        mock_handoff.assert_called_once()
        call_args = mock_handoff.call_args
        objectives = call_args.args[0]
        assert len(objectives) == 1
        assert objectives[0].startswith("Bridge pipeline to self-improve")
        assert call_args.kwargs["dry_run"] is True
        assert call_args.kwargs["require_approval"] is False
        assert call_args.kwargs["max_parallel"] == 4

    def test_handoff_runner_uses_dry_run_mode(self):
        from aragora.cli.commands.pipeline import _run_self_improve_handoff

        def _fake_config(**kwargs):
            payload = {"budget_limit_usd": 10.0}
            payload.update(kwargs)
            return SimpleNamespace(**payload)

        fake_runner = SimpleNamespace(dry_run=AsyncMock(return_value={"goals": [], "subtasks": []}))
        fake_module = SimpleNamespace(
            SelfImproveConfig=_fake_config,
            SelfImprovePipeline=lambda config: fake_runner,
        )

        with patch.dict("sys.modules", {"aragora.nomic.self_improve": fake_module}):
            _run_self_improve_handoff(
                ["Improve pipeline/nomic integration"],
                dry_run=True,
                require_approval=False,
                budget_limit=None,
                quick_mode=True,
                max_parallel=2,
            )

        fake_runner.dry_run.assert_awaited_once()
