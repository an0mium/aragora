"""Tests for ``aragora pipeline`` command behavior and parser wiring."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aragora.cli.commands.pipeline import (
    _cmd_pipeline_dogfood,
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
    run_calls = 0
    from_ideas_calls = 0
    from_assessment_calls = 0
    from_brain_dump_calls = 0
    force_run_error = False
    quality_gate_passed = True

    @classmethod
    def reset(cls) -> None:
        cls.run_calls = 0
        cls.from_ideas_calls = 0
        cls.from_assessment_calls = 0
        cls.from_brain_dump_calls = 0
        cls.force_run_error = False
        cls.quality_gate_passed = True

    async def run(self, _ideas_text, config=None):
        type(self).run_calls += 1
        if type(self).force_run_error:
            raise RuntimeError("simulated live failure")

        min_score = float(getattr(config, "plan_quality_min_score", 6.0))
        min_practicality = float(getattr(config, "plan_quality_min_practicality", 5.0))
        gate_passed = bool(type(self).quality_gate_passed)
        score = min_score + 0.5 if gate_passed else max(0.0, min_score - 1.0)
        practicality = min_practicality + 0.5 if gate_passed else max(0.0, min_practicality - 1.0)
        result = self.from_ideas(["live-generated"])
        result.metadata = {
            "plan_quality": {
                "gate_passed": gate_passed,
                "quality_score_10": score,
                "practicality_score_10": practicality,
                "min_quality_score_10": min_score,
                "min_practicality_score_10": min_practicality,
            }
        }
        result.stage_results = [
            SimpleNamespace(
                stage_name="ideation",
                status="completed",
                duration=0.25,
                output={"debate_result": {"provider": "fake"}},
            ),
            SimpleNamespace(
                stage_name="goals",
                status="completed",
                duration=0.18,
                output={},
            ),
        ]
        result.duration = 0.6
        return result

    def from_ideas(self, _ideas, auto_advance=True):
        type(self).from_ideas_calls += 1
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

    @classmethod
    async def from_assessment(cls, assessment):
        cls.from_assessment_calls += 1
        result = cls().from_ideas(["assessment-generated"])
        result.metadata = {"assessment_id": assessment.assessment_id}
        result.to_dict = lambda: {
            "pipeline_id": result.pipeline_id,
            "metadata": result.metadata,
        }
        return result

    @classmethod
    async def from_brain_dump(cls, text, automation_level="full"):
        cls.from_brain_dump_calls += 1
        result = cls().from_ideas([text])
        result.metadata = {"brain_dump_chars": len(text), "automation_level": automation_level}
        result.to_dict = lambda: {
            "pipeline_id": result.pipeline_id,
            "metadata": result.metadata,
        }
        return result


class _FakeAssessment:
    def __init__(self, assessment_id: str = "ca-123") -> None:
        self.assessment_id = assessment_id

    def to_dict(self) -> dict[str, object]:
        return {"assessment_id": self.assessment_id, "health_report": {"health_score": 0.82}}


class _FakeCanonicalAssessmentCompiler:
    async def compile(self):
        return _FakeAssessment()


class _FakeCampaignManifest:
    def __init__(self) -> None:
        self.campaign_id = "camp-123"
        self.projects = [SimpleNamespace(project_id="proj-1")]
        self.budget_limit_usd = 10.0
        self.time_limit_hours = 8.0
        self.max_parallel_ready_projects = 2
        self.max_retries_per_project = 2
        self.planning_findings: list[str] = []

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_id": self.campaign_id,
            "projects": [{"project_id": "proj-1"}],
            "planning_findings": list(self.planning_findings),
            "budget_limit_usd": self.budget_limit_usd,
            "time_limit_hours": self.time_limit_hours,
            "max_parallel_ready_projects": self.max_parallel_ready_projects,
            "max_retries_per_project": self.max_retries_per_project,
        }

    def to_yaml(self) -> str:
        return "campaign_id: camp-123\nprojects:\n  - project_id: proj-1\n"


class _FakeCampaignPlanner:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def plan_from_items(self, items, *, source_kind: str, source_ref: str):
        manifest = _FakeCampaignManifest()
        manifest.source_kind = source_kind
        manifest.source_ref = source_ref
        manifest.items = list(items)
        return manifest


class _FakeBrainDumpParser:
    def parse_enriched(self, text: str):
        return SimpleNamespace(ideas=[text.strip()])


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


def _fake_dogfood_module_payload(
    saved_assessment_calls=None, load_latest=None
) -> dict[str, object]:
    if saved_assessment_calls is None:
        saved_assessment_calls = []
    if load_latest is None:
        load_latest = lambda: _FakeAssessment("ca-latest")  # noqa: E731

    def _save_assessment(assessment):
        saved_assessment_calls.append(assessment.assessment_id)
        return assessment.assessment_id

    def _save_campaign_manifest(path, manifest):
        path.write_text(manifest.to_yaml(), encoding="utf-8")

    return {
        "aragora.nomic.canonical_assessment": SimpleNamespace(
            CanonicalAssessmentCompiler=_FakeCanonicalAssessmentCompiler,
            load_assessment=lambda assessment_id: _FakeAssessment(assessment_id),
            load_latest_assessment=load_latest,
            save_assessment=_save_assessment,
        ),
        "aragora.pipeline.brain_dump_parser": SimpleNamespace(
            BrainDumpParser=_FakeBrainDumpParser,
        ),
        "aragora.pipeline.idea_to_execution": SimpleNamespace(
            IdeaToExecutionPipeline=_FakeIdeaToExecutionPipeline,
        ),
        "aragora.swarm.campaign": SimpleNamespace(
            CampaignPlanner=_FakeCampaignPlanner,
            save_campaign_manifest=_save_campaign_manifest,
        ),
        "aragora.ralph.supervisor": SimpleNamespace(
            RalphSupervisor=SimpleNamespace(
                start=lambda **kwargs: SimpleNamespace(
                    status=lambda: {
                        "status": "running",
                        "supervisor_id": "ralph-123",
                        "manifest_path": str(kwargs["manifest_path"]),
                    }
                )
            )
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
                "--pipeline-mode",
                "hybrid",
                "--plan-quality-min-score",
                "7.5",
                "--plan-quality-min-practicality",
                "6.0",
                "--plan-quality-fail-closed",
            ]
        )

        assert args.command == "pipeline"
        assert args.pipeline_action == "self-improve"
        assert args.execute is True
        assert args.max_goals == 3
        assert args.quick_mode is True
        assert args.max_parallel == 2
        assert args.pipeline_mode == "hybrid"
        assert args.plan_quality_min_score == 7.5
        assert args.plan_quality_min_practicality == 6.0
        assert args.plan_quality_fail_closed is True

    def test_pipeline_dogfood_parser_accepts_handoff_flags(self):
        from aragora.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "pipeline",
                "dogfood",
                "--latest-assessment",
                "--output-dir",
                ".aragora/dogfood/test",
                "--max-goals",
                "2",
                "--budget-limit",
                "12.5",
                "--start-ralph",
                "--json",
            ]
        )

        assert args.command == "pipeline"
        assert args.pipeline_action == "dogfood"
        assert args.latest_assessment is True
        assert args.max_goals == 2
        assert args.budget_limit == 12.5
        assert args.start_ralph is True
        assert args.json is True


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

        with patch(
            "aragora.cli.commands.pipeline._pipeline_similarity_backend",
            return_value=SimpleNamespace(compute_similarity=lambda _a, _b: 0.0),
        ):
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

        with patch(
            "aragora.cli.commands.pipeline._pipeline_similarity_backend",
            return_value=SimpleNamespace(compute_similarity=lambda _a, _b: 0.9),
        ):
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
            "pipeline_mode": "live",
            "plan_quality_contract_file": None,
            "plan_quality_fail_closed": False,
            "plan_quality_min_score": 6.0,
            "plan_quality_min_practicality": 5.0,
        }
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_self_improve_planning_only_does_not_call_handoff(self, capsys):
        args = self._args(execute=False, dry_run=False)
        _FakeIdeaToExecutionPipeline.reset()

        with (
            patch.dict("sys.modules", _fake_module_payload()),
            patch("aragora.cli.commands.pipeline._run_self_improve_handoff") as mock_handoff,
        ):
            _cmd_pipeline_self_improve(args)

        out = capsys.readouterr().out
        assert "Handoff not executed" in out
        mock_handoff.assert_not_called()
        assert _FakeIdeaToExecutionPipeline.run_calls == 1
        assert _FakeIdeaToExecutionPipeline.from_ideas_calls == 1

    def test_self_improve_dry_run_without_execute_skips_handoff(self, capsys):
        args = self._args(execute=False, dry_run=True)
        _FakeIdeaToExecutionPipeline.reset()

        with (
            patch.dict("sys.modules", _fake_module_payload()),
            patch("aragora.cli.commands.pipeline._run_self_improve_handoff") as mock_handoff,
        ):
            _cmd_pipeline_self_improve(args)

        out = capsys.readouterr().out
        assert "Handoff skipped in dry-run mode" in out
        mock_handoff.assert_not_called()
        assert _FakeIdeaToExecutionPipeline.run_calls == 1
        assert _FakeIdeaToExecutionPipeline.from_ideas_calls == 1

    def test_self_improve_execute_calls_handoff_with_ranked_objective(self):
        args = self._args(execute=True, max_goals=1)
        _FakeIdeaToExecutionPipeline.reset()

        with (
            patch.dict("sys.modules", _fake_module_payload()),
            patch("aragora.cli.commands.pipeline._run_self_improve_handoff") as mock_handoff,
        ):
            _cmd_pipeline_self_improve(args)

        mock_handoff.assert_called_once()
        call_args = mock_handoff.call_args
        objectives = call_args.args[0]
        assert len(objectives) == 1
        assert objectives[0] == "Make Aragora more useful"
        assert call_args.kwargs["dry_run"] is True
        assert call_args.kwargs["require_approval"] is False
        assert call_args.kwargs["max_parallel"] == 4
        assert _FakeIdeaToExecutionPipeline.run_calls == 1

    def test_self_improve_hybrid_falls_back_to_heuristic(self, capsys):
        args = self._args(execute=False, dry_run=False, pipeline_mode="hybrid")
        _FakeIdeaToExecutionPipeline.reset()
        _FakeIdeaToExecutionPipeline.force_run_error = True

        with (
            patch.dict("sys.modules", _fake_module_payload()),
            patch("aragora.cli.commands.pipeline._run_self_improve_handoff") as mock_handoff,
        ):
            _cmd_pipeline_self_improve(args)

        out = capsys.readouterr().out
        assert "heuristic fallback" in out.lower()
        assert "Execution path: heuristic-fallback" in out
        assert _FakeIdeaToExecutionPipeline.run_calls == 1
        assert _FakeIdeaToExecutionPipeline.from_ideas_calls >= 1
        mock_handoff.assert_not_called()

    def test_self_improve_fail_closed_blocks_handoff(self, capsys):
        args = self._args(
            execute=True,
            plan_quality_fail_closed=True,
            plan_quality_min_score=8.0,
            plan_quality_min_practicality=7.0,
        )
        _FakeIdeaToExecutionPipeline.reset()
        _FakeIdeaToExecutionPipeline.quality_gate_passed = False

        with (
            patch.dict("sys.modules", _fake_module_payload()),
            patch("aragora.cli.commands.pipeline._run_self_improve_handoff") as mock_handoff,
        ):
            _cmd_pipeline_self_improve(args)

        out = capsys.readouterr().out
        assert "Blocking handoff" in out
        mock_handoff.assert_not_called()

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


class TestPipelineDogfoodCommand:
    def _args(self, tmp_path, **overrides):
        base = {
            "source_file": None,
            "assessment_id": None,
            "latest_assessment": False,
            "save_assessment": False,
            "output_dir": str(tmp_path / "dogfood"),
            "max_goals": 1,
            "budget_limit": 10.0,
            "time_limit_hours": 4.0,
            "max_parallel_ready_projects": 1,
            "max_retries_per_project": 2,
            "planner_model": "claude",
            "worker_model": "codex",
            "review_model": "claude",
            "start_ralph": False,
            "merge_policy": "manual_review_required",
            "max_repair_attempts": 2,
            "json": False,
        }
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_dogfood_from_assessment_writes_handoff_artifacts(self, tmp_path, capsys):
        args = self._args(tmp_path, save_assessment=True)
        _FakeIdeaToExecutionPipeline.reset()
        saved_assessment_calls: list[str] = []

        with patch.dict("sys.modules", _fake_dogfood_module_payload(saved_assessment_calls)):
            _cmd_pipeline_dogfood(args)

        out = capsys.readouterr().out
        assert "DOGFOOD HANDOFF" in out
        assert _FakeIdeaToExecutionPipeline.from_assessment_calls == 1
        assert saved_assessment_calls == ["ca-123"]
        assert (tmp_path / "dogfood" / "assessment.json").exists()
        assert (tmp_path / "dogfood" / "pipeline_result.json").exists()
        assert (tmp_path / "dogfood" / "objectives.json").exists()
        assert (tmp_path / "dogfood" / "campaign_manifest.yaml").exists()

    def test_dogfood_from_source_file_uses_parsed_ideas(self, tmp_path):
        args = self._args(tmp_path, source_file=str(tmp_path / "ROADMAP.md"), json=True)
        (tmp_path / "ROADMAP.md").write_text("Ship the dogfood loop end to end.", encoding="utf-8")
        _FakeIdeaToExecutionPipeline.reset()

        with patch.dict("sys.modules", _fake_dogfood_module_payload()):
            _cmd_pipeline_dogfood(args)

        assert _FakeIdeaToExecutionPipeline.from_ideas_calls == 1
        payload = json.loads((tmp_path / "dogfood" / "pipeline_result.json").read_text())
        assert payload["pipeline_id"] == "pipe-123"

    def test_dogfood_can_start_ralph(self, tmp_path, capsys):
        args = self._args(tmp_path, latest_assessment=True, start_ralph=True, json=True)
        _FakeIdeaToExecutionPipeline.reset()

        with patch.dict("sys.modules", _fake_dogfood_module_payload()):
            _cmd_pipeline_dogfood(args)

        out = json.loads(capsys.readouterr().out)
        assert out["ralph_status"] == "running"
        assert out["ralph_supervisor_id"] == "ralph-123"
        assert out["manifest_path"].endswith("campaign_manifest.yaml")
