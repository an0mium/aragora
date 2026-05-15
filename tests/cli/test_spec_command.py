from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from aragora.agents.errors import AgentCircuitOpenError
from aragora.cli.commands.spec import _run_spec_pipeline, cmd_spec
from aragora.cli.parser import build_parser


class _FakeConductorConfig:
    last_profile: str | None = None

    def __init__(self) -> None:
        self.interrogation_depth = None
        self.skip_research = False
        self.skip_interrogation = False

    @classmethod
    def from_profile(cls, profile: str):
        cls.last_profile = profile
        return cls()


class _FakeOpenAIAPIAgent:
    last_run: dict[str, object] | None = None

    def __init__(self, *, name, model, role) -> None:
        type(self).last_run = {
            "name": name,
            "model": model,
            "role": role,
        }


class _FakeRecord:
    def __init__(self, **data) -> None:
        self._data = data
        for key, value in data.items():
            setattr(self, key, value)

    def to_dict(self) -> dict[str, object]:
        return dict(self._data)


class _FakeTiming:
    def to_dict(self) -> dict[str, object]:
        return {
            "total_duration_ms": 123.0,
            "target_duration_ms": 15_000.0,
            "tracking_coverage_pct": 100.0,
            "stage_breakdown": [
                {"stage": "specify", "duration_ms": 70.0, "share_of_total_pct": 56.9}
            ],
            "optimization_targets": [
                {
                    "operation": "specify.agent_generate",
                    "duration_ms": 70.0,
                    "share_of_total_pct": 56.9,
                    "optimization_hint": "Reduce prompt size, model latency, or round trips.",
                }
            ],
        }


class _FakePromptConductor:
    last_init: dict[str, object] | None = None
    last_run: dict[str, object] | None = None

    def __init__(self, *, config, agent) -> None:
        type(self).last_init = {
            "interrogation_depth": config.interrogation_depth,
            "skip_research": config.skip_research,
            "skip_interrogation": config.skip_interrogation,
            "agent": agent,
        }

    async def run(self, prompt: str):
        type(self).last_run = {
            "prompt": prompt,
        }
        return _FakeRecord(
            specification=_FakeRecord(
                problem_statement="Need a better onboarding flow.",
                proposed_solution="Add a guided setup sequence.",
                success_criteria=[_FakeRecord(description="Users finish setup faster.")],
                risks=[_FakeRecord(description="Higher implementation complexity.")],
                estimated_effort="medium",
                confidence=0.8,
            ),
            intent=_FakeRecord(intent_type="feature", scope_estimate="medium"),
            research=_FakeRecord(evidence_links=["km://onboarding"]),
            questions=[_FakeRecord(question="What should improve first?")],
            stages_completed=["decompose", "specify"],
            auto_approved=False,
            timing=_FakeRecord(
                total_duration_ms=42.5,
                slowest_stage={"stage": "specify", "duration_ms": 20.0},
                top_operations=[
                    {"operation": "specify.agent_generate", "duration_ms": 19.0},
                    {"operation": "decompose.agent_generate", "duration_ms": 14.0},
                ],
            ),
        )


class TestSpecParser:
    def test_spec_command_parses_flags_and_handler(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "spec",
                "Make onboarding better",
                "--depth",
                "thorough",
                "--profile",
                "cto",
                "--skip-research",
                "--skip-interrogation",
                "--format",
                "json",
                "--output",
                "spec.json",
                "--to-mission",
                "mission.yaml",
                "--dry-run",
            ]
        )

        assert args.command == "spec"
        assert args.prompt == "Make onboarding better"
        assert args.depth == "thorough"
        assert args.profile == "cto"
        assert args.skip_research is True
        assert args.skip_interrogation is True
        assert args.format == "json"
        assert args.output == "spec.json"
        assert args.to_mission == "mission.yaml"
        assert args.dry_run is True
        assert args.func.__name__ == "cmd_spec"


class TestRunSpecPipeline:
    @pytest.mark.asyncio
    async def test_run_spec_pipeline_maps_depth_profile_and_flags(self):
        fake_types = SimpleNamespace(
            InterrogationDepth=SimpleNamespace(
                QUICK="quick-depth",
                THOROUGH="thorough-depth",
                EXHAUSTIVE="exhaustive-depth",
            ),
            UserProfile=SimpleNamespace(
                FOUNDER="founder-profile",
                CTO="cto-profile",
                BUSINESS="business-profile",
                TEAM="team-profile",
            ),
        )

        with patch.dict(
            "sys.modules",
            {
                "aragora.prompt_engine.conductor": SimpleNamespace(
                    ConductorConfig=_FakeConductorConfig,
                    PromptConductor=_FakePromptConductor,
                ),
                "aragora.agents.api_agents.openai": SimpleNamespace(
                    OpenAIAPIAgent=_FakeOpenAIAPIAgent
                ),
                "aragora.prompt_engine.types": fake_types,
            },
        ):
            result = await _run_spec_pipeline(
                "Design a better onboarding flow",
                depth="thorough",
                skip_research=True,
                skip_interrogation=True,
                profile="cto",
            )

        assert _FakeConductorConfig.last_profile == "cto"
        assert _FakePromptConductor.last_init == {
            "interrogation_depth": "thorough-depth",
            "skip_research": True,
            "skip_interrogation": True,
            "agent": _FakePromptConductor.last_init["agent"],
        }
        assert _FakeOpenAIAPIAgent.last_run == {
            "name": "spec-agent",
            "model": "gpt-4o-mini",
            "role": "proposer",
        }
        assert _FakePromptConductor.last_run == {
            "prompt": "Design a better onboarding flow",
        }
        assert result["intent"]["intent_type"] == "feature"
        assert result["specification"]["problem_statement"] == "Need a better onboarding flow."
        assert result["questions"] == [{"question": "What should improve first?"}]
        assert result["stages_completed"] == ["decompose", "specify"]
        assert result["timing"]["slowest_stage"]["stage"] == "specify"


class TestCmdSpec:
    def test_cmd_spec_dry_run_does_not_execute_pipeline(self, capsys):
        args = argparse.Namespace(
            prompt="Make onboarding better",
            depth="quick",
            profile="founder",
            skip_research=False,
            skip_interrogation=False,
            format="text",
            dry_run=True,
            output=None,
        )

        with patch(
            "aragora.cli.commands.spec._run_spec_pipeline", new_callable=AsyncMock
        ) as run_spec:
            cmd_spec(args)

        out = capsys.readouterr().out
        assert "[dry-run]" in out
        run_spec.assert_not_called()

    def test_cmd_spec_runs_pipeline_and_writes_output(self, tmp_path, capsys):
        output_path = tmp_path / "spec.json"
        result = {
            "intent": {"intent_type": "feature", "scope_estimate": "medium"},
            "specification": {
                "problem_statement": "Need faster onboarding.",
                "proposed_solution": "Create a guided setup checklist.",
                "success_criteria": [{"description": "Reduce median setup time."}],
                "risks": [{"description": "Could overwhelm new users."}],
                "estimated_effort": "medium",
                "confidence": 0.75,
            },
            "research": {"evidence_links": ["km://onboarding"]},
            "timing": {
                "total_duration_ms": 84.0,
                "slowest_stage": {"stage": "specify", "duration_ms": 41.0},
                "top_operations": [
                    {"operation": "specify.agent_generate", "duration_ms": 39.0},
                ],
            },
        }
        args = argparse.Namespace(
            prompt="Make onboarding better",
            depth="quick",
            profile="founder",
            skip_research=False,
            skip_interrogation=False,
            format="json",
            dry_run=False,
            output=str(output_path),
        )

        with patch(
            "aragora.cli.commands.spec._run_spec_pipeline",
            new_callable=AsyncMock,
            return_value=result,
        ) as run_spec:
            cmd_spec(args)

        out = capsys.readouterr().out
        assert "ARAGORA SPEC" in out
        assert "Elapsed:" in out
        assert '"total_duration_ms": 84.0' in out
        assert '"stage": "specify"' in out
        assert "Spec saved to:" in out
        run_spec.assert_awaited_once_with(
            "Make onboarding better",
            depth="quick",
            skip_research=False,
            skip_interrogation=False,
            profile="founder",
            output_format="json",
            use_orchestrator=False,
        )
        assert json.loads(output_path.read_text()) == result

    def test_cmd_spec_writes_conductor_mission_without_dispatch(self, tmp_path, capsys):
        mission_path = tmp_path / "mission.yaml"
        result = {
            "intent": {"intent_type": "benchmark", "scope_estimate": "medium"},
            "specification": {
                "title": "Publish H1-01 rev-4 benchmark result",
                "problem_statement": "The benchmark result needs to become a public artifact.",
                "proposed_solution": (
                    "Run the existing benchmark publication path and write the result."
                ),
                "success_criteria": [{"description": "Mission dry-run handoff exists."}],
                "confidence": 0.8,
            },
            "research": None,
            "timing": {},
        }
        args = argparse.Namespace(
            prompt="publish H1-01 rev-4 benchmark result",
            depth="quick",
            profile="founder",
            skip_research=False,
            skip_interrogation=False,
            format="text",
            dry_run=False,
            output=None,
            orchestrator=False,
            to_mission=str(mission_path),
        )

        with patch(
            "aragora.cli.commands.spec._run_spec_pipeline",
            new_callable=AsyncMock,
            return_value=result,
        ):
            cmd_spec(args)

        out = capsys.readouterr().out
        mission = yaml.safe_load(mission_path.read_text())

        assert "Conductor mission saved to:" in out
        assert "goal_conductor.py run-once" in out
        assert mission["objective"] == "publish H1-01 rev-4 benchmark result"
        impl_lanes = [lane for lane in mission["lanes"] if lane["mode"] == "implementation"]
        assert len(impl_lanes) <= 2
        assert mission["lanes"][-1]["mode"] == "panel"

    def test_cmd_spec_writes_text_output_when_requested(self, tmp_path, capsys):
        output_path = tmp_path / "spec.txt"
        result = {
            "intent": {"intent_type": "feature", "scope_estimate": "medium"},
            "specification": {
                "problem_statement": "Need faster onboarding.",
                "proposed_solution": "Create a guided setup checklist.",
                "success_criteria": [{"description": "Reduce median setup time."}],
                "risks": [{"description": "Could overwhelm new users."}],
                "estimated_effort": "medium",
                "confidence": 0.75,
            },
            "research": {"evidence_links": ["km://onboarding"]},
            "pipeline": "orchestrator",
        }
        args = argparse.Namespace(
            prompt="Make onboarding better",
            depth="quick",
            profile="founder",
            skip_research=False,
            skip_interrogation=False,
            format="text",
            dry_run=False,
            output=str(output_path),
        )

        with patch(
            "aragora.cli.commands.spec._run_spec_pipeline",
            new_callable=AsyncMock,
            return_value=result,
        ):
            cmd_spec(args)

        out = capsys.readouterr().out
        saved = output_path.read_text()

        assert "ARAGORA SPEC" in out
        assert "SPECIFICATION" in saved
        assert "Need faster onboarding." in saved
        assert "Create a guided setup checklist." in saved
        assert "Pipeline:   orchestrator" in saved
        assert saved.lstrip().startswith("=")

    def test_cmd_spec_uses_truthful_fallback_when_spec_agent_circuit_is_open(
        self, tmp_path, capsys
    ):
        output_path = tmp_path / "spec-fallback.txt"
        args = argparse.Namespace(
            prompt="Make onboarding better",
            depth="quick",
            profile="founder",
            skip_research=False,
            skip_interrogation=False,
            format="text",
            dry_run=False,
            output=str(output_path),
        )

        with patch(
            "aragora.cli.commands.spec._run_spec_pipeline",
            new_callable=AsyncMock,
            side_effect=AgentCircuitOpenError(
                "Circuit breaker open for spec-agent",
                agent_name="spec-agent",
            ),
        ) as run_spec:
            cmd_spec(args)

        out = capsys.readouterr().out
        saved = output_path.read_text()

        assert "ARAGORA SPEC" in out
        assert "Pipeline:   spec_fallback" in out
        assert "Fallback spec (spec-agent circuit breaker open)" in out
        assert "SPECIFICATION" in saved
        assert "Pipeline:   spec_fallback" in saved
        assert "Fallback spec (spec-agent circuit breaker open)" in saved
        assert "Make onboarding better" in saved
        run_spec.assert_awaited_once_with(
            "Make onboarding better",
            depth="quick",
            skip_research=False,
            skip_interrogation=False,
            profile="founder",
            output_format="text",
            use_orchestrator=False,
        )

    def test_cmd_spec_requires_prompt(self):
        args = argparse.Namespace(
            prompt=None,
            depth="quick",
            profile="founder",
            skip_research=False,
            skip_interrogation=False,
            format="text",
            dry_run=False,
            output=None,
        )

        with pytest.raises(SystemExit) as exc:
            cmd_spec(args)

        assert exc.value.code == 1
