from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from aragora.cli.commands.spec import _run_spec_pipeline, cmd_spec
from aragora.cli.parser import build_parser


class _FakeSpecConductor:
    last_init: dict[str, object] | None = None
    last_run: dict[str, object] | None = None

    def __init__(self, *, interrogation_depth, user_profile) -> None:
        type(self).last_init = {
            "interrogation_depth": interrogation_depth,
            "user_profile": user_profile,
        }

    async def run(self, prompt: str, *, skip_research: bool, skip_interrogation: bool):
        type(self).last_run = {
            "prompt": prompt,
            "skip_research": skip_research,
            "skip_interrogation": skip_interrogation,
        }
        return {
            "specification": {
                "problem_statement": "Need a better onboarding flow.",
                "proposed_solution": "Add a guided setup sequence.",
                "success_criteria": [{"description": "Users finish setup faster."}],
                "risks": [{"description": "Higher implementation complexity."}],
                "estimated_effort": "medium",
                "confidence": 0.8,
            }
        }


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
                    SpecConductor=_FakeSpecConductor
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

        assert _FakeSpecConductor.last_init == {
            "interrogation_depth": "thorough-depth",
            "user_profile": "cto-profile",
        }
        assert _FakeSpecConductor.last_run == {
            "prompt": "Design a better onboarding flow",
            "skip_research": True,
            "skip_interrogation": True,
        }
        assert "specification" in result


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
        assert "Spec saved to:" in out
        run_spec.assert_awaited_once_with(
            "Make onboarding better",
            depth="quick",
            skip_research=False,
            skip_interrogation=False,
            profile="founder",
            output_format="json",
        )
        assert json.loads(output_path.read_text()) == result

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
