"""Tests for ``aragora swarm`` CLI parser and command entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aragora.cli.commands.swarm import cmd_swarm


class _FakeSpec:
    def __init__(self, yaml_text: str = "id: test-spec\n") -> None:
        self._yaml_text = yaml_text

    def to_yaml(self) -> str:
        return self._yaml_text


class TestSwarmParser:
    def test_swarm_registered_in_root_parser(self):
        from aragora.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["swarm", "improve onboarding"])
        assert args.command == "swarm"
        assert args.goal == "improve onboarding"
        assert args.spec is None
        assert args.dry_run is False

    def test_swarm_parser_accepts_options(self):
        from aragora.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "swarm",
                "reduce latency",
                "--skip-interrogation",
                "--budget-limit",
                "12.5",
                "--require-approval",
                "--dry-run",
                "--save-spec",
                "swarm-spec.yaml",
            ]
        )
        assert args.command == "swarm"
        assert args.goal == "reduce latency"
        assert args.skip_interrogation is True
        assert args.budget_limit == 12.5
        assert args.require_approval is True
        assert args.dry_run is True
        assert args.save_spec == "swarm-spec.yaml"


class TestSwarmCommand:
    def test_cmd_swarm_requires_goal_or_spec(self, capsys):
        args = argparse.Namespace(
            goal=None,
            spec=None,
            skip_interrogation=False,
            dry_run=False,
            budget_limit=5.0,
            require_approval=False,
            save_spec=None,
        )
        cmd_swarm(args)
        out = capsys.readouterr().out
        assert "provide a goal or --spec file" in out

    def test_cmd_swarm_dry_run_saves_spec(self, tmp_path: Path):
        output_spec = tmp_path / "generated-spec.yaml"
        fake_spec = _FakeSpec("id: generated\n")
        mock_commander = SimpleNamespace(dry_run=AsyncMock(return_value=fake_spec))

        args = argparse.Namespace(
            goal="ship swarm",
            spec=None,
            skip_interrogation=False,
            dry_run=True,
            budget_limit=7.0,
            require_approval=False,
            save_spec=str(output_spec),
        )

        with patch("aragora.swarm.SwarmCommander", return_value=mock_commander):
            cmd_swarm(args)

        mock_commander.dry_run.assert_awaited_once()
        assert output_spec.exists()
        assert output_spec.read_text() == "id: generated\n"

    def test_cmd_swarm_skip_interrogation_dispatches(self):
        mock_commander = SimpleNamespace(run_from_spec=AsyncMock(return_value=None))
        args = argparse.Namespace(
            goal="harden CI",
            spec=None,
            skip_interrogation=True,
            dry_run=False,
            budget_limit=9.0,
            require_approval=True,
            save_spec=None,
        )

        with patch("aragora.swarm.SwarmCommander", return_value=mock_commander):
            cmd_swarm(args)

        mock_commander.run_from_spec.assert_awaited_once()
