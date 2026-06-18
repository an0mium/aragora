"""Tests for the native mission CLI commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.cli.parser import build_parser
from aragora.cli.commands.mission import cmd_mission
from aragora.nomic.mission import MissionSpec, WorkItem, WorkItemStatus


def test_mission_parser_args() -> None:
    """Test that the mission subcommand parser parses arguments correctly."""
    parser = build_parser()

    # 1. Parsing with all options
    args = parser.parse_args(
        [
            "mission",
            "Refactor auth",
            "--budget",
            "150.50",
            "--max-hours",
            "6.5",
            "--relay",
            "slack",
            "--auto-settle-max-tier",
            "1",
            "--tracks",
            "sme,qa",
        ]
    )

    assert args.command == "mission"
    assert args.goal == "Refactor auth"
    assert args.budget == 150.50
    assert args.max_hours == 6.5
    assert args.relay == "slack"
    assert args.auto_settle_max_tier == 1
    assert args.tracks == "sme,qa"

    # 2. Parsing with defaults
    args_default = parser.parse_args(["mission", "Do something"])
    assert args_default.command == "mission"
    assert args_default.goal == "Do something"
    assert args_default.budget is None
    assert args_default.max_hours is None
    assert args_default.relay == "none"
    assert args_default.auto_settle_max_tier == 2
    assert args_default.tracks is None


def test_mission_parser_invalid_relay() -> None:
    """Test that the parser rejects invalid relay choices."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["mission", "goal", "--relay", "invalid-relay"])


def test_cmd_mission_success(capsys: pytest.CaptureFixture[str]) -> None:
    """Test cmd_mission executes successfully when the feature flag is enabled."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "mission",
            "Refactor auth",
            "--budget",
            "50",
            "--relay",
            "email",
            "--tracks",
            "sme",
        ]
    )

    dummy_items = (
        WorkItem(
            item_id="wi-1",
            description="Refactor auth",
            status=WorkItemStatus.PENDING,
            complexity="medium",
        ),
    )

    # Mock NativeMissionRunner
    mock_runner_instance = MagicMock()
    mock_runner_instance.ingest_mission = AsyncMock(return_value=dummy_items)

    with patch(
        "aragora.cli.commands.mission.NativeMissionRunner", return_value=mock_runner_instance
    ):
        exit_code = cmd_mission(args)

        assert exit_code == 0

        captured = capsys.readouterr()
        assert "Ingesting mission" in captured.out
        assert "Refactor auth" in captured.out
        assert "Success: Mission" in captured.out
        assert "Decomposed into 1 work items" in captured.out
        assert "wi-1" in captured.out
        assert "medium" in captured.out

        # Verify runner was called correctly
        mock_runner_instance.ingest_mission.assert_called_once()
        called_spec = mock_runner_instance.ingest_mission.call_args[0][0]
        assert isinstance(called_spec, MissionSpec)
        assert called_spec.goal == "Refactor auth"
        assert called_spec.budget_usd == 50.0
        assert called_spec.relay == "email"
        assert mock_runner_instance.ingest_mission.call_args[1]["tracks"] == ["sme"]


def test_cmd_mission_disabled_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """Test cmd_mission handles RuntimeError when the feature flag is disabled."""
    parser = build_parser()
    args = parser.parse_args(["mission", "Refactor auth"])

    # Mock NativeMissionRunner to raise RuntimeError (matching disabled flag behavior)
    mock_runner_instance = MagicMock()
    mock_runner_instance.ingest_mission = AsyncMock(
        side_effect=RuntimeError("Native mission orchestrator is disabled")
    )

    with patch(
        "aragora.cli.commands.mission.NativeMissionRunner", return_value=mock_runner_instance
    ):
        exit_code = cmd_mission(args)

        assert exit_code == 1

        captured = capsys.readouterr()
        assert "Error: Native mission orchestrator is disabled" in captured.err


def test_cmd_mission_validation_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Test cmd_mission handles MissionSpec validation errors gracefully."""
    parser = build_parser()
    # Let's parse valid CLI arguments, but trigger a value error at Spec validation,
    # e.g., by passing a negative budget (which is allowed by parser float type but rejected by MissionSpec)
    args = parser.parse_args(["mission", "Goal", "--budget", "-10"])

    exit_code = cmd_mission(args)
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "Validation error: budget_usd must be non-negative" in captured.err
