"""Tests for CLI gt command - Gas Town multi-agent orchestration."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.cli.gt import (
    _format_timestamp,
    _print_table,
    _normalize_priority,
    cmd_workspace_init,
    cmd_gt,
    add_gt_subparsers,
)


class TestFormatTimestamp:
    """Test _format_timestamp helper."""

    def test_formats_datetime(self):
        dt = datetime(2025, 6, 15, 14, 30, 0)
        result = _format_timestamp(dt)
        assert result == "2025-06-15 14:30:00"

    def test_handles_none(self):
        assert _format_timestamp(None) == "N/A"


class TestPrintTable:
    """Test _print_table helper."""

    def test_prints_header_and_rows(self, capsys):
        headers = ["Name", "Age"]
        rows = [["Alice", "30"], ["Bob", "25"]]
        _print_table(headers, rows)
        output = capsys.readouterr().out
        assert "Name" in output
        assert "Alice" in output
        assert "Bob" in output

    def test_custom_widths(self, capsys):
        headers = ["A", "B"]
        rows = [["x", "y"]]
        _print_table(headers, rows, widths=[10, 10])
        output = capsys.readouterr().out
        assert "A" in output

    def test_empty_rows(self, capsys):
        headers = ["Col1"]
        rows = []
        _print_table(headers, rows)
        output = capsys.readouterr().out
        assert "Col1" in output


class TestNormalizePriority:
    """Test _normalize_priority helper."""

    def test_uppercase(self):
        assert _normalize_priority("high") == "HIGH"

    def test_critical_to_urgent(self):
        assert _normalize_priority("critical") == "URGENT"

    def test_normal(self):
        assert _normalize_priority("normal") == "NORMAL"

    def test_with_whitespace(self):
        assert _normalize_priority("  low  ") == "LOW"

    def test_already_uppercase(self):
        assert _normalize_priority("HIGH") == "HIGH"


class TestCmdWorkspaceInit:
    """Test cmd_workspace_init command."""

    def test_creates_directories(self, tmp_path, capsys):
        target = tmp_path / "workspace"
        target.mkdir()
        args = argparse.Namespace(directory=str(target), force=False)
        result = cmd_workspace_init(args)
        assert result == 0

        # Check directories were created
        assert (target / ".gt").is_dir()
        assert (target / ".gt" / "convoys").is_dir()
        assert (target / ".gt" / "beads").is_dir()
        assert (target / ".gt" / "agents").is_dir()
        assert (target / ".gt" / "hooks").is_dir()

        # Check config file
        config_file = target / ".gt" / "config.json"
        assert config_file.exists()
        config = json.loads(config_file.read_text())
        assert config["version"] == "1.0"

    def test_default_directory(self, capsys):
        args = argparse.Namespace(directory=None, force=False)
        result = cmd_workspace_init(args)
        assert result == 0

    def test_force_overwrites_config(self, tmp_path, capsys):
        target = tmp_path / "workspace"
        target.mkdir()
        gt_dir = target / ".gt"
        gt_dir.mkdir()
        config_file = gt_dir / "config.json"
        config_file.write_text('{"version": "old"}')

        args = argparse.Namespace(directory=str(target), force=True)
        result = cmd_workspace_init(args)
        assert result == 0

        config = json.loads(config_file.read_text())
        assert config["version"] == "1.0"

    def test_does_not_overwrite_existing_config(self, tmp_path, capsys):
        target = tmp_path / "workspace"
        target.mkdir()
        gt_dir = target / ".gt"
        gt_dir.mkdir()
        config_file = gt_dir / "config.json"
        config_file.write_text('{"version": "existing"}')

        args = argparse.Namespace(directory=str(target), force=False)
        result = cmd_workspace_init(args)
        assert result == 0

        config = json.loads(config_file.read_text())
        assert config["version"] == "existing"


class TestCmdGtDispatcher:
    """Test cmd_gt dispatcher."""

    def test_no_subcommand(self, capsys):
        args = argparse.Namespace()
        # No func attribute
        result = cmd_gt(args)
        assert result == 0
        output = capsys.readouterr().out
        assert "Gas Town CLI" in output

    def test_with_func(self):
        mock_func = MagicMock(return_value=0)
        args = argparse.Namespace(func=mock_func)
        result = cmd_gt(args)
        assert result == 0
        mock_func.assert_called_once_with(args)

    def test_func_is_none(self, capsys):
        args = argparse.Namespace(func=None)
        result = cmd_gt(args)
        assert result == 0


class TestAddGtSubparsers:
    """Test that GT subparsers are properly configured."""

    def test_creates_gt_parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_gt_subparsers(subparsers)

        # Should create all subcommand groups
        args = parser.parse_args(["gt", "workspace", "init", "--force"])
        assert args.force is True

    def test_convoy_create_parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_gt_subparsers(subparsers)

        args = parser.parse_args(
            ["gt", "convoy", "create", "My Convoy", "--beads", "task1,task2", "--priority", "high"]
        )
        assert args.title == "My Convoy"
        assert args.beads == "task1,task2"
        assert args.priority == "high"

    def test_convoy_list_parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_gt_subparsers(subparsers)

        args = parser.parse_args(["gt", "convoy", "list", "--status", "active", "--limit", "5"])
        assert args.status == "active"
        assert args.limit == 5

    def test_bead_list_parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_gt_subparsers(subparsers)

        args = parser.parse_args(["gt", "bead", "list", "--status", "pending"])
        assert args.status == "pending"

    def test_bead_assign_parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_gt_subparsers(subparsers)

        args = parser.parse_args(["gt", "bead", "assign", "bead-123", "agent-456"])
        assert args.bead_id == "bead-123"
        assert args.agent_id == "agent-456"

    def test_agent_list_parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_gt_subparsers(subparsers)

        args = parser.parse_args(["gt", "agent", "list", "--role", "mayor"])
        assert args.role == "mayor"

    def test_agent_promote_parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_gt_subparsers(subparsers)

        args = parser.parse_args(["gt", "agent", "promote", "agent-1", "mayor"])
        assert args.agent_id == "agent-1"
        assert args.role == "mayor"

    def test_migrate_parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_gt_subparsers(subparsers)

        args = parser.parse_args(["gt", "migrate", "--apply", "--mode", "coordinator"])
        assert args.apply is True
        assert args.mode == "coordinator"


class TestConvoyCommands:
    """Test convoy command handlers with mocked stores."""

    def test_convoy_list_no_convoys(self, capsys):
        from aragora.cli.gt import cmd_convoy_list

        args = argparse.Namespace(status=None, limit=20)
        with patch("aragora.cli.gt._init_convoy_manager") as mock_init:
            mock_manager = MagicMock()
            mock_init.return_value = (MagicMock(), mock_manager)
            with patch("aragora.cli.gt._run_async", return_value=[]):
                result = cmd_convoy_list(args)
        assert result == 0

    def test_convoy_list_import_error(self, capsys):
        from aragora.cli.gt import cmd_convoy_list

        args = argparse.Namespace(status=None, limit=20)
        with patch("builtins.__import__", side_effect=ImportError("no stores")):
            result = cmd_convoy_list(args)
        assert result == 1

    def test_convoy_status_not_found(self, capsys):
        from aragora.cli.gt import cmd_convoy_status

        args = argparse.Namespace(convoy_id="nonexistent")
        with patch("aragora.cli.gt._init_convoy_manager") as mock_init:
            mock_init.return_value = (MagicMock(), MagicMock())
            with patch("aragora.cli.gt._run_async", return_value=None):
                result = cmd_convoy_status(args)
        assert result == 1


class TestBeadCommands:
    """Test bead command handlers."""

    def test_bead_assign_success(self, capsys):
        from aragora.cli.gt import cmd_bead_assign

        args = argparse.Namespace(bead_id="b1", agent_id="a1")
        with patch("aragora.cli.gt._init_bead_store") as mock_init:
            mock_store = MagicMock()
            mock_init.return_value = mock_store
            with patch("aragora.cli.gt._run_async", return_value=True):
                result = cmd_bead_assign(args)
        assert result == 0

    def test_bead_assign_failure(self, capsys):
        from aragora.cli.gt import cmd_bead_assign

        args = argparse.Namespace(bead_id="b1", agent_id="a1")
        with patch("aragora.cli.gt._init_bead_store") as mock_init:
            mock_init.return_value = MagicMock()
            with patch("aragora.cli.gt._run_async", return_value=False):
                result = cmd_bead_assign(args)
        assert result == 1

    def test_bead_list_invalid_status(self, capsys):
        from aragora.cli.gt import cmd_bead_list

        args = argparse.Namespace(status="invalid_status", convoy=None, limit=20)
        with patch("aragora.cli.gt.BeadStatus", create=True) as MockStatus:
            MockStatus.side_effect = ValueError("invalid")
            try:
                result = cmd_bead_list(args)
            except (ImportError, ValueError):
                result = 1
        assert result == 1
