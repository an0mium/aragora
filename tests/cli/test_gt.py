"""
Tests for aragora.cli.gt module.

Tests Gas Town CLI commands for multi-agent orchestration.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.cli.gt import (
    _format_timestamp,
    _print_table,
    _run_async,
    cmd_bead_assign,
    cmd_bead_list,
    cmd_convoy_create,
    cmd_convoy_list,
    cmd_convoy_status,
    cmd_agent_list,
    cmd_agent_promote,
    cmd_witness_status,
    cmd_workspace_init,
    cmd_gt,
    add_gt_subparsers,
)


# ===========================================================================
# Test Fixtures and Mock Classes
# ===========================================================================


class MockStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class MockPriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class MockRole(Enum):
    MAYOR = "mayor"
    WITNESS = "witness"
    POLECAT = "polecat"
    CREW = "crew"


@dataclass
class MockBead:
    """Mock bead for testing."""

    id: str = "bead-12345678"
    title: str = "Test Bead"
    status: MockStatus = MockStatus.PENDING
    priority: MockPriority = MockPriority.NORMAL
    assigned_to: str | None = None
    convoy_id: str | None = None


@dataclass
class MockConvoy:
    """Mock convoy for testing."""

    id: str = "convoy-12345678"
    title: str = "Test Convoy"
    status: MockStatus = MockStatus.IN_PROGRESS
    total_beads: int = 5
    completed_beads: int = 2
    created_at: datetime = None
    beads: list = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.beads is None:
            self.beads = [MockBead()]


@dataclass
class MockAgent:
    """Mock agent for testing."""

    agent_id: str = "agent-12345678"
    role: MockRole = MockRole.CREW
    supervised_by: str | None = None
    assigned_at: datetime = None

    def __post_init__(self):
        if self.assigned_at is None:
            self.assigned_at = datetime.now()


# ===========================================================================
# Tests: Helper Functions
# ===========================================================================


class TestRunAsync:
    """Tests for _run_async function."""

    def test_runs_coroutine(self):
        """Test running an async coroutine."""

        async def sample_coro():
            return "result"

        result = _run_async(sample_coro())
        assert result == "result"


class TestFormatTimestamp:
    """Tests for _format_timestamp function."""

    def test_formats_datetime(self):
        """Test formatting a datetime."""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        result = _format_timestamp(dt)
        assert result == "2024-01-15 10:30:45"

    def test_handles_none(self):
        """Test handling None datetime."""
        result = _format_timestamp(None)
        assert result == "N/A"


class TestPrintTable:
    """Tests for _print_table function."""

    def test_prints_table(self, capsys):
        """Test printing a table."""
        headers = ["A", "B", "C"]
        rows = [["1", "2", "3"], ["4", "5", "6"]]
        _print_table(headers, rows)

        captured = capsys.readouterr()
        assert "A" in captured.out
        assert "B" in captured.out
        assert "C" in captured.out
        assert "-" in captured.out

    def test_with_custom_widths(self, capsys):
        """Test printing with custom widths."""
        headers = ["Name", "Value"]
        rows = [["X", "Y"]]
        _print_table(headers, rows, [10, 15])

        captured = capsys.readouterr()
        assert "Name" in captured.out


# ===========================================================================
# Tests: Convoy Commands
# ===========================================================================


class TestCmdConvoyList:
    """Tests for cmd_convoy_list function."""

    def test_empty_list(self, capsys):
        """Test with no convoys."""
        args = argparse.Namespace(status=None, limit=20)

        mock_manager = MagicMock()
        mock_module = MagicMock()
        mock_module.ConvoyManager.return_value = mock_manager

        with patch.dict("sys.modules", {"aragora.nomic.convoys": mock_module}):
            with patch("aragora.cli.gt._run_async", return_value=[]):
                result = cmd_convoy_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No convoys found" in captured.out

    def test_with_convoys(self, capsys):
        """Test with convoys in list."""
        args = argparse.Namespace(status=None, limit=20)

        mock_convoy = MockConvoy()
        mock_manager = MagicMock()
        mock_module = MagicMock()
        mock_module.ConvoyManager.return_value = mock_manager
        mock_module.ConvoyStatus = MockStatus

        with patch.dict("sys.modules", {"aragora.nomic.convoys": mock_module}):
            with patch("aragora.cli.gt._run_async", return_value=[mock_convoy]):
                result = cmd_convoy_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Found 1 convoy" in captured.out
        assert "Test Convoy" in captured.out

    def test_invalid_status_filter(self, capsys):
        """Test with invalid status filter."""
        args = argparse.Namespace(status="invalid_status", limit=20)

        mock_module = MagicMock()
        mock_module.ConvoyManager.return_value = MagicMock()
        mock_module.ConvoyStatus = MockStatus
        mock_module.ConvoyStatus.side_effect = ValueError()

        with patch.dict("sys.modules", {"aragora.nomic.convoys": mock_module}):
            result = cmd_convoy_list(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Invalid status" in captured.out

    def test_import_error(self, capsys):
        """Test handling import error."""
        args = argparse.Namespace(status=None, limit=20)

        with patch.dict("sys.modules", {"aragora.nomic.convoys": None}):
            result = cmd_convoy_list(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "not available" in captured.out


class TestCmdConvoyCreate:
    """Tests for cmd_convoy_create function."""

    def test_create_success(self, capsys):
        """Test creating a convoy."""
        args = argparse.Namespace(
            title="New Convoy",
            beads="task1,task2,task3",
            description="Test description",
            priority="normal",
        )

        mock_convoy = MockConvoy(title="New Convoy")
        mock_module = MagicMock()
        mock_beads_module = MagicMock()
        mock_beads_module.BeadPriority = {"NORMAL": MockPriority.NORMAL}

        with patch.dict(
            "sys.modules",
            {
                "aragora.nomic.convoys": mock_module,
                "aragora.nomic.beads": mock_beads_module,
            },
        ):
            with patch("aragora.cli.gt._run_async", return_value=mock_convoy):
                result = cmd_convoy_create(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Convoy created" in captured.out

    def test_no_beads(self, capsys):
        """Test error when no beads specified."""
        args = argparse.Namespace(
            title="New Convoy",
            beads="",  # Empty beads
            description=None,
            priority="normal",
        )

        mock_module = MagicMock()
        mock_beads_module = MagicMock()
        mock_beads_module.BeadPriority = {"NORMAL": MockPriority.NORMAL}

        with patch.dict(
            "sys.modules",
            {
                "aragora.nomic.convoys": mock_module,
                "aragora.nomic.beads": mock_beads_module,
            },
        ):
            result = cmd_convoy_create(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "At least one bead is required" in captured.out


class TestCmdConvoyStatus:
    """Tests for cmd_convoy_status function."""

    def test_convoy_not_found(self, capsys):
        """Test when convoy not found."""
        args = argparse.Namespace(convoy_id="nonexistent")

        mock_module = MagicMock()
        mock_module.ConvoyManager.return_value = MagicMock()

        with patch.dict("sys.modules", {"aragora.nomic.convoys": mock_module}):
            with patch("aragora.cli.gt._run_async", return_value=None):
                result = cmd_convoy_status(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_convoy_found(self, capsys):
        """Test when convoy is found."""
        args = argparse.Namespace(convoy_id="convoy-123")

        mock_convoy = MockConvoy()
        mock_module = MagicMock()

        with patch.dict("sys.modules", {"aragora.nomic.convoys": mock_module}):
            with patch("aragora.cli.gt._run_async", return_value=mock_convoy):
                result = cmd_convoy_status(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Test Convoy" in captured.out
        assert "Progress:" in captured.out


# ===========================================================================
# Tests: Bead Commands
# ===========================================================================


class TestCmdBeadList:
    """Tests for cmd_bead_list function."""

    def test_empty_list(self, capsys):
        """Test with no beads."""
        args = argparse.Namespace(status=None, convoy=None, limit=20)

        mock_module = MagicMock()
        mock_module.BeadStatus = MockStatus

        with patch.dict("sys.modules", {"aragora.nomic.beads": mock_module}):
            with patch("aragora.cli.gt._run_async", return_value=[]):
                result = cmd_bead_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No beads found" in captured.out

    def test_with_beads(self, capsys):
        """Test with beads in list."""
        args = argparse.Namespace(status=None, convoy=None, limit=20)

        mock_bead = MockBead()
        mock_module = MagicMock()
        mock_module.BeadStatus = MockStatus

        with patch.dict("sys.modules", {"aragora.nomic.beads": mock_module}):
            with patch("aragora.cli.gt._run_async", return_value=[mock_bead]):
                result = cmd_bead_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Found 1 bead" in captured.out
        assert "Test Bead" in captured.out


class TestCmdBeadAssign:
    """Tests for cmd_bead_assign function."""

    def test_assign_success(self, capsys):
        """Test successful bead assignment."""
        args = argparse.Namespace(bead_id="bead-123", agent_id="agent-456")

        mock_module = MagicMock()

        with patch.dict("sys.modules", {"aragora.nomic.beads": mock_module}):
            with patch("aragora.cli.gt._run_async", return_value=True):
                result = cmd_bead_assign(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "assigned" in captured.out

    def test_assign_failure(self, capsys):
        """Test failed bead assignment."""
        args = argparse.Namespace(bead_id="bead-123", agent_id="agent-456")

        mock_module = MagicMock()

        with patch.dict("sys.modules", {"aragora.nomic.beads": mock_module}):
            with patch("aragora.cli.gt._run_async", return_value=False):
                result = cmd_bead_assign(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Failed to assign" in captured.out


# ===========================================================================
# Tests: Agent Commands
# ===========================================================================


class TestCmdAgentList:
    """Tests for cmd_agent_list function."""

    def test_empty_list(self, capsys):
        """Test with no agents."""
        args = argparse.Namespace(role=None)

        mock_module = MagicMock()
        mock_module.AgentRole = MockRole

        with patch.dict("sys.modules", {"aragora.nomic.agent_roles": mock_module}):
            with patch("aragora.cli.gt._run_async", return_value=[]):
                result = cmd_agent_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No agents found" in captured.out

    def test_with_agents(self, capsys):
        """Test with agents in list."""
        args = argparse.Namespace(role=None)

        mock_agent = MockAgent()
        mock_module = MagicMock()
        mock_module.AgentRole = MockRole

        with patch.dict("sys.modules", {"aragora.nomic.agent_roles": mock_module}):
            with patch("aragora.cli.gt._run_async", return_value=[mock_agent]):
                result = cmd_agent_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Found 1 agent" in captured.out


class TestCmdAgentPromote:
    """Tests for cmd_agent_promote function."""

    def test_promote_success(self, capsys):
        """Test successful agent promotion."""
        args = argparse.Namespace(agent_id="agent-123", role="mayor")

        mock_module = MagicMock()
        mock_module.AgentRole = MockRole

        with patch.dict("sys.modules", {"aragora.nomic.agent_roles": mock_module}):
            with patch("aragora.cli.gt._run_async", return_value=True):
                result = cmd_agent_promote(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "promoted" in captured.out
        assert "MAYOR" in captured.out

    def test_promote_invalid_role(self, capsys):
        """Test promotion with invalid role."""
        args = argparse.Namespace(agent_id="agent-123", role="invalid")

        mock_module = MagicMock()
        # Make AgentRole raise ValueError for invalid roles
        mock_role_enum = MagicMock()
        mock_role_enum.side_effect = ValueError("Invalid role")
        mock_module.AgentRole = mock_role_enum

        with patch.dict("sys.modules", {"aragora.nomic.agent_roles": mock_module}):
            result = cmd_agent_promote(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Invalid role" in captured.out


# ===========================================================================
# Tests: Witness Commands
# ===========================================================================


class TestCmdWitnessStatus:
    """Tests for cmd_witness_status function."""

    def test_not_initialized(self, capsys):
        """Test when witness not initialized."""
        args = argparse.Namespace()

        mock_startup = MagicMock()
        mock_startup.get_witness_behavior.return_value = None

        with patch.dict("sys.modules", {"aragora.server.startup": mock_startup}):
            result = cmd_witness_status(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "not initialized" in captured.out

    def test_status_success(self, capsys):
        """Test successful status check."""
        args = argparse.Namespace()

        mock_witness = MagicMock()
        mock_witness._running = True
        mock_witness.config.patrol_interval_seconds = 30
        mock_witness.config.heartbeat_timeout_seconds = 60

        mock_startup = MagicMock()
        mock_startup.get_witness_behavior.return_value = mock_witness

        with patch.dict("sys.modules", {"aragora.server.startup": mock_startup}):
            result = cmd_witness_status(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Witness Patrol Status" in captured.out
        assert "Patrolling: Yes" in captured.out


# ===========================================================================
# Tests: Workspace Commands
# ===========================================================================


class TestCmdWorkspaceInit:
    """Tests for cmd_workspace_init function."""

    def test_init_default_directory(self, tmp_path, capsys):
        """Test initializing in default directory."""
        args = argparse.Namespace(directory=str(tmp_path), force=False)

        result = cmd_workspace_init(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "initialized" in captured.out

        # Check directories created
        assert (tmp_path / ".gt").exists()
        assert (tmp_path / ".gt" / "convoys").exists()
        assert (tmp_path / ".gt" / "beads").exists()
        assert (tmp_path / ".gt" / "agents").exists()
        assert (tmp_path / ".gt" / "hooks").exists()

        # Check config file
        config_file = tmp_path / ".gt" / "config.json"
        assert config_file.exists()
        config = json.loads(config_file.read_text())
        assert config["version"] == "1.0"

    def test_init_force_overwrite(self, tmp_path, capsys):
        """Test force overwriting existing config."""
        args = argparse.Namespace(directory=str(tmp_path), force=True)

        # Create existing config
        gt_dir = tmp_path / ".gt"
        gt_dir.mkdir()
        config_file = gt_dir / "config.json"
        config_file.write_text('{"version": "old"}')

        result = cmd_workspace_init(args)

        assert result == 0
        config = json.loads(config_file.read_text())
        assert config["version"] == "1.0"


# ===========================================================================
# Tests: Main Command Handler
# ===========================================================================


class TestCmdGt:
    """Tests for cmd_gt function."""

    def test_no_subcommand(self, capsys):
        """Test with no subcommand shows help."""
        args = argparse.Namespace()
        # No func attribute

        result = cmd_gt(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Gas Town CLI" in captured.out
        assert "convoy" in captured.out
        assert "bead" in captured.out
        assert "agent" in captured.out

    def test_with_func(self):
        """Test with func attribute calls function."""
        mock_func = MagicMock(return_value=0)
        args = argparse.Namespace(func=mock_func)

        result = cmd_gt(args)

        assert result == 0
        mock_func.assert_called_once_with(args)


class TestAddGtSubparsers:
    """Tests for add_gt_subparsers function."""

    def test_adds_all_commands(self):
        """Test all GT commands are added."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        add_gt_subparsers(subparsers)

        # Test we can parse gt commands
        args = parser.parse_args(["gt", "convoy", "list"])
        assert args.gt_command == "convoy"
        assert args.convoy_action == "list"

    def test_convoy_create_args(self):
        """Test convoy create arguments."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_gt_subparsers(subparsers)

        args = parser.parse_args(
            ["gt", "convoy", "create", "Test", "--beads", "a,b,c", "--priority", "high"]
        )
        assert args.title == "Test"
        assert args.beads == "a,b,c"
        assert args.priority == "high"

    def test_bead_list_args(self):
        """Test bead list arguments."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_gt_subparsers(subparsers)

        args = parser.parse_args(["gt", "bead", "list", "--status", "pending", "--limit", "10"])
        assert args.status == "pending"
        assert args.limit == 10

    def test_agent_promote_args(self):
        """Test agent promote arguments."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_gt_subparsers(subparsers)

        args = parser.parse_args(["gt", "agent", "promote", "agent-123", "mayor"])
        assert args.agent_id == "agent-123"
        assert args.role == "mayor"

    def test_workspace_init_args(self):
        """Test workspace init arguments."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_gt_subparsers(subparsers)

        args = parser.parse_args(["gt", "workspace", "init", "/path/to/dir", "--force"])
        assert args.directory == "/path/to/dir"
        assert args.force is True
