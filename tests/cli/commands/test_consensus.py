"""Tests for the 'aragora consensus' CLI command.

Validates:
- consensus detect (with --proposals, --file, --stdin)
- consensus status
- Parser registration
"""

import argparse
import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from aragora.cli.commands.consensus import (
    add_consensus_parser,
    cmd_consensus,
    cmd_consensus_detect,
    cmd_consensus_status,
    _normalize_proposals,
)


@pytest.fixture
def parser():
    """Create an argparse parser with consensus subcommand registered."""
    p = argparse.ArgumentParser()
    subparsers = p.add_subparsers(dest="command")
    add_consensus_parser(subparsers)
    return p


class TestConsensusParserRegistration:
    """Test parser registration and help text."""

    def test_parser_registers(self, parser):
        args = parser.parse_args(["consensus"])
        assert args.command == "consensus"

    def test_detect_subcommand(self, parser):
        args = parser.parse_args([
            "consensus", "detect",
            "--task", "Choose a DB",
            "--proposals", '["Use PostgreSQL"]',
        ])
        assert args.consensus_command == "detect"
        assert args.task == "Choose a DB"

    def test_status_subcommand(self, parser):
        args = parser.parse_args(["consensus", "status", "debate-123"])
        assert args.consensus_command == "status"
        assert args.debate_id == "debate-123"


class TestNormalizeProposals:
    """Test _normalize_proposals helper."""

    def test_string_proposals(self):
        result = _normalize_proposals(["Use Postgres", "Use MySQL"])
        assert len(result) == 2
        assert result[0]["agent"] == "agent-1"
        assert result[0]["content"] == "Use Postgres"

    def test_dict_proposals(self):
        result = _normalize_proposals([
            {"agent": "claude", "content": "Use Postgres"},
            {"agent": "gpt", "content": "Use MySQL"},
        ])
        assert len(result) == 2
        assert result[0]["agent"] == "claude"

    def test_empty_content_filtered(self):
        result = _normalize_proposals([
            {"agent": "a", "content": ""},
            {"agent": "b", "content": "Use Postgres"},
        ])
        assert len(result) == 1

    def test_mixed_types(self):
        result = _normalize_proposals([
            "Plain string proposal",
            {"agent": "claude", "content": "Dict proposal"},
        ])
        assert len(result) == 2


class TestConsensusDetect:
    """Test cmd_consensus_detect."""

    @patch("aragora.cli.commands.consensus._try_api_detect", return_value=None)
    @patch("aragora.cli.commands.consensus._try_local_detect")
    def test_detect_with_inline_proposals(self, mock_local, mock_api, capsys):
        """Test detect with inline --proposals."""
        mock_local.return_value = {
            "debate_id": "detect-abc",
            "consensus_reached": True,
            "confidence": 0.85,
            "threshold": 0.7,
            "agreement_ratio": 0.9,
            "has_strong_consensus": True,
            "final_claim": "Use PostgreSQL",
            "reasoning_summary": "High agreement",
            "supporting_agents": ["a1", "a2"],
            "dissenting_agents": [],
            "claims_count": 2,
            "evidence_count": 2,
            "checksum": "abc123",
        }

        args = argparse.Namespace(
            task="Choose a DB",
            proposals='["Use PostgreSQL", "PostgreSQL is best"]',
            file=None,
            stdin=False,
            threshold=0.7,
            output_format="text",
            api_url=None,
            api_key=None,
        )
        result = cmd_consensus_detect(args)
        assert result == 0

        captured = capsys.readouterr()
        assert "REACHED" in captured.out
        assert "PostgreSQL" in captured.out

    @patch("aragora.cli.commands.consensus._try_api_detect", return_value=None)
    @patch("aragora.cli.commands.consensus._try_local_detect")
    def test_detect_json_output(self, mock_local, mock_api, capsys):
        """Test detect with JSON output format."""
        mock_local.return_value = {
            "debate_id": "detect-abc",
            "consensus_reached": True,
            "confidence": 0.85,
        }

        args = argparse.Namespace(
            task="Choose a DB",
            proposals='["Use PostgreSQL"]',
            file=None,
            stdin=False,
            threshold=0.7,
            output_format="json",
            api_url=None,
            api_key=None,
        )
        result = cmd_consensus_detect(args)
        assert result == 0

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["consensus_reached"] is True

    def test_detect_missing_task(self, capsys):
        """Test detect without task returns error."""
        args = argparse.Namespace(
            task=None,
            proposals='["Use PostgreSQL"]',
            file=None,
            stdin=False,
            threshold=0.7,
            output_format="text",
            api_url=None,
            api_key=None,
        )
        result = cmd_consensus_detect(args)
        assert result == 1

    def test_detect_no_proposals(self, capsys):
        """Test detect without any proposal source returns error."""
        args = argparse.Namespace(
            task="Choose a DB",
            proposals=None,
            file=None,
            stdin=False,
            threshold=0.7,
            output_format="text",
            api_url=None,
            api_key=None,
        )
        result = cmd_consensus_detect(args)
        assert result == 1


class TestConsensusStatus:
    """Test cmd_consensus_status."""

    @patch("aragora.cli.commands.consensus._try_api_status")
    def test_status_success(self, mock_api, capsys):
        """Test status with successful API response."""
        mock_api.return_value = {
            "debate_id": "debate-123",
            "consensus_reached": True,
            "confidence": 0.9,
            "agreement_ratio": 0.85,
            "has_strong_consensus": True,
            "final_claim": "Use PostgreSQL",
            "supporting_agents": ["claude", "gpt-4"],
            "dissenting_agents": [],
            "claims_count": 3,
            "dissents_count": 0,
            "unresolved_tensions_count": 0,
            "partial_consensus": {"items": []},
            "checksum": "abc123",
        }

        args = argparse.Namespace(
            debate_id="debate-123",
            output_format="text",
            api_url=None,
            api_key=None,
        )
        result = cmd_consensus_status(args)
        assert result == 0

        captured = capsys.readouterr()
        assert "REACHED" in captured.out
        assert "debate-123" in captured.out

    @patch("aragora.cli.commands.consensus._try_api_status", return_value=None)
    def test_status_not_found(self, mock_api, capsys):
        """Test status when debate not found."""
        args = argparse.Namespace(
            debate_id="nonexistent",
            output_format="text",
            api_url=None,
            api_key=None,
        )
        result = cmd_consensus_status(args)
        assert result == 1

    def test_status_missing_debate_id(self, capsys):
        """Test status without debate_id."""
        args = argparse.Namespace(
            debate_id=None,
            output_format="text",
            api_url=None,
            api_key=None,
        )
        result = cmd_consensus_status(args)
        assert result == 1


class TestConsensusMainCommand:
    """Test the main consensus command routing."""

    def test_routes_to_detect(self):
        """Test that consensus_command=detect routes correctly."""
        with patch("aragora.cli.commands.consensus.cmd_consensus_detect") as mock_detect:
            args = argparse.Namespace(consensus_command="detect")
            cmd_consensus(args)
            mock_detect.assert_called_once()

    def test_routes_to_status(self):
        """Test that consensus_command=status routes correctly."""
        with patch("aragora.cli.commands.consensus.cmd_consensus_status") as mock_status:
            args = argparse.Namespace(consensus_command="status")
            cmd_consensus(args)
            mock_status.assert_called_once()

    def test_no_subcommand_shows_help(self, capsys):
        """Test that no subcommand shows help hint."""
        args = argparse.Namespace(consensus_command=None, _parser=None)
        cmd_consensus(args)

        captured = capsys.readouterr()
        assert "aragora consensus" in captured.out
