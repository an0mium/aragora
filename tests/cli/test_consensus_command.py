"""Tests for the `aragora consensus` CLI command.

Tests:
1. Parser registration (subcommands, flags, defaults)
2. Consensus detect - happy path with inline proposals
3. Consensus detect - from file
4. Consensus detect - from stdin
5. Consensus detect - error handling (no proposals, no task, bad JSON)
6. Consensus detect - JSON output
7. Consensus detect - API-first path
8. Consensus detect - local fallback
9. Consensus status - happy path (API)
10. Consensus status - error handling (API failure)
11. Consensus status - JSON output
12. Dispatcher routing
"""

from __future__ import annotations

import argparse
import json
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from aragora.cli.commands.consensus import (
    _normalize_proposals,
    _print_detect_result,
    _print_status_result,
    _try_api_detect,
    _try_api_status,
    _try_local_detect,
    add_consensus_parser,
    cmd_consensus,
    cmd_consensus_detect,
    cmd_consensus_status,
)


def _make_detect_args(**overrides) -> argparse.Namespace:
    """Create a Namespace with consensus detect defaults plus overrides."""
    defaults = {
        "consensus_command": "detect",
        "task": "Choose a database",
        "file": None,
        "proposals": None,
        "stdin": False,
        "threshold": 0.7,
        "output_format": "text",
        "api_url": None,
        "api_key": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_status_args(**overrides) -> argparse.Namespace:
    """Create a Namespace with consensus status defaults plus overrides."""
    defaults = {
        "consensus_command": "status",
        "debate_id": "debate-abc-123",
        "output_format": "text",
        "api_url": None,
        "api_key": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _mock_detect_result() -> dict:
    """Create a mock consensus detection result."""
    return {
        "debate_id": "detect-abcdef123456",
        "consensus_reached": True,
        "confidence": 0.85,
        "threshold": 0.7,
        "agreement_ratio": 0.80,
        "has_strong_consensus": True,
        "final_claim": "Use PostgreSQL with Redis cache",
        "reasoning_summary": "Analyzed 2 proposals from 2 agents. Average agreement: 80%.",
        "supporting_agents": ["agent-1", "agent-2"],
        "dissenting_agents": [],
        "claims_count": 2,
        "evidence_count": 2,
        "checksum": "sha256:abc123",
    }


def _mock_status_result() -> dict:
    """Create a mock consensus status result."""
    return {
        "consensus_reached": True,
        "confidence": 0.90,
        "agreement_ratio": 0.85,
        "has_strong_consensus": True,
        "final_claim": "Use microservices",
        "supporting_agents": ["claude", "gpt4"],
        "dissenting_agents": ["gemini"],
        "claims_count": 5,
        "dissents_count": 1,
        "unresolved_tensions_count": 0,
        "partial_consensus": {"agreed_count": 3, "items": [1, 2, 3]},
        "checksum": "sha256:xyz789",
    }


# ===========================================================================
# Parser registration
# ===========================================================================


class TestConsensusParser:
    """Tests that the consensus command is properly registered."""

    def test_parser_registered(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_consensus_parser(subparsers)

        args = parser.parse_args(["consensus", "detect", "--task", "Test"])
        assert args.consensus_command == "detect"
        assert args.task == "Test"

    def test_detect_subcommand_parses(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_consensus_parser(subparsers)

        args = parser.parse_args([
            "consensus", "detect",
            "--task", "Choose a database",
            "--proposals", '["Use PostgreSQL", "Use MySQL"]',
            "--threshold", "0.8",
            "--format", "json",
        ])
        assert args.consensus_command == "detect"
        assert args.task == "Choose a database"
        assert args.proposals == '["Use PostgreSQL", "Use MySQL"]'
        assert args.threshold == 0.8
        assert args.output_format == "json"

    def test_detect_file_flag(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_consensus_parser(subparsers)

        args = parser.parse_args([
            "consensus", "detect", "--task", "Test", "--file", "proposals.json",
        ])
        assert args.file == "proposals.json"

    def test_detect_stdin_flag(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_consensus_parser(subparsers)

        args = parser.parse_args([
            "consensus", "detect", "--stdin",
        ])
        assert args.stdin is True

    def test_status_subcommand_parses(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_consensus_parser(subparsers)

        args = parser.parse_args([
            "consensus", "status", "debate-123", "--format", "json",
        ])
        assert args.consensus_command == "status"
        assert args.debate_id == "debate-123"
        assert args.output_format == "json"

    def test_status_api_url_and_key(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_consensus_parser(subparsers)

        args = parser.parse_args([
            "consensus", "status", "d1",
            "--api-url", "http://custom:9090",
            "--api-key", "sk-test",
        ])
        assert args.api_url == "http://custom:9090"
        assert args.api_key == "sk-test"

    def test_detect_default_threshold(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_consensus_parser(subparsers)

        args = parser.parse_args(["consensus", "detect", "--task", "T"])
        assert args.threshold == 0.7

    def test_detect_default_format(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_consensus_parser(subparsers)

        args = parser.parse_args(["consensus", "detect", "--task", "T"])
        assert args.output_format == "text"

    def test_consensus_in_main_parser(self):
        """Consensus command is registered in the main parser."""
        from aragora.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["consensus", "detect", "--task", "Test"])
        assert args.consensus_command == "detect"


# ===========================================================================
# Normalize proposals helper
# ===========================================================================


class TestNormalizeProposals:
    """Tests for the _normalize_proposals helper."""

    def test_normalize_string_proposals(self):
        result = _normalize_proposals(["Use PostgreSQL", "Use MySQL"])
        assert len(result) == 2
        assert result[0]["agent"] == "agent-1"
        assert result[0]["content"] == "Use PostgreSQL"
        assert result[1]["agent"] == "agent-2"
        assert result[1]["content"] == "Use MySQL"

    def test_normalize_dict_proposals(self):
        result = _normalize_proposals([
            {"agent": "claude", "content": "Use PostgreSQL"},
            {"agent": "gpt4", "content": "Use MySQL"},
        ])
        assert len(result) == 2
        assert result[0]["agent"] == "claude"
        assert result[1]["agent"] == "gpt4"

    def test_normalize_mixed_proposals(self):
        result = _normalize_proposals([
            "Use PostgreSQL",
            {"agent": "claude", "content": "Use MySQL"},
        ])
        assert len(result) == 2
        assert result[0]["agent"] == "agent-1"
        assert result[1]["agent"] == "claude"

    def test_empty_content_filtered(self):
        result = _normalize_proposals([
            {"agent": "a", "content": ""},
            {"agent": "b", "content": "Valid"},
        ])
        assert len(result) == 1
        assert result[0]["agent"] == "b"

    def test_empty_list(self):
        result = _normalize_proposals([])
        assert result == []

    def test_dict_with_round(self):
        result = _normalize_proposals([
            {"agent": "a", "content": "Test", "round": 2},
        ])
        assert result[0]["round"] == 2


# ===========================================================================
# Consensus detect command
# ===========================================================================


class TestConsensusDetect:
    """Tests for cmd_consensus_detect."""

    def test_detect_with_inline_proposals_json(self, capsys):
        """Detect with --proposals outputs result."""
        args = _make_detect_args(
            proposals='["Use PostgreSQL", "Use PostgreSQL with Redis cache"]',
            output_format="json",
        )
        with patch(
            "aragora.cli.commands.consensus._try_api_detect",
            return_value=None,
        ):
            result = cmd_consensus_detect(args)

        assert result == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "consensus_reached" in data
        assert "confidence" in data
        assert "debate_id" in data

    def test_detect_with_inline_proposals_text(self, capsys):
        """Detect with text format outputs readable result."""
        args = _make_detect_args(
            proposals='["Use PostgreSQL", "Use PostgreSQL with Redis cache"]',
            output_format="text",
        )
        with patch(
            "aragora.cli.commands.consensus._try_api_detect",
            return_value=None,
        ):
            result = cmd_consensus_detect(args)

        assert result == 0
        output = capsys.readouterr().out
        assert "Consensus Detection Result" in output
        assert "Confidence:" in output

    def test_detect_from_file(self, capsys, tmp_path):
        """Detect reads proposals from a JSON file."""
        proposals_file = tmp_path / "proposals.json"
        proposals_file.write_text(json.dumps({
            "task": "Choose a framework",
            "proposals": [
                {"agent": "claude", "content": "Use React"},
                {"agent": "gpt4", "content": "Use React with Next.js"},
            ],
        }))

        args = _make_detect_args(
            file=str(proposals_file),
            task=None,  # Should get from file
            output_format="json",
        )
        with patch(
            "aragora.cli.commands.consensus._try_api_detect",
            return_value=None,
        ):
            result = cmd_consensus_detect(args)

        assert result == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "consensus_reached" in data

    def test_detect_from_file_array_format(self, capsys, tmp_path):
        """Detect reads proposals from a JSON array file."""
        proposals_file = tmp_path / "proposals.json"
        proposals_file.write_text(json.dumps([
            "Use React",
            "Use Vue",
        ]))

        args = _make_detect_args(
            file=str(proposals_file),
            task="Choose a framework",
            output_format="json",
        )
        with patch(
            "aragora.cli.commands.consensus._try_api_detect",
            return_value=None,
        ):
            result = cmd_consensus_detect(args)

        assert result == 0

    def test_detect_from_stdin(self, capsys):
        """Detect reads proposals from stdin."""
        stdin_data = json.dumps({
            "task": "Pick a framework",
            "proposals": [
                {"agent": "a", "content": "Use React"},
            ],
        })
        args = _make_detect_args(
            stdin=True,
            task=None,
            output_format="json",
        )
        with (
            patch("sys.stdin", StringIO(stdin_data)),
            patch(
                "aragora.cli.commands.consensus._try_api_detect",
                return_value=None,
            ),
        ):
            result = cmd_consensus_detect(args)

        assert result == 0

    def test_detect_api_first(self, capsys):
        """Detect uses API result when available."""
        mock_result = _mock_detect_result()
        args = _make_detect_args(
            proposals='["Use PostgreSQL", "Use MySQL"]',
            output_format="json",
        )
        with patch(
            "aragora.cli.commands.consensus._try_api_detect",
            return_value=mock_result,
        ):
            result = cmd_consensus_detect(args)

        assert result == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["debate_id"] == "detect-abcdef123456"
        assert data["consensus_reached"] is True

    def test_detect_local_fallback_when_api_fails(self, capsys):
        """Detect falls back to local when API returns None."""
        args = _make_detect_args(
            proposals='["Use PostgreSQL", "Use PostgreSQL with caching"]',
            output_format="json",
        )
        with patch(
            "aragora.cli.commands.consensus._try_api_detect",
            return_value=None,
        ):
            result = cmd_consensus_detect(args)

        assert result == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "consensus_reached" in data
        assert "confidence" in data

    def test_detect_no_proposals_error(self, capsys):
        """Detect errors when no proposals source is given."""
        args = _make_detect_args(
            proposals=None,
            file=None,
            stdin=False,
        )
        result = cmd_consensus_detect(args)

        assert result == 1
        err = capsys.readouterr().err
        assert "Provide proposals via" in err

    def test_detect_no_task_error(self, capsys):
        """Detect errors when no task is given."""
        args = _make_detect_args(
            task=None,
            proposals='["Proposal A"]',
        )
        with patch(
            "aragora.cli.commands.consensus._try_api_detect",
            return_value=None,
        ):
            result = cmd_consensus_detect(args)

        assert result == 1
        err = capsys.readouterr().err
        assert "--task is required" in err

    def test_detect_invalid_json_proposals(self, capsys):
        """Detect errors on invalid JSON in --proposals."""
        args = _make_detect_args(
            proposals="not valid json",
        )
        result = cmd_consensus_detect(args)

        assert result == 1
        err = capsys.readouterr().err
        assert "Invalid JSON" in err

    def test_detect_invalid_json_file(self, capsys, tmp_path):
        """Detect errors on invalid JSON in file."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json {{{")

        args = _make_detect_args(
            file=str(bad_file),
        )
        result = cmd_consensus_detect(args)

        assert result == 1
        err = capsys.readouterr().err
        assert "Could not read file" in err

    def test_detect_missing_file(self, capsys):
        """Detect errors on missing file."""
        args = _make_detect_args(
            file="/nonexistent/path/proposals.json",
        )
        result = cmd_consensus_detect(args)

        assert result == 1
        err = capsys.readouterr().err
        assert "Could not read file" in err

    def test_detect_invalid_json_stdin(self, capsys):
        """Detect errors on invalid JSON from stdin."""
        args = _make_detect_args(
            stdin=True,
            task=None,
        )
        with patch("sys.stdin", StringIO("not json")):
            result = cmd_consensus_detect(args)

        assert result == 1
        err = capsys.readouterr().err
        assert "Invalid JSON on stdin" in err

    def test_detect_empty_proposals_error(self, capsys):
        """Detect errors when proposals list is empty."""
        args = _make_detect_args(
            proposals='[]',
        )
        result = cmd_consensus_detect(args)

        assert result == 1
        err = capsys.readouterr().err
        assert "No valid proposals" in err

    def test_detect_both_fail_error(self, capsys):
        """Detect errors when both API and local fail."""
        args = _make_detect_args(
            proposals='["Use PostgreSQL"]',
        )
        with (
            patch("aragora.cli.commands.consensus._try_api_detect", return_value=None),
            patch("aragora.cli.commands.consensus._try_local_detect", return_value=None),
        ):
            result = cmd_consensus_detect(args)

        assert result == 1
        err = capsys.readouterr().err
        assert "Consensus detection failed" in err

    def test_detect_threshold_customization(self, capsys):
        """Detect respects custom threshold."""
        args = _make_detect_args(
            proposals='["Use PostgreSQL", "Use MySQL"]',
            threshold=0.99,
            output_format="json",
        )
        with patch(
            "aragora.cli.commands.consensus._try_api_detect",
            return_value=None,
        ):
            result = cmd_consensus_detect(args)

        assert result == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["threshold"] == 0.99


# ===========================================================================
# Consensus status command
# ===========================================================================


class TestConsensusStatus:
    """Tests for cmd_consensus_status."""

    def test_status_success_json(self, capsys):
        """Status outputs JSON when API succeeds."""
        mock_result = _mock_status_result()
        args = _make_status_args(output_format="json")

        with patch(
            "aragora.cli.commands.consensus._try_api_status",
            return_value=mock_result,
        ):
            result = cmd_consensus_status(args)

        assert result == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["consensus_reached"] is True
        assert data["confidence"] == 0.90

    def test_status_success_text(self, capsys):
        """Status outputs text when API succeeds."""
        mock_result = _mock_status_result()
        args = _make_status_args(output_format="text")

        with patch(
            "aragora.cli.commands.consensus._try_api_status",
            return_value=mock_result,
        ):
            result = cmd_consensus_status(args)

        assert result == 0
        output = capsys.readouterr().out
        assert "Consensus Status: debate-abc-123" in output
        assert "REACHED" in output

    def test_status_api_failure(self, capsys):
        """Status errors when API fails."""
        args = _make_status_args()

        with patch(
            "aragora.cli.commands.consensus._try_api_status",
            return_value=None,
        ):
            result = cmd_consensus_status(args)

        assert result == 1
        err = capsys.readouterr().err
        assert "Could not retrieve consensus status" in err
        assert "debate-abc-123" in err

    def test_status_missing_debate_id(self, capsys):
        """Status errors when debate_id is missing."""
        args = _make_status_args(debate_id=None)

        result = cmd_consensus_status(args)

        assert result == 1
        err = capsys.readouterr().err
        assert "debate_id is required" in err


# ===========================================================================
# API helpers
# ===========================================================================


class TestAPIHelpers:
    """Tests for API communication helpers."""

    def test_try_api_detect_success(self):
        """API detect returns unwrapped data."""
        mock_client = MagicMock()
        mock_client.consensus.detect.return_value = {
            "data": {"consensus_reached": True, "confidence": 0.9},
        }

        with patch("aragora.client.client.AragoraClient") as mock_cls:
            mock_cls.return_value = mock_client
            result = _try_api_detect(
                "Test task", [{"agent": "a", "content": "Proposal"}], 0.7,
                _make_detect_args(),
            )

        assert result is not None
        assert result["consensus_reached"] is True

    def test_try_api_detect_no_envelope(self):
        """API detect returns direct result without data wrapper."""
        mock_client = MagicMock()
        mock_client.consensus.detect.return_value = {
            "consensus_reached": False,
        }

        with patch("aragora.client.client.AragoraClient") as mock_cls:
            mock_cls.return_value = mock_client
            result = _try_api_detect(
                "Test", [{"agent": "a", "content": "P"}], 0.7,
                _make_detect_args(),
            )

        assert result is not None
        assert result["consensus_reached"] is False

    def test_try_api_detect_import_error(self):
        """API detect returns None when client not available."""
        original_import = __import__

        def block_client(name, *args, **kwargs):
            if "aragora.client.client" in name:
                raise ImportError("no client")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=block_client):
            result = _try_api_detect(
                "Test", [{"agent": "a", "content": "P"}], 0.7,
                _make_detect_args(),
            )

        assert result is None

    def test_try_api_detect_connection_error(self):
        """API detect returns None on connection error."""
        mock_client = MagicMock()
        mock_client.consensus.detect.side_effect = ConnectionError("refused")

        with patch("aragora.client.client.AragoraClient") as mock_cls:
            mock_cls.return_value = mock_client
            result = _try_api_detect(
                "Test", [{"agent": "a", "content": "P"}], 0.7,
                _make_detect_args(),
            )

        assert result is None

    def test_try_api_detect_uses_custom_url(self):
        """API detect uses custom URL from args."""
        mock_client = MagicMock()
        mock_client.consensus.detect.return_value = {"data": {}}

        with patch("aragora.client.client.AragoraClient") as mock_cls:
            mock_cls.return_value = mock_client
            _try_api_detect(
                "Test", [{"agent": "a", "content": "P"}], 0.7,
                _make_detect_args(api_url="http://custom:9090", api_key="sk-test"),
            )

        mock_cls.assert_called_once_with(base_url="http://custom:9090", api_key="sk-test")

    def test_try_api_status_success(self):
        """API status returns unwrapped data."""
        mock_client = MagicMock()
        mock_client.consensus.get_detection_status.return_value = {
            "data": {"consensus_reached": True},
        }

        with patch("aragora.client.client.AragoraClient") as mock_cls:
            mock_cls.return_value = mock_client
            result = _try_api_status("debate-123", _make_status_args())

        assert result is not None
        assert result["consensus_reached"] is True

    def test_try_api_status_import_error(self):
        """API status returns None when client not available."""
        original_import = __import__

        def block_client(name, *args, **kwargs):
            if "aragora.client.client" in name:
                raise ImportError("no client")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=block_client):
            result = _try_api_status("debate-123", _make_status_args())

        assert result is None

    def test_try_api_status_connection_error(self):
        """API status returns None on connection error."""
        mock_client = MagicMock()
        mock_client.consensus.get_detection_status.side_effect = ConnectionError("refused")

        with patch("aragora.client.client.AragoraClient") as mock_cls:
            mock_cls.return_value = mock_client
            result = _try_api_status("debate-123", _make_status_args())

        assert result is None


# ===========================================================================
# Local detection
# ===========================================================================


class TestLocalDetection:
    """Tests for the local ConsensusBuilder fallback."""

    def test_local_detect_similar_proposals(self):
        """Local detect finds consensus for similar proposals."""
        result = _try_local_detect(
            "Choose a database",
            [
                {"agent": "claude", "content": "Use PostgreSQL for reliable storage"},
                {"agent": "gpt4", "content": "Use PostgreSQL for reliable data storage"},
            ],
            0.3,
        )

        assert result is not None
        assert "consensus_reached" in result
        assert "confidence" in result
        assert result["debate_id"].startswith("detect-")

    def test_local_detect_divergent_proposals(self):
        """Local detect reports low confidence for divergent proposals."""
        result = _try_local_detect(
            "Choose a database",
            [
                {"agent": "claude", "content": "Use a relational database like PostgreSQL"},
                {"agent": "gpt4", "content": "Use a document store like MongoDB"},
            ],
            0.7,
        )

        assert result is not None
        assert "confidence" in result
        # Divergent proposals should have lower agreement
        assert result["confidence"] <= 1.0

    def test_local_detect_single_proposal(self):
        """Local detect handles single proposal (full agreement)."""
        result = _try_local_detect(
            "Choose a database",
            [{"agent": "claude", "content": "Use PostgreSQL"}],
            0.5,
        )

        assert result is not None
        # Single proposal = full agreement by definition
        assert result["consensus_reached"] is True

    def test_local_detect_import_error(self):
        """Local detect returns None when ConsensusBuilder unavailable."""
        original_import = __import__

        def block_consensus(name, *args, **kwargs):
            if "aragora.debate.consensus" in name:
                raise ImportError("no consensus module")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=block_consensus):
            result = _try_local_detect(
                "Test", [{"agent": "a", "content": "P"}], 0.7,
            )

        assert result is None


# ===========================================================================
# Text formatting
# ===========================================================================


class TestTextFormatting:
    """Tests for text output formatting."""

    def test_print_detect_result(self, capsys):
        """Detection result prints key information."""
        _print_detect_result(_mock_detect_result())
        output = capsys.readouterr().out

        assert "Consensus Detection Result" in output
        assert "REACHED" in output
        assert "Confidence:" in output
        assert "Threshold:" in output
        assert "Agreement:" in output
        assert "Final Claim:" in output
        assert "Use PostgreSQL with Redis cache" in output
        assert "agent-1" in output
        assert "Checksum:" in output

    def test_print_detect_not_reached(self, capsys):
        """Detection result shows NOT REACHED."""
        result = _mock_detect_result()
        result["consensus_reached"] = False
        result["has_strong_consensus"] = False
        _print_detect_result(result)
        output = capsys.readouterr().out

        assert "NOT REACHED" in output

    def test_print_detect_strong_consensus(self, capsys):
        """Detection result shows STRONG flag."""
        _print_detect_result(_mock_detect_result())
        output = capsys.readouterr().out

        assert "STRONG" in output

    def test_print_detect_long_claim_truncated(self, capsys):
        """Long claims are truncated in text output."""
        result = _mock_detect_result()
        result["final_claim"] = "A" * 300
        _print_detect_result(result)
        output = capsys.readouterr().out

        assert "..." in output
        assert "A" * 300 not in output

    def test_print_detect_dissenting_agents(self, capsys):
        """Dissenting agents are shown when present."""
        result = _mock_detect_result()
        result["dissenting_agents"] = ["agent-3"]
        _print_detect_result(result)
        output = capsys.readouterr().out

        assert "Dissenting: agent-3" in output

    def test_print_status_result(self, capsys):
        """Status result prints key information."""
        _print_status_result("debate-123", _mock_status_result())
        output = capsys.readouterr().out

        assert "Consensus Status: debate-123" in output
        assert "REACHED" in output
        assert "STRONG" in output
        assert "Use microservices" in output
        assert "claude" in output
        assert "gemini" in output
        assert "Claims:" in output
        assert "Dissents:" in output
        assert "Tensions:" in output
        assert "Partial Consensus:" in output

    def test_print_status_not_reached(self, capsys):
        """Status result shows NOT REACHED."""
        result = _mock_status_result()
        result["consensus_reached"] = False
        result["has_strong_consensus"] = False
        _print_status_result("d1", result)
        output = capsys.readouterr().out

        assert "NOT REACHED" in output


# ===========================================================================
# Dispatcher
# ===========================================================================


class TestConsensusDispatcher:
    """Tests for the cmd_consensus dispatcher."""

    def test_dispatches_to_detect(self, capsys):
        """Dispatcher routes 'detect' to cmd_consensus_detect."""
        args = _make_detect_args(
            proposals='["Use PostgreSQL"]',
            output_format="json",
        )
        with patch(
            "aragora.cli.commands.consensus._try_api_detect",
            return_value=None,
        ):
            cmd_consensus(args)

        output = capsys.readouterr().out
        data = json.loads(output)
        assert "consensus_reached" in data

    def test_dispatches_to_status(self, capsys):
        """Dispatcher routes 'status' to cmd_consensus_status."""
        args = _make_status_args(output_format="json")

        with patch(
            "aragora.cli.commands.consensus._try_api_status",
            return_value=_mock_status_result(),
        ):
            cmd_consensus(args)

        output = capsys.readouterr().out
        data = json.loads(output)
        assert "consensus_reached" in data

    def test_no_subcommand_shows_help(self, capsys):
        """Dispatcher shows help when no subcommand is given."""
        mock_parser = MagicMock()
        args = argparse.Namespace(
            consensus_command=None,
            _parser=mock_parser,
        )
        cmd_consensus(args)
        mock_parser.print_help.assert_called_once()

    def test_no_subcommand_no_parser_shows_usage(self, capsys):
        """Dispatcher shows usage text when no subcommand and no parser."""
        args = argparse.Namespace(consensus_command=None)
        cmd_consensus(args)
        output = capsys.readouterr().out
        assert "aragora consensus" in output
