"""
Tests for OpenClaw CLI commands.

Tests cover:
- Init command (scaffold, dry-run, force, templates)
- Status command (online, offline, stats)
- Policy command (list, validate)
- Audit command (query with filters)
- Parser registration
"""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.cli.openclaw import (
    cmd_init,
    cmd_status,
    cmd_policy,
    cmd_audit,
    create_openclaw_parser,
    DOCKER_COMPOSE_BASIC,
    POLICY_STRICT,
    POLICY_PERMISSIVE,
    GATEWAY_ENV,
)


class TestCmdInit:
    """Tests for the init command."""

    def test_init_creates_files(self, tmp_path):
        """Should create deployment files in output directory."""
        args = argparse.Namespace(
            output_dir=str(tmp_path / "deploy"),
            template="basic",
            policy_preset="strict",
            dry_run=False,
            force=False,
        )
        result = cmd_init(args)
        assert result == 0
        assert (tmp_path / "deploy" / "docker-compose.yml").exists()
        assert (tmp_path / "deploy" / "policies" / "policy.yaml").exists()
        assert (tmp_path / "deploy" / "gateway.env").exists()

    def test_init_strict_policy(self, tmp_path):
        """Should use strict policy preset by default."""
        args = argparse.Namespace(
            output_dir=str(tmp_path / "deploy"),
            template="basic",
            policy_preset="strict",
            dry_run=False,
            force=False,
        )
        cmd_init(args)
        policy = (tmp_path / "deploy" / "policies" / "policy.yaml").read_text()
        assert "default_decision: deny" in policy
        assert "block_system_directories" in policy

    def test_init_permissive_policy(self, tmp_path):
        """Should use permissive policy when requested."""
        args = argparse.Namespace(
            output_dir=str(tmp_path / "deploy"),
            template="basic",
            policy_preset="permissive",
            dry_run=False,
            force=False,
        )
        cmd_init(args)
        policy = (tmp_path / "deploy" / "policies" / "policy.yaml").read_text()
        assert "default_decision: allow" in policy

    def test_init_dry_run(self, tmp_path, capsys):
        """Dry run should not create files."""
        args = argparse.Namespace(
            output_dir=str(tmp_path / "deploy"),
            template="basic",
            policy_preset="strict",
            dry_run=True,
            force=False,
        )
        result = cmd_init(args)
        assert result == 0
        assert not (tmp_path / "deploy").exists()
        captured = capsys.readouterr()
        assert "Would create" in captured.out

    def test_init_refuses_overwrite_without_force(self, tmp_path):
        """Should refuse to overwrite existing files without --force."""
        deploy = tmp_path / "deploy"
        deploy.mkdir()
        (deploy / "docker-compose.yml").write_text("existing")

        args = argparse.Namespace(
            output_dir=str(deploy),
            template="basic",
            policy_preset="strict",
            dry_run=False,
            force=False,
        )
        result = cmd_init(args)
        assert result == 1

    def test_init_force_overwrites(self, tmp_path):
        """Should overwrite existing files with --force."""
        deploy = tmp_path / "deploy"
        deploy.mkdir()
        (deploy / "docker-compose.yml").write_text("old")

        args = argparse.Namespace(
            output_dir=str(deploy),
            template="basic",
            policy_preset="strict",
            dry_run=False,
            force=True,
        )
        result = cmd_init(args)
        assert result == 0
        content = (deploy / "docker-compose.yml").read_text()
        assert "aragora-gateway" in content

    def test_init_docker_compose_content(self, tmp_path):
        """Should write valid docker-compose content."""
        args = argparse.Namespace(
            output_dir=str(tmp_path / "deploy"),
            template="basic",
            policy_preset="strict",
            dry_run=False,
            force=False,
        )
        cmd_init(args)
        content = (tmp_path / "deploy" / "docker-compose.yml").read_text()
        assert "aragora-gateway" in content
        assert "openclaw" in content
        assert "8080:8080" in content

    def test_init_gateway_env(self, tmp_path):
        """Should create gateway.env with configuration."""
        args = argparse.Namespace(
            output_dir=str(tmp_path / "deploy"),
            template="basic",
            policy_preset="strict",
            dry_run=False,
            force=False,
        )
        cmd_init(args)
        content = (tmp_path / "deploy" / "gateway.env").read_text()
        assert "ARAGORA_ENV=production" in content
        assert "OPENCLAW_URL" in content


class TestCmdStatus:
    """Tests for the status command."""

    @patch("httpx.get")
    def test_status_online(self, mock_get, capsys):
        """Should report gateway as ONLINE when healthy."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "active_sessions": 3,
            "actions_allowed": 100,
            "actions_denied": 5,
            "pending_approvals": 2,
            "policy_rules": 10,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        args = argparse.Namespace(server="http://localhost:8080")
        result = cmd_status(args)
        assert result == 0
        captured = capsys.readouterr()
        assert "ONLINE" in captured.out

    @patch("httpx.get")
    def test_status_offline(self, mock_get, capsys):
        """Should report gateway as OFFLINE when unreachable."""
        mock_get.side_effect = OSError("Connection refused")

        args = argparse.Namespace(server="http://localhost:8080")
        result = cmd_status(args)
        assert result == 1
        captured = capsys.readouterr()
        assert "OFFLINE" in captured.out


class TestCmdPolicy:
    """Tests for the policy command."""

    @patch("httpx.get")
    def test_policy_list(self, mock_get, capsys):
        """Should list policy rules from gateway."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "rules": [
                {
                    "name": "block_system",
                    "decision": "deny",
                    "priority": 100,
                    "action_types": ["file_read"],
                },
                {
                    "name": "allow_workspace",
                    "decision": "allow",
                    "priority": 10,
                    "action_types": ["file_write"],
                },
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        args = argparse.Namespace(
            policy_action="list",
            server="http://localhost:8080",
        )
        result = cmd_policy(args)
        assert result == 0
        captured = capsys.readouterr()
        assert "block_system" in captured.out
        assert "allow_workspace" in captured.out

    def test_policy_validate_missing_file(self):
        """Should fail when policy file doesn't exist."""
        args = argparse.Namespace(
            policy_action="validate",
            file="/nonexistent/policy.yaml",
            server="http://localhost:8080",
        )
        result = cmd_policy(args)
        assert result == 1


class TestCmdAudit:
    """Tests for the audit command."""

    @patch("httpx.get")
    def test_audit_query(self, mock_get, capsys):
        """Should query and display audit records."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "records": [
                {
                    "timestamp": 1700000000,
                    "event_type": "action_executed",
                    "user_id": "user-1",
                    "action_type": "shell",
                    "success": True,
                },
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        args = argparse.Namespace(
            server="http://localhost:8080",
            user_id="user-1",
            session_id=None,
            event_type=None,
            limit=50,
        )
        result = cmd_audit(args)
        assert result == 0
        captured = capsys.readouterr()
        assert "action_executed" in captured.out

    @patch("httpx.get")
    def test_audit_offline(self, mock_get):
        """Should return error when gateway is offline."""
        mock_get.side_effect = OSError("Connection refused")

        args = argparse.Namespace(
            server="http://localhost:8080",
            user_id=None,
            session_id=None,
            event_type=None,
            limit=50,
        )
        result = cmd_audit(args)
        assert result == 1


class TestCreateOpenclawParser:
    """Tests for parser creation."""

    def test_creates_subparser(self):
        """Should register the openclaw subcommand."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_openclaw_parser(subparsers)
        # Should not raise
        args = parser.parse_args(["openclaw", "init", "--dry-run"])
        assert args.dry_run is True

    def test_status_parser(self):
        """Should parse status command."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_openclaw_parser(subparsers)
        args = parser.parse_args(["openclaw", "status", "--server", "http://test:9090"])
        assert args.server == "http://test:9090"

    def test_audit_parser(self):
        """Should parse audit command with filters."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_openclaw_parser(subparsers)
        args = parser.parse_args(["openclaw", "audit", "--user-id", "u1", "--limit", "10"])
        assert args.user_id == "u1"
        assert args.limit == 10
