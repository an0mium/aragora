"""
Tests for CLI knowledge module.

Tests knowledge base CLI commands.
"""

from __future__ import annotations

import argparse
import json
from unittest.mock import MagicMock, patch

import pytest


class TestCreateKnowledgeParser:
    """Tests for create_knowledge_parser function."""

    def test_creates_subparser(self):
        """create_knowledge_parser creates knowledge subparser."""
        from aragora.cli.knowledge import create_knowledge_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_knowledge_parser(subparsers)

        # Should be able to parse 'knowledge' command
        args = parser.parse_args(["knowledge"])
        assert hasattr(args, "func")


class TestCmdFacts:
    """Tests for cmd_facts function."""

    def test_facts_show_requires_fact_id(self, capsys):
        """cmd_facts show requires fact_id."""
        from aragora.cli.knowledge import cmd_facts

        args = argparse.Namespace(
            action="show",
            fact_id=None,
            workspace="default",
            topic=None,
            status=None,
            min_confidence=0.0,
            limit=20,
            json=False,
        )

        # Need to mock the import that happens inside cmd_facts
        with patch.dict("sys.modules", {
            "aragora.knowledge": MagicMock(),
        }):
            result = cmd_facts(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "fact_id required" in captured.out.lower()

    def test_facts_verify_not_implemented(self, capsys):
        """cmd_facts verify is not implemented."""
        from aragora.cli.knowledge import cmd_facts

        args = argparse.Namespace(
            action="verify",
            fact_id="test-fact-id",
            workspace="default",
            topic=None,
            status=None,
            min_confidence=0.0,
            limit=20,
            json=False,
        )

        with patch.dict("sys.modules", {
            "aragora.knowledge": MagicMock(),
        }):
            result = cmd_facts(args)

        assert result == 1


class TestCmdJobs:
    """Tests for cmd_jobs function."""

    def test_jobs_show_requires_job_id(self, capsys):
        """cmd_jobs show requires job_id."""
        from aragora.cli.knowledge import cmd_jobs

        args = argparse.Namespace(
            action="show",
            job_id=None,
            workspace=None,
            status=None,
            limit=20,
            json=False,
        )

        mock_knowledge = MagicMock()
        mock_knowledge.get_all_jobs = MagicMock(return_value=[])
        mock_knowledge.get_job_status = MagicMock(return_value=None)

        with patch.dict("sys.modules", {
            "aragora.knowledge": mock_knowledge,
        }):
            result = cmd_jobs(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "job_id required" in captured.out.lower()


class TestCmdProcess:
    """Tests for cmd_process function."""

    def test_process_file_not_found(self, capsys):
        """cmd_process returns error for missing file."""
        from aragora.cli.knowledge import cmd_process

        args = argparse.Namespace(
            file="/nonexistent/file.pdf",
            workspace="default",
            sync=False,
            no_facts=False,
            json=False,
        )

        result = cmd_process(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()


class TestMain:
    """Tests for main entry point."""

    def test_main_no_command_shows_usage(self, capsys):
        """main shows usage when no subcommand."""
        from aragora.cli.knowledge import main

        args = argparse.Namespace()  # No func attribute

        result = main(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Usage" in captured.out or "knowledge" in captured.out
