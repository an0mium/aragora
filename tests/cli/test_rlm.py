"""Tests for RLM CLI commands."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import argparse

import pytest

from aragora.cli.rlm import (
    cmd_compress,
    cmd_query,
    cmd_stats,
    cmd_clear_cache,
    cmd_rlm,
    create_rlm_parser,
)


class TestCmdStats:
    """Tests for 'rlm stats' command."""

    def test_stats_shows_cache_info(self, capsys):
        """Test that stats command displays cache statistics."""
        args = argparse.Namespace()

        result = cmd_stats(args)

        captured = capsys.readouterr()
        assert result == 0
        assert "RLM CACHE STATISTICS" in captured.out
        assert "Cache size:" in captured.out
        assert "Max size:" in captured.out
        assert "Cache hits:" in captured.out
        assert "Hit rate:" in captured.out


class TestCmdClearCache:
    """Tests for 'rlm clear-cache' command."""

    def test_clear_cache_clears_cache(self, capsys):
        """Test that clear-cache command clears the cache."""
        args = argparse.Namespace()

        result = cmd_clear_cache(args)

        captured = capsys.readouterr()
        assert result == 0
        assert "Cleared" in captured.out


class TestCmdCompress:
    """Tests for 'rlm compress' command."""

    def test_compress_file_not_found(self, capsys):
        """Test error when input file doesn't exist."""
        args = argparse.Namespace(
            input="/nonexistent/file.txt",
            output=None,
            type=None,
            levels=4,
            no_cache=False,
        )

        result = cmd_compress(args)

        captured = capsys.readouterr()
        assert result == 1
        assert "File not found" in captured.out

    def test_compress_detects_code_type(self, capsys):
        """Test that compress detects code file type."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("def hello():\n    print('Hello world')\n")
            temp_path = f.name

        try:
            args = argparse.Namespace(
                input=temp_path,
                output=None,
                type=None,
                levels=4,
                no_cache=True,
            )

            result = cmd_compress(args)

            captured = capsys.readouterr()
            assert result == 0
            assert "Source type: code" in captured.out
            assert "COMPRESSION RESULTS" in captured.out
        finally:
            Path(temp_path).unlink()

    def test_compress_saves_output(self, capsys):
        """Test that compress saves output to file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("Test content for compression.\n")
            input_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            args = argparse.Namespace(
                input=input_path,
                output=output_path,
                type="text",
                levels=4,
                no_cache=True,
            )

            result = cmd_compress(args)

            assert result == 0

            # Check output file exists and is valid JSON
            with open(output_path) as f:
                data = json.load(f)

            assert "original_tokens" in data
            assert "levels" in data
            assert "compression_ratio" in data
        finally:
            Path(input_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()


class TestCmdQuery:
    """Tests for 'rlm query' command."""

    def test_query_context_not_found(self, capsys):
        """Test error when context file doesn't exist."""
        args = argparse.Namespace(
            query="What is X?",
            context="/nonexistent/context.json",
            strategy="auto",
            refine=False,
            max_iterations=3,
            stream=False,
        )

        result = cmd_query(args)

        captured = capsys.readouterr()
        assert result == 1
        assert "Context file not found" in captured.out

    def test_query_invalid_json(self, capsys):
        """Test error when context file is invalid JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("not valid json")
            temp_path = f.name

        try:
            args = argparse.Namespace(
                query="What is X?",
                context=temp_path,
                strategy="auto",
                refine=False,
                max_iterations=3,
                stream=False,
            )

            result = cmd_query(args)

            captured = capsys.readouterr()
            assert result == 1
            assert "Invalid JSON" in captured.out
        finally:
            Path(temp_path).unlink()

    def test_query_with_valid_context(self, capsys):
        """Test query with valid context file."""
        context_data = {
            "original_tokens": 100,
            "levels": {
                "SUMMARY": [
                    {
                        "id": "summary_1",
                        "content": "Test summary content about feature flags.",
                        "token_count": 20,
                        "key_topics": ["feature flags"],
                    }
                ]
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(context_data, f)
            temp_path = f.name

        try:
            args = argparse.Namespace(
                query="What is this about?",
                context=temp_path,
                strategy="auto",
                refine=False,
                max_iterations=3,
                stream=False,
            )

            result = cmd_query(args)

            captured = capsys.readouterr()
            assert result == 0
            assert "Loaded context with" in captured.out
            assert "ANSWER" in captured.out
        finally:
            Path(temp_path).unlink()


class TestCmdRlm:
    """Tests for main 'rlm' command."""

    def test_no_action_shows_help(self, capsys):
        """Test that no action shows usage help."""
        args = argparse.Namespace(action=None)

        result = cmd_rlm(args)

        captured = capsys.readouterr()
        assert result == 0
        assert "Usage: aragora rlm" in captured.out
        assert "compress" in captured.out
        assert "query" in captured.out
        assert "stats" in captured.out
        assert "clear-cache" in captured.out

    def test_stats_action(self, capsys):
        """Test stats action through main command."""
        args = argparse.Namespace(action="stats")

        result = cmd_rlm(args)

        captured = capsys.readouterr()
        assert result == 0
        assert "RLM CACHE STATISTICS" in captured.out


class TestCreateRlmParser:
    """Tests for CLI parser creation."""

    def test_parser_creation(self):
        """Test that parser is created correctly."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        create_rlm_parser(subparsers)

        # Parse a test command
        args = parser.parse_args(["rlm", "stats"])
        assert args.action == "stats"

    def test_compress_parser_options(self):
        """Test compress subcommand options."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        create_rlm_parser(subparsers)

        args = parser.parse_args([
            "rlm", "compress", "test.txt",
            "--output", "output.json",
            "--type", "code",
            "--levels", "3",
            "--no-cache",
        ])

        assert args.input == "test.txt"
        assert args.output == "output.json"
        assert args.type == "code"
        assert args.levels == 3
        assert args.no_cache is True

    def test_query_parser_options(self):
        """Test query subcommand options."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        create_rlm_parser(subparsers)

        args = parser.parse_args([
            "rlm", "query", "What is X?",
            "--context", "context.json",
            "--strategy", "grep",
            "--refine",
            "--max-iterations", "5",
            "--stream",
        ])

        assert args.query == "What is X?"
        assert args.context == "context.json"
        assert args.strategy == "grep"
        assert args.refine is True
        assert args.max_iterations == 5
        assert args.stream is True
