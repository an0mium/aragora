"""Tests for CLI memory operations commands."""

import argparse
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aragora.cli.commands.memory_ops import (
    _cmd_promote,
    _cmd_query,
    _cmd_stats,
    _cmd_store,
    add_memory_ops_parser,
    cmd_memory_ops,
)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestMemoryOpsParser:
    """Test argument parser setup."""

    def _build_parser(self):
        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers(dest="command")
        add_memory_ops_parser(subs)
        return parser

    def test_query_subcommand_parses(self):
        parser = self._build_parser()
        args = parser.parse_args(["memory", "query", "hello world"])
        assert args.memory_command == "query"
        assert args.text == "hello world"

    def test_query_with_tier_filter(self):
        parser = self._build_parser()
        args = parser.parse_args(["memory", "query", "test", "--tier", "slow"])
        assert args.tier == "slow"

    def test_query_with_limit(self):
        parser = self._build_parser()
        args = parser.parse_args(["memory", "query", "test", "--limit", "5"])
        assert args.limit == 5

    def test_store_subcommand_parses(self):
        parser = self._build_parser()
        args = parser.parse_args(["memory", "store", "important fact"])
        assert args.memory_command == "store"
        assert args.text == "important fact"
        assert args.tier == "fast"  # default

    def test_store_with_tier(self):
        parser = self._build_parser()
        args = parser.parse_args(["memory", "store", "data", "--tier", "glacial"])
        assert args.tier == "glacial"

    def test_stats_subcommand_parses(self):
        parser = self._build_parser()
        args = parser.parse_args(["memory", "stats"])
        assert args.memory_command == "stats"

    def test_promote_subcommand_parses(self):
        parser = self._build_parser()
        args = parser.parse_args(["memory", "promote", "abc123", "--to", "medium"])
        assert args.memory_command == "promote"
        assert args.id == "abc123"
        assert args.to == "medium"

    def test_promote_requires_to_flag(self):
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["memory", "promote", "abc123"])

    def test_invalid_tier_rejected(self):
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["memory", "store", "x", "--tier", "invalid"])

    def test_json_flag_on_query(self):
        parser = self._build_parser()
        args = parser.parse_args(["memory", "query", "test", "--json"])
        assert args.json is True


# ---------------------------------------------------------------------------
# Dispatch tests
# ---------------------------------------------------------------------------


class TestCmdMemoryOpsDispatch:
    """Test command dispatch routing."""

    @patch("aragora.cli.commands.memory_ops.asyncio")
    def test_dispatch_query(self, mock_asyncio):
        args = argparse.Namespace(memory_command="query")
        cmd_memory_ops(args)
        mock_asyncio.run.assert_called_once()

    @patch("aragora.cli.commands.memory_ops.asyncio")
    def test_dispatch_store(self, mock_asyncio):
        args = argparse.Namespace(memory_command="store")
        cmd_memory_ops(args)
        mock_asyncio.run.assert_called_once()

    @patch("aragora.cli.commands.memory_ops.asyncio")
    def test_dispatch_stats(self, mock_asyncio):
        args = argparse.Namespace(memory_command="stats")
        cmd_memory_ops(args)
        mock_asyncio.run.assert_called_once()

    @patch("aragora.cli.commands.memory_ops.asyncio")
    def test_dispatch_promote(self, mock_asyncio):
        args = argparse.Namespace(memory_command="promote")
        cmd_memory_ops(args)
        mock_asyncio.run.assert_called_once()

    def test_dispatch_none_prints_help(self, capsys):
        args = argparse.Namespace(memory_command=None)
        cmd_memory_ops(args)
        captured = capsys.readouterr()
        assert "Usage:" in captured.out


# ---------------------------------------------------------------------------
# API call tests (mocked)
# ---------------------------------------------------------------------------


class TestCmdQuery:
    """Test the query subcommand."""

    @pytest.mark.asyncio
    async def test_query_success(self, capsys):
        mock_result = {
            "results": [
                {"id": "abc123def456", "tier": "fast", "score": 0.95, "content": "test memory"}
            ],
            "total": 1,
        }
        args = argparse.Namespace(text="test", tier=None, limit=10, json=False)
        with patch(
            "aragora.cli.commands.memory_ops._api_get",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await _cmd_query(args)
        captured = capsys.readouterr()
        assert "1 memory entries" in captured.out
        assert "abc123def456" in captured.out

    @pytest.mark.asyncio
    async def test_query_json_output(self, capsys):
        mock_result = {"results": [], "total": 0}
        args = argparse.Namespace(text="test", tier=None, limit=10, json=True)
        with patch(
            "aragora.cli.commands.memory_ops._api_get",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await _cmd_query(args)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["total"] == 0

    @pytest.mark.asyncio
    async def test_query_empty_text(self, capsys):
        args = argparse.Namespace(text="", tier=None, limit=10, json=False)
        await _cmd_query(args)
        captured = capsys.readouterr()
        assert "Error" in captured.out

    @pytest.mark.asyncio
    async def test_query_connection_error(self, capsys):
        args = argparse.Namespace(text="test", tier=None, limit=10, json=False)
        with patch(
            "aragora.cli.commands.memory_ops._api_get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            await _cmd_query(args)
        captured = capsys.readouterr()
        assert "Could not connect" in captured.out

    @pytest.mark.asyncio
    async def test_query_no_results(self, capsys):
        mock_result = {"results": [], "total": 0}
        args = argparse.Namespace(text="nonexistent", tier=None, limit=10, json=False)
        with patch(
            "aragora.cli.commands.memory_ops._api_get",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await _cmd_query(args)
        captured = capsys.readouterr()
        assert "No matching entries" in captured.out


class TestCmdStore:
    """Test the store subcommand."""

    @pytest.mark.asyncio
    async def test_store_success(self, capsys):
        mock_result = {"id": "new123", "tier": "fast"}
        args = argparse.Namespace(text="important fact", tier="fast", json=False)
        with patch(
            "aragora.cli.commands.memory_ops._api_post",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await _cmd_store(args)
        captured = capsys.readouterr()
        assert "stored successfully" in captured.out
        assert "new123" in captured.out

    @pytest.mark.asyncio
    async def test_store_empty_text(self, capsys):
        args = argparse.Namespace(text="", tier="fast", json=False)
        await _cmd_store(args)
        captured = capsys.readouterr()
        assert "Error" in captured.out

    @pytest.mark.asyncio
    async def test_store_json_output(self, capsys):
        mock_result = {"id": "x1", "tier": "slow"}
        args = argparse.Namespace(text="data", tier="slow", json=True)
        with patch(
            "aragora.cli.commands.memory_ops._api_post",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await _cmd_store(args)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["id"] == "x1"


class TestCmdStats:
    """Test the stats subcommand."""

    @pytest.mark.asyncio
    async def test_stats_success(self, capsys):
        mock_result = {
            "tiers": {
                "fast": {"count": 100, "hit_rate": 0.85},
                "medium": {"count": 50, "hit_rate": 0.6},
                "slow": {"count": 20, "hit_rate": 0.3},
                "glacial": {"count": 5, "hit_rate": 0.1},
            },
            "total_entries": 175,
            "promotions": 12,
            "demotions": 3,
        }
        args = argparse.Namespace(json=False)
        with patch(
            "aragora.cli.commands.memory_ops._api_get",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await _cmd_stats(args)
        captured = capsys.readouterr()
        assert "MEMORY TIER STATISTICS" in captured.out
        assert "175" in captured.out
        assert "Promotions: 12" in captured.out

    @pytest.mark.asyncio
    async def test_stats_json_output(self, capsys):
        mock_result = {"tiers": {}, "total_entries": 0}
        args = argparse.Namespace(json=True)
        with patch(
            "aragora.cli.commands.memory_ops._api_get",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await _cmd_stats(args)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["total_entries"] == 0


class TestCmdPromote:
    """Test the promote subcommand."""

    @pytest.mark.asyncio
    async def test_promote_success(self, capsys):
        mock_result = {"success": True, "previous_tier": "fast"}
        args = argparse.Namespace(id="abc123", to="medium", json=False)
        with patch(
            "aragora.cli.commands.memory_ops._api_post",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await _cmd_promote(args)
        captured = capsys.readouterr()
        assert "promoted successfully" in captured.out
        assert "fast" in captured.out
        assert "medium" in captured.out

    @pytest.mark.asyncio
    async def test_promote_missing_id(self, capsys):
        args = argparse.Namespace(id=None, to="medium", json=False)
        await _cmd_promote(args)
        captured = capsys.readouterr()
        assert "Error" in captured.out

    @pytest.mark.asyncio
    async def test_promote_missing_to(self, capsys):
        args = argparse.Namespace(id="abc123", to=None, json=False)
        await _cmd_promote(args)
        captured = capsys.readouterr()
        assert "Error" in captured.out

    @pytest.mark.asyncio
    async def test_promote_failure_response(self, capsys):
        mock_result = {"success": False, "error": "Entry not found"}
        args = argparse.Namespace(id="missing", to="slow", json=False)
        with patch(
            "aragora.cli.commands.memory_ops._api_post",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await _cmd_promote(args)
        captured = capsys.readouterr()
        assert "failed" in captured.out
        assert "Entry not found" in captured.out

    @pytest.mark.asyncio
    async def test_promote_http_error(self, capsys):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "Not found"}
        error = httpx.HTTPStatusError("not found", request=MagicMock(), response=mock_response)
        args = argparse.Namespace(id="abc", to="slow", json=False)
        with patch(
            "aragora.cli.commands.memory_ops._api_post", new_callable=AsyncMock, side_effect=error
        ):
            await _cmd_promote(args)
        captured = capsys.readouterr()
        assert "404" in captured.out
