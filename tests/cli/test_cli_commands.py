"""Tests for Phase W CLI command modules.

Covers argument parsing and command dispatch for:
- computer_use
- connectors
- rbac_ops
- knowledge (km)
- billing_ops (costs)
"""

import argparse
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from aragora.cli.commands.computer_use import (
    _cmd_list as cu_cmd_list,
    _cmd_run as cu_cmd_run,
    _cmd_status as cu_cmd_status,
    add_computer_use_parser,
    cmd_computer_use,
)
from aragora.cli.commands.connectors import (
    _cmd_list as conn_cmd_list,
    _cmd_status as conn_cmd_status,
    _cmd_test as conn_cmd_test,
    add_connectors_parser,
    cmd_connectors,
)
from aragora.cli.commands.rbac_ops import (
    _cmd_assign,
    _cmd_check,
    _cmd_roles,
    _cmd_permissions,
    add_rbac_ops_parser,
    cmd_rbac_ops,
)
from aragora.cli.commands.knowledge import (
    _cmd_query as km_cmd_query,
    _cmd_store as km_cmd_store,
    _cmd_stats as km_cmd_stats,
    add_knowledge_ops_parser,
    cmd_knowledge,
)
from aragora.cli.commands.billing_ops import (
    _cmd_usage,
    _cmd_budget,
    _cmd_forecast,
    _cmd_report,
    _cmd_agents,
    add_billing_ops_parser,
    cmd_billing_ops,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_parser(add_fn, name):
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command")
    add_fn(subs)
    return parser


# ===========================================================================
# computer-use
# ===========================================================================


class TestComputerUseParser:
    def _parser(self):
        return _build_parser(add_computer_use_parser, "computer-use")

    def test_run_parses(self):
        args = self._parser().parse_args(["computer-use", "run", "open browser"])
        assert args.cu_command == "run"
        assert args.goal == "open browser"

    def test_status_parses(self):
        args = self._parser().parse_args(["computer-use", "status", "task-42"])
        assert args.cu_command == "status"
        assert args.task_id == "task-42"

    def test_list_parses(self):
        args = self._parser().parse_args(["computer-use", "list"])
        assert args.cu_command == "list"

    def test_list_limit(self):
        args = self._parser().parse_args(["computer-use", "list", "--limit", "5"])
        assert args.limit == 5


class TestComputerUseDispatch:
    @patch("aragora.cli.commands.computer_use.asyncio")
    def test_dispatch_run(self, mock_asyncio):
        args = argparse.Namespace(cu_command="run")
        cmd_computer_use(args)
        mock_asyncio.run.assert_called_once()

    @patch("aragora.cli.commands.computer_use.asyncio")
    def test_dispatch_status(self, mock_asyncio):
        args = argparse.Namespace(cu_command="status")
        cmd_computer_use(args)
        mock_asyncio.run.assert_called_once()

    @patch("aragora.cli.commands.computer_use.asyncio")
    def test_dispatch_list(self, mock_asyncio):
        args = argparse.Namespace(cu_command="list")
        cmd_computer_use(args)
        mock_asyncio.run.assert_called_once()

    def test_dispatch_none_prints_help(self, capsys):
        args = argparse.Namespace(cu_command=None)
        cmd_computer_use(args)
        captured = capsys.readouterr()
        assert "Usage:" in captured.out


class TestComputerUseCommands:
    @pytest.mark.asyncio
    async def test_run_success(self, capsys):
        mock_result = {"task_id": "t-123", "status": "created"}
        args = argparse.Namespace(goal="open browser", json=False)
        with patch(
            "aragora.cli.commands.computer_use._api_post",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await cu_cmd_run(args)
        out = capsys.readouterr().out
        assert "t-123" in out
        assert "created" in out

    @pytest.mark.asyncio
    async def test_run_empty_goal(self, capsys):
        args = argparse.Namespace(goal="", json=False)
        await cu_cmd_run(args)
        assert "Error" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_status_connection_error(self, capsys):
        args = argparse.Namespace(task_id="t-1", json=False)
        with patch(
            "aragora.cli.commands.computer_use._api_get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            await cu_cmd_status(args)
        assert "Could not connect" in capsys.readouterr().out


# ===========================================================================
# connectors
# ===========================================================================


class TestConnectorsParser:
    def _parser(self):
        return _build_parser(add_connectors_parser, "connectors")

    def test_list_parses(self):
        args = self._parser().parse_args(["connectors", "list"])
        assert args.conn_command == "list"

    def test_status_parses(self):
        args = self._parser().parse_args(["connectors", "status", "slack"])
        assert args.conn_command == "status"
        assert args.name == "slack"

    def test_test_parses(self):
        args = self._parser().parse_args(["connectors", "test", "teams"])
        assert args.conn_command == "test"
        assert args.name == "teams"


class TestConnectorsDispatch:
    @patch("aragora.cli.commands.connectors.asyncio")
    def test_dispatch_list(self, mock_asyncio):
        args = argparse.Namespace(conn_command="list")
        cmd_connectors(args)
        mock_asyncio.run.assert_called_once()

    @patch("aragora.cli.commands.connectors.asyncio")
    def test_dispatch_status(self, mock_asyncio):
        args = argparse.Namespace(conn_command="status")
        cmd_connectors(args)
        mock_asyncio.run.assert_called_once()

    def test_dispatch_none_prints_help(self, capsys):
        args = argparse.Namespace(conn_command=None)
        cmd_connectors(args)
        assert "Usage:" in capsys.readouterr().out


class TestConnectorsCommands:
    @pytest.mark.asyncio
    async def test_list_success(self, capsys):
        mock = {"connectors": [{"name": "slack", "type": "slack", "status": "connected"}]}
        args = argparse.Namespace(json=False, type=None)
        with patch(
            "aragora.cli.commands.connectors._api_get", new_callable=AsyncMock, return_value=mock
        ):
            await conn_cmd_list(args)
        out = capsys.readouterr().out
        assert "slack" in out

    @pytest.mark.asyncio
    async def test_test_success(self, capsys):
        mock = {"success": True, "latency_ms": 42}
        args = argparse.Namespace(name="slack", json=False)
        with patch(
            "aragora.cli.commands.connectors._api_post", new_callable=AsyncMock, return_value=mock
        ):
            await conn_cmd_test(args)
        out = capsys.readouterr().out
        assert "passed" in out

    @pytest.mark.asyncio
    async def test_test_missing_name(self, capsys):
        args = argparse.Namespace(name=None, json=False)
        await conn_cmd_test(args)
        assert "Error" in capsys.readouterr().out


# ===========================================================================
# rbac
# ===========================================================================


class TestRbacParser:
    def _parser(self):
        return _build_parser(add_rbac_ops_parser, "rbac")

    def test_roles_parses(self):
        args = self._parser().parse_args(["rbac", "roles"])
        assert args.rbac_command == "roles"

    def test_permissions_parses(self):
        args = self._parser().parse_args(["rbac", "permissions"])
        assert args.rbac_command == "permissions"

    def test_assign_parses(self):
        args = self._parser().parse_args(["rbac", "assign", "user-1", "admin"])
        assert args.rbac_command == "assign"
        assert args.user_id == "user-1"
        assert args.role == "admin"

    def test_check_parses(self):
        args = self._parser().parse_args(["rbac", "check", "user-1", "debates:read"])
        assert args.rbac_command == "check"
        assert args.user_id == "user-1"
        assert args.permission == "debates:read"


class TestRbacDispatch:
    @patch("aragora.cli.commands.rbac_ops.asyncio")
    def test_dispatch_roles(self, mock_asyncio):
        args = argparse.Namespace(rbac_command="roles")
        cmd_rbac_ops(args)
        mock_asyncio.run.assert_called_once()

    @patch("aragora.cli.commands.rbac_ops.asyncio")
    def test_dispatch_assign(self, mock_asyncio):
        args = argparse.Namespace(rbac_command="assign")
        cmd_rbac_ops(args)
        mock_asyncio.run.assert_called_once()

    def test_dispatch_none_prints_help(self, capsys):
        args = argparse.Namespace(rbac_command=None)
        cmd_rbac_ops(args)
        assert "Usage:" in capsys.readouterr().out


class TestRbacCommands:
    @pytest.mark.asyncio
    async def test_roles_success(self, capsys):
        mock = {
            "roles": [{"name": "admin", "description": "Full access", "permissions": ["a", "b"]}]
        }
        args = argparse.Namespace(json=False)
        with patch(
            "aragora.cli.commands.rbac_ops._api_get", new_callable=AsyncMock, return_value=mock
        ):
            await _cmd_roles(args)
        out = capsys.readouterr().out
        assert "admin" in out
        assert "2 permissions" in out

    @pytest.mark.asyncio
    async def test_assign_success(self, capsys):
        mock = {"success": True}
        args = argparse.Namespace(user_id="u1", role="viewer", json=False)
        with patch(
            "aragora.cli.commands.rbac_ops._api_post", new_callable=AsyncMock, return_value=mock
        ):
            await _cmd_assign(args)
        out = capsys.readouterr().out
        assert "viewer" in out
        assert "u1" in out

    @pytest.mark.asyncio
    async def test_check_allowed(self, capsys):
        mock = {"allowed": True, "reason": ""}
        args = argparse.Namespace(user_id="u1", permission="debates:read", json=False)
        with patch(
            "aragora.cli.commands.rbac_ops._api_get", new_callable=AsyncMock, return_value=mock
        ):
            await _cmd_check(args)
        out = capsys.readouterr().out
        assert "HAS permission" in out

    @pytest.mark.asyncio
    async def test_check_denied(self, capsys):
        mock = {"allowed": False, "reason": "no role"}
        args = argparse.Namespace(user_id="u1", permission="admin:write", json=False)
        with patch(
            "aragora.cli.commands.rbac_ops._api_get", new_callable=AsyncMock, return_value=mock
        ):
            await _cmd_check(args)
        out = capsys.readouterr().out
        assert "DOES NOT have" in out


# ===========================================================================
# knowledge (km)
# ===========================================================================


class TestKnowledgeParser:
    def _parser(self):
        return _build_parser(add_knowledge_ops_parser, "km")

    def test_query_parses(self):
        args = self._parser().parse_args(["km", "query", "rate limiter"])
        assert args.km_command == "query"
        assert args.text == "rate limiter"

    def test_store_parses(self):
        args = self._parser().parse_args(["km", "store", "fact", "--source", "doc"])
        assert args.km_command == "store"
        assert args.text == "fact"
        assert args.source == "doc"

    def test_stats_parses(self):
        args = self._parser().parse_args(["km", "stats"])
        assert args.km_command == "stats"


class TestKnowledgeCommands:
    @pytest.mark.asyncio
    async def test_query_success(self, capsys):
        mock = {
            "results": [{"id": "k1", "source": "doc", "score": 0.9, "content": "fact"}],
            "total": 1,
        }
        args = argparse.Namespace(text="test", limit=10, json=False)
        with patch(
            "aragora.cli.commands.knowledge._api_get", new_callable=AsyncMock, return_value=mock
        ):
            await km_cmd_query(args)
        out = capsys.readouterr().out
        assert "1 knowledge entries" in out

    @pytest.mark.asyncio
    async def test_query_empty(self, capsys):
        args = argparse.Namespace(text="", limit=10, json=False)
        await km_cmd_query(args)
        assert "Error" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_store_success(self, capsys):
        mock = {"id": "k-new", "source": "cli"}
        args = argparse.Namespace(text="important", source="cli", json=False)
        with patch(
            "aragora.cli.commands.knowledge._api_post", new_callable=AsyncMock, return_value=mock
        ):
            await km_cmd_store(args)
        out = capsys.readouterr().out
        assert "stored successfully" in out

    @pytest.mark.asyncio
    async def test_stats_json(self, capsys):
        mock = {"total_entries": 42, "adapters": 5, "sources": {}}
        args = argparse.Namespace(json=True)
        with patch(
            "aragora.cli.commands.knowledge._api_get", new_callable=AsyncMock, return_value=mock
        ):
            await km_cmd_stats(args)
        parsed = json.loads(capsys.readouterr().out)
        assert parsed["total_entries"] == 42


# ===========================================================================
# billing (costs)
# ===========================================================================


class TestBillingParser:
    def _parser(self):
        return _build_parser(add_billing_ops_parser, "costs")

    def test_usage_parses(self):
        args = self._parser().parse_args(["costs", "usage"])
        assert args.billing_command == "usage"

    def test_budget_parses(self):
        args = self._parser().parse_args(["costs", "budget"])
        assert args.billing_command == "budget"

    def test_forecast_parses(self):
        args = self._parser().parse_args(["costs", "forecast"])
        assert args.billing_command == "forecast"

    def test_forecast_days(self):
        args = self._parser().parse_args(["costs", "forecast", "--days", "7"])
        assert args.days == 7


class TestBillingDispatch:
    @patch("aragora.cli.commands.billing_ops.asyncio")
    def test_dispatch_usage(self, mock_asyncio):
        args = argparse.Namespace(billing_command="usage")
        cmd_billing_ops(args)
        mock_asyncio.run.assert_called_once()

    def test_dispatch_none_prints_help(self, capsys):
        args = argparse.Namespace(billing_command=None)
        cmd_billing_ops(args)
        assert "Usage:" in capsys.readouterr().out


class TestBillingCommands:
    @pytest.mark.asyncio
    async def test_usage_success(self, capsys):
        mock = {
            "total_cost": 1.2345,
            "total_tokens": 5000,
            "period": "Jan 2026",
            "by_provider": {"anthropic": {"cost": 1.0}},
        }
        args = argparse.Namespace(json=False, period="current")
        with patch(
            "aragora.cli.commands.billing_ops._api_get", new_callable=AsyncMock, return_value=mock
        ):
            await _cmd_usage(args)
        out = capsys.readouterr().out
        assert "USAGE SUMMARY" in out
        assert "1.2345" in out

    @pytest.mark.asyncio
    async def test_budget_success(self, capsys):
        mock = {
            "limit": 100.0,
            "spent": 25.5,
            "remaining": 74.5,
            "utilization": 0.255,
            "alerts": [],
        }
        args = argparse.Namespace(json=False)
        with patch(
            "aragora.cli.commands.billing_ops._api_get", new_callable=AsyncMock, return_value=mock
        ):
            await _cmd_budget(args)
        out = capsys.readouterr().out
        assert "BUDGET STATUS" in out
        assert "100.00" in out

    @pytest.mark.asyncio
    async def test_forecast_json(self, capsys):
        mock = {
            "projected_cost": 50.0,
            "daily_average": 1.5,
            "horizon_days": 30,
            "confidence": 0.85,
        }
        args = argparse.Namespace(json=True, days=30)
        with patch(
            "aragora.cli.commands.billing_ops._api_get", new_callable=AsyncMock, return_value=mock
        ):
            await _cmd_forecast(args)
        parsed = json.loads(capsys.readouterr().out)
        assert parsed["projected_cost"] == 50.0

    @pytest.mark.asyncio
    async def test_usage_connection_error(self, capsys):
        args = argparse.Namespace(json=False, period="current")
        with patch(
            "aragora.cli.commands.billing_ops._api_get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            await _cmd_usage(args)
        assert "Could not connect" in capsys.readouterr().out


class TestBillingLocalCostCommands:
    """Regression tests for `costs report` / `costs agents`.

    Both commands previously always printed a misleading
    'CostTracker.<method>() not available.' error because they called the
    async CostTracker methods with an unsupported ``days=`` kwarg and never
    awaited the returned coroutine. These tests pin the corrected behavior:
    the real (async) tracker methods are invoked with period bounds and the
    results are rendered.
    """

    def _report_parser(self):
        return _build_parser(add_billing_ops_parser, "costs")

    def test_report_workspace_flag_parses(self):
        args = self._report_parser().parse_args(
            ["costs", "report", "--days", "7", "--workspace", "ws1"]
        )
        assert args.days == 7
        assert args.workspace == "ws1"

    def test_agents_workspace_flag_parses(self):
        args = self._report_parser().parse_args(["costs", "agents", "--workspace", "ws2"])
        assert args.workspace == "ws2"

    def test_report_renders_async_costreport(self, capsys):
        """generate_report is async + returns a CostReport; it must be awaited
        and normalized via .to_dict(), not reported as 'not available'."""
        report_obj = Mock()
        report_obj.to_dict.return_value = {
            "period_start": "2026-05-01",
            "period_end": "2026-05-31",
            "total_cost_usd": "1.2345",
            "total_tokens_in": 1000,
            "total_tokens_out": 500,
            "total_api_calls": 3,
            "cost_by_provider": {"anthropic": "1.2345"},
            "top_agents_by_cost": [{"agent": "claude", "cost_usd": "1.2345"}],
        }
        tracker = Mock()
        tracker.generate_report = AsyncMock(return_value=report_obj)

        args = argparse.Namespace(json=False, days=30, workspace=None)
        with patch("aragora.cli.commands.billing_ops._get_cost_tracker", return_value=tracker):
            _cmd_report(args)

        out = capsys.readouterr().out
        assert "not available" not in out
        assert "COST REPORT" in out
        assert "1.2345" in out
        assert "claude" in out
        # The async method must actually have been called (and awaited).
        tracker.generate_report.assert_awaited_once()
        _, kwargs = tracker.generate_report.call_args
        assert "days" not in kwargs  # the buggy kwarg must be gone
        assert kwargs["period_start"] is not None
        assert kwargs["period_end"] is not None

    def test_agents_renders_async_agent_costs(self, capsys):
        """get_agent_costs is async, requires a workspace_id, and returns a
        mapping of agent -> {cost_usd, percentage}."""
        tracker = Mock()
        tracker.get_agent_costs = AsyncMock(
            return_value={
                "claude": {"cost_usd": "0.9000", "percentage": 75.0},
                "gpt": {"cost_usd": "0.3000", "percentage": 25.0},
            }
        )

        args = argparse.Namespace(json=False, days=30, workspace="ws1")
        with patch("aragora.cli.commands.billing_ops._get_cost_tracker", return_value=tracker):
            _cmd_agents(args)

        out = capsys.readouterr().out
        assert "not available" not in out
        assert "AGENT COSTS" in out
        assert "claude" in out
        assert "0.9000" in out
        tracker.get_agent_costs.assert_awaited_once()
        # workspace_id passed positionally; days kwarg must be gone
        call = tracker.get_agent_costs.call_args
        assert call.args[0] == "ws1"
        assert "days" not in call.kwargs

    def test_agents_json_output(self, capsys):
        tracker = Mock()
        tracker.get_agent_costs = AsyncMock(
            return_value={"claude": {"cost_usd": "0.9", "percentage": 100.0}}
        )
        args = argparse.Namespace(json=True, days=30, workspace=None)
        with patch("aragora.cli.commands.billing_ops._get_cost_tracker", return_value=tracker):
            _cmd_agents(args)
        parsed = json.loads(capsys.readouterr().out)
        assert parsed["claude"]["cost_usd"] == "0.9"
        # default workspace used when none supplied
        assert tracker.get_agent_costs.call_args.args[0] == "default"

    def test_report_empty_tracker_still_succeeds(self, capsys):
        """Empty data must render a real (zero) report, never the misleading
        'not available' error."""
        report_obj = Mock()
        report_obj.to_dict.return_value = {
            "period_start": "2026-05-01",
            "period_end": "2026-05-31",
            "total_cost_usd": "0",
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_api_calls": 0,
        }
        tracker = Mock()
        tracker.generate_report = AsyncMock(return_value=report_obj)
        args = argparse.Namespace(json=False, days=30, workspace=None)
        with patch("aragora.cli.commands.billing_ops._get_cost_tracker", return_value=tracker):
            _cmd_report(args)
        out = capsys.readouterr().out
        assert "not available" not in out
        assert "COST REPORT" in out
