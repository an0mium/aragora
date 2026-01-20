"""Tests for CLI billing command - billing and usage management."""

import json
import os
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aragora.cli.billing import (
    get_api_token,
    api_request,
    cmd_status,
    cmd_status_local,
    cmd_usage,
    cmd_usage_local,
    cmd_subscribe,
    cmd_portal,
    cmd_invoices,
    cmd_billing_default,
    create_billing_parser,
    main,
    DEFAULT_API_URL,
)


@pytest.fixture
def mock_args():
    """Create mock args object."""
    args = MagicMock()
    args.server = DEFAULT_API_URL
    return args


@pytest.fixture
def mock_api_response():
    """Create mock API response."""
    return {
        "plan": {
            "name": "Pro",
            "status": "active",
            "billing_period_end": "2026-02-01",
        },
        "current_usage": {
            "debates": 50,
            "tokens": 100000,
            "cost_usd": "12.50",
        },
        "limits": {
            "debates": 500,
            "tokens": 1000000,
        },
        "overages": {
            "debates": 0,
            "tokens": 0,
        },
    }


class TestGetApiToken:
    """Tests for get_api_token function."""

    def test_returns_api_token(self, monkeypatch):
        """Return ARAGORA_API_TOKEN when set."""
        monkeypatch.setenv("ARAGORA_API_TOKEN", "test-token")

        token = get_api_token()

        assert token == "test-token"

    def test_falls_back_to_api_key(self, monkeypatch):
        """Fall back to ARAGORA_API_KEY."""
        monkeypatch.delenv("ARAGORA_API_TOKEN", raising=False)
        monkeypatch.setenv("ARAGORA_API_KEY", "fallback-key")

        token = get_api_token()

        assert token == "fallback-key"

    def test_returns_none_when_not_set(self, monkeypatch):
        """Return None when no token set."""
        monkeypatch.delenv("ARAGORA_API_TOKEN", raising=False)
        monkeypatch.delenv("ARAGORA_API_KEY", raising=False)

        token = get_api_token()

        assert token is None


class TestApiRequest:
    """Tests for api_request function."""

    @patch("urllib.request.urlopen")
    def test_makes_get_request(self, mock_urlopen):
        """Make GET request successfully."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "ok"}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = api_request("GET", "/api/test")

        assert result == {"status": "ok"}
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_makes_post_request_with_data(self, mock_urlopen):
        """Make POST request with JSON data."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"created": true}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = api_request("POST", "/api/test", data={"key": "value"})

        assert result == {"created": True}

    @patch("urllib.request.urlopen")
    def test_includes_auth_header(self, mock_urlopen, monkeypatch):
        """Include Authorization header when token present."""
        monkeypatch.setenv("ARAGORA_API_TOKEN", "my-token")

        mock_response = MagicMock()
        mock_response.read.return_value = b"{}"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        api_request("GET", "/api/test")

        call_args = mock_urlopen.call_args[0][0]
        assert call_args.get_header("Authorization") == "Bearer my-token"

    @patch("urllib.request.urlopen")
    def test_raises_on_http_error(self, mock_urlopen):
        """Raise RuntimeError on HTTP error."""
        import urllib.error

        mock_error = urllib.error.HTTPError("http://test", 401, "Unauthorized", {}, None)
        mock_error.read = MagicMock(return_value=b"Unauthorized")
        mock_urlopen.side_effect = mock_error

        with pytest.raises(RuntimeError) as exc_info:
            api_request("GET", "/api/test")

        assert "API error (401)" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    def test_raises_on_connection_error(self, mock_urlopen):
        """Raise RuntimeError on connection error."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        with pytest.raises(RuntimeError) as exc_info:
            api_request("GET", "/api/test")

        assert "Connection error" in str(exc_info.value)


class TestCmdStatus:
    """Tests for cmd_status function."""

    @patch("aragora.cli.billing.api_request")
    def test_shows_billing_status(self, mock_request, mock_args, mock_api_response, capsys):
        """Show billing status from API."""
        mock_request.return_value = mock_api_response

        result = cmd_status(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "BILLING STATUS" in captured.out
        assert "Plan: Pro" in captured.out
        assert "Status: active" in captured.out
        assert "Debates: 50 / 500" in captured.out

    @patch("aragora.cli.billing.api_request")
    def test_shows_overages(self, mock_request, mock_args, mock_api_response, capsys):
        """Show overages when present."""
        mock_api_response["overages"] = {"debates": 10, "tokens": 5000}
        mock_request.return_value = mock_api_response

        result = cmd_status(mock_args)

        captured = capsys.readouterr()
        assert "Overages:" in captured.out
        assert "10 extra" in captured.out

    @patch("aragora.cli.billing.api_request")
    @patch("aragora.cli.billing.cmd_status_local")
    def test_falls_back_to_local(self, mock_local, mock_request, mock_args):
        """Fall back to local when connection fails."""
        mock_request.side_effect = RuntimeError("Connection refused")
        mock_local.return_value = 0

        result = cmd_status(mock_args)

        mock_local.assert_called_once()

    @patch("aragora.cli.billing.api_request")
    def test_handles_api_error(self, mock_request, mock_args, capsys):
        """Handle API errors gracefully."""
        mock_request.side_effect = RuntimeError("API error (500): Server error")

        result = cmd_status(mock_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.out


class TestCmdStatusLocal:
    """Tests for cmd_status_local function."""

    @patch("aragora.billing.usage.UsageTracker")
    def test_shows_local_usage(self, mock_tracker_cls, mock_args, capsys, monkeypatch):
        """Show local usage data."""
        monkeypatch.setenv("ARAGORA_ORG_ID", "test-org")

        mock_summary = MagicMock()
        mock_summary.period_start = MagicMock()
        mock_summary.period_start.date.return_value = "2026-01-01"
        mock_summary.period_end = MagicMock()
        mock_summary.period_end.date.return_value = "2026-01-31"
        mock_summary.total_debates = 25
        mock_summary.total_api_calls = 100
        mock_summary.total_agent_calls = 50
        mock_summary.total_tokens_in = 50000
        mock_summary.total_tokens_out = 25000
        mock_summary.total_cost_usd = 5.25
        mock_summary.cost_by_provider = {"anthropic": 3.00, "openai": 2.25}

        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = mock_summary
        mock_tracker_cls.return_value = mock_tracker

        result = cmd_status_local(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Local usage data" in captured.out
        assert "Organization: test-org" in captured.out
        assert "Debates: 25" in captured.out

    def test_handles_import_error(self, mock_args, capsys):
        """Handle missing billing module."""
        with patch.dict("sys.modules", {"aragora.billing.usage": None}):
            # Force import to fail
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                result = cmd_status_local(mock_args)

        assert result == 1


class TestCmdUsage:
    """Tests for cmd_usage function."""

    @patch("aragora.cli.billing.api_request")
    def test_shows_current_month_usage(self, mock_request, mock_args, capsys):
        """Show current month usage."""
        mock_args.month = None
        mock_args.verbose = False
        mock_request.return_value = {
            "total_debates": 50,
            "total_api_calls": 200,
            "total_agent_calls": 100,
            "total_tokens_in": 50000,
            "total_tokens_out": 25000,
            "total_cost_usd": "12.50",
        }

        result = cmd_usage(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "USAGE REPORT" in captured.out
        assert "Debates: 50" in captured.out

    @patch("aragora.cli.billing.api_request")
    def test_shows_specific_month_usage(self, mock_request, mock_args, capsys):
        """Show specific month usage."""
        mock_args.month = "2026-01"
        mock_args.verbose = False
        mock_request.return_value = {"total_debates": 30}

        result = cmd_usage(mock_args)

        captured = capsys.readouterr()
        assert "2026-01-01" in captured.out

    def test_handles_invalid_month_format(self, mock_args, capsys):
        """Handle invalid month format."""
        mock_args.month = "invalid"
        mock_args.verbose = False

        result = cmd_usage(mock_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Invalid month format" in captured.out

    @patch("aragora.cli.billing.api_request")
    def test_shows_verbose_daily_breakdown(self, mock_request, mock_args, capsys):
        """Show daily breakdown in verbose mode."""
        mock_args.month = None
        mock_args.verbose = True
        mock_request.return_value = {
            "total_debates": 10,
            "total_api_calls": 0,
            "total_agent_calls": 0,
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_cost_usd": "0",
            "debates_by_day": {
                "2026-01-15": 5,
                "2026-01-16": 3,
                "2026-01-17": 2,
            },
        }

        result = cmd_usage(mock_args)

        captured = capsys.readouterr()
        assert "Debates by day:" in captured.out
        assert "2026-01-15:" in captured.out


class TestCmdSubscribe:
    """Tests for cmd_subscribe function."""

    @patch("aragora.cli.billing.api_request")
    def test_shows_checkout_url(self, mock_request, mock_args, capsys):
        """Show checkout URL for subscription."""
        mock_args.plan = "pro"
        mock_args.open = False
        mock_request.return_value = {"checkout_url": "https://checkout.stripe.com/test"}

        result = cmd_subscribe(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "checkout.stripe.com" in captured.out

    @patch("aragora.cli.billing.api_request")
    @patch("webbrowser.open")
    def test_opens_browser_with_flag(self, mock_browser, mock_request, mock_args, capsys):
        """Open browser when --open flag set."""
        mock_args.plan = "pro"
        mock_args.open = True
        mock_request.return_value = {"checkout_url": "https://checkout.stripe.com/test"}

        result = cmd_subscribe(mock_args)

        mock_browser.assert_called_once_with("https://checkout.stripe.com/test")

    @patch("aragora.cli.billing.api_request")
    def test_shows_success_message(self, mock_request, mock_args, capsys):
        """Show success message for immediate subscription."""
        mock_args.plan = "pro"
        mock_args.open = False
        mock_request.return_value = {"success": True}

        result = cmd_subscribe(mock_args)

        captured = capsys.readouterr()
        assert "Successfully subscribed" in captured.out

    @patch("aragora.cli.billing.api_request")
    def test_handles_error(self, mock_request, mock_args, capsys):
        """Handle subscription error."""
        mock_args.plan = "pro"
        mock_args.open = False
        mock_request.side_effect = RuntimeError("Payment failed")

        result = cmd_subscribe(mock_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.out


class TestCmdPortal:
    """Tests for cmd_portal function."""

    @patch("aragora.cli.billing.api_request")
    @patch("webbrowser.open")
    def test_opens_portal(self, mock_browser, mock_request, mock_args, capsys):
        """Open billing portal in browser."""
        mock_args.no_open = False
        mock_request.return_value = {"url": "https://billing.stripe.com/portal"}

        result = cmd_portal(mock_args)

        assert result == 0
        mock_browser.assert_called_once()
        captured = capsys.readouterr()
        assert "Opened in browser" in captured.out

    @patch("aragora.cli.billing.api_request")
    def test_shows_url_without_opening(self, mock_request, mock_args, capsys):
        """Show URL without opening browser."""
        mock_args.no_open = True
        mock_request.return_value = {"url": "https://billing.stripe.com/portal"}

        result = cmd_portal(mock_args)

        captured = capsys.readouterr()
        assert "billing.stripe.com" in captured.out
        assert "Opened in browser" not in captured.out


class TestCmdInvoices:
    """Tests for cmd_invoices function."""

    @patch("aragora.cli.billing.api_request")
    def test_lists_invoices(self, mock_request, mock_args, capsys):
        """List invoices."""
        mock_args.limit = 10
        mock_request.return_value = {
            "invoices": [
                {"date": "2026-01-01T00:00:00", "amount": 2500, "status": "paid", "id": "inv_123"},
                {"date": "2025-12-01T00:00:00", "amount": 2500, "status": "paid", "id": "inv_122"},
            ]
        }

        result = cmd_invoices(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "INVOICES" in captured.out
        assert "25.00" in captured.out  # Amount formatted with spacing
        assert "paid" in captured.out

    @patch("aragora.cli.billing.api_request")
    def test_shows_no_invoices_message(self, mock_request, mock_args, capsys):
        """Show message when no invoices."""
        mock_args.limit = 10
        mock_request.return_value = {"invoices": []}

        result = cmd_invoices(mock_args)

        captured = capsys.readouterr()
        assert "No invoices found" in captured.out


class TestCmdBillingDefault:
    """Tests for cmd_billing_default function."""

    @patch("aragora.cli.billing.cmd_status")
    def test_defaults_to_status(self, mock_status, mock_args):
        """Default to status command."""
        mock_args.billing_action = None
        mock_status.return_value = 0

        result = cmd_billing_default(mock_args)

        mock_status.assert_called_once()


class TestCreateBillingParser:
    """Tests for create_billing_parser function."""

    def test_creates_parser(self):
        """Create billing subparser."""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        create_billing_parser(subparsers)

        # Should be able to parse billing commands
        args = parser.parse_args(["billing", "status"])
        assert hasattr(args, "func")

    def test_parses_usage_options(self):
        """Parse usage command options."""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_billing_parser(subparsers)

        args = parser.parse_args(["billing", "usage", "--month", "2026-01", "-v"])

        assert args.month == "2026-01"
        assert args.verbose is True

    def test_parses_subscribe_plan(self):
        """Parse subscribe command plan."""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_billing_parser(subparsers)

        args = parser.parse_args(["billing", "subscribe", "--plan", "pro"])

        assert args.plan == "pro"


class TestMain:
    """Tests for main function."""

    @patch("aragora.cli.billing.cmd_status")
    def test_calls_func(self, mock_status, mock_args):
        """Call the function from args."""
        mock_args.func = mock_status
        mock_status.return_value = 0

        result = main(mock_args)

        mock_status.assert_called_once()
        assert result == 0

    @patch("aragora.cli.billing.cmd_billing_default")
    def test_defaults_when_no_func(self, mock_default, mock_args):
        """Use default when no func attribute."""
        del mock_args.func
        mock_default.return_value = 0

        result = main(mock_args)

        mock_default.assert_called_once()
