"""Tests for CLI billing module."""

import argparse
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from aragora.cli.billing import (
    api_request,
    cmd_billing_default,
    cmd_invoices,
    cmd_portal,
    cmd_status,
    cmd_subscribe,
    cmd_usage,
    create_billing_parser,
    get_api_token,
)


class TestGetApiToken:
    """Test API token retrieval."""

    def test_get_api_token_from_token_env(self, monkeypatch):
        """Test getting token from ARAGORA_API_TOKEN."""
        monkeypatch.setenv("ARAGORA_API_TOKEN", "test-token-123")
        monkeypatch.delenv("ARAGORA_API_KEY", raising=False)
        assert get_api_token() == "test-token-123"

    def test_get_api_token_from_key_env(self, monkeypatch):
        """Test getting token from ARAGORA_API_KEY."""
        monkeypatch.delenv("ARAGORA_API_TOKEN", raising=False)
        monkeypatch.setenv("ARAGORA_API_KEY", "api-key-456")
        assert get_api_token() == "api-key-456"

    def test_get_api_token_prefers_token(self, monkeypatch):
        """Test that ARAGORA_API_TOKEN takes precedence."""
        monkeypatch.setenv("ARAGORA_API_TOKEN", "token-first")
        monkeypatch.setenv("ARAGORA_API_KEY", "key-second")
        assert get_api_token() == "token-first"

    def test_get_api_token_returns_none(self, monkeypatch):
        """Test returns None when no token set."""
        monkeypatch.delenv("ARAGORA_API_TOKEN", raising=False)
        monkeypatch.delenv("ARAGORA_API_KEY", raising=False)
        assert get_api_token() is None


class TestApiRequest:
    """Test API request helper."""

    def test_api_request_success(self):
        """Test successful API request."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"success": true}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = api_request("GET", "/api/test")
            assert result == {"success": True}

    def test_api_request_with_data(self):
        """Test API request with POST data."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"created": true}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_open:
            result = api_request("POST", "/api/create", data={"name": "test"})
            assert result == {"created": True}

            # Verify request was called
            mock_open.assert_called_once()
            req = mock_open.call_args[0][0]
            assert req.method == "POST"
            assert b'"name"' in req.data

    def test_api_request_with_auth_token(self, monkeypatch):
        """Test API request includes auth token."""
        monkeypatch.setenv("ARAGORA_API_TOKEN", "my-token")

        mock_response = MagicMock()
        mock_response.read.return_value = b"{}"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_open:
            api_request("GET", "/api/test")

            req = mock_open.call_args[0][0]
            assert req.get_header("Authorization") == "Bearer my-token"

    def test_api_request_http_error(self):
        """Test handling HTTP errors."""
        import urllib.error

        mock_error = urllib.error.HTTPError("http://test/api", 404, "Not Found", {}, None)
        mock_error.read = MagicMock(return_value=b"Resource not found")

        with patch("urllib.request.urlopen", side_effect=mock_error):
            with pytest.raises(RuntimeError) as exc_info:
                api_request("GET", "/api/missing")
            assert "API error (404)" in str(exc_info.value)

    def test_api_request_connection_error(self):
        """Test handling connection errors."""
        import urllib.error

        mock_error = urllib.error.URLError("Connection refused")

        with patch("urllib.request.urlopen", side_effect=mock_error):
            with pytest.raises(RuntimeError) as exc_info:
                api_request("GET", "/api/test")
            assert "Connection error" in str(exc_info.value)


class TestCmdStatus:
    """Test status command."""

    def test_cmd_status_success(self, capsys):
        """Test successful status display."""
        mock_result = {
            "plan": {"name": "Pro", "status": "active", "billing_period_end": "2026-02-01"},
            "current_usage": {"debates": 50, "tokens": 100000, "cost_usd": "25.50"},
            "limits": {"debates": 100, "tokens": 500000},
        }

        args = argparse.Namespace(server="http://localhost:8080")

        with patch("aragora.cli.billing.api_request", return_value=mock_result):
            result = cmd_status(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Pro" in captured.out
        assert "active" in captured.out
        assert "50" in captured.out
        assert "25.50" in captured.out

    def test_cmd_status_with_overages(self, capsys):
        """Test status display with overages."""
        mock_result = {
            "plan": {"name": "Free", "status": "active"},
            "current_usage": {"debates": 15, "tokens": 60000, "cost_usd": "0"},
            "limits": {"debates": 10, "tokens": 50000},
            "overages": {"debates": 5, "tokens": 10000},
        }

        args = argparse.Namespace(server="http://localhost:8080")

        with patch("aragora.cli.billing.api_request", return_value=mock_result):
            result = cmd_status(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Overages" in captured.out
        assert "5 extra" in captured.out

    def test_cmd_status_api_error(self, capsys):
        """Test status with API error."""
        args = argparse.Namespace(server="http://localhost:8080")

        with patch(
            "aragora.cli.billing.api_request",
            side_effect=RuntimeError("API error (500): Server error"),
        ):
            result = cmd_status(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.out

    def test_cmd_status_connection_fallback(self, capsys, monkeypatch):
        """Test status falls back to local data on connection error."""
        args = argparse.Namespace(server="http://localhost:8080")

        # Mock local usage tracker
        mock_summary = MagicMock()
        mock_summary.period_start = MagicMock()
        mock_summary.period_start.date.return_value = "2026-01-01"
        mock_summary.period_end = MagicMock()
        mock_summary.period_end.date.return_value = "2026-01-31"
        mock_summary.total_debates = 10
        mock_summary.total_api_calls = 100
        mock_summary.total_agent_calls = 50
        mock_summary.total_tokens_in = 5000
        mock_summary.total_tokens_out = 3000
        mock_summary.total_cost_usd = 1.50
        mock_summary.cost_by_provider = {"anthropic": 1.0, "openai": 0.5}

        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = mock_summary

        with patch(
            "aragora.cli.billing.api_request",
            side_effect=RuntimeError("Connection refused"),
        ):
            with patch(
                "aragora.billing.usage.UsageTracker",
                return_value=mock_tracker,
            ):
                result = cmd_status(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Local usage data" in captured.out


class TestCmdUsage:
    """Test usage command."""

    def test_cmd_usage_current_month(self, capsys):
        """Test usage for current month."""
        mock_result = {
            "usage": {
                "total_debates": 25,
                "total_api_calls": 500,
                "total_agent_calls": 200,
                "total_tokens_in": 50000,
                "total_tokens_out": 30000,
                "total_cost_usd": "12.50",
                "cost_by_provider": {"anthropic": "8.00", "openai": "4.50"},
            }
        }

        args = argparse.Namespace(month=None, verbose=False, server="http://localhost:8080")

        with patch("aragora.cli.billing.api_request", return_value=mock_result):
            result = cmd_usage(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "25" in captured.out
        assert "12.50" in captured.out
        assert "anthropic" in captured.out

    def test_cmd_usage_specific_month(self, capsys):
        """Test usage for specific month."""
        mock_result = {
            "usage": {
                "total_debates": 15,
                "total_api_calls": 300,
                "total_agent_calls": 100,
                "total_tokens_in": 30000,
                "total_tokens_out": 20000,
                "total_cost_usd": "8.00",
            }
        }

        args = argparse.Namespace(month="2026-01", verbose=False, server="http://localhost:8080")

        with patch("aragora.cli.billing.api_request", return_value=mock_result):
            result = cmd_usage(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "2026-01-01" in captured.out

    def test_cmd_usage_invalid_month(self, capsys):
        """Test usage with invalid month format."""
        args = argparse.Namespace(month="invalid", verbose=False, server="http://localhost:8080")

        result = cmd_usage(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Invalid month format" in captured.out

    def test_cmd_usage_verbose(self, capsys):
        """Test usage with verbose daily breakdown."""
        mock_result = {
            "usage": {
                "total_debates": 10,
                "total_api_calls": 100,
                "total_agent_calls": 50,
                "total_tokens_in": 10000,
                "total_tokens_out": 5000,
                "total_cost_usd": "5.00",
                "debates_by_day": {"2026-01-01": 3, "2026-01-02": 7},
            }
        }

        args = argparse.Namespace(month=None, verbose=True, server="http://localhost:8080")

        with patch("aragora.cli.billing.api_request", return_value=mock_result):
            result = cmd_usage(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "2026-01-01" in captured.out
        assert "█" in captured.out  # Bar chart


class TestCmdSubscribe:
    """Test subscribe command."""

    def test_cmd_subscribe_with_checkout(self, capsys):
        """Test subscribe returns checkout URL."""
        mock_result = {"checkout_url": "https://checkout.stripe.com/123"}

        args = argparse.Namespace(plan="pro", open=False, server="http://localhost:8080")

        with patch("aragora.cli.billing.api_request", return_value=mock_result):
            result = cmd_subscribe(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "checkout.stripe.com" in captured.out

    def test_cmd_subscribe_success(self, capsys):
        """Test immediate subscription success."""
        mock_result = {"success": True}

        args = argparse.Namespace(plan="team", open=False, server="http://localhost:8080")

        with patch("aragora.cli.billing.api_request", return_value=mock_result):
            result = cmd_subscribe(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Successfully subscribed" in captured.out
        assert "team" in captured.out

    def test_cmd_subscribe_error(self, capsys):
        """Test subscribe with API error."""
        args = argparse.Namespace(plan="enterprise", open=False, server="http://localhost:8080")

        with patch(
            "aragora.cli.billing.api_request",
            side_effect=RuntimeError("API error (403): Unauthorized"),
        ):
            result = cmd_subscribe(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.out


class TestCmdPortal:
    """Test portal command."""

    def test_cmd_portal_success(self, capsys):
        """Test portal returns URL."""
        mock_result = {"url": "https://billing.stripe.com/portal/123"}

        args = argparse.Namespace(no_open=True, server="http://localhost:8080")

        with patch("aragora.cli.billing.api_request", return_value=mock_result):
            result = cmd_portal(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "billing.stripe.com" in captured.out

    def test_cmd_portal_opens_browser(self, capsys):
        """Test portal opens browser when not --no-open."""
        mock_result = {"url": "https://billing.stripe.com/portal/123"}

        args = argparse.Namespace(no_open=False, server="http://localhost:8080")

        with patch("aragora.cli.billing.api_request", return_value=mock_result):
            with patch("webbrowser.open") as mock_browser:
                result = cmd_portal(args)

        assert result == 0
        mock_browser.assert_called_once_with("https://billing.stripe.com/portal/123")


class TestCmdInvoices:
    """Test invoices command."""

    def test_cmd_invoices_list(self, capsys):
        """Test listing invoices."""
        mock_result = {
            "invoices": [
                {"id": "inv_123", "date": "2026-01-15T00:00:00Z", "amount": 2999, "status": "paid"},
                {
                    "id": "inv_456",
                    "date": "2026-02-15T00:00:00Z",
                    "amount": 4999,
                    "status": "pending",
                },
            ]
        }

        args = argparse.Namespace(limit=10, server="http://localhost:8080")

        with patch("aragora.cli.billing.api_request", return_value=mock_result):
            result = cmd_invoices(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "2026-01-15" in captured.out
        assert "29.99" in captured.out
        assert "paid" in captured.out
        assert "inv_123" in captured.out

    def test_cmd_invoices_empty(self, capsys):
        """Test empty invoice list."""
        mock_result = {"invoices": []}

        args = argparse.Namespace(limit=10, server="http://localhost:8080")

        with patch("aragora.cli.billing.api_request", return_value=mock_result):
            result = cmd_invoices(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No invoices found" in captured.out

    def test_cmd_invoices_error(self, capsys):
        """Test invoices with API error."""
        args = argparse.Namespace(limit=10, server="http://localhost:8080")

        with patch(
            "aragora.cli.billing.api_request",
            side_effect=RuntimeError("API error (401): Unauthorized"),
        ):
            result = cmd_invoices(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.out


class TestCreateBillingParser:
    """Test parser creation."""

    def test_parser_has_all_subcommands(self):
        """Test that parser has all expected subcommands."""
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers()

        create_billing_parser(subparsers)

        # Parse with each subcommand to verify they exist
        args = main_parser.parse_args(["billing", "status"])
        assert hasattr(args, "func")

        args = main_parser.parse_args(["billing", "usage"])
        assert hasattr(args, "func")

        args = main_parser.parse_args(["billing", "invoices"])
        assert hasattr(args, "func")

    def test_subscribe_requires_plan(self):
        """Test that subscribe requires --plan argument."""
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers()

        create_billing_parser(subparsers)

        with pytest.raises(SystemExit):
            main_parser.parse_args(["billing", "subscribe"])  # Missing --plan

    def test_subscribe_validates_plan_choices(self):
        """Test that subscribe validates plan choices."""
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers()

        create_billing_parser(subparsers)

        with pytest.raises(SystemExit):
            main_parser.parse_args(["billing", "subscribe", "--plan", "invalid"])


class TestCmdBillingDefault:
    """Test default billing command."""

    def test_default_calls_status(self, capsys):
        """Test that default billing command shows status."""
        mock_result = {
            "plan": {"name": "Free", "status": "active"},
            "current_usage": {"debates": 0, "tokens": 0, "cost_usd": "0"},
            "limits": {"debates": 10, "tokens": 50000},
        }

        args = argparse.Namespace(billing_action=None)

        with patch("aragora.cli.billing.api_request", return_value=mock_result):
            result = cmd_billing_default(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Free" in captured.out
