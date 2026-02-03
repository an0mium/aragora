"""
Tests for aragora.cli.tenant module.

Tests multi-tenant CLI commands including:
- List tenants
- Create tenants
- Delete tenants
- Quota management
- Data export
- Status management (suspend/activate)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from aragora.cli.tenant import (
    api_request,
    cmd_activate,
    cmd_create,
    cmd_delete,
    cmd_export,
    cmd_list,
    cmd_quota_get,
    cmd_quota_set,
    cmd_suspend,
    create_tenant_parser,
    get_api_token,
    main,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_tenants():
    """Sample tenant data for testing."""
    return [
        {
            "id": "tenant_acme_corp_123456",
            "name": "Acme Corp",
            "tier": "professional",
            "status": "active",
        },
        {
            "id": "tenant_startup_xyz_789",
            "name": "Startup XYZ",
            "tier": "starter",
            "status": "active",
        },
        {
            "id": "tenant_suspended_456",
            "name": "Suspended Inc",
            "tier": "enterprise",
            "status": "suspended",
        },
    ]


@pytest.fixture
def mock_quotas():
    """Sample quota data for testing."""
    return {
        "quotas": {
            "max_debates_per_day": 100,
            "max_concurrent_debates": 10,
            "max_agents_per_debate": 5,
            "max_rounds_per_debate": 10,
            "max_users": 50,
            "max_connectors": 5,
            "tokens_per_month": 10000000,
            "tokens_per_debate": 50000,
            "api_requests_per_minute": 100,
            "api_requests_per_day": 10000,
        },
        "usage": {
            "debates_today": 25,
            "tokens_this_month": 2500000,
            "storage_bytes": 52428800,
        },
    }


# =============================================================================
# API Helper Tests
# =============================================================================


class TestGetApiToken:
    """Tests for get_api_token function."""

    def test_returns_token_from_env(self, monkeypatch):
        """Test returns ARAGORA_API_TOKEN."""
        monkeypatch.setenv("ARAGORA_API_TOKEN", "test-token-123")
        assert get_api_token() == "test-token-123"

    def test_returns_key_from_env(self, monkeypatch):
        """Test returns ARAGORA_API_KEY if TOKEN not set."""
        monkeypatch.delenv("ARAGORA_API_TOKEN", raising=False)
        monkeypatch.setenv("ARAGORA_API_KEY", "test-key-456")
        assert get_api_token() == "test-key-456"

    def test_returns_none_if_not_set(self, monkeypatch):
        """Test returns None if no token set."""
        monkeypatch.delenv("ARAGORA_API_TOKEN", raising=False)
        monkeypatch.delenv("ARAGORA_API_KEY", raising=False)
        assert get_api_token() is None


class TestApiRequest:
    """Tests for api_request function."""

    def test_successful_get_request(self):
        """Test successful GET request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"tenants": []}
        mock_response.raise_for_status = MagicMock()

        with patch("aragora.cli.tenant.httpx.request", return_value=mock_response):
            result = api_request("GET", "/api/v1/tenants")

        assert result == {"tenants": []}

    def test_successful_post_request(self):
        """Test successful POST request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"tenant": {"id": "new-tenant"}}
        mock_response.raise_for_status = MagicMock()

        with patch("aragora.cli.tenant.httpx.request", return_value=mock_response):
            result = api_request("POST", "/api/v1/tenants", data={"name": "Test"})

        assert result["tenant"]["id"] == "new-tenant"

    def test_includes_auth_header(self, monkeypatch):
        """Test includes authorization header when token set."""
        monkeypatch.setenv("ARAGORA_API_TOKEN", "test-token")

        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        with patch("aragora.cli.tenant.httpx.request", return_value=mock_response) as mock_req:
            api_request("GET", "/test")

        call_kwargs = mock_req.call_args.kwargs
        assert "Authorization" in call_kwargs["headers"]
        assert "Bearer test-token" in call_kwargs["headers"]["Authorization"]

    def test_http_error_raises_runtime_error(self):
        """Test HTTP errors raise RuntimeError."""
        response = httpx.Response(404, text="Not Found")
        request = httpx.Request("GET", "http://test/api")
        mock_error = httpx.HTTPStatusError("Not Found", request=request, response=response)

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = mock_error
        mock_response.text = "Not Found"

        with patch("aragora.cli.tenant.httpx.request", return_value=mock_response):
            with pytest.raises(RuntimeError) as exc_info:
                api_request("GET", "/test")

        assert "API error" in str(exc_info.value)
        assert "404" in str(exc_info.value)

    def test_connection_error_raises_runtime_error(self):
        """Test connection errors raise RuntimeError."""
        request = httpx.Request("GET", "http://test/api")
        mock_error = httpx.RequestError("Connection refused", request=request)

        with patch("aragora.cli.tenant.httpx.request", side_effect=mock_error):
            with pytest.raises(RuntimeError) as exc_info:
                api_request("GET", "/test")

        assert "Connection error" in str(exc_info.value)


# =============================================================================
# List Tenants Tests
# =============================================================================


class TestCmdList:
    """Tests for cmd_list function."""

    def test_lists_tenants(self, mock_tenants, capsys):
        """Test listing tenants from API."""
        args = argparse.Namespace(
            server="http://localhost:8080",
            status=None,
            tier=None,
            limit=None,
        )

        with patch("aragora.cli.tenant.api_request") as mock_api:
            mock_api.return_value = {"tenants": mock_tenants, "total": 3}
            result = cmd_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "ARAGORA TENANTS" in captured.out
        assert "Total: 3" in captured.out
        assert "Acme Corp" in captured.out
        assert "professional" in captured.out

    def test_lists_empty_tenants(self, capsys):
        """Test listing when no tenants exist."""
        args = argparse.Namespace(
            server="http://localhost:8080",
            status=None,
            tier=None,
            limit=None,
        )

        with patch("aragora.cli.tenant.api_request") as mock_api:
            mock_api.return_value = {"tenants": [], "total": 0}
            result = cmd_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No tenants found" in captured.out

    def test_lists_with_status_filter(self, mock_tenants):
        """Test listing with status filter."""
        args = argparse.Namespace(
            server="http://localhost:8080",
            status="active",
            tier=None,
            limit=None,
        )

        with patch("aragora.cli.tenant.api_request") as mock_api:
            mock_api.return_value = {"tenants": mock_tenants[:2], "total": 2}
            cmd_list(args)

        call_args = mock_api.call_args[1]
        assert "status=active" in call_args.get("server_url", "") or mock_api.call_args[0][1]

    def test_lists_with_tier_filter(self, mock_tenants):
        """Test listing with tier filter."""
        args = argparse.Namespace(
            server="http://localhost:8080",
            status=None,
            tier="professional",
            limit=None,
        )

        with patch("aragora.cli.tenant.api_request") as mock_api:
            mock_api.return_value = {"tenants": [mock_tenants[0]], "total": 1}
            cmd_list(args)

        # Verify API was called
        mock_api.assert_called_once()

    def test_falls_back_to_local_on_connection_error(self, capsys):
        """Test falls back to local mode on connection error."""
        args = argparse.Namespace(
            server="http://localhost:8080",
            status=None,
            tier=None,
            limit=None,
        )

        with patch("aragora.cli.tenant.api_request", side_effect=RuntimeError("Connection error")):
            with patch("aragora.cli.tenant.cmd_list_local", return_value=0) as mock_local:
                result = cmd_list(args)

        mock_local.assert_called_once()
        assert result == 0


# =============================================================================
# Create Tenant Tests
# =============================================================================


class TestCmdCreate:
    """Tests for cmd_create function."""

    def test_creates_tenant(self, capsys):
        """Test creating a new tenant."""
        args = argparse.Namespace(
            server="http://localhost:8080",
            name="New Tenant",
            tier="starter",
            domain=None,
            admin_email=None,
        )

        mock_response = {
            "tenant": {
                "id": "tenant_new_123",
                "name": "New Tenant",
                "tier": "starter",
                "status": "active",
                "api_key": "sk-new-tenant-key-abc123",
            }
        }

        with patch("aragora.cli.tenant.api_request", return_value=mock_response):
            result = cmd_create(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Tenant created successfully" in captured.out
        assert "tenant_new_123" in captured.out
        assert "New Tenant" in captured.out
        assert "sk-new-tenant-key-abc123" in captured.out

    def test_creates_tenant_with_domain(self, capsys):
        """Test creating tenant with custom domain."""
        args = argparse.Namespace(
            server="http://localhost:8080",
            name="Enterprise Corp",
            tier="enterprise",
            domain="enterprise.example.com",
            admin_email="admin@enterprise.com",
        )

        mock_response = {
            "tenant": {
                "id": "tenant_enterprise",
                "name": "Enterprise Corp",
                "tier": "enterprise",
                "status": "active",
            }
        }

        with patch("aragora.cli.tenant.api_request") as mock_api:
            mock_api.return_value = mock_response
            result = cmd_create(args)

        assert result == 0
        call_data = mock_api.call_args.kwargs["data"]
        assert call_data["domain"] == "enterprise.example.com"
        assert call_data["admin_email"] == "admin@enterprise.com"

    def test_falls_back_to_local_on_connection_error(self, capsys):
        """Test falls back to local mode on connection error."""
        args = argparse.Namespace(
            server="http://localhost:8080",
            name="Test",
            tier="starter",
            domain=None,
            admin_email=None,
        )

        with patch("aragora.cli.tenant.api_request", side_effect=RuntimeError("Connection error")):
            with patch("aragora.cli.tenant.cmd_create_local", return_value=0) as mock_local:
                result = cmd_create(args)

        mock_local.assert_called_once()


# =============================================================================
# Delete Tenant Tests
# =============================================================================


class TestCmdDelete:
    """Tests for cmd_delete function."""

    def test_deletes_tenant_with_force(self, capsys):
        """Test deleting tenant with --force flag."""
        args = argparse.Namespace(
            server="http://localhost:8080",
            tenant="tenant_to_delete",
            force=True,
        )

        with patch("aragora.cli.tenant.api_request") as mock_api:
            mock_api.return_value = {}
            result = cmd_delete(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "deleted successfully" in captured.out
        mock_api.assert_called_once()

    def test_deletes_tenant_with_confirmation(self, capsys, monkeypatch):
        """Test deleting tenant with confirmation."""
        args = argparse.Namespace(
            server="http://localhost:8080",
            tenant="tenant_to_delete",
            force=False,
        )

        monkeypatch.setattr("builtins.input", lambda _: "y")

        with patch("aragora.cli.tenant.api_request") as mock_api:
            mock_api.return_value = {}
            result = cmd_delete(args)

        assert result == 0

    def test_cancels_delete_without_confirmation(self, capsys, monkeypatch):
        """Test cancel delete when not confirmed."""
        args = argparse.Namespace(
            server="http://localhost:8080",
            tenant="tenant_to_delete",
            force=False,
        )

        monkeypatch.setattr("builtins.input", lambda _: "n")

        with patch("aragora.cli.tenant.api_request") as mock_api:
            result = cmd_delete(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Cancelled" in captured.out
        mock_api.assert_not_called()


# =============================================================================
# Quota Management Tests
# =============================================================================


class TestCmdQuotaGet:
    """Tests for cmd_quota_get function."""

    def test_gets_quotas(self, mock_quotas, capsys):
        """Test getting tenant quotas."""
        args = argparse.Namespace(
            server="http://localhost:8080",
            tenant="tenant_123",
        )

        with patch("aragora.cli.tenant.api_request", return_value=mock_quotas):
            result = cmd_quota_get(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "QUOTAS FOR: tenant_123" in captured.out
        assert "Debates/day:" in captured.out
        assert "100" in captured.out
        assert "Tokens/month:" in captured.out
        assert "10,000,000" in captured.out

    def test_shows_usage(self, mock_quotas, capsys):
        """Test shows current usage."""
        args = argparse.Namespace(
            server="http://localhost:8080",
            tenant="tenant_123",
        )

        with patch("aragora.cli.tenant.api_request", return_value=mock_quotas):
            cmd_quota_get(args)

        captured = capsys.readouterr()
        assert "Current Usage:" in captured.out
        assert "Debates today:" in captured.out
        assert "25" in captured.out


class TestCmdQuotaSet:
    """Tests for cmd_quota_set function."""

    def test_sets_debates_quota(self, capsys):
        """Test setting debates quota."""
        args = argparse.Namespace(
            server="http://localhost:8080",
            tenant="tenant_123",
            debates=200,
            concurrent=None,
            agents=None,
            rounds=None,
            users=None,
            tokens_month=None,
            tokens_debate=None,
        )

        with patch("aragora.cli.tenant.api_request") as mock_api:
            mock_api.return_value = {}
            result = cmd_quota_set(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Quotas updated" in captured.out
        assert "max_debates_per_day: 200" in captured.out

    def test_sets_multiple_quotas(self, capsys):
        """Test setting multiple quotas."""
        args = argparse.Namespace(
            server="http://localhost:8080",
            tenant="tenant_123",
            debates=200,
            concurrent=15,
            agents=6,
            rounds=12,
            users=100,
            tokens_month=20000000,
            tokens_debate=100000,
        )

        with patch("aragora.cli.tenant.api_request") as mock_api:
            mock_api.return_value = {}
            cmd_quota_set(args)

        call_data = mock_api.call_args.kwargs["data"]
        assert call_data["max_debates_per_day"] == 200
        assert call_data["max_concurrent_debates"] == 15
        assert call_data["tokens_per_month"] == 20000000

    def test_no_quotas_specified(self, capsys):
        """Test error when no quotas specified."""
        args = argparse.Namespace(
            server="http://localhost:8080",
            tenant="tenant_123",
            debates=None,
            concurrent=None,
            agents=None,
            rounds=None,
            users=None,
            tokens_month=None,
            tokens_debate=None,
        )

        result = cmd_quota_set(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "No quotas specified" in captured.out


# =============================================================================
# Export Tests
# =============================================================================


class TestCmdExport:
    """Tests for cmd_export function."""

    def test_exports_tenant_data(self, tmp_path, capsys):
        """Test exporting tenant data."""
        output_file = tmp_path / "export.json"
        args = argparse.Namespace(
            server="http://localhost:8080",
            tenant="tenant_123",
            output=str(output_file),
            format="json",
            include_debates=False,
            include_knowledge=False,
            include_audit=False,
        )

        mock_response = {
            "tenant": {"id": "tenant_123", "name": "Test"},
            "statistics": {"debates": 100, "documents": 50, "users": 10},
        }

        with patch("aragora.cli.tenant.api_request", return_value=mock_response):
            result = cmd_export(args)

        assert result == 0
        assert output_file.exists()
        captured = capsys.readouterr()
        assert "Export complete" in captured.out

    def test_exports_with_all_includes(self, tmp_path, capsys):
        """Test export with all include flags."""
        output_file = tmp_path / "full_export.json"
        args = argparse.Namespace(
            server="http://localhost:8080",
            tenant="tenant_123",
            output=str(output_file),
            format="json",
            include_debates=True,
            include_knowledge=True,
            include_audit=True,
        )

        mock_response = {"tenant": {}, "statistics": {}}

        with patch("aragora.cli.tenant.api_request") as mock_api:
            mock_api.return_value = mock_response
            cmd_export(args)

        # Verify query params include the flags
        call_path = mock_api.call_args[0][1]
        assert "include_debates=true" in call_path
        assert "include_knowledge=true" in call_path
        assert "include_audit=true" in call_path


# =============================================================================
# Status Management Tests
# =============================================================================


class TestCmdSuspend:
    """Tests for cmd_suspend function."""

    def test_suspends_tenant(self, capsys):
        """Test suspending a tenant."""
        args = argparse.Namespace(
            server="http://localhost:8080",
            tenant="tenant_123",
            reason="Non-payment",
        )

        with patch("aragora.cli.tenant.api_request") as mock_api:
            mock_api.return_value = {}
            result = cmd_suspend(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "suspended" in captured.out
        assert "Non-payment" in captured.out

    def test_suspends_with_default_reason(self, capsys):
        """Test suspending with default reason."""
        args = argparse.Namespace(
            server="http://localhost:8080",
            tenant="tenant_123",
            reason=None,
        )

        with patch("aragora.cli.tenant.api_request") as mock_api:
            mock_api.return_value = {}
            cmd_suspend(args)

        call_data = mock_api.call_args.kwargs["data"]
        assert call_data["reason"] == "Administrative action"


class TestCmdActivate:
    """Tests for cmd_activate function."""

    def test_activates_tenant(self, capsys):
        """Test activating a suspended tenant."""
        args = argparse.Namespace(
            server="http://localhost:8080",
            tenant="tenant_123",
        )

        with patch("aragora.cli.tenant.api_request") as mock_api:
            mock_api.return_value = {}
            result = cmd_activate(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "activated" in captured.out

    def test_activate_api_error(self, capsys):
        """Test activate handles API error."""
        args = argparse.Namespace(
            server="http://localhost:8080",
            tenant="nonexistent",
        )

        with patch("aragora.cli.tenant.api_request", side_effect=RuntimeError("Not found")):
            result = cmd_activate(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.out


# =============================================================================
# Parser Tests
# =============================================================================


class TestCreateTenantParser:
    """Tests for create_tenant_parser function."""

    def test_creates_parser(self):
        """Test parser is created."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        create_tenant_parser(subparsers)

        # Should be able to parse tenant list
        args = parser.parse_args(["tenant", "list"])
        assert args.tenant_action == "list"

    def test_parses_create_command(self):
        """Test parsing create command."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_tenant_parser(subparsers)

        args = parser.parse_args(
            ["tenant", "create", "--name", "Test Corp", "--tier", "professional"]
        )

        assert args.tenant_action == "create"
        assert args.name == "Test Corp"
        assert args.tier == "professional"

    def test_parses_quota_set_command(self):
        """Test parsing quota-set command."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_tenant_parser(subparsers)

        args = parser.parse_args(
            [
                "tenant",
                "quota-set",
                "--tenant",
                "t123",
                "--debates",
                "500",
                "--tokens-month",
                "50000000",
            ]
        )

        assert args.tenant_action == "quota-set"
        assert args.tenant == "t123"
        assert args.debates == 500
        assert args.tokens_month == 50000000

    def test_parses_export_command(self):
        """Test parsing export command."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_tenant_parser(subparsers)

        args = parser.parse_args(
            [
                "tenant",
                "export",
                "--tenant",
                "t123",
                "--output",
                "data.json",
                "--format",
                "json",
                "--include-debates",
                "--include-audit",
            ]
        )

        assert args.tenant_action == "export"
        assert args.tenant == "t123"
        assert args.output == "data.json"
        assert args.format == "json"
        assert args.include_debates is True
        assert args.include_audit is True


# =============================================================================
# Main Entry Point Tests
# =============================================================================


class TestMain:
    """Tests for main function."""

    def test_main_default_to_list(self):
        """Test main defaults to list command."""
        args = argparse.Namespace(
            tenant_action=None,
            server="http://localhost:8080",
            status=None,
            tier=None,
            limit=None,
        )

        with patch("aragora.cli.tenant.cmd_list", return_value=0) as mock_list:
            result = main(args)

        mock_list.assert_called_once()
        assert result == 0

    def test_main_dispatches_to_handler(self):
        """Test main dispatches to correct handler."""
        mock_handler = MagicMock(return_value=0)
        args = argparse.Namespace(
            tenant_action="create",
            func=mock_handler,
        )

        result = main(args)

        mock_handler.assert_called_once_with(args)
        assert result == 0

    def test_main_unknown_action(self, capsys):
        """Test main handles unknown action."""
        args = argparse.Namespace(tenant_action="unknown")
        # Don't set func attribute

        result = main(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown tenant action" in captured.out
