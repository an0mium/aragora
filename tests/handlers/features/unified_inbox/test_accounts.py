"""Tests for the Unified Inbox accounts module.

Covers all functions in aragora/server/handlers/features/unified_inbox/accounts.py:
- handle_gmail_oauth_url   - Generate Gmail OAuth authorization URL
- handle_outlook_oauth_url - Generate Outlook OAuth authorization URL
- connect_gmail            - Connect Gmail account via OAuth
- connect_outlook          - Connect Outlook account via OAuth
- disconnect_account       - Stop and remove sync service for an account

Tests include:
- Happy paths for all functions
- Parameter validation (missing redirect_uri, missing auth_code, etc.)
- ImportError fallback paths (connector not available / mock mode)
- Connector misconfiguration (not configured)
- Authentication failures
- Missing refresh tokens
- Sync service registry management
- Token persistence (success and failure)
- Profile extraction edge cases
- Connection errors (ConnectionError, TimeoutError, OSError, ValueError)
- Disconnect with running sync service
- Disconnect with no sync service
- Disconnect when sync service stop() raises
- Security tests (path traversal in params, injection in state)
- Edge cases (empty strings, None values, unicode)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from aragora.server.handlers.features.unified_inbox.models import (
    AccountStatus,
    ConnectedAccount,
    EmailProvider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)


def _make_account(
    account_id: str = "acct-test-001",
    provider: EmailProvider = EmailProvider.GMAIL,
    status: AccountStatus = AccountStatus.PENDING,
    email: str = "",
) -> ConnectedAccount:
    """Create a ConnectedAccount for testing."""
    return ConnectedAccount(
        id=account_id,
        provider=provider,
        email_address=email,
        display_name="",
        status=status,
        connected_at=_NOW,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_sync_services():
    """Clear the global sync services registry between tests."""
    from aragora.server.handlers.features.unified_inbox.sync import (
        _sync_services,
    )

    _sync_services.clear()
    yield
    _sync_services.clear()


@pytest.fixture
def mock_gmail_connector():
    """Create a mock GmailConnector."""
    connector = MagicMock()
    connector.is_configured = True
    connector.get_oauth_url.return_value = "https://accounts.google.com/o/oauth2/auth?client_id=xxx"
    connector.authenticate = AsyncMock(return_value=True)
    connector.refresh_token = "gmail-refresh-token-xyz"
    connector.access_token = "gmail-access-token-xyz"
    connector.token_expiry = _NOW
    connector.get_user_info = AsyncMock(
        return_value={"emailAddress": "user@gmail.com"}
    )
    return connector


@pytest.fixture
def mock_outlook_connector():
    """Create a mock OutlookConnector."""
    connector = MagicMock()
    connector.is_configured = True
    connector.get_oauth_url.return_value = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize?client_id=xxx"
    connector.authenticate = AsyncMock(return_value=True)
    connector.refresh_token = "outlook-refresh-token-xyz"
    connector.access_token = "outlook-access-token-xyz"
    connector.token_expiry = _NOW
    connector.get_user_info = AsyncMock(
        return_value={"mail": "user@outlook.com", "userPrincipalName": "user@outlook.com"}
    )
    return connector


@pytest.fixture
def mock_schedule_persist():
    """Create a mock schedule_message_persist callback."""
    return MagicMock()


# ===========================================================================
# handle_gmail_oauth_url
# ===========================================================================


class TestHandleGmailOAuthUrl:
    """Tests for handle_gmail_oauth_url."""

    @pytest.mark.asyncio
    async def test_success_returns_auth_url(self, mock_gmail_connector):
        mock_module = MagicMock()
        mock_module.GmailConnector = MagicMock(return_value=mock_gmail_connector)
        with patch.dict("sys.modules", {
            "aragora.connectors.enterprise.communication.gmail": mock_module,
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                handle_gmail_oauth_url,
            )

            result = await handle_gmail_oauth_url(
                {"redirect_uri": "http://localhost/callback", "state": "my-state"},
                "tenant-1",
            )
            assert result["success"] is True
            assert "auth_url" in result["data"]
            assert result["data"]["provider"] == "gmail"
            assert result["data"]["state"] == "my-state"

    @pytest.mark.asyncio
    async def test_success_generates_state_when_not_provided(self, mock_gmail_connector):
        mock_module = MagicMock()
        mock_module.GmailConnector = MagicMock(return_value=mock_gmail_connector)
        with patch.dict("sys.modules", {
            "aragora.connectors.enterprise.communication.gmail": mock_module,
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                handle_gmail_oauth_url,
            )

            result = await handle_gmail_oauth_url(
                {"redirect_uri": "http://localhost/callback"},
                "tenant-1",
            )
            assert result["success"] is True
            # State should be a UUID string
            assert len(result["data"]["state"]) > 0

    @pytest.mark.asyncio
    async def test_missing_redirect_uri_returns_400(self):
        from aragora.server.handlers.features.unified_inbox.accounts import (
            handle_gmail_oauth_url,
        )

        result = await handle_gmail_oauth_url({}, "tenant-1")
        assert result["success"] is False
        assert result["status_code"] == 400
        assert "redirect_uri" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_redirect_uri_returns_400(self):
        from aragora.server.handlers.features.unified_inbox.accounts import (
            handle_gmail_oauth_url,
        )

        result = await handle_gmail_oauth_url({"redirect_uri": ""}, "tenant-1")
        assert result["success"] is False
        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_connector_not_configured_returns_503(self):
        mock_connector = MagicMock()
        mock_connector.is_configured = False
        mock_module = MagicMock()
        mock_module.GmailConnector = MagicMock(return_value=mock_connector)
        with patch.dict("sys.modules", {
            "aragora.connectors.enterprise.communication.gmail": mock_module,
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                handle_gmail_oauth_url,
            )

            result = await handle_gmail_oauth_url(
                {"redirect_uri": "http://localhost/callback"},
                "tenant-1",
            )
            assert result["success"] is False
            assert result["status_code"] == 503
            assert "not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_import_error_returns_503(self):
        from aragora.server.handlers.features.unified_inbox import accounts as mod

        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def fake_import(name, *args, **kwargs):
            if "gmail" in name.lower():
                raise ImportError("Gmail connector not available")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = await mod.handle_gmail_oauth_url(
                {"redirect_uri": "http://localhost/callback"},
                "tenant-1",
            )
            assert result["success"] is False
            assert result["status_code"] == 503
            assert "not available" in result["error"]

    @pytest.mark.asyncio
    async def test_state_with_special_characters(self, mock_gmail_connector):
        mock_module = MagicMock()
        mock_module.GmailConnector = MagicMock(return_value=mock_gmail_connector)
        with patch.dict("sys.modules", {
            "aragora.connectors.enterprise.communication.gmail": mock_module,
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                handle_gmail_oauth_url,
            )

            result = await handle_gmail_oauth_url(
                {"redirect_uri": "http://localhost/callback", "state": "state-with-special/chars&="},
                "tenant-1",
            )
            assert result["success"] is True
            assert result["data"]["state"] == "state-with-special/chars&="


# ===========================================================================
# handle_outlook_oauth_url
# ===========================================================================


class TestHandleOutlookOAuthUrl:
    """Tests for handle_outlook_oauth_url."""

    @pytest.mark.asyncio
    async def test_success_returns_auth_url(self, mock_outlook_connector):
        mock_module = MagicMock()
        mock_module.OutlookConnector = MagicMock(return_value=mock_outlook_connector)
        with patch.dict("sys.modules", {
            "aragora.connectors.enterprise.communication.outlook": mock_module,
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                handle_outlook_oauth_url,
            )

            result = await handle_outlook_oauth_url(
                {"redirect_uri": "http://localhost/callback", "state": "my-state"},
                "tenant-1",
            )
            assert result["success"] is True
            assert "auth_url" in result["data"]
            assert result["data"]["provider"] == "outlook"
            assert result["data"]["state"] == "my-state"

    @pytest.mark.asyncio
    async def test_success_generates_state_when_not_provided(self, mock_outlook_connector):
        mock_module = MagicMock()
        mock_module.OutlookConnector = MagicMock(return_value=mock_outlook_connector)
        with patch.dict("sys.modules", {
            "aragora.connectors.enterprise.communication.outlook": mock_module,
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                handle_outlook_oauth_url,
            )

            result = await handle_outlook_oauth_url(
                {"redirect_uri": "http://localhost/callback"},
                "tenant-1",
            )
            assert result["success"] is True
            assert len(result["data"]["state"]) > 0

    @pytest.mark.asyncio
    async def test_missing_redirect_uri_returns_400(self):
        from aragora.server.handlers.features.unified_inbox.accounts import (
            handle_outlook_oauth_url,
        )

        result = await handle_outlook_oauth_url({}, "tenant-1")
        assert result["success"] is False
        assert result["status_code"] == 400
        assert "redirect_uri" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_redirect_uri_returns_400(self):
        from aragora.server.handlers.features.unified_inbox.accounts import (
            handle_outlook_oauth_url,
        )

        result = await handle_outlook_oauth_url({"redirect_uri": ""}, "tenant-1")
        assert result["success"] is False
        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_connector_not_configured_returns_503(self):
        mock_connector = MagicMock()
        mock_connector.is_configured = False
        mock_module = MagicMock()
        mock_module.OutlookConnector = MagicMock(return_value=mock_connector)
        with patch.dict("sys.modules", {
            "aragora.connectors.enterprise.communication.outlook": mock_module,
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                handle_outlook_oauth_url,
            )

            result = await handle_outlook_oauth_url(
                {"redirect_uri": "http://localhost/callback"},
                "tenant-1",
            )
            assert result["success"] is False
            assert result["status_code"] == 503
            assert "not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_import_error_returns_503(self):
        from aragora.server.handlers.features.unified_inbox import accounts as mod

        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def fake_import(name, *args, **kwargs):
            if "outlook" in name.lower():
                raise ImportError("Outlook connector not available")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = await mod.handle_outlook_oauth_url(
                {"redirect_uri": "http://localhost/callback"},
                "tenant-1",
            )
            assert result["success"] is False
            assert result["status_code"] == 503
            assert "not available" in result["error"]


# ===========================================================================
# connect_gmail
# ===========================================================================


class TestConnectGmail:
    """Tests for connect_gmail."""

    @pytest.mark.asyncio
    async def test_success_full_flow(self, mock_gmail_connector, mock_schedule_persist):
        """Full happy path: auth, profile, sync service, token persist, start."""
        mock_sync_service = AsyncMock()
        mock_sync_service.start = AsyncMock()

        account = _make_account(provider=EmailProvider.GMAIL)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                GmailSyncService=MagicMock(return_value=mock_sync_service),
                GmailSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.gmail": MagicMock(
                GmailConnector=MagicMock(return_value=mock_gmail_connector),
            ),
            "aragora.storage.gmail_token_store": MagicMock(
                GmailUserState=MagicMock(),
                get_gmail_token_store=MagicMock(return_value=AsyncMock()),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_gmail,
            )

            result = await connect_gmail(
                account=account,
                auth_code="auth-code-123",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is True
            assert account.status == AccountStatus.CONNECTED
            assert account.email_address == "user@gmail.com"
            assert account.display_name == "user"

    @pytest.mark.asyncio
    async def test_auth_failure_returns_error(self, mock_schedule_persist):
        """When connector.authenticate returns False."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=False)

        account = _make_account(provider=EmailProvider.GMAIL)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                GmailSyncService=MagicMock(),
                GmailSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.gmail": MagicMock(
                GmailConnector=MagicMock(return_value=mock_connector),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_gmail,
            )

            result = await connect_gmail(
                account=account,
                auth_code="bad-code",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is False
            assert "authentication failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_missing_refresh_token_returns_error(self, mock_schedule_persist):
        """When connector has no refresh token after auth."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.refresh_token = ""

        account = _make_account(provider=EmailProvider.GMAIL)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                GmailSyncService=MagicMock(),
                GmailSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.gmail": MagicMock(
                GmailConnector=MagicMock(return_value=mock_connector),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_gmail,
            )

            result = await connect_gmail(
                account=account,
                auth_code="auth-code-123",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is False
            assert "refresh token" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_none_refresh_token_returns_error(self, mock_schedule_persist):
        """When connector.refresh_token is None."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.refresh_token = None

        account = _make_account(provider=EmailProvider.GMAIL)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                GmailSyncService=MagicMock(),
                GmailSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.gmail": MagicMock(
                GmailConnector=MagicMock(return_value=mock_connector),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_gmail,
            )

            result = await connect_gmail(
                account=account,
                auth_code="auth-code-123",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is False
            assert "refresh token" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_import_error_falls_back_to_mock_mode(self, mock_schedule_persist):
        """When GmailSyncService import fails, should use mock mode."""
        account = _make_account(account_id="abcd1234-rest-of-id", provider=EmailProvider.GMAIL)

        # Patch builtins to fail on gmail imports
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def fake_import(name, *args, **kwargs):
            if "aragora.connectors.email" in name or "aragora.connectors.enterprise.communication.gmail" in name:
                raise ImportError("Not available")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_gmail,
            )

            result = await connect_gmail(
                account=account,
                auth_code="auth-code-123",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is True
            assert account.status == AccountStatus.CONNECTED
            assert account.display_name == "Gmail User"
            assert "gmail.com" in account.email_address

    @pytest.mark.asyncio
    async def test_connection_error_returns_failure(self, mock_schedule_persist):
        """When a ConnectionError occurs during connect."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(side_effect=ConnectionError("refused"))

        account = _make_account(provider=EmailProvider.GMAIL)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                GmailSyncService=MagicMock(),
                GmailSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.gmail": MagicMock(
                GmailConnector=MagicMock(return_value=mock_connector),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_gmail,
            )

            result = await connect_gmail(
                account=account,
                auth_code="auth-code-123",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is False
            assert "connection failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_timeout_error_returns_failure(self, mock_schedule_persist):
        """When a TimeoutError occurs during connect."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(side_effect=TimeoutError("timed out"))

        account = _make_account(provider=EmailProvider.GMAIL)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                GmailSyncService=MagicMock(),
                GmailSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.gmail": MagicMock(
                GmailConnector=MagicMock(return_value=mock_connector),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_gmail,
            )

            result = await connect_gmail(
                account=account,
                auth_code="auth-code-123",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is False
            assert "connection failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_os_error_returns_failure(self, mock_schedule_persist):
        """When an OSError occurs during connect."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(side_effect=OSError("disk error"))

        account = _make_account(provider=EmailProvider.GMAIL)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                GmailSyncService=MagicMock(),
                GmailSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.gmail": MagicMock(
                GmailConnector=MagicMock(return_value=mock_connector),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_gmail,
            )

            result = await connect_gmail(
                account=account,
                auth_code="auth-code-123",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is False
            assert "connection failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_value_error_returns_failure(self, mock_schedule_persist):
        """When a ValueError occurs during connect."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(side_effect=ValueError("bad value"))

        account = _make_account(provider=EmailProvider.GMAIL)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                GmailSyncService=MagicMock(),
                GmailSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.gmail": MagicMock(
                GmailConnector=MagicMock(return_value=mock_connector),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_gmail,
            )

            result = await connect_gmail(
                account=account,
                auth_code="auth-code-123",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is False
            assert "connection failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_profile_with_empty_email_uses_default_name(self, mock_schedule_persist):
        """When profile returns empty emailAddress."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.refresh_token = "refresh-token"
        mock_connector.access_token = "access-token"
        mock_connector.token_expiry = _NOW
        mock_connector.get_user_info = AsyncMock(return_value={"emailAddress": ""})

        mock_sync_service = AsyncMock()
        mock_sync_service.start = AsyncMock()

        account = _make_account(provider=EmailProvider.GMAIL, email="")

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                GmailSyncService=MagicMock(return_value=mock_sync_service),
                GmailSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.gmail": MagicMock(
                GmailConnector=MagicMock(return_value=mock_connector),
            ),
            "aragora.storage.gmail_token_store": MagicMock(
                GmailUserState=MagicMock(),
                get_gmail_token_store=MagicMock(return_value=AsyncMock()),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_gmail,
            )

            result = await connect_gmail(
                account=account,
                auth_code="auth-code-123",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is True
            assert account.display_name == "Gmail User"

    @pytest.mark.asyncio
    async def test_token_persist_failure_is_non_fatal(self, mock_schedule_persist):
        """When token store save fails, connect should still succeed."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.refresh_token = "refresh-token"
        mock_connector.access_token = "access-token"
        mock_connector.token_expiry = _NOW
        mock_connector.get_user_info = AsyncMock(return_value={"emailAddress": "user@gmail.com"})

        mock_sync_service = AsyncMock()
        mock_sync_service.start = AsyncMock()

        mock_token_store = AsyncMock()
        mock_token_store.save = AsyncMock(side_effect=OSError("disk full"))

        account = _make_account(provider=EmailProvider.GMAIL)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                GmailSyncService=MagicMock(return_value=mock_sync_service),
                GmailSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.gmail": MagicMock(
                GmailConnector=MagicMock(return_value=mock_connector),
            ),
            "aragora.storage.gmail_token_store": MagicMock(
                GmailUserState=MagicMock(),
                get_gmail_token_store=MagicMock(return_value=mock_token_store),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_gmail,
            )

            result = await connect_gmail(
                account=account,
                auth_code="auth-code-123",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            # Token persist failure should not block connect
            assert result["success"] is True
            assert account.status == AccountStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_sync_service_registered_in_global_dict(self, mock_schedule_persist):
        """Verify sync service is stored in global registry."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.refresh_token = "refresh-token"
        mock_connector.access_token = "access-token"
        mock_connector.token_expiry = _NOW
        mock_connector.get_user_info = AsyncMock(return_value={"emailAddress": "user@gmail.com"})

        mock_sync_service = AsyncMock()
        mock_sync_service.start = AsyncMock()

        account = _make_account(account_id="acct-sync-test", provider=EmailProvider.GMAIL)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                GmailSyncService=MagicMock(return_value=mock_sync_service),
                GmailSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.gmail": MagicMock(
                GmailConnector=MagicMock(return_value=mock_connector),
            ),
            "aragora.storage.gmail_token_store": MagicMock(
                GmailUserState=MagicMock(),
                get_gmail_token_store=MagicMock(return_value=AsyncMock()),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_gmail,
            )
            from aragora.server.handlers.features.unified_inbox.sync import (
                get_sync_services,
            )

            result = await connect_gmail(
                account=account,
                auth_code="auth-code-123",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-reg",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is True
            services = get_sync_services()
            assert "tenant-reg" in services
            assert "acct-sync-test" in services["tenant-reg"]


# ===========================================================================
# connect_outlook
# ===========================================================================


class TestConnectOutlook:
    """Tests for connect_outlook."""

    @pytest.mark.asyncio
    async def test_success_full_flow(self, mock_outlook_connector, mock_schedule_persist):
        """Full happy path: auth, profile, sync service, token persist, start."""
        mock_sync_service = AsyncMock()
        mock_sync_service.start = AsyncMock()

        account = _make_account(provider=EmailProvider.OUTLOOK)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                OutlookSyncService=MagicMock(return_value=mock_sync_service),
                OutlookSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.outlook": MagicMock(
                OutlookConnector=MagicMock(return_value=mock_outlook_connector),
            ),
            "aragora.storage.integration_store": MagicMock(
                IntegrationConfig=MagicMock(),
                get_integration_store=MagicMock(return_value=AsyncMock()),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_outlook,
            )

            result = await connect_outlook(
                account=account,
                auth_code="auth-code-456",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is True
            assert account.status == AccountStatus.CONNECTED
            assert account.email_address == "user@outlook.com"
            assert account.display_name == "user"

    @pytest.mark.asyncio
    async def test_auth_failure_returns_error(self, mock_schedule_persist):
        """When connector.authenticate returns False."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=False)

        account = _make_account(provider=EmailProvider.OUTLOOK)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                OutlookSyncService=MagicMock(),
                OutlookSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.outlook": MagicMock(
                OutlookConnector=MagicMock(return_value=mock_connector),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_outlook,
            )

            result = await connect_outlook(
                account=account,
                auth_code="bad-code",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is False
            assert "authentication failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_missing_refresh_token_returns_error(self, mock_schedule_persist):
        """When connector has no refresh token after auth."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.refresh_token = ""

        account = _make_account(provider=EmailProvider.OUTLOOK)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                OutlookSyncService=MagicMock(),
                OutlookSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.outlook": MagicMock(
                OutlookConnector=MagicMock(return_value=mock_connector),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_outlook,
            )

            result = await connect_outlook(
                account=account,
                auth_code="auth-code-456",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is False
            assert "refresh token" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_none_refresh_token_returns_error(self, mock_schedule_persist):
        """When connector.refresh_token is None."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.refresh_token = None

        account = _make_account(provider=EmailProvider.OUTLOOK)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                OutlookSyncService=MagicMock(),
                OutlookSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.outlook": MagicMock(
                OutlookConnector=MagicMock(return_value=mock_connector),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_outlook,
            )

            result = await connect_outlook(
                account=account,
                auth_code="auth-code-456",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is False
            assert "refresh token" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_import_error_falls_back_to_mock_mode(self, mock_schedule_persist):
        """When OutlookSyncService import fails, should use mock mode."""
        account = _make_account(account_id="abcd1234-rest-of-id", provider=EmailProvider.OUTLOOK)

        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def fake_import(name, *args, **kwargs):
            if "aragora.connectors.email" in name or "aragora.connectors.enterprise.communication.outlook" in name:
                raise ImportError("Not available")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_outlook,
            )

            result = await connect_outlook(
                account=account,
                auth_code="auth-code-456",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is True
            assert account.status == AccountStatus.CONNECTED
            assert account.display_name == "Outlook User"
            assert "outlook.com" in account.email_address

    @pytest.mark.asyncio
    async def test_connection_error_returns_failure(self, mock_schedule_persist):
        """When a ConnectionError occurs during connect."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(side_effect=ConnectionError("refused"))

        account = _make_account(provider=EmailProvider.OUTLOOK)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                OutlookSyncService=MagicMock(),
                OutlookSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.outlook": MagicMock(
                OutlookConnector=MagicMock(return_value=mock_connector),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_outlook,
            )

            result = await connect_outlook(
                account=account,
                auth_code="auth-code-456",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is False
            assert "connection failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_timeout_error_returns_failure(self, mock_schedule_persist):
        """When a TimeoutError occurs during connect."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(side_effect=TimeoutError("timed out"))

        account = _make_account(provider=EmailProvider.OUTLOOK)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                OutlookSyncService=MagicMock(),
                OutlookSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.outlook": MagicMock(
                OutlookConnector=MagicMock(return_value=mock_connector),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_outlook,
            )

            result = await connect_outlook(
                account=account,
                auth_code="auth-code-456",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is False
            assert "connection failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_os_error_returns_failure(self, mock_schedule_persist):
        """When an OSError occurs during connect."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(side_effect=OSError("disk error"))

        account = _make_account(provider=EmailProvider.OUTLOOK)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                OutlookSyncService=MagicMock(),
                OutlookSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.outlook": MagicMock(
                OutlookConnector=MagicMock(return_value=mock_connector),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_outlook,
            )

            result = await connect_outlook(
                account=account,
                auth_code="auth-code-456",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is False
            assert "connection failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_value_error_returns_failure(self, mock_schedule_persist):
        """When a ValueError occurs during connect."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(side_effect=ValueError("bad value"))

        account = _make_account(provider=EmailProvider.OUTLOOK)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                OutlookSyncService=MagicMock(),
                OutlookSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.outlook": MagicMock(
                OutlookConnector=MagicMock(return_value=mock_connector),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_outlook,
            )

            result = await connect_outlook(
                account=account,
                auth_code="auth-code-456",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is False
            assert "connection failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_profile_uses_user_principal_name_fallback(self, mock_schedule_persist):
        """When profile has no 'mail' but has 'userPrincipalName'."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.refresh_token = "refresh-token"
        mock_connector.access_token = "access-token"
        mock_connector.token_expiry = _NOW
        mock_connector.get_user_info = AsyncMock(
            return_value={"mail": "", "userPrincipalName": "principal@outlook.com"}
        )

        mock_sync_service = AsyncMock()
        mock_sync_service.start = AsyncMock()

        account = _make_account(provider=EmailProvider.OUTLOOK)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                OutlookSyncService=MagicMock(return_value=mock_sync_service),
                OutlookSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.outlook": MagicMock(
                OutlookConnector=MagicMock(return_value=mock_connector),
            ),
            "aragora.storage.integration_store": MagicMock(
                IntegrationConfig=MagicMock(),
                get_integration_store=MagicMock(return_value=AsyncMock()),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_outlook,
            )

            result = await connect_outlook(
                account=account,
                auth_code="auth-code-456",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is True
            # Should fall back to userPrincipalName since mail is empty
            assert account.email_address == "principal@outlook.com"
            assert account.display_name == "principal"

    @pytest.mark.asyncio
    async def test_profile_with_empty_email_uses_default_name(self, mock_schedule_persist):
        """When profile returns empty fields."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.refresh_token = "refresh-token"
        mock_connector.access_token = "access-token"
        mock_connector.token_expiry = _NOW
        mock_connector.get_user_info = AsyncMock(
            return_value={"mail": "", "userPrincipalName": ""}
        )

        mock_sync_service = AsyncMock()
        mock_sync_service.start = AsyncMock()

        account = _make_account(provider=EmailProvider.OUTLOOK, email="")

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                OutlookSyncService=MagicMock(return_value=mock_sync_service),
                OutlookSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.outlook": MagicMock(
                OutlookConnector=MagicMock(return_value=mock_connector),
            ),
            "aragora.storage.integration_store": MagicMock(
                IntegrationConfig=MagicMock(),
                get_integration_store=MagicMock(return_value=AsyncMock()),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_outlook,
            )

            result = await connect_outlook(
                account=account,
                auth_code="auth-code-456",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is True
            assert account.display_name == "Outlook User"

    @pytest.mark.asyncio
    async def test_token_persist_failure_is_non_fatal(self, mock_schedule_persist):
        """When integration store save fails, connect should still succeed."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.refresh_token = "refresh-token"
        mock_connector.access_token = "access-token"
        mock_connector.token_expiry = _NOW
        mock_connector.get_user_info = AsyncMock(
            return_value={"mail": "user@outlook.com"}
        )

        mock_sync_service = AsyncMock()
        mock_sync_service.start = AsyncMock()

        mock_integration_store = AsyncMock()
        mock_integration_store.save = AsyncMock(side_effect=ValueError("invalid config"))

        account = _make_account(provider=EmailProvider.OUTLOOK)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                OutlookSyncService=MagicMock(return_value=mock_sync_service),
                OutlookSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.outlook": MagicMock(
                OutlookConnector=MagicMock(return_value=mock_connector),
            ),
            "aragora.storage.integration_store": MagicMock(
                IntegrationConfig=MagicMock(),
                get_integration_store=MagicMock(return_value=mock_integration_store),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_outlook,
            )

            result = await connect_outlook(
                account=account,
                auth_code="auth-code-456",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is True
            assert account.status == AccountStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_sync_service_registered_in_global_dict(self, mock_schedule_persist):
        """Verify sync service is stored in global registry."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.refresh_token = "refresh-token"
        mock_connector.access_token = "access-token"
        mock_connector.token_expiry = _NOW
        mock_connector.get_user_info = AsyncMock(
            return_value={"mail": "user@outlook.com"}
        )

        mock_sync_service = AsyncMock()
        mock_sync_service.start = AsyncMock()

        account = _make_account(account_id="acct-outlook-sync", provider=EmailProvider.OUTLOOK)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                OutlookSyncService=MagicMock(return_value=mock_sync_service),
                OutlookSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.outlook": MagicMock(
                OutlookConnector=MagicMock(return_value=mock_connector),
            ),
            "aragora.storage.integration_store": MagicMock(
                IntegrationConfig=MagicMock(),
                get_integration_store=MagicMock(return_value=AsyncMock()),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_outlook,
            )
            from aragora.server.handlers.features.unified_inbox.sync import (
                get_sync_services,
            )

            result = await connect_outlook(
                account=account,
                auth_code="auth-code-456",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-outlook",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is True
            services = get_sync_services()
            assert "tenant-outlook" in services
            assert "acct-outlook-sync" in services["tenant-outlook"]


# ===========================================================================
# disconnect_account
# ===========================================================================


class TestDisconnectAccount:
    """Tests for disconnect_account."""

    @pytest.mark.asyncio
    async def test_disconnect_with_running_sync_service(self):
        """When a sync service exists, it should be stopped and removed."""
        from aragora.server.handlers.features.unified_inbox.accounts import (
            disconnect_account,
        )
        from aragora.server.handlers.features.unified_inbox.sync import (
            get_sync_services,
        )

        mock_service = AsyncMock()
        mock_service.stop = AsyncMock()

        services = get_sync_services()
        services["tenant-1"] = {"acct-1": mock_service}

        await disconnect_account("tenant-1", "acct-1")

        mock_service.stop.assert_awaited_once()
        assert "acct-1" not in services.get("tenant-1", {})

    @pytest.mark.asyncio
    async def test_disconnect_no_sync_service(self):
        """When no sync service exists, should be a no-op."""
        from aragora.server.handlers.features.unified_inbox.accounts import (
            disconnect_account,
        )

        # Should not raise
        await disconnect_account("tenant-1", "nonexistent-acct")

    @pytest.mark.asyncio
    async def test_disconnect_no_tenant(self):
        """When tenant doesn't exist in registry."""
        from aragora.server.handlers.features.unified_inbox.accounts import (
            disconnect_account,
        )

        # Should not raise
        await disconnect_account("unknown-tenant", "acct-1")

    @pytest.mark.asyncio
    async def test_disconnect_stop_raises_os_error(self):
        """When sync_service.stop() raises OSError, should handle gracefully."""
        from aragora.server.handlers.features.unified_inbox.accounts import (
            disconnect_account,
        )
        from aragora.server.handlers.features.unified_inbox.sync import (
            get_sync_services,
        )

        mock_service = AsyncMock()
        mock_service.stop = AsyncMock(side_effect=OSError("IO error"))

        services = get_sync_services()
        services["tenant-1"] = {"acct-1": mock_service}

        # Should not raise, error is caught
        await disconnect_account("tenant-1", "acct-1")

        mock_service.stop.assert_awaited_once()
        # Account should still be removed from registry
        assert "acct-1" not in services.get("tenant-1", {})

    @pytest.mark.asyncio
    async def test_disconnect_stop_raises_value_error(self):
        """When sync_service.stop() raises ValueError."""
        from aragora.server.handlers.features.unified_inbox.accounts import (
            disconnect_account,
        )
        from aragora.server.handlers.features.unified_inbox.sync import (
            get_sync_services,
        )

        mock_service = AsyncMock()
        mock_service.stop = AsyncMock(side_effect=ValueError("bad state"))

        services = get_sync_services()
        services["tenant-1"] = {"acct-1": mock_service}

        await disconnect_account("tenant-1", "acct-1")
        mock_service.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_stop_raises_attribute_error(self):
        """When sync_service.stop() raises AttributeError."""
        from aragora.server.handlers.features.unified_inbox.accounts import (
            disconnect_account,
        )
        from aragora.server.handlers.features.unified_inbox.sync import (
            get_sync_services,
        )

        mock_service = AsyncMock()
        mock_service.stop = AsyncMock(side_effect=AttributeError("no stop"))

        services = get_sync_services()
        services["tenant-1"] = {"acct-1": mock_service}

        await disconnect_account("tenant-1", "acct-1")
        mock_service.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_service_without_stop_method(self):
        """When sync service has no stop() method (hasattr check)."""
        from aragora.server.handlers.features.unified_inbox.accounts import (
            disconnect_account,
        )
        from aragora.server.handlers.features.unified_inbox.sync import (
            get_sync_services,
        )

        # Create object without stop method
        mock_service = MagicMock(spec=[])

        services = get_sync_services()
        services["tenant-1"] = {"acct-1": mock_service}

        # Should not raise
        await disconnect_account("tenant-1", "acct-1")
        assert "acct-1" not in services.get("tenant-1", {})

    @pytest.mark.asyncio
    async def test_disconnect_only_removes_specified_account(self):
        """Other accounts in the same tenant should remain."""
        from aragora.server.handlers.features.unified_inbox.accounts import (
            disconnect_account,
        )
        from aragora.server.handlers.features.unified_inbox.sync import (
            get_sync_services,
        )

        mock_service_1 = AsyncMock()
        mock_service_1.stop = AsyncMock()
        mock_service_2 = AsyncMock()
        mock_service_2.stop = AsyncMock()

        services = get_sync_services()
        services["tenant-1"] = {
            "acct-1": mock_service_1,
            "acct-2": mock_service_2,
        }

        await disconnect_account("tenant-1", "acct-1")

        assert "acct-1" not in services["tenant-1"]
        assert "acct-2" in services["tenant-1"]
        mock_service_1.stop.assert_awaited_once()
        mock_service_2.stop.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_disconnect_other_tenants_unaffected(self):
        """Sync services in other tenants should remain."""
        from aragora.server.handlers.features.unified_inbox.accounts import (
            disconnect_account,
        )
        from aragora.server.handlers.features.unified_inbox.sync import (
            get_sync_services,
        )

        mock_service = AsyncMock()
        mock_service.stop = AsyncMock()
        other_service = AsyncMock()

        services = get_sync_services()
        services["tenant-1"] = {"acct-1": mock_service}
        services["tenant-2"] = {"acct-1": other_service}

        await disconnect_account("tenant-1", "acct-1")

        assert "acct-1" not in services["tenant-1"]
        assert "acct-1" in services["tenant-2"]
        other_service.stop.assert_not_awaited()


# ===========================================================================
# Security Tests
# ===========================================================================


class TestSecurityEdgeCases:
    """Security-related edge case tests."""

    @pytest.mark.asyncio
    async def test_gmail_oauth_redirect_uri_with_path_traversal(self, mock_gmail_connector):
        """Path traversal in redirect_uri should be passed through (validation is server-side)."""
        mock_module = MagicMock()
        mock_module.GmailConnector = MagicMock(return_value=mock_gmail_connector)
        with patch.dict("sys.modules", {
            "aragora.connectors.enterprise.communication.gmail": mock_module,
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                handle_gmail_oauth_url,
            )

            result = await handle_gmail_oauth_url(
                {"redirect_uri": "http://evil.com/../../etc/passwd"},
                "tenant-1",
            )
            # The function passes redirect_uri through to the connector;
            # it does not validate the URI itself
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_outlook_oauth_redirect_uri_with_injection(self, mock_outlook_connector):
        """Injection attempt in redirect_uri."""
        mock_module = MagicMock()
        mock_module.OutlookConnector = MagicMock(return_value=mock_outlook_connector)
        with patch.dict("sys.modules", {
            "aragora.connectors.enterprise.communication.outlook": mock_module,
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                handle_outlook_oauth_url,
            )

            result = await handle_outlook_oauth_url(
                {"redirect_uri": "http://localhost/callback?param=<script>alert(1)</script>"},
                "tenant-1",
            )
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_disconnect_with_path_traversal_account_id(self):
        """Path traversal in account_id should not affect other tenants."""
        from aragora.server.handlers.features.unified_inbox.accounts import (
            disconnect_account,
        )
        from aragora.server.handlers.features.unified_inbox.sync import (
            get_sync_services,
        )

        services = get_sync_services()
        services["tenant-1"] = {}

        # Should not raise, just a no-op
        await disconnect_account("tenant-1", "../../../etc/passwd")
        assert len(services["tenant-1"]) == 0

    @pytest.mark.asyncio
    async def test_disconnect_with_empty_account_id(self):
        """Empty account_id should be a no-op."""
        from aragora.server.handlers.features.unified_inbox.accounts import (
            disconnect_account,
        )

        await disconnect_account("tenant-1", "")

    @pytest.mark.asyncio
    async def test_disconnect_with_empty_tenant_id(self):
        """Empty tenant_id should be a no-op."""
        from aragora.server.handlers.features.unified_inbox.accounts import (
            disconnect_account,
        )

        await disconnect_account("", "acct-1")

    @pytest.mark.asyncio
    async def test_gmail_oauth_with_unicode_state(self, mock_gmail_connector):
        """Unicode characters in state parameter."""
        mock_module = MagicMock()
        mock_module.GmailConnector = MagicMock(return_value=mock_gmail_connector)
        with patch.dict("sys.modules", {
            "aragora.connectors.enterprise.communication.gmail": mock_module,
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                handle_gmail_oauth_url,
            )

            result = await handle_gmail_oauth_url(
                {"redirect_uri": "http://localhost/callback", "state": "unicode-test"},
                "tenant-1",
            )
            assert result["success"] is True
            assert result["data"]["state"] == "unicode-test"


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge case tests for account functions."""

    @pytest.mark.asyncio
    async def test_gmail_oauth_empty_params(self):
        """Empty params dict returns 400."""
        from aragora.server.handlers.features.unified_inbox.accounts import (
            handle_gmail_oauth_url,
        )

        result = await handle_gmail_oauth_url({}, "tenant-1")
        assert result["success"] is False
        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_outlook_oauth_empty_params(self):
        """Empty params dict returns 400."""
        from aragora.server.handlers.features.unified_inbox.accounts import (
            handle_outlook_oauth_url,
        )

        result = await handle_outlook_oauth_url({}, "tenant-1")
        assert result["success"] is False
        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_connect_gmail_mock_mode_email_format(self, mock_schedule_persist):
        """In mock mode, email is derived from first 8 chars of account id."""
        account = _make_account(account_id="12345678abcdef", provider=EmailProvider.GMAIL)

        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def fake_import(name, *args, **kwargs):
            if "aragora.connectors.email" in name or "aragora.connectors.enterprise.communication.gmail" in name:
                raise ImportError("Not available")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_gmail,
            )

            result = await connect_gmail(
                account=account,
                auth_code="auth-code",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is True
            # First 8 chars of "12345678abcdef" = "12345678"
            assert account.email_address == "user_12345678@gmail.com"

    @pytest.mark.asyncio
    async def test_connect_outlook_mock_mode_email_format(self, mock_schedule_persist):
        """In mock mode, email is derived from first 8 chars of account id."""
        account = _make_account(account_id="abcdef1234567890", provider=EmailProvider.OUTLOOK)

        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def fake_import(name, *args, **kwargs):
            if "aragora.connectors.email" in name or "aragora.connectors.enterprise.communication.outlook" in name:
                raise ImportError("Not available")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_outlook,
            )

            result = await connect_outlook(
                account=account,
                auth_code="auth-code",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is True
            assert account.email_address == "user_abcdef12@outlook.com"

    @pytest.mark.asyncio
    async def test_gmail_profile_extracts_display_name_from_email(self, mock_schedule_persist):
        """Display name should be local part of email address."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.refresh_token = "refresh-token"
        mock_connector.access_token = "access-token"
        mock_connector.token_expiry = _NOW
        mock_connector.get_user_info = AsyncMock(
            return_value={"emailAddress": "john.doe@gmail.com"}
        )

        mock_sync_service = AsyncMock()
        mock_sync_service.start = AsyncMock()

        account = _make_account(provider=EmailProvider.GMAIL)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                GmailSyncService=MagicMock(return_value=mock_sync_service),
                GmailSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.gmail": MagicMock(
                GmailConnector=MagicMock(return_value=mock_connector),
            ),
            "aragora.storage.gmail_token_store": MagicMock(
                GmailUserState=MagicMock(),
                get_gmail_token_store=MagicMock(return_value=AsyncMock()),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_gmail,
            )

            result = await connect_gmail(
                account=account,
                auth_code="auth-code-123",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is True
            assert account.display_name == "john.doe"

    @pytest.mark.asyncio
    async def test_outlook_profile_mail_takes_precedence(self, mock_schedule_persist):
        """Profile 'mail' field should take precedence over 'userPrincipalName'."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.refresh_token = "refresh-token"
        mock_connector.access_token = "access-token"
        mock_connector.token_expiry = _NOW
        mock_connector.get_user_info = AsyncMock(
            return_value={"mail": "preferred@outlook.com", "userPrincipalName": "fallback@outlook.com"}
        )

        mock_sync_service = AsyncMock()
        mock_sync_service.start = AsyncMock()

        account = _make_account(provider=EmailProvider.OUTLOOK)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                OutlookSyncService=MagicMock(return_value=mock_sync_service),
                OutlookSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.outlook": MagicMock(
                OutlookConnector=MagicMock(return_value=mock_connector),
            ),
            "aragora.storage.integration_store": MagicMock(
                IntegrationConfig=MagicMock(),
                get_integration_store=MagicMock(return_value=AsyncMock()),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_outlook,
            )

            result = await connect_outlook(
                account=account,
                auth_code="auth-code-456",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is True
            assert account.email_address == "preferred@outlook.com"


# ===========================================================================
# Connected Account Model Tests
# ===========================================================================


class TestConnectedAccountModel:
    """Tests for the ConnectedAccount dataclass used by accounts module."""

    def test_create_account_defaults(self):
        account = _make_account()
        assert account.id == "acct-test-001"
        assert account.provider == EmailProvider.GMAIL
        assert account.status == AccountStatus.PENDING
        assert account.total_messages == 0
        assert account.unread_count == 0
        assert account.sync_errors == 0
        assert account.metadata == {}

    def test_account_status_transitions(self):
        account = _make_account()
        assert account.status == AccountStatus.PENDING
        account.status = AccountStatus.CONNECTED
        assert account.status == AccountStatus.CONNECTED
        account.status = AccountStatus.DISCONNECTED
        assert account.status == AccountStatus.DISCONNECTED

    def test_account_to_dict(self):
        account = _make_account(email="user@example.com")
        d = account.to_dict()
        assert d["id"] == "acct-test-001"
        assert d["provider"] == "gmail"
        assert d["email_address"] == "user@example.com"
        assert d["status"] == "pending"

    def test_email_provider_values(self):
        assert EmailProvider.GMAIL.value == "gmail"
        assert EmailProvider.OUTLOOK.value == "outlook"

    def test_account_status_values(self):
        assert AccountStatus.PENDING.value == "pending"
        assert AccountStatus.CONNECTED.value == "connected"
        assert AccountStatus.SYNCING.value == "syncing"
        assert AccountStatus.ERROR.value == "error"
        assert AccountStatus.DISCONNECTED.value == "disconnected"


# ===========================================================================
# Parametrized Tests
# ===========================================================================


class TestParametrizedOAuthValidation:
    """Parametrized tests for OAuth URL parameter validation."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("params", [
        {},
        {"redirect_uri": ""},
        {"redirect_uri": None},
        {"state": "some-state"},
    ])
    async def test_gmail_oauth_invalid_params(self, params):
        from aragora.server.handlers.features.unified_inbox.accounts import (
            handle_gmail_oauth_url,
        )

        # None or empty redirect_uri should fail
        result = await handle_gmail_oauth_url(params, "tenant-1")
        assert result["success"] is False
        assert result["status_code"] == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize("params", [
        {},
        {"redirect_uri": ""},
        {"redirect_uri": None},
        {"state": "some-state"},
    ])
    async def test_outlook_oauth_invalid_params(self, params):
        from aragora.server.handlers.features.unified_inbox.accounts import (
            handle_outlook_oauth_url,
        )

        result = await handle_outlook_oauth_url(params, "tenant-1")
        assert result["success"] is False
        assert result["status_code"] == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error_cls", [
        ConnectionError,
        TimeoutError,
        OSError,
        ValueError,
    ])
    async def test_gmail_connect_various_errors(self, error_cls, mock_schedule_persist):
        """All caught error types should return failure."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(side_effect=error_cls("test error"))

        account = _make_account(provider=EmailProvider.GMAIL)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                GmailSyncService=MagicMock(),
                GmailSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.gmail": MagicMock(
                GmailConnector=MagicMock(return_value=mock_connector),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_gmail,
            )

            result = await connect_gmail(
                account=account,
                auth_code="auth-code",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error_cls", [
        ConnectionError,
        TimeoutError,
        OSError,
        ValueError,
    ])
    async def test_outlook_connect_various_errors(self, error_cls, mock_schedule_persist):
        """All caught error types should return failure."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(side_effect=error_cls("test error"))

        account = _make_account(provider=EmailProvider.OUTLOOK)

        with patch.dict("sys.modules", {
            "aragora.connectors.email": MagicMock(
                OutlookSyncService=MagicMock(),
                OutlookSyncConfig=MagicMock(return_value=MagicMock()),
            ),
            "aragora.connectors.enterprise.communication.outlook": MagicMock(
                OutlookConnector=MagicMock(return_value=mock_connector),
            ),
        }):
            from aragora.server.handlers.features.unified_inbox.accounts import (
                connect_outlook,
            )

            result = await connect_outlook(
                account=account,
                auth_code="auth-code",
                redirect_uri="http://localhost/callback",
                tenant_id="tenant-1",
                schedule_message_persist=mock_schedule_persist,
            )

            assert result["success"] is False
