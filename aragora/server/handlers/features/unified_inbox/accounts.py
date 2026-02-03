"""Account connection and management for unified inbox.

Handles OAuth-based account connection for Gmail and Outlook,
account listing, and disconnection.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, cast
from uuid import uuid4

from .models import (
    AccountStatus,
    ConnectedAccount,
    EmailProvider,
)
from .sync import (
    convert_synced_message_to_unified,
    get_sync_services,
    get_sync_services_lock,
)

logger = logging.getLogger(__name__)


async def handle_gmail_oauth_url(
    params: dict[str, str],
    tenant_id: str,
) -> dict[str, Any]:
    """Generate Gmail OAuth authorization URL.

    Returns dict with 'success' bool and either 'data' or 'error'/'status_code'.
    """
    redirect_uri = params.get("redirect_uri")
    if not redirect_uri:
        return {"success": False, "error": "Missing redirect_uri parameter", "status_code": 400}

    state = params.get("state") or str(uuid4())

    try:
        from aragora.connectors.enterprise.communication.gmail import GmailConnector

        connector = cast(Any, GmailConnector)()

        if not connector.is_configured:
            return {
                "success": False,
                "error": "Gmail OAuth not configured. Set GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET.",
                "status_code": 503,
            }

        auth_url = connector.get_oauth_url(redirect_uri=redirect_uri, state=state)

        logger.info(f"[UnifiedInbox] Generated Gmail OAuth URL for tenant {tenant_id}")

        return {
            "success": True,
            "data": {
                "auth_url": auth_url,
                "provider": "gmail",
                "state": state,
            },
        }

    except ImportError:
        return {"success": False, "error": "Gmail connector not available", "status_code": 503}


async def handle_outlook_oauth_url(
    params: dict[str, str],
    tenant_id: str,
) -> dict[str, Any]:
    """Generate Outlook OAuth authorization URL.

    Returns dict with 'success' bool and either 'data' or 'error'/'status_code'.
    """
    redirect_uri = params.get("redirect_uri")
    if not redirect_uri:
        return {"success": False, "error": "Missing redirect_uri parameter", "status_code": 400}

    state = params.get("state") or str(uuid4())

    try:
        from aragora.connectors.enterprise.communication.outlook import OutlookConnector

        connector = OutlookConnector()

        if not connector.is_configured:
            return {
                "success": False,
                "error": "Outlook OAuth not configured. Set OUTLOOK_CLIENT_ID and OUTLOOK_CLIENT_SECRET.",
                "status_code": 503,
            }

        auth_url = connector.get_oauth_url(redirect_uri=redirect_uri, state=state)

        logger.info(f"[UnifiedInbox] Generated Outlook OAuth URL for tenant {tenant_id}")

        return {
            "success": True,
            "data": {
                "auth_url": auth_url,
                "provider": "outlook",
                "state": state,
            },
        }

    except ImportError:
        return {"success": False, "error": "Outlook connector not available", "status_code": 503}


async def connect_gmail(
    account: ConnectedAccount,
    auth_code: str,
    redirect_uri: str,
    tenant_id: str,
    schedule_message_persist: Any,
) -> dict[str, Any]:
    """Connect Gmail account via OAuth."""
    try:
        from aragora.connectors.email import GmailSyncService, GmailSyncConfig
        from aragora.connectors.enterprise.communication.gmail import GmailConnector

        config = GmailSyncConfig(
            enable_prioritization=True,
            initial_sync_days=7,
        )

        # Exchange auth code for tokens via Gmail connector
        connector: GmailConnector = cast(Any, GmailConnector)()
        auth_ok = await connector.authenticate(code=auth_code, redirect_uri=redirect_uri)
        if not auth_ok:
            return {"success": False, "error": "Gmail authentication failed"}

        refresh_token = connector.refresh_token or ""
        if not refresh_token:
            return {"success": False, "error": "Gmail refresh token not returned"}

        # Load profile for display details
        profile = await connector.get_user_info()
        account.email_address = profile.get("emailAddress", "") or account.email_address
        if account.email_address:
            account.display_name = account.email_address.split("@")[0]
        else:
            account.display_name = "Gmail User"

        _sync_services = get_sync_services()
        _sync_services_lock = get_sync_services_lock()

        # Initialize tenant sync registry (thread-safe)
        async with _sync_services_lock:
            if tenant_id not in _sync_services:
                _sync_services[tenant_id] = {}

        # Create message callback that stores unified messages
        def on_message_synced(synced_msg: Any) -> None:
            try:
                unified = convert_synced_message_to_unified(
                    synced_msg, account.id, EmailProvider.GMAIL
                )
                schedule_message_persist(tenant_id, unified)
            except Exception as e:
                logger.warning(f"[UnifiedInbox] Error converting message: {e}")

        # Create sync service
        sync_service = GmailSyncService(
            tenant_id=tenant_id,
            user_id=account.id,
            config=config,
            gmail_connector=connector,
            on_message_synced=on_message_synced,
        )

        # Store sync service
        _sync_services[tenant_id][account.id] = sync_service

        # Persist OAuth state for restart safety
        try:
            from aragora.storage.gmail_token_store import GmailUserState, get_gmail_token_store

            state = GmailUserState(
                user_id=account.id,
                org_id=tenant_id,
                email_address=account.email_address,
                access_token=connector.access_token or "",
                refresh_token=refresh_token,
                token_expiry=connector.token_expiry,
                connected_at=datetime.now(timezone.utc),
            )
            store = get_gmail_token_store()
            await store.save(state)
        except Exception as e:
            logger.warning(f"[UnifiedInbox] Failed to persist Gmail tokens: {e}")

        # Start sync using the authenticated connector
        await sync_service.start()

        account.status = AccountStatus.CONNECTED

        logger.info(f"[UnifiedInbox] Gmail sync service registered for {account.id}")
        return {"success": True}

    except ImportError:
        # GmailSyncService not available, use mock mode
        account.email_address = f"user_{account.id[:8]}@gmail.com"
        account.display_name = "Gmail User"
        account.status = AccountStatus.CONNECTED
        logger.warning("[UnifiedInbox] GmailSyncService not available, using mock mode")
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def connect_outlook(
    account: ConnectedAccount,
    auth_code: str,
    redirect_uri: str,
    tenant_id: str,
    schedule_message_persist: Any,
) -> dict[str, Any]:
    """Connect Outlook account via OAuth."""
    try:
        from aragora.connectors.email import OutlookSyncService, OutlookSyncConfig
        from aragora.connectors.enterprise.communication.outlook import OutlookConnector

        config = OutlookSyncConfig(
            enable_prioritization=True,
            initial_sync_days=7,
        )

        connector = OutlookConnector()
        auth_ok = await connector.authenticate(code=auth_code, redirect_uri=redirect_uri)
        if not auth_ok:
            return {"success": False, "error": "Outlook authentication failed"}

        refresh_token = connector.refresh_token or ""
        if not refresh_token:
            return {"success": False, "error": "Outlook refresh token not returned"}

        # Load profile for display details
        profile = await connector.get_user_info()
        account.email_address = profile.get("mail") or profile.get("userPrincipalName", "")
        if account.email_address:
            account.display_name = account.email_address.split("@")[0]
        else:
            account.display_name = "Outlook User"

        _sync_services = get_sync_services()
        _sync_services_lock = get_sync_services_lock()

        # Initialize tenant sync registry (thread-safe)
        async with _sync_services_lock:
            if tenant_id not in _sync_services:
                _sync_services[tenant_id] = {}

        # Create message callback that stores unified messages
        def on_message_synced(synced_msg: Any) -> None:
            try:
                unified = convert_synced_message_to_unified(
                    synced_msg, account.id, EmailProvider.OUTLOOK
                )
                schedule_message_persist(tenant_id, unified)
            except Exception as e:
                logger.warning(f"[UnifiedInbox] Error converting message: {e}")

        # Create sync service
        sync_service = OutlookSyncService(
            tenant_id=tenant_id,
            user_id=account.id,
            config=config,
            outlook_connector=connector,
            on_message_synced=on_message_synced,
        )

        # Store sync service (thread-safe)
        async with _sync_services_lock:
            _sync_services[tenant_id][account.id] = sync_service

        # Persist OAuth state for restart safety
        try:
            from aragora.storage.integration_store import (
                IntegrationConfig,
                get_integration_store,
            )

            integration = IntegrationConfig(
                type="outlook_email",
                enabled=True,
                settings={
                    "refresh_token": refresh_token,
                    "access_token": connector.access_token or "",
                    "token_expiry": (
                        connector.token_expiry.isoformat() if connector.token_expiry else None
                    ),
                    "account_id": account.id,
                    "tenant_id": tenant_id,
                    "email_address": account.email_address,
                },
                user_id=account.id,
                workspace_id=tenant_id,
            )
            store = get_integration_store()
            await store.save(integration)
        except Exception as e:
            logger.warning(f"[UnifiedInbox] Failed to persist Outlook tokens: {e}")

        # Start sync using the authenticated connector
        await sync_service.start()

        account.status = AccountStatus.CONNECTED

        logger.info(f"[UnifiedInbox] Outlook sync service registered for {account.id}")
        return {"success": True}

    except ImportError:
        # OutlookSyncService not available, use mock mode
        account.email_address = f"user_{account.id[:8]}@outlook.com"
        account.display_name = "Outlook User"
        account.status = AccountStatus.CONNECTED
        logger.warning("[UnifiedInbox] OutlookSyncService not available, using mock mode")
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def disconnect_account(
    tenant_id: str,
    account_id: str,
) -> None:
    """Stop and remove sync service for an account (if running)."""
    _sync_services = get_sync_services()
    _sync_services_lock = get_sync_services_lock()

    sync_service = None
    async with _sync_services_lock:
        if tenant_id in _sync_services and account_id in _sync_services[tenant_id]:
            sync_service = _sync_services[tenant_id].pop(account_id)
    if sync_service:
        try:
            if hasattr(sync_service, "stop"):
                await sync_service.stop()
            logger.info(f"[UnifiedInbox] Stopped sync service for account {account_id}")
        except Exception as e:
            logger.warning(f"[UnifiedInbox] Error stopping sync service: {e}")
