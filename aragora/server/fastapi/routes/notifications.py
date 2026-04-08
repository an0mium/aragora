"""
FastAPI v2 routes for Notifications Management.

Migrated from aragora.server.handlers.social.notifications.NotificationsHandler.
Provides endpoints for email/telegram notification configuration and delivery.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from aragora.integrations.email import EmailConfig, EmailRecipient
from aragora.rbac.models import AuthorizationContext
from aragora.server.fastapi.dependencies.auth import require_authenticated
from aragora.storage.notification_config_store import (
    StoredEmailConfig,
    StoredEmailRecipient,
    StoredTelegramConfig,
    get_notification_config_store,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["Notifications"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class EmailConfigRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    provider: str = "smtp"
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    use_tls: bool = True
    use_ssl: bool = False
    sendgrid_api_key: str = ""
    ses_region: str = "us-east-1"
    ses_access_key_id: str = ""
    ses_secret_access_key: str = ""
    from_email: str = "debates@aragora.ai"
    from_name: str = "Aragora Debates"
    notify_on_consensus: bool = True
    notify_on_debate_end: bool = True
    notify_on_error: bool = True
    enable_digest: bool = True
    digest_frequency: str = "daily"
    min_consensus_confidence: float = 0.7
    max_emails_per_hour: int = 50


class TelegramConfigRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    bot_token: str = ""
    chat_id: str = ""
    notify_on_consensus: bool = True
    notify_on_debate_end: bool = True
    notify_on_error: bool = True
    min_consensus_confidence: float = 0.7
    max_messages_per_minute: int = 20


class RecipientRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    email: str
    name: str | None = None
    preferences: dict[str, Any] = {}


class TestNotificationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "all"


class SendNotificationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "all"
    subject: str = "Aragora Notification"
    message: str = ""
    html_message: str | None = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


async def _get_email_integration(org_id: str | None = None):
    """Get email integration, preferring org-specific config."""
    try:
        from aragora.server.handlers.social.notifications import (
            get_email_integration_for_org,
        )

        return await get_email_integration_for_org(org_id)
    except (ImportError, TypeError, ValueError, OSError, RuntimeError):
        return None


async def _get_telegram_integration(org_id: str | None = None):
    """Get telegram integration, preferring org-specific config."""
    try:
        from aragora.server.handlers.social.notifications import (
            get_telegram_integration_for_org,
        )

        return await get_telegram_integration_for_org(org_id)
    except (ImportError, TypeError, ValueError, OSError, RuntimeError):
        return None


def _build_email_config(body: EmailConfigRequest) -> EmailConfig:
    """Normalize request data into the shared email config model."""
    return EmailConfig(
        provider=body.provider,
        smtp_host=body.smtp_host,
        smtp_port=body.smtp_port,
        smtp_username=body.smtp_username,
        smtp_password=body.smtp_password,
        use_tls=body.use_tls,
        use_ssl=body.use_ssl,
        sendgrid_api_key=body.sendgrid_api_key,
        ses_region=body.ses_region,
        ses_access_key_id=body.ses_access_key_id,
        ses_secret_access_key=body.ses_secret_access_key,
        from_email=body.from_email,
        from_name=body.from_name,
        notify_on_consensus=body.notify_on_consensus,
        notify_on_debate_end=body.notify_on_debate_end,
        notify_on_error=body.notify_on_error,
        enable_digest=body.enable_digest,
        digest_frequency=body.digest_frequency,
        min_consensus_confidence=body.min_consensus_confidence,
        max_emails_per_hour=body.max_emails_per_hour,
    )


def require_notification_permission(action: str) -> Any:
    """Preserve legacy notification permission compatibility in FastAPI routes."""

    async def _require_notification_auth(
        auth: AuthorizationContext = Depends(require_authenticated),
    ) -> AuthorizationContext:
        if auth.has_any_role("admin", "owner"):
            return auth

        if action in auth.permissions:
            return auth

        for permission in (
            f"notifications:{action}",
            f"notification:{action}",
        ):
            if auth.has_permission(permission):
                return auth

        raise HTTPException(status_code=403, detail=f"Permission denied: notifications:{action}")

    return _require_notification_auth


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/notifications/status")
async def get_notification_status(
    auth: AuthorizationContext = Depends(require_notification_permission("read")),
):
    """Get status of notification integrations."""
    email = await _get_email_integration(auth.org_id)
    telegram = await _get_telegram_integration(auth.org_id)

    return {
        "data": {
            "email": {
                "configured": email is not None,
                "host": email.config.smtp_host if email else None,
                "recipients_count": len(email.recipients) if email else 0,
                "settings": (
                    {
                        "notify_on_consensus": email.config.notify_on_consensus,
                        "notify_on_debate_end": email.config.notify_on_debate_end,
                        "notify_on_error": email.config.notify_on_error,
                        "enable_digest": email.config.enable_digest,
                        "digest_frequency": email.config.digest_frequency,
                    }
                    if email
                    else None
                ),
            },
            "telegram": {
                "configured": telegram is not None,
                "chat_id": telegram.config.chat_id[:8] + "..." if telegram else None,
                "settings": (
                    {
                        "notify_on_consensus": telegram.config.notify_on_consensus,
                        "notify_on_debate_end": telegram.config.notify_on_debate_end,
                        "notify_on_error": telegram.config.notify_on_error,
                    }
                    if telegram
                    else None
                ),
            },
        }
    }


@router.get("/notifications/email/recipients")
async def get_email_recipients(
    auth: AuthorizationContext = Depends(require_notification_permission("read")),
):
    """Get list of email recipients."""
    email = await _get_email_integration(auth.org_id)
    if not email:
        return {"data": {"recipients": [], "error": "Email not configured"}}

    return {
        "data": {
            "recipients": [{"email": r.email, "name": r.name} for r in email.recipients],
            "count": len(email.recipients),
        }
    }


@router.post("/notifications/email/config")
async def configure_email(
    body: EmailConfigRequest,
    auth: AuthorizationContext = Depends(require_notification_permission("write")),
):
    """Configure email integration settings."""
    try:
        from aragora.server.handlers.social.notifications import (
            configure_email_integration,
            invalidate_org_integration_cache,
        )

        config = _build_email_config(body)
        if auth.org_id:
            stored_config = StoredEmailConfig(
                org_id=auth.org_id,
                provider=config.provider,
                smtp_host=config.smtp_host,
                smtp_port=config.smtp_port,
                smtp_username=config.smtp_username,
                smtp_password=config.smtp_password,
                use_tls=config.use_tls,
                use_ssl=config.use_ssl,
                sendgrid_api_key=config.sendgrid_api_key,
                ses_region=config.ses_region,
                ses_access_key_id=config.ses_access_key_id,
                ses_secret_access_key=config.ses_secret_access_key,
                from_email=config.from_email,
                from_name=config.from_name,
                notify_on_consensus=config.notify_on_consensus,
                notify_on_debate_end=config.notify_on_debate_end,
                notify_on_error=config.notify_on_error,
                enable_digest=config.enable_digest,
                digest_frequency=config.digest_frequency,
                min_consensus_confidence=config.min_consensus_confidence,
                max_emails_per_hour=config.max_emails_per_hour,
            )
            store = get_notification_config_store()
            await store.save_email_config(stored_config)
            invalidate_org_integration_cache(auth.org_id)
            return {
                "data": {
                    "success": True,
                    "message": f"Email configured for org {auth.org_id} with host: {config.smtp_host}",
                    "org_id": auth.org_id,
                }
            }

        configure_email_integration(config)

        return {
            "data": {
                "success": True,
                "message": f"Email configured with host: {config.smtp_host}",
            }
        }
    except ValueError as e:
        logger.warning("Invalid email config: %s", e)
        raise HTTPException(status_code=400, detail="Invalid configuration")
    except (TypeError, KeyError, OSError, ImportError) as e:
        logger.error("Failed to configure email: %s", e)
        raise HTTPException(status_code=500, detail="Failed to configure email")


@router.post("/notifications/telegram/config")
async def configure_telegram(
    body: TelegramConfigRequest,
    auth: AuthorizationContext = Depends(require_notification_permission("write")),
):
    """Configure Telegram integration settings."""
    try:
        from aragora.integrations.telegram import TelegramConfig
        from aragora.server.handlers.social.notifications import (
            configure_telegram_integration,
            invalidate_org_integration_cache,
        )

        config = TelegramConfig(
            bot_token=body.bot_token,
            chat_id=body.chat_id,
            notify_on_consensus=body.notify_on_consensus,
            notify_on_debate_end=body.notify_on_debate_end,
            notify_on_error=body.notify_on_error,
            min_consensus_confidence=body.min_consensus_confidence,
            max_messages_per_minute=body.max_messages_per_minute,
        )
        if auth.org_id:
            store = get_notification_config_store()
            await store.save_telegram_config(
                StoredTelegramConfig(
                    org_id=auth.org_id,
                    bot_token=config.bot_token,
                    chat_id=config.chat_id,
                    notify_on_consensus=config.notify_on_consensus,
                    notify_on_debate_end=config.notify_on_debate_end,
                    notify_on_error=config.notify_on_error,
                    min_consensus_confidence=config.min_consensus_confidence,
                    max_messages_per_minute=config.max_messages_per_minute,
                )
            )
            invalidate_org_integration_cache(auth.org_id)
            return {
                "data": {
                    "success": True,
                    "message": f"Telegram configured for org {auth.org_id}",
                    "org_id": auth.org_id,
                }
            }

        configure_telegram_integration(config)

        return {
            "data": {
                "success": True,
                "message": "Telegram configured successfully",
            }
        }
    except ValueError as e:
        logger.warning("Invalid telegram config: %s", e)
        raise HTTPException(status_code=400, detail="Invalid configuration")
    except (TypeError, KeyError, OSError, ImportError) as e:
        logger.error("Failed to configure telegram: %s", e)
        raise HTTPException(status_code=500, detail="Failed to configure telegram")


@router.post("/notifications/email/recipient")
async def add_email_recipient(
    body: RecipientRequest,
    auth: AuthorizationContext = Depends(require_notification_permission("write")),
):
    """Add an email recipient."""
    if not body.email or "@" not in body.email:
        raise HTTPException(status_code=400, detail="Valid email address required")

    try:
        if auth.org_id:
            from aragora.server.handlers.social.notifications import (
                invalidate_org_integration_cache,
            )

            store = get_notification_config_store()
            await store.add_recipient(
                StoredEmailRecipient(
                    org_id=auth.org_id,
                    email=body.email,
                    name=body.name,
                    preferences=body.preferences,
                )
            )
            invalidate_org_integration_cache(auth.org_id)
            recipients = await store.get_recipients(auth.org_id)
            return {
                "data": {
                    "success": True,
                    "message": f"Recipient added: {body.email}",
                    "recipients_count": len(recipients),
                    "org_id": auth.org_id,
                }
            }

        email = await _get_email_integration()
        if not email:
            raise HTTPException(status_code=503, detail="Email integration not configured")

        recipient = EmailRecipient(
            email=body.email,
            name=body.name,
            preferences=body.preferences,
        )
        email.add_recipient(recipient)

        return {
            "data": {
                "success": True,
                "message": f"Recipient added: {body.email}",
                "recipients_count": len(email.recipients),
            }
        }
    except HTTPException:
        raise
    except (TypeError, ValueError, ImportError, OSError) as e:
        logger.error("Failed to add recipient: %s", e)
        raise HTTPException(status_code=500, detail="Failed to add recipient")


@router.delete("/notifications/email/recipient")
async def remove_email_recipient(
    email: str,
    auth: AuthorizationContext = Depends(require_notification_permission("delete")),
):
    """Remove an email recipient."""
    if not email:
        raise HTTPException(status_code=400, detail="email parameter required")

    try:
        if auth.org_id:
            from aragora.server.handlers.social.notifications import (
                invalidate_org_integration_cache,
            )

            store = get_notification_config_store()
            removed = await store.remove_recipient(auth.org_id, email)
            if not removed:
                raise HTTPException(status_code=404, detail=f"Recipient not found: {email}")

            invalidate_org_integration_cache(auth.org_id)
            recipients = await store.get_recipients(auth.org_id)
            return {
                "data": {
                    "success": True,
                    "message": f"Recipient removed: {email}",
                    "recipients_count": len(recipients),
                    "org_id": auth.org_id,
                }
            }

        integration = await _get_email_integration()
        if not integration:
            raise HTTPException(status_code=503, detail="Email integration not configured")

        removed = integration.remove_recipient(email)
        if not removed:
            raise HTTPException(status_code=404, detail=f"Recipient not found: {email}")

        return {
            "data": {
                "success": True,
                "message": f"Recipient removed: {email}",
                "recipients_count": len(integration.recipients),
            }
        }
    except HTTPException:
        raise
    except (TypeError, ValueError, ImportError, OSError) as e:
        logger.error("Failed to remove recipient: %s", e)
        raise HTTPException(status_code=500, detail="Failed to remove recipient")


@router.post("/notifications/test")
async def send_test_notification(
    body: TestNotificationRequest,
    auth: AuthorizationContext = Depends(require_notification_permission("write")),
):
    """Send a test notification."""
    results: dict[str, Any] = {}

    if body.type in ("all", "email"):
        email = await _get_email_integration(auth.org_id)
        if email and email.recipients:
            try:
                success = await email._send_email(
                    email.recipients[0],
                    "Aragora Test Notification",
                    "<h1>Test Notification</h1><p>Your email integration is working correctly!</p>",
                    "Test Notification - Your email integration is working correctly!",
                )
                results["email"] = {"success": success, "recipient": email.recipients[0].email}
            except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
                logger.warning("Test email failed: %s", e)
                results["email"] = {"success": False, "error": "Send failed"}
        else:
            results["email"] = {
                "success": False,
                "error": "Email not configured" if not email else "No recipients configured",
            }

    if body.type in ("all", "telegram"):
        telegram = await _get_telegram_integration(auth.org_id)
        if telegram:
            try:
                from aragora.integrations.telegram import TelegramMessage

                msg = TelegramMessage(
                    text="<b>Test Notification</b>\n\nYour Telegram integration is working correctly!",
                )
                success = await telegram._send_message(msg)
                results["telegram"] = {"success": success}
            except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
                logger.warning("Test telegram failed: %s", e)
                results["telegram"] = {"success": False, "error": "Send failed"}
        else:
            results["telegram"] = {"success": False, "error": "Telegram not configured"}

    all_success = all(r.get("success", False) for r in results.values())
    return {"data": {"success": all_success, "results": results}}


@router.post("/notifications/send")
async def send_notification(
    body: SendNotificationRequest,
    auth: AuthorizationContext = Depends(require_notification_permission("write")),
):
    """Send a notification with custom content."""
    results: dict[str, Any] = {}
    html_message = body.html_message or f"<p>{body.message}</p>"

    if body.type in ("all", "email"):
        email = await _get_email_integration(auth.org_id)
        if email and email.recipients:
            try:
                sent = 0
                for recipient in email.recipients:
                    success = await email._send_email(
                        recipient, body.subject, html_message, body.message
                    )
                    if success:
                        sent += 1
                results["email"] = {
                    "success": sent > 0,
                    "sent": sent,
                    "total": len(email.recipients),
                }
            except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
                logger.warning("Email send failed: %s", e)
                results["email"] = {"success": False, "error": "Send failed"}
        else:
            results["email"] = {
                "success": False,
                "error": "Email not configured or no recipients",
            }

    if body.type in ("all", "telegram"):
        telegram = await _get_telegram_integration(auth.org_id)
        if telegram:
            try:
                from aragora.integrations.telegram import TelegramMessage

                telegram_text = f"<b>{body.subject}</b>\n\n{body.message}"
                msg = TelegramMessage(text=telegram_text)
                success = await telegram._send_message(msg)
                results["telegram"] = {"success": success}
            except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
                logger.warning("Telegram send failed: %s", e)
                results["telegram"] = {"success": False, "error": "Send failed"}
        else:
            results["telegram"] = {"success": False, "error": "Telegram not configured"}

    all_success = all(r.get("success", False) for r in results.values())
    return {"data": {"success": all_success, "results": results}}
