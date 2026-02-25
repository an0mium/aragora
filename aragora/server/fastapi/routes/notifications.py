"""
FastAPI v2 routes for Notifications Management.

Migrated from aragora.server.handlers.social.notifications.NotificationsHandler.
Provides endpoints for email/telegram notification configuration and delivery.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/notifications/status")
async def get_notification_status():
    """Get status of notification integrations."""
    email = await _get_email_integration()
    telegram = await _get_telegram_integration()

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
async def get_email_recipients():
    """Get list of email recipients."""
    email = await _get_email_integration()
    if not email:
        return {"data": {"recipients": [], "error": "Email not configured"}}

    return {
        "data": {
            "recipients": [{"email": r.email, "name": r.name} for r in email.recipients],
            "count": len(email.recipients),
        }
    }


@router.post("/notifications/email/config")
async def configure_email(body: EmailConfigRequest):
    """Configure email integration settings."""
    try:
        from aragora.integrations.email import EmailConfig, EmailIntegration
        from aragora.server.handlers.social.notifications import configure_email_integration

        config = EmailConfig(
            smtp_host=body.smtp_host,
            smtp_port=body.smtp_port,
            smtp_username=body.smtp_username,
            smtp_password=body.smtp_password,
            use_tls=body.use_tls,
            use_ssl=body.use_ssl,
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
async def configure_telegram(body: TelegramConfigRequest):
    """Configure Telegram integration settings."""
    try:
        from aragora.integrations.telegram import TelegramConfig
        from aragora.server.handlers.social.notifications import configure_telegram_integration

        config = TelegramConfig(
            bot_token=body.bot_token,
            chat_id=body.chat_id,
            notify_on_consensus=body.notify_on_consensus,
            notify_on_debate_end=body.notify_on_debate_end,
            notify_on_error=body.notify_on_error,
            min_consensus_confidence=body.min_consensus_confidence,
            max_messages_per_minute=body.max_messages_per_minute,
        )
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
async def add_email_recipient(body: RecipientRequest):
    """Add an email recipient."""
    if not body.email or "@" not in body.email:
        raise HTTPException(status_code=400, detail="Valid email address required")

    try:
        from aragora.integrations.email import EmailRecipient
        from aragora.server.handlers.social.notifications import get_email_integration

        email = get_email_integration()
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
async def remove_email_recipient(email: str):
    """Remove an email recipient."""
    if not email:
        raise HTTPException(status_code=400, detail="email parameter required")

    try:
        from aragora.server.handlers.social.notifications import get_email_integration

        integration = get_email_integration()
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
async def send_test_notification(body: TestNotificationRequest):
    """Send a test notification."""
    results: dict[str, Any] = {}

    if body.type in ("all", "email"):
        email = await _get_email_integration()
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
        telegram = await _get_telegram_integration()
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
async def send_notification(body: SendNotificationRequest):
    """Send a notification with custom content."""
    results: dict[str, Any] = {}
    html_message = body.html_message or f"<p>{body.message}</p>"

    if body.type in ("all", "email"):
        email = await _get_email_integration()
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
        telegram = await _get_telegram_integration()
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
