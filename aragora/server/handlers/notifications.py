"""
Notifications handler for Email and Telegram integrations.

Provides endpoints for configuring and managing notification channels:
- Email notifications via SMTP
- Telegram bot notifications
- Test notification delivery
- Status and configuration management
"""

import logging
import os
from typing import Any, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
)
from aragora.server.handlers.utils.rate_limit import RateLimiter, get_client_ip
from aragora.server.validation.schema import (
    validate_against_schema,
    EMAIL_CONFIG_SCHEMA,
    TELEGRAM_CONFIG_SCHEMA,
    NOTIFICATION_SEND_SCHEMA,
)
from aragora.integrations.email import EmailConfig, EmailIntegration, EmailRecipient
from aragora.integrations.telegram import TelegramConfig, TelegramIntegration

logger = logging.getLogger(__name__)

# Rate limiter for notification endpoints (30 requests per minute - can trigger external calls)
_notifications_limiter = RateLimiter(requests_per_minute=30)


def _run_async_in_thread(coro):
    """Run an async coroutine in a thread-safe manner.

    Creates a new event loop for the thread to avoid RuntimeError when
    asyncio.run() is called from within a ThreadPoolExecutor.
    """
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Module-level integration instances (singletons)
_email_integration: Optional[EmailIntegration] = None
_telegram_integration: Optional[TelegramIntegration] = None


def get_email_integration() -> Optional[EmailIntegration]:
    """Get or create the email integration singleton."""
    global _email_integration
    if _email_integration is not None:
        return _email_integration

    # Try to initialize from environment
    smtp_host = os.getenv("SMTP_HOST")
    if smtp_host:
        try:
            config = EmailConfig(
                smtp_host=smtp_host,
                smtp_port=int(os.getenv("SMTP_PORT", "587")),
                smtp_username=os.getenv("SMTP_USERNAME", ""),
                smtp_password=os.getenv("SMTP_PASSWORD", ""),
                use_tls=os.getenv("SMTP_USE_TLS", "true").lower() == "true",
                use_ssl=os.getenv("SMTP_USE_SSL", "false").lower() == "true",
                from_email=os.getenv("SMTP_FROM_EMAIL", "debates@aragora.ai"),
                from_name=os.getenv("SMTP_FROM_NAME", "Aragora Debates"),
            )
            _email_integration = EmailIntegration(config)
            logger.info(f"Email integration initialized with host: {smtp_host}")
        except Exception as e:
            logger.warning(f"Failed to initialize email integration: {e}")

    return _email_integration


def get_telegram_integration() -> Optional[TelegramIntegration]:
    """Get or create the telegram integration singleton."""
    global _telegram_integration
    if _telegram_integration is not None:
        return _telegram_integration

    # Try to initialize from environment
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if bot_token and chat_id:
        try:
            config = TelegramConfig(
                bot_token=bot_token,
                chat_id=chat_id,
            )
            _telegram_integration = TelegramIntegration(config)
            logger.info("Telegram integration initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize telegram integration: {e}")

    return _telegram_integration


def configure_email_integration(config: EmailConfig) -> EmailIntegration:
    """Configure and set the email integration."""
    global _email_integration
    _email_integration = EmailIntegration(config)
    logger.info(f"Email integration configured with host: {config.smtp_host}")
    return _email_integration


def configure_telegram_integration(config: TelegramConfig) -> TelegramIntegration:
    """Configure and set the telegram integration."""
    global _telegram_integration
    _telegram_integration = TelegramIntegration(config)
    logger.info("Telegram integration configured")
    return _telegram_integration


class NotificationsHandler(BaseHandler):
    """Handler for notification-related endpoints.

    Endpoints:
        GET  /api/notifications/status - Get integration status
        POST /api/notifications/email/config - Configure email settings
        POST /api/notifications/telegram/config - Configure Telegram settings
        POST /api/notifications/email/recipient - Add email recipient
        DELETE /api/notifications/email/recipient - Remove email recipient
        POST /api/notifications/test - Send test notification
        POST /api/notifications/send - Send a notification
    """

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path.startswith("/api/notifications")

    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle GET requests."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _notifications_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for notifications endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        if path == "/api/notifications/status":
            return self._get_status()

        if path == "/api/notifications/email/recipients":
            return self._get_email_recipients()

        return None

    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _notifications_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for notifications endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Require authentication for all POST endpoints
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        if path == "/api/notifications/email/config":
            return self._configure_email(handler)

        if path == "/api/notifications/telegram/config":
            return self._configure_telegram(handler)

        if path == "/api/notifications/email/recipient":
            return self._add_email_recipient(handler)

        if path == "/api/notifications/test":
            return self._send_test_notification(handler)

        if path == "/api/notifications/send":
            return self._send_notification(handler)

        return None

    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle DELETE requests."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        if path == "/api/notifications/email/recipient":
            return self._remove_email_recipient(handler, query_params)

        return None

    def _get_status(self) -> HandlerResult:
        """Get status of all notification integrations."""
        email = get_email_integration()
        telegram = get_telegram_integration()

        return json_response(
            {
                "email": {
                    "configured": email is not None,
                    "host": email.config.smtp_host if email else None,
                    "recipients_count": len(email.recipients) if email else 0,
                    "settings": (
                        {
                            "notify_on_consensus": (
                                email.config.notify_on_consensus if email else False
                            ),
                            "notify_on_debate_end": (
                                email.config.notify_on_debate_end if email else False
                            ),
                            "notify_on_error": email.config.notify_on_error if email else False,
                            "enable_digest": email.config.enable_digest if email else False,
                            "digest_frequency": email.config.digest_frequency if email else "daily",
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
                            "notify_on_consensus": (
                                telegram.config.notify_on_consensus if telegram else False
                            ),
                            "notify_on_debate_end": (
                                telegram.config.notify_on_debate_end if telegram else False
                            ),
                            "notify_on_error": (
                                telegram.config.notify_on_error if telegram else False
                            ),
                        }
                        if telegram
                        else None
                    ),
                },
            }
        )

    def _get_email_recipients(self) -> HandlerResult:
        """Get list of email recipients."""
        email = get_email_integration()
        if not email:
            return json_response({"recipients": [], "error": "Email not configured"})

        return json_response(
            {
                "recipients": [{"email": r.email, "name": r.name} for r in email.recipients],
                "count": len(email.recipients),
            }
        )

    def _configure_email(self, handler: Any) -> HandlerResult:
        """Configure email integration settings."""
        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        # Schema validation for input sanitization
        validation_result = validate_against_schema(body, EMAIL_CONFIG_SCHEMA)
        if not validation_result.is_valid:
            return error_response(validation_result.error, 400)

        try:
            config = EmailConfig(
                smtp_host=body.get("smtp_host", ""),
                smtp_port=body.get("smtp_port", 587),
                smtp_username=body.get("smtp_username", ""),
                smtp_password=body.get("smtp_password", ""),
                use_tls=body.get("use_tls", True),
                use_ssl=body.get("use_ssl", False),
                from_email=body.get("from_email", "debates@aragora.ai"),
                from_name=body.get("from_name", "Aragora Debates"),
                notify_on_consensus=body.get("notify_on_consensus", True),
                notify_on_debate_end=body.get("notify_on_debate_end", True),
                notify_on_error=body.get("notify_on_error", True),
                enable_digest=body.get("enable_digest", True),
                digest_frequency=body.get("digest_frequency", "daily"),
                min_consensus_confidence=body.get("min_consensus_confidence", 0.7),
                max_emails_per_hour=body.get("max_emails_per_hour", 50),
            )
            configure_email_integration(config)
            return json_response(
                {
                    "success": True,
                    "message": f"Email configured with host: {config.smtp_host}",
                }
            )
        except ValueError as e:
            return error_response(f"Invalid configuration: {e}", 400)
        except Exception as e:
            logger.error(f"Failed to configure email: {e}")
            return error_response("Failed to configure email", 500)

    def _configure_telegram(self, handler: Any) -> HandlerResult:
        """Configure Telegram integration settings."""
        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        # Schema validation for input sanitization
        validation_result = validate_against_schema(body, TELEGRAM_CONFIG_SCHEMA)
        if not validation_result.is_valid:
            return error_response(validation_result.error, 400)

        bot_token = body.get("bot_token", "")
        chat_id = body.get("chat_id", "")

        try:
            config = TelegramConfig(
                bot_token=bot_token,
                chat_id=chat_id,
                notify_on_consensus=body.get("notify_on_consensus", True),
                notify_on_debate_end=body.get("notify_on_debate_end", True),
                notify_on_error=body.get("notify_on_error", True),
                min_consensus_confidence=body.get("min_consensus_confidence", 0.7),
                max_messages_per_minute=body.get("max_messages_per_minute", 20),
            )
            configure_telegram_integration(config)
            return json_response(
                {
                    "success": True,
                    "message": "Telegram configured successfully",
                }
            )
        except ValueError as e:
            return error_response(f"Invalid configuration: {e}", 400)
        except Exception as e:
            logger.error(f"Failed to configure telegram: {e}")
            return error_response("Failed to configure telegram", 500)

    def _add_email_recipient(self, handler: Any) -> HandlerResult:
        """Add an email recipient."""
        email = get_email_integration()
        if not email:
            return error_response("Email integration not configured", 503)

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        recipient_email = body.get("email", "")
        if not recipient_email or "@" not in recipient_email:
            return error_response("Valid email address required", 400)

        recipient = EmailRecipient(
            email=recipient_email,
            name=body.get("name"),
            preferences=body.get("preferences", {}),
        )
        email.add_recipient(recipient)

        return json_response(
            {
                "success": True,
                "message": f"Recipient added: {recipient_email}",
                "recipients_count": len(email.recipients),
            }
        )

    def _remove_email_recipient(self, handler: Any, query_params: dict) -> HandlerResult:
        """Remove an email recipient."""
        email = get_email_integration()
        if not email:
            return error_response("Email integration not configured", 503)

        recipient_email = query_params.get("email", "")
        if not recipient_email:
            return error_response("email parameter required", 400)

        removed = email.remove_recipient(recipient_email)
        if removed:
            return json_response(
                {
                    "success": True,
                    "message": f"Recipient removed: {recipient_email}",
                    "recipients_count": len(email.recipients),
                }
            )
        else:
            return error_response(f"Recipient not found: {recipient_email}", 404)

    def _send_test_notification(self, handler: Any) -> HandlerResult:
        """Send a test notification."""
        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        notification_type = body.get("type", "all")
        results = {}

        # Test email
        if notification_type in ("all", "email"):
            email = get_email_integration()
            if email:
                if email.recipients:
                    # Import asyncio for running async in sync context
                    import asyncio

                    async def send_test_email():
                        return await email._send_email(
                            email.recipients[0],
                            "Aragora Test Notification",
                            "<h1>Test Notification</h1><p>Your email integration is working correctly!</p>",
                            "Test Notification - Your email integration is working correctly!",
                        )

                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Already in async context, run in thread with new loop
                            import concurrent.futures

                            with concurrent.futures.ThreadPoolExecutor() as pool:
                                success = pool.submit(
                                    _run_async_in_thread, send_test_email()
                                ).result()
                        else:
                            success = loop.run_until_complete(send_test_email())
                        results["email"] = {
                            "success": success,
                            "recipient": email.recipients[0].email,
                        }
                    except Exception as e:
                        results["email"] = {"success": False, "error": str(e)}
                else:
                    results["email"] = {"success": False, "error": "No recipients configured"}
            else:
                results["email"] = {"success": False, "error": "Email not configured"}

        # Test telegram
        if notification_type in ("all", "telegram"):
            telegram = get_telegram_integration()
            if telegram:
                import asyncio
                from aragora.integrations.telegram import TelegramMessage

                async def send_test_telegram():
                    msg = TelegramMessage(
                        text="<b>Test Notification</b>\n\nYour Telegram integration is working correctly! 🎉",
                    )
                    return await telegram._send_message(msg)

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            success = pool.submit(
                                _run_async_in_thread, send_test_telegram()
                            ).result()
                    else:
                        success = loop.run_until_complete(send_test_telegram())
                    results["telegram"] = {"success": success}
                except Exception as e:
                    results["telegram"] = {"success": False, "error": str(e)}
            else:
                results["telegram"] = {"success": False, "error": "Telegram not configured"}

        all_success = all(r.get("success", False) for r in results.values())
        return json_response(
            {
                "success": all_success,
                "results": results,
            }
        )

    def _send_notification(self, handler: Any) -> HandlerResult:
        """Send a notification with custom content."""
        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        # Schema validation for input sanitization
        validation_result = validate_against_schema(body, NOTIFICATION_SEND_SCHEMA)
        if not validation_result.is_valid:
            return error_response(validation_result.error, 400)

        notification_type = body.get("type", "all")
        subject = body.get("subject", "Aragora Notification")
        message = body.get("message", "")
        html_message = body.get("html_message", f"<p>{message}</p>")

        results = {}
        import asyncio

        # Send email
        if notification_type in ("all", "email"):
            email = get_email_integration()
            if email and email.recipients:

                async def send_emails():
                    sent = 0
                    for recipient in email.recipients:
                        success = await email._send_email(recipient, subject, html_message, message)
                        if success:
                            sent += 1
                    return sent

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            sent = pool.submit(_run_async_in_thread, send_emails()).result()
                    else:
                        sent = loop.run_until_complete(send_emails())
                    results["email"] = {
                        "success": sent > 0,
                        "sent": sent,
                        "total": len(email.recipients),
                    }
                except Exception as e:
                    results["email"] = {"success": False, "error": str(e)}
            else:
                results["email"] = {
                    "success": False,
                    "error": "Email not configured or no recipients",
                }

        # Send telegram
        if notification_type in ("all", "telegram"):
            telegram = get_telegram_integration()
            if telegram:
                from aragora.integrations.telegram import TelegramMessage

                # Convert to Telegram HTML format
                telegram_text = f"<b>{subject}</b>\n\n{message}"

                async def send_telegram():
                    msg = TelegramMessage(text=telegram_text)
                    return await telegram._send_message(msg)

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            success = pool.submit(_run_async_in_thread, send_telegram()).result()
                    else:
                        success = loop.run_until_complete(send_telegram())
                    results["telegram"] = {"success": success}
                except Exception as e:
                    results["telegram"] = {"success": False, "error": str(e)}
            else:
                results["telegram"] = {"success": False, "error": "Telegram not configured"}

        all_success = all(r.get("success", False) for r in results.values())
        return json_response(
            {
                "success": all_success,
                "results": results,
            }
        )


# Utility functions for use by other handlers/orchestrator
async def notify_debate_completed(result: Any) -> dict[str, bool]:
    """Notify all configured channels about a completed debate.

    This function is designed to be called from the debate orchestrator
    after a debate completes.

    Args:
        result: DebateResult object

    Returns:
        Dict with success status for each channel
    """
    results = {}

    email = get_email_integration()
    if email:
        try:
            sent = await email.send_debate_summary(result)
            results["email"] = sent > 0
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            results["email"] = False

    telegram = get_telegram_integration()
    if telegram:
        try:
            success = await telegram.post_debate_summary(result)
            results["telegram"] = success
        except Exception as e:
            logger.error(f"Failed to send telegram notification: {e}")
            results["telegram"] = False

    return results


async def notify_consensus_reached(
    debate_id: str,
    confidence: float,
    winner: Optional[str] = None,
    task: Optional[str] = None,
) -> dict[str, bool]:
    """Notify all configured channels about consensus being reached.

    Args:
        debate_id: ID of the debate
        confidence: Consensus confidence score
        winner: Winning agent name
        task: Task description

    Returns:
        Dict with success status for each channel
    """
    results = {}

    email = get_email_integration()
    if email:
        try:
            sent = await email.send_consensus_alert(debate_id, confidence, winner, task)
            results["email"] = sent > 0
        except Exception as e:
            logger.error(f"Failed to send email consensus alert: {e}")
            results["email"] = False

    telegram = get_telegram_integration()
    if telegram:
        try:
            success = await telegram.send_consensus_alert(debate_id, confidence, winner, task)
            results["telegram"] = success
        except Exception as e:
            logger.error(f"Failed to send telegram consensus alert: {e}")
            results["telegram"] = False

    return results


async def notify_error(
    error_type: str,
    error_message: str,
    debate_id: Optional[str] = None,
    severity: str = "warning",
) -> dict[str, bool]:
    """Notify configured channels about an error.

    Args:
        error_type: Type of error
        error_message: Error details
        debate_id: Optional debate ID
        severity: One of "info", "warning", "error", "critical"

    Returns:
        Dict with success status for each channel
    """
    results = {}

    telegram = get_telegram_integration()
    if telegram:
        try:
            success = await telegram.send_error_alert(
                error_type, error_message, debate_id, severity
            )
            results["telegram"] = success
        except Exception as e:
            logger.error(f"Failed to send telegram error alert: {e}")
            results["telegram"] = False

    return results


__all__ = [
    "NotificationsHandler",
    "get_email_integration",
    "get_telegram_integration",
    "configure_email_integration",
    "configure_telegram_integration",
    "notify_debate_completed",
    "notify_consensus_reached",
    "notify_error",
]
