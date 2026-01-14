"""
Billing Notifications for Aragora.

Handles payment failure notifications, trial expiration warnings, and dunning emails.

Environment Variables:
    ARAGORA_SMTP_HOST: SMTP server host
    ARAGORA_SMTP_PORT: SMTP server port (default: 587)
    ARAGORA_SMTP_USER: SMTP username
    ARAGORA_SMTP_PASSWORD: SMTP password
    ARAGORA_SMTP_FROM: From email address
    ARAGORA_NOTIFICATION_WEBHOOK: Optional webhook URL for notifications
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# SMTP Configuration
SMTP_HOST = os.environ.get("ARAGORA_SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("ARAGORA_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("ARAGORA_SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("ARAGORA_SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("ARAGORA_SMTP_FROM", "billing@aragora.ai")

# Webhook for external notification systems (Slack, etc.)
NOTIFICATION_WEBHOOK = os.environ.get("ARAGORA_NOTIFICATION_WEBHOOK", "")


@dataclass
class NotificationResult:
    """Result of sending a notification."""

    success: bool
    method: str  # "email", "webhook", "log"
    error: Optional[str] = None


class BillingNotifier:
    """
    Handles billing notifications for payment failures and trial expiration.

    Supports multiple notification channels:
    - Email via SMTP
    - Webhook (for Slack, Discord, etc.)
    - Logging fallback
    """

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        smtp_from: Optional[str] = None,
        webhook_url: Optional[str] = None,
    ):
        self.smtp_host = smtp_host or SMTP_HOST
        self.smtp_port = smtp_port or SMTP_PORT
        self.smtp_user = smtp_user or SMTP_USER
        self.smtp_password = smtp_password or SMTP_PASSWORD
        self.smtp_from = smtp_from or SMTP_FROM
        self.webhook_url = webhook_url or NOTIFICATION_WEBHOOK

    def _is_smtp_configured(self) -> bool:
        """Check if SMTP is configured."""
        return bool(self.smtp_host and self.smtp_user and self.smtp_password)

    def _send_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None,
    ) -> NotificationResult:
        """Send email via SMTP."""
        if not self._is_smtp_configured():
            return NotificationResult(
                success=False,
                method="email",
                error="SMTP not configured",
            )

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.smtp_from
            msg["To"] = to_email

            # Add text and HTML parts
            if text_body:
                msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            # Create secure connection
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.smtp_from, to_email, msg.as_string())

            logger.info(f"Sent billing email to {to_email}: {subject}")
            return NotificationResult(success=True, method="email")

        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return NotificationResult(success=False, method="email", error=str(e))

    def _send_webhook(self, payload: dict) -> NotificationResult:
        """Send notification via webhook."""
        if not self.webhook_url:
            return NotificationResult(
                success=False,
                method="webhook",
                error="Webhook not configured",
            )

        try:
            data = json.dumps(payload).encode("utf-8")
            req = Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=10) as response:
                response.read()

            logger.info(f"Sent webhook notification: {payload.get('event')}")
            return NotificationResult(success=True, method="webhook")

        except URLError as e:
            logger.error(f"Failed to send webhook: {e}")
            return NotificationResult(success=False, method="webhook", error=str(e))

    def notify_payment_failed(
        self,
        org_id: str,
        org_name: str,
        email: str,
        attempt_count: int = 1,
        invoice_url: Optional[str] = None,
        days_until_downgrade: int = 7,
    ) -> NotificationResult:
        """
        Send payment failure notification.

        Args:
            org_id: Organization ID
            org_name: Organization name
            email: Email address to notify
            attempt_count: Number of failed payment attempts
            invoice_url: URL to the failed invoice
            days_until_downgrade: Days until subscription downgrade (default 7)

        Returns:
            NotificationResult indicating success/failure
        """
        subject = "[Aragora] Payment Failed - Action Required"

        # Determine urgency based on attempt count
        if attempt_count >= 3:
            urgency = "URGENT"
            urgency_message = (
                "This is your final notice. Your subscription will be suspended "
                "if payment is not received within 48 hours."
            )
        elif attempt_count >= 2:
            urgency = "IMPORTANT"
            urgency_message = (
                "This is a follow-up notice. Please update your payment method "
                "to avoid service interruption."
            )
        else:
            urgency = "NOTICE"
            urgency_message = (
                "We were unable to process your payment. Please update your payment information."
            )

        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Monaco', 'Menlo', monospace; background: #0a0a0a; color: #00ff00; padding: 20px; }}
        .container {{ max-width: 600px; margin: 0 auto; border: 1px solid #00ff00; padding: 20px; }}
        .header {{ font-size: 18px; margin-bottom: 20px; }}
        .urgency {{ color: {"#ff6600" if urgency != "NOTICE" else "#00ff00"}; font-weight: bold; }}
        .message {{ margin: 20px 0; line-height: 1.6; }}
        .button {{ display: inline-block; padding: 10px 20px; background: #00ff00; color: #0a0a0a; text-decoration: none; margin-top: 20px; }}
        .footer {{ margin-top: 30px; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">[ARAGORA BILLING]</div>
        <div class="urgency">{urgency}: Payment Failed</div>
        <div class="message">
            <p>Hi {org_name},</p>
            <p>{urgency_message}</p>
            <p><strong>Organization:</strong> {org_name}</p>
            <p><strong>Attempt:</strong> {attempt_count} of 3</p>
            {'<p><a href="' + invoice_url + '" class="button">UPDATE PAYMENT</a></p>' if invoice_url else ""}
        </div>
        <div class="footer">
            <p>If you believe this is an error, please contact support@aragora.ai</p>
            <p>— Aragora Billing System</p>
        </div>
    </div>
</body>
</html>
"""

        text_body = f"""
[ARAGORA BILLING]

{urgency}: Payment Failed

Hi {org_name},

{urgency_message}

Organization: {org_name}
Attempt: {attempt_count} of 3

{"Update your payment here: " + invoice_url if invoice_url else "Please log in to update your payment method."}

If you believe this is an error, please contact support@aragora.ai

— Aragora Billing System
"""

        # Try to send email first
        result = self._send_email(email, subject, html_body, text_body)
        if result.success:
            return result

        # Fall back to webhook
        webhook_result = self._send_webhook(
            {
                "event": "payment_failed",
                "org_id": org_id,
                "org_name": org_name,
                "email": email,
                "attempt_count": attempt_count,
                "urgency": urgency,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        if webhook_result.success:
            return webhook_result

        # Log as final fallback
        logger.warning(
            f"PAYMENT_FAILED: org={org_id} name={org_name} email={email} "
            f"attempt={attempt_count} urgency={urgency}"
        )
        return NotificationResult(success=True, method="log")

    def notify_trial_ending(
        self,
        org_id: str,
        org_name: str,
        email: str,
        days_remaining: int,
        trial_end: datetime,
    ) -> NotificationResult:
        """
        Send trial expiration warning notification.

        Args:
            org_id: Organization ID
            org_name: Organization name
            email: Email address to notify
            days_remaining: Days until trial expires
            trial_end: Trial end datetime

        Returns:
            NotificationResult indicating success/failure
        """
        if days_remaining <= 1:
            subject = "[Aragora] Your Trial Ends Tomorrow!"
            urgency = "URGENT"
        elif days_remaining <= 3:
            subject = f"[Aragora] Your Trial Ends in {days_remaining} Days"
            urgency = "REMINDER"
        else:
            subject = f"[Aragora] {days_remaining} Days Left in Your Trial"
            urgency = "INFO"

        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Monaco', 'Menlo', monospace; background: #0a0a0a; color: #00ff00; padding: 20px; }}
        .container {{ max-width: 600px; margin: 0 auto; border: 1px solid #00ffff; padding: 20px; }}
        .header {{ font-size: 18px; margin-bottom: 20px; color: #00ffff; }}
        .highlight {{ color: #00ffff; font-size: 24px; margin: 20px 0; }}
        .message {{ margin: 20px 0; line-height: 1.6; }}
        .button {{ display: inline-block; padding: 10px 20px; background: #00ffff; color: #0a0a0a; text-decoration: none; margin-top: 20px; }}
        .footer {{ margin-top: 30px; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">[ARAGORA TRIAL]</div>
        <div class="highlight">{days_remaining} Days Remaining</div>
        <div class="message">
            <p>Hi {org_name},</p>
            <p>Your Aragora trial will end on {trial_end.strftime("%B %d, %Y")}.</p>
            <p>Upgrade now to keep access to:</p>
            <ul>
                <li>Unlimited AI debates</li>
                <li>Advanced analytics</li>
                <li>Team collaboration</li>
                <li>API access</li>
            </ul>
            <p><a href="https://aragora.ai/pricing" class="button">UPGRADE NOW</a></p>
        </div>
        <div class="footer">
            <p>Questions? Contact us at support@aragora.ai</p>
            <p>— Aragora Team</p>
        </div>
    </div>
</body>
</html>
"""

        text_body = f"""
[ARAGORA TRIAL]

{days_remaining} Days Remaining

Hi {org_name},

Your Aragora trial will end on {trial_end.strftime("%B %d, %Y")}.

Upgrade now to keep access to:
- Unlimited AI debates
- Advanced analytics
- Team collaboration
- API access

Upgrade at: https://aragora.ai/pricing

Questions? Contact us at support@aragora.ai

— Aragora Team
"""

        # Try to send email first
        result = self._send_email(email, subject, html_body, text_body)
        if result.success:
            return result

        # Fall back to webhook
        webhook_result = self._send_webhook(
            {
                "event": "trial_ending",
                "org_id": org_id,
                "org_name": org_name,
                "email": email,
                "days_remaining": days_remaining,
                "trial_end": trial_end.isoformat(),
                "urgency": urgency,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        if webhook_result.success:
            return webhook_result

        # Log as final fallback
        logger.info(
            f"TRIAL_ENDING: org={org_id} name={org_name} email={email} "
            f"days_remaining={days_remaining}"
        )
        return NotificationResult(success=True, method="log")

    def notify_subscription_canceled(
        self,
        org_id: str,
        org_name: str,
        email: str,
        reason: Optional[str] = None,
    ) -> NotificationResult:
        """
        Send subscription cancellation confirmation.

        Args:
            org_id: Organization ID
            org_name: Organization name
            email: Email address to notify
            reason: Optional cancellation reason

        Returns:
            NotificationResult indicating success/failure
        """
        subject = "[Aragora] Subscription Canceled - We're Sorry to See You Go"

        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Monaco', 'Menlo', monospace; background: #0a0a0a; color: #00ff00; padding: 20px; }}
        .container {{ max-width: 600px; margin: 0 auto; border: 1px solid #ff6600; padding: 20px; }}
        .header {{ font-size: 18px; margin-bottom: 20px; color: #ff6600; }}
        .message {{ margin: 20px 0; line-height: 1.6; }}
        .button {{ display: inline-block; padding: 10px 20px; background: #00ff00; color: #0a0a0a; text-decoration: none; margin-top: 20px; }}
        .footer {{ margin-top: 30px; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">[ARAGORA]</div>
        <div class="message">
            <p>Hi {org_name},</p>
            <p>Your Aragora subscription has been canceled.</p>
            <p>Your access will continue until the end of your current billing period.</p>
            <p>Changed your mind? You can reactivate anytime:</p>
            <p><a href="https://aragora.ai/billing" class="button">REACTIVATE</a></p>
            <p>We'd love to hear your feedback. What could we have done better?</p>
        </div>
        <div class="footer">
            <p>— Aragora Team</p>
        </div>
    </div>
</body>
</html>
"""

        text_body = f"""
[ARAGORA]

Hi {org_name},

Your Aragora subscription has been canceled.

Your access will continue until the end of your current billing period.

Changed your mind? You can reactivate anytime at https://aragora.ai/billing

We'd love to hear your feedback. What could we have done better?

— Aragora Team
"""

        # Try to send email
        result = self._send_email(email, subject, html_body, text_body)
        if result.success:
            return result

        # Fall back to webhook
        webhook_result = self._send_webhook(
            {
                "event": "subscription_canceled",
                "org_id": org_id,
                "org_name": org_name,
                "email": email,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        if webhook_result.success:
            return webhook_result

        # Log as final fallback
        logger.info(
            f"SUBSCRIPTION_CANCELED: org={org_id} name={org_name} email={email} reason={reason}"
        )
        return NotificationResult(success=True, method="log")


# Default notifier instance
_default_notifier: Optional[BillingNotifier] = None


def get_billing_notifier() -> BillingNotifier:
    """Get the default billing notifier instance."""
    global _default_notifier
    if _default_notifier is None:
        _default_notifier = BillingNotifier()
    return _default_notifier


__all__ = [
    "BillingNotifier",
    "NotificationResult",
    "get_billing_notifier",
]
