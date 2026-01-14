"""
Email notification integration for aragora debates.

Sends debate summaries, consensus alerts, and digest emails.
Supports HTML email templates with inline CSS.
"""

import asyncio
import logging
import smtplib
import ssl
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Optional

from aragora.core import DebateResult

logger = logging.getLogger(__name__)


@dataclass
class EmailConfig:
    """Configuration for Email integration."""

    # SMTP settings
    smtp_host: str
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    use_tls: bool = True
    use_ssl: bool = False

    # Email settings
    from_email: str = "debates@aragora.ai"
    from_name: str = "Aragora Debates"

    # Notification settings
    notify_on_consensus: bool = True
    notify_on_debate_end: bool = True
    notify_on_error: bool = True

    # Digest settings
    enable_digest: bool = True
    digest_frequency: str = "daily"  # "daily" or "weekly"

    # Minimum confidence for consensus alerts
    min_consensus_confidence: float = 0.7

    # Rate limiting
    max_emails_per_hour: int = 50

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 2.0

    def __post_init__(self) -> None:
        if not self.smtp_host:
            raise ValueError("SMTP host is required")


@dataclass
class EmailRecipient:
    """An email recipient."""

    email: str
    name: Optional[str] = None
    preferences: dict[str, Any] = field(default_factory=dict)

    @property
    def formatted(self) -> str:
        """Get formatted email address."""
        if self.name:
            return f"{self.name} <{self.email}>"
        return self.email


class EmailIntegration:
    """
    Email integration for sending debate notifications.

    Usage:
        email = EmailIntegration(EmailConfig(
            smtp_host="smtp.sendgrid.net",
            smtp_username="apikey",
            smtp_password="your-api-key",
        ))

        # Add recipient
        email.add_recipient(EmailRecipient(
            email="user@example.com",
            name="User Name"
        ))

        # Send debate summary
        await email.send_debate_summary(debate_result)

        # Send daily digest
        await email.send_digest()
    """

    def __init__(self, config: EmailConfig):
        self.config = config
        self.recipients: list[EmailRecipient] = []
        self._email_count = 0
        self._last_reset = datetime.now()
        self._pending_digests: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._rate_limit_lock = asyncio.Lock()  # Thread-safe rate limiting

    def add_recipient(self, recipient: EmailRecipient) -> None:
        """Add an email recipient."""
        self.recipients.append(recipient)

    def remove_recipient(self, email: str) -> bool:
        """Remove a recipient by email address."""
        for i, r in enumerate(self.recipients):
            if r.email == email:
                del self.recipients[i]
                return True
        return False

    async def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits (thread-safe).

        Uses asyncio.Lock to prevent race conditions when multiple
        coroutines check and increment the counter concurrently.
        """
        async with self._rate_limit_lock:
            now = datetime.now()
            elapsed = (now - self._last_reset).total_seconds()

            if elapsed >= 3600:  # 1 hour
                self._email_count = 0
                self._last_reset = now

            if self._email_count >= self.config.max_emails_per_hour:
                logger.warning("Email rate limit reached, skipping email")
                return False

            self._email_count += 1
            return True

    async def _send_email(
        self,
        recipient: EmailRecipient,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None,
    ) -> bool:
        """Send an email with retry logic."""
        if not await self._check_rate_limit():
            return False

        for attempt in range(self.config.max_retries):
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = f"{self.config.from_name} <{self.config.from_email}>"
                msg["To"] = recipient.formatted
                msg["List-Unsubscribe"] = (
                    f"<mailto:unsubscribe@aragora.ai?subject=unsubscribe-{recipient.email}>"
                )

                # Add plain text part
                if text_body:
                    msg.attach(MIMEText(text_body, "plain"))

                # Add HTML part
                msg.attach(MIMEText(html_body, "html"))

                # Send via SMTP (run in thread pool to avoid blocking)
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._smtp_send,
                    msg,
                    recipient.email,
                )

                logger.debug(f"Email sent to {recipient.email}")
                return True

            except smtplib.SMTPException as e:
                logger.error(f"SMTP error sending to {recipient.email}: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2**attempt))
                    continue
                return False
            except Exception as e:
                logger.error(f"Email send failed: {e}")
                return False

        return False

    def _smtp_send(self, msg: MIMEMultipart, to_email: str) -> None:
        """Send email via SMTP (synchronous, called from executor)."""
        if self.config.use_ssl:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(
                self.config.smtp_host, self.config.smtp_port, context=context
            ) as server:
                if self.config.smtp_username:
                    server.login(self.config.smtp_username, self.config.smtp_password)
                server.sendmail(self.config.from_email, to_email, msg.as_string())
        else:
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                if self.config.use_tls:
                    server.starttls()
                if self.config.smtp_username:
                    server.login(self.config.smtp_username, self.config.smtp_password)
                server.sendmail(self.config.from_email, to_email, msg.as_string())

    def _get_email_styles(self) -> str:
        """Get inline CSS styles for emails."""
        return """
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }
            .container { max-width: 600px; margin: 0 auto; padding: 20px; }
            .header { background: linear-gradient(135deg, #00ff00, #00cc00); color: #000; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }
            .content { background: #f9f9f9; padding: 20px; border: 1px solid #e0e0e0; }
            .footer { background: #333; color: #999; padding: 15px; text-align: center; font-size: 12px; border-radius: 0 0 8px 8px; }
            .status-success { color: #00cc00; }
            .status-fail { color: #cc0000; }
            .metric { display: inline-block; background: #fff; border: 1px solid #e0e0e0; padding: 10px 15px; margin: 5px; border-radius: 4px; }
            .metric-label { font-size: 12px; color: #666; display: block; }
            .metric-value { font-size: 18px; font-weight: bold; color: #333; }
            .button { display: inline-block; background: #00cc00; color: #000; padding: 10px 20px; text-decoration: none; border-radius: 4px; font-weight: bold; margin: 10px 0; }
            .code { background: #f0f0f0; padding: 15px; border-radius: 4px; font-family: monospace; font-size: 13px; overflow-x: auto; }
            .divider { border-top: 1px solid #e0e0e0; margin: 20px 0; }
        </style>
        """

    async def send_debate_summary(
        self,
        result: DebateResult,
        recipients: Optional[list[EmailRecipient]] = None,
    ) -> int:
        """Send a debate summary to recipients.

        Args:
            result: The debate result to summarize
            recipients: Optional list of recipients (defaults to all)

        Returns:
            Number of emails sent successfully
        """
        if not self.config.notify_on_debate_end:
            return 0

        recipients = recipients or self.recipients
        if not recipients:
            return 0

        subject = f"Debate Completed: {result.task[:50]}..."
        html_body = self._build_debate_summary_html(result)
        text_body = self._build_debate_summary_text(result)

        # Add to pending digest
        if self.config.enable_digest:
            self._add_to_digest(
                {
                    "type": "debate_summary",
                    "result": result,
                    "timestamp": datetime.now(),
                }
            )

        sent = 0
        for recipient in recipients:
            if await self._send_email(recipient, subject, html_body, text_body):
                sent += 1

        return sent

    def _build_debate_summary_html(self, result: DebateResult) -> str:
        """Build HTML email for debate summary."""
        status_class = "status-success" if result.consensus_reached else "status-fail"
        status_text = "Consensus Reached" if result.consensus_reached else "No Consensus"
        confidence = getattr(result, "confidence", 0.0)

        final_answer_html = ""
        if result.consensus_reached and result.final_answer:
            preview = result.final_answer[:600].replace("\n", "<br>")
            if len(result.final_answer) > 600:
                preview += "..."
            final_answer_html = f"""
            <div class="divider"></div>
            <h3>Final Proposal</h3>
            <div class="code">{preview}</div>
            """

        debate_id = getattr(result, "debate_id", "unknown")

        return f"""
        <!DOCTYPE html>
        <html>
        <head>{self._get_email_styles()}</head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Debate Completed</h1>
                </div>
                <div class="content">
                    <h2>{result.task}</h2>

                    <div style="text-align: center; margin: 20px 0;">
                        <div class="metric">
                            <span class="metric-label">Status</span>
                            <span class="metric-value {status_class}">{status_text}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Confidence</span>
                            <span class="metric-value">{confidence:.0%}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Rounds</span>
                            <span class="metric-value">{result.rounds_used}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Winner</span>
                            <span class="metric-value">{result.winner or "None"}</span>
                        </div>
                    </div>

                    {final_answer_html}

                    <div style="text-align: center; margin-top: 20px;">
                        <a href="https://aragora.ai/debate/{debate_id}" class="button">View Full Debate</a>
                    </div>
                </div>
                <div class="footer">
                    <p>Aragora AI Debate System</p>
                    <p><a href="mailto:unsubscribe@aragora.ai?subject=unsubscribe" style="color: #666;">Unsubscribe</a></p>
                </div>
            </div>
        </body>
        </html>
        """

    def _build_debate_summary_text(self, result: DebateResult) -> str:
        """Build plain text email for debate summary."""
        status = "REACHED" if result.consensus_reached else "NOT REACHED"
        confidence = getattr(result, "confidence", 0.0)

        lines = [
            "DEBATE COMPLETED",
            "=" * 40,
            "",
            f"Task: {result.task}",
            "",
            f"Consensus: {status}",
            f"Confidence: {confidence:.0%}",
            f"Rounds: {result.rounds_used}",
            f"Winner: {result.winner or 'None'}",
        ]

        if result.consensus_reached and result.final_answer:
            preview = result.final_answer[:400]
            if len(result.final_answer) > 400:
                preview += "..."
            lines.extend(["", "Final Proposal:", preview])

        lines.extend(
            [
                "",
                "-" * 40,
                "Aragora AI Debate System",
                "To unsubscribe, reply with UNSUBSCRIBE",
            ]
        )

        return "\n".join(lines)

    async def send_consensus_alert(
        self,
        debate_id: str,
        confidence: float,
        winner: Optional[str] = None,
        task: Optional[str] = None,
        recipients: Optional[list[EmailRecipient]] = None,
    ) -> int:
        """Send a consensus reached notification.

        Returns:
            Number of emails sent successfully
        """
        if not self.config.notify_on_consensus:
            return 0

        if confidence < self.config.min_consensus_confidence:
            return 0

        recipients = recipients or self.recipients
        if not recipients:
            return 0

        subject = f"Consensus Reached! ({confidence:.0%} confidence)"
        html_body = self._build_consensus_alert_html(debate_id, confidence, winner, task)
        text_body = self._build_consensus_alert_text(debate_id, confidence, winner, task)

        sent = 0
        for recipient in recipients:
            if await self._send_email(recipient, subject, html_body, text_body):
                sent += 1

        return sent

    def _build_consensus_alert_html(
        self,
        debate_id: str,
        confidence: float,
        winner: Optional[str],
        task: Optional[str],
    ) -> str:
        """Build HTML email for consensus alert."""
        task_html = f"<p><strong>Task:</strong> {task}</p>" if task else ""
        winner_html = f"<p><strong>Winner:</strong> {winner}</p>" if winner else ""

        return f"""
        <!DOCTYPE html>
        <html>
        <head>{self._get_email_styles()}</head>
        <body>
            <div class="container">
                <div class="header" style="background: linear-gradient(135deg, #ffd700, #ffcc00);">
                    <h1>Consensus Reached!</h1>
                </div>
                <div class="content">
                    {task_html}

                    <div style="text-align: center; margin: 20px 0;">
                        <div class="metric">
                            <span class="metric-label">Debate</span>
                            <span class="metric-value">{debate_id[:8]}...</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Confidence</span>
                            <span class="metric-value status-success">{confidence:.0%}</span>
                        </div>
                    </div>

                    {winner_html}

                    <div style="text-align: center;">
                        <a href="https://aragora.ai/debate/{debate_id}" class="button">View Debate</a>
                    </div>
                </div>
                <div class="footer">
                    <p>Aragora AI Debate System</p>
                </div>
            </div>
        </body>
        </html>
        """

    def _build_consensus_alert_text(
        self,
        debate_id: str,
        confidence: float,
        winner: Optional[str],
        task: Optional[str],
    ) -> str:
        """Build plain text email for consensus alert."""
        lines = [
            "CONSENSUS REACHED!",
            "=" * 40,
            "",
            f"Debate: {debate_id[:8]}...",
            f"Confidence: {confidence:.0%}",
        ]
        if winner:
            lines.append(f"Winner: {winner}")
        if task:
            lines.extend(["", f"Task: {task}"])
        lines.extend(
            [
                "",
                f"View: https://aragora.ai/debate/{debate_id}",
            ]
        )
        return "\n".join(lines)

    def _add_to_digest(self, item: dict[str, Any]) -> None:
        """Add an item to the pending digest."""
        date_key = datetime.now().strftime("%Y-%m-%d")
        self._pending_digests[date_key].append(item)

    async def send_digest(
        self,
        recipients: Optional[list[EmailRecipient]] = None,
    ) -> int:
        """Send a digest of recent debates.

        Returns:
            Number of emails sent successfully
        """
        if not self.config.enable_digest:
            return 0

        recipients = recipients or self.recipients
        if not recipients:
            return 0

        # Get items for digest
        items: list[dict[str, Any]] = []
        cutoff = datetime.now() - timedelta(
            days=7 if self.config.digest_frequency == "weekly" else 1
        )

        for date_key, day_items in list(self._pending_digests.items()):
            date = datetime.strptime(date_key, "%Y-%m-%d")
            if date >= cutoff:
                items.extend(day_items)
            else:
                # Clean up old items
                del self._pending_digests[date_key]

        if not items:
            return 0

        subject = f"Aragora Debate Digest - {len(items)} debates"
        html_body = self._build_digest_html(items)
        text_body = self._build_digest_text(items)

        sent = 0
        for recipient in recipients:
            if await self._send_email(recipient, subject, html_body, text_body):
                sent += 1

        return sent

    def _build_digest_html(self, items: list[dict[str, Any]]) -> str:
        """Build HTML email for digest."""
        debates_html = ""
        for item in items:
            if item["type"] == "debate_summary":
                result = item["result"]
                status = "Consensus" if result.consensus_reached else "No Consensus"
                debates_html += f"""
                <div style="border-bottom: 1px solid #e0e0e0; padding: 15px 0;">
                    <strong>{result.task[:80]}{"..." if len(result.task) > 80 else ""}</strong>
                    <br>
                    <span style="color: #666;">Status: {status} | Winner: {result.winner or "None"}</span>
                </div>
                """

        return f"""
        <!DOCTYPE html>
        <html>
        <head>{self._get_email_styles()}</head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Debate Digest</h1>
                    <p>{len(items)} debates in the last {self.config.digest_frequency.replace("ly", "")}</p>
                </div>
                <div class="content">
                    {debates_html}
                    <div style="text-align: center; margin-top: 20px;">
                        <a href="https://aragora.ai/debates" class="button">View All Debates</a>
                    </div>
                </div>
                <div class="footer">
                    <p>Aragora AI Debate System</p>
                    <p><a href="mailto:unsubscribe@aragora.ai?subject=unsubscribe" style="color: #666;">Unsubscribe</a></p>
                </div>
            </div>
        </body>
        </html>
        """

    def _build_digest_text(self, items: list[dict[str, Any]]) -> str:
        """Build plain text email for digest."""
        lines = [
            "ARAGORA DEBATE DIGEST",
            f"{len(items)} debates in the last {self.config.digest_frequency.replace('ly', '')}",
            "=" * 40,
            "",
        ]

        for item in items:
            if item["type"] == "debate_summary":
                result = item["result"]
                status = "Consensus" if result.consensus_reached else "No Consensus"
                lines.extend(
                    [
                        f"- {result.task[:60]}{'...' if len(result.task) > 60 else ''}",
                        f"  Status: {status} | Winner: {result.winner or 'None'}",
                        "",
                    ]
                )

        lines.extend(
            [
                "View all: https://aragora.ai/debates",
                "",
                "To unsubscribe, reply with UNSUBSCRIBE",
            ]
        )

        return "\n".join(lines)


__all__ = [
    "EmailConfig",
    "EmailRecipient",
    "EmailIntegration",
]
