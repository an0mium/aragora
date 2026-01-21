"""
WhatsApp integration for aragora debates.

Sends debate summaries and consensus alerts via WhatsApp Business API
or Twilio's WhatsApp API.

Supports:
    - Meta WhatsApp Business API (cloud-hosted)
    - Twilio WhatsApp API (alternative)

Requires one of:
    WHATSAPP_PHONE_NUMBER_ID + WHATSAPP_ACCESS_TOKEN (Meta)
    TWILIO_ACCOUNT_SID + TWILIO_AUTH_TOKEN + TWILIO_WHATSAPP_NUMBER (Twilio)

Usage:
    whatsapp = WhatsAppIntegration(WhatsAppConfig(
        phone_number_id="1234567890",
        access_token="EAAxxxxxxxxx",
        recipient="+1234567890"
    ))
    await whatsapp.send_debate_summary(debate_result)
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import aiohttp

from aragora.core import DebateResult
from aragora.http_client import DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)


class WhatsAppProvider(Enum):
    """WhatsApp API provider."""

    META = "meta"
    TWILIO = "twilio"


@dataclass
class WhatsAppConfig:
    """Configuration for WhatsApp integration."""

    # Meta WhatsApp Business API
    phone_number_id: str = ""
    access_token: str = ""

    # Twilio WhatsApp API
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_number: str = ""

    # Target recipient (required)
    recipient: str = ""

    # Notification settings
    notify_on_consensus: bool = True
    notify_on_debate_end: bool = True
    notify_on_error: bool = True

    # Minimum confidence for consensus alerts
    min_consensus_confidence: float = 0.7

    # Rate limiting
    max_messages_per_minute: int = 5
    max_messages_per_day: int = 100

    # API version (Meta)
    api_version: str = "v18.0"

    def __post_init__(self) -> None:
        # Load from environment if not provided
        if not self.phone_number_id:
            self.phone_number_id = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
        if not self.access_token:
            self.access_token = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
        if not self.twilio_account_sid:
            self.twilio_account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
        if not self.twilio_auth_token:
            self.twilio_auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
        if not self.twilio_whatsapp_number:
            self.twilio_whatsapp_number = os.environ.get("TWILIO_WHATSAPP_NUMBER", "")

    @property
    def provider(self) -> Optional[WhatsAppProvider]:
        """Determine which provider to use based on config."""
        if self.phone_number_id and self.access_token:
            return WhatsAppProvider.META
        elif self.twilio_account_sid and self.twilio_auth_token and self.twilio_whatsapp_number:
            return WhatsAppProvider.TWILIO
        return None


class WhatsAppIntegration:
    """
    WhatsApp integration for sending debate notifications.

    Supports both Meta WhatsApp Business API and Twilio WhatsApp API.

    Usage:
        # Meta WhatsApp Business API
        whatsapp = WhatsAppIntegration(WhatsAppConfig(
            phone_number_id="1234567890",
            access_token="EAAxxxxxxxxx",
            recipient="+1234567890"
        ))

        # Twilio WhatsApp API
        whatsapp = WhatsAppIntegration(WhatsAppConfig(
            twilio_account_sid="ACxxxxxxxxx",
            twilio_auth_token="xxxxxxxxx",
            twilio_whatsapp_number="+14155238886",
            recipient="+1234567890"
        ))

        await whatsapp.send_debate_summary(debate_result)
    """

    # API endpoints
    META_API_BASE = "https://graph.facebook.com"
    TWILIO_API_BASE = "https://api.twilio.com/2010-04-01"

    def __init__(self, config: Optional[WhatsAppConfig] = None):
        self.config = config or WhatsAppConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._message_count_minute = 0
        self._message_count_day = 0
        self._last_minute_reset = datetime.now()
        self._last_day_reset = datetime.now()

    @property
    def is_configured(self) -> bool:
        """Check if WhatsApp integration is configured."""
        return self.config.provider is not None and bool(self.config.recipient)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with timeout protection."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT)
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()

        # Reset minute counter
        minute_elapsed = (now - self._last_minute_reset).total_seconds()
        if minute_elapsed >= 60:
            self._message_count_minute = 0
            self._last_minute_reset = now

        # Reset day counter
        day_elapsed = (now - self._last_day_reset).total_seconds()
        if day_elapsed >= 86400:
            self._message_count_day = 0
            self._last_day_reset = now

        if self._message_count_minute >= self.config.max_messages_per_minute:
            logger.warning("WhatsApp per-minute rate limit reached")
            return False

        if self._message_count_day >= self.config.max_messages_per_day:
            logger.warning("WhatsApp daily rate limit reached")
            return False

        self._message_count_minute += 1
        self._message_count_day += 1
        return True

    def _format_phone_number(self, number: str) -> str:
        """Format phone number for API (remove + prefix for some APIs)."""
        return number.lstrip("+")

    async def _send_via_meta(self, message: str) -> bool:
        """Send message via Meta WhatsApp Business API."""
        session = await self._get_session()
        url = (
            f"{self.META_API_BASE}/{self.config.api_version}/{self.config.phone_number_id}/messages"
        )

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": self._format_phone_number(self.config.recipient),
            "type": "text",
            "text": {"body": message},
        }

        headers = {
            "Authorization": f"Bearer {self.config.access_token}",
            "Content-Type": "application/json",
        }

        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    logger.debug("WhatsApp message sent via Meta API")
                    return True
                else:
                    text = await response.text()
                    logger.error(f"WhatsApp Meta API error: {response.status} - {text}")
                    return False
        except aiohttp.ClientError as e:
            logger.error(f"WhatsApp Meta API connection error: {e}")
            return False
        except asyncio.TimeoutError:
            logger.error("WhatsApp Meta API request timed out")
            return False

    async def _send_via_twilio(self, message: str) -> bool:
        """Send message via Twilio WhatsApp API."""
        session = await self._get_session()
        url = f"{self.TWILIO_API_BASE}/Accounts/{self.config.twilio_account_sid}/Messages.json"

        data = {
            "From": f"whatsapp:{self.config.twilio_whatsapp_number}",
            "To": f"whatsapp:{self.config.recipient}",
            "Body": message,
        }

        auth = aiohttp.BasicAuth(
            self.config.twilio_account_sid,
            self.config.twilio_auth_token,
        )

        try:
            async with session.post(
                url,
                data=data,
                auth=auth,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status in (200, 201):
                    logger.debug("WhatsApp message sent via Twilio")
                    return True
                else:
                    text = await response.text()
                    logger.error(f"Twilio API error: {response.status} - {text}")
                    return False
        except aiohttp.ClientError as e:
            logger.error(f"Twilio API connection error: {e}")
            return False
        except asyncio.TimeoutError:
            logger.error("Twilio API request timed out")
            return False

    async def send_message(self, message: str) -> bool:
        """Send a text message via WhatsApp.

        Args:
            message: The message text to send

        Returns:
            True if message was sent successfully
        """
        if not self.is_configured:
            logger.warning("WhatsApp not configured, skipping message")
            return False

        if not self._check_rate_limit():
            return False

        # Truncate message if too long (WhatsApp limit is ~4096 chars)
        if len(message) > 4000:
            message = message[:3997] + "..."

        if self.config.provider == WhatsAppProvider.META:
            return await self._send_via_meta(message)
        elif self.config.provider == WhatsAppProvider.TWILIO:
            return await self._send_via_twilio(message)
        else:
            logger.error("No WhatsApp provider configured")
            return False

    async def send_debate_summary(self, result: DebateResult) -> bool:
        """Send a debate summary via WhatsApp.

        Args:
            result: The debate result to summarize

        Returns:
            True if message was sent successfully
        """
        if not self.config.notify_on_debate_end:
            return False

        # Build message
        lines = [
            "ARAGORA DEBATE COMPLETE",
            "",
            f"Question: {result.task[:200]}",
        ]

        if result.final_answer:
            answer_preview = result.final_answer[:500]
            if len(result.final_answer) > 500:
                answer_preview += "..."
            lines.extend(["", f"Answer: {answer_preview}"])

        # Stats
        stats = [f"Rounds: {result.rounds_used}"]
        if result.confidence:
            stats.append(f"Confidence: {result.confidence:.0%}")
        if result.participants:
            stats.append(f"Agents: {len(result.participants)}")

        lines.extend(["", " | ".join(stats)])

        # Link
        lines.extend(
            [
                "",
                f"View: https://aragora.ai/debate/{result.debate_id}",
            ]
        )

        return await self.send_message("\n".join(lines))

    async def send_consensus_alert(
        self,
        debate_id: str,
        answer: str,
        confidence: float,
    ) -> bool:
        """Send a consensus reached alert.

        Args:
            debate_id: ID of the debate
            answer: The consensus answer
            confidence: Confidence level (0-1)

        Returns:
            True if message was sent successfully
        """
        if not self.config.notify_on_consensus:
            return False

        if confidence < self.config.min_consensus_confidence:
            return False

        answer_preview = answer[:400]
        if len(answer) > 400:
            answer_preview += "..."

        message = (
            f"CONSENSUS REACHED\n"
            f"\n"
            f"{answer_preview}\n"
            f"\n"
            f"Confidence: {confidence:.0%}\n"
            f"\n"
            f"View: https://aragora.ai/debate/{debate_id}"
        )

        return await self.send_message(message)

    async def send_error_alert(
        self,
        debate_id: str,
        error: str,
    ) -> bool:
        """Send an error notification.

        Args:
            debate_id: ID of the debate
            error: Error message

        Returns:
            True if message was sent successfully
        """
        if not self.config.notify_on_error:
            return False

        message = f"ARAGORA ERROR\n" f"\n" f"Debate: {debate_id}\n" f"Error: {error[:500]}"

        return await self.send_message(message)

    async def __aenter__(self) -> "WhatsAppIntegration":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
