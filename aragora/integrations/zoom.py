"""
Zoom integration for aragora debates.

Provides:
    - Send debate notifications to Zoom chat
    - Join Zoom meetings as a bot (for live debate discussions)
    - Transcribe Zoom meeting recordings
    - Webhook handling for meeting events

Requires:
    ZOOM_CLIENT_ID - OAuth client ID
    ZOOM_CLIENT_SECRET - OAuth client secret
    ZOOM_WEBHOOK_SECRET - Webhook verification token
    ZOOM_BOT_JID - Bot JID for Zoom Chat (optional)

Setup:
    1. Create a Zoom Marketplace app at https://marketplace.zoom.us
    2. Add OAuth credentials
    3. Configure webhook subscriptions for meeting events
    4. Set environment variables

Usage:
    zoom = ZoomIntegration(ZoomConfig.from_env())
    await zoom.send_chat_message(channel_id, "Debate complete!")
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import aiohttp

from aragora.core import DebateResult
from aragora.http_client import DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)


@dataclass
class ZoomConfig:
    """Configuration for Zoom integration."""

    # OAuth credentials
    client_id: str = ""
    client_secret: str = ""
    account_id: str = ""  # For Server-to-Server OAuth

    # Webhook verification
    webhook_secret: str = ""

    # Bot configuration
    bot_jid: str = ""  # For Zoom Chat messaging

    # Meeting bot settings
    enable_meeting_bot: bool = False
    bot_name: str = "Aragora Bot"

    # Notification settings
    notify_on_consensus: bool = True
    notify_on_debate_end: bool = True
    notify_on_error: bool = True

    # Rate limiting
    max_requests_per_minute: int = 30
    max_requests_per_day: int = 1000

    # API settings
    api_base_url: str = "https://api.zoom.us/v2"
    oauth_url: str = "https://zoom.us/oauth/token"

    def __post_init__(self) -> None:
        # Load from environment if not provided
        if not self.client_id:
            self.client_id = os.environ.get("ZOOM_CLIENT_ID", "")
        if not self.client_secret:
            self.client_secret = os.environ.get("ZOOM_CLIENT_SECRET", "")
        if not self.account_id:
            self.account_id = os.environ.get("ZOOM_ACCOUNT_ID", "")
        if not self.webhook_secret:
            self.webhook_secret = os.environ.get("ZOOM_WEBHOOK_SECRET", "")
        if not self.bot_jid:
            self.bot_jid = os.environ.get("ZOOM_BOT_JID", "")

    @classmethod
    def from_env(cls) -> "ZoomConfig":
        """Create config from environment variables."""
        return cls()

    @property
    def is_configured(self) -> bool:
        """Check if Zoom integration is configured."""
        return bool(self.client_id and self.client_secret and self.account_id)


@dataclass
class ZoomMeetingInfo:
    """Information about a Zoom meeting."""

    meeting_id: str
    topic: str
    start_time: Optional[datetime] = None
    duration: int = 0  # minutes
    host_id: str = ""
    join_url: str = ""
    password: str = ""
    recording_url: Optional[str] = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "ZoomMeetingInfo":
        """Create from Zoom API response."""
        start_time = None
        if data.get("start_time"):
            try:
                start_time = datetime.fromisoformat(data["start_time"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return cls(
            meeting_id=str(data.get("id", "")),
            topic=data.get("topic", ""),
            start_time=start_time,
            duration=data.get("duration", 0),
            host_id=data.get("host_id", ""),
            join_url=data.get("join_url", ""),
            password=data.get("password", ""),
        )


@dataclass
class ZoomWebhookEvent:
    """A Zoom webhook event."""

    event_type: str
    payload: dict[str, Any]
    event_ts: int
    account_id: str

    @classmethod
    def from_request(cls, data: dict[str, Any]) -> "ZoomWebhookEvent":
        """Create from webhook request body."""
        return cls(
            event_type=data.get("event", ""),
            payload=data.get("payload", {}),
            event_ts=data.get("event_ts", 0),
            account_id=data.get("payload", {}).get("account_id", ""),
        )


class ZoomIntegration:
    """
    Zoom integration for meetings and chat.

    Supports:
        - Server-to-Server OAuth authentication
        - Sending chat messages
        - Creating meetings
        - Webhook event handling
        - Meeting recording transcription

    Usage:
        zoom = ZoomIntegration(ZoomConfig.from_env())

        # Check if configured
        if zoom.is_configured:
            # Send chat message
            await zoom.send_chat_message(channel_id, "Hello!")

            # Create a meeting
            meeting = await zoom.create_meeting("Debate Discussion")

            # Handle webhook
            event = await zoom.handle_webhook(request_body, signature)
    """

    def __init__(self, config: Optional[ZoomConfig] = None):
        self.config = config or ZoomConfig.from_env()
        self._session: Optional[aiohttp.ClientSession] = None
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        self._request_count_minute = 0
        self._request_count_day = 0
        self._last_minute_reset = datetime.now()
        self._last_day_reset = datetime.now()

    @property
    def is_configured(self) -> bool:
        """Check if Zoom integration is configured."""
        return self.config.is_configured

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
        if (now - self._last_minute_reset).total_seconds() >= 60:
            self._request_count_minute = 0
            self._last_minute_reset = now

        # Reset day counter
        if (now - self._last_day_reset).total_seconds() >= 86400:
            self._request_count_day = 0
            self._last_day_reset = now

        if self._request_count_minute >= self.config.max_requests_per_minute:
            logger.warning("Zoom per-minute rate limit reached")
            return False

        if self._request_count_day >= self.config.max_requests_per_day:
            logger.warning("Zoom daily rate limit reached")
            return False

        self._request_count_minute += 1
        self._request_count_day += 1
        return True

    async def _get_access_token(self) -> str:
        """Get OAuth access token using Server-to-Server OAuth."""
        # Check if we have a valid cached token
        if self._access_token and self._token_expires and datetime.now() < self._token_expires:
            return self._access_token

        if not self.config.is_configured:
            raise ValueError("Zoom credentials not configured")

        session = await self._get_session()

        # Server-to-Server OAuth token request
        auth = aiohttp.BasicAuth(
            self.config.client_id,
            self.config.client_secret,
        )

        data = {
            "grant_type": "account_credentials",
            "account_id": self.config.account_id,
        }

        try:
            async with session.post(
                self.config.oauth_url,
                data=data,
                auth=auth,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"Zoom OAuth error: {response.status} - {text}")
                    raise RuntimeError(f"Failed to get Zoom token: {text}")

                result = await response.json()
                self._access_token = result["access_token"]
                expires_in = result.get("expires_in", 3600)
                self._token_expires = datetime.now() + timedelta(seconds=expires_in - 60)

                return self._access_token

        except aiohttp.ClientError as e:
            logger.error(f"Zoom OAuth connection error: {e}")
            raise

    async def _api_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Make an authenticated API request."""
        if not self._check_rate_limit():
            raise RuntimeError("Rate limit exceeded")

        token = await self._get_access_token()
        session = await self._get_session()

        url = f"{self.config.api_base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        try:
            async with session.request(
                method,
                url,
                json=data,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status >= 400:
                    text = await response.text()
                    logger.error(f"Zoom API error: {response.status} - {text}")
                    raise RuntimeError(f"Zoom API error: {response.status}")

                if response.status == 204:
                    return {}

                result: dict[str, Any] = await response.json()
                return result

        except aiohttp.ClientError as e:
            logger.error(f"Zoom API connection error: {e}")
            raise

    def verify_webhook(self, body: bytes, signature: str, timestamp: str) -> bool:
        """Verify Zoom webhook signature.

        Args:
            body: Raw request body
            signature: x-zm-signature header value
            timestamp: x-zm-request-timestamp header value

        Returns:
            True if signature is valid
        """
        if not self.config.webhook_secret:
            logger.warning("Webhook secret not configured")
            return False

        message = f"v0:{timestamp}:{body.decode()}"
        expected = hmac.new(
            self.config.webhook_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(f"v0={expected}", signature)

    async def handle_webhook(
        self,
        body: bytes,
        signature: str,
        timestamp: str,
    ) -> Optional[ZoomWebhookEvent]:
        """Handle incoming Zoom webhook.

        Args:
            body: Raw request body
            signature: x-zm-signature header
            timestamp: x-zm-request-timestamp header

        Returns:
            ZoomWebhookEvent if valid, None otherwise
        """
        if not self.verify_webhook(body, signature, timestamp):
            logger.warning("Invalid Zoom webhook signature")
            return None

        import json

        try:
            data = json.loads(body)
            return ZoomWebhookEvent.from_request(data)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in webhook body")
            return None

    async def create_meeting(
        self,
        topic: str,
        duration: int = 60,
        start_time: Optional[datetime] = None,
        password: Optional[str] = None,
        settings: Optional[dict[str, Any]] = None,
    ) -> ZoomMeetingInfo:
        """Create a Zoom meeting.

        Args:
            topic: Meeting topic
            duration: Duration in minutes
            start_time: Scheduled start time (instant meeting if None)
            password: Meeting password
            settings: Additional meeting settings

        Returns:
            ZoomMeetingInfo with meeting details
        """
        data: dict[str, Any] = {
            "topic": topic,
            "type": 2 if start_time else 1,  # 1=instant, 2=scheduled
            "duration": duration,
        }

        if start_time:
            data["start_time"] = start_time.isoformat()

        if password:
            data["password"] = password

        if settings:
            data["settings"] = settings

        result = await self._api_request("POST", "/users/me/meetings", data=data)
        return ZoomMeetingInfo.from_api_response(result)

    async def get_meeting(self, meeting_id: str) -> ZoomMeetingInfo:
        """Get meeting details.

        Args:
            meeting_id: Zoom meeting ID

        Returns:
            ZoomMeetingInfo with meeting details
        """
        result = await self._api_request("GET", f"/meetings/{meeting_id}")
        return ZoomMeetingInfo.from_api_response(result)

    async def list_recordings(
        self,
        user_id: str = "me",
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """List cloud recordings.

        Args:
            user_id: User ID or "me"
            from_date: Start date for search
            to_date: End date for search

        Returns:
            List of recording objects
        """
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date.strftime("%Y-%m-%d")
        if to_date:
            params["to"] = to_date.strftime("%Y-%m-%d")

        result = await self._api_request("GET", f"/users/{user_id}/recordings", params=params)
        meetings: list[dict[str, Any]] = result.get("meetings", [])
        return meetings

    async def get_recording_transcript(
        self,
        meeting_id: str,
    ) -> Optional[str]:
        """Get transcript for a meeting recording.

        Args:
            meeting_id: Zoom meeting ID

        Returns:
            Transcript text if available
        """
        try:
            result = await self._api_request("GET", f"/meetings/{meeting_id}/recordings")

            # Find VTT transcript file
            for file in result.get("recording_files", []):
                if file.get("file_type") == "TRANSCRIPT":
                    # Download transcript
                    download_url = file.get("download_url")
                    if download_url:
                        token = await self._get_access_token()
                        session = await self._get_session()
                        async with session.get(
                            download_url,
                            headers={"Authorization": f"Bearer {token}"},
                        ) as response:
                            if response.status == 200:
                                return await response.text()

            return None

        except Exception as e:
            logger.error(f"Failed to get transcript: {e}")
            return None

    async def send_chat_message(
        self,
        to_jid: str,
        message: str,
        is_channel: bool = False,
    ) -> bool:
        """Send a Zoom chat message.

        Args:
            to_jid: Recipient JID (user or channel)
            message: Message text
            is_channel: True if sending to a channel

        Returns:
            True if message sent successfully
        """
        if not self.config.bot_jid:
            logger.warning("Zoom bot JID not configured")
            return False

        endpoint = "/chat/users/me/messages"
        data = {
            "message": message,
            "to_jid": to_jid,
            "robot_jid": self.config.bot_jid,
        }

        if is_channel:
            data["to_channel"] = to_jid
            del data["to_jid"]

        try:
            await self._api_request("POST", endpoint, data=data)
            logger.debug("Zoom chat message sent")
            return True
        except Exception as e:
            logger.error(f"Failed to send Zoom chat: {e}")
            return False

    async def send_debate_summary(
        self,
        to_jid: str,
        result: DebateResult,
        is_channel: bool = False,
    ) -> bool:
        """Send a debate summary via Zoom chat.

        Args:
            to_jid: Recipient JID
            result: Debate result
            is_channel: True if sending to a channel

        Returns:
            True if message sent successfully
        """
        if not self.config.notify_on_debate_end:
            return False

        lines = [
            "**ARAGORA DEBATE COMPLETE**",
            "",
            f"**Question:** {result.task[:200]}",
        ]

        if result.final_answer:
            answer_preview = result.final_answer[:400]
            if len(result.final_answer) > 400:
                answer_preview += "..."
            lines.extend(["", f"**Answer:** {answer_preview}"])

        stats = [f"Rounds: {result.rounds_used}"]
        if result.confidence:
            stats.append(f"Confidence: {result.confidence:.0%}")
        lines.extend(["", " | ".join(stats)])

        lines.extend(["", f"View: https://aragora.ai/debate/{result.debate_id}"])

        return await self.send_chat_message(to_jid, "\n".join(lines), is_channel)

    async def __aenter__(self) -> "ZoomIntegration":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
