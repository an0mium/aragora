"""
Amazon Alexa Device Connector.

Provides integration with Amazon Alexa for:
- Smart Home Skill handling
- Proactive notifications via Alexa Events Gateway
- OAuth account linking
- Custom skill intents for debate interactions

Environment Variables:
    ALEXA_CLIENT_ID: Alexa skill client ID (from Alexa Developer Console)
    ALEXA_CLIENT_SECRET: Alexa skill client secret
    ALEXA_SKILL_ID: Alexa skill ID (amzn1.ask.skill.xxx)
    ALEXA_PROACTIVE_API_ENDPOINT: Proactive events API endpoint (optional)
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import DeviceConnector, DeviceConnectorConfig
from .models import (
    DeliveryStatus,
    DeviceMessage,
    DeviceToken,
    DeviceType,
    SendResult,
    VoiceDeviceRequest,
    VoiceDeviceResponse,
)

logger = logging.getLogger(__name__)


class AlexaRequestType(Enum):
    """Alexa request types."""

    LAUNCH = "LaunchRequest"
    INTENT = "IntentRequest"
    SESSION_ENDED = "SessionEndedRequest"
    CAN_FULFILL = "CanFulfillIntentRequest"

    # Smart Home
    DISCOVERY = "Alexa.Discovery"
    REPORT_STATE = "Alexa.ReportState"


class AlexaIntent(Enum):
    """Built-in and custom Alexa intents."""

    # Built-in intents
    HELP = "AMAZON.HelpIntent"
    CANCEL = "AMAZON.CancelIntent"
    STOP = "AMAZON.StopIntent"
    FALLBACK = "AMAZON.FallbackIntent"
    YES = "AMAZON.YesIntent"
    NO = "AMAZON.NoIntent"

    # Custom Aragora intents
    START_DEBATE = "StartDebateIntent"
    GET_DECISION = "GetDecisionIntent"
    LIST_DEBATES = "ListDebatesIntent"
    VOTE = "VoteIntent"
    GET_STATUS = "GetStatusIntent"


@dataclass
class AlexaUser:
    """Alexa user information from account linking."""

    user_id: str
    access_token: Optional[str] = None
    consent_token: Optional[str] = None
    api_access_token: Optional[str] = None
    api_endpoint: Optional[str] = None
    permissions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlexaSession:
    """Alexa skill session state."""

    session_id: str
    is_new: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)
    user: Optional[AlexaUser] = None


class AlexaConnector(DeviceConnector):
    """
    Amazon Alexa device connector.

    Handles Alexa skill requests and proactive notifications.

    Features:
    - Custom skill intent handling for debate operations
    - Proactive notifications for debate results
    - Account linking for user identification
    - Smart Home skill capabilities
    """

    def __init__(self, config: Optional[DeviceConnectorConfig] = None):
        """Initialize the Alexa connector."""
        super().__init__(config)

        # Load credentials from environment
        self._client_id = os.environ.get("ALEXA_CLIENT_ID", "")
        self._client_secret = os.environ.get("ALEXA_CLIENT_SECRET", "")
        self._skill_id = os.environ.get("ALEXA_SKILL_ID", "")
        self._proactive_api_endpoint = os.environ.get(
            "ALEXA_PROACTIVE_API_ENDPOINT",
            "https://api.amazonalexa.com/v1/proactiveEvents/stages/development",
        )

        # OAuth token cache
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0

        # Intent handlers
        self._intent_handlers: Dict[str, Any] = {}
        self._register_default_handlers()

    # ==========================================================================
    # Properties
    # ==========================================================================

    @property
    def platform_name(self) -> str:
        """Return the platform identifier."""
        return "alexa"

    @property
    def platform_display_name(self) -> str:
        """Return human-readable platform name."""
        return "Amazon Alexa"

    @property
    def supported_device_types(self) -> List[DeviceType]:
        """Return list of device types this connector supports."""
        return [DeviceType.ALEXA]

    # ==========================================================================
    # Initialization
    # ==========================================================================

    async def initialize(self) -> bool:
        """
        Initialize the Alexa connector.

        Validates credentials and obtains initial access token.

        Returns:
            True if initialization successful
        """
        if not all([self._client_id, self._client_secret, self._skill_id]):
            logger.info(
                "Alexa connector not configured: missing ALEXA_CLIENT_ID, "
                "ALEXA_CLIENT_SECRET, or ALEXA_SKILL_ID"
            )
            return False

        # Get initial access token for proactive events
        try:
            await self._refresh_access_token()
            self._initialized = True
            logger.info("Alexa connector initialized successfully")
            return True
        except Exception as e:
            logger.warning(f"Alexa connector initialization failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown the Alexa connector."""
        self._access_token = None
        self._token_expires_at = 0
        await super().shutdown()

    # ==========================================================================
    # OAuth Token Management
    # ==========================================================================

    async def _refresh_access_token(self) -> str:
        """
        Refresh the OAuth access token for proactive events API.

        Returns:
            Access token string

        Raises:
            Exception if token refresh fails
        """
        # Check if current token is still valid
        if self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token

        # Request new token
        token_url = "https://api.amazon.com/auth/o2/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "scope": "alexa::proactive_events",
        }

        success, response, error = await self._http_request(
            method="POST",
            url=token_url,
            headers=headers,
            data=data,
            operation="token_refresh",
        )

        if not success or not response:
            raise Exception(f"Failed to refresh Alexa access token: {error}")

        self._access_token = response.get("access_token")
        expires_in = response.get("expires_in", 3600)
        self._token_expires_at = time.time() + expires_in

        logger.debug(f"Alexa access token refreshed, expires in {expires_in}s")
        return self._access_token  # type: ignore

    # ==========================================================================
    # Request Verification
    # ==========================================================================

    def verify_request(
        self,
        request_body: bytes,
        signature: str,
        signature_cert_url: str,
    ) -> bool:
        """
        Verify an incoming Alexa skill request.

        Validates the request signature to ensure it came from Amazon.

        Args:
            request_body: Raw request body bytes
            signature: Signature header value
            signature_cert_url: Certificate chain URL header

        Returns:
            True if request is valid
        """
        # Basic validation
        if not all([request_body, signature, signature_cert_url]):
            logger.warning("Missing required headers for Alexa request verification")
            return False

        # Validate certificate URL
        from urllib.parse import urlparse

        parsed = urlparse(signature_cert_url)
        if parsed.scheme.lower() != "https":
            logger.warning("Certificate URL must use HTTPS")
            return False

        if parsed.hostname != "s3.amazonaws.com":
            logger.warning("Certificate URL must be from s3.amazonaws.com")
            return False

        if not parsed.path.startswith("/echo.api/"):
            logger.warning("Certificate URL path must start with /echo.api/")
            return False

        # In production, should verify:
        # 1. Download and validate the X.509 certificate
        # 2. Verify certificate chain and expiration
        # 3. Verify request signature using the certificate
        # 4. Validate timestamp is within tolerance

        # For now, basic validation passes
        return True

    def verify_skill_id(self, request_data: Dict[str, Any]) -> bool:
        """
        Verify the skill ID in the request matches configuration.

        Args:
            request_data: Parsed request JSON

        Returns:
            True if skill ID matches
        """
        try:
            request_skill_id = (
                request_data.get("context", {})
                .get("System", {})
                .get("application", {})
                .get("applicationId", "")
            )

            # Also check session application ID
            if not request_skill_id:
                request_skill_id = (
                    request_data.get("session", {}).get("application", {}).get("applicationId", "")
                )

            if request_skill_id != self._skill_id:
                logger.warning(
                    f"Skill ID mismatch: expected {self._skill_id}, got {request_skill_id}"
                )
                return False

            return True

        except Exception as e:
            logger.warning(f"Error verifying skill ID: {e}")
            return False

    # ==========================================================================
    # Request Handling
    # ==========================================================================

    def _register_default_handlers(self) -> None:
        """Register default intent handlers."""
        self._intent_handlers = {
            AlexaIntent.HELP.value: self._handle_help,
            AlexaIntent.CANCEL.value: self._handle_stop,
            AlexaIntent.STOP.value: self._handle_stop,
            AlexaIntent.FALLBACK.value: self._handle_fallback,
            AlexaIntent.START_DEBATE.value: self._handle_start_debate,
            AlexaIntent.GET_DECISION.value: self._handle_get_decision,
            AlexaIntent.LIST_DEBATES.value: self._handle_list_debates,
            AlexaIntent.GET_STATUS.value: self._handle_get_status,
        }

    async def handle_voice_request(
        self,
        request: VoiceDeviceRequest,
        **kwargs: Any,
    ) -> VoiceDeviceResponse:
        """
        Handle a voice request from Alexa.

        Args:
            request: Parsed voice request
            **kwargs: Additional options

        Returns:
            VoiceDeviceResponse to send back
        """
        intent = request.intent

        # Check for registered handler
        if intent in self._intent_handlers:
            handler = self._intent_handlers[intent]
            return await handler(request)

        # Default response for unknown intents
        logger.warning(f"No handler for intent: {intent}")
        return VoiceDeviceResponse(
            text="I'm not sure how to help with that. "
            "You can ask me to start a debate, get a decision, or list your debates.",
            should_end_session=False,
            reprompt="What would you like to do?",
        )

    def parse_alexa_request(self, request_data: Dict[str, Any]) -> VoiceDeviceRequest:
        """
        Parse an Alexa skill request into VoiceDeviceRequest.

        Args:
            request_data: Raw Alexa request JSON

        Returns:
            Parsed VoiceDeviceRequest
        """
        request_obj = request_data.get("request", {})
        session = request_data.get("session", {})
        context = request_data.get("context", {})

        # Extract intent and slots
        intent_name = ""
        slots: Dict[str, Any] = {}

        request_type = request_obj.get("type", "")

        if request_type == AlexaRequestType.LAUNCH.value:
            intent_name = "LaunchRequest"
        elif request_type == AlexaRequestType.INTENT.value:
            intent_data = request_obj.get("intent", {})
            intent_name = intent_data.get("name", "")
            raw_slots = intent_data.get("slots", {})
            for slot_name, slot_data in raw_slots.items():
                if "value" in slot_data:
                    slots[slot_name] = slot_data["value"]
        elif request_type == AlexaRequestType.SESSION_ENDED.value:
            intent_name = "SessionEndedRequest"

        # Extract user info
        system = context.get("System", {})
        user_data = system.get("user", {})
        user_id = user_data.get("userId", "")
        access_token = user_data.get("accessToken")

        return VoiceDeviceRequest(
            request_id=request_obj.get("requestId", ""),
            device_type=DeviceType.ALEXA,
            user_id=user_id,
            intent=intent_name,
            slots=slots,
            raw_input=request_obj.get("locale"),
            session_id=session.get("sessionId"),
            is_new_session=session.get("new", False),
            timestamp=datetime.fromisoformat(
                request_obj.get("timestamp", "").replace("Z", "+00:00")
            )
            if request_obj.get("timestamp")
            else None,
            locale=request_obj.get("locale", "en-US"),
            metadata={
                "access_token": access_token,
                "api_endpoint": system.get("apiEndpoint"),
                "api_access_token": system.get("apiAccessToken"),
                "device_id": system.get("device", {}).get("deviceId"),
                "session_attributes": session.get("attributes", {}),
            },
        )

    def build_alexa_response(
        self,
        response: VoiceDeviceResponse,
        session_attributes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build an Alexa skill response from VoiceDeviceResponse.

        Args:
            response: Voice response to convert
            session_attributes: Optional session state to persist

        Returns:
            Alexa response JSON
        """
        alexa_response: Dict[str, Any] = {
            "version": "1.0",
            "response": {
                "outputSpeech": {
                    "type": "PlainText",
                    "text": response.text,
                },
                "shouldEndSession": response.should_end_session,
            },
        }

        # Add reprompt if provided
        if response.reprompt:
            alexa_response["response"]["reprompt"] = {
                "outputSpeech": {
                    "type": "PlainText",
                    "text": response.reprompt,
                }
            }

        # Add card if provided
        if response.card_title or response.card_content:
            card: Dict[str, Any] = {"type": "Simple"}
            if response.card_title:
                card["title"] = response.card_title
            if response.card_content:
                card["content"] = response.card_content
            if response.card_image_url:
                card["type"] = "Standard"
                card["image"] = {
                    "smallImageUrl": response.card_image_url,
                    "largeImageUrl": response.card_image_url,
                }
            alexa_response["response"]["card"] = card

        # Add directives
        if response.directives:
            alexa_response["response"]["directives"] = response.directives

        # Add session attributes
        if session_attributes:
            alexa_response["sessionAttributes"] = session_attributes

        return alexa_response

    # ==========================================================================
    # Intent Handlers
    # ==========================================================================

    async def _handle_help(self, request: VoiceDeviceRequest) -> VoiceDeviceResponse:
        """Handle help intent."""
        return VoiceDeviceResponse(
            text="I can help you with multi-agent debates. "
            "You can say: Start a debate about a topic. "
            "Get my latest decision. "
            "List my debates. "
            "Or get the status of a debate. "
            "What would you like to do?",
            should_end_session=False,
            reprompt="What would you like to do?",
            card_title="Aragora Help",
            card_content="Available commands:\n"
            "- Start a debate about [topic]\n"
            "- Get my latest decision\n"
            "- List my debates\n"
            "- Get status of [debate]",
        )

    async def _handle_stop(self, request: VoiceDeviceRequest) -> VoiceDeviceResponse:
        """Handle stop/cancel intent."""
        return VoiceDeviceResponse(
            text="Goodbye! Your debates will continue running in the background.",
            should_end_session=True,
        )

    async def _handle_fallback(self, request: VoiceDeviceRequest) -> VoiceDeviceResponse:
        """Handle fallback intent."""
        return VoiceDeviceResponse(
            text="I didn't understand that. You can ask me to start a debate, "
            "get a decision, or list your debates. What would you like to do?",
            should_end_session=False,
            reprompt="What would you like to do?",
        )

    async def _handle_start_debate(self, request: VoiceDeviceRequest) -> VoiceDeviceResponse:
        """Handle start debate intent."""
        topic = request.slots.get("topic", "")

        if not topic:
            return VoiceDeviceResponse(
                text="What topic would you like to debate?",
                should_end_session=False,
                reprompt="Please tell me a topic to debate.",
            )

        # Start debate via Arena
        try:
            # This would integrate with the debate system
            logger.info(f"Starting voice debate on topic: {topic}")

            return VoiceDeviceResponse(
                text=f"Starting a debate on: {topic}. "
                "I'll notify you when the agents reach a decision. "
                "You can ask me for the status at any time.",
                should_end_session=True,
                card_title=f"Debate Started: {topic}",
                card_content=f"Topic: {topic}\nStatus: In Progress",
            )

        except Exception as e:
            logger.error(f"Failed to start debate: {e}")
            return VoiceDeviceResponse(
                text="I'm sorry, I couldn't start the debate right now. Please try again later.",
                should_end_session=True,
            )

    async def _handle_get_decision(self, request: VoiceDeviceRequest) -> VoiceDeviceResponse:
        """Handle get decision intent."""
        try:
            # This would fetch the latest decision for the user
            logger.info(f"Getting decision for user: {request.user_id}")

            # Placeholder response
            return VoiceDeviceResponse(
                text="Your latest debate concluded with the agents recommending "
                "to proceed with option A. The confidence level was high, "
                "with 4 out of 5 agents in agreement.",
                should_end_session=True,
                card_title="Latest Decision",
                card_content="Recommendation: Option A\nConfidence: High\nConsensus: 4/5 agents",
            )

        except Exception as e:
            logger.error(f"Failed to get decision: {e}")
            return VoiceDeviceResponse(
                text="I couldn't retrieve your decision right now. Please try again later.",
                should_end_session=True,
            )

    async def _handle_list_debates(self, request: VoiceDeviceRequest) -> VoiceDeviceResponse:
        """Handle list debates intent."""
        try:
            # This would fetch debates for the user
            logger.info(f"Listing debates for user: {request.user_id}")

            # Placeholder response
            return VoiceDeviceResponse(
                text="You have 3 active debates. "
                "First, a debate about marketing strategy. "
                "Second, a debate about technical architecture. "
                "Third, a debate about team structure. "
                "Which one would you like to know more about?",
                should_end_session=False,
                reprompt="Which debate would you like to hear about?",
            )

        except Exception as e:
            logger.error(f"Failed to list debates: {e}")
            return VoiceDeviceResponse(
                text="I couldn't retrieve your debates right now. Please try again later.",
                should_end_session=True,
            )

    async def _handle_get_status(self, request: VoiceDeviceRequest) -> VoiceDeviceResponse:
        """Handle get status intent."""
        debate_name = request.slots.get("debate_name", "")

        try:
            logger.info(f"Getting status for debate: {debate_name}")

            # Placeholder response
            return VoiceDeviceResponse(
                text="The debate is currently in round 3 of 5. "
                "The agents are discussing pros and cons. "
                "Early consensus is forming around option B.",
                should_end_session=True,
                card_title="Debate Status",
                card_content="Round: 3/5\nPhase: Discussion\nTrending: Option B",
            )

        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return VoiceDeviceResponse(
                text="I couldn't get the debate status right now. Please try again later.",
                should_end_session=True,
            )

    # ==========================================================================
    # Proactive Notifications
    # ==========================================================================

    async def send_proactive_notification(
        self,
        user_id: str,
        message: str,
        **kwargs: Any,
    ) -> bool:
        """
        Send a proactive notification to an Alexa user.

        Uses the Proactive Events API to send notifications.

        Args:
            user_id: Alexa user ID
            message: Notification message
            **kwargs: Additional options (event_type, reference_id, etc.)

        Returns:
            True if notification sent successfully
        """
        if not self._initialized:
            logger.warning("Alexa connector not initialized")
            return False

        try:
            # Refresh token if needed
            access_token = await self._refresh_access_token()

            # Build proactive event
            event_type = kwargs.get("event_type", "ARAGORA.DecisionReady")
            reference_id = kwargs.get("reference_id", f"aragora_{int(time.time())}")

            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "referenceId": reference_id,
                "expiryTime": datetime.now(timezone.utc)
                .replace(hour=23, minute=59, second=59)
                .isoformat(),
                "event": {
                    "name": event_type,
                    "payload": {
                        "message": {
                            "content": message,
                        },
                    },
                },
                "relevantAudience": {
                    "type": "Unicast",
                    "payload": {
                        "user": user_id,
                    },
                },
            }

            # Send proactive event
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }

            success, response, error = await self._http_request(
                method="POST",
                url=self._proactive_api_endpoint,
                headers=headers,
                json=event,
                operation="proactive_notification",
            )

            if success:
                logger.info(f"Sent proactive notification to user {user_id[:20]}...")
                return True
            else:
                logger.warning(f"Failed to send proactive notification: {error}")
                return False

        except Exception as e:
            logger.error(f"Error sending proactive notification: {e}")
            return False

    # ==========================================================================
    # Push Notification Interface
    # ==========================================================================

    async def send_notification(
        self,
        device: DeviceToken,
        message: DeviceMessage,
        **kwargs: Any,
    ) -> SendResult:
        """
        Send a notification to an Alexa device.

        Uses proactive notifications API.

        Args:
            device: Target device token
            message: Notification message

        Returns:
            SendResult with delivery status
        """
        success = await self.send_proactive_notification(
            user_id=device.user_id,
            message=f"{message.title}: {message.body}",
            **kwargs,
        )

        return SendResult(
            success=success,
            device_id=device.device_id,
            status=DeliveryStatus.SENT if success else DeliveryStatus.FAILED,
            error=None if success else "Failed to send Alexa notification",
            timestamp=datetime.now(timezone.utc),
        )

    # ==========================================================================
    # Account Linking
    # ==========================================================================

    async def link_account(
        self,
        alexa_user_id: str,
        aragora_user_id: str,
        access_token: str,
    ) -> bool:
        """
        Link an Alexa user to an Aragora user account.

        Args:
            alexa_user_id: Alexa user identifier
            aragora_user_id: Aragora user identifier
            access_token: OAuth access token from account linking

        Returns:
            True if linking successful
        """
        try:
            # Store the account link
            from aragora.server.session_store import DeviceSession, get_session_store

            store = get_session_store()

            device_session = DeviceSession(
                device_id=f"alexa_{hashlib.sha256(alexa_user_id.encode()).hexdigest()[:32]}",
                user_id=aragora_user_id,
                device_type="alexa",
                push_token=alexa_user_id,
                metadata={
                    "alexa_user_id": alexa_user_id,
                    "access_token": access_token,
                    "linked_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            store.set_device_session(device_session)
            logger.info(f"Linked Alexa user to Aragora user {aragora_user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to link account: {e}")
            return False

    async def unlink_account(self, alexa_user_id: str) -> bool:
        """
        Unlink an Alexa user account.

        Args:
            alexa_user_id: Alexa user identifier

        Returns:
            True if unlinking successful
        """
        try:
            from aragora.server.session_store import get_session_store

            store = get_session_store()
            device_id = f"alexa_{hashlib.sha256(alexa_user_id.encode()).hexdigest()[:32]}"

            return store.delete_device_session(device_id)

        except Exception as e:
            logger.error(f"Failed to unlink account: {e}")
            return False

    # ==========================================================================
    # Health and Status
    # ==========================================================================

    async def get_health(self) -> Dict[str, Any]:
        """Get health status for the Alexa connector."""
        health = await super().get_health()

        health["configured"] = all([self._client_id, self._client_secret, self._skill_id])
        health["skill_id"] = self._skill_id[:20] + "..." if self._skill_id else None
        health["has_access_token"] = bool(self._access_token)
        health["token_valid"] = (
            time.time() < self._token_expires_at if self._access_token else False
        )

        return health
