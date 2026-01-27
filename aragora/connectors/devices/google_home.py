"""
Google Home Device Connector.

Provides integration with Google Assistant/Home for:
- Conversational Actions handling
- Home Graph for device sync
- Broadcast announcements for debate results
- Account linking for user identification

Environment Variables:
    GOOGLE_HOME_PROJECT_ID: Google Cloud project ID
    GOOGLE_HOME_CREDENTIALS: Path to service account JSON or JSON string
    GOOGLE_APPLICATION_CREDENTIALS: Alternative path to service account JSON
"""

from __future__ import annotations

import base64
import hashlib
import json
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


class GoogleActionType(Enum):
    """Google Actions request types."""

    # Conversation actions
    MAIN = "actions.intent.MAIN"
    TEXT = "actions.intent.TEXT"
    CANCEL = "actions.intent.CANCEL"

    # Smart Home
    SYNC = "action.devices.SYNC"
    QUERY = "action.devices.QUERY"
    EXECUTE = "action.devices.EXECUTE"
    DISCONNECT = "action.devices.DISCONNECT"


class GoogleIntent(Enum):
    """Custom intents for Aragora."""

    START_DEBATE = "start_debate"
    GET_DECISION = "get_decision"
    LIST_DEBATES = "list_debates"
    GET_STATUS = "get_status"
    VOTE = "vote"
    HELP = "help"


@dataclass
class GoogleUser:
    """Google user information."""

    user_id: str
    access_token: Optional[str] = None
    profile: Dict[str, Any] = field(default_factory=dict)
    locale: str = "en"


@dataclass
class GoogleConversation:
    """Conversation state for Google Actions."""

    conversation_id: str
    conversation_token: Optional[str] = None
    user: Optional[GoogleUser] = None
    is_new: bool = False
    surface_capabilities: List[str] = field(default_factory=list)


class GoogleHomeConnector(DeviceConnector):
    """
    Google Home/Assistant device connector.

    Handles Google Actions requests and broadcast announcements.

    Features:
    - Conversational Actions for debate operations
    - Home Graph integration for device sync
    - Broadcast announcements for debate results
    - Account linking for user identification
    """

    def __init__(self, config: Optional[DeviceConnectorConfig] = None):
        """Initialize the Google Home connector."""
        super().__init__(config)

        # Load configuration
        self._project_id = os.environ.get("GOOGLE_HOME_PROJECT_ID", "")
        self._credentials_path = os.environ.get(
            "GOOGLE_HOME_CREDENTIALS",
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
        )

        # Service account credentials
        self._credentials: Optional[Dict[str, Any]] = None
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0

        # Home Graph API endpoint
        self._home_graph_url = "https://homegraph.googleapis.com/v1"

        # Intent handlers
        self._intent_handlers: Dict[str, Any] = {}
        self._register_default_handlers()

    # ==========================================================================
    # Properties
    # ==========================================================================

    @property
    def platform_name(self) -> str:
        """Return the platform identifier."""
        return "google_home"

    @property
    def platform_display_name(self) -> str:
        """Return human-readable platform name."""
        return "Google Home"

    @property
    def supported_device_types(self) -> List[DeviceType]:
        """Return list of device types this connector supports."""
        return [DeviceType.GOOGLE_HOME]

    # ==========================================================================
    # Initialization
    # ==========================================================================

    async def initialize(self) -> bool:
        """
        Initialize the Google Home connector.

        Loads service account credentials and validates configuration.

        Returns:
            True if initialization successful
        """
        if not self._project_id:
            logger.info("Google Home connector not configured: missing GOOGLE_HOME_PROJECT_ID")
            return False

        # Load credentials
        try:
            if self._credentials_path:
                # Check if it's a JSON string or file path
                if self._credentials_path.startswith("{"):
                    self._credentials = json.loads(self._credentials_path)
                elif os.path.exists(self._credentials_path):
                    with open(self._credentials_path) as f:
                        self._credentials = json.load(f)
                else:
                    logger.warning(f"Credentials file not found: {self._credentials_path}")
                    return False

            self._initialized = True
            logger.info("Google Home connector initialized successfully")
            return True

        except Exception as e:
            logger.warning(f"Google Home connector initialization failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown the Google Home connector."""
        self._access_token = None
        self._token_expires_at = 0
        await super().shutdown()

    # ==========================================================================
    # OAuth Token Management
    # ==========================================================================

    async def _get_access_token(self) -> str:
        """
        Get OAuth access token for Google APIs.

        Uses service account credentials to generate JWT and exchange
        for an access token.

        Returns:
            Access token string
        """
        # Check if current token is valid
        if self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token

        if not self._credentials:
            raise Exception("No credentials configured")

        try:
            # Build JWT for service account auth
            now = int(time.time())
            payload = {
                "iss": self._credentials.get("client_email"),
                "sub": self._credentials.get("client_email"),
                "aud": "https://oauth2.googleapis.com/token",
                "iat": now,
                "exp": now + 3600,
                "scope": "https://www.googleapis.com/auth/homegraph",
            }

            # Sign JWT
            header = {"alg": "RS256", "typ": "JWT"}
            header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
            payload_b64 = (
                base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
            )

            # For actual implementation, use cryptography library to sign
            # For now, use basic auth approach
            private_key = self._credentials.get("private_key", "")

            try:
                from cryptography.hazmat.primitives import hashes, serialization
                from cryptography.hazmat.primitives.asymmetric import padding

                key = serialization.load_pem_private_key(
                    private_key.encode(),
                    password=None,
                )
                message = f"{header_b64}.{payload_b64}".encode()
                signature = key.sign(  # type: ignore
                    message,
                    padding.PKCS1v15(),
                    hashes.SHA256(),
                )
                signature_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
                jwt = f"{header_b64}.{payload_b64}.{signature_b64}"

            except ImportError:
                logger.warning("cryptography library not available, using mock token")
                jwt = f"{header_b64}.{payload_b64}.mock_signature"

            # Exchange JWT for access token
            token_url = "https://oauth2.googleapis.com/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            data = {
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": jwt,
            }

            success, response, error = await self._http_request(
                method="POST",
                url=token_url,
                headers=headers,
                data=data,
                operation="token_exchange",
            )

            if success and response:
                self._access_token = response.get("access_token")
                expires_in = response.get("expires_in", 3600)
                self._token_expires_at = time.time() + expires_in
                return self._access_token  # type: ignore
            else:
                raise Exception(f"Token exchange failed: {error}")

        except Exception as e:
            logger.error(f"Failed to get access token: {e}")
            raise

    # ==========================================================================
    # Request Handling
    # ==========================================================================

    def _register_default_handlers(self) -> None:
        """Register default intent handlers."""
        self._intent_handlers = {
            GoogleIntent.HELP.value: self._handle_help,
            GoogleIntent.START_DEBATE.value: self._handle_start_debate,
            GoogleIntent.GET_DECISION.value: self._handle_get_decision,
            GoogleIntent.LIST_DEBATES.value: self._handle_list_debates,
            GoogleIntent.GET_STATUS.value: self._handle_get_status,
        }

    async def handle_voice_request(
        self,
        request: VoiceDeviceRequest,
        **kwargs: Any,
    ) -> VoiceDeviceResponse:
        """
        Handle a voice request from Google Assistant.

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

    def parse_google_request(self, request_data: Dict[str, Any]) -> VoiceDeviceRequest:
        """
        Parse a Google Actions request into VoiceDeviceRequest.

        Args:
            request_data: Raw Google Actions request JSON

        Returns:
            Parsed VoiceDeviceRequest
        """
        # Handle different request formats (Actions Builder vs Legacy)
        handler = request_data.get("handler", {})
        intent_name = handler.get("name", "")

        # Extract user info
        user_data = request_data.get("user", {})
        user_id = user_data.get("params", {}).get("userId", "")

        # Extract parameters
        session = request_data.get("session", {})
        params = session.get("params", {})

        # Extract slots from intent params
        intent = request_data.get("intent", {})
        intent_params = intent.get("params", {})
        slots: Dict[str, Any] = {}
        for param_name, param_data in intent_params.items():
            if "resolved" in param_data:
                slots[param_name] = param_data["resolved"]
            elif "original" in param_data:
                slots[param_name] = param_data["original"]

        # Get raw input
        scene = request_data.get("scene", {})
        raw_input = intent.get("query", "")

        return VoiceDeviceRequest(
            request_id=session.get("id", ""),
            device_type=DeviceType.GOOGLE_HOME,
            user_id=user_id,
            intent=intent_name,
            slots=slots,
            raw_input=raw_input,
            session_id=session.get("id"),
            is_new_session=request_data.get("conversation", {}).get("type") == "NEW",
            timestamp=datetime.now(timezone.utc),
            locale=user_data.get("locale", "en"),
            metadata={
                "scene_name": scene.get("name"),
                "session_params": params,
                "device": request_data.get("device", {}),
            },
        )

    def build_google_response(
        self,
        response: VoiceDeviceResponse,
        session_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a Google Actions response from VoiceDeviceResponse.

        Args:
            response: Voice response to convert
            session_params: Optional session state to persist

        Returns:
            Google Actions response JSON
        """
        google_response: Dict[str, Any] = {
            "prompt": {
                "firstSimple": {
                    "speech": response.text,
                    "text": response.text,
                },
            },
        }

        # Handle session ending
        if response.should_end_session:
            google_response["scene"] = {"name": "EndConversation"}
        elif response.reprompt:
            google_response["prompt"]["suggestions"] = [
                {"title": "Start a debate"},
                {"title": "Get decision"},
                {"title": "Help"},
            ]

        # Add card if provided
        if response.card_title or response.card_content:
            card: Dict[str, Any] = {
                "title": response.card_title or "Aragora",
                "text": response.card_content or response.text,
            }
            if response.card_image_url:
                card["image"] = {
                    "url": response.card_image_url,
                    "alt": response.card_title or "Aragora",
                }
            google_response["prompt"]["content"] = {"card": card}

        # Add session params
        if session_params:
            google_response["session"] = {"params": session_params}

        return google_response

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
            "Or get the status of a debate.",
            should_end_session=False,
            reprompt="What would you like to do?",
            card_title="Aragora Help",
            card_content="Available commands:\n"
            "- Start a debate about [topic]\n"
            "- Get my latest decision\n"
            "- List my debates\n"
            "- Get status",
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

        logger.info(f"Starting voice debate on topic: {topic}")

        return VoiceDeviceResponse(
            text=f"Starting a debate on: {topic}. "
            "I'll send you a notification when the decision is ready.",
            should_end_session=True,
            card_title="Debate Started",
            card_content=f"Topic: {topic}\nStatus: In Progress",
        )

    async def _handle_get_decision(self, request: VoiceDeviceRequest) -> VoiceDeviceResponse:
        """Handle get decision intent."""
        logger.info(f"Getting decision for user: {request.user_id}")

        return VoiceDeviceResponse(
            text="Your latest debate concluded with the agents recommending "
            "to proceed with option A. The confidence level was high.",
            should_end_session=True,
            card_title="Latest Decision",
            card_content="Recommendation: Option A\nConfidence: High",
        )

    async def _handle_list_debates(self, request: VoiceDeviceRequest) -> VoiceDeviceResponse:
        """Handle list debates intent."""
        logger.info(f"Listing debates for user: {request.user_id}")

        return VoiceDeviceResponse(
            text="You have 3 active debates. "
            "A marketing strategy debate, a technical architecture debate, "
            "and a team structure debate.",
            should_end_session=False,
            reprompt="Which debate would you like to hear about?",
        )

    async def _handle_get_status(self, request: VoiceDeviceRequest) -> VoiceDeviceResponse:
        """Handle get status intent."""
        debate_name = request.slots.get("debate_name", "")
        logger.info(f"Getting status for debate: {debate_name}")

        return VoiceDeviceResponse(
            text="The debate is in round 3 of 5. Early consensus is forming around option B.",
            should_end_session=True,
            card_title="Debate Status",
            card_content="Round: 3/5\nTrending: Option B",
        )

    # ==========================================================================
    # Home Graph Operations
    # ==========================================================================

    async def request_sync(self, agent_user_id: str) -> bool:
        """
        Request a SYNC for a user's devices.

        Triggers Google to refresh the device list for a user.

        Args:
            agent_user_id: The user's agent user ID

        Returns:
            True if sync requested successfully
        """
        if not self._initialized:
            logger.warning("Google Home connector not initialized")
            return False

        try:
            access_token = await self._get_access_token()

            url = f"{self._home_graph_url}/devices:requestSync"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }
            body = {"agentUserId": agent_user_id}

            success, response, error = await self._http_request(
                method="POST",
                url=url,
                headers=headers,
                json=body,
                operation="request_sync",
            )

            if success:
                logger.info(f"Requested sync for user {agent_user_id}")
                return True
            else:
                logger.warning(f"Failed to request sync: {error}")
                return False

        except Exception as e:
            logger.error(f"Error requesting sync: {e}")
            return False

    async def report_state(
        self,
        agent_user_id: str,
        devices: List[Dict[str, Any]],
    ) -> bool:
        """
        Report device state to Home Graph.

        Args:
            agent_user_id: The user's agent user ID
            devices: List of device states to report

        Returns:
            True if state reported successfully
        """
        if not self._initialized:
            logger.warning("Google Home connector not initialized")
            return False

        try:
            access_token = await self._get_access_token()

            url = f"{self._home_graph_url}/devices:reportStateAndNotification"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }
            body = {
                "agentUserId": agent_user_id,
                "requestId": f"aragora_{int(time.time())}",
                "payload": {
                    "devices": {
                        "states": {d["id"]: d["state"] for d in devices},
                    },
                },
            }

            success, response, error = await self._http_request(
                method="POST",
                url=url,
                headers=headers,
                json=body,
                operation="report_state",
            )

            if success:
                logger.info(f"Reported state for user {agent_user_id}")
                return True
            else:
                logger.warning(f"Failed to report state: {error}")
                return False

        except Exception as e:
            logger.error(f"Error reporting state: {e}")
            return False

    # ==========================================================================
    # Broadcast Announcements
    # ==========================================================================

    async def send_proactive_notification(
        self,
        user_id: str,
        message: str,
        **kwargs: Any,
    ) -> bool:
        """
        Send a broadcast announcement to Google Home devices.

        Uses the Home Graph API to send notifications.

        Args:
            user_id: Google user ID
            message: Message to broadcast

        Returns:
            True if broadcast sent successfully
        """
        if not self._initialized:
            logger.warning("Google Home connector not initialized")
            return False

        try:
            access_token = await self._get_access_token()

            # Use notifications API
            url = f"{self._home_graph_url}/devices:reportStateAndNotification"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }

            # Build notification payload
            body = {
                "agentUserId": user_id,
                "requestId": f"aragora_notification_{int(time.time())}",
                "eventId": f"event_{int(time.time())}",
                "payload": {
                    "devices": {
                        "notifications": {
                            "aragora-debate-notifier": {
                                "ObjectNotification": {
                                    "objects": {
                                        "decision": {
                                            "priority": 0,
                                            "name": "Debate Decision Ready",
                                            "description": message,
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            }

            success, response, error = await self._http_request(
                method="POST",
                url=url,
                headers=headers,
                json=body,
                operation="broadcast_notification",
            )

            if success:
                logger.info(f"Sent broadcast to user {user_id[:20]}...")
                return True
            else:
                logger.warning(f"Failed to send broadcast: {error}")
                return False

        except Exception as e:
            logger.error(f"Error sending broadcast: {e}")
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
        Send a notification to a Google Home device.

        Uses broadcast announcements.

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
            error=None if success else "Failed to send Google Home notification",
            timestamp=datetime.now(timezone.utc),
        )

    # ==========================================================================
    # Account Linking
    # ==========================================================================

    async def link_account(
        self,
        google_user_id: str,
        aragora_user_id: str,
        access_token: str,
    ) -> bool:
        """
        Link a Google user to an Aragora user account.

        Args:
            google_user_id: Google user identifier
            aragora_user_id: Aragora user identifier
            access_token: OAuth access token from account linking

        Returns:
            True if linking successful
        """
        try:
            from aragora.server.session_store import DeviceSession, get_session_store

            store = get_session_store()

            device_session = DeviceSession(
                device_id=f"google_{hashlib.sha256(google_user_id.encode()).hexdigest()[:32]}",
                user_id=aragora_user_id,
                device_type="google_home",
                push_token=google_user_id,
                metadata={
                    "google_user_id": google_user_id,
                    "access_token": access_token,
                    "linked_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            store.set_device_session(device_session)
            logger.info(f"Linked Google user to Aragora user {aragora_user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to link account: {e}")
            return False

    async def unlink_account(self, google_user_id: str) -> bool:
        """
        Unlink a Google user account.

        Args:
            google_user_id: Google user identifier

        Returns:
            True if unlinking successful
        """
        try:
            from aragora.server.session_store import get_session_store

            store = get_session_store()
            device_id = f"google_{hashlib.sha256(google_user_id.encode()).hexdigest()[:32]}"

            return store.delete_device_session(device_id)

        except Exception as e:
            logger.error(f"Failed to unlink account: {e}")
            return False

    # ==========================================================================
    # Smart Home Handlers
    # ==========================================================================

    async def handle_sync(
        self,
        request_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Handle SYNC intent from Google.

        Returns list of devices for the user.

        Args:
            request_id: Request identifier
            user_id: User identifier

        Returns:
            SYNC response with device list
        """
        # Return Aragora as a virtual "notifier" device
        return {
            "requestId": request_id,
            "payload": {
                "agentUserId": user_id,
                "devices": [
                    {
                        "id": "aragora-debate-notifier",
                        "type": "action.devices.types.SENSOR",
                        "traits": ["action.devices.traits.ObjectDetection"],
                        "name": {
                            "defaultNames": ["Aragora Debate Notifier"],
                            "name": "Debate Notifier",
                            "nicknames": ["Aragora", "Debate Agent"],
                        },
                        "willReportState": True,
                        "notificationSupportedByAgent": True,
                        "attributes": {
                            "objectDetectionConfiguration": [
                                {
                                    "object": "decision",
                                    "priority": 0,
                                    "notifyOnce": False,
                                },
                            ],
                        },
                        "deviceInfo": {
                            "manufacturer": "Aragora",
                            "model": "Debate Engine v1",
                            "hwVersion": "1.0",
                            "swVersion": "1.0",
                        },
                    },
                ],
            },
        }

    async def handle_query(
        self,
        request_id: str,
        devices: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Handle QUERY intent from Google.

        Returns current state of requested devices.

        Args:
            request_id: Request identifier
            devices: List of devices to query

        Returns:
            QUERY response with device states
        """
        device_states = {}
        for device in devices:
            device_id = device.get("id")
            if device_id == "aragora-debate-notifier":
                device_states[device_id] = {
                    "online": True,
                    "status": "SUCCESS",
                }

        return {
            "requestId": request_id,
            "payload": {
                "devices": device_states,
            },
        }

    async def handle_execute(
        self,
        request_id: str,
        commands: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Handle EXECUTE intent from Google.

        Executes commands on devices.

        Args:
            request_id: Request identifier
            commands: List of commands to execute

        Returns:
            EXECUTE response with results
        """
        # Aragora notifier doesn't support direct commands
        command_results = []
        for command in commands:
            for device in command.get("devices", []):
                command_results.append(
                    {
                        "ids": [device.get("id")],
                        "status": "SUCCESS",
                        "states": {"online": True},
                    }
                )

        return {
            "requestId": request_id,
            "payload": {
                "commands": command_results,
            },
        }

    async def handle_disconnect(
        self,
        request_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Handle DISCONNECT intent from Google.

        User has unlinked their account.

        Args:
            request_id: Request identifier
            user_id: User identifier

        Returns:
            Empty response acknowledging disconnect
        """
        await self.unlink_account(user_id)
        return {}

    # ==========================================================================
    # Health and Status
    # ==========================================================================

    async def get_health(self) -> Dict[str, Any]:
        """Get health status for the Google Home connector."""
        health = await super().get_health()

        health["configured"] = bool(self._project_id and self._credentials)
        health["project_id"] = self._project_id if self._project_id else None
        health["has_credentials"] = bool(self._credentials)
        health["has_access_token"] = bool(self._access_token)
        health["token_valid"] = (
            time.time() < self._token_expires_at if self._access_token else False
        )

        return health
