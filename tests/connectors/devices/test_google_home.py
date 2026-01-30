"""
Tests for GoogleHomeConnector - Google Home/Assistant device integration.

Tests cover:
- Connector initialization and configuration
- Voice request handling and parsing
- Response building
- Intent handlers (help, start_debate, get_decision, list_debates, get_status)
- Home Graph operations (sync, query, execute, disconnect)
- Broadcast announcements
- Account linking
- Notification sending
- Health status
- Error handling
"""

import json
import os
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


class TestGoogleHomeConnectorInit:
    """Tests for GoogleHomeConnector initialization."""

    def test_default_init(self):
        """Should initialize with default values."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()

        assert connector.platform_name == "google_home"
        assert connector.platform_display_name == "Google Home"

    def test_init_with_config(self):
        """Should accept custom configuration."""
        from aragora.connectors.devices.base import DeviceConnectorConfig
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        config = DeviceConnectorConfig(
            enable_circuit_breaker=False,
            request_timeout=60.0,
        )
        connector = GoogleHomeConnector(config=config)

        assert connector.config.enable_circuit_breaker is False
        assert connector.config.request_timeout == 60.0

    def test_supported_device_types(self):
        """Should support Google Home device type."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector
        from aragora.connectors.devices.models import DeviceType

        connector = GoogleHomeConnector()

        assert DeviceType.GOOGLE_HOME in connector.supported_device_types

    def test_intent_handlers_registered(self):
        """Should register default intent handlers."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()

        assert "help" in connector._intent_handlers
        assert "start_debate" in connector._intent_handlers
        assert "get_decision" in connector._intent_handlers
        assert "list_debates" in connector._intent_handlers
        assert "get_status" in connector._intent_handlers


class TestGoogleHomeInitialization:
    """Tests for connector initialization lifecycle."""

    @pytest.mark.asyncio
    async def test_initialize_without_project_id(self):
        """Should fail initialization without project ID."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        with patch.dict(os.environ, {}, clear=True):
            connector = GoogleHomeConnector()
            result = await connector.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_with_valid_json_credentials(self):
        """Should initialize with valid JSON credentials string."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        credentials = json.dumps(
            {
                "client_email": "test@project.iam.gserviceaccount.com",
                "private_key": "-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----",
            }
        )

        with patch.dict(
            os.environ,
            {
                "GOOGLE_HOME_PROJECT_ID": "test-project",
                "GOOGLE_HOME_CREDENTIALS": credentials,
            },
        ):
            connector = GoogleHomeConnector()
            result = await connector.initialize()

        assert result is True
        assert connector._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_with_invalid_json_credentials(self):
        """Should fail with invalid JSON credentials."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        with patch.dict(
            os.environ,
            {
                "GOOGLE_HOME_PROJECT_ID": "test-project",
                "GOOGLE_HOME_CREDENTIALS": "not-valid-json",
            },
        ):
            connector = GoogleHomeConnector()
            result = await connector.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_with_nonexistent_file(self):
        """Should fail when credentials file does not exist."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        with patch.dict(
            os.environ,
            {
                "GOOGLE_HOME_PROJECT_ID": "test-project",
                "GOOGLE_HOME_CREDENTIALS": "/nonexistent/path/credentials.json",
            },
        ):
            connector = GoogleHomeConnector()
            result = await connector.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Should clear state on shutdown."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()
        connector._initialized = True
        connector._access_token = "test-token"
        connector._token_expires_at = 12345

        await connector.shutdown()

        assert connector._initialized is False
        assert connector._access_token is None
        assert connector._token_expires_at == 0


class TestGoogleHomeRequestParsing:
    """Tests for parsing Google Actions requests."""

    def test_parse_google_request_basic(self):
        """Should parse basic Google Actions request."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector
        from aragora.connectors.devices.models import DeviceType

        connector = GoogleHomeConnector()

        request_data = {
            "handler": {"name": "start_debate"},
            "user": {
                "params": {"userId": "user123"},
                "locale": "en-US",
            },
            "session": {
                "id": "session123",
                "params": {"context": "test"},
            },
            "intent": {
                "params": {
                    "topic": {"resolved": "climate change"},
                },
                "query": "start a debate about climate change",
            },
            "scene": {"name": "MainScene"},
            "conversation": {"type": "NEW"},
            "device": {"capabilities": ["SPEECH"]},
        }

        result = connector.parse_google_request(request_data)

        assert result.intent == "start_debate"
        assert result.user_id == "user123"
        assert result.device_type == DeviceType.GOOGLE_HOME
        assert result.slots["topic"] == "climate change"
        assert result.raw_input == "start a debate about climate change"
        assert result.session_id == "session123"
        assert result.is_new_session is True
        assert result.locale == "en-US"

    def test_parse_google_request_with_original_param(self):
        """Should fall back to original param value."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()

        request_data = {
            "handler": {"name": "test"},
            "user": {"params": {}},
            "session": {"id": "s1"},
            "intent": {
                "params": {
                    "name": {"original": "test value"},
                },
            },
        }

        result = connector.parse_google_request(request_data)

        assert result.slots["name"] == "test value"

    def test_parse_google_request_minimal(self):
        """Should handle minimal request data."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()

        request_data = {
            "handler": {},
            "user": {},
            "session": {},
            "intent": {},
        }

        result = connector.parse_google_request(request_data)

        assert result.intent == ""
        assert result.user_id == ""


class TestGoogleHomeResponseBuilding:
    """Tests for building Google Actions responses."""

    def test_build_google_response_basic(self):
        """Should build basic response."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector
        from aragora.connectors.devices.models import VoiceDeviceResponse

        connector = GoogleHomeConnector()
        response = VoiceDeviceResponse(text="Hello, world!")

        result = connector.build_google_response(response)

        assert result["prompt"]["firstSimple"]["speech"] == "Hello, world!"
        assert result["prompt"]["firstSimple"]["text"] == "Hello, world!"

    def test_build_google_response_with_end_session(self):
        """Should include EndConversation scene when ending session."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector
        from aragora.connectors.devices.models import VoiceDeviceResponse

        connector = GoogleHomeConnector()
        response = VoiceDeviceResponse(text="Goodbye!", should_end_session=True)

        result = connector.build_google_response(response)

        assert result["scene"]["name"] == "EndConversation"

    def test_build_google_response_with_reprompt(self):
        """Should include suggestions when reprompt provided."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector
        from aragora.connectors.devices.models import VoiceDeviceResponse

        connector = GoogleHomeConnector()
        response = VoiceDeviceResponse(
            text="What next?",
            should_end_session=False,
            reprompt="Choose an option.",
        )

        result = connector.build_google_response(response)

        assert "suggestions" in result["prompt"]
        assert len(result["prompt"]["suggestions"]) > 0

    def test_build_google_response_with_card(self):
        """Should include card content."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector
        from aragora.connectors.devices.models import VoiceDeviceResponse

        connector = GoogleHomeConnector()
        response = VoiceDeviceResponse(
            text="Result",
            card_title="Decision",
            card_content="Choose option A",
            card_image_url="https://example.com/image.png",
        )

        result = connector.build_google_response(response)

        card = result["prompt"]["content"]["card"]
        assert card["title"] == "Decision"
        assert card["text"] == "Choose option A"
        assert card["image"]["url"] == "https://example.com/image.png"

    def test_build_google_response_with_session_params(self):
        """Should include session parameters."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector
        from aragora.connectors.devices.models import VoiceDeviceResponse

        connector = GoogleHomeConnector()
        response = VoiceDeviceResponse(text="Continue")
        session_params = {"debate_id": "123", "step": 2}

        result = connector.build_google_response(response, session_params)

        assert result["session"]["params"] == session_params


class TestGoogleHomeIntentHandlers:
    """Tests for intent handlers."""

    @pytest.mark.asyncio
    async def test_handle_help_intent(self):
        """Should handle help intent."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector
        from aragora.connectors.devices.models import VoiceDeviceRequest, DeviceType

        connector = GoogleHomeConnector()
        request = VoiceDeviceRequest(
            request_id="r1",
            device_type=DeviceType.GOOGLE_HOME,
            user_id="u1",
            intent="help",
        )

        result = await connector.handle_voice_request(request)

        assert "debate" in result.text.lower()
        assert result.should_end_session is False
        assert result.card_title == "Aragora Help"

    @pytest.mark.asyncio
    async def test_handle_start_debate_with_topic(self):
        """Should start debate with provided topic."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector
        from aragora.connectors.devices.models import VoiceDeviceRequest, DeviceType

        connector = GoogleHomeConnector()
        request = VoiceDeviceRequest(
            request_id="r1",
            device_type=DeviceType.GOOGLE_HOME,
            user_id="u1",
            intent="start_debate",
            slots={"topic": "AI ethics"},
        )

        result = await connector.handle_voice_request(request)

        assert "AI ethics" in result.text
        assert result.should_end_session is True
        assert result.card_title == "Debate Started"

    @pytest.mark.asyncio
    async def test_handle_start_debate_without_topic(self):
        """Should prompt for topic when not provided."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector
        from aragora.connectors.devices.models import VoiceDeviceRequest, DeviceType

        connector = GoogleHomeConnector()
        request = VoiceDeviceRequest(
            request_id="r1",
            device_type=DeviceType.GOOGLE_HOME,
            user_id="u1",
            intent="start_debate",
            slots={},
        )

        result = await connector.handle_voice_request(request)

        assert "topic" in result.text.lower()
        assert result.should_end_session is False

    @pytest.mark.asyncio
    async def test_handle_get_decision(self):
        """Should return latest decision."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector
        from aragora.connectors.devices.models import VoiceDeviceRequest, DeviceType

        connector = GoogleHomeConnector()
        request = VoiceDeviceRequest(
            request_id="r1",
            device_type=DeviceType.GOOGLE_HOME,
            user_id="u1",
            intent="get_decision",
        )

        result = await connector.handle_voice_request(request)

        assert result.should_end_session is True
        assert result.card_title == "Latest Decision"

    @pytest.mark.asyncio
    async def test_handle_list_debates(self):
        """Should list user debates."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector
        from aragora.connectors.devices.models import VoiceDeviceRequest, DeviceType

        connector = GoogleHomeConnector()
        request = VoiceDeviceRequest(
            request_id="r1",
            device_type=DeviceType.GOOGLE_HOME,
            user_id="u1",
            intent="list_debates",
        )

        result = await connector.handle_voice_request(request)

        assert "debates" in result.text.lower()
        assert result.should_end_session is False

    @pytest.mark.asyncio
    async def test_handle_get_status(self):
        """Should return debate status."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector
        from aragora.connectors.devices.models import VoiceDeviceRequest, DeviceType

        connector = GoogleHomeConnector()
        request = VoiceDeviceRequest(
            request_id="r1",
            device_type=DeviceType.GOOGLE_HOME,
            user_id="u1",
            intent="get_status",
            slots={"debate_name": "marketing"},
        )

        result = await connector.handle_voice_request(request)

        assert result.should_end_session is True
        assert result.card_title == "Debate Status"

    @pytest.mark.asyncio
    async def test_handle_unknown_intent(self):
        """Should handle unknown intents gracefully."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector
        from aragora.connectors.devices.models import VoiceDeviceRequest, DeviceType

        connector = GoogleHomeConnector()
        request = VoiceDeviceRequest(
            request_id="r1",
            device_type=DeviceType.GOOGLE_HOME,
            user_id="u1",
            intent="unknown_intent",
        )

        result = await connector.handle_voice_request(request)

        assert result.should_end_session is False
        assert result.reprompt is not None


class TestGoogleHomeGraphOperations:
    """Tests for Home Graph API operations."""

    @pytest.mark.asyncio
    async def test_request_sync_not_initialized(self):
        """Should fail sync request when not initialized."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()
        connector._initialized = False

        result = await connector.request_sync("user123")

        assert result is False

    @pytest.mark.asyncio
    async def test_request_sync_success(self):
        """Should successfully request sync."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()
        connector._initialized = True
        connector._access_token = "test-token"
        connector._token_expires_at = 9999999999

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = (True, {}, None)

            result = await connector.request_sync("user123")

        assert result is True
        mock_http.assert_called_once()

    @pytest.mark.asyncio
    async def test_report_state_not_initialized(self):
        """Should fail state report when not initialized."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()
        connector._initialized = False

        result = await connector.report_state("user123", [])

        assert result is False

    @pytest.mark.asyncio
    async def test_report_state_success(self):
        """Should successfully report device state."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()
        connector._initialized = True
        connector._access_token = "test-token"
        connector._token_expires_at = 9999999999

        devices = [
            {"id": "device1", "state": {"online": True}},
            {"id": "device2", "state": {"online": False}},
        ]

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = (True, {}, None)

            result = await connector.report_state("user123", devices)

        assert result is True

    @pytest.mark.asyncio
    async def test_handle_sync(self):
        """Should return correct SYNC response."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()

        result = await connector.handle_sync("request123", "user123")

        assert result["requestId"] == "request123"
        assert result["payload"]["agentUserId"] == "user123"
        assert len(result["payload"]["devices"]) == 1
        assert result["payload"]["devices"][0]["id"] == "aragora-debate-notifier"

    @pytest.mark.asyncio
    async def test_handle_query(self):
        """Should return correct QUERY response."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()
        devices = [{"id": "aragora-debate-notifier"}]

        result = await connector.handle_query("request123", devices)

        assert result["requestId"] == "request123"
        assert result["payload"]["devices"]["aragora-debate-notifier"]["online"] is True

    @pytest.mark.asyncio
    async def test_handle_execute(self):
        """Should return correct EXECUTE response."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()
        commands = [
            {"devices": [{"id": "aragora-debate-notifier"}], "execution": []},
        ]

        result = await connector.handle_execute("request123", commands)

        assert result["requestId"] == "request123"
        assert result["payload"]["commands"][0]["status"] == "SUCCESS"

    @pytest.mark.asyncio
    async def test_handle_disconnect(self):
        """Should handle disconnect and unlink account."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()

        with patch.object(connector, "unlink_account", new_callable=AsyncMock) as mock_unlink:
            mock_unlink.return_value = True

            result = await connector.handle_disconnect("request123", "user123")

        assert result == {}
        mock_unlink.assert_called_once_with("user123")


class TestGoogleHomeNotifications:
    """Tests for notification sending."""

    @pytest.mark.asyncio
    async def test_send_proactive_notification_not_initialized(self):
        """Should fail when not initialized."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()
        connector._initialized = False

        result = await connector.send_proactive_notification("user123", "Test message")

        assert result is False

    @pytest.mark.asyncio
    async def test_send_proactive_notification_success(self):
        """Should send proactive notification successfully."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()
        connector._initialized = True
        connector._access_token = "test-token"
        connector._token_expires_at = 9999999999

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = (True, {}, None)

            result = await connector.send_proactive_notification(
                "user123",
                "Your debate has concluded.",
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_send_notification_success(self):
        """Should send notification via broadcast."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector
        from aragora.connectors.devices.models import (
            DeviceToken,
            DeviceType,
            DeviceMessage,
            DeliveryStatus,
        )

        connector = GoogleHomeConnector()
        connector._initialized = True
        connector._access_token = "test-token"
        connector._token_expires_at = 9999999999

        device = DeviceToken(
            device_id="d1",
            user_id="user123",
            device_type=DeviceType.GOOGLE_HOME,
            push_token="token",
        )
        message = DeviceMessage(title="Decision Ready", body="Your debate has concluded.")

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = (True, {}, None)

            result = await connector.send_notification(device, message)

        assert result.success is True
        assert result.status == DeliveryStatus.SENT

    @pytest.mark.asyncio
    async def test_send_notification_failure(self):
        """Should handle notification failure."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector
        from aragora.connectors.devices.models import (
            DeviceToken,
            DeviceType,
            DeviceMessage,
            DeliveryStatus,
        )

        connector = GoogleHomeConnector()
        connector._initialized = True
        connector._access_token = "test-token"
        connector._token_expires_at = 9999999999

        device = DeviceToken(
            device_id="d1",
            user_id="user123",
            device_type=DeviceType.GOOGLE_HOME,
            push_token="token",
        )
        message = DeviceMessage(title="Test", body="Test")

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = (False, None, "API Error")

            result = await connector.send_notification(device, message)

        assert result.success is False
        assert result.status == DeliveryStatus.FAILED


class TestGoogleHomeAccountLinking:
    """Tests for account linking."""

    @pytest.mark.asyncio
    async def test_link_account_success(self):
        """Should link account successfully."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()

        with patch("aragora.connectors.devices.google_home.get_session_store") as mock_store:
            mock_instance = MagicMock()
            mock_store.return_value = mock_instance

            result = await connector.link_account(
                google_user_id="google123",
                aragora_user_id="aragora456",
                access_token="oauth-token",
            )

        assert result is True
        mock_instance.set_device_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_link_account_import_error(self):
        """Should handle import error gracefully."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()

        with patch(
            "aragora.connectors.devices.google_home.get_session_store",
            side_effect=ImportError("Module not found"),
        ):
            # The function catches ImportError inside
            result = await connector.link_account("g1", "a1", "token")

        # Should fail gracefully
        assert result is False

    @pytest.mark.asyncio
    async def test_unlink_account_success(self):
        """Should unlink account successfully."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()

        with patch("aragora.connectors.devices.google_home.get_session_store") as mock_store:
            mock_instance = MagicMock()
            mock_instance.delete_device_session.return_value = True
            mock_store.return_value = mock_instance

            result = await connector.unlink_account("google123")

        assert result is True


class TestGoogleHomeHealth:
    """Tests for health status."""

    @pytest.mark.asyncio
    async def test_get_health_uninitialized(self):
        """Should return health status for uninitialized connector."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()

        health = await connector.get_health()

        assert health["platform"] == "google_home"
        assert health["initialized"] is False
        assert health["configured"] is False

    @pytest.mark.asyncio
    async def test_get_health_initialized(self):
        """Should return health status for initialized connector."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()
        connector._initialized = True
        connector._project_id = "test-project"
        connector._credentials = {"client_email": "test@test.com"}
        connector._access_token = "token"
        connector._token_expires_at = 9999999999

        health = await connector.get_health()

        assert health["platform"] == "google_home"
        assert health["initialized"] is True
        assert health["configured"] is True
        assert health["has_credentials"] is True
        assert health["has_access_token"] is True
        assert health["token_valid"] is True

    @pytest.mark.asyncio
    async def test_get_health_expired_token(self):
        """Should indicate expired token."""
        from aragora.connectors.devices.google_home import GoogleHomeConnector

        connector = GoogleHomeConnector()
        connector._initialized = True
        connector._project_id = "test-project"
        connector._credentials = {}
        connector._access_token = "token"
        connector._token_expires_at = 0  # Expired

        health = await connector.get_health()

        assert health["token_valid"] is False
