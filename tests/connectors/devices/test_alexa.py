"""
Tests for AlexaConnector - Amazon Alexa device integration.

Tests cover:
- Connector initialization and configuration
- Request verification (signature, skill ID)
- Voice request parsing and handling
- Response building
- Intent handlers (help, stop, fallback, start_debate, get_decision, list_debates, get_status)
- Proactive notifications
- Account linking
- Token management
- Health status
- Error handling
"""

import json
import os
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


class TestAlexaConnectorInit:
    """Tests for AlexaConnector initialization."""

    def test_default_init(self):
        """Should initialize with default values."""
        from aragora.connectors.devices.alexa import AlexaConnector

        connector = AlexaConnector()

        assert connector.platform_name == "alexa"
        assert connector.platform_display_name == "Amazon Alexa"

    def test_init_with_config(self):
        """Should accept custom configuration."""
        from aragora.connectors.devices.base import DeviceConnectorConfig
        from aragora.connectors.devices.alexa import AlexaConnector

        config = DeviceConnectorConfig(
            enable_circuit_breaker=False,
            request_timeout=45.0,
        )
        connector = AlexaConnector(config=config)

        assert connector.config.enable_circuit_breaker is False
        assert connector.config.request_timeout == 45.0

    def test_supported_device_types(self):
        """Should support Alexa device type."""
        from aragora.connectors.devices.alexa import AlexaConnector
        from aragora.connectors.devices.models import DeviceType

        connector = AlexaConnector()

        assert DeviceType.ALEXA in connector.supported_device_types

    def test_intent_handlers_registered(self):
        """Should register default intent handlers."""
        from aragora.connectors.devices.alexa import AlexaConnector, AlexaIntent

        connector = AlexaConnector()

        assert AlexaIntent.HELP.value in connector._intent_handlers
        assert AlexaIntent.STOP.value in connector._intent_handlers
        assert AlexaIntent.CANCEL.value in connector._intent_handlers
        assert AlexaIntent.FALLBACK.value in connector._intent_handlers
        assert AlexaIntent.START_DEBATE.value in connector._intent_handlers
        assert AlexaIntent.GET_DECISION.value in connector._intent_handlers

    def test_proactive_api_endpoint(self):
        """Should use correct proactive API endpoint."""
        from aragora.connectors.devices.alexa import AlexaConnector

        connector = AlexaConnector()

        assert "proactiveEvents" in connector._proactive_api_endpoint


class TestAlexaInitialization:
    """Tests for connector initialization lifecycle."""

    @pytest.mark.asyncio
    async def test_initialize_without_credentials(self):
        """Should fail initialization without credentials."""
        from aragora.connectors.devices.alexa import AlexaConnector

        with patch.dict(os.environ, {}, clear=True):
            connector = AlexaConnector()
            result = await connector.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_with_partial_credentials(self):
        """Should fail with only partial credentials."""
        from aragora.connectors.devices.alexa import AlexaConnector

        with patch.dict(
            os.environ,
            {
                "ALEXA_CLIENT_ID": "client123",
                # Missing client_secret and skill_id
            },
            clear=True,
        ):
            connector = AlexaConnector()
            result = await connector.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Should initialize with valid credentials."""
        from aragora.connectors.devices.alexa import AlexaConnector

        with patch.dict(
            os.environ,
            {
                "ALEXA_CLIENT_ID": "client123",
                "ALEXA_CLIENT_SECRET": "secret456",
                "ALEXA_SKILL_ID": "amzn1.ask.skill.test",
            },
        ):
            connector = AlexaConnector()

            with patch.object(
                connector, "_refresh_access_token", new_callable=AsyncMock
            ) as mock_refresh:
                mock_refresh.return_value = "test-token"
                result = await connector.initialize()

        assert result is True
        assert connector._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_token_failure(self):
        """Should fail if token refresh fails."""
        from aragora.connectors.devices.alexa import AlexaConnector

        with patch.dict(
            os.environ,
            {
                "ALEXA_CLIENT_ID": "client123",
                "ALEXA_CLIENT_SECRET": "secret456",
                "ALEXA_SKILL_ID": "amzn1.ask.skill.test",
            },
        ):
            connector = AlexaConnector()

            with patch.object(
                connector,
                "_refresh_access_token",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Token error"),
            ):
                result = await connector.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Should clear state on shutdown."""
        from aragora.connectors.devices.alexa import AlexaConnector

        connector = AlexaConnector()
        connector._initialized = True
        connector._access_token = "test-token"
        connector._token_expires_at = 12345

        await connector.shutdown()

        assert connector._initialized is False
        assert connector._access_token is None
        assert connector._token_expires_at == 0


class TestAlexaRequestVerification:
    """Tests for request verification."""

    def test_verify_request_missing_headers(self):
        """Should fail verification with missing headers."""
        from aragora.connectors.devices.alexa import AlexaConnector

        connector = AlexaConnector()

        result = connector.verify_request(b"body", "", "")

        assert result is False

    def test_verify_request_invalid_scheme(self):
        """Should fail verification with non-HTTPS certificate URL."""
        from aragora.connectors.devices.alexa import AlexaConnector

        connector = AlexaConnector()

        result = connector.verify_request(
            b"body",
            "signature",
            "http://s3.amazonaws.com/echo.api/cert",
        )

        assert result is False

    def test_verify_request_invalid_host(self):
        """Should fail verification with wrong hostname."""
        from aragora.connectors.devices.alexa import AlexaConnector

        connector = AlexaConnector()

        result = connector.verify_request(
            b"body",
            "signature",
            "https://evil.com/echo.api/cert",
        )

        assert result is False

    def test_verify_request_invalid_path(self):
        """Should fail verification with wrong path."""
        from aragora.connectors.devices.alexa import AlexaConnector

        connector = AlexaConnector()

        result = connector.verify_request(
            b"body",
            "signature",
            "https://s3.amazonaws.com/wrong/path/cert",
        )

        assert result is False

    def test_verify_request_valid(self):
        """Should pass basic verification with valid URL format."""
        from aragora.connectors.devices.alexa import AlexaConnector

        connector = AlexaConnector()

        result = connector.verify_request(
            b"request body",
            "signature",
            "https://s3.amazonaws.com/echo.api/cert.pem",
        )

        assert result is True

    def test_verify_skill_id_success(self):
        """Should verify matching skill ID."""
        from aragora.connectors.devices.alexa import AlexaConnector

        with patch.dict(os.environ, {"ALEXA_SKILL_ID": "amzn1.ask.skill.test123"}):
            connector = AlexaConnector()

        request_data = {
            "context": {
                "System": {
                    "application": {
                        "applicationId": "amzn1.ask.skill.test123",
                    },
                },
            },
        }

        result = connector.verify_skill_id(request_data)

        assert result is True

    def test_verify_skill_id_mismatch(self):
        """Should reject mismatched skill ID."""
        from aragora.connectors.devices.alexa import AlexaConnector

        with patch.dict(os.environ, {"ALEXA_SKILL_ID": "amzn1.ask.skill.correct"}):
            connector = AlexaConnector()

        request_data = {
            "context": {
                "System": {
                    "application": {
                        "applicationId": "amzn1.ask.skill.wrong",
                    },
                },
            },
        }

        result = connector.verify_skill_id(request_data)

        assert result is False

    def test_verify_skill_id_from_session(self):
        """Should verify skill ID from session if not in context."""
        from aragora.connectors.devices.alexa import AlexaConnector

        with patch.dict(os.environ, {"ALEXA_SKILL_ID": "amzn1.ask.skill.test123"}):
            connector = AlexaConnector()

        request_data = {
            "context": {"System": {}},
            "session": {
                "application": {
                    "applicationId": "amzn1.ask.skill.test123",
                },
            },
        }

        result = connector.verify_skill_id(request_data)

        assert result is True


class TestAlexaRequestParsing:
    """Tests for parsing Alexa requests."""

    def test_parse_launch_request(self):
        """Should parse LaunchRequest."""
        from aragora.connectors.devices.alexa import AlexaConnector
        from aragora.connectors.devices.models import DeviceType

        connector = AlexaConnector()

        request_data = {
            "request": {
                "type": "LaunchRequest",
                "requestId": "req123",
                "timestamp": "2024-01-15T10:30:00Z",
                "locale": "en-US",
            },
            "session": {
                "sessionId": "sess123",
                "new": True,
            },
            "context": {
                "System": {
                    "user": {"userId": "user123"},
                    "device": {"deviceId": "device123"},
                    "apiEndpoint": "https://api.amazonalexa.com",
                    "apiAccessToken": "api-token",
                },
            },
        }

        result = connector.parse_alexa_request(request_data)

        assert result.intent == "LaunchRequest"
        assert result.user_id == "user123"
        assert result.device_type == DeviceType.ALEXA
        assert result.session_id == "sess123"
        assert result.is_new_session is True
        assert result.locale == "en-US"

    def test_parse_intent_request(self):
        """Should parse IntentRequest with slots."""
        from aragora.connectors.devices.alexa import AlexaConnector

        connector = AlexaConnector()

        request_data = {
            "request": {
                "type": "IntentRequest",
                "requestId": "req123",
                "timestamp": "2024-01-15T10:30:00Z",
                "locale": "en-US",
                "intent": {
                    "name": "StartDebateIntent",
                    "slots": {
                        "topic": {
                            "name": "topic",
                            "value": "climate change",
                        },
                    },
                },
            },
            "session": {"sessionId": "s1", "new": False},
            "context": {"System": {"user": {"userId": "u1"}}},
        }

        result = connector.parse_alexa_request(request_data)

        assert result.intent == "StartDebateIntent"
        assert result.slots["topic"] == "climate change"
        assert result.is_new_session is False

    def test_parse_session_ended_request(self):
        """Should parse SessionEndedRequest."""
        from aragora.connectors.devices.alexa import AlexaConnector

        connector = AlexaConnector()

        request_data = {
            "request": {
                "type": "SessionEndedRequest",
                "requestId": "req123",
            },
            "session": {"sessionId": "s1"},
            "context": {"System": {"user": {"userId": "u1"}}},
        }

        result = connector.parse_alexa_request(request_data)

        assert result.intent == "SessionEndedRequest"

    def test_parse_request_with_access_token(self):
        """Should extract access token from account linking."""
        from aragora.connectors.devices.alexa import AlexaConnector

        connector = AlexaConnector()

        request_data = {
            "request": {
                "type": "IntentRequest",
                "intent": {"name": "TestIntent"},
            },
            "session": {"sessionId": "s1"},
            "context": {
                "System": {
                    "user": {
                        "userId": "u1",
                        "accessToken": "oauth-token-123",
                    },
                },
            },
        }

        result = connector.parse_alexa_request(request_data)

        assert result.metadata["access_token"] == "oauth-token-123"


class TestAlexaResponseBuilding:
    """Tests for building Alexa responses."""

    def test_build_alexa_response_basic(self):
        """Should build basic response."""
        from aragora.connectors.devices.alexa import AlexaConnector
        from aragora.connectors.devices.models import VoiceDeviceResponse

        connector = AlexaConnector()
        response = VoiceDeviceResponse(text="Hello, world!")

        result = connector.build_alexa_response(response)

        assert result["version"] == "1.0"
        assert result["response"]["outputSpeech"]["type"] == "PlainText"
        assert result["response"]["outputSpeech"]["text"] == "Hello, world!"
        assert result["response"]["shouldEndSession"] is True

    def test_build_alexa_response_with_reprompt(self):
        """Should include reprompt when provided."""
        from aragora.connectors.devices.alexa import AlexaConnector
        from aragora.connectors.devices.models import VoiceDeviceResponse

        connector = AlexaConnector()
        response = VoiceDeviceResponse(
            text="What next?",
            should_end_session=False,
            reprompt="Please choose an option.",
        )

        result = connector.build_alexa_response(response)

        assert result["response"]["shouldEndSession"] is False
        assert result["response"]["reprompt"]["outputSpeech"]["text"] == "Please choose an option."

    def test_build_alexa_response_with_simple_card(self):
        """Should include simple card."""
        from aragora.connectors.devices.alexa import AlexaConnector
        from aragora.connectors.devices.models import VoiceDeviceResponse

        connector = AlexaConnector()
        response = VoiceDeviceResponse(
            text="Result",
            card_title="My Title",
            card_content="My content",
        )

        result = connector.build_alexa_response(response)

        card = result["response"]["card"]
        assert card["type"] == "Simple"
        assert card["title"] == "My Title"
        assert card["content"] == "My content"

    def test_build_alexa_response_with_image_card(self):
        """Should include standard card with image."""
        from aragora.connectors.devices.alexa import AlexaConnector
        from aragora.connectors.devices.models import VoiceDeviceResponse

        connector = AlexaConnector()
        response = VoiceDeviceResponse(
            text="Result",
            card_title="My Title",
            card_content="My content",
            card_image_url="https://example.com/image.png",
        )

        result = connector.build_alexa_response(response)

        card = result["response"]["card"]
        assert card["type"] == "Standard"
        assert card["image"]["smallImageUrl"] == "https://example.com/image.png"

    def test_build_alexa_response_with_directives(self):
        """Should include directives."""
        from aragora.connectors.devices.alexa import AlexaConnector
        from aragora.connectors.devices.models import VoiceDeviceResponse

        connector = AlexaConnector()
        response = VoiceDeviceResponse(
            text="Playing audio",
            directives=[{"type": "AudioPlayer.Play"}],
        )

        result = connector.build_alexa_response(response)

        assert result["response"]["directives"] == [{"type": "AudioPlayer.Play"}]

    def test_build_alexa_response_with_session_attributes(self):
        """Should include session attributes."""
        from aragora.connectors.devices.alexa import AlexaConnector
        from aragora.connectors.devices.models import VoiceDeviceResponse

        connector = AlexaConnector()
        response = VoiceDeviceResponse(text="Continue", should_end_session=False)
        attributes = {"debate_id": "123", "step": 2}

        result = connector.build_alexa_response(response, attributes)

        assert result["sessionAttributes"] == attributes


class TestAlexaIntentHandlers:
    """Tests for intent handlers."""

    @pytest.mark.asyncio
    async def test_handle_help_intent(self):
        """Should handle help intent."""
        from aragora.connectors.devices.alexa import AlexaConnector, AlexaIntent
        from aragora.connectors.devices.models import VoiceDeviceRequest, DeviceType

        connector = AlexaConnector()
        request = VoiceDeviceRequest(
            request_id="r1",
            device_type=DeviceType.ALEXA,
            user_id="u1",
            intent=AlexaIntent.HELP.value,
        )

        result = await connector.handle_voice_request(request)

        assert "debate" in result.text.lower()
        assert result.should_end_session is False
        assert result.card_title == "Aragora Help"

    @pytest.mark.asyncio
    async def test_handle_stop_intent(self):
        """Should handle stop intent."""
        from aragora.connectors.devices.alexa import AlexaConnector, AlexaIntent
        from aragora.connectors.devices.models import VoiceDeviceRequest, DeviceType

        connector = AlexaConnector()
        request = VoiceDeviceRequest(
            request_id="r1",
            device_type=DeviceType.ALEXA,
            user_id="u1",
            intent=AlexaIntent.STOP.value,
        )

        result = await connector.handle_voice_request(request)

        assert "goodbye" in result.text.lower()
        assert result.should_end_session is True

    @pytest.mark.asyncio
    async def test_handle_cancel_intent(self):
        """Should handle cancel intent same as stop."""
        from aragora.connectors.devices.alexa import AlexaConnector, AlexaIntent
        from aragora.connectors.devices.models import VoiceDeviceRequest, DeviceType

        connector = AlexaConnector()
        request = VoiceDeviceRequest(
            request_id="r1",
            device_type=DeviceType.ALEXA,
            user_id="u1",
            intent=AlexaIntent.CANCEL.value,
        )

        result = await connector.handle_voice_request(request)

        assert result.should_end_session is True

    @pytest.mark.asyncio
    async def test_handle_fallback_intent(self):
        """Should handle fallback intent."""
        from aragora.connectors.devices.alexa import AlexaConnector, AlexaIntent
        from aragora.connectors.devices.models import VoiceDeviceRequest, DeviceType

        connector = AlexaConnector()
        request = VoiceDeviceRequest(
            request_id="r1",
            device_type=DeviceType.ALEXA,
            user_id="u1",
            intent=AlexaIntent.FALLBACK.value,
        )

        result = await connector.handle_voice_request(request)

        assert "didn't understand" in result.text.lower()
        assert result.should_end_session is False

    @pytest.mark.asyncio
    async def test_handle_start_debate_with_topic(self):
        """Should start debate with provided topic."""
        from aragora.connectors.devices.alexa import AlexaConnector, AlexaIntent
        from aragora.connectors.devices.models import VoiceDeviceRequest, DeviceType

        connector = AlexaConnector()
        request = VoiceDeviceRequest(
            request_id="r1",
            device_type=DeviceType.ALEXA,
            user_id="u1",
            intent=AlexaIntent.START_DEBATE.value,
            slots={"topic": "AI ethics"},
        )

        result = await connector.handle_voice_request(request)

        assert "AI ethics" in result.text
        assert result.should_end_session is True

    @pytest.mark.asyncio
    async def test_handle_start_debate_without_topic(self):
        """Should prompt for topic when not provided."""
        from aragora.connectors.devices.alexa import AlexaConnector, AlexaIntent
        from aragora.connectors.devices.models import VoiceDeviceRequest, DeviceType

        connector = AlexaConnector()
        request = VoiceDeviceRequest(
            request_id="r1",
            device_type=DeviceType.ALEXA,
            user_id="u1",
            intent=AlexaIntent.START_DEBATE.value,
            slots={},
        )

        result = await connector.handle_voice_request(request)

        assert "topic" in result.text.lower()
        assert result.should_end_session is False

    @pytest.mark.asyncio
    async def test_handle_get_decision(self):
        """Should return latest decision."""
        from aragora.connectors.devices.alexa import AlexaConnector, AlexaIntent
        from aragora.connectors.devices.models import VoiceDeviceRequest, DeviceType

        connector = AlexaConnector()
        request = VoiceDeviceRequest(
            request_id="r1",
            device_type=DeviceType.ALEXA,
            user_id="u1",
            intent=AlexaIntent.GET_DECISION.value,
        )

        result = await connector.handle_voice_request(request)

        assert result.should_end_session is True
        assert result.card_title == "Latest Decision"

    @pytest.mark.asyncio
    async def test_handle_list_debates(self):
        """Should list user debates."""
        from aragora.connectors.devices.alexa import AlexaConnector, AlexaIntent
        from aragora.connectors.devices.models import VoiceDeviceRequest, DeviceType

        connector = AlexaConnector()
        request = VoiceDeviceRequest(
            request_id="r1",
            device_type=DeviceType.ALEXA,
            user_id="u1",
            intent=AlexaIntent.LIST_DEBATES.value,
        )

        result = await connector.handle_voice_request(request)

        assert "debates" in result.text.lower()
        assert result.should_end_session is False

    @pytest.mark.asyncio
    async def test_handle_get_status(self):
        """Should return debate status."""
        from aragora.connectors.devices.alexa import AlexaConnector, AlexaIntent
        from aragora.connectors.devices.models import VoiceDeviceRequest, DeviceType

        connector = AlexaConnector()
        request = VoiceDeviceRequest(
            request_id="r1",
            device_type=DeviceType.ALEXA,
            user_id="u1",
            intent=AlexaIntent.GET_STATUS.value,
            slots={"debate_name": "marketing"},
        )

        result = await connector.handle_voice_request(request)

        assert result.should_end_session is True
        assert result.card_title == "Debate Status"

    @pytest.mark.asyncio
    async def test_handle_unknown_intent(self):
        """Should handle unknown intents gracefully."""
        from aragora.connectors.devices.alexa import AlexaConnector
        from aragora.connectors.devices.models import VoiceDeviceRequest, DeviceType

        connector = AlexaConnector()
        request = VoiceDeviceRequest(
            request_id="r1",
            device_type=DeviceType.ALEXA,
            user_id="u1",
            intent="UnknownCustomIntent",
        )

        result = await connector.handle_voice_request(request)

        assert result.should_end_session is False
        assert result.reprompt is not None


class TestAlexaProactiveNotifications:
    """Tests for proactive notifications."""

    @pytest.mark.asyncio
    async def test_send_proactive_notification_not_initialized(self):
        """Should fail when not initialized."""
        from aragora.connectors.devices.alexa import AlexaConnector

        connector = AlexaConnector()
        connector._initialized = False

        result = await connector.send_proactive_notification("user123", "Test message")

        assert result is False

    @pytest.mark.asyncio
    async def test_send_proactive_notification_success(self):
        """Should send proactive notification successfully."""
        from aragora.connectors.devices.alexa import AlexaConnector

        connector = AlexaConnector()
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
        mock_http.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_proactive_notification_with_event_type(self):
        """Should use custom event type."""
        from aragora.connectors.devices.alexa import AlexaConnector

        connector = AlexaConnector()
        connector._initialized = True
        connector._access_token = "test-token"
        connector._token_expires_at = 9999999999

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = (True, {}, None)

            await connector.send_proactive_notification(
                "user123",
                "Test",
                event_type="CUSTOM.Event",
                reference_id="ref123",
            )

            call_kwargs = mock_http.call_args[1]
            payload = call_kwargs["json"]
            assert payload["event"]["name"] == "CUSTOM.Event"
            assert payload["referenceId"] == "ref123"

    @pytest.mark.asyncio
    async def test_send_proactive_notification_failure(self):
        """Should handle notification failure."""
        from aragora.connectors.devices.alexa import AlexaConnector

        connector = AlexaConnector()
        connector._initialized = True
        connector._access_token = "test-token"
        connector._token_expires_at = 9999999999

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = (False, None, "API Error")

            result = await connector.send_proactive_notification("user123", "Test")

        assert result is False

    @pytest.mark.asyncio
    async def test_send_notification_success(self):
        """Should send notification via proactive API."""
        from aragora.connectors.devices.alexa import AlexaConnector
        from aragora.connectors.devices.models import (
            DeviceToken,
            DeviceType,
            DeviceMessage,
            DeliveryStatus,
        )

        connector = AlexaConnector()
        connector._initialized = True
        connector._access_token = "test-token"
        connector._token_expires_at = 9999999999

        device = DeviceToken(
            device_id="d1",
            user_id="user123",
            device_type=DeviceType.ALEXA,
            push_token="token",
        )
        message = DeviceMessage(title="Decision Ready", body="Your debate has concluded.")

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = (True, {}, None)

            result = await connector.send_notification(device, message)

        assert result.success is True
        assert result.status == DeliveryStatus.SENT


class TestAlexaAccountLinking:
    """Tests for account linking."""

    @pytest.mark.asyncio
    async def test_link_account_success(self):
        """Should link account successfully."""
        from aragora.connectors.devices.alexa import AlexaConnector

        connector = AlexaConnector()

        with patch("aragora.connectors.devices.alexa.get_session_store") as mock_store:
            mock_instance = MagicMock()
            mock_store.return_value = mock_instance

            result = await connector.link_account(
                alexa_user_id="alexa123",
                aragora_user_id="aragora456",
                access_token="oauth-token",
            )

        assert result is True
        mock_instance.set_device_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_link_account_import_error(self):
        """Should handle import error gracefully."""
        from aragora.connectors.devices.alexa import AlexaConnector

        connector = AlexaConnector()

        with patch(
            "aragora.connectors.devices.alexa.get_session_store",
            side_effect=ImportError("Module not found"),
        ):
            result = await connector.link_account("a1", "a2", "token")

        assert result is False

    @pytest.mark.asyncio
    async def test_unlink_account_success(self):
        """Should unlink account successfully."""
        from aragora.connectors.devices.alexa import AlexaConnector

        connector = AlexaConnector()

        with patch("aragora.connectors.devices.alexa.get_session_store") as mock_store:
            mock_instance = MagicMock()
            mock_instance.delete_device_session.return_value = True
            mock_store.return_value = mock_instance

            result = await connector.unlink_account("alexa123")

        assert result is True


class TestAlexaTokenManagement:
    """Tests for OAuth token management."""

    @pytest.mark.asyncio
    async def test_refresh_access_token_success(self):
        """Should refresh access token successfully."""
        from aragora.connectors.devices.alexa import AlexaConnector

        with patch.dict(
            os.environ,
            {
                "ALEXA_CLIENT_ID": "client123",
                "ALEXA_CLIENT_SECRET": "secret456",
                "ALEXA_SKILL_ID": "amzn1.ask.skill.test",
            },
        ):
            connector = AlexaConnector()

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = (True, {"access_token": "new-token", "expires_in": 3600}, None)

            token = await connector._refresh_access_token()

        assert token == "new-token"
        assert connector._access_token == "new-token"

    @pytest.mark.asyncio
    async def test_refresh_access_token_uses_cached(self):
        """Should use cached token if not expired."""
        from aragora.connectors.devices.alexa import AlexaConnector

        connector = AlexaConnector()
        connector._access_token = "cached-token"
        connector._token_expires_at = 9999999999  # Far in the future

        token = await connector._refresh_access_token()

        assert token == "cached-token"

    @pytest.mark.asyncio
    async def test_refresh_access_token_failure(self):
        """Should raise exception on token refresh failure."""
        from aragora.connectors.devices.alexa import AlexaConnector

        with patch.dict(
            os.environ,
            {
                "ALEXA_CLIENT_ID": "client123",
                "ALEXA_CLIENT_SECRET": "secret456",
                "ALEXA_SKILL_ID": "amzn1.ask.skill.test",
            },
        ):
            connector = AlexaConnector()

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = (False, None, "Auth error")

            with pytest.raises(Exception) as exc_info:
                await connector._refresh_access_token()

            assert "token" in str(exc_info.value).lower()


class TestAlexaHealth:
    """Tests for health status."""

    @pytest.mark.asyncio
    async def test_get_health_uninitialized(self):
        """Should return health status for uninitialized connector."""
        from aragora.connectors.devices.alexa import AlexaConnector

        connector = AlexaConnector()

        health = await connector.get_health()

        assert health["platform"] == "alexa"
        assert health["initialized"] is False
        assert health["configured"] is False

    @pytest.mark.asyncio
    async def test_get_health_initialized(self):
        """Should return health status for initialized connector."""
        from aragora.connectors.devices.alexa import AlexaConnector

        with patch.dict(
            os.environ,
            {
                "ALEXA_CLIENT_ID": "client123",
                "ALEXA_CLIENT_SECRET": "secret456",
                "ALEXA_SKILL_ID": "amzn1.ask.skill.test123456789",
            },
        ):
            connector = AlexaConnector()

        connector._initialized = True
        connector._access_token = "token"
        connector._token_expires_at = 9999999999

        health = await connector.get_health()

        assert health["platform"] == "alexa"
        assert health["initialized"] is True
        assert health["configured"] is True
        assert health["has_access_token"] is True
        assert health["token_valid"] is True
        # Skill ID should be truncated
        assert "..." in health["skill_id"]

    @pytest.mark.asyncio
    async def test_get_health_expired_token(self):
        """Should indicate expired token."""
        from aragora.connectors.devices.alexa import AlexaConnector

        with patch.dict(
            os.environ,
            {
                "ALEXA_CLIENT_ID": "client123",
                "ALEXA_CLIENT_SECRET": "secret456",
                "ALEXA_SKILL_ID": "amzn1.ask.skill.test",
            },
        ):
            connector = AlexaConnector()

        connector._initialized = True
        connector._access_token = "token"
        connector._token_expires_at = 0  # Expired

        health = await connector.get_health()

        assert health["token_valid"] is False
