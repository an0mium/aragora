"""Comprehensive tests for WhatsApp Cloud API messaging utilities.

Covers all functions in aragora/server/handlers/social/whatsapp/messaging.py:
- send_text_message() - text message sending via Cloud API
- send_interactive_buttons() - interactive button message sending + fallback
- send_voice_summary() - TTS-based voice summary of debate results
- send_voice_message() - audio upload + send via WhatsApp Media API
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Module path for patching
_MSG = "aragora.server.handlers.social.whatsapp.messaging"
_CONFIG = "aragora.server.handlers.social.whatsapp.config"
_HTTP_POOL = "aragora.server.http_client_pool.get_http_pool"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(status_code: int = 200, json_data: dict | None = None):
    """Create a mock HTTP response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


def _mock_session_and_pool(response=None):
    """Create a mock aiohttp-style session and pool pair.

    Returns (pool, session) where pool.get_session() is the async context manager.
    """
    mock_session = AsyncMock()
    if response is not None:
        mock_session.post.return_value = response
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.get_session.return_value = mock_session
    return mock_pool, mock_session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_config(monkeypatch):
    """Set config tokens so messaging functions don't early-return."""
    monkeypatch.setattr(f"{_MSG}.WHATSAPP_ACCESS_TOKEN", "test-token")
    monkeypatch.setattr(f"{_MSG}.WHATSAPP_PHONE_NUMBER_ID", "123456789")
    monkeypatch.setattr(f"{_MSG}.WHATSAPP_API_BASE", "https://graph.facebook.com/v18.0")


@pytest.fixture(autouse=True)
def _patch_telemetry(monkeypatch):
    """Patch telemetry to avoid side effects."""
    monkeypatch.setattr(f"{_MSG}.record_api_call", MagicMock())
    monkeypatch.setattr(f"{_MSG}.record_api_latency", MagicMock())


@pytest.fixture
def mock_pool_success():
    """Create a pool/session pair that returns 200."""
    return _mock_session_and_pool(_mock_response(200))


@pytest.fixture
def mock_pool_error():
    """Create a pool/session pair that returns 500."""
    return _mock_session_and_pool(_mock_response(500, {"error": "Internal error"}))


# ===========================================================================
# send_text_message Tests
# ===========================================================================


class TestSendTextMessage:
    """Tests for send_text_message()."""

    @pytest.mark.asyncio
    async def test_sends_text_message_successfully(self, mock_pool_success):
        """Successful text message sends with correct payload."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        pool, session = mock_pool_success
        with patch(_HTTP_POOL, return_value=pool):
            await send_text_message("+1234567890", "Hello from Aragora")

        session.post.assert_called_once()
        call_kwargs = session.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["messaging_product"] == "whatsapp"
        assert payload["recipient_type"] == "individual"
        assert payload["to"] == "+1234567890"
        assert payload["type"] == "text"
        assert payload["text"]["body"] == "Hello from Aragora"
        assert payload["text"]["preview_url"] is False

    @pytest.mark.asyncio
    async def test_sends_to_correct_url(self, mock_pool_success):
        """URL includes the PHONE_NUMBER_ID."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        pool, session = mock_pool_success
        with patch(_HTTP_POOL, return_value=pool):
            await send_text_message("+1234567890", "Test")

        url = session.post.call_args[0][0]
        assert "123456789/messages" in url
        assert "graph.facebook.com" in url

    @pytest.mark.asyncio
    async def test_sends_correct_headers(self, mock_pool_success):
        """Request includes Bearer token and content type."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        pool, session = mock_pool_success
        with patch(_HTTP_POOL, return_value=pool):
            await send_text_message("+1234567890", "Test")

        headers = session.post.call_args.kwargs.get("headers") or session.post.call_args[1].get("headers")
        assert headers["Authorization"] == "Bearer test-token"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_sends_with_30s_timeout(self, mock_pool_success):
        """Text messages use a 30-second timeout."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        pool, session = mock_pool_success
        with patch(_HTTP_POOL, return_value=pool):
            await send_text_message("+1234567890", "Test")

        timeout = session.post.call_args.kwargs.get("timeout") or session.post.call_args[1].get("timeout")
        assert timeout == 30

    @pytest.mark.asyncio
    async def test_returns_none_when_token_missing(self, monkeypatch):
        """Returns early when access token is not configured."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        monkeypatch.setattr(f"{_MSG}.WHATSAPP_ACCESS_TOKEN", "")
        with patch(_HTTP_POOL) as mock_get_pool:
            await send_text_message("+1234567890", "Test")
            mock_get_pool.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_none_when_token_none(self, monkeypatch):
        """Returns early when access token is None."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        monkeypatch.setattr(f"{_MSG}.WHATSAPP_ACCESS_TOKEN", None)
        with patch(_HTTP_POOL) as mock_get_pool:
            await send_text_message("+1234567890", "Test")
            mock_get_pool.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_none_when_phone_id_missing(self, monkeypatch):
        """Returns early when phone number ID is not configured."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        monkeypatch.setattr(f"{_MSG}.WHATSAPP_PHONE_NUMBER_ID", "")
        with patch(_HTTP_POOL) as mock_get_pool:
            await send_text_message("+1234567890", "Test")
            mock_get_pool.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_none_when_phone_id_none(self, monkeypatch):
        """Returns early when phone number ID is None."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        monkeypatch.setattr(f"{_MSG}.WHATSAPP_PHONE_NUMBER_ID", None)
        with patch(_HTTP_POOL) as mock_get_pool:
            await send_text_message("+1234567890", "Test")
            mock_get_pool.assert_not_called()

    @pytest.mark.asyncio
    async def test_logs_warning_on_api_error(self, mock_pool_error):
        """Logs a warning when the API returns a non-200 status."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        pool, _ = mock_pool_error
        with patch(_HTTP_POOL, return_value=pool):
            with patch(f"{_MSG}.logger") as mock_logger:
                await send_text_message("+1234567890", "Test")
                mock_logger.warning.assert_called()
                assert "WhatsApp API error" in mock_logger.warning.call_args[0][0]

    @pytest.mark.asyncio
    async def test_handles_os_error(self):
        """Catches OSError without raising."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        pool, session = _mock_session_and_pool()
        session.post.side_effect = OSError("network unreachable")
        with patch(_HTTP_POOL, return_value=pool):
            await send_text_message("+1234567890", "Test")  # Should not raise

    @pytest.mark.asyncio
    async def test_handles_connection_error(self):
        """Catches ConnectionError without raising."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        pool, session = _mock_session_and_pool()
        session.post.side_effect = ConnectionError("connection refused")
        with patch(_HTTP_POOL, return_value=pool):
            await send_text_message("+1234567890", "Test")

    @pytest.mark.asyncio
    async def test_handles_timeout_error(self):
        """Catches TimeoutError without raising."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        pool, session = _mock_session_and_pool()
        session.post.side_effect = TimeoutError("request timed out")
        with patch(_HTTP_POOL, return_value=pool):
            await send_text_message("+1234567890", "Test")

    @pytest.mark.asyncio
    async def test_handles_value_error(self):
        """Catches ValueError without raising."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        pool, session = _mock_session_and_pool()
        session.post.side_effect = ValueError("invalid json")
        with patch(_HTTP_POOL, return_value=pool):
            await send_text_message("+1234567890", "Test")

    @pytest.mark.asyncio
    async def test_handles_runtime_error(self):
        """Catches RuntimeError without raising."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        pool, session = _mock_session_and_pool()
        session.post.side_effect = RuntimeError("event loop closed")
        with patch(_HTTP_POOL, return_value=pool):
            await send_text_message("+1234567890", "Test")

    @pytest.mark.asyncio
    async def test_records_success_telemetry(self, mock_pool_success):
        """Records API call and latency on success."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        mock_record_call = MagicMock()
        mock_record_latency = MagicMock()
        pool, _ = mock_pool_success
        with (
            patch(_HTTP_POOL, return_value=pool),
            patch(f"{_MSG}.record_api_call", mock_record_call),
            patch(f"{_MSG}.record_api_latency", mock_record_latency),
        ):
            await send_text_message("+1234567890", "Test")

        mock_record_call.assert_called_once_with("whatsapp", "sendMessage", "success")
        mock_record_latency.assert_called_once()
        args = mock_record_latency.call_args[0]
        assert args[0] == "whatsapp"
        assert args[1] == "sendMessage"
        assert isinstance(args[2], float)

    @pytest.mark.asyncio
    async def test_records_error_telemetry_on_api_error(self, mock_pool_error):
        """Records error status telemetry when API returns non-200."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        mock_record_call = MagicMock()
        pool, _ = mock_pool_error
        with (
            patch(_HTTP_POOL, return_value=pool),
            patch(f"{_MSG}.record_api_call", mock_record_call),
            patch(f"{_MSG}.record_api_latency", MagicMock()),
        ):
            await send_text_message("+1234567890", "Test")

        mock_record_call.assert_called_once_with("whatsapp", "sendMessage", "error")

    @pytest.mark.asyncio
    async def test_records_error_telemetry_on_exception(self):
        """Records error status telemetry when exception occurs."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        mock_record_call = MagicMock()
        pool, session = _mock_session_and_pool()
        session.post.side_effect = ConnectionError("fail")
        with (
            patch(_HTTP_POOL, return_value=pool),
            patch(f"{_MSG}.record_api_call", mock_record_call),
            patch(f"{_MSG}.record_api_latency", MagicMock()),
        ):
            await send_text_message("+1234567890", "Test")

        mock_record_call.assert_called_once_with("whatsapp", "sendMessage", "error")

    @pytest.mark.asyncio
    async def test_records_latency_on_exception(self):
        """Records latency even when exception occurs (finally block)."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        mock_record_latency = MagicMock()
        pool, session = _mock_session_and_pool()
        session.post.side_effect = TimeoutError("slow")
        with (
            patch(_HTTP_POOL, return_value=pool),
            patch(f"{_MSG}.record_api_call", MagicMock()),
            patch(f"{_MSG}.record_api_latency", mock_record_latency),
        ):
            await send_text_message("+1234567890", "Test")

        mock_record_latency.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_whatsapp_session_name(self, mock_pool_success):
        """Pool session is requested with 'whatsapp' as the session name."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        pool, _ = mock_pool_success
        with patch(_HTTP_POOL, return_value=pool):
            await send_text_message("+1234567890", "Test")

        pool.get_session.assert_called_once_with("whatsapp")

    @pytest.mark.asyncio
    async def test_empty_text_still_sends(self, mock_pool_success):
        """Empty text body is allowed (WhatsApp may reject it, but we send it)."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        pool, session = mock_pool_success
        with patch(_HTTP_POOL, return_value=pool):
            await send_text_message("+1234567890", "")

        payload = session.post.call_args.kwargs.get("json") or session.post.call_args[1].get("json")
        assert payload["text"]["body"] == ""

    @pytest.mark.asyncio
    async def test_long_text_sent_untruncated(self, mock_pool_success):
        """Long text is sent as-is (API enforces its own limits)."""
        from aragora.server.handlers.social.whatsapp.messaging import send_text_message

        long_text = "A" * 5000
        pool, session = mock_pool_success
        with patch(_HTTP_POOL, return_value=pool):
            await send_text_message("+1234567890", long_text)

        payload = session.post.call_args.kwargs.get("json") or session.post.call_args[1].get("json")
        assert payload["text"]["body"] == long_text


# ===========================================================================
# send_interactive_buttons Tests
# ===========================================================================


class TestSendInteractiveButtons:
    """Tests for send_interactive_buttons()."""

    @pytest.mark.asyncio
    async def test_sends_interactive_buttons_successfully(self, mock_pool_success):
        """Successful interactive button message with correct payload structure."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        pool, session = mock_pool_success
        buttons = [
            {"id": "btn_1", "title": "Agree"},
            {"id": "btn_2", "title": "Disagree"},
        ]
        with patch(_HTTP_POOL, return_value=pool):
            await send_interactive_buttons("+1234567890", "Do you agree?", buttons)

        session.post.assert_called_once()
        payload = session.post.call_args.kwargs.get("json") or session.post.call_args[1].get("json")
        assert payload["type"] == "interactive"
        assert payload["messaging_product"] == "whatsapp"
        assert payload["to"] == "+1234567890"
        interactive = payload["interactive"]
        assert interactive["type"] == "button"
        assert interactive["body"]["text"] == "Do you agree?"
        assert len(interactive["action"]["buttons"]) == 2

    @pytest.mark.asyncio
    async def test_button_format_correct(self, mock_pool_success):
        """Buttons have correct reply structure."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        pool, session = mock_pool_success
        buttons = [{"id": "vote_yes", "title": "Yes"}]
        with patch(_HTTP_POOL, return_value=pool):
            await send_interactive_buttons("+1234567890", "Question", buttons)

        payload = session.post.call_args.kwargs.get("json") or session.post.call_args[1].get("json")
        btn = payload["interactive"]["action"]["buttons"][0]
        assert btn["type"] == "reply"
        assert btn["reply"]["id"] == "vote_yes"
        assert btn["reply"]["title"] == "Yes"

    @pytest.mark.asyncio
    async def test_max_3_buttons_enforced(self, mock_pool_success):
        """Only first 3 buttons are sent (WhatsApp limit)."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        pool, session = mock_pool_success
        buttons = [
            {"id": f"btn_{i}", "title": f"Option {i}"} for i in range(5)
        ]
        with patch(_HTTP_POOL, return_value=pool):
            await send_interactive_buttons("+1234567890", "Choose", buttons)

        payload = session.post.call_args.kwargs.get("json") or session.post.call_args[1].get("json")
        assert len(payload["interactive"]["action"]["buttons"]) == 3

    @pytest.mark.asyncio
    async def test_button_title_truncated_to_20_chars(self, mock_pool_success):
        """Button titles are truncated to 20 characters."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        pool, session = mock_pool_success
        buttons = [{"id": "btn_1", "title": "A" * 30}]
        with patch(_HTTP_POOL, return_value=pool):
            await send_interactive_buttons("+1234567890", "Choose", buttons)

        payload = session.post.call_args.kwargs.get("json") or session.post.call_args[1].get("json")
        btn_title = payload["interactive"]["action"]["buttons"][0]["reply"]["title"]
        assert len(btn_title) == 20

    @pytest.mark.asyncio
    async def test_body_text_truncated_to_1024_chars(self, mock_pool_success):
        """Body text is truncated to 1024 characters."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        pool, session = mock_pool_success
        long_body = "B" * 2000
        buttons = [{"id": "btn_1", "title": "OK"}]
        with patch(_HTTP_POOL, return_value=pool):
            await send_interactive_buttons("+1234567890", long_body, buttons)

        payload = session.post.call_args.kwargs.get("json") or session.post.call_args[1].get("json")
        body_text = payload["interactive"]["body"]["text"]
        assert len(body_text) == 1024

    @pytest.mark.asyncio
    async def test_header_text_included_when_provided(self, mock_pool_success):
        """Optional header text is included in the interactive payload."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        pool, session = mock_pool_success
        buttons = [{"id": "btn_1", "title": "OK"}]
        with patch(_HTTP_POOL, return_value=pool):
            await send_interactive_buttons("+1234567890", "Body", buttons, header_text="My Header")

        payload = session.post.call_args.kwargs.get("json") or session.post.call_args[1].get("json")
        header = payload["interactive"]["header"]
        assert header["type"] == "text"
        assert header["text"] == "My Header"

    @pytest.mark.asyncio
    async def test_header_text_truncated_to_60_chars(self, mock_pool_success):
        """Header text is truncated to 60 characters."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        pool, session = mock_pool_success
        buttons = [{"id": "btn_1", "title": "OK"}]
        with patch(_HTTP_POOL, return_value=pool):
            await send_interactive_buttons("+1234567890", "Body", buttons, header_text="H" * 100)

        payload = session.post.call_args.kwargs.get("json") or session.post.call_args[1].get("json")
        assert len(payload["interactive"]["header"]["text"]) == 60

    @pytest.mark.asyncio
    async def test_no_header_when_none(self, mock_pool_success):
        """No header field in interactive payload when header_text is None."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        pool, session = mock_pool_success
        buttons = [{"id": "btn_1", "title": "OK"}]
        with patch(_HTTP_POOL, return_value=pool):
            await send_interactive_buttons("+1234567890", "Body", buttons, header_text=None)

        payload = session.post.call_args.kwargs.get("json") or session.post.call_args[1].get("json")
        assert "header" not in payload["interactive"]

    @pytest.mark.asyncio
    async def test_no_header_when_empty_string(self, mock_pool_success):
        """No header field in interactive payload when header_text is empty string."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        pool, session = mock_pool_success
        buttons = [{"id": "btn_1", "title": "OK"}]
        with patch(_HTTP_POOL, return_value=pool):
            await send_interactive_buttons("+1234567890", "Body", buttons, header_text="")

        payload = session.post.call_args.kwargs.get("json") or session.post.call_args[1].get("json")
        assert "header" not in payload["interactive"]

    @pytest.mark.asyncio
    async def test_returns_none_when_token_missing(self, monkeypatch):
        """Returns early when access token is not configured."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        monkeypatch.setattr(f"{_MSG}.WHATSAPP_ACCESS_TOKEN", "")
        with patch(_HTTP_POOL) as mock_get_pool:
            await send_interactive_buttons("+1234567890", "Body", [])
            mock_get_pool.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_none_when_phone_id_missing(self, monkeypatch):
        """Returns early when phone number ID is not configured."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        monkeypatch.setattr(f"{_MSG}.WHATSAPP_PHONE_NUMBER_ID", "")
        with patch(_HTTP_POOL) as mock_get_pool:
            await send_interactive_buttons("+1234567890", "Body", [])
            mock_get_pool.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_to_text_on_api_error(self, mock_pool_error):
        """Falls back to text message when interactive message API returns error."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        pool, _ = mock_pool_error
        with (
            patch(_HTTP_POOL, return_value=pool),
            patch(f"{_MSG}.send_text_message", new_callable=AsyncMock) as mock_send_text,
        ):
            await send_interactive_buttons("+1234567890", "Fallback body", [{"id": "1", "title": "OK"}])

        mock_send_text.assert_called_once_with("+1234567890", "Fallback body")

    @pytest.mark.asyncio
    async def test_fallback_to_text_on_connection_error(self):
        """Falls back to text message on ConnectionError."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        pool, session = _mock_session_and_pool()
        session.post.side_effect = ConnectionError("fail")
        with (
            patch(_HTTP_POOL, return_value=pool),
            patch(f"{_MSG}.send_text_message", new_callable=AsyncMock) as mock_send_text,
        ):
            await send_interactive_buttons("+1234567890", "Fallback body", [{"id": "1", "title": "OK"}])

        mock_send_text.assert_called_once_with("+1234567890", "Fallback body")

    @pytest.mark.asyncio
    async def test_fallback_to_text_on_timeout_error(self):
        """Falls back to text message on TimeoutError."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        pool, session = _mock_session_and_pool()
        session.post.side_effect = TimeoutError("slow")
        with (
            patch(_HTTP_POOL, return_value=pool),
            patch(f"{_MSG}.send_text_message", new_callable=AsyncMock) as mock_send_text,
        ):
            await send_interactive_buttons("+1234567890", "Fallback body", [{"id": "1", "title": "OK"}])

        mock_send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_to_text_on_os_error(self):
        """Falls back to text message on OSError."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        pool, session = _mock_session_and_pool()
        session.post.side_effect = OSError("broken pipe")
        with (
            patch(_HTTP_POOL, return_value=pool),
            patch(f"{_MSG}.send_text_message", new_callable=AsyncMock) as mock_send_text,
        ):
            await send_interactive_buttons("+1234567890", "Fallback", [{"id": "1", "title": "OK"}])

        mock_send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_to_text_on_value_error(self):
        """Falls back to text message on ValueError."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        pool, session = _mock_session_and_pool()
        session.post.side_effect = ValueError("bad json")
        with (
            patch(_HTTP_POOL, return_value=pool),
            patch(f"{_MSG}.send_text_message", new_callable=AsyncMock) as mock_send_text,
        ):
            await send_interactive_buttons("+1234567890", "Fallback", [{"id": "1", "title": "OK"}])

        mock_send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_to_text_on_runtime_error(self):
        """Falls back to text message on RuntimeError."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        pool, session = _mock_session_and_pool()
        session.post.side_effect = RuntimeError("event loop issue")
        with (
            patch(_HTTP_POOL, return_value=pool),
            patch(f"{_MSG}.send_text_message", new_callable=AsyncMock) as mock_send_text,
        ):
            await send_interactive_buttons("+1234567890", "Fallback", [{"id": "1", "title": "OK"}])

        mock_send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_records_success_telemetry(self, mock_pool_success):
        """Records sendInteractive telemetry on success."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        mock_record_call = MagicMock()
        pool, _ = mock_pool_success
        with (
            patch(_HTTP_POOL, return_value=pool),
            patch(f"{_MSG}.record_api_call", mock_record_call),
            patch(f"{_MSG}.record_api_latency", MagicMock()),
        ):
            await send_interactive_buttons("+1234567890", "Body", [{"id": "1", "title": "OK"}])

        mock_record_call.assert_called_once_with("whatsapp", "sendInteractive", "success")

    @pytest.mark.asyncio
    async def test_records_error_telemetry_on_api_error(self, mock_pool_error):
        """Records error telemetry when API returns non-200."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        mock_record_call = MagicMock()
        pool, _ = mock_pool_error
        with (
            patch(_HTTP_POOL, return_value=pool),
            patch(f"{_MSG}.record_api_call", mock_record_call),
            patch(f"{_MSG}.record_api_latency", MagicMock()),
            patch(f"{_MSG}.send_text_message", new_callable=AsyncMock),
        ):
            await send_interactive_buttons("+1234567890", "Body", [{"id": "1", "title": "OK"}])

        mock_record_call.assert_called_once_with("whatsapp", "sendInteractive", "error")

    @pytest.mark.asyncio
    async def test_records_error_telemetry_on_exception(self):
        """Records error telemetry when exception occurs."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        mock_record_call = MagicMock()
        pool, session = _mock_session_and_pool()
        session.post.side_effect = ConnectionError("fail")
        with (
            patch(_HTTP_POOL, return_value=pool),
            patch(f"{_MSG}.record_api_call", mock_record_call),
            patch(f"{_MSG}.record_api_latency", MagicMock()),
            patch(f"{_MSG}.send_text_message", new_callable=AsyncMock),
        ):
            await send_interactive_buttons("+1234567890", "Body", [{"id": "1", "title": "OK"}])

        mock_record_call.assert_called_once_with("whatsapp", "sendInteractive", "error")

    @pytest.mark.asyncio
    async def test_uses_whatsapp_session_name(self, mock_pool_success):
        """Pool session is requested with 'whatsapp' name."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        pool, _ = mock_pool_success
        with patch(_HTTP_POOL, return_value=pool):
            await send_interactive_buttons("+1234567890", "Body", [{"id": "1", "title": "OK"}])

        pool.get_session.assert_called_once_with("whatsapp")

    @pytest.mark.asyncio
    async def test_empty_buttons_list(self, mock_pool_success):
        """Sending with empty buttons list still makes the API call."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        pool, session = mock_pool_success
        with patch(_HTTP_POOL, return_value=pool):
            await send_interactive_buttons("+1234567890", "Body", [])

        payload = session.post.call_args.kwargs.get("json") or session.post.call_args[1].get("json")
        assert payload["interactive"]["action"]["buttons"] == []

    @pytest.mark.asyncio
    async def test_sends_correct_headers(self, mock_pool_success):
        """Request includes Bearer token and content type."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        pool, session = mock_pool_success
        with patch(_HTTP_POOL, return_value=pool):
            await send_interactive_buttons("+1234567890", "Body", [{"id": "1", "title": "OK"}])

        headers = session.post.call_args.kwargs.get("headers") or session.post.call_args[1].get("headers")
        assert headers["Authorization"] == "Bearer test-token"

    @pytest.mark.asyncio
    async def test_sends_with_30s_timeout(self, mock_pool_success):
        """Interactive messages use a 30-second timeout."""
        from aragora.server.handlers.social.whatsapp.messaging import send_interactive_buttons

        pool, session = mock_pool_success
        with patch(_HTTP_POOL, return_value=pool):
            await send_interactive_buttons("+1234567890", "Body", [{"id": "1", "title": "OK"}])

        timeout = session.post.call_args.kwargs.get("timeout") or session.post.call_args[1].get("timeout")
        assert timeout == 30


# ===========================================================================
# send_voice_summary Tests
# ===========================================================================


class TestSendVoiceSummary:
    """Tests for send_voice_summary()."""

    @pytest.mark.asyncio
    async def test_sends_voice_when_tts_available(self):
        """Sends voice message when TTS helper is available and synthesis succeeds."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_summary

        mock_result = MagicMock()
        mock_result.audio_bytes = b"fake-audio-data"
        mock_result.format = "mp3"

        mock_helper = MagicMock()
        mock_helper.is_available = True
        mock_helper.synthesize_debate_result = AsyncMock(return_value=mock_result)

        with (
            patch("aragora.server.handlers.social.tts_helper.get_tts_helper", return_value=mock_helper),
            patch(f"{_MSG}.send_voice_message", new_callable=AsyncMock) as mock_send_voice,
        ):
            await send_voice_summary(
                "+1234567890", "AI Safety", "Yes, with guardrails",
                consensus_reached=True, confidence=0.85, rounds_used=3,
            )

        mock_helper.synthesize_debate_result.assert_called_once_with(
            task="AI Safety",
            final_answer="Yes, with guardrails",
            consensus_reached=True,
            confidence=0.85,
            rounds_used=3,
        )
        mock_send_voice.assert_called_once_with("+1234567890", b"fake-audio-data", "mp3")

    @pytest.mark.asyncio
    async def test_no_send_when_tts_unavailable(self):
        """Does not send when TTS helper reports not available."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_summary

        mock_helper = MagicMock()
        mock_helper.is_available = False

        with (
            patch("aragora.server.handlers.social.tts_helper.get_tts_helper", return_value=mock_helper),
            patch(f"{_MSG}.send_voice_message", new_callable=AsyncMock) as mock_send_voice,
        ):
            await send_voice_summary(
                "+1234567890", "Topic", "Answer",
                consensus_reached=True, confidence=0.8, rounds_used=2,
            )

        mock_send_voice.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_send_when_synthesis_returns_none(self):
        """Does not send when TTS synthesis returns None."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_summary

        mock_helper = MagicMock()
        mock_helper.is_available = True
        mock_helper.synthesize_debate_result = AsyncMock(return_value=None)

        with (
            patch("aragora.server.handlers.social.tts_helper.get_tts_helper", return_value=mock_helper),
            patch(f"{_MSG}.send_voice_message", new_callable=AsyncMock) as mock_send_voice,
        ):
            await send_voice_summary(
                "+1234567890", "Topic", "Answer",
                consensus_reached=True, confidence=0.8, rounds_used=2,
            )

        mock_send_voice.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_import_error(self):
        """Handles ImportError when TTS helper module is not available."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_summary

        with patch("aragora.server.handlers.social.tts_helper.get_tts_helper", side_effect=ImportError("no tts")):
            await send_voice_summary(
                "+1234567890", "Topic", "Answer",
                consensus_reached=True, confidence=0.8, rounds_used=2,
            )
        # Should not raise

    @pytest.mark.asyncio
    async def test_handles_os_error(self):
        """Handles OSError during voice summary."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_summary

        mock_helper = MagicMock()
        mock_helper.is_available = True
        mock_helper.synthesize_debate_result = AsyncMock(side_effect=OSError("disk full"))

        with patch("aragora.server.handlers.social.tts_helper.get_tts_helper", return_value=mock_helper):
            await send_voice_summary(
                "+1234567890", "Topic", "Answer",
                consensus_reached=True, confidence=0.8, rounds_used=2,
            )

    @pytest.mark.asyncio
    async def test_handles_connection_error(self):
        """Handles ConnectionError during voice summary."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_summary

        mock_helper = MagicMock()
        mock_helper.is_available = True
        mock_helper.synthesize_debate_result = AsyncMock(side_effect=ConnectionError("network down"))

        with patch("aragora.server.handlers.social.tts_helper.get_tts_helper", return_value=mock_helper):
            await send_voice_summary(
                "+1234567890", "Topic", "Answer",
                consensus_reached=True, confidence=0.8, rounds_used=2,
            )

    @pytest.mark.asyncio
    async def test_handles_timeout_error(self):
        """Handles TimeoutError during voice summary."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_summary

        mock_helper = MagicMock()
        mock_helper.is_available = True
        mock_helper.synthesize_debate_result = AsyncMock(side_effect=TimeoutError("slow tts"))

        with patch("aragora.server.handlers.social.tts_helper.get_tts_helper", return_value=mock_helper):
            await send_voice_summary(
                "+1234567890", "Topic", "Answer",
                consensus_reached=True, confidence=0.8, rounds_used=2,
            )

    @pytest.mark.asyncio
    async def test_handles_value_error(self):
        """Handles ValueError during voice summary."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_summary

        mock_helper = MagicMock()
        mock_helper.is_available = True
        mock_helper.synthesize_debate_result = AsyncMock(side_effect=ValueError("bad data"))

        with patch("aragora.server.handlers.social.tts_helper.get_tts_helper", return_value=mock_helper):
            await send_voice_summary(
                "+1234567890", "Topic", "Answer",
                consensus_reached=True, confidence=0.8, rounds_used=2,
            )

    @pytest.mark.asyncio
    async def test_handles_runtime_error(self):
        """Handles RuntimeError during voice summary."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_summary

        mock_helper = MagicMock()
        mock_helper.is_available = True
        mock_helper.synthesize_debate_result = AsyncMock(side_effect=RuntimeError("tts broken"))

        with patch("aragora.server.handlers.social.tts_helper.get_tts_helper", return_value=mock_helper):
            await send_voice_summary(
                "+1234567890", "Topic", "Answer",
                consensus_reached=True, confidence=0.8, rounds_used=2,
            )

    @pytest.mark.asyncio
    async def test_passes_none_final_answer(self):
        """None final_answer is passed through to TTS helper."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_summary

        mock_result = MagicMock()
        mock_result.audio_bytes = b"audio"
        mock_result.format = "ogg"

        mock_helper = MagicMock()
        mock_helper.is_available = True
        mock_helper.synthesize_debate_result = AsyncMock(return_value=mock_result)

        with (
            patch("aragora.server.handlers.social.tts_helper.get_tts_helper", return_value=mock_helper),
            patch(f"{_MSG}.send_voice_message", new_callable=AsyncMock),
        ):
            await send_voice_summary(
                "+1234567890", "Topic", None,
                consensus_reached=False, confidence=0.3, rounds_used=5,
            )

        call_kwargs = mock_helper.synthesize_debate_result.call_args.kwargs
        assert call_kwargs["final_answer"] is None
        assert call_kwargs["consensus_reached"] is False


# ===========================================================================
# send_voice_message Tests
# ===========================================================================


class TestSendVoiceMessage:
    """Tests for send_voice_message()."""

    def _make_upload_and_send_pool(
        self,
        upload_status: int = 200,
        upload_json: dict | None = None,
        send_status: int = 200,
        send_json: dict | None = None,
    ):
        """Create a pool that handles both upload and send requests."""
        upload_resp = _mock_response(upload_status, upload_json or {"id": "media-12345"})
        send_resp = _mock_response(send_status, send_json or {"messages": [{"id": "msg-1"}]})

        mock_session = AsyncMock()
        mock_session.post = AsyncMock(side_effect=[upload_resp, send_resp])
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session
        return mock_pool, mock_session

    @pytest.mark.asyncio
    async def test_uploads_then_sends_audio(self):
        """Completes the two-step process: upload media then send message."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool()
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio-data", "mp3")

        assert session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_upload_url_correct(self):
        """First request goes to the media upload endpoint."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool()
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio-data", "mp3")

        upload_call = session.post.call_args_list[0]
        assert "123456789/media" in upload_call[0][0]

    @pytest.mark.asyncio
    async def test_send_url_correct(self):
        """Second request goes to the messages endpoint."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool()
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio-data", "mp3")

        send_call = session.post.call_args_list[1]
        assert "123456789/messages" in send_call[0][0]

    @pytest.mark.asyncio
    async def test_upload_includes_file_data(self):
        """Upload request includes multipart file data."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool()
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"my-audio", "mp3")

        upload_kwargs = session.post.call_args_list[0].kwargs
        files = upload_kwargs.get("files") or session.post.call_args_list[0][1].get("files")
        assert "file" in files
        filename, audio_bytes, mime = files["file"]
        assert filename == "voice.mp3"
        assert audio_bytes == b"my-audio"
        assert mime == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_upload_includes_messaging_product(self):
        """Upload request includes messaging_product field."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool()
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio", "mp3")

        upload_kwargs = session.post.call_args_list[0].kwargs
        data = upload_kwargs.get("data") or session.post.call_args_list[0][1].get("data")
        assert data["messaging_product"] == "whatsapp"

    @pytest.mark.asyncio
    async def test_send_message_includes_media_id(self):
        """Send request includes the media ID from the upload response."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool(upload_json={"id": "media-xyz"})
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio", "mp3")

        send_kwargs = session.post.call_args_list[1].kwargs
        payload = send_kwargs.get("json") or session.post.call_args_list[1][1].get("json")
        assert payload["audio"]["id"] == "media-xyz"
        assert payload["type"] == "audio"
        assert payload["to"] == "+1234567890"

    @pytest.mark.asyncio
    async def test_mp3_mime_type(self):
        """MP3 format uses audio/mpeg MIME type."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool()
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio", "mp3")

        files = session.post.call_args_list[0].kwargs.get("files")
        assert files["file"][2] == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_ogg_mime_type(self):
        """OGG format uses audio/ogg MIME type."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool()
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio", "ogg")

        files = session.post.call_args_list[0].kwargs.get("files")
        assert files["file"][0] == "voice.ogg"
        assert files["file"][2] == "audio/ogg"

    @pytest.mark.asyncio
    async def test_wav_mime_type(self):
        """WAV format uses audio/wav MIME type."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool()
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio", "wav")

        files = session.post.call_args_list[0].kwargs.get("files")
        assert files["file"][2] == "audio/wav"

    @pytest.mark.asyncio
    async def test_m4a_mime_type(self):
        """M4A format uses audio/mp4 MIME type."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool()
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio", "m4a")

        files = session.post.call_args_list[0].kwargs.get("files")
        assert files["file"][2] == "audio/mp4"

    @pytest.mark.asyncio
    async def test_unknown_format_defaults_to_mpeg(self):
        """Unknown audio format defaults to audio/mpeg MIME type."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool()
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio", "flac")

        files = session.post.call_args_list[0].kwargs.get("files")
        assert files["file"][2] == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_returns_early_when_token_missing(self, monkeypatch):
        """Returns early when access token is not configured."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        monkeypatch.setattr(f"{_MSG}.WHATSAPP_ACCESS_TOKEN", "")
        with patch(_HTTP_POOL) as mock_get_pool:
            await send_voice_message("+1234567890", b"audio", "mp3")
            mock_get_pool.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_early_when_token_none(self, monkeypatch):
        """Returns early when access token is None."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        monkeypatch.setattr(f"{_MSG}.WHATSAPP_ACCESS_TOKEN", None)
        with patch(_HTTP_POOL) as mock_get_pool:
            await send_voice_message("+1234567890", b"audio", "mp3")
            mock_get_pool.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_early_when_phone_id_missing(self, monkeypatch):
        """Returns early when phone number ID is not configured."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        monkeypatch.setattr(f"{_MSG}.WHATSAPP_PHONE_NUMBER_ID", "")
        with patch(_HTTP_POOL) as mock_get_pool:
            await send_voice_message("+1234567890", b"audio", "mp3")
            mock_get_pool.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_early_when_phone_id_none(self, monkeypatch):
        """Returns early when phone number ID is None."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        monkeypatch.setattr(f"{_MSG}.WHATSAPP_PHONE_NUMBER_ID", None)
        with patch(_HTTP_POOL) as mock_get_pool:
            await send_voice_message("+1234567890", b"audio", "mp3")
            mock_get_pool.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_early_on_upload_failure(self):
        """Returns without sending when upload returns non-200."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        upload_resp = _mock_response(400, {"error": "Bad request"})
        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=upload_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session

        with patch(_HTTP_POOL, return_value=mock_pool):
            await send_voice_message("+1234567890", b"audio", "mp3")

        # Only upload call, no send call
        assert mock_session.post.call_count == 1

    @pytest.mark.asyncio
    async def test_returns_early_when_no_media_id(self):
        """Returns without sending when upload response lacks media ID."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        upload_resp = _mock_response(200, {})  # No "id" field
        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=upload_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session

        with patch(_HTTP_POOL, return_value=mock_pool):
            await send_voice_message("+1234567890", b"audio", "mp3")

        assert mock_session.post.call_count == 1

    @pytest.mark.asyncio
    async def test_logs_warning_on_upload_failure(self):
        """Logs a warning when the upload fails."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        upload_resp = _mock_response(500, {"error": "server error"})
        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=upload_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session

        with (
            patch(_HTTP_POOL, return_value=mock_pool),
            patch(f"{_MSG}.logger") as mock_logger,
        ):
            await send_voice_message("+1234567890", b"audio", "mp3")

        mock_logger.warning.assert_called()
        assert "upload failed" in mock_logger.warning.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_logs_warning_on_no_media_id(self):
        """Logs a warning when no media ID is returned."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        upload_resp = _mock_response(200, {})
        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=upload_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session

        with (
            patch(_HTTP_POOL, return_value=mock_pool),
            patch(f"{_MSG}.logger") as mock_logger,
        ):
            await send_voice_message("+1234567890", b"audio", "mp3")

        mock_logger.warning.assert_called()
        assert "No media ID" in mock_logger.warning.call_args[0][0]

    @pytest.mark.asyncio
    async def test_logs_warning_on_send_failure(self):
        """Logs a warning when the send message step fails."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool(send_status=500, send_json={"error": "fail"})
        with (
            patch(_HTTP_POOL, return_value=pool),
            patch(f"{_MSG}.logger") as mock_logger,
        ):
            await send_voice_message("+1234567890", b"audio", "mp3")

        # Check that warning about send failure was logged
        warning_calls = [c[0][0] for c in mock_logger.warning.call_args_list]
        assert any("audio send failed" in w.lower() for w in warning_calls)

    @pytest.mark.asyncio
    async def test_logs_info_on_success(self):
        """Logs info message on successful send."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool()
        with (
            patch(_HTTP_POOL, return_value=pool),
            patch(f"{_MSG}.logger") as mock_logger,
        ):
            await send_voice_message("+1234567890", b"audio", "mp3")

        mock_logger.info.assert_called()
        assert "voice message sent" in mock_logger.info.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_handles_os_error(self):
        """Catches OSError without raising."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = _mock_session_and_pool()
        session.post.side_effect = OSError("broken pipe")
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio", "mp3")

    @pytest.mark.asyncio
    async def test_handles_connection_error(self):
        """Catches ConnectionError without raising."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = _mock_session_and_pool()
        session.post.side_effect = ConnectionError("refused")
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio", "mp3")

    @pytest.mark.asyncio
    async def test_handles_timeout_error(self):
        """Catches TimeoutError without raising."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = _mock_session_and_pool()
        session.post.side_effect = TimeoutError("slow")
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio", "mp3")

    @pytest.mark.asyncio
    async def test_handles_value_error(self):
        """Catches ValueError without raising."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = _mock_session_and_pool()
        session.post.side_effect = ValueError("json decode fail")
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio", "mp3")

    @pytest.mark.asyncio
    async def test_handles_runtime_error(self):
        """Catches RuntimeError without raising."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = _mock_session_and_pool()
        session.post.side_effect = RuntimeError("loop closed")
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio", "mp3")

    @pytest.mark.asyncio
    async def test_upload_uses_60s_timeout(self):
        """Upload uses a 60-second timeout (larger files)."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool()
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio", "mp3")

        upload_kwargs = session.post.call_args_list[0].kwargs
        assert upload_kwargs.get("timeout") == 60

    @pytest.mark.asyncio
    async def test_send_uses_30s_timeout(self):
        """Send message uses a 30-second timeout."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool()
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio", "mp3")

        send_kwargs = session.post.call_args_list[1].kwargs
        assert send_kwargs.get("timeout") == 30

    @pytest.mark.asyncio
    async def test_upload_uses_bearer_token(self):
        """Upload request includes Bearer authorization."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool()
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio", "mp3")

        upload_headers = session.post.call_args_list[0].kwargs.get("headers")
        assert upload_headers["Authorization"] == "Bearer test-token"

    @pytest.mark.asyncio
    async def test_send_uses_bearer_token_and_json_content(self):
        """Send request includes Bearer auth and JSON content type."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool()
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio", "mp3")

        send_headers = session.post.call_args_list[1].kwargs.get("headers")
        assert send_headers["Authorization"] == "Bearer test-token"
        assert send_headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_uses_whatsapp_media_session_name(self):
        """Pool session is requested with 'whatsapp_media' as the session name."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool()
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio", "mp3")

        pool.get_session.assert_called_once_with("whatsapp_media")

    @pytest.mark.asyncio
    async def test_default_format_is_mp3(self):
        """Default audio_format parameter is mp3."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool()
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio")

        files = session.post.call_args_list[0].kwargs.get("files")
        assert files["file"][0] == "voice.mp3"
        assert files["file"][2] == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_upload_data_includes_mime_type(self):
        """Upload data field includes the correct MIME type."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool()
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+1234567890", b"audio", "ogg")

        upload_kwargs = session.post.call_args_list[0].kwargs
        data = upload_kwargs.get("data")
        assert data["type"] == "audio/ogg"

    @pytest.mark.asyncio
    async def test_send_payload_structure(self):
        """Send payload has the correct structure for audio messages."""
        from aragora.server.handlers.social.whatsapp.messaging import send_voice_message

        pool, session = self._make_upload_and_send_pool(upload_json={"id": "media-abc"})
        with patch(_HTTP_POOL, return_value=pool):
            await send_voice_message("+9876543210", b"audio", "mp3")

        payload = session.post.call_args_list[1].kwargs.get("json")
        assert payload["messaging_product"] == "whatsapp"
        assert payload["recipient_type"] == "individual"
        assert payload["to"] == "+9876543210"
        assert payload["type"] == "audio"
        assert payload["audio"]["id"] == "media-abc"
