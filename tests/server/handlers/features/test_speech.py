"""Tests for the Speech handler 501 stub."""

from aragora.server.handlers.features.speech import SpeechHandler


class TestSpeechHandlerStub:
    """Verify all speech routes return 501 after module removal."""

    def setup_method(self):
        self.handler = SpeechHandler()

    def test_can_handle_transcribe(self):
        assert self.handler.can_handle("/api/v1/speech/transcribe")

    def test_can_handle_transcribe_url(self):
        assert self.handler.can_handle("/api/v1/speech/transcribe-url")

    def test_can_handle_providers(self):
        assert self.handler.can_handle("/api/v1/speech/providers")

    def test_cannot_handle_unknown(self):
        assert not self.handler.can_handle("/api/v1/other")

    def test_get_returns_501(self):
        result = self.handler.handle("/api/v1/speech/providers", {}, None)
        assert result is not None
        assert result.status_code == 501

    def test_post_transcribe_returns_501(self):
        result = self.handler.handle_post("/api/v1/speech/transcribe", {}, None)
        assert result is not None
        assert result.status_code == 501

    def test_post_transcribe_url_returns_501(self):
        result = self.handler.handle_post("/api/v1/speech/transcribe-url", {}, None)
        assert result is not None
        assert result.status_code == 501
