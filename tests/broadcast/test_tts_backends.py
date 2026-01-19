"""Tests for TTS backend abstraction layer."""

import os
import pytest
from unittest.mock import patch

from aragora.broadcast.tts_backends import (
    _parse_csv,
    _parse_json_env,
    _normalize_backend_name,
    TTSConfig,
)


class TestParseCSV:
    """Test CSV parsing utility."""

    def test_parse_single_value(self):
        """Test parsing single value."""
        result = _parse_csv("value")
        assert result == ["value"]

    def test_parse_multiple_values(self):
        """Test parsing multiple comma-separated values."""
        result = _parse_csv("a,b,c")
        assert result == ["a", "b", "c"]

    def test_parse_with_whitespace(self):
        """Test parsing handles whitespace."""
        result = _parse_csv("a , b , c")
        assert result == ["a", "b", "c"]

    def test_parse_empty_string(self):
        """Test parsing empty string returns None."""
        result = _parse_csv("")
        assert result is None

    def test_parse_none(self):
        """Test parsing None returns None."""
        result = _parse_csv(None)
        assert result is None

    def test_parse_filters_empty_items(self):
        """Test parsing filters out empty items."""
        result = _parse_csv("a,,b,  ,c")
        assert result == ["a", "b", "c"]


class TestParseJsonEnv:
    """Test JSON environment variable parsing."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON from env."""
        with patch.dict(os.environ, {"TEST_JSON": '{"key": "value"}'}):
            result = _parse_json_env("TEST_JSON")
            assert result == {"key": "value"}

    def test_parse_array_json(self):
        """Test parsing JSON array."""
        with patch.dict(os.environ, {"TEST_JSON": '["a", "b", "c"]'}):
            result = _parse_json_env("TEST_JSON")
            assert result == ["a", "b", "c"]

    def test_parse_missing_env(self):
        """Test parsing missing env var returns None."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove if exists
            os.environ.pop("MISSING_VAR", None)
            result = _parse_json_env("MISSING_VAR")
            assert result is None

    def test_parse_invalid_json(self, caplog):
        """Test parsing invalid JSON returns None and logs warning."""
        with patch.dict(os.environ, {"BAD_JSON": "not valid json"}):
            result = _parse_json_env("BAD_JSON")
            assert result is None
            # Warning should be logged
            assert any("Invalid JSON" in r.message for r in caplog.records)

    def test_parse_empty_env(self):
        """Test parsing empty env var returns None."""
        with patch.dict(os.environ, {"EMPTY": ""}):
            result = _parse_json_env("EMPTY")
            assert result is None


class TestNormalizeBackendName:
    """Test backend name normalization."""

    def test_normalize_eleven(self):
        """Test 'eleven' normalizes to 'elevenlabs'."""
        assert _normalize_backend_name("eleven") == "elevenlabs"

    def test_normalize_11labs(self):
        """Test '11labs' normalizes to 'elevenlabs'."""
        assert _normalize_backend_name("11labs") == "elevenlabs"

    def test_normalize_edge(self):
        """Test 'edge' normalizes to 'edge-tts'."""
        assert _normalize_backend_name("edge") == "edge-tts"

    def test_normalize_aws(self):
        """Test 'aws' normalizes to 'polly'."""
        assert _normalize_backend_name("aws") == "polly"

    def test_normalize_aws_polly(self):
        """Test 'aws-polly' normalizes to 'polly'."""
        assert _normalize_backend_name("aws-polly") == "polly"

    def test_normalize_amazon_polly(self):
        """Test 'amazon-polly' normalizes to 'polly'."""
        assert _normalize_backend_name("amazon-polly") == "polly"

    def test_normalize_coqui(self):
        """Test 'coqui' normalizes to 'xtts'."""
        assert _normalize_backend_name("coqui") == "xtts"

    def test_normalize_xtts_v2(self):
        """Test 'xtts-v2' normalizes to 'xtts'."""
        assert _normalize_backend_name("xtts-v2") == "xtts"

    def test_normalize_fallback(self):
        """Test 'fallback' normalizes to 'pyttsx3'."""
        assert _normalize_backend_name("fallback") == "pyttsx3"

    def test_normalize_passthrough(self):
        """Test unknown names pass through unchanged."""
        assert _normalize_backend_name("elevenlabs") == "elevenlabs"
        assert _normalize_backend_name("polly") == "polly"
        assert _normalize_backend_name("unknown") == "unknown"


class TestTTSConfig:
    """Test TTSConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TTSConfig()
        assert "elevenlabs" in config.backend_priority
        assert "polly" in config.backend_priority
        assert "xtts" in config.backend_priority
        assert config.elevenlabs_api_key is None
        assert config.elevenlabs_model == "eleven_multilingual_v2"
        assert config.xtts_device == "auto"
        assert config.xtts_language == "en"

    def test_custom_backend_priority(self):
        """Test setting custom backend priority."""
        config = TTSConfig(backend_priority=["polly", "xtts"])
        assert config.backend_priority == ["polly", "xtts"]

    def test_elevenlabs_settings(self):
        """Test ElevenLabs configuration."""
        config = TTSConfig(
            elevenlabs_api_key="sk-test",
            elevenlabs_model="eleven_monolingual_v1",
            elevenlabs_voice_map={"narrator": "voice-id-123"},
        )
        assert config.elevenlabs_api_key == "sk-test"
        assert config.elevenlabs_model == "eleven_monolingual_v1"
        assert config.elevenlabs_voice_map["narrator"] == "voice-id-123"

    def test_xtts_settings(self):
        """Test XTTS configuration."""
        config = TTSConfig(
            xtts_model_path="/models/xtts",
            xtts_device="cuda",
            xtts_language="es",
            xtts_speaker_wav="/voices/speaker.wav",
        )
        assert config.xtts_model_path == "/models/xtts"
        assert config.xtts_device == "cuda"
        assert config.xtts_language == "es"
        assert config.xtts_speaker_wav == "/voices/speaker.wav"

    def test_voice_maps_default_to_empty_dict(self):
        """Test that voice maps default to empty dicts."""
        config = TTSConfig()
        assert config.elevenlabs_voice_map == {}
        assert config.xtts_speaker_wav_map == {}

    def test_voice_maps_are_independent(self):
        """Test that voice maps are independent instances."""
        config1 = TTSConfig()
        config2 = TTSConfig()
        config1.elevenlabs_voice_map["test"] = "value"
        assert "test" not in config2.elevenlabs_voice_map
