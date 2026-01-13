"""Tests for TTS backends module.

Comprehensive tests for all TTS backend implementations:
- Configuration parsing and environment variables
- ElevenLabs, Polly, XTTS, Edge-TTS, pyttsx3 backends
- Fallback chain logic
- Factory functions
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.broadcast.tts_backends import (
    TTSConfig,
    TTSBackend,
    ElevenLabsBackend,
    XTTSBackend,
    EdgeTTSBackend,
    PollyBackend,
    Pyttsx3Backend,
    FallbackTTSBackend,
    get_tts_backend,
    get_fallback_backend,
    BACKEND_REGISTRY,
    ELEVENLABS_VOICES,
    EDGE_TTS_VOICES,
    POLLY_VOICES,
    XTTS_SPEAKERS,
    _parse_csv,
    _parse_json_env,
    _normalize_backend_name,
)


# =============================================================================
# Helper Functions Tests
# =============================================================================


class TestParseCsv:
    """Tests for _parse_csv helper."""

    def test_parse_csv_simple(self):
        """Parse simple CSV string."""
        result = _parse_csv("elevenlabs,polly,xtts")
        assert result == ["elevenlabs", "polly", "xtts"]

    def test_parse_csv_with_whitespace(self):
        """Strips whitespace from items."""
        result = _parse_csv(" elevenlabs , polly , xtts ")
        assert result == ["elevenlabs", "polly", "xtts"]

    def test_parse_csv_empty_string(self):
        """Empty string returns None."""
        result = _parse_csv("")
        assert result is None

    def test_parse_csv_none(self):
        """None returns None."""
        result = _parse_csv(None)
        assert result is None

    def test_parse_csv_filters_empty_items(self):
        """Empty items are filtered out."""
        result = _parse_csv("elevenlabs,,polly,")
        assert result == ["elevenlabs", "polly"]


class TestParseJsonEnv:
    """Tests for _parse_json_env helper."""

    def test_parse_valid_json(self):
        """Valid JSON is parsed."""
        with patch.dict(os.environ, {"TEST_JSON": '{"key": "value"}'}):
            result = _parse_json_env("TEST_JSON")
            assert result == {"key": "value"}

    def test_parse_json_list(self):
        """JSON list is parsed."""
        with patch.dict(os.environ, {"TEST_JSON": '["a", "b", "c"]'}):
            result = _parse_json_env("TEST_JSON")
            assert result == ["a", "b", "c"]

    def test_parse_invalid_json(self):
        """Invalid JSON returns None."""
        with patch.dict(os.environ, {"TEST_JSON": "not valid json"}):
            result = _parse_json_env("TEST_JSON")
            assert result is None

    def test_parse_missing_env(self):
        """Missing env var returns None."""
        with patch.dict(os.environ, {}, clear=True):
            result = _parse_json_env("NONEXISTENT_VAR")
            assert result is None


class TestNormalizeBackendName:
    """Tests for _normalize_backend_name helper."""

    def test_known_aliases(self):
        """Known aliases are normalized."""
        assert _normalize_backend_name("eleven") == "elevenlabs"
        assert _normalize_backend_name("11labs") == "elevenlabs"
        assert _normalize_backend_name("edge") == "edge-tts"
        assert _normalize_backend_name("aws") == "polly"
        assert _normalize_backend_name("aws-polly") == "polly"
        assert _normalize_backend_name("amazon-polly") == "polly"
        assert _normalize_backend_name("coqui") == "xtts"
        assert _normalize_backend_name("xtts-v2") == "xtts"
        assert _normalize_backend_name("fallback") == "pyttsx3"

    def test_unknown_names_unchanged(self):
        """Unknown names pass through unchanged."""
        assert _normalize_backend_name("elevenlabs") == "elevenlabs"
        assert _normalize_backend_name("polly") == "polly"
        assert _normalize_backend_name("custom") == "custom"


# =============================================================================
# TTSConfig Tests
# =============================================================================


class TestTTSConfig:
    """Tests for TTSConfig dataclass."""

    def test_default_values(self):
        """Config has correct default values."""
        config = TTSConfig()
        assert config.backend_priority == ["elevenlabs", "polly", "xtts", "edge-tts", "pyttsx3"]
        assert config.elevenlabs_api_key is None
        assert config.elevenlabs_model == "eleven_multilingual_v2"
        assert config.xtts_device == "auto"
        assert config.xtts_language == "en"
        assert config.polly_engine == "neural"
        assert config.enable_cache is True

    def test_from_env_backend_priority(self):
        """from_env parses ARAGORA_TTS_ORDER."""
        with patch.dict(os.environ, {"ARAGORA_TTS_ORDER": "edge-tts,pyttsx3"}, clear=True):
            config = TTSConfig.from_env()
            assert config.backend_priority == ["edge-tts", "pyttsx3"]

    def test_from_env_backend_priority_aliases(self):
        """from_env normalizes backend aliases."""
        with patch.dict(os.environ, {"ARAGORA_TTS_ORDER": "eleven,aws,edge"}, clear=True):
            config = TTSConfig.from_env()
            assert config.backend_priority == ["elevenlabs", "polly", "edge-tts"]

    def test_from_env_elevenlabs_api_key(self):
        """from_env reads ElevenLabs API key."""
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}, clear=True):
            config = TTSConfig.from_env()
            assert config.elevenlabs_api_key == "test-key"

    def test_from_env_aragora_prefix_takes_precedence(self):
        """ARAGORA_ prefixed vars take precedence."""
        with patch.dict(
            os.environ,
            {"ARAGORA_ELEVENLABS_API_KEY": "aragora-key", "ELEVENLABS_API_KEY": "other-key"},
            clear=True,
        ):
            config = TTSConfig.from_env()
            assert config.elevenlabs_api_key == "aragora-key"

    def test_from_env_polly_region_fallback(self):
        """Polly region falls back to AWS_REGION."""
        with patch.dict(os.environ, {"AWS_REGION": "us-west-2"}, clear=True):
            config = TTSConfig.from_env()
            assert config.polly_region == "us-west-2"

    def test_from_env_voice_map_json(self):
        """from_env parses voice map JSON."""
        voice_map = {"narrator": "custom-voice-id"}
        with patch.dict(
            os.environ,
            {"ARAGORA_ELEVENLABS_VOICE_MAP": '{"narrator": "custom-voice-id"}'},
            clear=True,
        ):
            config = TTSConfig.from_env()
            assert config.elevenlabs_voice_map == voice_map

    def test_from_env_invalid_voice_map_defaults_empty(self):
        """Invalid voice map JSON defaults to empty dict."""
        with patch.dict(os.environ, {"ARAGORA_ELEVENLABS_VOICE_MAP": "not json"}, clear=True):
            config = TTSConfig.from_env()
            assert config.elevenlabs_voice_map == {}

    def test_from_env_polly_lexicons(self):
        """from_env parses Polly lexicons CSV."""
        with patch.dict(os.environ, {"ARAGORA_POLLY_LEXICONS": "lex1,lex2"}, clear=True):
            config = TTSConfig.from_env()
            assert config.polly_lexicons == ["lex1", "lex2"]


# =============================================================================
# ElevenLabs Backend Tests
# =============================================================================


class TestElevenLabsBackend:
    """Tests for ElevenLabsBackend."""

    @pytest.fixture
    def config_with_key(self):
        """Config with ElevenLabs API key."""
        return TTSConfig(elevenlabs_api_key="test-api-key")

    @pytest.fixture
    def config_without_key(self):
        """Config without API key."""
        return TTSConfig(elevenlabs_api_key=None)

    def test_is_available_with_key_and_module(self, config_with_key):
        """Available when API key set and module installed."""
        backend = ElevenLabsBackend(config_with_key)
        with patch.dict("sys.modules", {"elevenlabs": MagicMock()}):
            assert backend.is_available() is True

    def test_is_available_without_key(self, config_without_key):
        """Not available without API key."""
        backend = ElevenLabsBackend(config_without_key)
        assert backend.is_available() is False

    def test_is_available_without_module(self, config_with_key):
        """Not available without elevenlabs module."""
        backend = ElevenLabsBackend(config_with_key)
        with patch.dict("sys.modules", {"elevenlabs": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                assert backend.is_available() is False

    def test_get_voice_id_from_config_map(self):
        """Voice ID from config map takes precedence."""
        config = TTSConfig(elevenlabs_api_key="key", elevenlabs_voice_map={"narrator": "custom-id"})
        backend = ElevenLabsBackend(config)
        assert backend.get_voice_id("narrator") == "custom-id"

    def test_get_voice_id_from_default_voices(self, config_with_key):
        """Voice ID from default ELEVENLABS_VOICES mapping."""
        backend = ElevenLabsBackend(config_with_key)
        assert backend.get_voice_id("narrator") == ELEVENLABS_VOICES["narrator"]

    def test_get_voice_id_default_fallback(self, config_with_key):
        """Unknown voice falls back to default."""
        backend = ElevenLabsBackend(config_with_key)
        result = backend.get_voice_id("unknown-voice")
        # Falls back to speaker itself or default
        assert result in [ELEVENLABS_VOICES["default"], "unknown-voice"]

    def test_get_voice_id_uses_config_default(self):
        """Uses config default voice ID."""
        config = TTSConfig(elevenlabs_api_key="key", elevenlabs_default_voice_id="config-default")
        backend = ElevenLabsBackend(config)
        # Unknown voice should use config default
        assert backend.get_voice_id("unknown") == "config-default"

    @pytest.mark.asyncio
    async def test_synthesize_not_available_returns_none(self, config_without_key):
        """Synthesize returns None when not available."""
        backend = ElevenLabsBackend(config_without_key)
        result = await backend.synthesize("Hello world")
        assert result is None

    @pytest.mark.asyncio
    async def test_synthesize_success(self, config_with_key, tmp_path):
        """Successful synthesis returns output path."""
        backend = ElevenLabsBackend(config_with_key)

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = [b"audio data"]

        with patch.dict("sys.modules", {"elevenlabs": MagicMock()}):
            with patch.object(backend, "_get_client", return_value=mock_client):
                with patch.object(backend, "is_available", return_value=True):
                    output = tmp_path / "output.mp3"
                    result = await backend.synthesize("Hello", output_path=output)

                    assert result == output
                    mock_client.text_to_speech.convert.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_api_error_returns_none(self, config_with_key):
        """API error returns None."""
        backend = ElevenLabsBackend(config_with_key)

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.side_effect = Exception("API Error")

        with patch.dict("sys.modules", {"elevenlabs": MagicMock()}):
            with patch.object(backend, "_get_client", return_value=mock_client):
                with patch.object(backend, "is_available", return_value=True):
                    result = await backend.synthesize("Hello")
                    assert result is None


# =============================================================================
# Polly Backend Tests
# =============================================================================


class TestPollyBackend:
    """Tests for PollyBackend."""

    @pytest.fixture
    def config(self):
        """Config with Polly settings."""
        return TTSConfig(polly_region="us-east-1")

    def test_is_available_with_boto3_and_creds(self, config):
        """Available when boto3 installed and credentials present."""
        backend = PollyBackend(config)

        mock_session = MagicMock()
        mock_session.get_credentials.return_value = MagicMock()

        with patch.dict("sys.modules", {"boto3": MagicMock()}):
            with patch("boto3.Session", return_value=mock_session):
                assert backend.is_available() is True

    def test_is_available_without_boto3(self, config):
        """Not available without boto3."""
        backend = PollyBackend(config)
        with patch.dict("sys.modules", {"boto3": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                assert backend.is_available() is False

    def test_is_available_without_credentials(self, config):
        """Not available without AWS credentials."""
        backend = PollyBackend(config)

        mock_session = MagicMock()
        mock_session.get_credentials.return_value = None

        with patch.dict("sys.modules", {"boto3": MagicMock()}):
            with patch("boto3.Session", return_value=mock_session):
                assert backend.is_available() is False

    def test_get_voice_id_from_config_map(self):
        """Voice ID from config map takes precedence."""
        config = TTSConfig(polly_voice_map={"narrator": "CustomVoice"})
        backend = PollyBackend(config)
        assert backend.get_voice_id("narrator") == "CustomVoice"

    def test_get_voice_id_from_default_voices(self):
        """Voice ID from default POLLY_VOICES mapping."""
        backend = PollyBackend(TTSConfig())
        assert backend.get_voice_id("narrator") == POLLY_VOICES["narrator"]

    def test_get_voice_id_default_fallback(self):
        """Unknown voice falls back to default."""
        config = TTSConfig(polly_default_voice_id="DefaultVoice")
        backend = PollyBackend(config)
        assert backend.get_voice_id("unknown") == "DefaultVoice"

    @pytest.mark.asyncio
    async def test_synthesize_success(self, config, tmp_path):
        """Successful synthesis with Polly."""
        backend = PollyBackend(config)

        mock_stream = MagicMock()
        mock_stream.read.return_value = b"audio data"

        mock_client = MagicMock()
        mock_client.synthesize_speech.return_value = {"AudioStream": mock_stream}

        with patch.object(backend, "is_available", return_value=True):
            with patch.object(backend, "_get_client", return_value=mock_client):
                output = tmp_path / "output.mp3"
                result = await backend.synthesize("Hello", output_path=output)

                assert result == output
                mock_client.synthesize_speech.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_includes_lexicons(self, tmp_path):
        """Lexicons included in Polly request."""
        config = TTSConfig(polly_region="us-east-1", polly_lexicons=["lex1", "lex2"])
        backend = PollyBackend(config)

        mock_stream = MagicMock()
        mock_stream.read.return_value = b"audio"
        mock_client = MagicMock()
        mock_client.synthesize_speech.return_value = {"AudioStream": mock_stream}

        with patch.object(backend, "is_available", return_value=True):
            with patch.object(backend, "_get_client", return_value=mock_client):
                output = tmp_path / "output.mp3"
                await backend.synthesize("Hello", output_path=output)

                call_args = mock_client.synthesize_speech.call_args
                assert call_args[1]["LexiconNames"] == ["lex1", "lex2"]

    @pytest.mark.asyncio
    async def test_synthesize_not_available_returns_none(self):
        """Returns None when not available."""
        backend = PollyBackend(TTSConfig())
        with patch.object(backend, "is_available", return_value=False):
            result = await backend.synthesize("Hello")
            assert result is None


# =============================================================================
# XTTS Backend Tests
# =============================================================================


class TestXTTSBackend:
    """Tests for XTTSBackend."""

    @pytest.fixture
    def config(self):
        """Basic XTTS config."""
        return TTSConfig(xtts_device="cpu")

    def test_is_available_with_dependencies(self, config):
        """Available when torch and TTS installed."""
        backend = XTTSBackend(config)
        with patch.dict("sys.modules", {"torch": MagicMock(), "TTS.api": MagicMock()}):
            with patch("builtins.__import__", return_value=MagicMock()):
                # Mock the actual import check
                assert backend.is_available() is True

    def test_is_available_without_torch(self, config):
        """Not available without torch."""
        backend = XTTSBackend(config)

        def mock_import(name, *args, **kwargs):
            if "torch" in name:
                raise ImportError("No torch")
            return MagicMock()

        with patch("builtins.__import__", side_effect=mock_import):
            assert backend.is_available() is False

    def test_get_device_auto_cuda(self, config):
        """Auto device selects CUDA when available."""
        config.xtts_device = "auto"
        backend = XTTSBackend(config)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            device = backend._get_device()
            assert device == "cuda"

    def test_get_device_auto_cpu(self, config):
        """Auto device selects CPU when CUDA unavailable."""
        config.xtts_device = "auto"
        backend = XTTSBackend(config)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            device = backend._get_device()
            assert device == "cpu"

    def test_get_device_explicit(self):
        """Explicit device setting is used."""
        config = TTSConfig(xtts_device="cuda:1")
        backend = XTTSBackend(config)
        device = backend._get_device()
        assert device == "cuda:1"

    def test_get_voice_id_existing_path(self, config, tmp_path):
        """Existing file path returned as-is."""
        backend = XTTSBackend(config)
        wav_file = tmp_path / "voice.wav"
        wav_file.touch()

        result = backend.get_voice_id(str(wav_file))
        assert result == str(wav_file)

    def test_get_voice_id_from_config_map(self, config, tmp_path):
        """Voice from speaker wav map."""
        wav_file = tmp_path / "narrator.wav"
        wav_file.touch()

        config.xtts_speaker_wav_map = {"narrator": str(wav_file)}
        backend = XTTSBackend(config)

        result = backend.get_voice_id("narrator")
        assert result == str(wav_file)

    def test_get_voice_id_missing_returns_none(self, config):
        """Missing speaker wav returns None."""
        backend = XTTSBackend(config)
        result = backend.get_voice_id("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_synthesize_not_available_returns_none(self, config):
        """Returns None when not available."""
        backend = XTTSBackend(config)
        with patch.object(backend, "is_available", return_value=False):
            result = await backend.synthesize("Hello")
            assert result is None


# =============================================================================
# Edge-TTS Backend Tests
# =============================================================================


class TestEdgeTTSBackend:
    """Tests for EdgeTTSBackend."""

    @pytest.fixture
    def config(self):
        """Basic config."""
        return TTSConfig()

    def test_is_available_with_cli(self, config):
        """Available when edge-tts CLI found."""
        backend = EdgeTTSBackend(config)
        with patch("shutil.which", return_value="/usr/bin/edge-tts"):
            assert backend.is_available() is True

    def test_is_available_with_module(self, config):
        """Available when edge_tts module installed."""
        backend = EdgeTTSBackend(config)
        with patch("shutil.which", return_value=None):
            with patch("importlib.util.find_spec") as mock_spec:
                mock_spec.return_value = MagicMock()  # Module found
                assert backend.is_available() is True

    def test_is_available_neither(self, config):
        """Not available without CLI or module."""
        backend = EdgeTTSBackend(config)
        with patch("shutil.which", return_value=None):
            with patch("importlib.util.find_spec", return_value=None):
                assert backend.is_available() is False

    def test_get_voice_id_from_mapping(self, config):
        """Voice ID from EDGE_TTS_VOICES mapping."""
        backend = EdgeTTSBackend(config)
        assert backend.get_voice_id("narrator") == EDGE_TTS_VOICES["narrator"]

    def test_get_voice_id_default(self, config):
        """Unknown voice returns default."""
        backend = EdgeTTSBackend(config)
        assert backend.get_voice_id("unknown") == EDGE_TTS_VOICES["default"]

    def test_get_command_cli(self, config):
        """Returns CLI command when available."""
        backend = EdgeTTSBackend(config)
        with patch("shutil.which", return_value="/usr/bin/edge-tts"):
            cmd = backend._get_command()
            assert cmd == ["/usr/bin/edge-tts"]

    def test_get_command_module(self, config):
        """Returns module command when CLI not available."""
        backend = EdgeTTSBackend(config)
        with patch("shutil.which", return_value=None):
            with patch("importlib.util.find_spec") as mock_spec:
                mock_spec.return_value = MagicMock()
                cmd = backend._get_command()
                assert cmd is not None
                assert "-m" in cmd
                assert "edge_tts" in cmd

    @pytest.mark.asyncio
    async def test_synthesize_success(self, config, tmp_path):
        """Successful synthesis with Edge-TTS."""
        backend = EdgeTTSBackend(config)
        output = tmp_path / "output.mp3"

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch.object(backend, "is_available", return_value=True):
            with patch.object(backend, "_get_command", return_value=["/usr/bin/edge-tts"]):
                with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                    # Create the output file to simulate success
                    output.touch()
                    result = await backend.synthesize("Hello", output_path=output)
                    assert result == output

    @pytest.mark.asyncio
    async def test_synthesize_timeout(self, config, tmp_path):
        """Timeout during synthesis returns None."""
        backend = EdgeTTSBackend(config)

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()

        with patch.object(backend, "is_available", return_value=True):
            with patch.object(backend, "_get_command", return_value=["/usr/bin/edge-tts"]):
                with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                    result = await backend.synthesize("Hello", timeout=0.1)
                    assert result is None
                    mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_process_error(self, config, tmp_path):
        """Process error returns None."""
        backend = EdgeTTSBackend(config)

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Error message"))

        with patch.object(backend, "is_available", return_value=True):
            with patch.object(backend, "_get_command", return_value=["/usr/bin/edge-tts"]):
                with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                    result = await backend.synthesize("Hello")
                    assert result is None


# =============================================================================
# Pyttsx3 Backend Tests
# =============================================================================


class TestPyttsx3Backend:
    """Tests for Pyttsx3Backend."""

    @pytest.fixture
    def config(self):
        """Basic config."""
        return TTSConfig()

    def test_is_available_with_module(self, config):
        """Available when pyttsx3 installed."""
        backend = Pyttsx3Backend(config)
        with patch.dict("sys.modules", {"pyttsx3": MagicMock()}):
            assert backend.is_available() is True

    def test_is_available_without_module(self, config):
        """Not available without pyttsx3."""
        backend = Pyttsx3Backend(config)
        with patch.dict("sys.modules", {"pyttsx3": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                assert backend.is_available() is False

    @pytest.mark.asyncio
    async def test_synthesize_success(self, config, tmp_path):
        """Successful synthesis with pyttsx3."""
        backend = Pyttsx3Backend(config)
        output = tmp_path / "output.mp3"

        mock_engine = MagicMock()

        with patch.object(backend, "is_available", return_value=True):
            with patch("pyttsx3.init", return_value=mock_engine):
                # Create output to simulate success
                output.touch()
                result = await backend.synthesize("Hello", output_path=output)

                assert result == output
                mock_engine.save_to_file.assert_called_once()
                mock_engine.runAndWait.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_not_available_returns_none(self, config):
        """Returns None when not available."""
        backend = Pyttsx3Backend(config)
        with patch.object(backend, "is_available", return_value=False):
            result = await backend.synthesize("Hello")
            assert result is None

    @pytest.mark.asyncio
    async def test_synthesize_engine_error(self, config):
        """Engine error returns None."""
        backend = Pyttsx3Backend(config)

        with patch.object(backend, "is_available", return_value=True):
            with patch("pyttsx3.init", side_effect=Exception("Engine init failed")):
                result = await backend.synthesize("Hello")
                assert result is None


# =============================================================================
# Fallback Backend Tests
# =============================================================================


class TestFallbackTTSBackend:
    """Tests for FallbackTTSBackend."""

    @pytest.fixture
    def config(self):
        """Config with specific priority."""
        return TTSConfig(backend_priority=["elevenlabs", "edge-tts", "pyttsx3"])

    def test_init_filters_unavailable_backends(self, config):
        """Init only keeps available backends."""
        # Mock all backends as unavailable except pyttsx3
        with patch.object(ElevenLabsBackend, "is_available", return_value=False):
            with patch.object(EdgeTTSBackend, "is_available", return_value=False):
                with patch.object(Pyttsx3Backend, "is_available", return_value=True):
                    backend = FallbackTTSBackend(config)
                    assert len(backend._backends) == 1
                    assert backend._backends[0].name == "pyttsx3"

    def test_is_available_with_backends(self, config):
        """Available when at least one backend available."""
        with patch.object(ElevenLabsBackend, "is_available", return_value=False):
            with patch.object(EdgeTTSBackend, "is_available", return_value=True):
                with patch.object(Pyttsx3Backend, "is_available", return_value=False):
                    backend = FallbackTTSBackend(config)
                    assert backend.is_available() is True

    def test_is_available_without_backends(self, config):
        """Not available when no backends available."""
        with patch.object(ElevenLabsBackend, "is_available", return_value=False):
            with patch.object(EdgeTTSBackend, "is_available", return_value=False):
                with patch.object(Pyttsx3Backend, "is_available", return_value=False):
                    backend = FallbackTTSBackend(config)
                    assert backend.is_available() is False

    @pytest.mark.asyncio
    async def test_synthesize_uses_first_available(self, config, tmp_path):
        """Synthesize uses first available backend."""
        output = tmp_path / "output.mp3"
        output.touch()

        mock_backend1 = AsyncMock(spec=TTSBackend)
        mock_backend1.name = "mock1"
        mock_backend1.synthesize = AsyncMock(return_value=output)

        mock_backend2 = AsyncMock(spec=TTSBackend)
        mock_backend2.name = "mock2"
        mock_backend2.synthesize = AsyncMock(return_value=output)

        backend = FallbackTTSBackend.__new__(FallbackTTSBackend)
        backend.config = config
        backend._backends = [mock_backend1, mock_backend2]

        result = await backend.synthesize("Hello")

        assert result == output
        mock_backend1.synthesize.assert_called_once()
        mock_backend2.synthesize.assert_not_called()

    @pytest.mark.asyncio
    async def test_synthesize_falls_back_on_failure(self, config, tmp_path):
        """Synthesize falls back when first backend fails."""
        output = tmp_path / "output.mp3"
        output.touch()

        mock_backend1 = AsyncMock(spec=TTSBackend)
        mock_backend1.name = "mock1"
        mock_backend1.synthesize = AsyncMock(return_value=None)

        mock_backend2 = AsyncMock(spec=TTSBackend)
        mock_backend2.name = "mock2"
        mock_backend2.synthesize = AsyncMock(return_value=output)

        backend = FallbackTTSBackend.__new__(FallbackTTSBackend)
        backend.config = config
        backend._backends = [mock_backend1, mock_backend2]

        result = await backend.synthesize("Hello")

        assert result == output
        mock_backend1.synthesize.assert_called_once()
        mock_backend2.synthesize.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_falls_back_on_exception(self, config, tmp_path):
        """Synthesize falls back when backend raises exception."""
        output = tmp_path / "output.mp3"
        output.touch()

        mock_backend1 = AsyncMock(spec=TTSBackend)
        mock_backend1.name = "mock1"
        mock_backend1.synthesize = AsyncMock(side_effect=Exception("API Error"))

        mock_backend2 = AsyncMock(spec=TTSBackend)
        mock_backend2.name = "mock2"
        mock_backend2.synthesize = AsyncMock(return_value=output)

        backend = FallbackTTSBackend.__new__(FallbackTTSBackend)
        backend.config = config
        backend._backends = [mock_backend1, mock_backend2]

        result = await backend.synthesize("Hello")

        assert result == output

    @pytest.mark.asyncio
    async def test_synthesize_all_fail_returns_none(self, config):
        """Returns None when all backends fail."""
        mock_backend1 = AsyncMock(spec=TTSBackend)
        mock_backend1.name = "mock1"
        mock_backend1.synthesize = AsyncMock(return_value=None)

        mock_backend2 = AsyncMock(spec=TTSBackend)
        mock_backend2.name = "mock2"
        mock_backend2.synthesize = AsyncMock(return_value=None)

        backend = FallbackTTSBackend.__new__(FallbackTTSBackend)
        backend.config = config
        backend._backends = [mock_backend1, mock_backend2]

        result = await backend.synthesize("Hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_synthesize_adjusts_extension_for_xtts(self, config, tmp_path):
        """Output extension adjusted to .wav for XTTS."""
        output = tmp_path / "output.mp3"

        mock_backend = AsyncMock(spec=TTSBackend)
        mock_backend.name = "xtts"
        mock_backend.synthesize = AsyncMock(return_value=tmp_path / "output.wav")

        backend = FallbackTTSBackend.__new__(FallbackTTSBackend)
        backend.config = config
        backend._backends = [mock_backend]

        await backend.synthesize("Hello", output_path=output)

        # Check that .wav extension was passed
        call_args = mock_backend.synthesize.call_args
        assert call_args[1]["output_path"].suffix == ".wav"


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestGetTTSBackend:
    """Tests for get_tts_backend factory function."""

    def test_specific_backend_found(self):
        """Returns specific backend when requested and available."""
        with patch.object(EdgeTTSBackend, "is_available", return_value=True):
            backend = get_tts_backend("edge-tts")
            assert isinstance(backend, EdgeTTSBackend)

    def test_specific_backend_alias(self):
        """Normalizes backend name aliases."""
        with patch.object(ElevenLabsBackend, "is_available", return_value=True):
            backend = get_tts_backend("eleven")
            assert isinstance(backend, ElevenLabsBackend)

    def test_specific_backend_not_available(self):
        """Raises RuntimeError when specific backend not available."""
        with patch.object(ElevenLabsBackend, "is_available", return_value=False):
            with pytest.raises(RuntimeError, match="not available"):
                get_tts_backend("elevenlabs")

    def test_unknown_backend_raises_value_error(self):
        """Raises ValueError for unknown backend name."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_tts_backend("nonexistent")

    def test_auto_select_first_available(self):
        """Auto-selects first available backend from priority."""
        config = TTSConfig(backend_priority=["elevenlabs", "polly", "edge-tts"])

        with patch.object(ElevenLabsBackend, "is_available", return_value=False):
            with patch.object(PollyBackend, "is_available", return_value=False):
                with patch.object(EdgeTTSBackend, "is_available", return_value=True):
                    backend = get_tts_backend(config=config)
                    assert isinstance(backend, EdgeTTSBackend)

    def test_no_backends_available_raises(self):
        """Raises RuntimeError when no backends available."""
        config = TTSConfig(backend_priority=["elevenlabs", "polly"])

        with patch.object(ElevenLabsBackend, "is_available", return_value=False):
            with patch.object(PollyBackend, "is_available", return_value=False):
                with pytest.raises(RuntimeError, match="No TTS backends available"):
                    get_tts_backend(config=config)


class TestGetFallbackBackend:
    """Tests for get_fallback_backend function."""

    def test_returns_fallback_backend(self):
        """Returns FallbackTTSBackend instance."""
        with patch.object(ElevenLabsBackend, "is_available", return_value=False):
            with patch.object(PollyBackend, "is_available", return_value=False):
                with patch.object(XTTSBackend, "is_available", return_value=False):
                    with patch.object(EdgeTTSBackend, "is_available", return_value=False):
                        with patch.object(Pyttsx3Backend, "is_available", return_value=True):
                            backend = get_fallback_backend()
                            assert isinstance(backend, FallbackTTSBackend)


# =============================================================================
# Voice Mapping Tests
# =============================================================================


class TestVoiceMappings:
    """Tests for voice mapping constants."""

    def test_elevenlabs_voices_has_default(self):
        """ELEVENLABS_VOICES has default entry."""
        assert "default" in ELEVENLABS_VOICES
        assert isinstance(ELEVENLABS_VOICES["default"], str)

    def test_edge_tts_voices_has_default(self):
        """EDGE_TTS_VOICES has default entry."""
        assert "default" in EDGE_TTS_VOICES
        assert isinstance(EDGE_TTS_VOICES["default"], str)

    def test_polly_voices_has_default(self):
        """POLLY_VOICES has default entry."""
        assert "default" in POLLY_VOICES
        assert isinstance(POLLY_VOICES["default"], str)

    def test_xtts_speakers_has_default(self):
        """XTTS_SPEAKERS has default entry."""
        assert "default" in XTTS_SPEAKERS

    def test_all_mappings_have_narrator(self):
        """All voice mappings have narrator entry."""
        assert "narrator" in ELEVENLABS_VOICES
        assert "narrator" in EDGE_TTS_VOICES
        assert "narrator" in POLLY_VOICES
        assert "narrator" in XTTS_SPEAKERS


# =============================================================================
# Backend Registry Tests
# =============================================================================


class TestBackendRegistry:
    """Tests for BACKEND_REGISTRY."""

    def test_all_backends_registered(self):
        """All backend classes are registered."""
        assert "elevenlabs" in BACKEND_REGISTRY
        assert "polly" in BACKEND_REGISTRY
        assert "xtts" in BACKEND_REGISTRY
        assert "edge-tts" in BACKEND_REGISTRY
        assert "pyttsx3" in BACKEND_REGISTRY

    def test_registry_values_are_classes(self):
        """Registry values are backend classes."""
        for name, cls in BACKEND_REGISTRY.items():
            assert issubclass(cls, TTSBackend)

    def test_registry_classes_instantiable(self):
        """Registry classes can be instantiated."""
        for name, cls in BACKEND_REGISTRY.items():
            instance = cls(TTSConfig())
            assert instance.name == name or name == "edge-tts"
