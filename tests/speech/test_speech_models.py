"""Tests for speech module data models and imports."""

import pytest


class TestImports:
    """Verify speech module exports are importable."""

    def test_import_base_models(self):
        from aragora.speech.providers.base import (
            STTProvider,
            STTProviderConfig,
            TranscriptionResult,
            TranscriptionSegment,
        )

    def test_import_from_package(self):
        from aragora.speech import (
            STTProvider,
            STTProviderConfig,
            TranscriptionResult,
            TranscriptionSegment,
        )

    def test_import_transcribe_functions(self):
        from aragora.speech.transcribe import get_provider, transcribe_audio, transcribe_audio_file

    def test_import_get_provider_from_transcribe(self):
        from aragora.speech.transcribe import get_provider

        assert callable(get_provider)


class TestSTTProviderConfig:
    """Test STTProviderConfig dataclass."""

    def test_defaults(self):
        from aragora.speech.providers.base import STTProviderConfig

        cfg = STTProviderConfig()
        assert cfg.provider_name == "base"
        assert cfg.default_language is None
        assert cfg.model == "default"
        assert cfg.include_timestamps is True
        assert cfg.include_word_timestamps is False
        assert cfg.extra_options == {}

    def test_custom_values(self):
        from aragora.speech.providers.base import STTProviderConfig

        cfg = STTProviderConfig(
            provider_name="whisper",
            default_language="en",
            model="whisper-1",
            include_timestamps=False,
            include_word_timestamps=True,
            extra_options={"response_format": "verbose_json"},
        )
        assert cfg.provider_name == "whisper"
        assert cfg.default_language == "en"
        assert cfg.model == "whisper-1"
        assert cfg.include_timestamps is False
        assert cfg.include_word_timestamps is True
        assert cfg.extra_options == {"response_format": "verbose_json"}

    def test_extra_options_independent(self):
        """Each instance gets its own extra_options dict."""
        from aragora.speech.providers.base import STTProviderConfig

        a = STTProviderConfig()
        b = STTProviderConfig()
        a.extra_options["key"] = "value"
        assert "key" not in b.extra_options


class TestTranscriptionSegment:
    """Test TranscriptionSegment dataclass."""

    def test_required_fields(self):
        from aragora.speech.providers.base import TranscriptionSegment

        seg = TranscriptionSegment(text="hello", start=0.0, end=1.5)
        assert seg.text == "hello"
        assert seg.start == 0.0
        assert seg.end == 1.5
        assert seg.confidence is None
        assert seg.words is None

    def test_optional_fields(self):
        from aragora.speech.providers.base import TranscriptionSegment

        words = [{"word": "hello", "start": 0.0, "end": 0.5}]
        seg = TranscriptionSegment(text="hello", start=0.0, end=1.5, confidence=0.95, words=words)
        assert seg.confidence == 0.95
        assert seg.words == words


class TestTranscriptionResult:
    """Test TranscriptionResult dataclass."""

    def test_required_fields(self):
        from aragora.speech.providers.base import TranscriptionResult

        result = TranscriptionResult(text="hello world", language="en", duration=2.0)
        assert result.text == "hello world"
        assert result.language == "en"
        assert result.duration == 2.0
        assert result.segments == []
        assert result.provider == "unknown"
        assert result.model == "unknown"

    def test_custom_values(self):
        from aragora.speech.providers.base import TranscriptionResult, TranscriptionSegment

        seg = TranscriptionSegment(text="hello", start=0.0, end=1.0)
        result = TranscriptionResult(
            text="hello",
            language="en",
            duration=1.0,
            segments=[seg],
            provider="openai_whisper",
            model="whisper-1",
        )
        assert len(result.segments) == 1
        assert result.provider == "openai_whisper"
        assert result.model == "whisper-1"

    def test_to_dict(self):
        from aragora.speech.providers.base import TranscriptionResult, TranscriptionSegment

        seg = TranscriptionSegment(text="hi", start=0.0, end=0.5, confidence=0.99)
        result = TranscriptionResult(
            text="hi", language="en", duration=0.5, segments=[seg], provider="test", model="m1"
        )
        d = result.to_dict()
        assert d["text"] == "hi"
        assert d["language"] == "en"
        assert d["duration"] == 0.5
        assert d["provider"] == "test"
        assert d["model"] == "m1"
        assert len(d["segments"]) == 1
        assert d["segments"][0]["text"] == "hi"
        assert d["segments"][0]["confidence"] == 0.99

    def test_to_dict_empty_segments(self):
        from aragora.speech.providers.base import TranscriptionResult

        result = TranscriptionResult(text="", language="en", duration=0.0)
        d = result.to_dict()
        assert d["segments"] == []

    def test_segments_independent(self):
        """Each instance gets its own segments list."""
        from aragora.speech.providers.base import TranscriptionResult

        a = TranscriptionResult(text="a", language="en", duration=1.0)
        b = TranscriptionResult(text="b", language="en", duration=1.0)
        a.segments.append("x")
        assert len(b.segments) == 0


class TestSTTProviderAbstract:
    """Test that STTProvider cannot be directly instantiated."""

    def test_cannot_instantiate(self):
        from aragora.speech.providers.base import STTProvider

        with pytest.raises(TypeError):
            STTProvider()

    def test_is_abstract(self):
        from abc import ABC
        from aragora.speech.providers.base import STTProvider

        assert issubclass(STTProvider, ABC)
