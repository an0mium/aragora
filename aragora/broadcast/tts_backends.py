"""
TTS Backend Abstraction Layer.

Provides multiple text-to-speech backends with fallback support:
1. ElevenLabs - Highest quality, most voice variety (cloud, paid)
2. Amazon Polly - High quality neural voices (cloud, AWS)
3. Coqui XTTS v2 - High quality local TTS (GPU recommended)
4. Edge-TTS - Free Microsoft TTS (cloud, free)
5. pyttsx3 - Offline fallback (low quality)

Usage:
    from aragora.broadcast.tts_backends import get_tts_backend, TTSConfig

    # Auto-select best available backend
    backend = get_tts_backend()
    audio_path = await backend.synthesize("Hello world", voice="narrator")

    # Or specify backend explicitly
    backend = get_tts_backend("elevenlabs")
"""

import asyncio
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aragora.exceptions import ConfigurationError, ExternalServiceError

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

def _parse_csv(value: Optional[str]) -> Optional[list[str]]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_json_env(name: str) -> Optional[Any]:
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in %s", name)
        return None


def _normalize_backend_name(name: str) -> str:
    aliases = {
        "eleven": "elevenlabs",
        "11labs": "elevenlabs",
        "edge": "edge-tts",
        "aws": "polly",
        "aws-polly": "polly",
        "amazon-polly": "polly",
        "coqui": "xtts",
        "xtts-v2": "xtts",
        "fallback": "pyttsx3",
    }
    return aliases.get(name, name)


@dataclass
class TTSConfig:
    """Configuration for TTS backends."""

    # Backend priority (first available is used)
    backend_priority: List[str] = field(
        default_factory=lambda: ["elevenlabs", "polly", "xtts", "edge-tts", "pyttsx3"]
    )

    # ElevenLabs settings
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_model: str = "eleven_multilingual_v2"
    elevenlabs_voice_map: Dict[str, str] = field(default_factory=dict)
    elevenlabs_default_voice_id: Optional[str] = None

    # XTTS settings
    xtts_model_path: Optional[str] = None
    xtts_device: str = "auto"  # auto, cuda, cpu
    xtts_language: str = "en"
    xtts_speaker_wav: Optional[str] = None
    xtts_speaker_wav_map: Dict[str, str] = field(default_factory=dict)

    # Amazon Polly settings
    polly_region: Optional[str] = None
    polly_engine: str = "neural"  # neural or standard
    polly_text_type: str = "text"  # text or ssml
    polly_voice_map: Dict[str, str] = field(default_factory=dict)
    polly_default_voice_id: Optional[str] = None
    polly_lexicons: Optional[List[str]] = None

    # Cache settings
    cache_dir: Optional[Path] = None
    enable_cache: bool = True

    @classmethod
    def from_env(cls) -> "TTSConfig":
        """Create config from environment variables."""
        backend_order = (
            os.getenv("ARAGORA_TTS_ORDER")
            or os.getenv("ARAGORA_TTS_BACKEND_PRIORITY")
            or os.getenv("TTS_BACKEND_PRIORITY")
        )
        parsed_backend = _parse_csv(backend_order)
        if parsed_backend:
            parsed_backend = [_normalize_backend_name(name.lower()) for name in parsed_backend]

        voice_map = (
            _parse_json_env("ARAGORA_ELEVENLABS_VOICE_MAP")
            or _parse_json_env("ELEVENLABS_VOICE_MAP")
            or {}
        )
        if not isinstance(voice_map, dict):
            voice_map = {}

        speaker_wav_map = (
            _parse_json_env("ARAGORA_XTTS_SPEAKER_WAV_MAP")
            or _parse_json_env("XTTS_SPEAKER_WAV_MAP")
            or {}
        )
        if not isinstance(speaker_wav_map, dict):
            speaker_wav_map = {}

        polly_voice_map = _parse_json_env("ARAGORA_POLLY_VOICE_MAP") or {}
        if not isinstance(polly_voice_map, dict):
            polly_voice_map = {}

        polly_lexicons = _parse_csv(os.getenv("ARAGORA_POLLY_LEXICONS"))

        return cls(
            backend_priority=parsed_backend or cls().backend_priority,
            elevenlabs_api_key=os.getenv("ARAGORA_ELEVENLABS_API_KEY") or os.getenv("ELEVENLABS_API_KEY"),
            elevenlabs_model=os.getenv("ARAGORA_ELEVENLABS_MODEL_ID", os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")),
            elevenlabs_voice_map=voice_map,
            elevenlabs_default_voice_id=(
                os.getenv("ARAGORA_ELEVENLABS_VOICE_ID")
                or os.getenv("ARAGORA_ELEVENLABS_DEFAULT_VOICE_ID")
                or os.getenv("ELEVENLABS_VOICE_ID")
            ),
            xtts_model_path=os.getenv("ARAGORA_XTTS_MODEL_PATH") or os.getenv("ARAGORA_XTTS_MODEL") or os.getenv("XTTS_MODEL_PATH"),
            xtts_device=os.getenv("ARAGORA_XTTS_DEVICE", os.getenv("XTTS_DEVICE", "auto")),
            xtts_language=os.getenv("ARAGORA_XTTS_LANGUAGE", os.getenv("XTTS_LANGUAGE", "en")),
            xtts_speaker_wav=os.getenv("ARAGORA_XTTS_SPEAKER_WAV") or os.getenv("XTTS_SPEAKER_WAV"),
            xtts_speaker_wav_map=speaker_wav_map,
            polly_region=(
                os.getenv("ARAGORA_POLLY_REGION")
                or os.getenv("AWS_REGION")
                or os.getenv("AWS_DEFAULT_REGION")
            ),
            polly_engine=os.getenv("ARAGORA_POLLY_ENGINE", "neural"),
            polly_text_type=os.getenv("ARAGORA_POLLY_TEXT_TYPE", "text"),
            polly_voice_map=polly_voice_map,
            polly_default_voice_id=(
                os.getenv("ARAGORA_POLLY_VOICE_ID")
                or os.getenv("ARAGORA_POLLY_DEFAULT_VOICE_ID")
            ),
            polly_lexicons=polly_lexicons,
            cache_dir=Path(os.getenv("ARAGORA_TTS_CACHE_DIR", os.getenv("TTS_CACHE_DIR", ".cache/tts"))),
        )


# =============================================================================
# Voice Mappings
# =============================================================================

# ElevenLabs voice IDs (these are examples - get actual IDs from your account)
ELEVENLABS_VOICES: Dict[str, str] = {
    # Character voices (diverse, expressive)
    "claude-visionary": "pNInz6obpgDQGcFmaJgB",  # Adam - deep, authoritative
    "codex-engineer": "VR6AewLTigWG4xSOukaG",   # Arnold - technical
    "gemini-visionary": "EXAVITQu4vr4xnSDxMaL",  # Bella - warm, expressive
    "grok-lateral-thinker": "TxGEqnHWrfWFTfGW9XjX",  # Josh - energetic
    "narrator": "21m00Tcm4TlvDq8ikWAM",  # Rachel - clear narrator
    # Fallback
    "default": "21m00Tcm4TlvDq8ikWAM",
}

# XTTS speaker references (sample audio files for voice cloning)
XTTS_SPEAKERS: Dict[str, str] = {
    "claude-visionary": "voices/claude.wav",
    "codex-engineer": "voices/codex.wav",
    "gemini-visionary": "voices/gemini.wav",
    "grok-lateral-thinker": "voices/grok.wav",
    "narrator": "voices/narrator.wav",
    "default": None,  # Use default XTTS voice
}

# Edge-TTS voices (Microsoft Azure neural voices)
EDGE_TTS_VOICES: Dict[str, str] = {
    "claude-visionary": "en-GB-SoniaNeural",
    "codex-engineer": "en-US-GuyNeural",
    "gemini-visionary": "en-AU-NatashaNeural",
    "grok-lateral-thinker": "en-US-ChristopherNeural",
    "narrator": "en-US-AriaNeural",
    "default": "en-US-AriaNeural",
}

# Amazon Polly voices (AWS)
POLLY_VOICES: Dict[str, str] = {
    "claude-visionary": "Matthew",
    "codex-engineer": "Joanna",
    "gemini-visionary": "Salli",
    "grok-lateral-thinker": "Joey",
    "narrator": "Joanna",
    "default": "Joanna",
}


# =============================================================================
# Base Backend Interface
# =============================================================================

class TTSBackend(ABC):
    """Abstract base class for TTS backends."""

    name: str = "base"

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        output_path: Optional[Path] = None,
        **kwargs,
    ) -> Optional[Path]:
        """
        Synthesize speech from text.

        Args:
            text: Text to convert to speech
            voice: Voice/speaker identifier
            output_path: Path to save audio (auto-generated if None)
            **kwargs: Backend-specific options

        Returns:
            Path to generated audio file, or None if failed
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available and configured."""
        pass

    def get_voice_id(self, speaker: str) -> str:
        """Get the voice ID for a speaker name."""
        return speaker


# =============================================================================
# ElevenLabs Backend
# =============================================================================

class ElevenLabsBackend(TTSBackend):
    """
    ElevenLabs TTS backend.

    Highest quality voices with excellent prosody and emotion.
    Requires API key: https://elevenlabs.io

    Features:
    - 29+ languages
    - Voice cloning
    - Emotion/style control
    - Low latency streaming
    """

    name = "elevenlabs"

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig.from_env()
        self._client = None

    def is_available(self) -> bool:
        """Check if ElevenLabs is configured."""
        if not self.config.elevenlabs_api_key:
            return False

        try:
            import elevenlabs
            return True
        except ImportError:
            return False

    def _get_client(self):
        """Get or create ElevenLabs client."""
        if self._client is None:
            try:
                from elevenlabs.client import ElevenLabs
                self._client = ElevenLabs(api_key=self.config.elevenlabs_api_key)
            except ImportError:
                raise ConfigurationError(
                    component="ElevenLabsTTS",
                    reason="elevenlabs package not installed. Run: pip install elevenlabs"
                )
        return self._client

    def get_voice_id(self, speaker: str) -> str:
        """Get ElevenLabs voice ID for speaker."""
        if speaker in self.config.elevenlabs_voice_map:
            return self.config.elevenlabs_voice_map[speaker]
        if speaker in ELEVENLABS_VOICES:
            return ELEVENLABS_VOICES[speaker]
        if self.config.elevenlabs_default_voice_id:
            return self.config.elevenlabs_default_voice_id
        return speaker or ELEVENLABS_VOICES["default"]

    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        output_path: Optional[Path] = None,
        **kwargs,
    ) -> Optional[Path]:
        """Synthesize speech using ElevenLabs."""
        if not self.is_available():
            return None

        try:
            client = self._get_client()
            voice_id = self.get_voice_id(voice)

            # Generate output path if not provided
            if output_path is None:
                text_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
                output_path = Path(tempfile.gettempdir()) / f"elevenlabs_{voice}_{text_hash}.mp3"

            # Run sync API in thread pool
            def _generate():
                # Use text_to_speech.convert() API (elevenlabs SDK v2+)
                audio_iterator = client.text_to_speech.convert(
                    text=text,
                    voice_id=voice_id,
                    model_id=self.config.elevenlabs_model,
                )
                # Write audio to file
                with open(output_path, "wb") as f:
                    for chunk in audio_iterator:
                        f.write(chunk)
                return output_path

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _generate)

            if result and result.exists():
                logger.debug(f"ElevenLabs generated: {result}")
                return result

        except Exception as e:
            logger.warning(f"ElevenLabs synthesis failed: {e}")

        return None


# =============================================================================
# Coqui XTTS Backend
# =============================================================================

class XTTSBackend(TTSBackend):
    """
    Coqui XTTS v2 TTS backend.

    High-quality local TTS with voice cloning capability.
    GPU recommended for reasonable speed.

    Features:
    - Zero-shot voice cloning
    - 17 languages
    - Runs locally (no API costs)
    - Good for privacy-sensitive use
    """

    name = "xtts"

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig.from_env()
        self._model = None
        self._device: Optional[str] = None

    def is_available(self) -> bool:
        """Check if XTTS is available."""
        try:
            import torch
            from TTS.api import TTS
            return True
        except ImportError:
            return False

    def _get_device(self) -> str:
        """Determine the device to use."""
        if self._device is not None:
            return self._device

        device = self.config.xtts_device
        if device == "auto":
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        else:
            self._device = device

        return self._device

    def _get_model(self):
        """Get or create XTTS model."""
        if self._model is None:
            try:
                from TTS.api import TTS

                device = self._get_device()
                logger.info(f"Loading XTTS model on {device}...")

                # Use XTTS v2 model
                model_name = self.config.xtts_model_path or "tts_models/multilingual/multi-dataset/xtts_v2"
                self._model = TTS(model_name).to(device)

                logger.info("XTTS model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load XTTS model: {e}")
                raise

        return self._model

    def get_voice_id(self, speaker: str) -> Optional[str]:
        """Get speaker reference audio path."""
        if speaker:
            candidate = Path(speaker).expanduser()
            if candidate.exists():
                return str(candidate)

        ref_path = self.config.xtts_speaker_wav_map.get(speaker) or self.config.xtts_speaker_wav
        if ref_path:
            full_path = Path(ref_path).expanduser()
            if full_path.exists():
                return str(full_path)
            logger.warning("XTTS speaker wav not found: %s", full_path)

        ref_path = XTTS_SPEAKERS.get(speaker, XTTS_SPEAKERS.get("default"))
        if ref_path:
            full_path = Path(__file__).parent / ref_path
            if full_path.exists():
                return str(full_path)
        return None

    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        output_path: Optional[Path] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> Optional[Path]:
        """Synthesize speech using XTTS."""
        if not self.is_available():
            return None

        try:
            model = self._get_model()
            speaker_wav = self.get_voice_id(voice)
            language = language or self.config.xtts_language

            # Generate output path if not provided
            if output_path is None:
                text_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
                output_path = Path(tempfile.gettempdir()) / f"xtts_{voice}_{text_hash}.wav"

            # Run model inference in thread pool (CPU-bound)
            def _generate():
                if speaker_wav:
                    # Voice cloning with reference audio
                    model.tts_to_file(
                        text=text,
                        file_path=str(output_path),
                        speaker_wav=speaker_wav,
                        language=language,
                    )
                else:
                    # Use default speaker
                    model.tts_to_file(
                        text=text,
                        file_path=str(output_path),
                        language=language,
                    )
                return output_path

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _generate)

            if result and result.exists():
                logger.debug(f"XTTS generated: {result}")
                return result

        except Exception as e:
            logger.warning(f"XTTS synthesis failed: {e}")

        return None


# =============================================================================
# Edge-TTS Backend
# =============================================================================

class EdgeTTSBackend(TTSBackend):
    """
    Edge-TTS backend (Microsoft Azure neural voices).

    Free, good quality, many voices.
    Requires internet connection.
    """

    name = "edge-tts"

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig.from_env()

    def is_available(self) -> bool:
        """Check if edge-tts is available."""
        # Check CLI or module
        if shutil.which("edge-tts"):
            return True
        if importlib.util.find_spec("edge_tts") is not None:
            return True
        return False

    def get_voice_id(self, speaker: str) -> str:
        """Get Edge-TTS voice for speaker."""
        return EDGE_TTS_VOICES.get(speaker, EDGE_TTS_VOICES["default"])

    def _get_command(self) -> Optional[List[str]]:
        """Get edge-tts command."""
        cmd = shutil.which("edge-tts")
        if cmd:
            return [cmd]
        if importlib.util.find_spec("edge_tts") is not None:
            return [sys.executable, "-m", "edge_tts"]
        return None

    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        output_path: Optional[Path] = None,
        timeout: float = 60.0,
        **kwargs,
    ) -> Optional[Path]:
        """Synthesize speech using Edge-TTS."""
        if not self.is_available():
            return None

        cmd = self._get_command()
        if not cmd:
            return None

        try:
            voice_id = self.get_voice_id(voice)

            # Generate output path if not provided
            if output_path is None:
                text_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
                output_path = Path(tempfile.gettempdir()) / f"edge_{voice}_{text_hash}.mp3"

            full_cmd = cmd + [
                "--voice", voice_id,
                "--text", text,
                "--write-media", str(output_path),
            ]

            process = await asyncio.create_subprocess_exec(
                *full_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                logger.warning(f"Edge-TTS timed out after {timeout}s")
                return None

            if process.returncode == 0 and output_path.exists():
                logger.debug(f"Edge-TTS generated: {output_path}")
                return output_path

            error_msg = stderr.decode("utf-8", errors="replace").strip()
            logger.warning(f"Edge-TTS failed: {error_msg[:200]}")

        except Exception as e:
            logger.warning(f"Edge-TTS synthesis failed: {e}")

        return None


# =============================================================================
# Amazon Polly Backend
# =============================================================================

class PollyBackend(TTSBackend):
    """
    Amazon Polly TTS backend.

    High-quality neural voices via AWS. Requires AWS credentials.
    """

    name = "polly"

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig.from_env()
        self._client = None

    def is_available(self) -> bool:
        """Check if Amazon Polly is available and configured."""
        try:
            import boto3
        except ImportError:
            return False

        try:
            session = boto3.Session(region_name=self.config.polly_region)
            creds = session.get_credentials()
            if creds is None:
                return False
        except Exception as e:
            logger.debug(f"AWS credentials check failed: {e}")
            return False
        return True

    def _get_client(self):
        """Get or create Amazon Polly client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client("polly", region_name=self.config.polly_region)
            except Exception as e:
                raise ExternalServiceError(
                    service="Amazon Polly",
                    reason=f"Failed to initialize client: {e}"
                ) from e
        return self._client

    def get_voice_id(self, speaker: str) -> str:
        """Get Polly voice ID for speaker."""
        if speaker in self.config.polly_voice_map:
            return self.config.polly_voice_map[speaker]
        if speaker in POLLY_VOICES:
            return POLLY_VOICES[speaker]
        if self.config.polly_default_voice_id:
            return self.config.polly_default_voice_id
        return speaker or POLLY_VOICES["default"]

    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        output_path: Optional[Path] = None,
        **kwargs,
    ) -> Optional[Path]:
        """Synthesize speech using Amazon Polly."""
        if not self.is_available():
            return None

        try:
            client = self._get_client()
            voice_id = self.get_voice_id(voice)
            engine = (self.config.polly_engine or "neural").lower()
            text_type = (self.config.polly_text_type or "text").lower()
            if text_type not in ("text", "ssml"):
                text_type = "text"

            if output_path is None:
                text_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
                output_path = Path(tempfile.gettempdir()) / f"polly_{voice}_{text_hash}.mp3"

            def _generate():
                params: Dict[str, Any] = {
                    "Text": text,
                    "VoiceId": voice_id,
                    "OutputFormat": "mp3",
                    "Engine": engine,
                    "TextType": text_type,
                }
                if self.config.polly_lexicons:
                    params["LexiconNames"] = self.config.polly_lexicons

                response = client.synthesize_speech(**params)
                stream = response.get("AudioStream")
                if stream is None:
                    return None
                with open(output_path, "wb") as f:
                    f.write(stream.read())
                return output_path

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _generate)

            if result and result.exists():
                logger.debug(f"Polly generated: {result}")
                return result

        except Exception as e:
            logger.warning(f"Polly synthesis failed: {e}")

        return None


# =============================================================================
# pyttsx3 Backend (Offline Fallback)
# =============================================================================

class Pyttsx3Backend(TTSBackend):
    """
    pyttsx3 offline TTS backend.

    Low quality but works offline without any API.
    Last resort fallback.
    """

    name = "pyttsx3"

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig.from_env()

    def is_available(self) -> bool:
        """Check if pyttsx3 is available."""
        try:
            import pyttsx3
            return True
        except ImportError:
            return False

    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        output_path: Optional[Path] = None,
        **kwargs,
    ) -> Optional[Path]:
        """Synthesize speech using pyttsx3."""
        if not self.is_available():
            return None

        try:
            import pyttsx3

            # Generate output path if not provided
            if output_path is None:
                text_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
                output_path = Path(tempfile.gettempdir()) / f"pyttsx3_{voice}_{text_hash}.mp3"

            def _generate():
                engine = pyttsx3.init()
                engine.save_to_file(text, str(output_path))
                engine.runAndWait()
                return output_path

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _generate)

            if result and result.exists():
                logger.debug(f"pyttsx3 generated: {result}")
                return result

        except Exception as e:
            logger.warning(f"pyttsx3 synthesis failed: {e}")

        return None


# =============================================================================
# Backend Factory
# =============================================================================

# Registry of available backends
BACKEND_REGISTRY: Dict[str, type] = {
    "elevenlabs": ElevenLabsBackend,
    "polly": PollyBackend,
    "xtts": XTTSBackend,
    "edge-tts": EdgeTTSBackend,
    "pyttsx3": Pyttsx3Backend,
}


def get_tts_backend(
    backend_name: Optional[str] = None,
    config: Optional[TTSConfig] = None,
) -> TTSBackend:
    """
    Get a TTS backend instance.

    Args:
        backend_name: Specific backend name, or None for auto-selection
        config: TTS configuration

    Returns:
        TTSBackend instance

    Raises:
        RuntimeError: If no backends are available
    """
    config = config or TTSConfig.from_env()

    if backend_name:
        # Specific backend requested
        backend_name = _normalize_backend_name(backend_name.lower())
        if backend_name not in BACKEND_REGISTRY:
            raise ValueError(f"Unknown backend: {backend_name}")

        backend_cls = BACKEND_REGISTRY[backend_name]
        backend = backend_cls(config)

        if not backend.is_available():
            raise ConfigurationError(
                component="TTSBackend",
                reason=f"Backend '{backend_name}' is not available"
            )

        return backend

    # Auto-select first available backend
    for name in config.backend_priority:
        if name not in BACKEND_REGISTRY:
            continue

        backend_cls = BACKEND_REGISTRY[name]
        backend = backend_cls(config)

        if backend.is_available():
            logger.info(f"Selected TTS backend: {name}")
            return backend

    raise ConfigurationError(
        component="TTSBackend",
        reason="No TTS backends available. Install at least one: elevenlabs, boto3, edge-tts, or pyttsx3"
    )


class FallbackTTSBackend(TTSBackend):
    """
    TTS backend with automatic fallback chain.

    Tries backends in priority order until one succeeds.
    """

    name = "fallback"

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig.from_env()
        self._backends: List[TTSBackend] = []
        self._init_backends()

    def _init_backends(self):
        """Initialize available backends in priority order."""
        for name in self.config.backend_priority:
            if name not in BACKEND_REGISTRY:
                continue

            backend_cls = BACKEND_REGISTRY[name]
            backend = backend_cls(self.config)

            if backend.is_available():
                self._backends.append(backend)
                logger.debug(f"TTS backend available: {name}")

        if self._backends:
            logger.info(f"TTS fallback chain: {[b.name for b in self._backends]}")

    def is_available(self) -> bool:
        """Check if any backend is available."""
        return len(self._backends) > 0

    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        output_path: Optional[Path] = None,
        **kwargs,
    ) -> Optional[Path]:
        """Synthesize with fallback through available backends."""
        for backend in self._backends:
            try:
                backend_output = output_path
                if output_path is not None:
                    ext = ".wav" if backend.name == "xtts" else ".mp3"
                    backend_output = output_path.with_suffix(ext)
                result = await backend.synthesize(
                    text=text,
                    voice=voice,
                    output_path=backend_output,
                    **kwargs,
                )
                if result:
                    return result
            except Exception as e:
                logger.debug(f"{backend.name} failed, trying next: {e}")
                continue

        logger.error("All TTS backends failed")
        return None


def get_fallback_backend(config: Optional[TTSConfig] = None) -> FallbackTTSBackend:
    """Get a TTS backend with fallback support."""
    return FallbackTTSBackend(config)


__all__ = [
    # Config
    "TTSConfig",
    # Voice mappings
    "ELEVENLABS_VOICES",
    "XTTS_SPEAKERS",
    "EDGE_TTS_VOICES",
    # Base class
    "TTSBackend",
    # Backends
    "ElevenLabsBackend",
    "XTTSBackend",
    "EdgeTTSBackend",
    "Pyttsx3Backend",
    "FallbackTTSBackend",
    # Factory functions
    "get_tts_backend",
    "get_fallback_backend",
    "BACKEND_REGISTRY",
]
