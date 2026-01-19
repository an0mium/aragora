"""
Whisper Connector - Audio/Video transcription for aragora agents.

Provides speech-to-text transcription via OpenAI's Whisper API for:
- Audio files (mp3, m4a, wav, webm, mpga, mpeg)
- Video files (mp4, webm, mov - audio track extracted)
- Live voice streaming (chunked transcription)

Requires OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from aragora.connectors.base import BaseConnector, Evidence
from aragora.connectors.exceptions import (
    ConnectorAPIError,
    ConnectorConfigError,
    ConnectorRateLimitError,
)
from aragora.reasoning.provenance import ProvenanceManager, SourceType

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# OpenAI Whisper API endpoint
WHISPER_API_URL = "https://api.openai.com/v1/audio/transcriptions"

# Supported file formats
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".webm", ".mpga", ".mpeg"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
ALL_SUPPORTED_EXTENSIONS = SUPPORTED_AUDIO_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS

# MIME type mappings
MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".mpga": "audio/mpeg",
    ".mpeg": "audio/mpeg",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}

# Whisper API limits
MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@dataclass
class TranscriptionSegment:
    """A timestamped segment of transcription."""

    start: float  # seconds
    end: float  # seconds
    text: str
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptionSegment":
        return cls(
            start=data["start"],
            end=data["end"],
            text=data["text"],
            confidence=data.get("confidence", 0.0),
        )


@dataclass
class TranscriptionResult:
    """Result from audio/video transcription."""

    id: str
    text: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str = ""
    duration_seconds: float = 0.0
    source_filename: str = ""
    word_count: int = 0
    confidence: float = 0.0

    def __post_init__(self):
        if not self.word_count and self.text:
            self.word_count = len(self.text.split())

    def to_evidence(self) -> Evidence:
        """Convert to Evidence for debate context."""
        return Evidence(
            id=self.id,
            source_type=SourceType.AUDIO_TRANSCRIPT,
            source_id=self.source_filename,
            content=self.text,
            title=f"Transcript: {self.source_filename}",
            confidence=self.confidence or 0.85,  # Whisper is generally reliable
            metadata={
                "segments": [s.to_dict() for s in self.segments],
                "language": self.language,
                "duration_seconds": self.duration_seconds,
                "word_count": self.word_count,
            },
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "source_filename": self.source_filename,
            "word_count": self.word_count,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptionResult":
        return cls(
            id=data["id"],
            text=data["text"],
            segments=[TranscriptionSegment.from_dict(s) for s in data.get("segments", [])],
            language=data.get("language", ""),
            duration_seconds=data.get("duration_seconds", 0.0),
            source_filename=data.get("source_filename", ""),
            word_count=data.get("word_count", 0),
            confidence=data.get("confidence", 0.0),
        )


class WhisperConnector(BaseConnector):
    """
    Connector for OpenAI Whisper API transcription.

    Enables agents to:
    - Transcribe audio files (mp3, wav, m4a, etc.)
    - Transcribe video files (extracts audio track)
    - Get timestamped segments for precise citations
    - Stream transcription for live voice input

    Example:
        connector = WhisperConnector()

        # Transcribe a file
        with open("meeting.mp3", "rb") as f:
            result = await connector.transcribe(f.read(), "meeting.mp3")
        print(result.text)

        # Get as debate evidence
        evidence = result.to_evidence()
    """

    # Default model (currently only whisper-1 available)
    DEFAULT_MODEL = "whisper-1"

    # Rate limit: Whisper API allows ~50 RPM
    RATE_LIMIT_RPM = 50
    RATE_LIMIT_DELAY = 60.0 / RATE_LIMIT_RPM  # ~1.2 seconds between requests

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        language: Optional[str] = None,
        response_format: str = "verbose_json",
        provenance: Optional[ProvenanceManager] = None,
        default_confidence: float = 0.85,
        timeout: int = 120,
        max_cache_entries: int = 100,
        cache_ttl_seconds: float = 86400.0,  # 24 hour cache
    ):
        """
        Initialize WhisperConnector.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Whisper model to use (currently only "whisper-1")
            language: ISO-639-1 language code (auto-detect if None)
            response_format: "json", "text", "srt", "verbose_json", or "vtt"
            provenance: Optional provenance manager for tracking
            default_confidence: Base confidence for transcriptions
            timeout: HTTP request timeout in seconds
            max_cache_entries: Maximum cached entries
            cache_ttl_seconds: Cache TTL in seconds
        """
        super().__init__(
            provenance=provenance,
            default_confidence=default_confidence,
            max_cache_entries=max_cache_entries,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.language = language
        self.response_format = response_format
        self.timeout = timeout
        self._last_request_time: float = 0.0

        if not self.api_key:
            logger.warning("WhisperConnector: No OPENAI_API_KEY configured")

    @property
    def source_type(self) -> SourceType:
        """Transcriptions are audio transcript data."""
        return SourceType.AUDIO_TRANSCRIPT

    @property
    def name(self) -> str:
        """Human-readable connector name."""
        return "Whisper"

    @property
    def is_available(self) -> bool:
        """Check if connector is properly configured."""
        return HTTPX_AVAILABLE and bool(self.api_key)

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            await asyncio.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get_mime_type(self, filename: str) -> str:
        """Get MIME type for file extension."""
        ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
        return MIME_TYPES.get(ext, "application/octet-stream")

    def _validate_file(self, content: bytes, filename: str) -> None:
        """Validate file before transcription."""
        # Check size
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise ConnectorConfigError(
                f"File too large: {len(content) / 1024 / 1024:.1f}MB > {MAX_FILE_SIZE_MB}MB limit",
                connector_name=self.name,
            )

        # Check extension
        ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
        if ext not in ALL_SUPPORTED_EXTENSIONS:
            raise ConnectorConfigError(
                f"Unsupported file type: {ext}. Supported: {', '.join(sorted(ALL_SUPPORTED_EXTENSIONS))}",
                connector_name=self.name,
            )

    async def transcribe(
        self,
        audio_content: bytes,
        filename: str,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> TranscriptionResult:
        """
        Transcribe audio/video content using Whisper API.

        Args:
            audio_content: Raw audio/video bytes
            filename: Original filename (for extension detection)
            prompt: Optional context hint for transcription
            temperature: Sampling temperature (0.0 = deterministic)

        Returns:
            TranscriptionResult with text and segments

        Raises:
            ConnectorConfigError: Invalid file or missing API key
            ConnectorAPIError: API request failed
            ConnectorRateLimitError: Rate limit exceeded
        """
        if not self.is_available:
            if not HTTPX_AVAILABLE:
                raise ConnectorConfigError(
                    "httpx not installed. Install with: pip install httpx",
                    connector_name=self.name,
                )
            raise ConnectorConfigError(
                "No OPENAI_API_KEY configured",
                connector_name=self.name,
            )

        # Validate input
        self._validate_file(audio_content, filename)

        # Rate limit
        await self._rate_limit()

        # Build multipart form data
        mime_type = self._get_mime_type(filename)
        files = {
            "file": (filename, io.BytesIO(audio_content), mime_type),
        }
        data = {
            "model": self.model,
            "response_format": self.response_format,
            "temperature": str(temperature),
        }

        if self.language:
            data["language"] = self.language

        if prompt:
            data["prompt"] = prompt

        # Make API request with retry
        async def do_request():
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    WHISPER_API_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=files,
                    data=data,
                )

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    raise ConnectorRateLimitError(
                        "Whisper API rate limit exceeded",
                        connector_name=self.name,
                        retry_after=float(retry_after) if retry_after else None,
                    )

                response.raise_for_status()
                return response.json()

        try:
            result_data = await self._request_with_retry(do_request, "transcribe")
        except Exception as e:
            logger.error(f"[Whisper] Transcription failed: {e}")
            raise

        # Parse response
        return self._parse_response(result_data, filename)

    def _parse_response(self, data: dict, filename: str) -> TranscriptionResult:
        """Parse Whisper API response into TranscriptionResult."""
        result_id = f"trans_{uuid.uuid4().hex[:12]}"

        # Handle different response formats
        if isinstance(data, str):
            # Plain text response
            return TranscriptionResult(
                id=result_id,
                text=data,
                source_filename=filename,
            )

        text = data.get("text", "")
        language = data.get("language", "")
        duration = data.get("duration", 0.0)

        # Parse segments if available (verbose_json format)
        segments = []
        if "segments" in data:
            for seg in data["segments"]:
                segments.append(
                    TranscriptionSegment(
                        start=seg.get("start", 0.0),
                        end=seg.get("end", 0.0),
                        text=seg.get("text", ""),
                        confidence=seg.get("avg_logprob", 0.0),
                    )
                )

        return TranscriptionResult(
            id=result_id,
            text=text,
            segments=segments,
            language=language,
            duration_seconds=duration,
            source_filename=filename,
            confidence=0.85,  # Whisper is generally reliable
        )

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        chunk_duration_ms: int = 3000,
        filename_base: str = "stream",
    ) -> AsyncIterator[TranscriptionSegment]:
        """
        Stream transcription for live voice input.

        Accumulates audio chunks and transcribes in intervals
        for near-real-time transcription.

        Args:
            audio_chunks: Async iterator of audio bytes
            chunk_duration_ms: Target chunk duration before transcription
            filename_base: Base filename for chunks

        Yields:
            TranscriptionSegment for each transcribed chunk
        """
        if not self.is_available:
            raise ConnectorConfigError(
                "Whisper connector not configured",
                connector_name=self.name,
            )

        buffer = b""
        chunk_count = 0
        total_duration = 0.0

        # Assuming 16kHz, 16-bit mono audio
        bytes_per_second = 16000 * 2
        target_bytes = int(bytes_per_second * chunk_duration_ms / 1000)

        async for chunk in audio_chunks:
            buffer += chunk

            if len(buffer) >= target_bytes:
                # Transcribe accumulated buffer
                chunk_count += 1
                chunk_filename = f"{filename_base}_{chunk_count}.wav"

                try:
                    result = await self.transcribe(buffer, chunk_filename)

                    if result.text.strip():
                        yield TranscriptionSegment(
                            start=total_duration,
                            end=total_duration + result.duration_seconds,
                            text=result.text,
                            confidence=result.confidence,
                        )

                    total_duration += result.duration_seconds

                except Exception as e:
                    logger.warning(f"[Whisper] Stream chunk {chunk_count} failed: {e}")

                buffer = b""

        # Transcribe remaining buffer
        if buffer:
            chunk_count += 1
            chunk_filename = f"{filename_base}_{chunk_count}_final.wav"

            try:
                result = await self.transcribe(buffer, chunk_filename)

                if result.text.strip():
                    yield TranscriptionSegment(
                        start=total_duration,
                        end=total_duration + result.duration_seconds,
                        text=result.text,
                        confidence=result.confidence,
                    )

            except Exception as e:
                logger.warning(f"[Whisper] Final stream chunk failed: {e}")

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> list[Evidence]:
        """
        Search is not applicable for transcription.

        Raises:
            NotImplementedError: This connector doesn't support search
        """
        raise NotImplementedError(
            "WhisperConnector does not support search. Use transcribe() instead."
        )

    async def fetch(self, evidence_id: str) -> Optional[Evidence]:
        """
        Fetch a cached transcription by ID.

        Args:
            evidence_id: Transcription ID

        Returns:
            Evidence object or None if not found/expired
        """
        return self._cache_get(evidence_id)

    def cache_result(self, result: TranscriptionResult) -> Evidence:
        """
        Cache a transcription result and return as Evidence.

        Args:
            result: TranscriptionResult to cache

        Returns:
            Evidence object
        """
        evidence = result.to_evidence()
        self._cache_put(result.id, evidence)
        return evidence


# Convenience functions


def is_supported_audio(filename: str) -> bool:
    """Check if file is a supported audio format."""
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    return ext in SUPPORTED_AUDIO_EXTENSIONS


def is_supported_video(filename: str) -> bool:
    """Check if file is a supported video format."""
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    return ext in SUPPORTED_VIDEO_EXTENSIONS


def is_supported_media(filename: str) -> bool:
    """Check if file is a supported audio or video format."""
    return is_supported_audio(filename) or is_supported_video(filename)


def get_supported_formats() -> dict:
    """Get supported file formats and limits."""
    return {
        "audio": list(SUPPORTED_AUDIO_EXTENSIONS),
        "video": list(SUPPORTED_VIDEO_EXTENSIONS),
        "max_size_mb": MAX_FILE_SIZE_MB,
        "model": WhisperConnector.DEFAULT_MODEL,
    }
