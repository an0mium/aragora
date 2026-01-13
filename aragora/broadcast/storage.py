"""
Audio file storage for debate broadcasts.

Provides persistent storage for generated audio files with metadata tracking.
Audio files are stored in .nomic/audio/ with accompanying JSON metadata.
"""

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Security: Path Traversal Protection
# =============================================================================

# Pattern for valid debate IDs (alphanumeric, hyphens, underscores only)
VALID_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

# Allowed audio formats (whitelist)
ALLOWED_AUDIO_FORMATS = frozenset(["mp3", "wav", "m4a", "ogg", "flac", "aac"])

# Maximum file size (100 MB - reasonable for debate audio)
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

# Audio file magic bytes for validation
# Format: magic_bytes -> (format_name, offset)
AUDIO_MAGIC_BYTES = {
    b"\xff\xfb": "mp3",  # MP3 frame sync
    b"\xff\xfa": "mp3",  # MP3 frame sync (alt)
    b"\xff\xf3": "mp3",  # MP3 frame sync (alt)
    b"\xff\xf2": "mp3",  # MP3 frame sync (alt)
    b"ID3": "mp3",  # MP3 with ID3 tag
    b"RIFF": "wav",  # WAV container
    b"fLaC": "flac",  # FLAC
    b"OggS": "ogg",  # Ogg container
    b"\x00\x00\x00": "m4a",  # M4A/AAC (partial - needs ftyp check)
}


def _validate_audio_magic(data: bytes, claimed_format: str) -> bool:
    """Validate audio file magic bytes match claimed format.

    Args:
        data: Audio file data (at least first 12 bytes)
        claimed_format: The format the file claims to be

    Returns:
        True if magic bytes are consistent with format, False otherwise
    """
    if len(data) < 12:
        return False  # Too small to be valid audio

    # Check common magic bytes
    for magic, fmt in AUDIO_MAGIC_BYTES.items():
        if data.startswith(magic):
            # MP3 magic matches multiple patterns
            if fmt == "mp3" and claimed_format == "mp3":
                return True
            if fmt == claimed_format:
                return True

    # M4A/AAC special case - check for 'ftyp' at offset 4
    if claimed_format in ("m4a", "aac"):
        if data[4:8] == b"ftyp":
            return True

    # WAV special case - verify WAVE identifier
    if claimed_format == "wav" and data[:4] == b"RIFF":
        if data[8:12] == b"WAVE":
            return True

    # Allow if we couldn't detect (some formats are tricky)
    # But log for monitoring
    logger.debug(f"Could not validate magic bytes for claimed format {claimed_format}")
    return True  # Permissive for now - log helps identify issues


def _validate_debate_id(debate_id: str) -> bool:
    """Validate a debate ID to prevent path traversal.

    Args:
        debate_id: The debate ID to validate

    Returns:
        True if the ID is safe, False otherwise
    """
    if not debate_id:
        return False

    # Must match safe pattern (no path separators, no dots)
    if not VALID_ID_PATTERN.match(debate_id):
        return False

    # Reasonable length limit
    if len(debate_id) > 128:
        return False

    return True


@dataclass
class AudioMetadata:
    """Metadata for a stored audio file."""

    debate_id: str
    filename: str
    format: str  # mp3, wav, etc.
    duration_seconds: Optional[int] = None
    file_size_bytes: Optional[int] = None
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    task_summary: Optional[str] = None  # Brief debate topic
    agents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "debate_id": self.debate_id,
            "filename": self.filename,
            "format": self.format,
            "duration_seconds": self.duration_seconds,
            "file_size_bytes": self.file_size_bytes,
            "generated_at": self.generated_at,
            "task_summary": self.task_summary,
            "agents": self.agents,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AudioMetadata":
        """Create from dictionary."""
        return cls(
            debate_id=data["debate_id"],
            filename=data["filename"],
            format=data.get("format", "mp3"),
            duration_seconds=data.get("duration_seconds"),
            file_size_bytes=data.get("file_size_bytes"),
            generated_at=data.get("generated_at", datetime.now().isoformat()),
            task_summary=data.get("task_summary"),
            agents=data.get("agents", []),
        )


class AudioFileStore:
    """
    Persistent storage for debate audio files.

    Audio files are stored in .nomic/audio/ with JSON metadata files.
    Provides methods for saving, retrieving, listing, and deleting audio.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize audio file store.

        Args:
            storage_dir: Directory for audio storage. Defaults to .nomic/audio/
        """
        if storage_dir is None:
            storage_dir = Path.cwd() / ".nomic" / "audio"
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for metadata
        self._cache: dict[str, AudioMetadata] = {}

    def _audio_path(self, debate_id: str, format: str = "mp3") -> Optional[Path]:
        """Get path for an audio file.

        Args:
            debate_id: The debate ID
            format: Audio format extension

        Returns:
            Path if debate_id is valid, None otherwise (path traversal protection)
        """
        if not _validate_debate_id(debate_id):
            logger.warning(f"Invalid debate ID rejected: {debate_id!r}")
            return None

        # Validate format as well
        if not re.match(r"^[a-zA-Z0-9]+$", format):
            logger.warning(f"Invalid format rejected: {format!r}")
            return None

        path = self.storage_dir / f"{debate_id}.{format}"

        # Double-check path is within storage directory
        try:
            path.resolve().relative_to(self.storage_dir.resolve())
        except ValueError:
            logger.warning(f"Path traversal attempt: {debate_id!r}.{format}")
            return None

        return path

    def _metadata_path(self, debate_id: str) -> Optional[Path]:
        """Get path for metadata JSON file.

        Returns:
            Path if debate_id is valid, None otherwise (path traversal protection)
        """
        if not _validate_debate_id(debate_id):
            logger.warning(f"Invalid debate ID rejected: {debate_id!r}")
            return None

        path = self.storage_dir / f"{debate_id}.json"

        # Double-check path is within storage directory
        try:
            path.resolve().relative_to(self.storage_dir.resolve())
        except ValueError:
            logger.warning(f"Path traversal attempt: {debate_id!r}")
            return None

        return path

    def save(
        self,
        debate_id: str,
        audio_path: Path,
        format: str = "mp3",
        duration_seconds: Optional[int] = None,
        task_summary: Optional[str] = None,
        agents: Optional[list[str]] = None,
    ) -> Optional[Path]:
        """
        Save an audio file to the store.

        Args:
            debate_id: Unique debate identifier
            audio_path: Path to the source audio file
            format: Audio format (mp3, wav, etc.)
            duration_seconds: Audio duration in seconds
            task_summary: Brief debate topic for metadata
            agents: List of participating agents

        Returns:
            Path to the stored audio file, or None if validation fails
        """
        # Validate format against whitelist
        format_lower = format.lower()
        if format_lower not in ALLOWED_AUDIO_FORMATS:
            logger.error(
                f"Rejected audio format '{format}': not in whitelist {ALLOWED_AUDIO_FORMATS}"
            )
            return None

        dest_path = self._audio_path(debate_id, format_lower)
        if dest_path is None:
            logger.error(f"Cannot save audio: invalid debate_id {debate_id!r}")
            return None

        # Validate file size
        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error(f"Source audio file not found: {audio_path}")
            return None

        file_size = audio_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            logger.error(
                f"Audio file too large: {file_size:,} bytes > {MAX_FILE_SIZE_BYTES:,} bytes limit"
            )
            return None

        # Validate magic bytes
        with open(audio_path, "rb") as f:
            header = f.read(12)
        if not _validate_audio_magic(header, format_lower):
            logger.warning(f"Audio magic bytes don't match format {format_lower} for {audio_path}")
            # Continue anyway - logged for monitoring

        # Copy audio file to storage
        shutil.copy2(audio_path, dest_path)

        # Get file size
        file_size = dest_path.stat().st_size

        # Create metadata
        metadata = AudioMetadata(
            debate_id=debate_id,
            filename=dest_path.name,
            format=format,
            duration_seconds=duration_seconds,
            file_size_bytes=file_size,
            task_summary=task_summary,
            agents=agents or [],
        )

        # Save metadata to JSON
        metadata_path = self._metadata_path(debate_id)
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Update cache
        self._cache[debate_id] = metadata

        logger.info(f"Saved audio for debate {debate_id}: {dest_path} ({file_size} bytes)")
        return dest_path

    def save_bytes(
        self,
        debate_id: str,
        audio_data: bytes,
        format: str = "mp3",
        duration_seconds: Optional[int] = None,
        task_summary: Optional[str] = None,
        agents: Optional[list[str]] = None,
    ) -> Optional[Path]:
        """
        Save audio data directly from bytes.

        Args:
            debate_id: Unique debate identifier
            audio_data: Raw audio bytes
            format: Audio format (mp3, wav, etc.)
            duration_seconds: Audio duration in seconds
            task_summary: Brief debate topic for metadata
            agents: List of participating agents

        Returns:
            Path to the stored audio file, or None if validation fails
        """
        # Validate format against whitelist
        format_lower = format.lower()
        if format_lower not in ALLOWED_AUDIO_FORMATS:
            logger.error(
                f"Rejected audio format '{format}': not in whitelist {ALLOWED_AUDIO_FORMATS}"
            )
            return None

        # Validate size
        if len(audio_data) > MAX_FILE_SIZE_BYTES:
            logger.error(
                f"Audio data too large: {len(audio_data):,} bytes > {MAX_FILE_SIZE_BYTES:,} bytes limit"
            )
            return None

        # Validate magic bytes
        if not _validate_audio_magic(audio_data, format_lower):
            logger.warning(f"Audio magic bytes don't match claimed format {format_lower}")
            # Continue anyway - logged for monitoring

        dest_path = self._audio_path(debate_id, format_lower)
        if dest_path is None:
            logger.error(f"Cannot save audio: invalid debate_id {debate_id!r}")
            return None

        # Write audio data
        with open(dest_path, "wb") as f:
            f.write(audio_data)

        # Create metadata
        metadata = AudioMetadata(
            debate_id=debate_id,
            filename=dest_path.name,
            format=format,
            duration_seconds=duration_seconds,
            file_size_bytes=len(audio_data),
            task_summary=task_summary,
            agents=agents or [],
        )

        # Save metadata to JSON
        metadata_path = self._metadata_path(debate_id)
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Update cache
        self._cache[debate_id] = metadata

        logger.info(f"Saved audio for debate {debate_id}: {dest_path} ({len(audio_data)} bytes)")
        return dest_path

    def get_path(self, debate_id: str) -> Optional[Path]:
        """
        Get the path to an audio file.

        Args:
            debate_id: Debate identifier

        Returns:
            Path to audio file if exists, None otherwise (also None for invalid debate_id)
        """
        # Check common formats
        for ext in ["mp3", "wav", "m4a", "ogg"]:
            path = self._audio_path(debate_id, ext)
            if path is not None and path.exists():
                return path
        return None

    def get_metadata(self, debate_id: str) -> Optional[AudioMetadata]:
        """
        Get metadata for an audio file.

        Args:
            debate_id: Debate identifier

        Returns:
            AudioMetadata if exists, None otherwise (also None for invalid debate_id)
        """
        # Check cache first
        if debate_id in self._cache:
            return self._cache[debate_id]

        # Load from disk (returns None for invalid IDs - path traversal protection)
        metadata_path = self._metadata_path(debate_id)
        if metadata_path is None or not metadata_path.exists():
            return None

        try:
            with open(metadata_path) as f:
                data = json.load(f)
            metadata = AudioMetadata.from_dict(data)
            self._cache[debate_id] = metadata
            return metadata
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load metadata for {debate_id}: {e}")
            return None

    def exists(self, debate_id: str) -> bool:
        """Check if audio exists for a debate."""
        return self.get_path(debate_id) is not None

    def delete(self, debate_id: str) -> bool:
        """
        Delete audio and metadata for a debate.

        Args:
            debate_id: Debate identifier

        Returns:
            True if deleted, False if not found or invalid debate_id
        """
        # Validate debate_id first (path traversal protection)
        if not _validate_debate_id(debate_id):
            logger.warning(f"Cannot delete: invalid debate_id {debate_id!r}")
            return False

        # Remove from cache
        self._cache.pop(debate_id, None)

        deleted = False

        # Delete audio file
        audio_path = self.get_path(debate_id)
        if audio_path and audio_path.exists():
            audio_path.unlink()
            deleted = True

        # Delete metadata
        metadata_path = self._metadata_path(debate_id)
        if metadata_path is not None and metadata_path.exists():
            metadata_path.unlink()
            deleted = True

        if deleted:
            logger.info(f"Deleted audio for debate {debate_id}")

        return deleted

    def list_all(self) -> list[dict]:
        """
        List all stored audio files with metadata.

        Returns:
            List of metadata dictionaries
        """
        results = []
        for metadata_path in self.storage_dir.glob("*.json"):
            try:
                with open(metadata_path) as f:
                    data = json.load(f)

                # Verify audio file exists
                debate_id = data.get("debate_id")
                if debate_id and self.get_path(debate_id):
                    results.append(
                        {
                            "debate_id": debate_id,
                            "filename": data.get("filename"),
                            "format": data.get("format", "mp3"),
                            "duration_seconds": data.get("duration_seconds"),
                            "file_size_bytes": data.get("file_size_bytes"),
                            "generated_at": data.get("generated_at"),
                            "task_summary": data.get("task_summary", "")[:100],
                        }
                    )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to read metadata {metadata_path}: {e}")
                continue

        # Sort by generated_at descending
        results.sort(key=lambda x: x.get("generated_at", ""), reverse=True)
        return results

    def get_total_size(self) -> int:
        """Get total size of all stored audio files in bytes."""
        total = 0
        for audio_path in self.storage_dir.glob("*.mp3"):
            total += audio_path.stat().st_size
        for audio_path in self.storage_dir.glob("*.wav"):
            total += audio_path.stat().st_size
        return total

    def cleanup_orphaned(self) -> int:
        """
        Remove metadata files without corresponding audio.

        Returns:
            Number of orphaned files removed
        """
        removed = 0
        for metadata_path in self.storage_dir.glob("*.json"):
            try:
                with open(metadata_path) as f:
                    data = json.load(f)
                debate_id = data.get("debate_id")
                if debate_id and not self.get_path(debate_id):
                    metadata_path.unlink()
                    self._cache.pop(debate_id, None)
                    removed += 1
                    logger.info(f"Removed orphaned metadata: {metadata_path}")
            except (json.JSONDecodeError, KeyError):
                # Remove corrupted metadata files
                metadata_path.unlink()
                removed += 1

        return removed
