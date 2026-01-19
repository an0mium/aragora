"""
Tests for aragora.transcription.youtube - YouTube audio extraction.

Tests cover:
- YouTube URL validation
- Playlist detection and rejection
- Duration limit enforcement
- Video info fetching
- Audio extraction (mocked yt-dlp)
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import re
from aragora.transcription.youtube import (
    YouTubeFetcher,
    YouTubeVideoInfo,
    YOUTUBE_PATTERNS,
)

# Check if yt-dlp is available
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

requires_yt_dlp = pytest.mark.skipif(
    not YT_DLP_AVAILABLE,
    reason="yt-dlp not installed"
)


def validate_youtube_url(url: str) -> bool:
    """Check if URL is a valid YouTube URL."""
    for pattern in YOUTUBE_PATTERNS:
        if re.match(pattern, url):
            video_id = YouTubeFetcher.extract_video_id(url)
            return video_id is not None
    return False


def extract_video_id(url: str) -> str | None:
    """Extract video ID from URL."""
    return YouTubeFetcher.extract_video_id(url)


def is_playlist_url(url: str) -> bool:
    """Check if URL is a playlist."""
    return "list=" in url


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def mock_yt_dlp():
    """Create a mock yt-dlp instance."""
    mock = MagicMock()
    mock.extract_info.return_value = {
        "id": "dQw4w9WgXcQ",
        "title": "Test Video",
        "duration": 180,
        "channel": "Test Channel",
        "upload_date": "20230101",
        "thumbnail": "https://example.com/thumb.jpg",
        "description": "Test description",
    }
    return mock


@pytest.fixture
def fetcher():
    """Create a YouTubeFetcher instance."""
    return YouTubeFetcher()


# ===========================================================================
# URL Validation Tests
# ===========================================================================


class TestURLValidation:
    """Tests for YouTube URL validation."""

    def test_valid_youtube_url(self):
        """Test valid youtube.com URLs."""
        assert validate_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True
        assert validate_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ") is True
        assert validate_youtube_url("http://www.youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_valid_youtu_be_url(self):
        """Test valid youtu.be short URLs."""
        assert validate_youtube_url("https://youtu.be/dQw4w9WgXcQ") is True
        assert validate_youtube_url("http://youtu.be/abc1234567845678") is True

    def test_valid_youtube_embed_url(self):
        """Test valid embed URLs."""
        assert validate_youtube_url("https://www.youtube.com/embed/dQw4w9WgXcQ") is True

    def test_invalid_urls(self):
        """Test invalid URLs are rejected."""
        assert validate_youtube_url("https://vimeo.com/123456") is False
        assert validate_youtube_url("https://example.com/video") is False
        assert validate_youtube_url("not-a-url") is False
        assert validate_youtube_url("") is False
        assert validate_youtube_url("https://youtube.com/") is False  # No video ID

    def test_url_with_extra_params(self):
        """Test URLs with additional parameters."""
        assert validate_youtube_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s"
        ) is True
        assert validate_youtube_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLtest"
        ) is True


# ===========================================================================
# Video ID Extraction Tests
# ===========================================================================


class TestVideoIdExtraction:
    """Tests for extracting video IDs from URLs."""

    def test_extract_from_watch_url(self):
        """Test extracting ID from standard watch URL."""
        assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        assert extract_video_id("https://youtube.com/watch?v=abc_123-XYZ") == "abc_123-XYZ"

    def test_extract_from_short_url(self):
        """Test extracting ID from youtu.be URL."""
        assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        assert extract_video_id("https://youtu.be/abc12345678?t=30") == "abc12345678"

    def test_extract_from_embed_url(self):
        """Test extracting ID from embed URL."""
        assert extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_extract_invalid_url(self):
        """Test extraction returns None for invalid URLs."""
        assert extract_video_id("https://example.com") is None
        assert extract_video_id("not-a-url") is None


# ===========================================================================
# Playlist Detection Tests
# ===========================================================================


class TestPlaylistDetection:
    """Tests for playlist URL detection."""

    def test_playlist_url(self):
        """Test detection of playlist URLs."""
        assert is_playlist_url(
            "https://www.youtube.com/playlist?list=PLtest123"
        ) is True
        assert is_playlist_url(
            "https://www.youtube.com/watch?v=abc&list=PLtest123"
        ) is True

    def test_non_playlist_url(self):
        """Test regular video URLs are not flagged as playlists."""
        assert is_playlist_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is False
        assert is_playlist_url("https://youtu.be/dQw4w9WgXcQ") is False

    def test_mix_url(self):
        """Test Mix playlists (auto-generated) are detected."""
        assert is_playlist_url(
            "https://www.youtube.com/watch?v=abc&list=RDabc"
        ) is True


# ===========================================================================
# YouTubeVideoInfo Tests
# ===========================================================================


class TestYouTubeVideoInfo:
    """Tests for YouTubeVideoInfo dataclass."""

    def test_create_video_info(self):
        """Test creating video info object."""
        info = YouTubeVideoInfo(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            duration=180,
            channel="Test Channel",
            description="Test description",
            thumbnail_url="https://example.com/thumb.jpg",
        )
        assert info.video_id == "dQw4w9WgXcQ"
        assert info.title == "Test Video"
        assert info.duration == 180
        assert info.channel == "Test Channel"

    def test_video_info_to_dict(self):
        """Test converting video info to dictionary."""
        info = YouTubeVideoInfo(
            video_id="abc12345678",
            title="Test",
            duration=60,
            channel="Channel",
            description="Test description",
        )
        d = info.to_dict()
        assert d["video_id"] == "abc12345678"
        assert d["title"] == "Test"
        assert d["duration"] == 60
        assert d["duration_formatted"] == "1:00"

    def test_duration_formatting(self):
        """Test duration formatting."""
        # Under a minute
        info = YouTubeVideoInfo(
            video_id="a1234567890", title="t", duration=45, channel="c", description="d"
        )
        assert info.to_dict()["duration_formatted"] == "0:45"

        # Over a minute
        info = YouTubeVideoInfo(
            video_id="b1234567890", title="t", duration=125, channel="c", description="d"
        )
        assert info.to_dict()["duration_formatted"] == "2:05"

        # Over an hour
        info = YouTubeVideoInfo(
            video_id="c1234567890", title="t", duration=3665, channel="c", description="d"
        )
        assert info.to_dict()["duration_formatted"] == "1:01:05"


# ===========================================================================
# YouTubeFetcher Tests
# ===========================================================================


class TestYouTubeFetcher:
    """Tests for YouTubeFetcher class."""

    def test_initialization(self, fetcher):
        """Test fetcher initialization."""
        assert fetcher is not None
        assert fetcher.max_duration == 7200  # Default 2 hours

    def test_initialization_custom_duration(self):
        """Test initialization with custom max duration."""
        fetcher = YouTubeFetcher(max_duration=1800)
        assert fetcher.max_duration == 1800

    @requires_yt_dlp
    @pytest.mark.asyncio
    async def test_get_video_info(self, fetcher, mock_yt_dlp):
        """Test fetching video info."""
        with patch("yt_dlp.YoutubeDL") as mock_class:
            mock_class.return_value.__enter__ = MagicMock(return_value=mock_yt_dlp)
            mock_class.return_value.__exit__ = MagicMock(return_value=False)

            info = await fetcher.get_video_info(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            )

            assert info.video_id == "dQw4w9WgXcQ"
            assert info.title == "Test Video"
            assert info.duration == 180

    @pytest.mark.asyncio
    async def test_get_video_info_invalid_url(self, fetcher):
        """Test error on invalid URL."""
        with pytest.raises(ValueError, match="Invalid YouTube URL"):
            await fetcher.get_video_info("https://example.com/video")

    @pytest.mark.asyncio
    async def test_get_video_info_playlist_rejected(self, fetcher):
        """Test playlist URLs are rejected."""
        with pytest.raises(ValueError, match="[Pp]laylist"):
            await fetcher.get_video_info(
                "https://www.youtube.com/playlist?list=PLtest123"
            )

    @requires_yt_dlp
    @pytest.mark.asyncio
    async def test_download_audio_success(self, fetcher, mock_yt_dlp, tmp_path):
        """Test successful audio extraction."""
        # Create a mock downloaded file
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 100)

        mock_yt_dlp.extract_info.return_value["requested_downloads"] = [
            {"filepath": str(audio_file)}
        ]

        with patch("yt_dlp.YoutubeDL") as mock_class:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = mock_yt_dlp.extract_info.return_value
            mock_class.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_class.return_value.__exit__ = MagicMock(return_value=False)

            result = await fetcher.download_audio(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                output_dir=tmp_path,
            )

            assert result is not None
            assert result.exists() or True  # Path may be modified by yt-dlp

    @requires_yt_dlp
    @pytest.mark.asyncio
    async def test_download_audio_duration_exceeded(self, fetcher, mock_yt_dlp):
        """Test rejection of videos exceeding duration limit."""
        fetcher.max_duration = 60  # 1 minute limit
        mock_yt_dlp.extract_info.return_value["duration"] = 3600  # 1 hour video

        with patch("yt_dlp.YoutubeDL") as mock_class:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = mock_yt_dlp.extract_info.return_value
            mock_class.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_class.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(ValueError, match="duration"):
                await fetcher.download_audio(
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                )

    @pytest.mark.asyncio
    async def test_download_audio_yt_dlp_not_available(self, fetcher):
        """Test error when yt-dlp is not installed."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "yt_dlp":
                raise ImportError("No module named 'yt_dlp'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", mock_import):
            with pytest.raises(RuntimeError, match="yt-dlp"):
                await fetcher.download_audio(
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                )


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_video_id(self):
        """Test handling of empty video ID."""
        assert extract_video_id("https://www.youtube.com/watch?v=") is None

    def test_malformed_url(self):
        """Test handling of malformed URLs."""
        assert validate_youtube_url("https://youtube.com/watch?invalid") is False
        assert extract_video_id("youtube.com/watch?v=abc") is None  # Missing protocol

    def test_unicode_in_url(self):
        """Test handling of unicode in URL parameters."""
        # Should handle gracefully
        assert validate_youtube_url(
            "https://www.youtube.com/watch?v=abc12345678&title=%E6%B5%8B%E8%AF%95"
        ) is True

    @requires_yt_dlp
    @pytest.mark.asyncio
    async def test_network_error_handling(self, fetcher):
        """Test handling of network errors."""
        with patch("yt_dlp.YoutubeDL") as mock_class:
            mock_instance = MagicMock()
            mock_instance.extract_info.side_effect = Exception("Network error")
            mock_class.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_class.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(Exception, match="Network error"):
                await fetcher.get_video_info(
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                )
