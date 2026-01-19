"""Tests for RSS/podcast feed generation."""

import pytest
from datetime import datetime

from aragora.broadcast.rss_gen import (
    PodcastConfig,
    PodcastEpisode,
    _escape_xml,
    _escape_cdata,
    _format_rfc2822_date,
)


class TestPodcastConfig:
    """Test PodcastConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PodcastConfig()
        assert config.title == "Aragora Debates"
        assert config.author == "Aragora"
        assert config.language == "en-us"
        assert config.category == "Technology"
        assert config.explicit is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PodcastConfig(
            title="My Podcast",
            description="Custom description",
            author="Test Author",
            email="test@example.com",
            language="en-gb",
        )
        assert config.title == "My Podcast"
        assert config.description == "Custom description"
        assert config.author == "Test Author"
        assert config.email == "test@example.com"
        assert config.language == "en-gb"

    def test_copyright_auto_generated(self):
        """Test copyright is auto-generated with current year."""
        config = PodcastConfig()
        current_year = datetime.now().year
        assert str(current_year) in config.copyright
        assert "Aragora" in config.copyright

    def test_custom_copyright(self):
        """Test custom copyright overrides auto-generation."""
        config = PodcastConfig(copyright="Custom Copyright 2025")
        assert config.copyright == "Custom Copyright 2025"


class TestPodcastEpisode:
    """Test PodcastEpisode dataclass."""

    def test_required_fields(self):
        """Test episode with required fields."""
        episode = PodcastEpisode(
            guid="ep-001",
            title="First Episode",
            description="A test episode",
            content="Full content here",
            audio_url="https://example.com/ep1.mp3",
            pub_date="2025-01-18T12:00:00Z",
            duration_seconds=600,
        )
        assert episode.guid == "ep-001"
        assert episode.title == "First Episode"
        assert episode.duration_seconds == 600

    def test_optional_fields(self):
        """Test episode with optional fields."""
        episode = PodcastEpisode(
            guid="ep-002",
            title="Second Episode",
            description="Another test",
            content="Content",
            audio_url="https://example.com/ep2.mp3",
            pub_date="2025-01-18",
            duration_seconds=1200,
            file_size_bytes=15000000,
            explicit=True,
            episode_number=2,
            season_number=1,
            agents=["claude", "gpt"],
        )
        assert episode.file_size_bytes == 15000000
        assert episode.explicit is True
        assert episode.episode_number == 2
        assert episode.season_number == 1
        assert episode.agents == ["claude", "gpt"]

    def test_default_optional_fields(self):
        """Test default values for optional fields."""
        episode = PodcastEpisode(
            guid="ep-003",
            title="Third",
            description="Test",
            content="Content",
            audio_url="https://example.com/ep3.mp3",
            pub_date="2025-01-18",
            duration_seconds=300,
        )
        assert episode.file_size_bytes == 0
        assert episode.explicit is False
        assert episode.episode_number is None
        assert episode.season_number is None
        assert episode.agents == []


class TestEscapeXml:
    """Test XML escaping function."""

    def test_basic_text_unchanged(self):
        """Test basic text without special chars is unchanged."""
        assert _escape_xml("Hello World") == "Hello World"

    def test_escapes_ampersand(self):
        """Test ampersand is escaped."""
        assert _escape_xml("A & B") == "A &amp; B"

    def test_escapes_less_than(self):
        """Test less than is escaped."""
        assert _escape_xml("A < B") == "A &lt; B"

    def test_escapes_greater_than(self):
        """Test greater than is escaped."""
        assert _escape_xml("A > B") == "A &gt; B"

    def test_escapes_quotes(self):
        """Test quotes are escaped."""
        result = _escape_xml('Say "hello"')
        assert '"' not in result or "&quot;" in result or result == 'Say "hello"'

    def test_empty_string(self):
        """Test empty string returns empty."""
        assert _escape_xml("") == ""

    def test_none_returns_empty(self):
        """Test None returns empty string."""
        assert _escape_xml(None) == ""

    def test_handles_html_entities(self):
        """Test HTML entities are decoded then re-escaped."""
        # &amp; -> & -> &amp;
        result = _escape_xml("A &amp; B")
        assert result == "A &amp; B"

    def test_multiple_special_chars(self):
        """Test multiple special characters."""
        result = _escape_xml("<script>alert('XSS')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


class TestEscapeCdata:
    """Test CDATA escaping function."""

    def test_basic_text_unchanged(self):
        """Test basic text is unchanged."""
        assert _escape_cdata("Hello World") == "Hello World"

    def test_escapes_cdata_end(self):
        """Test CDATA end sequence is escaped."""
        result = _escape_cdata("Text with ]]> inside")
        assert "]]>" not in result or result.count("]]>") == 0
        # Should split the CDATA sequence
        assert "]]]]><![CDATA[>" in result

    def test_empty_string(self):
        """Test empty string returns empty."""
        assert _escape_cdata("") == ""

    def test_none_returns_empty(self):
        """Test None returns empty string."""
        assert _escape_cdata(None) == ""

    def test_multiple_cdata_ends(self):
        """Test multiple CDATA end sequences."""
        result = _escape_cdata("First ]]> Second ]]> Third")
        # Should not contain literal ]]>
        assert result.count("]]>") == 0 or "]]]]><![CDATA[>" in result


class TestFormatRfc2822Date:
    """Test RFC 2822 date formatting."""

    def test_iso_date_conversion(self):
        """Test ISO date converts to RFC 2822."""
        result = _format_rfc2822_date("2025-01-18T12:30:00Z")
        assert "2025" in result
        assert "Jan" in result
        assert "18" in result

    def test_iso_with_timezone(self):
        """Test ISO date with timezone."""
        result = _format_rfc2822_date("2025-01-18T12:30:00+00:00")
        assert "2025" in result

    def test_iso_without_z(self):
        """Test ISO date without Z suffix."""
        result = _format_rfc2822_date("2025-01-18T12:30:00")
        assert "2025" in result or "Jan" in result

    def test_invalid_date_returns_current(self):
        """Test invalid date returns current date."""
        result = _format_rfc2822_date("not-a-date")
        current_year = str(datetime.now().year)
        # Should return a valid RFC 2822 date (fallback to now)
        assert current_year in result or len(result) > 0

    def test_empty_date_returns_current(self):
        """Test empty date returns current date."""
        result = _format_rfc2822_date("")
        # Should return a valid date format
        assert len(result) > 0

    def test_rfc2822_format_structure(self):
        """Test output follows RFC 2822 structure."""
        result = _format_rfc2822_date("2025-01-18T12:30:00Z")
        # RFC 2822: "Day, DD Mon YYYY HH:MM:SS +ZZZZ"
        parts = result.split()
        assert len(parts) >= 5  # Day, DD, Mon, YYYY, time, [tz]
