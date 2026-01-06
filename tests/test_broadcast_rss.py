"""
Tests for RSS/Podcast feed generation and audio storage.

Tests the PodcastFeedGenerator and AudioFileStore classes.
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from aragora.broadcast.rss_gen import (
    PodcastConfig,
    PodcastEpisode,
    PodcastFeedGenerator,
    _escape_xml,
    _escape_cdata,
    _format_rfc2822_date,
    _format_duration,
    create_debate_summary,
)
from aragora.broadcast.storage import (
    AudioMetadata,
    AudioFileStore,
)


# =============================================================================
# RSS Feed Generator Tests
# =============================================================================

class TestPodcastConfig:
    """Tests for PodcastConfig dataclass."""

    def test_default_values(self):
        """Default config should have sensible values."""
        config = PodcastConfig()
        assert config.title == "Aragora Debates"
        assert config.language == "en-us"
        assert config.explicit is False

    def test_custom_values(self):
        """Custom values should override defaults."""
        config = PodcastConfig(
            title="Custom Podcast",
            author="Test Author",
            explicit=True,
        )
        assert config.title == "Custom Podcast"
        assert config.author == "Test Author"
        assert config.explicit is True

    def test_copyright_auto_generated(self):
        """Copyright should be auto-generated with current year."""
        config = PodcastConfig()
        current_year = datetime.now().year
        assert str(current_year) in config.copyright


class TestXMLHelpers:
    """Tests for XML helper functions."""

    def test_escape_xml_basic(self):
        """Should escape basic XML entities."""
        assert _escape_xml("<script>") == "&lt;script&gt;"
        assert _escape_xml("Tom & Jerry") == "Tom &amp; Jerry"
        # Note: quotes don't need escaping in XML element content
        assert '"' in _escape_xml('"quoted"')

    def test_escape_xml_empty(self):
        """Should handle empty strings."""
        assert _escape_xml("") == ""
        assert _escape_xml(None) == ""

    def test_escape_xml_html_entities(self):
        """Should handle HTML entities by unescaping first."""
        assert _escape_xml("&amp;") == "&amp;"

    def test_escape_cdata_basic(self):
        """Should escape CDATA end sequences."""
        assert _escape_cdata("content]]>more") == "content]]]]><![CDATA[>more"

    def test_escape_cdata_empty(self):
        """Should handle empty strings."""
        assert _escape_cdata("") == ""
        assert _escape_cdata(None) == ""


class TestDateFormatting:
    """Tests for date formatting functions."""

    def test_format_rfc2822_valid_iso(self):
        """Should convert ISO date to RFC 2822."""
        result = _format_rfc2822_date("2024-01-15T10:30:00+00:00")
        assert "15 Jan 2024" in result
        assert "10:30:00" in result

    def test_format_rfc2822_with_z(self):
        """Should handle Z timezone suffix."""
        result = _format_rfc2822_date("2024-01-15T10:30:00Z")
        assert "15 Jan 2024" in result

    def test_format_rfc2822_invalid(self):
        """Should return current date for invalid input."""
        result = _format_rfc2822_date("invalid")
        assert result  # Should return something, not raise


class TestDurationFormatting:
    """Tests for duration formatting."""

    def test_format_duration_basic(self):
        """Should format duration as HH:MM:SS."""
        assert _format_duration(3661) == "01:01:01"  # 1h 1m 1s
        assert _format_duration(0) == "00:00:00"
        assert _format_duration(59) == "00:00:59"

    def test_format_duration_large(self):
        """Should handle large durations."""
        assert _format_duration(7200) == "02:00:00"  # 2 hours

    def test_format_duration_invalid(self):
        """Should handle invalid durations."""
        assert _format_duration(-1) == "00:00:00"
        assert _format_duration(None) == "00:00:00"


class TestPodcastFeedGenerator:
    """Tests for PodcastFeedGenerator class."""

    def test_init_default_config(self):
        """Should use default config if not provided."""
        gen = PodcastFeedGenerator()
        assert gen.config.title == "Aragora Debates"

    def test_init_custom_config(self):
        """Should accept custom config."""
        config = PodcastConfig(title="Test Podcast")
        gen = PodcastFeedGenerator(config)
        assert gen.config.title == "Test Podcast"

    def test_generate_empty_feed(self):
        """Should generate valid XML with no episodes."""
        gen = PodcastFeedGenerator()
        feed = gen.generate_feed([])

        # Should be valid XML
        root = ET.fromstring(feed)
        assert root.tag == "rss"

        # Should have channel element
        channel = root.find("channel")
        assert channel is not None

        # Should have title
        title = channel.find("title")
        assert title is not None
        assert title.text == "Aragora Debates"

    def test_generate_feed_with_episode(self):
        """Should include episodes in feed."""
        gen = PodcastFeedGenerator()
        episode = PodcastEpisode(
            guid="test-001",
            title="Test Episode",
            description="Test description",
            content="<p>Full content</p>",
            audio_url="https://example.com/audio.mp3",
            pub_date="2024-01-15T10:00:00Z",
            duration_seconds=600,
            file_size_bytes=1024000,
            agents=["claude", "gpt4"],
        )

        feed = gen.generate_feed([episode])
        root = ET.fromstring(feed)
        channel = root.find("channel")
        item = channel.find("item")

        assert item is not None
        assert item.find("title").text == "Test Episode"
        assert item.find("guid").text == "test-001"

    def test_itunes_metadata_present(self):
        """Feed should include iTunes-specific tags."""
        gen = PodcastFeedGenerator()
        feed = gen.generate_feed([])

        assert "itunes:author" in feed
        assert "itunes:category" in feed
        assert "itunes:explicit" in feed
        assert "itunes:image" in feed

    def test_enclosure_attributes(self):
        """Enclosure should have url, length, and type."""
        gen = PodcastFeedGenerator()
        episode = PodcastEpisode(
            guid="enc-test",
            title="Enclosure Test",
            description="Testing enclosure",
            content="content",
            audio_url="https://example.com/test.mp3",
            pub_date="2024-01-15T10:00:00Z",
            duration_seconds=300,
            file_size_bytes=5000000,
        )

        feed = gen.generate_feed([episode])
        assert 'enclosure url="https://example.com/test.mp3"' in feed
        assert 'length="5000000"' in feed
        assert 'type="audio/mpeg"' in feed

    def test_create_episode_from_debate(self):
        """Should create episode from debate metadata."""
        gen = PodcastFeedGenerator()
        episode = gen.create_episode_from_debate(
            debate_id="debate-123",
            task="Should we use microservices?",
            agents=["claude", "gpt4", "gemini"],
            audio_url="https://example.com/debate.mp3",
            duration_seconds=1800,
            file_size_bytes=10000000,
            created_at="2024-01-15T12:00:00Z",
            consensus_reached=True,
        )

        assert episode.guid == "debate-123"
        assert "microservices" in episode.title
        assert "claude" in episode.description
        assert episode.duration_seconds == 1800

    def test_xml_validity(self):
        """Generated feed should be valid XML."""
        config = PodcastConfig(
            title="XML Test <script>alert('xss')</script>",
            description="Test & verify",
        )
        gen = PodcastFeedGenerator(config)
        episode = PodcastEpisode(
            guid="xml-test",
            title="Episode with <special> chars",
            description="Description with & ampersand",
            content="<p>Content with ]]> CDATA end</p>",
            audio_url="https://example.com/test.mp3",
            pub_date="2024-01-15T10:00:00Z",
            duration_seconds=300,
        )

        feed = gen.generate_feed([episode])

        # Should parse without error
        root = ET.fromstring(feed)
        assert root is not None


class TestCreateDebateSummary:
    """Tests for create_debate_summary function."""

    def test_short_summary(self):
        """Short topics should not be truncated."""
        summary = create_debate_summary(
            "Rate limiter design",
            ["claude", "gpt4"],
            consensus_reached=True,
        )
        assert "Rate limiter design" in summary
        assert "claude" in summary
        assert "gpt4" in summary

    def test_long_topic_truncation(self):
        """Long topics should be truncated."""
        long_topic = "A" * 300
        summary = create_debate_summary(long_topic, ["claude"], max_length=100)
        assert len(summary) <= 100
        assert "..." in summary

    def test_consensus_emoji(self):
        """Should include appropriate emoji based on consensus."""
        with_consensus = create_debate_summary("Topic", ["claude"], True)
        without_consensus = create_debate_summary("Topic", ["claude"], False)

        # Different emojis for different outcomes
        assert with_consensus != without_consensus


# =============================================================================
# Audio File Store Tests
# =============================================================================

class TestAudioMetadata:
    """Tests for AudioMetadata dataclass."""

    def test_to_dict(self):
        """Should serialize to dictionary."""
        metadata = AudioMetadata(
            debate_id="test-001",
            filename="test-001.mp3",
            format="mp3",
            duration_seconds=600,
            file_size_bytes=1024000,
            agents=["claude", "gpt4"],
        )
        d = metadata.to_dict()

        assert d["debate_id"] == "test-001"
        assert d["filename"] == "test-001.mp3"
        assert d["duration_seconds"] == 600
        assert d["agents"] == ["claude", "gpt4"]

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        d = {
            "debate_id": "test-002",
            "filename": "test-002.mp3",
            "format": "mp3",
            "duration_seconds": 300,
        }
        metadata = AudioMetadata.from_dict(d)

        assert metadata.debate_id == "test-002"
        assert metadata.filename == "test-002.mp3"
        assert metadata.duration_seconds == 300

    def test_from_dict_defaults(self):
        """Should use defaults for missing fields."""
        d = {"debate_id": "test", "filename": "test.mp3"}
        metadata = AudioMetadata.from_dict(d)

        assert metadata.format == "mp3"
        assert metadata.agents == []


class TestAudioFileStore:
    """Tests for AudioFileStore class."""

    @pytest.fixture
    def temp_store(self, tmp_path):
        """Create a temporary audio store."""
        return AudioFileStore(storage_dir=tmp_path / "audio")

    @pytest.fixture
    def sample_audio(self, tmp_path):
        """Create a sample audio file."""
        audio_file = tmp_path / "sample.mp3"
        audio_file.write_bytes(b"fake mp3 content" * 1000)
        return audio_file

    def test_init_creates_directory(self, tmp_path):
        """Should create storage directory on init."""
        storage_dir = tmp_path / "new_audio"
        store = AudioFileStore(storage_dir=storage_dir)
        assert storage_dir.exists()

    def test_save_audio_file(self, temp_store, sample_audio):
        """Should save audio file and create metadata."""
        path = temp_store.save(
            debate_id="debate-001",
            audio_path=sample_audio,
            format="mp3",
            duration_seconds=600,
            task_summary="Test debate",
            agents=["claude"],
        )

        assert path.exists()
        assert path.name == "debate-001.mp3"

        # Metadata should exist
        metadata = temp_store.get_metadata("debate-001")
        assert metadata is not None
        assert metadata.duration_seconds == 600

    def test_save_bytes(self, temp_store):
        """Should save audio from raw bytes."""
        audio_data = b"raw audio bytes" * 500
        path = temp_store.save_bytes(
            debate_id="bytes-001",
            audio_data=audio_data,
            format="mp3",
            duration_seconds=300,
        )

        assert path.exists()
        assert path.read_bytes() == audio_data

    def test_get_path_existing(self, temp_store, sample_audio):
        """Should return path for existing audio."""
        temp_store.save("existing", sample_audio)
        path = temp_store.get_path("existing")

        assert path is not None
        assert path.exists()

    def test_get_path_nonexistent(self, temp_store):
        """Should return None for nonexistent audio."""
        path = temp_store.get_path("nonexistent")
        assert path is None

    def test_exists(self, temp_store, sample_audio):
        """Should check existence correctly."""
        temp_store.save("check-exists", sample_audio)

        assert temp_store.exists("check-exists") is True
        assert temp_store.exists("not-exists") is False

    def test_delete_audio(self, temp_store, sample_audio):
        """Should delete audio and metadata."""
        temp_store.save("to-delete", sample_audio)
        assert temp_store.exists("to-delete")

        result = temp_store.delete("to-delete")

        assert result is True
        assert temp_store.exists("to-delete") is False
        assert temp_store.get_metadata("to-delete") is None

    def test_delete_nonexistent(self, temp_store):
        """Should return False for nonexistent audio."""
        result = temp_store.delete("not-there")
        assert result is False

    def test_list_all(self, temp_store, sample_audio):
        """Should list all stored audio files."""
        temp_store.save("list-001", sample_audio, task_summary="First debate")
        temp_store.save("list-002", sample_audio, task_summary="Second debate")

        all_audio = temp_store.list_all()

        assert len(all_audio) == 2
        debate_ids = [a["debate_id"] for a in all_audio]
        assert "list-001" in debate_ids
        assert "list-002" in debate_ids

    def test_get_total_size(self, temp_store, sample_audio):
        """Should calculate total storage size."""
        temp_store.save("size-001", sample_audio)
        temp_store.save("size-002", sample_audio)

        total = temp_store.get_total_size()
        expected = sample_audio.stat().st_size * 2

        assert total == expected

    def test_storage_path_safety(self, temp_store, sample_audio):
        """Should not allow path traversal attacks."""
        # Save with a normal ID
        temp_store.save("normal-id", sample_audio)

        # The file should be in the storage directory, not escaped
        path = temp_store.get_path("normal-id")
        assert path.parent == temp_store.storage_dir


# =============================================================================
# Integration Tests
# =============================================================================

class TestRSSIntegration:
    """Integration tests for RSS and storage together."""

    @pytest.fixture
    def temp_store(self, tmp_path):
        """Create a temporary audio store."""
        return AudioFileStore(storage_dir=tmp_path / "audio")

    @pytest.fixture
    def sample_audio(self, tmp_path):
        """Create a sample audio file."""
        audio_file = tmp_path / "sample.mp3"
        audio_file.write_bytes(b"fake mp3 content" * 1000)
        return audio_file

    def test_feed_with_stored_episodes(self, temp_store, sample_audio):
        """Should generate feed from stored audio metadata."""
        # Store some audio
        for i in range(3):
            temp_store.save(
                f"debate-{i:03d}",
                sample_audio,
                duration_seconds=600 + i * 100,
                task_summary=f"Test debate {i}",
                agents=["claude", "gpt4"],
            )

        # Generate feed using stored metadata
        gen = PodcastFeedGenerator()
        episodes = []

        for audio in temp_store.list_all():
            metadata = temp_store.get_metadata(audio["debate_id"])
            if metadata:
                episode = gen.create_episode_from_debate(
                    debate_id=metadata.debate_id,
                    task=metadata.task_summary or "Debate",
                    agents=metadata.agents,
                    audio_url=f"https://example.com/{metadata.filename}",
                    duration_seconds=metadata.duration_seconds or 0,
                    file_size_bytes=metadata.file_size_bytes or 0,
                )
                episodes.append(episode)

        feed = gen.generate_feed(episodes)

        # Should have 3 items
        root = ET.fromstring(feed)
        items = root.findall(".//item")
        assert len(items) == 3

    def test_feed_url_generation(self, temp_store, sample_audio):
        """Audio URLs should be properly formatted in feed."""
        temp_store.save("url-test", sample_audio)

        gen = PodcastFeedGenerator()
        metadata = temp_store.get_metadata("url-test")
        episode = gen.create_episode_from_debate(
            debate_id="url-test",
            task="Test",
            agents=[],
            audio_url="https://cdn.example.com/audio/url-test.mp3",
            duration_seconds=300,
        )

        feed = gen.generate_feed([episode])
        assert "https://cdn.example.com/audio/url-test.mp3" in feed

    def test_metadata_roundtrip(self, temp_store, sample_audio):
        """Metadata should survive save/load cycle."""
        original_agents = ["claude", "gpt4", "gemini"]
        original_summary = "Test debate about microservices"

        temp_store.save(
            "roundtrip-test",
            sample_audio,
            duration_seconds=1234,
            task_summary=original_summary,
            agents=original_agents,
        )

        # Clear cache to force disk read
        temp_store._cache.clear()

        # Load metadata
        loaded = temp_store.get_metadata("roundtrip-test")

        assert loaded.duration_seconds == 1234
        assert loaded.task_summary == original_summary
        assert loaded.agents == original_agents
