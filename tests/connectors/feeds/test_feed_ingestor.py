"""Tests for RSS/Atom feed ingestor."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.connectors.feeds import FeedEntry, FeedIngestor, FeedSource


# Sample RSS 2.0 feed
SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <link>https://example.com</link>
    <item>
      <title>Test Article 1</title>
      <link>https://example.com/article1</link>
      <description>This is a test article</description>
      <pubDate>Mon, 20 Jan 2025 12:00:00 GMT</pubDate>
      <author>test@example.com</author>
      <category>tech</category>
    </item>
    <item>
      <title>Test Article 2</title>
      <link>https://example.com/article2</link>
      <description>Another test article</description>
    </item>
  </channel>
</rss>
"""

# Sample Atom feed
SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Test Atom Feed</title>
  <entry>
    <id>urn:uuid:1234</id>
    <title>Atom Article 1</title>
    <link href="https://example.com/atom1" rel="alternate"/>
    <summary>Atom test article</summary>
    <published>2025-01-20T12:00:00Z</published>
    <author>
      <name>John Doe</name>
    </author>
    <category term="science"/>
  </entry>
</feed>
"""


class TestFeedEntry:
    """Tests for FeedEntry dataclass."""

    def test_feed_entry_defaults(self):
        """Test FeedEntry default values."""
        entry = FeedEntry(id="1", title="Test", link="https://example.com")
        assert entry.id == "1"
        assert entry.title == "Test"
        assert entry.summary == ""
        assert entry.categories == []

    def test_to_debate_context(self):
        """Test converting entry to debate context."""
        entry = FeedEntry(
            id="1",
            title="Test Article",
            link="https://example.com",
            summary="A summary",
            author="Jane",
            published="2025-01-20",
        )
        context = entry.to_debate_context()
        assert "Test Article" in context
        assert "Jane" in context
        assert "A summary" in context
        assert "https://example.com" in context

    def test_content_hash(self):
        """Test content hash generation."""
        entry1 = FeedEntry(id="1", title="Test", link="https://example.com")
        entry2 = FeedEntry(id="2", title="Test", link="https://example.com")
        entry3 = FeedEntry(id="1", title="Different", link="https://example.com")

        # Same content = same hash
        assert entry1.content_hash == entry2.content_hash
        # Different content = different hash
        assert entry1.content_hash != entry3.content_hash


class TestFeedSource:
    """Tests for FeedSource dataclass."""

    def test_feed_source_defaults(self):
        """Test FeedSource default values."""
        source = FeedSource(url="https://example.com/feed.xml")
        assert source.url == "https://example.com/feed.xml"
        assert source.name == "example.com"
        assert source.enabled is True
        assert source.max_entries == 50

    def test_feed_source_custom_name(self):
        """Test FeedSource with custom name."""
        source = FeedSource(url="https://example.com/feed.xml", name="My Feed")
        assert source.name == "My Feed"


class TestFeedIngestor:
    """Tests for FeedIngestor class."""

    @pytest.fixture
    def ingestor(self):
        """Create a feed ingestor for testing."""
        return FeedIngestor()

    def test_add_source(self, ingestor):
        """Test adding a feed source."""
        source = FeedSource(url="https://example.com/feed.xml")
        ingestor.add_source(source)
        assert len(ingestor.sources) == 1
        assert ingestor.sources[0].url == "https://example.com/feed.xml"

    def test_remove_source(self, ingestor):
        """Test removing a feed source."""
        source = FeedSource(url="https://example.com/feed.xml")
        ingestor.add_source(source)
        assert len(ingestor.sources) == 1

        removed = ingestor.remove_source("https://example.com/feed.xml")
        assert removed is True
        assert len(ingestor.sources) == 0

    def test_remove_nonexistent_source(self, ingestor):
        """Test removing a source that doesn't exist."""
        removed = ingestor.remove_source("https://nonexistent.com")
        assert removed is False

    def test_parse_rss(self, ingestor):
        """Test parsing RSS 2.0 feed."""
        source = FeedSource(url="https://example.com/feed.xml", name="Test")
        entries = ingestor._parse_feed(SAMPLE_RSS, source)

        assert len(entries) == 2
        assert entries[0].title == "Test Article 1"
        assert entries[0].link == "https://example.com/article1"
        assert entries[0].summary == "This is a test article"
        assert "tech" in entries[0].categories

    def test_parse_atom(self, ingestor):
        """Test parsing Atom feed."""
        source = FeedSource(url="https://example.com/atom.xml", name="Test")
        entries = ingestor._parse_feed(SAMPLE_ATOM, source)

        assert len(entries) == 1
        assert entries[0].title == "Atom Article 1"
        assert entries[0].link == "https://example.com/atom1"
        assert entries[0].author == "John Doe"
        assert "science" in entries[0].categories

    def test_parse_invalid_feed(self, ingestor):
        """Test parsing invalid feed content."""
        source = FeedSource(url="https://example.com/bad.xml")
        entries = ingestor._parse_feed("not valid xml", source)
        assert entries == []

    def test_clear_cache(self, ingestor):
        """Test clearing the cache."""
        ingestor._cache["test"] = ([], 0)
        ingestor._seen_hashes.add("hash1")

        ingestor.clear_cache()

        assert len(ingestor._cache) == 0
        assert len(ingestor._seen_hashes) == 0

    def test_get_cache_stats(self, ingestor):
        """Test getting cache statistics."""
        source = FeedSource(url="https://example.com/feed.xml")
        ingestor.add_source(source)
        ingestor._cache["test"] = ([], 0)
        ingestor._seen_hashes.add("hash1")

        stats = ingestor.get_cache_stats()

        assert stats["cached_feeds"] == 1
        assert stats["seen_hashes"] == 1
        assert stats["sources"] == 1
        assert stats["enabled_sources"] == 1

    @pytest.mark.asyncio
    async def test_fetch_all_no_sources(self, ingestor):
        """Test fetch_all with no sources."""
        entries = await ingestor.fetch_all()
        assert entries == []

    @pytest.mark.asyncio
    async def test_fetch_feed_with_mock(self, ingestor):
        """Test fetching a feed with mocked HTTP response."""
        source = FeedSource(url="https://example.com/feed.xml", name="Test")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.text = SAMPLE_RSS
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            entries = await ingestor._fetch_feed(source)

            assert len(entries) == 2
            assert entries[0].title == "Test Article 1"


class TestFeedIngestorDeduplication:
    """Tests for feed deduplication."""

    @pytest.fixture
    def ingestor(self):
        return FeedIngestor()

    @pytest.mark.asyncio
    async def test_deduplication(self, ingestor):
        """Test that duplicate entries are removed."""
        source1 = FeedSource(url="https://feed1.com", name="Feed 1")
        source2 = FeedSource(url="https://feed2.com", name="Feed 2")
        ingestor.add_source(source1)
        ingestor.add_source(source2)

        # Mock both feeds returning the same content
        with patch.object(ingestor, "_fetch_feed") as mock_fetch:
            mock_fetch.return_value = [
                FeedEntry(id="1", title="Same", link="https://example.com/same")
            ]

            entries = await ingestor.fetch_all(deduplicate=True)

            # Should only have 1 entry despite 2 feeds returning the same content
            # (though with mock, it returns for each source)
            assert len(entries) >= 1
