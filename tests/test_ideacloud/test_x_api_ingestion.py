"""Tests for the live X API mode of the bookmark/like ingestors.

All tests use a fake connector — no network, no real tokens.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from aragora.ideacloud.ingestion.twitter_bookmarks import TwitterBookmarksIngestor
from aragora.ideacloud.ingestion.twitter_likes import TwitterLikesIngestor
from aragora.ideacloud.ingestion.x_api import (
    fetch_live_entries,
    is_api_source,
)


def _entry(tweet_id: str, text: str = "Some insight #ai") -> dict:
    return {
        "tweetId": tweet_id,
        "fullText": text,
        "screenName": "someone",
        "created_at": "2026-08-28T12:00:00.000Z",
        "public_metrics": {},
    }


class FakeConnector:
    """Pages of export-shaped entries, mimicking TwitterConnector's pager."""

    def __init__(self, pages: list[list[dict]], user_id: str | None = "42"):
        self.pages = pages
        self.user_id = user_id
        self.calls = 0

    async def get_authenticated_user_id(self) -> str | None:
        return self.user_id

    async def _page(self, pagination_token):
        index = int(pagination_token) if pagination_token else 0
        self.calls += 1
        if index >= len(self.pages):
            return [], None
        next_token = str(index + 1) if index + 1 < len(self.pages) else None
        return self.pages[index], next_token

    async def fetch_bookmarks_page(self, user_id, pagination_token=None, max_results=100):
        return await self._page(pagination_token)

    async def fetch_liked_page(self, user_id, pagination_token=None, max_results=100):
        return await self._page(pagination_token)


class TestIsApiSource:
    def test_recognizes_api_strings(self):
        assert is_api_source("api:")
        assert is_api_source("api:500")
        assert is_api_source(" API: ")

    def test_rejects_paths(self):
        assert not is_api_source("data/bookmarks.js")
        assert not is_api_source("/tmp/api:weird")
        from pathlib import Path

        assert not is_api_source(Path("bookmarks.js"))


class TestFetchLiveEntries:
    def test_fetches_across_pages(self, tmp_path):
        connector = FakeConnector([[_entry("3"), _entry("2")], [_entry("1")]])
        entries = asyncio.run(
            fetch_live_entries(
                "twitter_bookmark",
                "api:",
                connector=connector,
                state_path=tmp_path / "state.json",
            )
        )
        assert [e["tweetId"] for e in entries] == ["3", "2", "1"]

    def test_stops_at_seen_id_and_updates_state(self, tmp_path):
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps({"twitter_bookmark": {"seen_ids": ["2"]}}))
        connector = FakeConnector([[_entry("4"), _entry("3"), _entry("2"), _entry("1")]])

        entries = asyncio.run(
            fetch_live_entries(
                "twitter_bookmark", "api:", connector=connector, state_path=state_path
            )
        )

        assert [e["tweetId"] for e in entries] == ["4", "3"]
        state = json.loads(state_path.read_text())
        assert state["twitter_bookmark"]["seen_ids"][:2] == ["4", "3"]
        # Previously seen ids are retained behind the new ones
        assert "2" in state["twitter_bookmark"]["seen_ids"]

    def test_respects_max_items_suffix(self, tmp_path):
        connector = FakeConnector([[_entry(str(i)) for i in range(9, -1, -1)]])
        entries = asyncio.run(
            fetch_live_entries(
                "twitter_bookmark",
                "api:3",
                connector=connector,
                state_path=tmp_path / "state.json",
            )
        )
        assert len(entries) == 3

    def test_no_token_returns_empty(self, tmp_path):
        connector = FakeConnector([], user_id=None)
        entries = asyncio.run(
            fetch_live_entries(
                "twitter_like", "api:", connector=connector, state_path=tmp_path / "s.json"
            )
        )
        assert entries == []

    def test_state_is_per_source_type(self, tmp_path):
        state_path = tmp_path / "state.json"
        connector = FakeConnector([[_entry("7")]])
        asyncio.run(
            fetch_live_entries(
                "twitter_bookmark", "api:", connector=connector, state_path=state_path
            )
        )
        connector_likes = FakeConnector([[_entry("7")]])
        likes = asyncio.run(
            fetch_live_entries(
                "twitter_like", "api:", connector=connector_likes, state_path=state_path
            )
        )
        # A bookmark seen-id must not suppress the same tweet arriving as a like
        assert [e["tweetId"] for e in likes] == ["7"]


class TestIngestorApiMode:
    def test_bookmarks_api_mode_maps_nodes(self, monkeypatch, tmp_path):
        async def fake_fetch(source_type, source, **kwargs):
            assert source_type == "twitter_bookmark"
            return [_entry("11", "Adversarial review beats vibes #decisions")]

        monkeypatch.setattr("aragora.ideacloud.ingestion.x_api.fetch_live_entries", fake_fetch)
        nodes = asyncio.run(TwitterBookmarksIngestor().ingest("api:"))
        assert len(nodes) == 1
        assert nodes[0].source_type == "twitter_bookmark"
        assert nodes[0].source_url.endswith("/status/11")
        assert "decisions" in nodes[0].tags

    def test_likes_api_mode_sets_source_type(self, monkeypatch, tmp_path):
        async def fake_fetch(source_type, source, **kwargs):
            assert source_type == "twitter_like"
            return [_entry("12")]

        monkeypatch.setattr("aragora.ideacloud.ingestion.x_api.fetch_live_entries", fake_fetch)
        nodes = asyncio.run(TwitterLikesIngestor().ingest("api:"))
        assert len(nodes) == 1
        assert nodes[0].source_type == "twitter_like"

    def test_file_mode_still_works(self, tmp_path):
        export = tmp_path / "bookmarks.js"
        export.write_text(
            'window.YTD.bookmark.part0 = [{"bookmark": {"tweetId": "99", '
            '"fullText": "hello world", "screenName": "abc"}}]'
        )
        nodes = asyncio.run(TwitterBookmarksIngestor().ingest(export))
        assert len(nodes) == 1
        assert nodes[0].source_url == "https://x.com/abc/status/99"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
