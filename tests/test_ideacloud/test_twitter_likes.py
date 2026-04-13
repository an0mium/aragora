"""Focused tests for Twitter likes ingestion."""

from __future__ import annotations

import logging

import pytest

from aragora.ideacloud.ingestion.twitter_likes import (
    TwitterLikesIngestor,
    _like_entry_to_node,
)


def test_like_entry_to_node_converts_nested_like_entry():
    node = _like_entry_to_node(
        {
            "like": {
                "tweetId": "12345",
                "fullText": "First sentence. Second sentence.",
                "screenName": "alice",
            }
        }
    )

    assert node is not None
    assert node.source_type == "twitter_like"
    assert node.title == "First sentence"
    assert node.body == "First sentence. Second sentence."
    assert node.source_url == "https://x.com/alice/status/12345"
    assert node.source_author == "@alice"


def test_like_entry_to_node_supports_flat_entry_shape():
    node = _like_entry_to_node(
        {
            "tweetId": "777",
            "fullText": "Flat payload works",
            "screenName": "flat_case",
        }
    )

    assert node is not None
    assert node.source_type == "twitter_like"
    assert node.source_url == "https://x.com/flat_case/status/777"


def test_like_entry_to_node_returns_none_without_tweet_id():
    node = _like_entry_to_node({"like": {"fullText": "Missing identifier"}})
    assert node is None


def test_like_entry_to_node_preserves_override_source_type():
    node = _like_entry_to_node(
        {"like": {"tweetId": "999", "fullText": "Custom source type"}},
        source_type="custom_like",
    )

    assert node is not None
    assert node.source_type == "custom_like"


@pytest.mark.asyncio
async def test_ingest_parses_wrapped_twitter_export(tmp_path):
    likes_file = tmp_path / "like.js"
    likes_file.write_text(
        """window.YTD.like.part0 = [
  {"like": {"tweetId": "100", "fullText": "Wrapped export text", "screenName": "wrapped"}}
];""",
        encoding="utf-8",
    )

    nodes = await TwitterLikesIngestor().ingest(likes_file)

    assert len(nodes) == 1
    assert nodes[0].title == "Wrapped export text"
    assert nodes[0].source_type == "twitter_like"
    assert nodes[0].source_url == "https://x.com/wrapped/status/100"


@pytest.mark.asyncio
async def test_ingest_accepts_raw_json_array(tmp_path):
    likes_file = tmp_path / "like.js"
    likes_file.write_text(
        """
[
  {"like": {"tweetId": "101", "fullText": "One #tag", "screenName": "raw"}},
  {"like": {"tweetId": "102", "fullText": "Two", "screenName": "raw"}}
]
""",
        encoding="utf-8",
    )

    nodes = await TwitterLikesIngestor().ingest(likes_file)

    assert len(nodes) == 2
    assert nodes[0].tags == ["tag"]
    assert all(node.source_type == "twitter_like" for node in nodes)


@pytest.mark.asyncio
async def test_ingest_skips_entries_that_cannot_be_converted(tmp_path):
    likes_file = tmp_path / "like.js"
    likes_file.write_text(
        """
[
  {"like": {"tweetId": "201", "fullText": "Keep me"}},
  {"like": {"fullText": "Drop me"}},
  {"tweetId": "202", "text": "Keep me too"}
]
""",
        encoding="utf-8",
    )

    nodes = await TwitterLikesIngestor().ingest(likes_file)

    assert len(nodes) == 2
    assert [node.body for node in nodes] == ["Keep me", "Keep me too"]


@pytest.mark.asyncio
async def test_ingest_returns_empty_list_for_invalid_json(tmp_path):
    likes_file = tmp_path / "like.js"
    likes_file.write_text("window.YTD.like.part0 = not valid json;", encoding="utf-8")

    nodes = await TwitterLikesIngestor().ingest(likes_file)

    assert nodes == []


@pytest.mark.asyncio
async def test_ingest_raises_file_not_found_for_missing_source(tmp_path):
    missing_file = tmp_path / "missing-like.js"

    with pytest.raises(FileNotFoundError, match="Likes file not found"):
        await TwitterLikesIngestor().ingest(missing_file)


@pytest.mark.asyncio
async def test_ingest_logs_parsed_count(tmp_path, caplog):
    likes_file = tmp_path / "like.js"
    likes_file.write_text(
        """
[
  {"like": {"tweetId": "301", "fullText": "Logged entry"}}
]
""",
        encoding="utf-8",
    )

    with caplog.at_level(logging.INFO, logger="aragora.ideacloud.ingestion.twitter_likes"):
        await TwitterLikesIngestor().ingest(likes_file)

    assert "Parsed 1 likes from like.js" in caplog.text
