"""Tests for YouTubeUploaderConnector - init, config, metadata, rate limiter."""

from pathlib import Path

import pytest

from aragora.connectors.youtube_uploader import (
    UploadResult,
    YouTubeRateLimiter,
    YouTubeUploaderConnector,
    YouTubeVideoMetadata,
    create_video_metadata_from_debate,
)


class TestYouTubeUploaderInit:
    """Initialization and configuration."""

    def test_default_init_unconfigured(self, monkeypatch):
        monkeypatch.delenv("YOUTUBE_CLIENT_ID", raising=False)
        monkeypatch.delenv("YOUTUBE_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("YOUTUBE_REFRESH_TOKEN", raising=False)
        connector = YouTubeUploaderConnector()
        assert not connector.is_configured

    def test_explicit_credentials(self):
        connector = YouTubeUploaderConnector(
            client_id="cid", client_secret="cs", refresh_token="rt"
        )
        assert connector.is_configured

    def test_partial_credentials_not_configured(self):
        connector = YouTubeUploaderConnector(client_id="cid")
        assert not connector.is_configured

    def test_auth_url_generation(self):
        connector = YouTubeUploaderConnector(client_id="test-id")
        url = connector.get_auth_url("https://example.com/callback", state="abc")
        assert "test-id" in url
        assert "abc" in url
        assert "youtube.upload" in url


class TestUploadUnconfigured:
    """Upload without credentials returns graceful failure."""

    @pytest.mark.asyncio
    async def test_upload_unconfigured(self, tmp_path):
        connector = YouTubeUploaderConnector()
        meta = YouTubeVideoMetadata(title="Test", description="Desc")
        result = await connector.upload(tmp_path / "video.mp4", meta)
        assert isinstance(result, UploadResult)
        assert not result.success
        assert "not configured" in result.error

    @pytest.mark.asyncio
    async def test_upload_file_not_found(self, tmp_path):
        connector = YouTubeUploaderConnector(client_id="c", client_secret="s", refresh_token="r")
        meta = YouTubeVideoMetadata(title="Test", description="Desc")
        result = await connector.upload(tmp_path / "nonexistent.mp4", meta)
        assert not result.success
        assert "not found" in result.error


class TestVideoMetadata:
    """YouTubeVideoMetadata validation and conversion."""

    def test_title_truncation(self):
        meta = YouTubeVideoMetadata(title="x" * 200, description="desc")
        assert len(meta.title) <= 100
        assert meta.title.endswith("...")

    def test_description_truncation(self):
        meta = YouTubeVideoMetadata(title="t", description="y" * 6000)
        assert len(meta.description) <= 5000

    def test_tags_truncation(self):
        long_tags = [f"tag{'x' * 50}_{i}" for i in range(20)]
        meta = YouTubeVideoMetadata(title="t", description="d", tags=long_tags)
        total = sum(len(t) for t in meta.tags)
        assert total <= 500

    def test_to_api_body(self):
        meta = YouTubeVideoMetadata(
            title="Test Video",
            description="A test",
            tags=["ai", "debate"],
            category_id="28",
            privacy_status="unlisted",
        )
        body = meta.to_api_body()
        assert body["snippet"]["title"] == "Test Video"
        assert body["status"]["privacyStatus"] == "unlisted"
        assert "ai" in body["snippet"]["tags"]

    def test_defaults(self):
        meta = YouTubeVideoMetadata(title="T", description="D")
        assert meta.category_id == "28"
        assert meta.privacy_status == "public"
        assert meta.made_for_kids is False
        assert meta.language == "en"


class TestYouTubeRateLimiter:
    """Rate limiter quota tracking."""

    def test_can_upload_initially(self):
        limiter = YouTubeRateLimiter(daily_quota=10000)
        assert limiter.can_upload()
        assert limiter.remaining_quota == 10000

    def test_record_upload_reduces_quota(self):
        limiter = YouTubeRateLimiter(daily_quota=10000)
        limiter.record_upload()
        assert limiter.remaining_quota == 8400  # 10000 - 1600

    def test_quota_exhaustion(self):
        limiter = YouTubeRateLimiter(daily_quota=3000)
        limiter.record_upload()  # 1600 used, 1400 remaining
        assert not limiter.can_upload()  # Need 1600, only 1400 left

    def test_record_api_call(self):
        limiter = YouTubeRateLimiter(daily_quota=100)
        limiter.record_api_call(10)
        assert limiter.remaining_quota == 90


class TestCreateVideoMetadataFromDebate:
    """Helper function for creating metadata from debate info."""

    def test_basic_metadata(self):
        meta = create_video_metadata_from_debate(
            task="Should we use Kubernetes?",
            agents=["Claude", "GPT-4"],
            consensus_reached=True,
            debate_id="debate-123",
        )
        assert "Kubernetes" in meta.title
        assert "Claude" in meta.description
        assert "Consensus" in meta.description
        assert meta.category_id == "28"

    def test_long_task_truncated(self):
        meta = create_video_metadata_from_debate(
            task="x" * 200,
            agents=["A"],
            consensus_reached=False,
            debate_id="d1",
        )
        assert len(meta.title) <= 100

    def test_no_consensus(self):
        meta = create_video_metadata_from_debate(
            task="Topic",
            agents=["A", "B"],
            consensus_reached=False,
            debate_id="d2",
        )
        assert "No consensus" in meta.description


class TestUploadResultDataclass:
    """UploadResult basic behavior."""

    def test_defaults(self):
        r = UploadResult(video_id="v1", title="T", url="u")
        assert r.success is True
        assert r.upload_status == "complete"
        assert r.error is None
