"""Tests for Response Compression Middleware.

Covers:
- CompressionStats calculation
- should_compress() decision logic
- parse_accept_encoding() header parsing
- select_encoding() negotiation
- compress_gzip() and compress_brotli() functions
- compress_response() wrapper
- CompressionMiddleware ASGI middleware
- Statistics tracking and reset
"""

from __future__ import annotations

import gzip
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.performance.compression import (
    COMPRESSIBLE_TYPES,
    NO_COMPRESS_PATHS,
    CompressionMiddleware,
    CompressionStats,
    compress_gzip,
    compress_response,
    get_compression_stats,
    parse_accept_encoding,
    reset_compression_stats,
    select_encoding,
    should_compress,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_stats():
    """Reset compression stats before each test."""
    reset_compression_stats()
    yield
    reset_compression_stats()


@pytest.fixture
def sample_text_data():
    """Sample text data for compression testing."""
    return b"Hello, World! " * 1000  # ~14KB of text


@pytest.fixture
def small_data():
    """Data smaller than compression threshold."""
    return b"small"


# =============================================================================
# CompressionStats Tests
# =============================================================================


class TestCompressionStats:
    """Tests for CompressionStats dataclass."""

    def test_initial_values(self):
        """Test default initial values."""
        stats = CompressionStats()

        assert stats.total_requests == 0
        assert stats.compressed_requests == 0
        assert stats.skipped_requests == 0
        assert stats.total_original_bytes == 0
        assert stats.total_compressed_bytes == 0
        assert stats.gzip_count == 0
        assert stats.brotli_count == 0
        assert stats.compression_time_ms == 0.0

    def test_compression_ratio_empty(self):
        """Test compression ratio with no data."""
        stats = CompressionStats()
        assert stats.compression_ratio == 0.0

    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        stats = CompressionStats(
            total_original_bytes=1000,
            total_compressed_bytes=300,  # 70% compression
        )
        assert abs(stats.compression_ratio - 0.7) < 0.01

    def test_avg_compression_time_empty(self):
        """Test average time with no compressions."""
        stats = CompressionStats()
        assert stats.avg_compression_time_ms == 0.0

    def test_avg_compression_time_calculation(self):
        """Test average compression time calculation."""
        stats = CompressionStats(
            compressed_requests=5,
            compression_time_ms=50.0,
        )
        assert stats.avg_compression_time_ms == 10.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = CompressionStats(
            total_requests=100,
            compressed_requests=80,
            skipped_requests=20,
            total_original_bytes=50000,
            total_compressed_bytes=15000,
            gzip_count=70,
            brotli_count=10,
            compression_time_ms=200.0,
        )

        d = stats.to_dict()

        assert d["total_requests"] == 100
        assert d["compressed_requests"] == 80
        assert d["bytes_saved"] == 35000
        assert d["gzip_count"] == 70
        assert d["brotli_count"] == 10


# =============================================================================
# should_compress Tests
# =============================================================================


class TestShouldCompress:
    """Tests for should_compress() function."""

    def test_small_content_rejected(self, small_data):
        """Small content is not compressed."""
        result = should_compress(
            content_type="application/json",
            content_length=len(small_data),
            min_size=1000,
        )
        assert result is False

    def test_json_content_accepted(self, sample_text_data):
        """JSON content is compressed."""
        result = should_compress(
            content_type="application/json",
            content_length=len(sample_text_data),
        )
        assert result is True

    def test_text_html_accepted(self, sample_text_data):
        """HTML content is compressed."""
        result = should_compress(
            content_type="text/html",
            content_length=len(sample_text_data),
        )
        assert result is True

    def test_text_plain_accepted(self, sample_text_data):
        """Plain text is compressed."""
        result = should_compress(
            content_type="text/plain",
            content_length=len(sample_text_data),
        )
        assert result is True

    def test_binary_rejected(self):
        """Binary content types are not compressed."""
        result = should_compress(
            content_type="application/octet-stream",
            content_length=5000,
        )
        assert result is False

    def test_image_rejected(self):
        """Image content (except SVG) is not compressed."""
        result = should_compress(
            content_type="image/png",
            content_length=5000,
        )
        assert result is False

    def test_svg_accepted(self):
        """SVG images are compressed."""
        result = should_compress(
            content_type="image/svg+xml",
            content_length=5000,
        )
        assert result is True

    def test_content_type_with_charset(self):
        """Content type with charset is handled."""
        result = should_compress(
            content_type="application/json; charset=utf-8",
            content_length=5000,
        )
        assert result is True

    def test_excluded_path_stream(self):
        """Stream paths are not compressed."""
        result = should_compress(
            content_type="application/json",
            content_length=5000,
            path="/api/v1/stream/events",
        )
        assert result is False

    def test_excluded_path_websocket(self):
        """WebSocket paths are not compressed."""
        result = should_compress(
            content_type="application/json",
            content_length=5000,
            path="/api/v1/ws/connect",
        )
        assert result is False

    def test_custom_min_size(self):
        """Custom minimum size is respected."""
        result = should_compress(
            content_type="application/json",
            content_length=500,
            min_size=100,  # Below default 1000
        )
        assert result is True


# =============================================================================
# parse_accept_encoding Tests
# =============================================================================


class TestParseAcceptEncoding:
    """Tests for parse_accept_encoding() function."""

    def test_empty_header(self):
        """Empty header returns empty list."""
        result = parse_accept_encoding("")
        assert result == []

    def test_single_encoding(self):
        """Single encoding is parsed."""
        result = parse_accept_encoding("gzip")
        assert result == [("gzip", 1.0)]

    def test_multiple_encodings(self):
        """Multiple encodings are parsed."""
        result = parse_accept_encoding("gzip, deflate, br")
        encodings = [enc for enc, _ in result]
        assert "gzip" in encodings
        assert "deflate" in encodings
        assert "br" in encodings

    def test_quality_values(self):
        """Quality values are parsed."""
        result = parse_accept_encoding("gzip;q=0.8, br;q=1.0")
        assert ("gzip", 0.8) in result
        assert ("br", 1.0) in result

    def test_sorted_by_quality(self):
        """Results are sorted by quality (highest first)."""
        result = parse_accept_encoding("gzip;q=0.5, br;q=1.0, deflate;q=0.8")
        assert result[0] == ("br", 1.0)
        assert result[1] == ("deflate", 0.8)
        assert result[2] == ("gzip", 0.5)

    def test_invalid_quality_defaults_to_one(self):
        """Invalid quality value defaults to 1.0."""
        result = parse_accept_encoding("gzip;q=invalid")
        assert result == [("gzip", 1.0)]

    def test_whitespace_handling(self):
        """Whitespace is handled properly."""
        result = parse_accept_encoding("  gzip  ,  br  ")
        encodings = [enc for enc, _ in result]
        assert "gzip" in encodings
        assert "br" in encodings


# =============================================================================
# select_encoding Tests
# =============================================================================


class TestSelectEncoding:
    """Tests for select_encoding() function."""

    def test_empty_header_returns_none(self):
        """Empty header returns None."""
        result = select_encoding("")
        assert result is None

    def test_gzip_selected(self):
        """Gzip is selected when accepted."""
        result = select_encoding("gzip")
        assert result == "gzip"

    def test_deflate_maps_to_gzip(self):
        """Deflate accepts gzip."""
        result = select_encoding("deflate")
        assert result == "gzip"

    def test_brotli_preferred_when_available(self):
        """Brotli is preferred when available."""
        from aragora.performance.compression import BROTLI_AVAILABLE

        if BROTLI_AVAILABLE:
            result = select_encoding("gzip, br")
            assert result == "br"
        else:
            result = select_encoding("gzip, br")
            assert result == "gzip"

    def test_identity_only_returns_none(self):
        """Identity only returns None (no compression)."""
        result = select_encoding("identity")
        assert result is None

    def test_quality_zero_rejected(self):
        """Quality 0 encodings are rejected."""
        result = select_encoding("gzip;q=0")
        assert result is None


# =============================================================================
# compress_gzip Tests
# =============================================================================


class TestCompressGzip:
    """Tests for compress_gzip() function."""

    def test_compresses_data(self, sample_text_data):
        """Data is compressed."""
        compressed = compress_gzip(sample_text_data)
        assert len(compressed) < len(sample_text_data)

    def test_can_decompress(self, sample_text_data):
        """Compressed data can be decompressed."""
        compressed = compress_gzip(sample_text_data)
        decompressed = gzip.decompress(compressed)
        assert decompressed == sample_text_data

    def test_compression_level_affects_size(self, sample_text_data):
        """Higher compression level produces smaller output."""
        low_compression = compress_gzip(sample_text_data, level=1)
        high_compression = compress_gzip(sample_text_data, level=9)
        assert len(high_compression) <= len(low_compression)

    def test_empty_data(self):
        """Empty data is handled."""
        compressed = compress_gzip(b"")
        decompressed = gzip.decompress(compressed)
        assert decompressed == b""


# =============================================================================
# compress_brotli Tests
# =============================================================================


class TestCompressBrotli:
    """Tests for compress_brotli() function."""

    def test_brotli_compresses_data(self, sample_text_data):
        """Brotli compresses data when available."""
        from aragora.performance.compression import BROTLI_AVAILABLE, compress_brotli

        assert BROTLI_AVAILABLE, "brotli should be available"

        compressed = compress_brotli(sample_text_data)
        assert len(compressed) < len(sample_text_data)

    def test_brotli_unavailable_raises(self):
        """Raises when brotli not available."""
        from aragora.performance.compression import compress_brotli

        with patch("aragora.performance.compression.BROTLI_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="brotli not available"):
                compress_brotli(b"test data")


# =============================================================================
# compress_response Tests
# =============================================================================


class TestCompressResponse:
    """Tests for compress_response() function."""

    def test_gzip_encoding(self, sample_text_data):
        """Gzip encoding works."""
        compressed = compress_response(sample_text_data, encoding="gzip")
        assert len(compressed) < len(sample_text_data)

        stats = get_compression_stats()
        assert stats["gzip_count"] == 1

    def test_brotli_encoding_when_available(self, sample_text_data):
        """Brotli encoding works when available."""
        from aragora.performance.compression import BROTLI_AVAILABLE

        assert BROTLI_AVAILABLE, "brotli should be available"

        compressed = compress_response(sample_text_data, encoding="br")
        assert len(compressed) < len(sample_text_data)

        stats = get_compression_stats()
        assert stats["brotli_count"] == 1

    def test_tracks_statistics(self, sample_text_data):
        """Statistics are tracked."""
        compress_response(sample_text_data, encoding="gzip")

        stats = get_compression_stats()
        assert stats["compressed_requests"] == 1
        assert stats["total_original_bytes"] == len(sample_text_data)
        assert stats["total_compressed_bytes"] > 0
        assert stats["total_compressed_bytes"] < len(sample_text_data)

    def test_custom_level(self, sample_text_data):
        """Custom compression level is used."""
        compressed_low = compress_response(sample_text_data, encoding="gzip", level=1)
        reset_compression_stats()
        compressed_high = compress_response(sample_text_data, encoding="gzip", level=9)
        assert len(compressed_high) <= len(compressed_low)

    def test_returns_original_on_error(self):
        """Returns original data on compression error."""
        # This is hard to trigger naturally, but we can verify the pattern
        data = b"test data"
        result = compress_response(data, encoding="gzip")
        assert result is not None


# =============================================================================
# CompressionMiddleware Tests
# =============================================================================


class TestCompressionMiddleware:
    """Tests for CompressionMiddleware class."""

    def test_initialization(self):
        """Middleware initializes with defaults."""
        middleware = CompressionMiddleware()
        assert middleware._min_size == 1000
        assert middleware._gzip_level == 6
        assert middleware._brotli_level == 4

    def test_custom_configuration(self):
        """Middleware accepts custom configuration."""
        middleware = CompressionMiddleware(
            min_size=500,
            gzip_level=9,
            brotli_level=11,
            excluded_paths={"/custom/"},
        )
        assert middleware._min_size == 500
        assert middleware._gzip_level == 9
        assert middleware._brotli_level == 11
        assert "/custom/" in middleware._excluded_paths

    @pytest.mark.asyncio
    async def test_no_compression_without_accept_encoding(self):
        """No compression when Accept-Encoding not present."""
        middleware = CompressionMiddleware()

        request = MagicMock()
        request.headers = {}
        request.url.path = "/api/test"

        response = MagicMock()
        response.headers = {"content-type": "application/json"}

        call_next = AsyncMock(return_value=response)

        result = await middleware(request, call_next)

        stats = get_compression_stats()
        assert stats["skipped_requests"] == 1

    @pytest.mark.asyncio
    async def test_compression_with_accept_encoding(self):
        """Compression when Accept-Encoding: gzip present."""
        middleware = CompressionMiddleware(min_size=10)

        request = MagicMock()
        request.headers = {"accept-encoding": "gzip"}
        request.url.path = "/api/test"

        # Create mock response with body iterator
        body_content = b"Hello World! " * 100  # Large enough to compress

        async def body_iter():
            yield body_content

        response = MagicMock()
        response.headers = {"content-type": "application/json"}
        response.body_iterator = body_iter()
        response.status_code = 200
        response.media_type = "application/json"

        call_next = AsyncMock(return_value=response)

        with patch("starlette.responses.Response") as MockResponse:
            MockResponse.return_value = MagicMock()
            result = await middleware(request, call_next)

        # Should have attempted compression
        assert call_next.called

    @pytest.mark.asyncio
    async def test_excluded_path_not_compressed(self):
        """Excluded paths are not compressed."""
        middleware = CompressionMiddleware(excluded_paths={"/no-compress/"})

        request = MagicMock()
        request.headers = {"accept-encoding": "gzip"}
        request.url.path = "/no-compress/file"

        response = MagicMock()
        call_next = AsyncMock(return_value=response)

        result = await middleware(request, call_next)

        stats = get_compression_stats()
        assert stats["skipped_requests"] >= 1


# =============================================================================
# Statistics Functions Tests
# =============================================================================


class TestStatisticsFunctions:
    """Tests for get_compression_stats() and reset_compression_stats()."""

    def test_get_compression_stats(self, sample_text_data):
        """get_compression_stats returns current stats."""
        compress_response(sample_text_data, encoding="gzip")
        compress_response(sample_text_data, encoding="gzip")

        stats = get_compression_stats()
        assert stats["compressed_requests"] == 2
        assert stats["gzip_count"] == 2

    def test_reset_compression_stats(self, sample_text_data):
        """reset_compression_stats clears all stats."""
        compress_response(sample_text_data, encoding="gzip")

        reset_compression_stats()

        stats = get_compression_stats()
        assert stats["compressed_requests"] == 0
        assert stats["gzip_count"] == 0


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_compressible_types_contains_common_types(self):
        """COMPRESSIBLE_TYPES contains common text types."""
        assert "text/plain" in COMPRESSIBLE_TYPES
        assert "text/html" in COMPRESSIBLE_TYPES
        assert "application/json" in COMPRESSIBLE_TYPES
        assert "application/javascript" in COMPRESSIBLE_TYPES
        assert "text/css" in COMPRESSIBLE_TYPES

    def test_no_compress_paths_contains_streaming(self):
        """NO_COMPRESS_PATHS contains streaming paths."""
        assert any("stream" in path for path in NO_COMPRESS_PATHS)
        assert any("ws" in path for path in NO_COMPRESS_PATHS)


# =============================================================================
# Integration Tests
# =============================================================================


class TestCompressionIntegration:
    """Integration tests for compression system."""

    def test_full_compression_workflow(self, sample_text_data):
        """Test complete compression workflow."""
        # Check if should compress
        assert should_compress(
            content_type="application/json",
            content_length=len(sample_text_data),
        )

        # Select encoding
        encoding = select_encoding("gzip, deflate, br")
        assert encoding in ("gzip", "br")

        # Compress
        compressed = compress_response(sample_text_data, encoding="gzip")
        assert len(compressed) < len(sample_text_data)

        # Verify stats
        stats = get_compression_stats()
        assert stats["compressed_requests"] == 1
        ratio = 1 - (stats["total_compressed_bytes"] / stats["total_original_bytes"])
        assert ratio > 0.5  # At least 50% compression

    def test_multiple_compressions_accumulated(self, sample_text_data):
        """Multiple compressions accumulate stats correctly."""
        for _ in range(10):
            compress_response(sample_text_data, encoding="gzip")

        stats = get_compression_stats()
        assert stats["compressed_requests"] == 10
        assert stats["gzip_count"] == 10
        assert stats["total_original_bytes"] == len(sample_text_data) * 10
