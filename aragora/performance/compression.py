"""
Response Compression Middleware.

Provides gzip and brotli compression for HTTP responses
to reduce bandwidth and improve load times.

Features:
- Automatic content-type detection
- Configurable minimum size threshold
- Accept-Encoding negotiation
- Compression level configuration
- Prometheus metrics for compression ratios

Usage:
    # As middleware
    app.middleware("http")(CompressionMiddleware(min_size=1000))

    # Direct usage
    compressed = compress_response(data, encoding="gzip")
"""

from __future__ import annotations

import gzip
import io
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Optional brotli support
try:
    import brotli

    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False
    logger.debug("brotli not available, falling back to gzip only")


@dataclass
class CompressionStats:
    """Statistics for compression operations."""

    total_requests: int = 0
    compressed_requests: int = 0
    skipped_requests: int = 0
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0
    gzip_count: int = 0
    brotli_count: int = 0
    compression_time_ms: float = 0.0

    @property
    def compression_ratio(self) -> float:
        """Average compression ratio."""
        if self.total_original_bytes == 0:
            return 0.0
        return 1 - (self.total_compressed_bytes / self.total_original_bytes)

    @property
    def avg_compression_time_ms(self) -> float:
        """Average compression time per request."""
        if self.compressed_requests == 0:
            return 0.0
        return self.compression_time_ms / self.compressed_requests

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "compressed_requests": self.compressed_requests,
            "skipped_requests": self.skipped_requests,
            "compression_ratio": f"{self.compression_ratio * 100:.1f}%",
            "total_original_bytes": self.total_original_bytes,
            "total_compressed_bytes": self.total_compressed_bytes,
            "bytes_saved": self.total_original_bytes - self.total_compressed_bytes,
            "gzip_count": self.gzip_count,
            "brotli_count": self.brotli_count,
            "avg_compression_time_ms": round(self.avg_compression_time_ms, 2),
        }


# Global stats
_compression_stats = CompressionStats()

# Content types that should be compressed
COMPRESSIBLE_TYPES: Set[str] = {
    "text/plain",
    "text/html",
    "text/css",
    "text/javascript",
    "text/xml",
    "application/json",
    "application/javascript",
    "application/xml",
    "application/xhtml+xml",
    "application/rss+xml",
    "application/atom+xml",
    "image/svg+xml",
    "application/x-javascript",
    "application/x-font-ttf",
    "font/opentype",
    "font/ttf",
    "font/otf",
}

# Paths that should not be compressed (streaming, binary)
NO_COMPRESS_PATHS: Set[str] = {
    "/api/v1/stream/",
    "/api/v1/ws/",
    "/api/v1/audio/",
    "/api/v1/download/",
}


def should_compress(
    content_type: str,
    content_length: int,
    path: str = "",
    min_size: int = 1000,
) -> bool:
    """
    Determine if content should be compressed.

    Args:
        content_type: Response content type
        content_length: Response size in bytes
        path: Request path
        min_size: Minimum size to compress

    Returns:
        True if content should be compressed
    """
    # Too small
    if content_length < min_size:
        return False

    # Check path exclusions
    for excluded in NO_COMPRESS_PATHS:
        if path.startswith(excluded):
            return False

    # Check content type
    base_type = content_type.split(";")[0].strip().lower()
    return base_type in COMPRESSIBLE_TYPES


def parse_accept_encoding(header: str) -> List[Tuple[str, float]]:
    """
    Parse Accept-Encoding header into ordered list.

    Args:
        header: Accept-Encoding header value

    Returns:
        List of (encoding, quality) tuples, highest quality first
    """
    if not header:
        return []

    encodings = []
    for part in header.split(","):
        part = part.strip()
        if not part:
            continue

        if ";q=" in part:
            encoding, q = part.split(";q=", 1)
            try:
                quality = float(q.strip())
            except ValueError:
                quality = 1.0
        else:
            encoding = part
            quality = 1.0

        encodings.append((encoding.strip().lower(), quality))

    # Sort by quality, highest first
    encodings.sort(key=lambda x: x[1], reverse=True)
    return encodings


def select_encoding(accept_encoding: str) -> Optional[str]:
    """
    Select best compression encoding based on Accept-Encoding.

    Prefers brotli > gzip > identity.

    Args:
        accept_encoding: Accept-Encoding header value

    Returns:
        Selected encoding or None
    """
    encodings = parse_accept_encoding(accept_encoding)

    if not encodings:
        return None

    # Build set of acceptable encodings
    acceptable = {enc for enc, q in encodings if q > 0}

    # Prefer brotli if available and accepted
    if BROTLI_AVAILABLE and "br" in acceptable:
        return "br"

    # Fall back to gzip
    if "gzip" in acceptable or "deflate" in acceptable:
        return "gzip"

    return None


def compress_gzip(
    data: bytes,
    level: int = 6,
) -> bytes:
    """
    Compress data using gzip.

    Args:
        data: Data to compress
        level: Compression level (1-9)

    Returns:
        Compressed data
    """
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=level) as f:
        f.write(data)
    return buffer.getvalue()


def compress_brotli(
    data: bytes,
    level: int = 4,
) -> bytes:
    """
    Compress data using brotli.

    Args:
        data: Data to compress
        level: Compression level (0-11)

    Returns:
        Compressed data
    """
    if not BROTLI_AVAILABLE:
        raise RuntimeError("brotli not available")
    return bytes(brotli.compress(data, quality=level))


def compress_response(
    data: bytes,
    encoding: str = "gzip",
    level: Optional[int] = None,
) -> bytes:
    """
    Compress response data.

    Args:
        data: Data to compress
        encoding: Compression encoding ("gzip" or "br")
        level: Optional compression level

    Returns:
        Compressed data
    """
    start = time.perf_counter()

    try:
        if encoding == "br":
            compressed = compress_brotli(data, level or 4)
            _compression_stats.brotli_count += 1
        else:
            compressed = compress_gzip(data, level or 6)
            _compression_stats.gzip_count += 1

        _compression_stats.compressed_requests += 1
        _compression_stats.total_original_bytes += len(data)
        _compression_stats.total_compressed_bytes += len(compressed)
        _compression_stats.compression_time_ms += (time.perf_counter() - start) * 1000

        return compressed

    except Exception as e:
        logger.warning(f"Compression failed: {e}")
        return data


class CompressionMiddleware:
    """
    HTTP middleware for response compression.

    Automatically compresses responses based on Accept-Encoding
    and content type.
    """

    def __init__(
        self,
        min_size: int = 1000,
        gzip_level: int = 6,
        brotli_level: int = 4,
        excluded_paths: Optional[Set[str]] = None,
    ):
        """
        Initialize compression middleware.

        Args:
            min_size: Minimum response size to compress
            gzip_level: Gzip compression level (1-9)
            brotli_level: Brotli compression level (0-11)
            excluded_paths: Additional paths to exclude
        """
        self._min_size = min_size
        self._gzip_level = gzip_level
        self._brotli_level = brotli_level
        self._excluded_paths = excluded_paths or set()

    async def __call__(
        self,
        request: Any,
        call_next: Callable[[Any], Any],
    ) -> Any:
        """
        Process request and optionally compress response.

        Compatible with Starlette/FastAPI middleware pattern.
        """
        _compression_stats.total_requests += 1

        # Get response
        response = await call_next(request)

        # Check if we should compress
        accept_encoding = request.headers.get("accept-encoding", "")
        encoding = select_encoding(accept_encoding)

        if not encoding:
            _compression_stats.skipped_requests += 1
            return response

        # Get response info
        content_type = response.headers.get("content-type", "")
        path = str(request.url.path)

        # Check path exclusions
        for excluded in self._excluded_paths:
            if path.startswith(excluded):
                _compression_stats.skipped_requests += 1
                return response

        # Check if already compressed
        if response.headers.get("content-encoding"):
            _compression_stats.skipped_requests += 1
            return response

        # Get response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk

        # Check if should compress
        if not should_compress(
            content_type,
            len(body),
            path,
            self._min_size,
        ):
            _compression_stats.skipped_requests += 1
            # Return original response
            from starlette.responses import Response

            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        # Compress
        level = self._brotli_level if encoding == "br" else self._gzip_level
        compressed = compress_response(body, encoding, level)

        # Build compressed response
        from starlette.responses import Response

        headers = dict(response.headers)
        headers["content-encoding"] = encoding
        headers["content-length"] = str(len(compressed))
        # Add Vary header for caching
        vary = headers.get("vary", "")
        if "accept-encoding" not in vary.lower():
            headers["vary"] = f"{vary}, Accept-Encoding" if vary else "Accept-Encoding"

        return Response(
            content=compressed,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
        )


def get_compression_stats() -> Dict[str, Any]:
    """Get compression statistics."""
    return _compression_stats.to_dict()


def reset_compression_stats() -> None:
    """Reset compression statistics."""
    global _compression_stats
    _compression_stats = CompressionStats()


__all__ = [
    "CompressionMiddleware",
    "compress_response",
    "compress_gzip",
    "compress_brotli",
    "should_compress",
    "select_encoding",
    "parse_accept_encoding",
    "get_compression_stats",
    "reset_compression_stats",
    "COMPRESSIBLE_TYPES",
    "CompressionStats",
]
