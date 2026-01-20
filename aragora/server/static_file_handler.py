"""
Static file serving with security protections.

Provides path traversal protection, symlink rejection,
SPA routing, and content type detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)

# Content type mapping for common file extensions
CONTENT_TYPES = {
    ".css": "text/css",
    ".js": "application/javascript",
    ".json": "application/json",
    ".ico": "image/x-icon",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".eot": "application/vnd.ms-fontobject",
    ".map": "application/json",
    ".html": "text/html",
    ".htm": "text/html",
    ".txt": "text/plain",
    ".xml": "application/xml",
}


class StaticFileHandler:
    """Handles static file serving with security protections.

    Security features:
    - Path traversal prevention
    - Symlink rejection
    - Content type validation

    Usage:
        handler = StaticFileHandler(static_dir=Path("./public"))
        result = handler.serve_file("css/style.css")
        if result:
            status, headers, content = result
    """

    def __init__(
        self,
        static_dir: Optional[Path] = None,
        enable_spa_routing: bool = True,
        spa_fallback: str = "index.html",
    ):
        """Initialize the static file handler.

        Args:
            static_dir: Root directory for static files
            enable_spa_routing: If True, serve index.html for missing files
            spa_fallback: File to serve for SPA routing (default: index.html)
        """
        self.static_dir = static_dir
        self.enable_spa_routing = enable_spa_routing
        self.spa_fallback = spa_fallback

    def validate_path(self, filename: str) -> Tuple[bool, Optional[Path], str]:
        """Validate and resolve a file path securely.

        Args:
            filename: Requested filename relative to static_dir

        Returns:
            Tuple of (is_valid, resolved_path, error_message)
        """
        if not self.static_dir:
            return False, None, "Static directory not configured"

        try:
            filepath = (self.static_dir / filename).resolve()
            static_dir_resolved = self.static_dir.resolve()

            # Ensure resolved path is within static directory
            if not str(filepath).startswith(str(static_dir_resolved)):
                return False, None, "Access denied"

            # Security: Reject symlinks to prevent escape attacks
            original_path = self.static_dir / filename
            if original_path.is_symlink():
                logger.warning(f"Symlink access denied: {filename}")
                return False, None, "Symlinks not allowed"

            return True, filepath, ""

        except (ValueError, OSError):
            return False, None, "Invalid path"

    def get_content_type(self, filename: str) -> str:
        """Determine content type from filename extension.

        Args:
            filename: Filename to check

        Returns:
            Content type string (defaults to text/html)
        """
        for ext, content_type in CONTENT_TYPES.items():
            if filename.endswith(ext):
                return content_type
        return "text/html"

    def serve_file(
        self, filename: str
    ) -> Optional[Tuple[int, dict, bytes]]:
        """Serve a static file with security protections.

        Args:
            filename: Requested filename relative to static_dir

        Returns:
            Tuple of (status_code, headers_dict, content_bytes) or None on error
        """
        # Validate path
        is_valid, filepath, error = self.validate_path(filename)
        if not is_valid:
            return None

        # Handle missing files with SPA routing
        if not filepath.exists():
            if self.enable_spa_routing:
                fallback_path = self.static_dir / self.spa_fallback
                if fallback_path.exists():
                    filepath = fallback_path
                else:
                    return None
            else:
                return None

        # Read and return file
        try:
            content = filepath.read_bytes()
            content_type = self.get_content_type(filename)

            headers = {
                "Content-Type": content_type,
                "Content-Length": str(len(content)),
            }

            return 200, headers, content

        except FileNotFoundError:
            return None
        except PermissionError:
            logger.warning(f"Permission denied: {filename}")
            return None
        except (IOError, OSError) as e:
            logger.error(f"File read error: {e}")
            return None


def serve_static_file(
    handler,
    filename: str,
    static_dir: Optional[Path],
    add_cors_fn: Callable,
    add_security_fn: Callable,
) -> bool:
    """Serve a static file using the handler's response methods.

    This is a convenience function that integrates StaticFileHandler
    with BaseHTTPRequestHandler.

    Args:
        handler: HTTP request handler instance
        filename: Requested filename
        static_dir: Root directory for static files
        add_cors_fn: Function to add CORS headers
        add_security_fn: Function to add security headers

    Returns:
        True if file was served, False otherwise
    """
    file_handler = StaticFileHandler(static_dir=static_dir)
    result = file_handler.serve_file(filename)

    if result is None:
        return False

    status, headers, content = result

    try:
        handler.send_response(status)
        for key, value in headers.items():
            handler.send_header(key, value)
        add_cors_fn()
        add_security_fn()
        handler.end_headers()
        handler.wfile.write(content)
        return True

    except (BrokenPipeError, ConnectionResetError) as e:
        logger.debug(f"Client disconnected during file serve: {type(e).__name__}")
        return False
