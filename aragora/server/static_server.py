"""
Static file server - serves static files with security protections.

Handles:
- Static file serving with path traversal protection
- Content type detection
- SPA routing fallback to index.html
- Audio file serving for debate broadcasts
"""

import logging
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from http.server import BaseHTTPRequestHandler

logger = logging.getLogger(__name__)

# Content type mappings
CONTENT_TYPES = {
    ".html": "text/html",
    ".htm": "text/html",
    ".css": "text/css",
    ".js": "application/javascript",
    ".mjs": "application/javascript",
    ".json": "application/json",
    ".ico": "image/x-icon",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".eot": "application/vnd.ms-fontobject",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".txt": "text/plain",
    ".xml": "application/xml",
    ".pdf": "application/pdf",
    ".zip": "application/zip",
}


def get_content_type(filename: str) -> str:
    """Get content type for a file based on extension."""
    ext = Path(filename).suffix.lower()
    if ext in CONTENT_TYPES:
        return CONTENT_TYPES[ext]

    # Fall back to mimetypes module
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"


def validate_path(
    filename: str,
    base_dir: Path,
) -> tuple[bool, Optional[Path], str]:
    """
    Validate and resolve a file path with security checks.

    Args:
        filename: Requested filename/path
        base_dir: Base directory for static files

    Returns:
        Tuple of (is_valid, resolved_path, error_message)
    """
    try:
        # Resolve both paths
        filepath = (base_dir / filename).resolve()
        base_resolved = base_dir.resolve()

        # Check if resolved path is within base directory
        if not str(filepath).startswith(str(base_resolved)):
            return False, None, "Access denied: path traversal detected"

        # Check for symlinks (prevent escape attacks)
        original_path = base_dir / filename
        if original_path.is_symlink():
            logger.warning(f"Symlink access denied: {filename}")
            return False, None, "Symlinks not allowed"

        return True, filepath, ""

    except (ValueError, OSError) as e:
        logger.debug(f"Path validation error: {e}")
        return False, None, "Invalid path"


def serve_static_file(
    handler: "BaseHTTPRequestHandler",
    filename: str,
    static_dir: Path,
    add_cors_headers: Optional[Callable] = None,
    add_security_headers: Optional[Callable] = None,
    spa_fallback: bool = True,
) -> bool:
    """
    Serve a static file with security protections.

    Args:
        handler: HTTP request handler instance
        filename: Requested filename/path
        static_dir: Base directory for static files
        add_cors_headers: Optional callback to add CORS headers
        add_security_headers: Optional callback to add security headers
        spa_fallback: Whether to fall back to index.html for missing files

    Returns:
        True if file was served, False on error
    """
    # Validate path
    is_valid, filepath, error = validate_path(filename, static_dir)
    if not is_valid:
        if "traversal" in error or "Symlinks" in error:
            handler.send_error(403, error)
        else:
            handler.send_error(400, error)
        return False

    # Type guard - filepath is guaranteed non-None if is_valid is True
    if filepath is None:
        handler.send_error(500, "Internal error: invalid path state")
        return False

    # Check if file exists
    if not filepath.exists():
        if spa_fallback:
            # Try index.html for SPA routing
            index_path = static_dir / "index.html"
            if index_path.exists():
                filepath = index_path
            else:
                handler.send_error(404, "File not found")
                return False
        else:
            handler.send_error(404, "File not found")
            return False

    # Determine content type
    content_type = get_content_type(str(filepath))

    try:
        content = filepath.read_bytes()

        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(content)))

        # Add optional headers
        if add_cors_headers:
            add_cors_headers()
        if add_security_headers:
            add_security_headers()

        handler.end_headers()
        handler.wfile.write(content)
        return True

    except FileNotFoundError:
        handler.send_error(404, "File not found")
        return False
    except PermissionError:
        handler.send_error(403, "Permission denied")
        return False
    except (IOError, OSError) as e:
        logger.error(f"File read error: {e}")
        handler.send_error(500, "Failed to read file")
        return False
    except (BrokenPipeError, ConnectionResetError):
        # Client disconnected, no response needed
        return False


def serve_audio_file(
    handler: "BaseHTTPRequestHandler",
    debate_id: str,
    audio_store,
    add_cors_headers: Optional[Callable] = None,
) -> bool:
    """
    Serve an audio file for a debate broadcast.

    Args:
        handler: HTTP request handler instance
        debate_id: ID of the debate
        audio_store: AudioFileStore instance
        add_cors_headers: Optional callback to add CORS headers

    Returns:
        True if file was served, False on error
    """
    if not audio_store:
        handler.send_error(503, "Audio storage not available")
        return False

    audio_path = audio_store.get_audio_path(debate_id)
    if not audio_path or not audio_path.exists():
        handler.send_error(404, "Audio not found for this debate")
        return False

    try:
        content = audio_path.read_bytes()

        # Determine audio format from extension
        ext = audio_path.suffix.lower()
        content_type = CONTENT_TYPES.get(ext, "audio/mpeg")

        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(content)))
        handler.send_header("Content-Disposition", f'inline; filename="{debate_id}{ext}"')

        if add_cors_headers:
            add_cors_headers()

        handler.end_headers()
        handler.wfile.write(content)
        return True

    except FileNotFoundError:
        handler.send_error(404, "Audio file not found")
        return False
    except PermissionError:
        handler.send_error(403, "Permission denied")
        return False
    except (IOError, OSError) as e:
        logger.error(f"Audio file read error: {e}")
        handler.send_error(500, "Failed to read audio file")
        return False
    except (BrokenPipeError, ConnectionResetError):
        # Client disconnected
        return False


class StaticFileHandler:
    """
    Reusable static file handler.

    Can be used as a mixin or standalone handler.
    """

    def __init__(
        self,
        static_dir: Path,
        spa_fallback: bool = True,
    ):
        self.static_dir = static_dir
        self.spa_fallback = spa_fallback

    def serve(
        self,
        handler: "BaseHTTPRequestHandler",
        path: str,
        add_cors_headers: Optional[Callable] = None,
        add_security_headers: Optional[Callable] = None,
    ) -> bool:
        """Serve a static file."""
        # Strip leading slash
        filename = path.lstrip("/")

        return serve_static_file(
            handler,
            filename,
            self.static_dir,
            add_cors_headers,
            add_security_headers,
            self.spa_fallback,
        )
