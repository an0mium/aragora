"""
Smart Upload Handler - Auto-detect file types and route to appropriate processors.

Routes:
- POST /api/upload/smart - Upload and auto-process files
- POST /api/upload/batch - Batch upload with per-file processing options
- GET  /api/upload/status/{id} - Check processing status
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import mimetypes
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import handler base
try:
    from ..base import (
        BaseHandler,
        HandlerResult,
        auto_error_response,  # noqa: F401
        error_response,
        json_response,
    )
    from ..utils.rate_limit import rate_limit  # noqa: F401
    from aragora.rbac.decorators import require_permission

    HANDLER_BASE_AVAILABLE = True
except ImportError:
    HANDLER_BASE_AVAILABLE = False
    logger.warning(
        "Handler base not available - SmartUploadHandler will have limited functionality"
    )


class FileCategory(str, Enum):
    """Categories of files for processing."""

    CODE = "code"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    DATA = "data"
    ARCHIVE = "archive"
    UNKNOWN = "unknown"


class ProcessingAction(str, Enum):
    """Processing actions based on file category."""

    INDEX = "index"  # Repository/code indexing
    EXTRACT = "extract"  # Document text extraction
    TRANSCRIBE = "transcribe"  # Audio/video transcription
    OCR = "ocr"  # Image text extraction
    PARSE = "parse"  # Data file parsing
    EXPAND = "expand"  # Archive extraction
    SKIP = "skip"  # No processing


# File type detection patterns
FILE_PATTERNS: Dict[FileCategory, Dict[str, List[str]]] = {
    FileCategory.CODE: {
        "extensions": [
            ".py",
            ".ts",
            ".tsx",
            ".js",
            ".jsx",
            ".go",
            ".rs",
            ".java",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".r",
            ".jl",
            ".lua",
            ".sh",
            ".bash",
            ".zsh",
            ".ps1",
            ".sql",
            ".graphql",
            ".proto",
            ".ex",
            ".exs",
        ],
        "mime_patterns": ["text/x-", "application/x-python", "application/javascript"],
    },
    FileCategory.DOCUMENT: {
        "extensions": [
            ".pdf",
            ".docx",
            ".doc",
            ".txt",
            ".md",
            ".markdown",
            ".rtf",
            ".odt",
            ".pptx",
            ".ppt",
            ".xlsx",
            ".xls",
            ".epub",
            ".mobi",
        ],
        "mime_patterns": [
            "application/pdf",
            "application/vnd.openxmlformats",
            "application/msword",
            "text/plain",
            "text/markdown",
        ],
    },
    FileCategory.AUDIO: {
        "extensions": [".mp3", ".m4a", ".wav", ".webm", ".ogg", ".flac", ".aac", ".wma"],
        "mime_patterns": ["audio/"],
    },
    FileCategory.VIDEO: {
        "extensions": [".mp4", ".webm", ".mov", ".mkv", ".avi", ".wmv", ".m4v"],
        "mime_patterns": ["video/"],
    },
    FileCategory.IMAGE: {
        "extensions": [".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff", ".bmp", ".svg", ".heic"],
        "mime_patterns": ["image/"],
    },
    FileCategory.DATA: {
        "extensions": [".json", ".xml", ".csv", ".yaml", ".yml", ".toml", ".ini", ".conf"],
        "mime_patterns": ["application/json", "application/xml", "text/csv", "text/yaml"],
    },
    FileCategory.ARCHIVE: {
        "extensions": [".zip", ".tar", ".gz", ".tgz", ".rar", ".7z", ".bz2", ".xz"],
        "mime_patterns": ["application/zip", "application/x-tar", "application/gzip"],
    },
}

# Map categories to default processing actions
CATEGORY_ACTIONS: Dict[FileCategory, ProcessingAction] = {
    FileCategory.CODE: ProcessingAction.INDEX,
    FileCategory.DOCUMENT: ProcessingAction.EXTRACT,
    FileCategory.AUDIO: ProcessingAction.TRANSCRIBE,
    FileCategory.VIDEO: ProcessingAction.TRANSCRIBE,
    FileCategory.IMAGE: ProcessingAction.OCR,
    FileCategory.DATA: ProcessingAction.PARSE,
    FileCategory.ARCHIVE: ProcessingAction.EXPAND,
    FileCategory.UNKNOWN: ProcessingAction.SKIP,
}


@dataclass
class UploadResult:
    """Result of a file upload and processing."""

    id: str
    filename: str
    size: int
    category: FileCategory
    action: ProcessingAction
    status: str  # pending, processing, completed, failed
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# In-memory storage for upload status (would use Redis/DB in production)
_upload_results: Dict[str, UploadResult] = {}


def detect_file_category(filename: str, mime_type: Optional[str] = None) -> FileCategory:
    """
    Detect file category from filename and optional MIME type.

    Args:
        filename: Name of the file
        mime_type: Optional MIME type hint

    Returns:
        Detected FileCategory
    """
    ext = Path(filename).suffix.lower()

    # Check extension-based patterns first
    for category, patterns in FILE_PATTERNS.items():
        if ext in patterns["extensions"]:
            return category

    # Fallback to MIME type patterns
    if mime_type:
        for category, patterns in FILE_PATTERNS.items():
            for pattern in patterns["mime_patterns"]:
                if mime_type.startswith(pattern):
                    return category

    # Try guessing MIME type from extension
    guessed_mime, _ = mimetypes.guess_type(filename)
    if guessed_mime:
        for category, patterns in FILE_PATTERNS.items():
            for pattern in patterns["mime_patterns"]:
                if guessed_mime.startswith(pattern):
                    return category

    return FileCategory.UNKNOWN


def get_processing_action(category: FileCategory) -> ProcessingAction:
    """Get the default processing action for a category."""
    return CATEGORY_ACTIONS.get(category, ProcessingAction.SKIP)


async def process_file(
    file_content: bytes,
    filename: str,
    category: FileCategory,
    action: ProcessingAction,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Process a file based on its category and action.

    Args:
        file_content: Raw file bytes
        filename: Original filename
        category: Detected file category
        action: Processing action to perform
        options: Optional processing options

    Returns:
        Processing result dict
    """
    options = options or {}
    result: Dict[str, Any] = {
        "filename": filename,
        "category": category.value,
        "action": action.value,
        "size": len(file_content),
    }

    try:
        if action == ProcessingAction.TRANSCRIBE:
            result.update(await _transcribe_audio_video(file_content, filename, options))

        elif action == ProcessingAction.EXTRACT:
            result.update(await _extract_document_text(file_content, filename, options))

        elif action == ProcessingAction.OCR:
            result.update(await _ocr_image(file_content, filename, options))

        elif action == ProcessingAction.PARSE:
            result.update(await _parse_data_file(file_content, filename, options))

        elif action == ProcessingAction.INDEX:
            result.update(await _index_code_file(file_content, filename, options))

        elif action == ProcessingAction.EXPAND:
            result.update(await _expand_archive(file_content, filename, options))

        else:
            # Skip or unknown - just store the file
            result["stored"] = True

    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        result["error"] = str(e)

    return result


async def _transcribe_audio_video(
    content: bytes,
    filename: str,
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """Transcribe audio/video using Whisper connector."""
    try:
        from aragora.connectors.whisper import WhisperConnector

        connector = WhisperConnector()

        # Write to temp file for processing
        with tempfile.NamedTemporaryFile(
            suffix=Path(filename).suffix,
            delete=False,
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Transcribe
            transcription = await connector.transcribe(  # type: ignore[call-arg]
                audio_path=tmp_path,
                language=options.get("language"),
                prompt=options.get("prompt"),
            )

            return {
                "transcription": transcription.text,
                "duration": transcription.duration,  # type: ignore[attr-defined]
                "language": transcription.language,
                "segments": [
                    {
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                    }
                    for s in (transcription.segments or [])
                ],
            }
        finally:
            os.unlink(tmp_path)

    except ImportError:
        # Fallback to simple metadata
        return {"transcription": "[Whisper connector not available]", "segments": []}


async def _extract_document_text(
    content: bytes,
    filename: str,
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract text from documents."""
    ext = Path(filename).suffix.lower()
    text = ""

    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF

            pdf = fitz.open(stream=content, filetype="pdf")
            text = "\n\n".join(page.get_text() for page in pdf)
            page_count = len(pdf)
            return {
                "text": text,
                "page_count": page_count,
                "word_count": len(text.split()),
            }
        except ImportError:
            return {"text": "[PDF extraction requires PyMuPDF]"}

    elif ext in (".txt", ".md", ".markdown"):
        text = content.decode("utf-8", errors="replace")
        return {"text": text, "word_count": len(text.split())}

    elif ext in (".docx",):
        try:
            from docx import Document

            doc = Document(io.BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs)
            return {"text": text, "word_count": len(text.split())}
        except ImportError:
            return {"text": "[DOCX extraction requires python-docx]"}

    return {"text": text or "[Unsupported document format]"}


async def _ocr_image(
    content: bytes,
    filename: str,
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract text from images using OCR."""
    try:
        from aragora.connectors.enterprise.documents.ocr import OCRConnector

        connector = OCRConnector()
        result = await connector.extract_text(content, filename)
        return {
            "text": result.text,
            "confidence": result.confidence,
            "regions": result.regions,
        }
    except ImportError:
        # Try tesseract directly
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(io.BytesIO(content))
            text = pytesseract.image_to_string(img, lang=options.get("language", "eng"))
            return {"text": text}
        except ImportError:
            return {"text": "[OCR requires tesseract or OCRConnector]"}


async def _parse_data_file(
    content: bytes,
    filename: str,
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """Parse data files (JSON, CSV, YAML, etc.)."""
    import csv
    import json

    ext = Path(filename).suffix.lower()
    text_content = content.decode("utf-8", errors="replace")

    if ext == ".json":
        try:
            data = json.loads(text_content)
            return {
                "parsed": True,
                "type": "json",
                "record_count": len(data) if isinstance(data, list) else 1,
                "preview": str(data)[:500],
            }
        except json.JSONDecodeError as e:
            return {"parsed": False, "error": str(e)}

    elif ext == ".csv":
        try:
            reader = csv.DictReader(io.StringIO(text_content))
            rows = list(reader)
            return {
                "parsed": True,
                "type": "csv",
                "columns": reader.fieldnames,
                "record_count": len(rows),
            }
        except Exception as e:
            return {"parsed": False, "error": str(e)}

    elif ext in (".yaml", ".yml"):
        try:
            import yaml

            data = yaml.safe_load(text_content)
            return {
                "parsed": True,
                "type": "yaml",
                "preview": str(data)[:500],
            }
        except ImportError:
            return {"parsed": False, "error": "YAML parsing requires PyYAML"}

    return {"parsed": False, "type": ext, "text": text_content[:1000]}


async def _index_code_file(
    content: bytes,
    filename: str,
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """Index a code file for search/analysis."""
    text_content = content.decode("utf-8", errors="replace")
    lines = text_content.split("\n")

    # Simple code analysis
    result = {
        "indexed": True,
        "line_count": len(lines),
        "char_count": len(text_content),
        "language": _detect_language(filename),
    }

    # Try to extract symbols (functions, classes)
    symbols = _extract_symbols(text_content, filename)
    if symbols:
        result["symbols"] = symbols

    return result


def _detect_language(filename: str) -> str:
    """Detect programming language from filename."""
    ext = Path(filename).suffix.lower()
    language_map = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
    }
    return language_map.get(ext, "unknown")


def _extract_symbols(content: str, filename: str) -> List[Dict[str, Any]]:
    """Extract function/class symbols from code."""
    import re

    symbols = []
    language = _detect_language(filename)

    if language == "python":
        # Match class and function definitions
        for match in re.finditer(r"^(class|def)\s+(\w+)", content, re.MULTILINE):
            symbols.append(
                {
                    "type": "class" if match.group(1) == "class" else "function",
                    "name": match.group(2),
                }
            )

    elif language in ("javascript", "typescript"):
        # Match function/class declarations
        for match in re.finditer(
            r"^(?:export\s+)?(?:async\s+)?(?:function|class)\s+(\w+)",
            content,
            re.MULTILINE,
        ):
            symbols.append({"type": "function", "name": match.group(1)})

    return symbols[:50]  # Limit to first 50 symbols


async def _expand_archive(
    content: bytes,
    filename: str,
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """List contents of an archive (don't extract for security)."""
    import zipfile
    import tarfile

    ext = Path(filename).suffix.lower()
    files = []

    if ext == ".zip":
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                for info in zf.infolist():
                    files.append(
                        {
                            "name": info.filename,
                            "size": info.file_size,
                            "compressed": info.compress_size,
                        }
                    )
        except Exception as e:
            return {"error": str(e)}

    elif ext in (".tar", ".gz", ".tgz"):
        try:
            mode = "r:gz" if ext in (".gz", ".tgz") else "r"
            with tarfile.open(fileobj=io.BytesIO(content), mode=mode) as tf:  # type: ignore[call-overload]
                for member in tf.getmembers():
                    files.append(
                        {
                            "name": member.name,
                            "size": member.size,
                        }
                    )
        except Exception as e:
            return {"error": str(e)}

    return {
        "expanded": True,
        "file_count": len(files),
        "files": files[:100],  # Limit listing
    }


async def smart_upload(
    file_content: bytes,
    filename: str,
    mime_type: Optional[str] = None,
    override_action: Optional[ProcessingAction] = None,
    options: Optional[Dict[str, Any]] = None,
) -> UploadResult:
    """
    Smart upload with auto-detection and processing.

    Args:
        file_content: Raw file bytes
        filename: Original filename
        mime_type: Optional MIME type hint
        override_action: Override the auto-detected action
        options: Processing options

    Returns:
        UploadResult with processing status
    """
    # Generate unique ID
    content_hash = hashlib.sha256(file_content).hexdigest()[:16]
    upload_id = f"{uuid.uuid4().hex[:8]}_{content_hash}"

    # Detect category and action
    category = detect_file_category(filename, mime_type)
    action = override_action or get_processing_action(category)

    # Create result entry
    result = UploadResult(
        id=upload_id,
        filename=filename,
        size=len(file_content),
        category=category,
        action=action,
        status="processing",
    )
    _upload_results[upload_id] = result

    try:
        # Process the file
        processing_result = await process_file(
            file_content,
            filename,
            category,
            action,
            options,
        )

        result.status = "completed"
        result.completed_at = time.time()
        result.result = processing_result

    except Exception as e:
        logger.error(f"Smart upload failed for {filename}: {e}")
        result.status = "failed"
        result.error = str(e)

    return result


def get_upload_status(upload_id: str) -> Optional[UploadResult]:
    """Get the status of an upload by ID."""
    return _upload_results.get(upload_id)


# HTTP Handler
if HANDLER_BASE_AVAILABLE:

    class SmartUploadHandler(BaseHandler):
        """HTTP handler for smart file uploads."""

        ROUTES = [
            "/api/v1/upload/smart",
            "/api/v1/upload/batch",
            "/api/v1/upload/status",
        ]

        def can_handle(self, path: str, method: str = "GET") -> bool:
            """Check if this handler can process the given path."""
            return path.startswith("/api/v1/upload/")

        def handle(
            self,
            path: str,
            query_params: Dict[str, Any],
            handler: Any,
        ) -> Optional[HandlerResult]:
            """Route upload requests."""
            if path.startswith("/api/v1/upload/status/"):
                upload_id = path.split("/")[-1]
                return self._get_status(upload_id)

            # POST endpoints handled by handle_post
            if handler.command != "POST":
                return error_response("Method not allowed", 405)

            return None

        def handle_post(
            self,
            path: str,
            body: Dict[str, Any],
            handler: Any,
        ) -> Optional[HandlerResult]:
            """Handle POST uploads."""

            if path == "/api/v1/upload/smart":
                return self._handle_smart_upload(body, handler)

            elif path == "/api/v1/upload/batch":
                return self._handle_batch_upload(body, handler)

            return error_response("Not found", 404)

        @require_permission("upload:create")
        def _handle_smart_upload(
            self,
            body: Dict[str, Any],
            handler: Any,
        ) -> HandlerResult:
            """Handle single smart upload."""
            # Get file from multipart form or base64 body
            file_content = body.get("content")
            filename = body.get("filename", "unknown")
            mime_type = body.get("mime_type")
            action = body.get("action")
            options = body.get("options", {})

            if not file_content:
                return error_response("No file content provided", 400)

            # Decode base64 if needed
            if isinstance(file_content, str):
                import base64

                file_content = base64.b64decode(file_content)

            # Process
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            override_action = ProcessingAction(action) if action else None
            result = loop.run_until_complete(
                smart_upload(file_content, filename, mime_type, override_action, options)
            )

            return json_response(
                {
                    "id": result.id,
                    "filename": result.filename,
                    "size": result.size,
                    "category": result.category.value,
                    "action": result.action.value,
                    "status": result.status,
                    "result": result.result,
                    "error": result.error,
                }
            )

        @require_permission("upload:create")
        def _handle_batch_upload(
            self,
            body: Dict[str, Any],
            handler: Any,
        ) -> HandlerResult:
            """Handle batch upload."""
            files = body.get("files", [])
            if not files:
                return error_response("No files provided", 400)

            # Process all files
            import asyncio
            import base64

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            async def process_all():
                results = []
                for file_info in files:
                    content = file_info.get("content", "")
                    if isinstance(content, str):
                        content = base64.b64decode(content)

                    result = await smart_upload(
                        content,
                        file_info.get("filename", "unknown"),
                        file_info.get("mime_type"),
                        ProcessingAction(file_info["action"]) if file_info.get("action") else None,
                        file_info.get("options"),
                    )
                    results.append(
                        {
                            "id": result.id,
                            "filename": result.filename,
                            "status": result.status,
                            "category": result.category.value,
                            "error": result.error,
                        }
                    )
                return results

            results = loop.run_until_complete(process_all())
            return json_response({"files": results, "count": len(results)})

        def _get_status(self, upload_id: str) -> HandlerResult:
            """Get upload status."""
            result = get_upload_status(upload_id)
            if not result:
                return error_response("Upload not found", 404)

            return json_response(
                {
                    "id": result.id,
                    "filename": result.filename,
                    "size": result.size,
                    "category": result.category.value,
                    "action": result.action.value,
                    "status": result.status,
                    "created_at": result.created_at,
                    "completed_at": result.completed_at,
                    "result": result.result,
                    "error": result.error,
                }
            )
