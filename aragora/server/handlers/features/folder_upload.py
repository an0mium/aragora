"""
Folder upload endpoint handlers.

Endpoints:
- POST /api/documents/folder/scan - Scan a folder and return what would be uploaded
- POST /api/documents/folder/upload - Start folder upload
- GET /api/documents/folder/upload/{folder_id}/status - Get upload progress
- GET /api/documents/folders - List uploaded folder sets
- GET /api/documents/folders/{folder_id} - Get folder details
- DELETE /api/documents/folders/{folder_id} - Delete an uploaded folder set
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    require_user_auth,
    safe_error_message,
)

logger = logging.getLogger(__name__)


class FolderUploadStatus(Enum):
    """Status of a folder upload job."""

    PENDING = "pending"
    SCANNING = "scanning"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class FolderUploadJob:
    """Tracks state of an in-progress folder upload."""

    folder_id: str
    root_path: str
    status: FolderUploadStatus
    created_at: datetime
    updated_at: datetime
    user_id: Optional[str] = None

    # Scan results
    total_files_found: int = 0
    included_count: int = 0
    excluded_count: int = 0
    total_size_bytes: int = 0

    # Upload progress
    files_uploaded: int = 0
    files_failed: int = 0
    bytes_uploaded: int = 0

    # Results
    document_ids: list[str] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    # Configuration used
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "folder_id": self.folder_id,
            "root_path": self.root_path,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "user_id": self.user_id,
            "scan": {
                "total_files_found": self.total_files_found,
                "included_count": self.included_count,
                "excluded_count": self.excluded_count,
                "total_size_bytes": self.total_size_bytes,
            },
            "progress": {
                "files_uploaded": self.files_uploaded,
                "files_failed": self.files_failed,
                "bytes_uploaded": self.bytes_uploaded,
                "percent_complete": (
                    round(self.files_uploaded / self.included_count * 100, 1)
                    if self.included_count > 0
                    else 0
                ),
            },
            "results": {
                "document_ids": self.document_ids,
                "errors": self.errors[-10:],  # Last 10 errors
                "error_count": len(self.errors),
            },
            "config": self.config,
        }


class FolderUploadHandler(BaseHandler):
    """Handler for folder upload endpoints."""

    ROUTES = [
        "/api/documents/folder/scan",
        "/api/documents/folder/upload",
        "/api/documents/folders",
    ]

    # In-memory job storage (would be persisted in production)
    _jobs: dict[str, FolderUploadJob] = {}
    _jobs_lock = threading.Lock()

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Handle /api/documents/folder/upload/{folder_id}/status
        if path.startswith("/api/documents/folder/upload/") and path.endswith("/status"):
            return True
        # Handle /api/documents/folders/{folder_id}
        if path.startswith("/api/documents/folders/") and path.count("/") == 4:
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route GET folder requests."""
        if path == "/api/documents/folders":
            return self._list_folders(query_params)

        # GET /api/documents/folder/upload/{folder_id}/status
        if path.startswith("/api/documents/folder/upload/") and path.endswith("/status"):
            folder_id = path.split("/")[-2]
            return self._get_upload_status(folder_id)

        # GET /api/documents/folders/{folder_id}
        if path.startswith("/api/documents/folders/"):
            folder_id, err = self.extract_path_param(path, 3, "folder_id")
            if err:
                return err
            return self._get_folder(folder_id)

        return None

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route POST folder requests."""
        if path == "/api/documents/folder/scan":
            return self._scan_folder(handler)

        if path == "/api/documents/folder/upload":
            return self._start_upload(handler)

        return None

    def handle_delete(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route DELETE folder requests."""
        if path.startswith("/api/documents/folders/"):
            folder_id, err = self.extract_path_param(path, 3, "folder_id")
            if err:
                return err
            return self._delete_folder(folder_id)
        return None

    @require_user_auth
    @handle_errors("folder scan")
    def _scan_folder(self, handler, user=None) -> HandlerResult:
        """Scan a folder and return what would be uploaded.

        Request body:
        {
            "path": "/absolute/path/to/folder",
            "config": {
                "maxDepth": 10,
                "excludePatterns": ["**/.git/**"],
                "maxFileSizeMb": 100,
                "maxTotalSizeMb": 500,
                "maxFileCount": 1000
            }
        }
        """
        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        folder_path = body.get("path")
        if not folder_path:
            return error_response("Missing required field: path", 400)

        # Validate path exists and is a directory
        path = Path(folder_path)
        if not path.exists():
            return error_response(f"Path does not exist: {folder_path}", 404)
        if not path.is_dir():
            return error_response(f"Path is not a directory: {folder_path}", 400)

        # Build config from request
        config_data = body.get("config", {})

        try:
            from aragora.documents.folder import FolderScanner, FolderUploadConfig

            config = FolderUploadConfig(
                max_depth=config_data.get("maxDepth", 10),
                follow_symlinks=config_data.get("followSymlinks", False),
                exclude_patterns=config_data.get("excludePatterns", [])
                + list(FolderUploadConfig().exclude_patterns),
                include_patterns=config_data.get("includePatterns", []),
                max_file_size_mb=config_data.get("maxFileSizeMb", 100),
                max_total_size_mb=config_data.get("maxTotalSizeMb", 500),
                max_file_count=config_data.get("maxFileCount", 1000),
            )

            scanner = FolderScanner(config)

            # Run scan async
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(scanner.scan(path))
            finally:
                loop.close()

            return json_response(result.to_dict())

        except ImportError as e:
            logger.error(f"Folder scanner not available: {e}")
            return error_response("Folder scanning not available", 503)
        except ValueError as e:
            return error_response(str(e), 400)
        except Exception as e:
            logger.error(f"Folder scan error: {e}")
            return error_response(safe_error_message(e, "Scan"), 500)

    @require_user_auth
    @handle_errors("folder upload")
    def _start_upload(self, handler, user=None) -> HandlerResult:
        """Start an async folder upload.

        Request body:
        {
            "path": "/absolute/path/to/folder",
            "config": {
                "maxDepth": 10,
                "excludePatterns": ["**/.git/**"],
                "maxFileSizeMb": 100,
                "maxTotalSizeMb": 500,
                "maxFileCount": 1000
            }
        }

        Returns:
        {
            "folder_id": "uuid",
            "status": "scanning",
            "message": "Upload started"
        }
        """
        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        folder_path = body.get("path")
        if not folder_path:
            return error_response("Missing required field: path", 400)

        # Validate path
        path = Path(folder_path)
        if not path.exists():
            return error_response(f"Path does not exist: {folder_path}", 404)
        if not path.is_dir():
            return error_response(f"Path is not a directory: {folder_path}", 400)

        # Create job
        folder_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        job = FolderUploadJob(
            folder_id=folder_id,
            root_path=str(path.resolve()),
            status=FolderUploadStatus.PENDING,
            created_at=now,
            updated_at=now,
            user_id=user.user_id if user else None,
            config=body.get("config", {}),
        )

        with FolderUploadHandler._jobs_lock:
            FolderUploadHandler._jobs[folder_id] = job

        # Start async upload in background
        thread = threading.Thread(
            target=self._run_upload_job,
            args=(folder_id, path, body.get("config", {})),
            daemon=True,
        )
        thread.start()

        return json_response(
            {
                "folder_id": folder_id,
                "status": "scanning",
                "message": "Upload started. Poll /api/documents/folder/upload/{folder_id}/status for progress.",
            }
        )

    def _run_upload_job(self, folder_id: str, path: Path, config_data: dict) -> None:
        """Run folder upload job in background thread."""
        try:
            self._update_job_status(folder_id, FolderUploadStatus.SCANNING)

            from aragora.documents.folder import FolderScanner, FolderUploadConfig

            config = FolderUploadConfig(
                max_depth=config_data.get("maxDepth", 10),
                follow_symlinks=config_data.get("followSymlinks", False),
                exclude_patterns=config_data.get("excludePatterns", [])
                + list(FolderUploadConfig().exclude_patterns),
                include_patterns=config_data.get("includePatterns", []),
                max_file_size_mb=config_data.get("maxFileSizeMb", 100),
                max_total_size_mb=config_data.get("maxTotalSizeMb", 500),
                max_file_count=config_data.get("maxFileCount", 1000),
            )

            scanner = FolderScanner(config)

            # Run scan
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                scan_result = loop.run_until_complete(scanner.scan(path))
            finally:
                loop.close()

            # Update job with scan results
            with FolderUploadHandler._jobs_lock:
                if folder_id in FolderUploadHandler._jobs:
                    job = FolderUploadHandler._jobs[folder_id]
                    job.total_files_found = scan_result.total_files_found
                    job.included_count = scan_result.included_count
                    job.excluded_count = scan_result.excluded_count
                    job.total_size_bytes = scan_result.included_size_bytes
                    job.updated_at = datetime.now(timezone.utc)

            if scan_result.included_count == 0:
                self._update_job_status(folder_id, FolderUploadStatus.COMPLETED)
                return

            # Start uploading
            self._update_job_status(folder_id, FolderUploadStatus.UPLOADING)

            store = self.get_document_store()
            if not store:
                self._update_job_error(folder_id, "Document storage not configured")
                self._update_job_status(folder_id, FolderUploadStatus.FAILED)
                return

            try:
                from aragora.server.documents import parse_document
            except ImportError:
                self._update_job_error(folder_id, "Document parsing not available")
                self._update_job_status(folder_id, FolderUploadStatus.FAILED)
                return

            # Upload each file
            for file_info in scan_result.included_files:
                try:
                    file_path = Path(file_info.absolute_path)
                    with open(file_path, "rb") as f:
                        content = f.read()

                    doc = parse_document(content, file_path.name)
                    doc_id = store.add(doc)

                    with FolderUploadHandler._jobs_lock:
                        if folder_id in FolderUploadHandler._jobs:
                            job = FolderUploadHandler._jobs[folder_id]
                            job.files_uploaded += 1
                            job.bytes_uploaded += file_info.size_bytes
                            job.document_ids.append(doc_id)
                            job.updated_at = datetime.now(timezone.utc)

                    logger.debug(f"Uploaded {file_path.name} -> {doc_id}")

                except Exception as e:
                    logger.warning(f"Failed to upload {file_info.path}: {e}")
                    with FolderUploadHandler._jobs_lock:
                        if folder_id in FolderUploadHandler._jobs:
                            job = FolderUploadHandler._jobs[folder_id]
                            job.files_failed += 1
                            job.errors.append(
                                {
                                    "file": file_info.path,
                                    "error": str(e),
                                }
                            )
                            job.updated_at = datetime.now(timezone.utc)

            self._update_job_status(folder_id, FolderUploadStatus.COMPLETED)

        except Exception as e:
            logger.error(f"Folder upload job {folder_id} failed: {e}")
            self._update_job_error(folder_id, str(e))
            self._update_job_status(folder_id, FolderUploadStatus.FAILED)

    def _update_job_status(self, folder_id: str, status: FolderUploadStatus) -> None:
        """Update job status."""
        with FolderUploadHandler._jobs_lock:
            if folder_id in FolderUploadHandler._jobs:
                job = FolderUploadHandler._jobs[folder_id]
                job.status = status
                job.updated_at = datetime.now(timezone.utc)

    def _update_job_error(self, folder_id: str, error: str) -> None:
        """Add error to job."""
        with FolderUploadHandler._jobs_lock:
            if folder_id in FolderUploadHandler._jobs:
                job = FolderUploadHandler._jobs[folder_id]
                job.errors.append({"error": error, "fatal": True})
                job.updated_at = datetime.now(timezone.utc)

    def get_document_store(self):
        """Get document store instance."""
        return self.ctx.get("document_store")

    def _get_upload_status(self, folder_id: str) -> HandlerResult:
        """Get status of a folder upload job."""
        with FolderUploadHandler._jobs_lock:
            job = FolderUploadHandler._jobs.get(folder_id)

        if not job:
            return error_response(f"Folder upload not found: {folder_id}", 404)

        return json_response(job.to_dict())

    def _list_folders(self, query_params: dict) -> HandlerResult:
        """List all folder upload jobs."""
        with FolderUploadHandler._jobs_lock:
            jobs = list(FolderUploadHandler._jobs.values())

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        # Apply limit
        limit = int(query_params.get("limit", 50))
        jobs = jobs[:limit]

        return json_response(
            {
                "folders": [job.to_dict() for job in jobs],
                "count": len(jobs),
            }
        )

    def _get_folder(self, folder_id: str) -> HandlerResult:
        """Get details of a specific folder upload."""
        with FolderUploadHandler._jobs_lock:
            job = FolderUploadHandler._jobs.get(folder_id)

        if not job:
            return error_response(f"Folder not found: {folder_id}", 404)

        return json_response(job.to_dict())

    @require_user_auth
    @handle_errors("folder delete")
    def _delete_folder(self, folder_id: str, handler=None, user=None) -> HandlerResult:
        """Delete a folder upload and optionally its documents."""
        with FolderUploadHandler._jobs_lock:
            job = FolderUploadHandler._jobs.get(folder_id)

            if not job:
                return error_response(f"Folder not found: {folder_id}", 404)

            # Check ownership
            if user and job.user_id and job.user_id != user.user_id:
                return error_response("Not authorized to delete this folder", 403)

            # Remove job
            del FolderUploadHandler._jobs[folder_id]

        logger.info(f"Deleted folder upload: {folder_id}")
        return json_response(
            {
                "success": True,
                "message": f"Folder {folder_id} deleted",
                "documents_deleted": 0,  # Documents are kept by default
            }
        )
