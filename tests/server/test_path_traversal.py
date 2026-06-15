"""
Tests for path traversal protection in storage modules.

Verifies that documents.py and broadcast/storage.py properly
reject malicious IDs that could lead to path traversal attacks.
"""

import pytest
from pathlib import Path
from unittest.mock import patch
import tempfile
import json

from aragora.server.documents import (
    DocumentStore,
    ParsedDocument,
    _validate_doc_id,
    _safe_path,
)
from aragora.broadcast.storage import (
    AudioFileStore,
    AudioMetadata,
    _validate_debate_id,
    VALID_ID_PATTERN,
)


class TestDocumentIdValidation:
    """Test document ID validation for path traversal prevention."""

    def test_valid_simple_id(self) -> None:
        """Simple alphanumeric IDs are valid."""
        assert _validate_doc_id("abc123") is True
        assert _validate_doc_id("test_document") is True
        assert _validate_doc_id("doc-with-dashes") is True

    def test_valid_mixed_case(self) -> None:
        """Mixed case IDs are valid."""
        assert _validate_doc_id("DocID123") is True
        assert _validate_doc_id("ABC_def_123") is True

    def test_rejects_empty(self) -> None:
        """Empty IDs are rejected."""
        assert _validate_doc_id("") is False
        assert _validate_doc_id(None) is False  # type: ignore[arg-type]

    def test_rejects_path_traversal(self) -> None:
        """Path traversal attempts are rejected."""
        assert _validate_doc_id("../secret") is False
        assert _validate_doc_id("..") is False
        assert _validate_doc_id("../../etc/passwd") is False
        assert _validate_doc_id("foo/../bar") is False

    def test_rejects_absolute_paths(self) -> None:
        """Absolute path components are rejected."""
        assert _validate_doc_id("/etc/passwd") is False
        assert _validate_doc_id("C:\\Windows") is False

    def test_rejects_dots(self) -> None:
        """Dots that could be path components are rejected."""
        assert _validate_doc_id(".hidden") is False
        assert _validate_doc_id("file.txt") is False  # Extension dots
        assert _validate_doc_id("..") is False

    def test_rejects_special_chars(self) -> None:
        """Special characters are rejected."""
        assert _validate_doc_id("doc/name") is False
        assert _validate_doc_id("doc\\name") is False
        assert _validate_doc_id("doc:name") is False
        assert _validate_doc_id("doc*name") is False
        assert _validate_doc_id("doc?name") is False
        assert _validate_doc_id("doc<name") is False
        assert _validate_doc_id("doc>name") is False

    def test_rejects_long_ids(self) -> None:
        """Excessively long IDs are rejected."""
        assert _validate_doc_id("a" * 128) is True  # At limit
        assert _validate_doc_id("a" * 129) is False  # Over limit


class TestSafePath:
    """Test safe path construction."""

    def test_returns_path_for_valid_id(self) -> None:
        """Returns path for valid document IDs."""
        storage_dir = Path("/tmp/docs")
        result = _safe_path(storage_dir, "valid_id")
        assert result is not None
        assert result.name == "valid_id.json"

    def test_returns_none_for_invalid_id(self) -> None:
        """Returns None for invalid document IDs."""
        storage_dir = Path("/tmp/docs")
        assert _safe_path(storage_dir, "../secret") is None
        assert _safe_path(storage_dir, "") is None


class TestDebateIdValidation:
    """Test debate ID validation for AudioFileStore."""

    def test_valid_debate_ids(self) -> None:
        """Valid debate IDs pass validation."""
        assert _validate_debate_id("debate-123") is True
        assert _validate_debate_id("debate_456") is True
        assert _validate_debate_id("abc123XYZ") is True

    def test_rejects_path_traversal(self) -> None:
        """Path traversal attempts are rejected."""
        assert _validate_debate_id("../secret") is False
        assert _validate_debate_id("../../etc/passwd") is False
        assert _validate_debate_id("foo/../bar") is False

    def test_rejects_special_chars(self) -> None:
        """Special characters are rejected."""
        assert _validate_debate_id("debate/123") is False
        assert _validate_debate_id("debate.mp3") is False


class TestDocumentStorePathTraversal:
    """Integration tests for DocumentStore path traversal protection."""

    def test_add_rejects_malicious_id(self) -> None:
        """DocumentStore.add rejects documents with malicious IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DocumentStore(storage_dir=Path(tmpdir))

            # Create a document with a malicious ID
            doc = ParsedDocument(
                id="../../../etc/passwd",
                filename="malicious.txt",
                content_type="text/plain",
                text="malicious content",
            )

            with pytest.raises(ValueError, match="Invalid document ID"):
                store.add(doc)

    def test_get_rejects_malicious_id(self) -> None:
        """DocumentStore.get returns None for malicious IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DocumentStore(storage_dir=Path(tmpdir))
            result = store.get("../../../etc/passwd")
            assert result is None

    def test_delete_rejects_malicious_id(self) -> None:
        """DocumentStore.delete returns False for malicious IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DocumentStore(storage_dir=Path(tmpdir))
            result = store.delete("../../../etc/passwd")
            assert result is False

    def test_normal_operations_work(self) -> None:
        """Normal operations still work with valid IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DocumentStore(storage_dir=Path(tmpdir))

            doc = ParsedDocument(
                id="valid_doc_id",
                filename="test.txt",
                content_type="text/plain",
                text="test content",
            )

            # Add should work
            doc_id = store.add(doc)
            assert doc_id == "valid_doc_id"

            # Get should work
            retrieved = store.get(doc_id)
            assert retrieved is not None
            assert retrieved.text == "test content"

            # Delete should work
            deleted = store.delete(doc_id)
            assert deleted is True


class TestAudioFileStorePathTraversal:
    """Integration tests for AudioFileStore path traversal protection."""

    def test_save_bytes_rejects_malicious_id(self) -> None:
        """AudioFileStore.save_bytes returns None for malicious IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AudioFileStore(storage_dir=Path(tmpdir))
            result = store.save_bytes(
                debate_id="../../../etc/passwd",
                audio_data=b"fake audio data",
            )
            assert result is None

    def test_get_path_rejects_malicious_id(self) -> None:
        """AudioFileStore.get_path returns None for malicious IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AudioFileStore(storage_dir=Path(tmpdir))
            result = store.get_path("../../../etc/passwd")
            assert result is None

    def test_get_metadata_rejects_malicious_id(self) -> None:
        """AudioFileStore.get_metadata returns None for malicious IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AudioFileStore(storage_dir=Path(tmpdir))
            result = store.get_metadata("../../../etc/passwd")
            assert result is None

    def test_delete_rejects_malicious_id(self) -> None:
        """AudioFileStore.delete returns False for malicious IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AudioFileStore(storage_dir=Path(tmpdir))
            result = store.delete("../../../etc/passwd")
            assert result is False

    def test_exists_rejects_malicious_id(self) -> None:
        """AudioFileStore.exists returns False for malicious IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AudioFileStore(storage_dir=Path(tmpdir))
            result = store.exists("../../../etc/passwd")
            assert result is False

    def test_rejects_malicious_format(self) -> None:
        """AudioFileStore rejects malicious format strings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AudioFileStore(storage_dir=Path(tmpdir))
            # Attempt path traversal via format parameter
            result = store.save_bytes(
                debate_id="valid_id",
                audio_data=b"fake audio data",
                format="../../../etc/passwd",
            )
            assert result is None

    def test_normal_operations_work(self) -> None:
        """Normal operations still work with valid IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AudioFileStore(storage_dir=Path(tmpdir))

            # Save should work
            path = store.save_bytes(
                debate_id="valid_debate_id",
                audio_data=b"fake audio data",
                task_summary="Test debate",
            )
            assert path is not None
            assert path.exists()

            # get_path should work
            retrieved_path = store.get_path("valid_debate_id")
            assert retrieved_path == path

            # get_metadata should work
            metadata = store.get_metadata("valid_debate_id")
            assert metadata is not None
            assert metadata.task_summary == "Test debate"

            # exists should work
            assert store.exists("valid_debate_id") is True

            # delete should work
            deleted = store.delete("valid_debate_id")
            assert deleted is True
            assert store.exists("valid_debate_id") is False


class TestEdgeCases:
    """Edge case tests for security hardening."""

    def test_unicode_ids_rejected(self) -> None:
        """Unicode characters that might bypass validation are rejected."""
        # Some unicode characters could be normalized to path separators
        assert _validate_doc_id("doc\u2215name") is False  # Division slash
        assert _validate_doc_id("doc\uff0fname") is False  # Fullwidth slash

    def test_null_byte_injection(self) -> None:
        """Null bytes in IDs are rejected."""
        assert _validate_doc_id("doc\x00.json") is False
        assert _validate_debate_id("debate\x00id") is False

    def test_very_long_path_components(self) -> None:
        """Very long IDs are rejected before path operations."""
        long_id = "a" * 1000
        assert _validate_doc_id(long_id) is False
        assert _validate_debate_id(long_id) is False
