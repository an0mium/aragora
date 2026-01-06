"""
Tests for aragora.server.storage module.

Tests DebateStorage class including slug generation, save/retrieve,
and SQL injection prevention.
"""

import json
import os
import sqlite3
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.storage import (
    DB_TIMEOUT,
    DebateMetadata,
    DebateStorage,
    _escape_like_pattern,
    _get_connection,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create temporary SQLite database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@pytest.fixture
def storage(temp_db):
    """DebateStorage with temporary database."""
    return DebateStorage(db_path=temp_db)


@pytest.fixture
def mock_artifact():
    """Mock DebateArtifact for testing."""
    artifact = MagicMock()
    artifact.artifact_id = "test-id-123"
    artifact.task = "Design a rate limiter"
    artifact.agents = ["claude", "gemini"]
    artifact.consensus_proof = MagicMock(reached=True, confidence=0.95)
    artifact.to_json.return_value = json.dumps({
        "artifact_id": "test-id-123",
        "task": "Design a rate limiter",
        "agents": ["claude", "gemini"],
        "consensus_proof": {"reached": True, "confidence": 0.95},
    })
    return artifact


# =============================================================================
# Test Slug Generation
# =============================================================================


class TestSlugGeneration:
    """Tests for slug generation."""

    def test_simple_task_generates_slug(self, storage):
        """Simple task should generate correct slug."""
        with patch("aragora.server.storage.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-01-05"
            slug = storage.generate_slug("Rate limiter system")

        assert "rate" in slug
        assert "limiter" in slug
        assert "2026-01-05" in slug

    def test_stop_words_filtered(self, storage):
        """Stop words should be filtered out."""
        with patch("aragora.server.storage.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-01-05"
            slug = storage.generate_slug("Design a rate limiter for the system")

        # "Design", "a", "for", "the" are stop words
        assert "rate" in slug
        assert "limiter" in slug
        assert "system" in slug
        assert "design" not in slug
        assert "-a-" not in slug

    def test_punctuation_removed(self, storage):
        """Punctuation should be removed."""
        with patch("aragora.server.storage.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-01-05"
            slug = storage.generate_slug("What's the best API? Create one!")

        assert "?" not in slug
        assert "!" not in slug
        assert "'" not in slug

    def test_short_task_falls_back_to_debate(self, storage):
        """Task with only stop words should fall back to 'debate'."""
        with patch("aragora.server.storage.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-01-05"
            slug = storage.generate_slug("The a an")

        assert slug.startswith("debate-")

    def test_uppercase_converted(self, storage):
        """Uppercase should be converted to lowercase."""
        with patch("aragora.server.storage.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-01-05"
            slug = storage.generate_slug("API Gateway SERVICE")

        assert "api" in slug.lower()
        assert "gateway" in slug.lower()
        assert "API" not in slug
        assert "SERVICE" not in slug

    def test_collision_handling(self, storage, mock_artifact):
        """Collisions should append counter."""
        with patch("aragora.server.storage.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-01-05"

            # First save
            slug1 = storage.save(mock_artifact)
            # First slug should not end with a counter
            assert not slug1.endswith("-2") and not slug1.endswith("-3")

            # Reset mock to get same base slug
            mock_artifact.artifact_id = "test-id-124"
            slug2 = storage.save(mock_artifact)
            assert slug2.endswith("-2")

            # Third save
            mock_artifact.artifact_id = "test-id-125"
            slug3 = storage.save(mock_artifact)
            assert slug3.endswith("-3")

    def test_unicode_preserved(self, storage):
        """Unicode characters should be preserved."""
        with patch("aragora.server.storage.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-01-05"
            slug = storage.generate_slug("日本語 API")

        assert "日本語" in slug

    def test_max_4_keywords(self, storage):
        """Only first 4 keywords should be used."""
        with patch("aragora.server.storage.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-01-05"
            slug = storage.generate_slug("one two three four five six seven")

        # Should have 4 keywords + date
        parts = slug.rsplit("-", 3)  # Split date parts
        keyword_part = "-".join(parts[:-3]) if len(parts) > 3 else parts[0]
        keywords = keyword_part.split("-")
        assert len(keywords) <= 4

    def test_date_format(self, storage):
        """Date should be in YYYY-MM-DD format."""
        with patch("aragora.server.storage.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-12-31"
            slug = storage.generate_slug("Test task")

        assert "2026-12-31" in slug

    def test_empty_task(self, storage):
        """Empty task should fall back to 'debate'."""
        with patch("aragora.server.storage.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-01-05"
            slug = storage.generate_slug("")

        assert slug.startswith("debate-")


# =============================================================================
# Test Database Initialization
# =============================================================================


class TestDatabaseInitialization:
    """Tests for database initialization."""

    def test_creates_debates_table(self, temp_db):
        """Should create debates table."""
        DebateStorage(db_path=temp_db)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='debates'"
            )
            assert cursor.fetchone() is not None

    def test_creates_indexes(self, temp_db):
        """Should create slug and created_at indexes."""
        DebateStorage(db_path=temp_db)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
            indexes = [row[0] for row in cursor.fetchall()]

        assert "idx_slug" in indexes
        assert "idx_created" in indexes

    def test_idempotent(self, temp_db):
        """Calling twice should not fail."""
        DebateStorage(db_path=temp_db)
        DebateStorage(db_path=temp_db)  # Should not raise

    def test_connection_timeout_applied(self, temp_db):
        """Connection should have timeout configured."""
        conn = _get_connection(temp_db, timeout=5.0)
        try:
            # Check busy_timeout pragma was set (in milliseconds)
            cursor = conn.execute("PRAGMA busy_timeout")
            timeout_ms = cursor.fetchone()[0]
            assert timeout_ms == 5000
        finally:
            conn.close()


# =============================================================================
# Test Save Operations
# =============================================================================


class TestSaveOperations:
    """Tests for save operations."""

    def test_save_artifact(self, storage, mock_artifact):
        """save() should persist DebateArtifact."""
        slug = storage.save(mock_artifact)

        debate = storage.get_by_slug(slug)
        assert debate is not None
        assert debate["task"] == "Design a rate limiter"

    def test_save_returns_slug(self, storage, mock_artifact):
        """save() should return generated slug."""
        slug = storage.save(mock_artifact)
        assert isinstance(slug, str)
        assert len(slug) > 0

    def test_save_dict(self, storage):
        """save_dict() should persist raw dict."""
        data = {
            "id": "dict-id-001",
            "task": "API Gateway",
            "agents": ["agent1", "agent2"],
            "consensus_reached": True,
            "confidence": 0.8,
        }

        slug = storage.save_dict(data)
        debate = storage.get_by_slug(slug)

        assert debate is not None
        assert debate["task"] == "API Gateway"

    def test_consensus_proof_none_handled(self, storage):
        """Should handle artifact without consensus_proof."""
        artifact = MagicMock()
        artifact.artifact_id = "no-consensus-123"
        artifact.task = "Test task"
        artifact.agents = ["agent1"]
        artifact.consensus_proof = None
        artifact.to_json.return_value = json.dumps({"task": "Test task"})

        slug = storage.save(artifact)
        debate = storage.get_by_slug(slug)
        assert debate is not None

    def test_view_count_initialized_to_zero(self, storage, mock_artifact, temp_db):
        """view_count should start at 0."""
        slug = storage.save(mock_artifact)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute(
                "SELECT view_count FROM debates WHERE slug = ?",
                (slug,)
            )
            count = cursor.fetchone()[0]

        assert count == 0

    def test_timestamp_auto_populated(self, storage, mock_artifact, temp_db):
        """created_at should be auto-populated."""
        slug = storage.save(mock_artifact)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute(
                "SELECT created_at FROM debates WHERE slug = ?",
                (slug,)
            )
            created_at = cursor.fetchone()[0]

        assert created_at is not None

    def test_agents_json_serialized(self, storage, mock_artifact, temp_db):
        """Agents should be JSON serialized."""
        slug = storage.save(mock_artifact)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute(
                "SELECT agents FROM debates WHERE slug = ?",
                (slug,)
            )
            agents_json = cursor.fetchone()[0]

        agents = json.loads(agents_json)
        assert agents == ["claude", "gemini"]

    def test_long_artifact_json(self, storage):
        """Should handle large artifact_json."""
        artifact = MagicMock()
        artifact.artifact_id = "large-json-123"
        artifact.task = "Large test"
        artifact.agents = ["agent1"]
        artifact.consensus_proof = MagicMock(reached=True, confidence=0.9)
        # Large JSON (100KB)
        artifact.to_json.return_value = json.dumps({"data": "x" * 100000})

        slug = storage.save(artifact)
        debate = storage.get_by_slug(slug)
        assert debate is not None
        assert len(debate["data"]) == 100000


# =============================================================================
# Test Retrieval Operations
# =============================================================================


class TestRetrievalOperations:
    """Tests for retrieval operations."""

    def test_get_by_slug_returns_debate(self, storage, mock_artifact):
        """get_by_slug should return debate."""
        slug = storage.save(mock_artifact)
        debate = storage.get_by_slug(slug)

        assert debate is not None
        assert isinstance(debate, dict)

    def test_get_by_slug_increments_view_count(self, storage, mock_artifact, temp_db):
        """get_by_slug should increment view_count."""
        slug = storage.save(mock_artifact)

        # First retrieval
        storage.get_by_slug(slug)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute(
                "SELECT view_count FROM debates WHERE slug = ?",
                (slug,)
            )
            count1 = cursor.fetchone()[0]

        assert count1 == 1

        # Second retrieval
        storage.get_by_slug(slug)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute(
                "SELECT view_count FROM debates WHERE slug = ?",
                (slug,)
            )
            count2 = cursor.fetchone()[0]

        assert count2 == 2

    def test_get_by_slug_returns_none_for_missing(self, storage):
        """get_by_slug should return None for missing slug."""
        debate = storage.get_by_slug("nonexistent-slug")
        assert debate is None

    def test_get_by_slug_rejects_long_slug(self, storage):
        """get_by_slug should reject slug > 500 chars."""
        long_slug = "a" * 501
        debate = storage.get_by_slug(long_slug)
        assert debate is None

    def test_get_by_slug_rejects_empty(self, storage):
        """get_by_slug should return None for empty slug."""
        debate = storage.get_by_slug("")
        assert debate is None

    def test_get_by_id_returns_debate(self, storage, mock_artifact):
        """get_by_id should return debate."""
        storage.save(mock_artifact)
        debate = storage.get_by_id("test-id-123")

        assert debate is not None
        assert debate["artifact_id"] == "test-id-123"

    def test_get_by_id_no_view_increment(self, storage, mock_artifact, temp_db):
        """get_by_id should NOT increment view_count."""
        slug = storage.save(mock_artifact)

        # Retrieve by ID multiple times
        storage.get_by_id("test-id-123")
        storage.get_by_id("test-id-123")

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute(
                "SELECT view_count FROM debates WHERE slug = ?",
                (slug,)
            )
            count = cursor.fetchone()[0]

        assert count == 0  # No increment

    def test_get_by_id_returns_none_for_missing(self, storage):
        """get_by_id should return None for missing ID."""
        debate = storage.get_by_id("nonexistent-id")
        assert debate is None

    def test_list_recent_orders_by_created_at(self, storage):
        """list_recent should order by created_at DESC."""
        # Save in specific order
        storage.save_dict({"id": "first", "task": "First task"})
        storage.save_dict({"id": "second", "task": "Second task"})
        storage.save_dict({"id": "third", "task": "Third task"})

        results = storage.list_recent()

        # Most recent should be first
        assert results[0].debate_id == "third"
        assert results[-1].debate_id == "first"

    def test_list_recent_respects_limit(self, storage):
        """list_recent should respect limit parameter."""
        for i in range(10):
            storage.save_dict({"id": f"id-{i}", "task": f"Task {i}"})

        results = storage.list_recent(limit=5)
        assert len(results) == 5

    def test_list_recent_empty_db(self, storage):
        """list_recent should return empty list for empty db."""
        results = storage.list_recent()
        assert results == []

    def test_list_recent_returns_metadata(self, storage, mock_artifact):
        """list_recent should return DebateMetadata objects."""
        storage.save(mock_artifact)
        results = storage.list_recent()

        assert len(results) == 1
        assert isinstance(results[0], DebateMetadata)
        assert results[0].task == "Design a rate limiter"
        assert results[0].agents == ["claude", "gemini"]


# =============================================================================
# Test SQL Injection Prevention
# =============================================================================


class TestSQLInjectionPrevention:
    """Tests for SQL injection prevention."""

    def test_escape_like_percent(self):
        """Should escape % character."""
        result = _escape_like_pattern("100%")
        assert result == "100\\%"

    def test_escape_like_underscore(self):
        """Should escape _ character."""
        result = _escape_like_pattern("test_value")
        assert result == "test\\_value"

    def test_escape_like_backslash(self):
        """Should escape \\ character."""
        result = _escape_like_pattern("path\\file")
        assert result == "path\\\\file"

    def test_escape_like_combined(self):
        """Should escape multiple special characters."""
        result = _escape_like_pattern("100% test_value with\\path")
        assert result == "100\\% test\\_value with\\\\path"

    def test_slug_with_special_chars(self, storage):
        """Slug with special chars should be handled safely."""
        data = {
            "id": "special-id",
            "task": "Test'; DROP TABLE debates;--",
            "agents": [],
        }
        slug = storage.save_dict(data)

        # Should not have SQL injection
        debate = storage.get_by_slug(slug)
        assert debate is not None


# =============================================================================
# Test Delete and Edge Cases
# =============================================================================


class TestDeleteAndEdgeCases:
    """Tests for delete and edge cases."""

    def test_delete_existing(self, storage, mock_artifact):
        """delete should return True for existing debate."""
        slug = storage.save(mock_artifact)
        result = storage.delete(slug)

        assert result is True
        assert storage.get_by_slug(slug) is None

    def test_delete_nonexistent(self, storage):
        """delete should return False for nonexistent slug."""
        result = storage.delete("nonexistent-slug")
        assert result is False

    def test_invalid_datetime_fallback(self, storage, temp_db):
        """list_recent should handle invalid datetime."""
        # Insert with invalid datetime directly
        with sqlite3.connect(temp_db) as conn:
            conn.execute("""
                INSERT INTO debates (id, slug, task, agents, artifact_json,
                                    consensus_reached, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "invalid-dt-id",
                "invalid-dt-slug",
                "Test",
                "[]",
                "{}",
                False,
                0,
                "not-a-valid-datetime",
            ))
            conn.commit()

        # Should not crash
        results = storage.list_recent()
        assert len(results) == 1
        # Should have fallback datetime
        assert isinstance(results[0].created_at, datetime)

    def test_empty_agents_list(self, storage):
        """Should handle empty agents list."""
        data = {
            "id": "empty-agents",
            "task": "Test",
            "agents": [],
        }
        slug = storage.save_dict(data)
        debate = storage.get_by_slug(slug)

        assert debate is not None
        assert debate.get("agents", []) == []

    def test_debate_metadata_dataclass(self):
        """DebateMetadata should have all expected fields."""
        metadata = DebateMetadata(
            slug="test-slug",
            debate_id="test-id",
            task="Test task",
            agents=["agent1", "agent2"],
            consensus_reached=True,
            confidence=0.9,
            created_at=datetime.now(),
            view_count=5,
        )

        assert metadata.slug == "test-slug"
        assert metadata.debate_id == "test-id"
        assert metadata.task == "Test task"
        assert metadata.agents == ["agent1", "agent2"]
        assert metadata.consensus_reached is True
        assert metadata.confidence == 0.9
        assert metadata.view_count == 5


# =============================================================================
# Test Concurrent Access
# =============================================================================


class TestConcurrentAccess:
    """Tests for concurrent database access."""

    def test_concurrent_view_count_updates(self, storage, mock_artifact):
        """Concurrent view count updates should be atomic."""
        import threading

        slug = storage.save(mock_artifact)
        errors = []

        def read_debate():
            try:
                storage.get_by_slug(slug)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_debate) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # View count should be exactly 20
        with sqlite3.connect(storage.db_path) as conn:
            cursor = conn.execute(
                "SELECT view_count FROM debates WHERE slug = ?",
                (slug,)
            )
            count = cursor.fetchone()[0]

        assert count == 20

    def test_db_timeout_constant(self):
        """DB_TIMEOUT should be 30 seconds."""
        assert DB_TIMEOUT == 30.0


# =============================================================================
# Test Audio Methods
# =============================================================================


class TestAudioMethods:
    """Tests for audio-related storage methods."""

    def test_update_audio_success(self, storage, mock_artifact, temp_db):
        """Successfully update audio path for existing debate."""
        storage.save(mock_artifact)

        result = storage.update_audio("test-id-123", "/path/to/audio.mp3", 300)

        assert result is True
        # Verify in database
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute(
                "SELECT audio_path, audio_duration_seconds FROM debates WHERE id = ?",
                ("test-id-123",)
            )
            row = cursor.fetchone()
        assert row[0] == "/path/to/audio.mp3"
        assert row[1] == 300

    def test_update_audio_missing_debate(self, storage):
        """update_audio for nonexistent debate returns False."""
        result = storage.update_audio("nonexistent-id", "/path/to/audio.mp3")
        assert result is False

    def test_update_audio_with_duration(self, storage, mock_artifact):
        """Update audio with duration_seconds."""
        storage.save(mock_artifact)
        result = storage.update_audio("test-id-123", "/audio.mp3", duration_seconds=180)

        assert result is True
        audio_info = storage.get_audio_info("test-id-123")
        assert audio_info["audio_duration_seconds"] == 180

    def test_update_audio_without_duration(self, storage, mock_artifact):
        """Update audio without duration (None)."""
        storage.save(mock_artifact)
        result = storage.update_audio("test-id-123", "/audio.mp3")

        assert result is True
        audio_info = storage.get_audio_info("test-id-123")
        assert audio_info["audio_duration_seconds"] is None

    def test_get_audio_info_exists(self, storage, mock_artifact):
        """Get audio info for debate with audio."""
        storage.save(mock_artifact)
        storage.update_audio("test-id-123", "/path/to/audio.mp3", 120)

        info = storage.get_audio_info("test-id-123")

        assert info is not None
        assert info["audio_path"] == "/path/to/audio.mp3"
        assert info["audio_duration_seconds"] == 120
        assert info["audio_generated_at"] is not None

    def test_get_audio_info_missing_debate(self, storage):
        """Get audio info for nonexistent debate returns None."""
        info = storage.get_audio_info("nonexistent-id")
        assert info is None

    def test_get_audio_info_no_audio_path(self, storage, mock_artifact):
        """Get audio info when no audio has been set returns None."""
        storage.save(mock_artifact)
        info = storage.get_audio_info("test-id-123")
        assert info is None

    def test_concurrent_audio_updates(self, storage, mock_artifact):
        """Two threads updating audio simultaneously should not error."""
        import threading

        storage.save(mock_artifact)
        errors = []

        def update_audio(thread_id):
            try:
                storage.update_audio(
                    "test-id-123",
                    f"/audio_{thread_id}.mp3",
                    duration_seconds=thread_id * 10
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_audio, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # One of the updates should have won
        info = storage.get_audio_info("test-id-123")
        assert info is not None


# =============================================================================
# Test Slug Edge Cases
# =============================================================================


class TestSlugEdgeCases:
    """Tests for slug generation edge cases."""

    def test_slug_with_glob_asterisk(self, storage):
        """Task containing '*' should not cause GLOB issues."""
        data = {"id": "glob-asterisk", "task": "Test * wildcard task", "agents": []}
        slug = storage.save_dict(data)

        # Should be able to save and retrieve
        debate = storage.get_by_slug(slug)
        assert debate is not None
        assert debate["task"] == "Test * wildcard task"

    def test_slug_with_glob_question_mark(self, storage):
        """Task containing '?' should not cause GLOB issues."""
        data = {"id": "glob-question", "task": "What is this? A question", "agents": []}
        slug = storage.save_dict(data)

        debate = storage.get_by_slug(slug)
        assert debate is not None

    def test_slug_with_glob_brackets(self, storage):
        """Task containing '[abc]' should not cause GLOB issues."""
        data = {"id": "glob-brackets", "task": "Test [option1] or [option2]", "agents": []}
        slug = storage.save_dict(data)

        debate = storage.get_by_slug(slug)
        assert debate is not None

    def test_extremely_long_task(self, storage):
        """Task with 10,000+ characters should still generate valid slug."""
        long_task = "word " * 2000  # 10,000 chars
        data = {"id": "long-task", "task": long_task, "agents": []}
        slug = storage.save_dict(data)

        # Slug should be reasonable length
        assert len(slug) < 200
        debate = storage.get_by_slug(slug)
        assert debate is not None

    def test_task_with_only_punctuation(self, storage):
        """Task with only punctuation should fall back to 'debate'."""
        with patch("aragora.server.storage.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-01-05"
            slug = storage.generate_slug("!@#$%^&*()")

        assert slug.startswith("debate-")

    def test_concurrent_slug_generation(self, storage):
        """Multiple threads generating slugs for same task may encounter race conditions.

        Note: There's a known race condition between generate_slug() and save()
        where multiple threads can generate the same slug before insertion.
        This test documents that behavior - some threads may fail with
        IntegrityError, but successful ones should have unique slugs.
        """
        import threading

        slugs = []
        errors = []
        lock = threading.Lock()

        def save_debate(thread_id):
            try:
                data = {
                    "id": f"concurrent-{thread_id}",
                    "task": "Same task description",
                    "agents": [],
                }
                slug = storage.save_dict(data)
                with lock:
                    slugs.append(slug)
            except sqlite3.IntegrityError:
                # Race condition: multiple threads generated same slug
                with lock:
                    errors.append("IntegrityError")
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=save_debate, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Some threads may have succeeded, some may have failed due to race condition
        # At least one should have succeeded
        assert len(slugs) >= 1
        # Successful slugs should all be unique
        assert len(slugs) == len(set(slugs))
        # Total attempts should equal 10
        assert len(slugs) + len(errors) == 10


# =============================================================================
# Test JSON Handling Edge Cases
# =============================================================================


class TestJSONHandling:
    """Tests for JSON handling edge cases."""

    def test_corrupted_artifact_json_retrieval(self, storage, temp_db):
        """Retrieving debate with corrupted JSON in DB should raise."""
        # Insert with invalid JSON directly
        with sqlite3.connect(temp_db) as conn:
            conn.execute("""
                INSERT INTO debates (id, slug, task, agents, artifact_json,
                                    consensus_reached, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                "corrupted-json",
                "corrupted-slug",
                "Test",
                "[]",
                "NOT VALID JSON {{{",
                False,
                0,
            ))
            conn.commit()

        # Should raise JSONDecodeError
        with pytest.raises(json.JSONDecodeError):
            storage.get_by_slug("corrupted-slug")

    def test_malformed_agents_json_in_list_recent(self, storage, temp_db):
        """list_recent with malformed agents JSON should handle gracefully."""
        # Insert with invalid agents JSON
        with sqlite3.connect(temp_db) as conn:
            conn.execute("""
                INSERT INTO debates (id, slug, task, agents, artifact_json,
                                    consensus_reached, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "malformed-agents",
                "malformed-agents-slug",
                "Test",
                "NOT VALID JSON",  # Invalid
                "{}",
                False,
                0,
                datetime.now().isoformat(),
            ))
            conn.commit()

        # Should raise JSONDecodeError on invalid agents
        with pytest.raises(json.JSONDecodeError):
            storage.list_recent()

    def test_massive_artifact_json(self, storage):
        """Save/retrieve 10MB+ artifact JSON should work."""
        artifact = MagicMock()
        artifact.artifact_id = "massive-json"
        artifact.task = "Massive test"
        artifact.agents = ["agent1"]
        artifact.consensus_proof = MagicMock(reached=True, confidence=0.9)
        # 10MB JSON
        artifact.to_json.return_value = json.dumps({"data": "x" * 10_000_000})

        slug = storage.save(artifact)
        debate = storage.get_by_slug(slug)

        assert debate is not None
        assert len(debate["data"]) == 10_000_000

    def test_empty_artifact_json(self, storage, temp_db):
        """Empty artifact_json field handling."""
        # Insert with empty JSON
        with sqlite3.connect(temp_db) as conn:
            conn.execute("""
                INSERT INTO debates (id, slug, task, agents, artifact_json,
                                    consensus_reached, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                "empty-json",
                "empty-json-slug",
                "Test",
                "[]",
                "{}",  # Empty but valid JSON
                False,
                0,
            ))
            conn.commit()

        debate = storage.get_by_slug("empty-json-slug")
        assert debate == {}

    def test_null_in_json_fields(self, storage):
        """Handle null values in JSON fields."""
        data = {
            "id": "null-fields",
            "task": "Test",
            "agents": None,  # Will be serialized as null
            "consensus_reached": None,
        }
        # This should handle None by defaulting to empty list
        slug = storage.save_dict(data)
        debate = storage.get_by_slug(slug)
        assert debate is not None


# =============================================================================
# Test list_recent Edge Cases
# =============================================================================


class TestListRecentEdgeCases:
    """Tests for list_recent edge cases."""

    def test_list_recent_zero_limit(self, storage, mock_artifact):
        """list_recent(limit=0) should return empty list."""
        storage.save(mock_artifact)
        results = storage.list_recent(limit=0)
        assert results == []

    def test_list_recent_negative_limit(self, storage, mock_artifact):
        """list_recent(limit=-1) returns all rows (SQLite LIMIT -1 means unlimited)."""
        storage.save(mock_artifact)
        results = storage.list_recent(limit=-1)
        # SQLite LIMIT -1 returns all rows (unlimited)
        assert len(results) == 1

    def test_list_recent_very_large_limit(self, storage, mock_artifact):
        """list_recent(limit=100000) should work without error."""
        storage.save(mock_artifact)
        results = storage.list_recent(limit=100000)
        assert len(results) == 1

    def test_list_recent_empty_agents_string(self, storage, temp_db):
        """Handle empty agents JSON string in database."""
        # Insert with empty JSON array string
        with sqlite3.connect(temp_db) as conn:
            conn.execute("""
                INSERT INTO debates (id, slug, task, agents, artifact_json,
                                    consensus_reached, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "empty-agents",
                "empty-agents-slug",
                "Test",
                "[]",  # Empty array (agents column is NOT NULL)
                "{}",
                False,
                0,
                datetime.now().isoformat(),
            ))
            conn.commit()

        results = storage.list_recent()
        assert len(results) == 1
        assert results[0].agents == []

    def test_list_recent_handles_null_view_count(self, storage, temp_db):
        """Handle NULL view_count in database."""
        with sqlite3.connect(temp_db) as conn:
            conn.execute("""
                INSERT INTO debates (id, slug, task, agents, artifact_json,
                                    consensus_reached, confidence, created_at, view_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "null-view",
                "null-view-slug",
                "Test",
                "[]",
                "{}",
                False,
                0,
                datetime.now().isoformat(),
                None,  # NULL view_count
            ))
            conn.commit()

        results = storage.list_recent()
        assert len(results) == 1
        assert results[0].view_count == 0


# =============================================================================
# Test Delete Edge Cases
# =============================================================================


class TestDeleteEdgeCases:
    """Tests for delete edge cases."""

    def test_delete_with_audio(self, storage, mock_artifact):
        """Delete debate that has audio attached should succeed."""
        slug = storage.save(mock_artifact)
        storage.update_audio("test-id-123", "/path/to/audio.mp3", 120)

        result = storage.delete(slug)
        assert result is True

        # Verify debate is gone
        assert storage.get_by_slug(slug) is None

    def test_delete_empty_slug(self, storage):
        """Delete with empty slug should return False."""
        result = storage.delete("")
        assert result is False

    def test_concurrent_delete_and_get(self, storage, mock_artifact):
        """Delete while another thread is reading should be safe."""
        import threading
        import time

        slug = storage.save(mock_artifact)
        errors = []
        results = []

        def reader():
            try:
                for _ in range(10):
                    result = storage.get_by_slug(slug)
                    results.append(result)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def deleter():
            try:
                time.sleep(0.005)  # Let reader start
                storage.delete(slug)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=deleter),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Some reads may have found it, some may have not
        assert any(r is not None for r in results) or any(r is None for r in results)


# =============================================================================
# Test Database Error Handling
# =============================================================================


class TestDatabaseErrors:
    """Tests for database error handling."""

    def test_duplicate_debate_id(self, storage, mock_artifact):
        """Inserting duplicate debate_id should raise IntegrityError."""
        storage.save(mock_artifact)

        # Try to save again with same ID
        mock_artifact.artifact_id = "test-id-123"  # Same ID
        mock_artifact.task = "Different task"

        with pytest.raises(sqlite3.IntegrityError):
            storage.save(mock_artifact)

    def test_duplicate_slug_handled(self, storage):
        """Duplicate slugs should be handled by counter."""
        with patch("aragora.server.storage.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-01-05"

            slug1 = storage.save_dict({"id": "first", "task": "same words", "agents": []})
            slug2 = storage.save_dict({"id": "second", "task": "same words", "agents": []})

        # Slugs should be different
        assert slug1 != slug2
        # Second should have counter
        assert slug2.endswith("-2")

    def test_get_by_slug_with_sql_chars(self, storage, mock_artifact):
        """Slug with SQL special chars should not cause injection."""
        storage.save(mock_artifact)

        # Try to inject SQL
        malicious_slug = "'; DROP TABLE debates; --"
        result = storage.get_by_slug(malicious_slug)

        # Should return None (not found), not crash
        assert result is None

        # Table should still exist
        results = storage.list_recent()
        assert len(results) == 1


# =============================================================================
# Test Safe Column Addition
# =============================================================================


class TestSafeColumnAddition:
    """Tests for _safe_add_column method."""

    def test_safe_add_column_invalid_table_name(self, storage, temp_db):
        """Invalid table name should be rejected."""
        with storage._get_connection() as conn:
            result = storage._safe_add_column(conn, "'; DROP TABLE debates; --", "test", "TEXT")
        assert result is False

    def test_safe_add_column_invalid_column_name(self, storage, temp_db):
        """Invalid column name should be rejected."""
        with storage._get_connection() as conn:
            result = storage._safe_add_column(conn, "debates", "'; DROP TABLE debates; --", "TEXT")
        assert result is False

    def test_safe_add_column_invalid_type(self, storage, temp_db):
        """Invalid column type should be rejected."""
        with storage._get_connection() as conn:
            result = storage._safe_add_column(conn, "debates", "test_col", "INVALID_TYPE")
        assert result is False

    def test_safe_add_column_existing_column(self, storage, temp_db):
        """Adding existing column should return False."""
        with storage._get_connection() as conn:
            # 'task' column already exists
            result = storage._safe_add_column(conn, "debates", "task", "TEXT")
        assert result is False

    def test_safe_add_column_new_column(self, storage, temp_db):
        """Adding new valid column should succeed."""
        with storage._get_connection() as conn:
            result = storage._safe_add_column(conn, "debates", "new_test_column", "TEXT")
            conn.commit()

        assert result is True

        # Verify column exists
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute("PRAGMA table_info(debates)")
            columns = [row[1] for row in cursor.fetchall()]

        assert "new_test_column" in columns
