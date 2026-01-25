"""
Tests for enhanced backup verification features.

Tests:
- Schema validation (columns, types, constraints, indexes)
- Referential integrity verification
- Per-table checksums
- Comprehensive verification combining all checks
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from aragora.backup.manager import (
    BackupManager,
    BackupMetadata,
    BackupStatus,
    BackupType,
    ComprehensiveVerificationResult,
    IntegrityResult,
    RetentionPolicy,
    SchemaValidationResult,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_db(temp_dir: Path) -> Path:
    """Create a sample database with tables, foreign keys, and indexes."""
    db_path = temp_dir / "sample.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON")

    # Create users table
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create posts table with foreign key to users
    cursor.execute("""
        CREATE TABLE posts (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Create comments table with foreign keys to posts and users
    cursor.execute("""
        CREATE TABLE comments (
            id INTEGER PRIMARY KEY,
            post_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            body TEXT NOT NULL,
            FOREIGN KEY (post_id) REFERENCES posts(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX idx_posts_user ON posts(user_id)")
    cursor.execute("CREATE INDEX idx_comments_post ON comments(post_id)")
    cursor.execute("CREATE UNIQUE INDEX idx_users_email ON users(email)")

    # Insert sample data
    cursor.executemany(
        "INSERT INTO users (name, email) VALUES (?, ?)",
        [
            ("Alice", "alice@example.com"),
            ("Bob", "bob@example.com"),
            ("Charlie", "charlie@example.com"),
        ],
    )

    cursor.executemany(
        "INSERT INTO posts (user_id, title, content) VALUES (?, ?, ?)",
        [
            (1, "First Post", "Hello world"),
            (1, "Second Post", "More content"),
            (2, "Bob's Post", "Bob writes stuff"),
        ],
    )

    cursor.executemany(
        "INSERT INTO comments (post_id, user_id, body) VALUES (?, ?, ?)",
        [
            (1, 2, "Great post!"),
            (1, 3, "Thanks for sharing"),
            (2, 3, "Interesting"),
        ],
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def backup_manager(temp_dir: Path) -> BackupManager:
    """Create a backup manager instance."""
    backup_dir = temp_dir / "backups"
    return BackupManager(
        backup_dir=backup_dir,
        retention_policy=RetentionPolicy(min_backups=1),
        compression=True,
        verify_after_backup=True,
        metrics_enabled=False,
    )


class TestSchemaInfo:
    """Tests for schema information extraction."""

    def test_get_schema_info_tables(self, backup_manager: BackupManager, sample_db: Path):
        """Test that schema info includes all tables."""
        schema_info = backup_manager._get_schema_info(sample_db)

        assert "users" in schema_info
        assert "posts" in schema_info
        assert "comments" in schema_info

    def test_get_schema_info_columns(self, backup_manager: BackupManager, sample_db: Path):
        """Test that schema info includes correct columns."""
        schema_info = backup_manager._get_schema_info(sample_db)

        users_cols = {c["name"] for c in schema_info["users"]["columns"]}
        assert users_cols == {"id", "name", "email", "created_at"}

        posts_cols = {c["name"] for c in schema_info["posts"]["columns"]}
        assert posts_cols == {"id", "user_id", "title", "content", "created_at"}

    def test_get_schema_info_column_types(self, backup_manager: BackupManager, sample_db: Path):
        """Test that schema info includes correct column types."""
        schema_info = backup_manager._get_schema_info(sample_db)

        users_cols = {c["name"]: c for c in schema_info["users"]["columns"]}
        assert users_cols["id"]["type"] == "INTEGER"
        assert users_cols["name"]["type"] == "TEXT"
        assert users_cols["id"]["pk"] is True
        assert users_cols["name"]["notnull"] is True

    def test_get_schema_info_foreign_keys(self, backup_manager: BackupManager, sample_db: Path):
        """Test that schema info includes foreign keys."""
        schema_info = backup_manager._get_schema_info(sample_db)

        posts_fks = schema_info["posts"]["foreign_keys"]
        assert len(posts_fks) == 1
        assert posts_fks[0] == ("user_id", "users", "id")

        comments_fks = schema_info["comments"]["foreign_keys"]
        assert len(comments_fks) == 2

    def test_get_schema_info_indexes(self, backup_manager: BackupManager, sample_db: Path):
        """Test that schema info includes indexes."""
        schema_info = backup_manager._get_schema_info(sample_db)

        posts_indexes = {idx[0] for idx in schema_info["posts"]["indexes"]}
        assert "idx_posts_user" in posts_indexes

        users_indexes = {idx[0] for idx in schema_info["users"]["indexes"]}
        assert "idx_users_email" in users_indexes


class TestSchemaHash:
    """Tests for schema hash computation."""

    def test_schema_hash_deterministic(self, backup_manager: BackupManager, sample_db: Path):
        """Test that schema hash is deterministic."""
        hash1 = backup_manager._compute_schema_hash(sample_db)
        hash2 = backup_manager._compute_schema_hash(sample_db)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest

    def test_schema_hash_changes_on_alter(self, backup_manager: BackupManager, sample_db: Path):
        """Test that schema hash changes when schema changes."""
        hash_before = backup_manager._compute_schema_hash(sample_db)

        # Add a new column
        conn = sqlite3.connect(str(sample_db))
        conn.execute("ALTER TABLE users ADD COLUMN age INTEGER")
        conn.close()

        hash_after = backup_manager._compute_schema_hash(sample_db)

        assert hash_before != hash_after

    def test_schema_hash_changes_on_new_table(self, backup_manager: BackupManager, sample_db: Path):
        """Test that schema hash changes when new table is added."""
        hash_before = backup_manager._compute_schema_hash(sample_db)

        # Add a new table
        conn = sqlite3.connect(str(sample_db))
        conn.execute("CREATE TABLE tags (id INTEGER PRIMARY KEY, name TEXT)")
        conn.close()

        hash_after = backup_manager._compute_schema_hash(sample_db)

        assert hash_before != hash_after


class TestSchemaValidation:
    """Tests for schema validation."""

    def test_validate_schema_identical(self, backup_manager: BackupManager, sample_db: Path):
        """Test schema validation with identical schemas."""
        # Create backup metadata
        schema_hash = backup_manager._compute_schema_hash(sample_db)
        tables, _ = backup_manager._get_database_info(sample_db)

        backup_meta = BackupMetadata(
            id="test",
            created_at=datetime.now(timezone.utc),
            backup_type=BackupType.FULL,
            status=BackupStatus.COMPLETED,
            source_path=str(sample_db),
            backup_path=str(sample_db),
            tables=tables,
            schema_hash=schema_hash,
        )

        result = backup_manager._validate_schema(sample_db, backup_meta)

        assert result.valid is True
        assert result.tables_match is True
        assert result.columns_match is True
        assert len(result.missing_tables) == 0
        assert len(result.extra_tables) == 0

    def test_validate_schema_missing_table(self, backup_manager: BackupManager, sample_db: Path):
        """Test schema validation detects missing tables."""
        tables, _ = backup_manager._get_database_info(sample_db)

        # Add a fake expected table
        backup_meta = BackupMetadata(
            id="test",
            created_at=datetime.now(timezone.utc),
            backup_type=BackupType.FULL,
            status=BackupStatus.COMPLETED,
            source_path=str(sample_db),
            backup_path=str(sample_db),
            tables=tables + ["nonexistent_table"],
        )

        result = backup_manager._validate_schema(sample_db, backup_meta)

        assert result.valid is False
        assert result.tables_match is False
        assert "nonexistent_table" in result.missing_tables

    def test_validate_schema_extra_table(self, backup_manager: BackupManager, sample_db: Path):
        """Test schema validation detects extra tables."""
        tables, _ = backup_manager._get_database_info(sample_db)

        # Remove a table from expected
        backup_meta = BackupMetadata(
            id="test",
            created_at=datetime.now(timezone.utc),
            backup_type=BackupType.FULL,
            status=BackupStatus.COMPLETED,
            source_path=str(sample_db),
            backup_path=str(sample_db),
            tables=[t for t in tables if t != "comments"],
        )

        result = backup_manager._validate_schema(sample_db, backup_meta)

        assert result.valid is False
        assert result.tables_match is False
        assert "comments" in result.extra_tables


class TestReferentialIntegrity:
    """Tests for referential integrity verification."""

    def test_integrity_valid_database(self, backup_manager: BackupManager, sample_db: Path):
        """Test integrity check on valid database."""
        result = backup_manager._verify_referential_integrity(sample_db)

        assert result.valid is True
        assert result.foreign_keys_valid is True
        assert len(result.orphaned_records) == 0
        assert len(result.foreign_key_errors) == 0

    def test_integrity_detects_orphaned_records(
        self, backup_manager: BackupManager, sample_db: Path
    ):
        """Test that integrity check detects orphaned records."""
        conn = sqlite3.connect(str(sample_db))
        # Disable foreign keys to insert invalid data
        conn.execute("PRAGMA foreign_keys = OFF")
        # Insert a post referencing non-existent user
        conn.execute(
            "INSERT INTO posts (user_id, title, content) VALUES (999, 'Orphan', 'Invalid')"
        )
        conn.commit()
        conn.close()

        result = backup_manager._verify_referential_integrity(sample_db)

        assert result.valid is False
        assert result.foreign_keys_valid is False
        assert len(result.foreign_key_errors) > 0
        assert "posts" in result.orphaned_records

    def test_integrity_basic_check(self, backup_manager: BackupManager, sample_db: Path):
        """Test basic SQLite integrity check passes."""
        result = backup_manager._verify_referential_integrity(sample_db)

        assert result.valid is True
        assert len(result.data_type_errors) == 0


class TestTableChecksums:
    """Tests for per-table checksums."""

    def test_compute_table_checksums(self, backup_manager: BackupManager, sample_db: Path):
        """Test computing checksums for all tables."""
        checksums = backup_manager._compute_table_checksums(sample_db)

        assert "users" in checksums
        assert "posts" in checksums
        assert "comments" in checksums

        # Checksums should be 64-char hex strings (SHA-256)
        for table, checksum in checksums.items():
            assert len(checksum) == 64 or checksum == "", f"Invalid checksum for {table}"

    def test_checksums_deterministic(self, backup_manager: BackupManager, sample_db: Path):
        """Test that checksums are deterministic."""
        checksums1 = backup_manager._compute_table_checksums(sample_db)
        checksums2 = backup_manager._compute_table_checksums(sample_db)

        assert checksums1 == checksums2

    def test_checksums_change_on_data_change(self, backup_manager: BackupManager, sample_db: Path):
        """Test that checksums change when data changes."""
        checksums_before = backup_manager._compute_table_checksums(sample_db)

        # Modify data
        conn = sqlite3.connect(str(sample_db))
        conn.execute("UPDATE users SET name = 'Modified' WHERE id = 1")
        conn.commit()
        conn.close()

        checksums_after = backup_manager._compute_table_checksums(sample_db)

        assert checksums_before["users"] != checksums_after["users"]
        # Other tables should be unchanged
        assert checksums_before["posts"] == checksums_after["posts"]

    def test_checksums_change_on_insert(self, backup_manager: BackupManager, sample_db: Path):
        """Test that checksums change when row is inserted."""
        checksums_before = backup_manager._compute_table_checksums(sample_db)

        # Insert data
        conn = sqlite3.connect(str(sample_db))
        conn.execute("INSERT INTO users (name, email) VALUES ('New', 'new@example.com')")
        conn.commit()
        conn.close()

        checksums_after = backup_manager._compute_table_checksums(sample_db)

        assert checksums_before["users"] != checksums_after["users"]

    def test_checksums_change_on_delete(self, backup_manager: BackupManager, sample_db: Path):
        """Test that checksums change when row is deleted."""
        checksums_before = backup_manager._compute_table_checksums(sample_db)

        # Delete data (remove a comment to avoid FK issues)
        conn = sqlite3.connect(str(sample_db))
        conn.execute("DELETE FROM comments WHERE id = 1")
        conn.commit()
        conn.close()

        checksums_after = backup_manager._compute_table_checksums(sample_db)

        assert checksums_before["comments"] != checksums_after["comments"]


class TestComprehensiveVerification:
    """Tests for comprehensive verification."""

    def test_comprehensive_verification_valid_backup(
        self, backup_manager: BackupManager, sample_db: Path
    ):
        """Test comprehensive verification of a valid backup."""
        # Create a backup
        backup_meta = backup_manager.create_backup(sample_db)

        result = backup_manager.verify_restore_comprehensive(backup_meta.id, backup_meta)

        assert result.verified is True
        assert result.basic_verification.verified is True
        assert result.schema_validation is not None
        assert result.schema_validation.valid is True
        assert result.integrity_check is not None
        assert result.integrity_check.valid is True
        assert result.table_checksums_valid is True
        assert len(result.all_errors) == 0

    def test_comprehensive_verification_returns_all_checks(
        self, backup_manager: BackupManager, sample_db: Path
    ):
        """Test that comprehensive verification includes all check types."""
        backup_meta = backup_manager.create_backup(sample_db)
        result = backup_manager.verify_restore_comprehensive(backup_meta.id, backup_meta)

        assert isinstance(result, ComprehensiveVerificationResult)
        assert result.basic_verification is not None
        assert result.schema_validation is not None
        assert result.integrity_check is not None
        assert result.duration_seconds > 0

    def test_comprehensive_verification_to_dict(
        self, backup_manager: BackupManager, sample_db: Path
    ):
        """Test that comprehensive result can be serialized to dict."""
        backup_meta = backup_manager.create_backup(sample_db)
        result = backup_manager.verify_restore_comprehensive(backup_meta.id, backup_meta)

        result_dict = result.to_dict()

        assert "backup_id" in result_dict
        assert "verified" in result_dict
        assert "basic_verification" in result_dict
        assert "schema_validation" in result_dict
        assert "integrity_check" in result_dict
        assert "table_checksums_valid" in result_dict
        assert "verified_at" in result_dict

    def test_comprehensive_verification_nonexistent_backup(self, backup_manager: BackupManager):
        """Test comprehensive verification of non-existent backup."""
        result = backup_manager.verify_restore_comprehensive("nonexistent")

        assert result.verified is False
        assert "not found" in result.all_errors[0].lower()


class TestBackupWithEnhancedFields:
    """Tests for backup creation with enhanced fields."""

    def test_backup_includes_schema_hash(self, backup_manager: BackupManager, sample_db: Path):
        """Test that backup metadata includes schema hash."""
        backup_meta = backup_manager.create_backup(sample_db)

        assert backup_meta.schema_hash != ""
        assert len(backup_meta.schema_hash) == 64

    def test_backup_includes_table_checksums(self, backup_manager: BackupManager, sample_db: Path):
        """Test that backup metadata includes table checksums."""
        backup_meta = backup_manager.create_backup(sample_db)

        assert len(backup_meta.table_checksums) > 0
        assert "users" in backup_meta.table_checksums
        assert "posts" in backup_meta.table_checksums

    def test_backup_includes_foreign_keys(self, backup_manager: BackupManager, sample_db: Path):
        """Test that backup metadata includes foreign key definitions."""
        backup_meta = backup_manager.create_backup(sample_db)

        assert len(backup_meta.foreign_keys) > 0
        # Check posts->users FK exists
        fk_tables = [fk[0] for fk in backup_meta.foreign_keys]
        assert "posts" in fk_tables
        assert "comments" in fk_tables

    def test_backup_includes_indexes(self, backup_manager: BackupManager, sample_db: Path):
        """Test that backup metadata includes index definitions."""
        backup_meta = backup_manager.create_backup(sample_db)

        assert len(backup_meta.indexes) > 0
        idx_names = [idx[0] for idx in backup_meta.indexes]
        assert "idx_posts_user" in idx_names
        assert "idx_users_email" in idx_names


class TestBackupMetadataSerialization:
    """Tests for BackupMetadata serialization with new fields."""

    def test_metadata_to_dict_includes_new_fields(
        self, backup_manager: BackupManager, sample_db: Path
    ):
        """Test that to_dict includes enhanced fields."""
        backup_meta = backup_manager.create_backup(sample_db)
        data = backup_meta.to_dict()

        assert "schema_hash" in data
        assert "table_checksums" in data
        assert "foreign_keys" in data
        assert "indexes" in data

    def test_metadata_from_dict_restores_new_fields(
        self, backup_manager: BackupManager, sample_db: Path
    ):
        """Test that from_dict restores enhanced fields."""
        backup_meta = backup_manager.create_backup(sample_db)
        data = backup_meta.to_dict()

        restored = BackupMetadata.from_dict(data)

        assert restored.schema_hash == backup_meta.schema_hash
        assert restored.table_checksums == backup_meta.table_checksums
        assert len(restored.foreign_keys) == len(backup_meta.foreign_keys)
        assert len(restored.indexes) == len(backup_meta.indexes)

    def test_metadata_roundtrip_preserves_data(
        self, backup_manager: BackupManager, sample_db: Path
    ):
        """Test that to_dict -> from_dict preserves all data."""
        backup_meta = backup_manager.create_backup(sample_db)

        data = backup_meta.to_dict()
        restored = BackupMetadata.from_dict(data)
        data2 = restored.to_dict()

        # Compare serialized forms
        assert data["schema_hash"] == data2["schema_hash"]
        assert data["table_checksums"] == data2["table_checksums"]
        assert data["foreign_keys"] == data2["foreign_keys"]
        assert data["indexes"] == data2["indexes"]


class TestVerificationResultDataclasses:
    """Tests for new verification result dataclasses."""

    def test_schema_validation_result_defaults(self):
        """Test SchemaValidationResult default values."""
        result = SchemaValidationResult(
            valid=True,
            tables_match=True,
            columns_match=True,
            types_match=True,
            constraints_match=True,
            indexes_match=True,
        )

        assert result.missing_tables == []
        assert result.extra_tables == []
        assert result.column_mismatches == []

    def test_integrity_result_defaults(self):
        """Test IntegrityResult default values."""
        result = IntegrityResult(
            valid=True,
            foreign_keys_valid=True,
        )

        assert result.orphaned_records == {}
        assert result.foreign_key_errors == []
        assert result.data_type_errors == []
        assert result.null_constraint_violations == []

    def test_comprehensive_result_aggregates_errors(self):
        """Test that ComprehensiveVerificationResult aggregates errors."""
        from aragora.backup.manager import VerificationResult

        basic = VerificationResult(
            backup_id="test",
            verified=False,
            checksum_valid=False,
            restore_tested=False,
            tables_valid=False,
            row_counts_valid=False,
            errors=["Basic error"],
        )

        result = ComprehensiveVerificationResult(
            backup_id="test",
            verified=False,
            basic_verification=basic,
        )
        result.all_errors.extend(basic.errors)
        result.all_errors.append("Schema error")

        assert len(result.all_errors) == 2
        assert "Basic error" in result.all_errors
        assert "Schema error" in result.all_errors


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_database(self, backup_manager: BackupManager, temp_dir: Path):
        """Test verification of empty database."""
        db_path = temp_dir / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()

        backup_meta = backup_manager.create_backup(db_path)
        result = backup_manager.verify_restore_comprehensive(backup_meta.id, backup_meta)

        assert result.verified is True
        assert result.integrity_check.valid is True

    def test_database_with_no_foreign_keys(self, backup_manager: BackupManager, temp_dir: Path):
        """Test verification of database without foreign keys."""
        db_path = temp_dir / "no_fk.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO items (name) VALUES ('test')")
        conn.commit()
        conn.close()

        backup_meta = backup_manager.create_backup(db_path)
        result = backup_manager.verify_restore_comprehensive(backup_meta.id, backup_meta)

        assert result.verified is True
        assert len(backup_meta.foreign_keys) == 0

    def test_database_with_blob_data(self, backup_manager: BackupManager, temp_dir: Path):
        """Test checksum computation with BLOB data."""
        db_path = temp_dir / "blob.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE files (id INTEGER PRIMARY KEY, data BLOB)")
        conn.execute("INSERT INTO files (data) VALUES (?)", (b"\x00\x01\x02\x03",))
        conn.commit()
        conn.close()

        checksums = backup_manager._compute_table_checksums(db_path)

        assert "files" in checksums
        assert checksums["files"] != ""

    def test_database_with_null_values(self, backup_manager: BackupManager, temp_dir: Path):
        """Test checksum computation with NULL values."""
        db_path = temp_dir / "nulls.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO items (name) VALUES (NULL)")
        conn.execute("INSERT INTO items (name) VALUES ('test')")
        conn.commit()
        conn.close()

        checksums = backup_manager._compute_table_checksums(db_path)

        assert "items" in checksums
        assert checksums["items"] != ""

    def test_multiple_foreign_keys_same_table(self, backup_manager: BackupManager, temp_dir: Path):
        """Test database with multiple FKs to same table."""
        db_path = temp_dir / "multi_fk.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("""
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                sender_id INTEGER REFERENCES users(id),
                receiver_id INTEGER REFERENCES users(id),
                content TEXT
            )
        """)
        conn.commit()
        conn.close()

        schema_info = backup_manager._get_schema_info(db_path)

        assert len(schema_info["messages"]["foreign_keys"]) == 2
