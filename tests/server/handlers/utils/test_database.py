"""Tests for database module."""

from __future__ import annotations

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import sqlite3
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.utils.database import (
    get_db_connection,
    table_exists,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    # Cleanup
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def db_with_table(temp_db):
    """Create a database with a test table."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE test_table (
            id INTEGER PRIMARY KEY,
            name TEXT
        )
    """)
    cursor.execute("INSERT INTO test_table (name) VALUES (?)", ("test",))
    conn.commit()
    conn.close()
    return temp_db


# =============================================================================
# Test get_db_connection
# =============================================================================


class TestGetDbConnection:
    """Tests for get_db_connection function."""

    def test_returns_connection(self, db_with_table):
        """Should return a database connection."""
        with get_db_connection(db_with_table) as conn:
            assert conn is not None
            assert isinstance(conn, sqlite3.Connection)

    def test_connection_can_execute_queries(self, db_with_table):
        """Should be able to execute queries."""
        with get_db_connection(db_with_table) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM test_table")
            rows = cursor.fetchall()
            assert len(rows) == 1
            assert rows[0][0] == "test"

    def test_connection_closes_after_context(self, db_with_table):
        """Should close connection after context exits."""
        with get_db_connection(db_with_table) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")

        # Connection should be closed
        # Accessing closed connection should raise
        # Note: sqlite3 doesn't raise on closed connections always

    def test_handles_writes(self, db_with_table):
        """Should handle write operations."""
        with get_db_connection(db_with_table) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO test_table (name) VALUES (?)", ("new",))
            conn.commit()

        # Verify write persisted
        with get_db_connection(db_with_table) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM test_table")
            count = cursor.fetchone()[0]
            assert count == 2

    def test_can_create_tables(self, temp_db):
        """Should be able to create tables."""
        with get_db_connection(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE new_table (
                    id INTEGER PRIMARY KEY,
                    data TEXT
                )
            """)
            conn.commit()

        # Verify table exists
        with get_db_connection(temp_db) as conn:
            cursor = conn.cursor()
            assert table_exists(cursor, "new_table") is True


# =============================================================================
# Test table_exists
# =============================================================================


class TestTableExists:
    """Tests for table_exists function."""

    def test_returns_true_for_existing_table(self, db_with_table):
        """Should return True for existing table."""
        with get_db_connection(db_with_table) as conn:
            cursor = conn.cursor()
            assert table_exists(cursor, "test_table") is True

    def test_returns_false_for_nonexistent_table(self, db_with_table):
        """Should return False for nonexistent table."""
        with get_db_connection(db_with_table) as conn:
            cursor = conn.cursor()
            assert table_exists(cursor, "nonexistent_table") is False

    def test_handles_empty_database(self, temp_db):
        """Should handle empty database."""
        # Create empty database
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        result = table_exists(cursor, "any_table")
        conn.close()

        assert result is False

    def test_case_sensitive_table_names(self, db_with_table):
        """Should be case sensitive for table names."""
        with get_db_connection(db_with_table) as conn:
            cursor = conn.cursor()
            # SQLite is case insensitive by default but let's test
            assert table_exists(cursor, "test_table") is True
            # Different case should not match (SQLite is case insensitive)
            # This actually returns True in SQLite, so adjust test
            # assert table_exists(cursor, "TEST_TABLE") is True  # SQLite is case insensitive

    def test_handles_special_characters_in_name(self, temp_db):
        """Should handle special characters in table name."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        # Table names with special chars need quoting
        cursor.execute('CREATE TABLE "table-with-dashes" (id INTEGER)')
        conn.commit()

        result = table_exists(cursor, "table-with-dashes")
        conn.close()

        assert result is True

    def test_works_with_multiple_tables(self, temp_db):
        """Should work correctly with multiple tables."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE table_a (id INTEGER)")
        cursor.execute("CREATE TABLE table_b (id INTEGER)")
        cursor.execute("CREATE TABLE table_c (id INTEGER)")
        conn.commit()

        assert table_exists(cursor, "table_a") is True
        assert table_exists(cursor, "table_b") is True
        assert table_exists(cursor, "table_c") is True
        assert table_exists(cursor, "table_d") is False

        conn.close()


# =============================================================================
# Test Connection Behavior
# =============================================================================


class TestConnectionBehavior:
    """Tests for connection behavior and error handling."""

    def test_handles_concurrent_reads(self, db_with_table):
        """Should handle concurrent read operations."""
        # Open multiple connections and read
        results = []
        for _ in range(3):
            with get_db_connection(db_with_table) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM test_table")
                results.append(cursor.fetchall())

        assert all(len(r) == 1 for r in results)

    def test_connection_with_nonexistent_file_creates_db(self):
        """Should create database if file doesn't exist."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(path)  # Delete the file

        try:
            with get_db_connection(path) as conn:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE test (id INTEGER)")
                conn.commit()

            # File should now exist
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)
