"""Tests for the calibration database module.

Tests cover:
- CalibrationDatabase class initialization
- Connection context manager
- Transaction context manager
- fetch_one method
- fetch_all method
- execute_write method
- executemany method
"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.ranking.calibration_database import CalibrationDatabase


class TestCalibrationDatabaseInit:
    """Tests for CalibrationDatabase initialization."""

    @pytest.fixture
    def temp_db_dir(self):
        """Create a temporary directory for database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_init_creates_instance(self, temp_db_dir):
        """Should create instance with db_path."""
        db_path = temp_db_dir / "calibration.db"
        with patch("aragora.ranking.calibration_database.DatabaseManager") as mock_dm:
            mock_dm.get_instance.return_value = MagicMock()
            db = CalibrationDatabase(db_path)
            assert db.db_path == db_path

    def test_init_accepts_string_path(self, temp_db_dir):
        """Should accept string path."""
        db_path = str(temp_db_dir / "calibration.db")
        with patch("aragora.ranking.calibration_database.DatabaseManager") as mock_dm:
            mock_dm.get_instance.return_value = MagicMock()
            db = CalibrationDatabase(db_path)
            assert db.db_path == Path(db_path)

    def test_init_uses_database_manager(self, temp_db_dir):
        """Should use DatabaseManager.get_instance."""
        db_path = temp_db_dir / "calibration.db"
        with patch("aragora.ranking.calibration_database.DatabaseManager") as mock_dm:
            mock_dm.get_instance.return_value = MagicMock()
            CalibrationDatabase(db_path)
            mock_dm.get_instance.assert_called_once()

    def test_repr(self, temp_db_dir):
        """Should have repr representation."""
        db_path = temp_db_dir / "calibration.db"
        with patch("aragora.ranking.calibration_database.DatabaseManager") as mock_dm:
            mock_dm.get_instance.return_value = MagicMock()
            db = CalibrationDatabase(db_path)
            assert "CalibrationDatabase" in repr(db)
            assert str(db_path) in repr(db)


class TestCalibrationDatabaseConnection:
    """Tests for connection context manager."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        with patch("aragora.ranking.calibration_database.DatabaseManager") as mock_dm:
            mock_manager = MagicMock()
            mock_conn = MagicMock()
            mock_manager.fresh_connection.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_manager.fresh_connection.return_value.__exit__ = MagicMock(
                return_value=None
            )
            mock_dm.get_instance.return_value = mock_manager
            db = CalibrationDatabase("/tmp/test.db")
            yield db, mock_manager, mock_conn

    def test_connection_context_manager(self, mock_db):
        """Should provide connection via context manager."""
        db, mock_manager, mock_conn = mock_db

        with db.connection() as conn:
            assert conn is not None

        mock_manager.fresh_connection.assert_called()


class TestCalibrationDatabaseTransaction:
    """Tests for transaction context manager."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        with patch("aragora.ranking.calibration_database.DatabaseManager") as mock_dm:
            mock_manager = MagicMock()
            mock_conn = MagicMock()
            mock_manager.fresh_connection.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_manager.fresh_connection.return_value.__exit__ = MagicMock(
                return_value=None
            )
            mock_dm.get_instance.return_value = mock_manager
            db = CalibrationDatabase("/tmp/test.db")
            yield db, mock_manager, mock_conn

    def test_transaction_begins(self, mock_db):
        """Should execute BEGIN on transaction start."""
        db, mock_manager, mock_conn = mock_db

        with db.transaction():
            pass

        # Should execute BEGIN
        mock_conn.execute.assert_any_call("BEGIN")

    def test_transaction_commits_on_success(self, mock_db):
        """Should execute COMMIT on success."""
        db, mock_manager, mock_conn = mock_db

        with db.transaction():
            pass

        # Should execute COMMIT
        mock_conn.execute.assert_any_call("COMMIT")

    def test_transaction_rollbacks_on_exception(self, mock_db):
        """Should execute ROLLBACK on exception."""
        db, mock_manager, mock_conn = mock_db

        with pytest.raises(ValueError):
            with db.transaction():
                raise ValueError("test error")

        # Should execute ROLLBACK
        mock_conn.execute.assert_any_call("ROLLBACK")


class TestCalibrationDatabaseFetchOne:
    """Tests for fetch_one method."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        with patch("aragora.ranking.calibration_database.DatabaseManager") as mock_dm:
            mock_manager = MagicMock()
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.execute.return_value = mock_cursor
            mock_manager.fresh_connection.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_manager.fresh_connection.return_value.__exit__ = MagicMock(
                return_value=None
            )
            mock_dm.get_instance.return_value = mock_manager
            db = CalibrationDatabase("/tmp/test.db")
            yield db, mock_conn, mock_cursor

    def test_fetch_one_executes_query(self, mock_db):
        """Should execute SQL query."""
        db, mock_conn, mock_cursor = mock_db
        mock_cursor.fetchone.return_value = (1, "test")

        db.fetch_one("SELECT * FROM test WHERE id = ?", ("123",))

        mock_conn.execute.assert_called_with("SELECT * FROM test WHERE id = ?", ("123",))

    def test_fetch_one_returns_row(self, mock_db):
        """Should return fetched row."""
        db, mock_conn, mock_cursor = mock_db
        mock_cursor.fetchone.return_value = (1, "test")

        result = db.fetch_one("SELECT * FROM test")

        assert result == (1, "test")

    def test_fetch_one_returns_none_for_no_results(self, mock_db):
        """Should return None when no results."""
        db, mock_conn, mock_cursor = mock_db
        mock_cursor.fetchone.return_value = None

        result = db.fetch_one("SELECT * FROM test")

        assert result is None


class TestCalibrationDatabaseFetchAll:
    """Tests for fetch_all method."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        with patch("aragora.ranking.calibration_database.DatabaseManager") as mock_dm:
            mock_manager = MagicMock()
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.execute.return_value = mock_cursor
            mock_manager.fresh_connection.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_manager.fresh_connection.return_value.__exit__ = MagicMock(
                return_value=None
            )
            mock_dm.get_instance.return_value = mock_manager
            db = CalibrationDatabase("/tmp/test.db")
            yield db, mock_conn, mock_cursor

    def test_fetch_all_executes_query(self, mock_db):
        """Should execute SQL query."""
        db, mock_conn, mock_cursor = mock_db
        mock_cursor.fetchall.return_value = []

        db.fetch_all("SELECT * FROM test ORDER BY id")

        mock_conn.execute.assert_called_with("SELECT * FROM test ORDER BY id", ())

    def test_fetch_all_returns_rows(self, mock_db):
        """Should return all fetched rows."""
        db, mock_conn, mock_cursor = mock_db
        mock_cursor.fetchall.return_value = [(1, "a"), (2, "b"), (3, "c")]

        result = db.fetch_all("SELECT * FROM test")

        assert result == [(1, "a"), (2, "b"), (3, "c")]

    def test_fetch_all_returns_empty_list_for_no_results(self, mock_db):
        """Should return empty list when no results."""
        db, mock_conn, mock_cursor = mock_db
        mock_cursor.fetchall.return_value = []

        result = db.fetch_all("SELECT * FROM test")

        assert result == []


class TestCalibrationDatabaseExecuteWrite:
    """Tests for execute_write method."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        with patch("aragora.ranking.calibration_database.DatabaseManager") as mock_dm:
            mock_manager = MagicMock()
            mock_conn = MagicMock()
            mock_manager.fresh_connection.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_manager.fresh_connection.return_value.__exit__ = MagicMock(
                return_value=None
            )
            mock_dm.get_instance.return_value = mock_manager
            db = CalibrationDatabase("/tmp/test.db")
            yield db, mock_conn

    def test_execute_write_executes_statement(self, mock_db):
        """Should execute SQL statement."""
        db, mock_conn = mock_db

        db.execute_write("INSERT INTO test VALUES (?)", ("value",))

        mock_conn.execute.assert_called_with("INSERT INTO test VALUES (?)", ("value",))


class TestCalibrationDatabaseExecuteMany:
    """Tests for executemany method."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        with patch("aragora.ranking.calibration_database.DatabaseManager") as mock_dm:
            mock_manager = MagicMock()
            mock_conn = MagicMock()
            mock_manager.fresh_connection.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_manager.fresh_connection.return_value.__exit__ = MagicMock(
                return_value=None
            )
            mock_dm.get_instance.return_value = mock_manager
            db = CalibrationDatabase("/tmp/test.db")
            yield db, mock_conn

    def test_executemany_executes_batch(self, mock_db):
        """Should execute statement with multiple parameter sets."""
        db, mock_conn = mock_db
        params_list = [("a",), ("b",), ("c",)]

        db.executemany("INSERT INTO test VALUES (?)", params_list)

        mock_conn.executemany.assert_called_with(
            "INSERT INTO test VALUES (?)", params_list
        )
