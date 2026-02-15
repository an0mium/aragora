"""Tests for aragora.audience.database module."""

from unittest.mock import patch, MagicMock

import pytest

from aragora.audience.database import AudienceDatabase
from aragora.storage.base_database import BaseDatabase


class TestAudienceDatabase:
    """Tests for AudienceDatabase."""

    def test_is_subclass_of_base_database(self):
        assert issubclass(AudienceDatabase, BaseDatabase)

    @patch("aragora.storage.base_database.DatabaseManager")
    @patch("aragora.storage.base_database.resolve_db_path", side_effect=lambda p: str(p))
    def test_instantiation_with_path(self, mock_resolve, mock_dm, tmp_path):
        db_path = tmp_path / "audience.db"
        mock_dm.get_instance.return_value = MagicMock()
        db = AudienceDatabase(str(db_path))
        assert str(db.db_path) == str(db_path)

    @patch("aragora.storage.base_database.DatabaseManager")
    @patch("aragora.storage.base_database.resolve_db_path", side_effect=lambda p: str(p))
    def test_inherits_fetch_methods(self, mock_resolve, mock_dm, tmp_path):
        db_path = tmp_path / "audience.db"
        mock_dm.get_instance.return_value = MagicMock()
        db = AudienceDatabase(str(db_path))
        assert hasattr(db, "fetch_one")
        assert hasattr(db, "fetch_all")
        assert hasattr(db, "execute_write")
        assert hasattr(db, "connection")
        assert hasattr(db, "transaction")

    @patch("aragora.storage.base_database.DatabaseManager")
    @patch("aragora.storage.base_database.resolve_db_path", side_effect=lambda p: str(p))
    def test_repr(self, mock_resolve, mock_dm, tmp_path):
        db_path = tmp_path / "audience.db"
        mock_dm.get_instance.return_value = MagicMock()
        db = AudienceDatabase(str(db_path))
        r = repr(db)
        assert "AudienceDatabase" in r
        assert "audience.db" in r

    @patch("aragora.storage.base_database.DatabaseManager")
    @patch("aragora.storage.base_database.resolve_db_path", side_effect=lambda p: str(p))
    def test_connection_context_manager_available(self, mock_resolve, mock_dm, tmp_path):
        db_path = tmp_path / "audience.db"
        mock_manager = MagicMock()
        mock_dm.get_instance.return_value = mock_manager
        db = AudienceDatabase(str(db_path))
        # Verify connection is a context manager method
        import inspect

        assert inspect.ismethod(db.connection) or callable(db.connection)
