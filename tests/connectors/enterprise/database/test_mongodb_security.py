"""
Tests for MongoDB connector security features.

Tests cover:
- Credential isolation from connection strings
- Password masking in logs and representations
- Secure credential passing via kwargs
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Credential Security Tests
# =============================================================================


class TestMongoDBCredentialSecurity:
    """Tests for credential security in MongoDB connector."""

    def test_credentials_not_in_connection_string(self):
        """
        Verify that passwords are not embedded in the connection URI.

        When credentials are provided via the credential provider,
        they should NOT be concatenated into the connection string.
        """
        from aragora.connectors.enterprise.database.mongodb import MongoDBConnector

        connector = MongoDBConnector(
            host="localhost",
            port=27017,
            database="testdb",
        )

        # The connection string built internally should not contain credentials
        # We verify by checking _get_client builds the URI without embedded password
        # The actual connection string format should be: mongodb://host:port/db
        # NOT: mongodb://user:pass@host:port/db

        # Since _get_client is async, we test the logic by examining how it builds URIs
        # The connection_string attribute should not be set with embedded credentials
        assert connector.connection_string is None

    @pytest.mark.asyncio
    async def test_credentials_passed_as_kwargs(self):
        """
        Verify credentials are passed to AsyncIOMotorClient as separate kwargs.

        This ensures credentials stay out of the URI and won't leak in logs/traces.
        """
        from aragora.connectors.enterprise.database.mongodb import MongoDBConnector

        connector = MongoDBConnector(
            host="localhost",
            port=27017,
            database="testdb",
        )

        # Mock the credential provider
        mock_credentials = AsyncMock()
        mock_credentials.get_credential = AsyncMock(
            side_effect=lambda key: {
                "MONGO_USER": "testuser",
                "MONGO_PASSWORD": "supersecretpassword123",
            }.get(key)
        )
        connector.credentials = mock_credentials

        # Mock AsyncIOMotorClient to capture how it's called
        with patch(
            "aragora.connectors.enterprise.database.mongodb.AsyncIOMotorClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.__getitem__ = MagicMock(return_value=MagicMock())
            mock_client_class.return_value = mock_client

            # Patch the import inside the method
            with patch.dict(
                "sys.modules",
                {"motor.motor_asyncio": MagicMock(AsyncIOMotorClient=mock_client_class)},
            ):
                await connector._get_client()

            # Verify AsyncIOMotorClient was called
            mock_client_class.assert_called_once()

            # Get the call arguments
            call_args = mock_client_class.call_args

            # The first positional arg should be the connection string WITHOUT credentials
            conn_str = call_args[0][0]
            assert "supersecretpassword123" not in conn_str
            assert "testuser:supersecretpassword123" not in conn_str
            assert conn_str == "mongodb://localhost:27017/testdb"

            # Credentials should be passed as kwargs
            kwargs = call_args[1]
            assert kwargs.get("username") == "testuser"
            assert kwargs.get("password") == "supersecretpassword123"
            assert kwargs.get("authSource") == "testdb"

    @pytest.mark.asyncio
    async def test_no_auth_kwargs_when_no_credentials(self):
        """
        Verify no auth kwargs are passed when credentials are not provided.
        """
        from aragora.connectors.enterprise.database.mongodb import MongoDBConnector

        connector = MongoDBConnector(
            host="localhost",
            port=27017,
            database="testdb",
        )

        # Mock the credential provider to return None (no credentials)
        mock_credentials = AsyncMock()
        mock_credentials.get_credential = AsyncMock(return_value=None)
        connector.credentials = mock_credentials

        # Mock AsyncIOMotorClient
        with patch(
            "aragora.connectors.enterprise.database.mongodb.AsyncIOMotorClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.__getitem__ = MagicMock(return_value=MagicMock())
            mock_client_class.return_value = mock_client

            with patch.dict(
                "sys.modules",
                {"motor.motor_asyncio": MagicMock(AsyncIOMotorClient=mock_client_class)},
            ):
                await connector._get_client()

            # Get the call arguments
            call_args = mock_client_class.call_args
            kwargs = call_args[1]

            # No auth kwargs should be present
            assert "username" not in kwargs
            assert "password" not in kwargs
            assert "authSource" not in kwargs


# =============================================================================
# Password Masking Tests
# =============================================================================


class TestPasswordMasking:
    """Tests for password masking functionality."""

    def test_mask_connection_string_hides_password(self):
        """
        Test that _mask_connection_string properly masks passwords.
        """
        from aragora.connectors.enterprise.database.mongodb import MongoDBConnector

        # Test standard MongoDB URI
        conn_str = "mongodb://myuser:mysecretpassword@localhost:27017/mydb"
        masked = MongoDBConnector._mask_connection_string(conn_str)
        assert "mysecretpassword" not in masked
        assert "****" in masked
        assert "myuser:" in masked
        assert "@localhost:27017/mydb" in masked

    def test_mask_connection_string_handles_srv(self):
        """
        Test masking works for mongodb+srv:// connection strings.
        """
        from aragora.connectors.enterprise.database.mongodb import MongoDBConnector

        conn_str = "mongodb+srv://admin:verysecret123@cluster.mongodb.net/production"
        masked = MongoDBConnector._mask_connection_string(conn_str)
        assert "verysecret123" not in masked
        assert "****" in masked
        assert "mongodb+srv://admin:" in masked

    def test_mask_connection_string_handles_special_chars(self):
        """
        Test masking handles passwords with special characters.
        """
        from aragora.connectors.enterprise.database.mongodb import MongoDBConnector

        # Password with special chars that might be URL-encoded
        conn_str = "mongodb://user:p%40ssw0rd%21@localhost:27017/db"
        masked = MongoDBConnector._mask_connection_string(conn_str)
        assert "p%40ssw0rd%21" not in masked
        assert "****" in masked

    def test_mask_connection_string_no_credentials(self):
        """
        Test masking leaves strings without credentials unchanged.
        """
        from aragora.connectors.enterprise.database.mongodb import MongoDBConnector

        conn_str = "mongodb://localhost:27017/mydb"
        masked = MongoDBConnector._mask_connection_string(conn_str)
        # Should be unchanged since there's no password to mask
        assert masked == conn_str

    def test_mask_connection_string_empty_string(self):
        """
        Test masking handles empty strings.
        """
        from aragora.connectors.enterprise.database.mongodb import MongoDBConnector

        masked = MongoDBConnector._mask_connection_string("")
        assert masked == ""


# =============================================================================
# __repr__ Security Tests
# =============================================================================


class TestReprSecurity:
    """Tests for __repr__ security."""

    def test_password_not_in_repr(self):
        """
        Verify __repr__ doesn't expose passwords in connection strings.
        """
        from aragora.connectors.enterprise.database.mongodb import MongoDBConnector

        # Create connector with a connection string containing a password
        connector = MongoDBConnector(
            host="localhost",
            port=27017,
            database="testdb",
            connection_string="mongodb://admin:supersecret@localhost:27017/testdb",
        )

        repr_str = repr(connector)

        # Password should NOT appear in repr
        assert "supersecret" not in repr_str
        # But the masked version should be present
        assert "****" in repr_str
        # Other info should be present
        assert "localhost" in repr_str
        assert "testdb" in repr_str

    def test_repr_without_connection_string(self):
        """
        Verify __repr__ works when no connection string is provided.
        """
        from aragora.connectors.enterprise.database.mongodb import MongoDBConnector

        connector = MongoDBConnector(
            host="myhost.example.com",
            port=27017,
            database="production",
            collections=["users", "orders"],
        )

        repr_str = repr(connector)

        assert "myhost.example.com" in repr_str
        assert "27017" in repr_str
        assert "production" in repr_str
        assert "users" in repr_str
        assert "orders" in repr_str
        assert "connection_string=None" in repr_str

    def test_repr_never_contains_raw_password(self):
        """
        Comprehensive test that repr never leaks password patterns.
        """
        from aragora.connectors.enterprise.database.mongodb import MongoDBConnector

        passwords = [
            "simple123",
            "p@ssw0rd!",
            "very-long-password-with-many-characters",
            "pass%20word",  # URL-encoded space
            "123456",
        ]

        for password in passwords:
            connector = MongoDBConnector(
                host="localhost",
                database="db",
                connection_string=f"mongodb://user:{password}@localhost:27017/db",
            )

            repr_str = repr(connector)
            assert password not in repr_str, f"Password '{password}' leaked in repr"


# =============================================================================
# Integration Security Tests
# =============================================================================


class TestSecurityIntegration:
    """Integration tests for security features."""

    @pytest.mark.asyncio
    async def test_logging_does_not_leak_credentials(self):
        """
        Verify that logging statements use masked connection strings.
        """
        import logging
        from aragora.connectors.enterprise.database.mongodb import MongoDBConnector

        connector = MongoDBConnector(
            host="localhost",
            port=27017,
            database="testdb",
            connection_string="mongodb://user:secretpass@localhost:27017/testdb",
        )

        # Capture log output
        log_records = []

        class LogCapture(logging.Handler):
            def emit(self, record):
                log_records.append(record.getMessage())

        logger = logging.getLogger("aragora.connectors.enterprise.database.mongodb")
        handler = LogCapture()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.DEBUG)

        try:
            # Mock AsyncIOMotorClient
            with patch(
                "aragora.connectors.enterprise.database.mongodb.AsyncIOMotorClient"
            ) as mock_client_class:
                mock_client = MagicMock()
                mock_client.__getitem__ = MagicMock(return_value=MagicMock())
                mock_client_class.return_value = mock_client

                with patch.dict(
                    "sys.modules",
                    {"motor.motor_asyncio": MagicMock(AsyncIOMotorClient=mock_client_class)},
                ):
                    await connector._get_client()

            # Check no log message contains the raw password
            for message in log_records:
                assert "secretpass" not in message, f"Password leaked in log: {message}"

        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)

    def test_str_representation_safe(self):
        """
        Verify that str() on the connector is also safe.
        """
        from aragora.connectors.enterprise.database.mongodb import MongoDBConnector

        connector = MongoDBConnector(
            host="localhost",
            database="db",
            connection_string="mongodb://admin:topsecret@localhost:27017/db",
        )

        # str() typically falls back to __repr__ if __str__ not defined
        str_repr = str(connector)
        assert "topsecret" not in str_repr

    def test_connector_attributes_dont_leak_password(self):
        """
        Verify that accessible attributes don't contain plaintext passwords.
        """
        from aragora.connectors.enterprise.database.mongodb import MongoDBConnector

        connector = MongoDBConnector(
            host="localhost",
            database="db",
            connection_string="mongodb://admin:mysecret@localhost:27017/db",
        )

        # These attributes should be safe to access/log
        safe_attrs = [
            connector.host,
            connector.port,
            connector.database_name,
            connector.name,
            connector.connector_id,
        ]

        for attr_value in safe_attrs:
            str_value = str(attr_value)
            assert "mysecret" not in str_value
