"""
Tests for QuickBooks Online connector security.

Tests cover:
- SQL injection prevention in query building
- Input validation for numeric IDs
- Value sanitization for string inputs
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


class TestQBOQuerySanitization:
    """Tests for QBO query value sanitization."""

    @pytest.fixture
    def qbo_connector(self):
        """Create QBO connector instance."""
        from aragora.connectors.accounting import QuickBooksConnector, QBOEnvironment

        return QuickBooksConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            redirect_uri="http://localhost:8080/callback/qbo",
            environment=QBOEnvironment.SANDBOX,
        )

    def test_sanitize_query_value_escapes_single_quotes(self, qbo_connector):
        """Test that single quotes are properly escaped."""
        # Standard SQL injection attempt
        result = qbo_connector._sanitize_query_value("O'Brien")
        assert result == "O''Brien"

    def test_sanitize_query_value_multiple_quotes(self, qbo_connector):
        """Test multiple single quotes are all escaped."""
        result = qbo_connector._sanitize_query_value("Test's 'Value'")
        assert result == "Test''s ''Value''"

    def test_sanitize_query_value_injection_attempt(self, qbo_connector):
        """Test SQL injection attempt is neutralized."""
        # Classic SQL injection payload
        malicious = "'; DROP TABLE Customer; --"
        result = qbo_connector._sanitize_query_value(malicious)
        # Single quote at start becomes '' (escaped), rendering the injection harmless
        assert result == "''; DROP TABLE Customer; --"

    def test_sanitize_query_value_nested_quotes(self, qbo_connector):
        """Test nested quote injection is handled."""
        result = qbo_connector._sanitize_query_value("'''")
        assert result == "''''''"

    def test_sanitize_query_value_converts_non_string(self, qbo_connector):
        """Test non-string values are converted before sanitization."""
        result = qbo_connector._sanitize_query_value(123)
        assert result == "123"

    def test_sanitize_query_value_empty_string(self, qbo_connector):
        """Test empty string is handled."""
        result = qbo_connector._sanitize_query_value("")
        assert result == ""


class TestQBONumericIdValidation:
    """Tests for numeric ID validation."""

    @pytest.fixture
    def qbo_connector(self):
        """Create QBO connector instance."""
        from aragora.connectors.accounting import QuickBooksConnector, QBOEnvironment

        return QuickBooksConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            redirect_uri="http://localhost:8080/callback/qbo",
            environment=QBOEnvironment.SANDBOX,
        )

    def test_validate_numeric_id_valid(self, qbo_connector):
        """Test valid numeric IDs pass validation."""
        assert qbo_connector._validate_numeric_id("123", "customer_id") == "123"
        assert qbo_connector._validate_numeric_id("1", "customer_id") == "1"
        assert qbo_connector._validate_numeric_id("999999999", "customer_id") == "999999999"

    def test_validate_numeric_id_strips_whitespace(self, qbo_connector):
        """Test whitespace is stripped from IDs."""
        assert qbo_connector._validate_numeric_id("  123  ", "customer_id") == "123"
        assert qbo_connector._validate_numeric_id("\t456\n", "customer_id") == "456"

    def test_validate_numeric_id_rejects_injection(self, qbo_connector):
        """Test SQL injection in ID is rejected."""
        with pytest.raises(ValueError, match="must be a numeric ID"):
            qbo_connector._validate_numeric_id("1; DROP TABLE", "customer_id")

    def test_validate_numeric_id_rejects_quotes(self, qbo_connector):
        """Test quote injection in ID is rejected."""
        with pytest.raises(ValueError, match="must be a numeric ID"):
            qbo_connector._validate_numeric_id("1'", "customer_id")

    def test_validate_numeric_id_rejects_letters(self, qbo_connector):
        """Test alphanumeric IDs are rejected."""
        with pytest.raises(ValueError, match="must be a numeric ID"):
            qbo_connector._validate_numeric_id("abc123", "customer_id")

    def test_validate_numeric_id_rejects_empty(self, qbo_connector):
        """Test empty ID is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            qbo_connector._validate_numeric_id("", "customer_id")

    def test_validate_numeric_id_rejects_whitespace_only(self, qbo_connector):
        """Test whitespace-only ID is rejected."""
        with pytest.raises(ValueError, match="must be a numeric ID"):
            qbo_connector._validate_numeric_id("   ", "customer_id")

    def test_validate_numeric_id_rejects_negative(self, qbo_connector):
        """Test negative IDs are rejected."""
        with pytest.raises(ValueError, match="must be a numeric ID"):
            qbo_connector._validate_numeric_id("-123", "customer_id")

    def test_validate_numeric_id_rejects_decimal(self, qbo_connector):
        """Test decimal IDs are rejected."""
        with pytest.raises(ValueError, match="must be a numeric ID"):
            qbo_connector._validate_numeric_id("123.45", "customer_id")


class TestQBOListInvoicesInjectionPrevention:
    """Tests for list_invoices SQL injection prevention."""

    @pytest.fixture
    def qbo_connector(self):
        """Create QBO connector with credentials."""
        from aragora.connectors.accounting import (
            QuickBooksConnector,
            QBOEnvironment,
            QBOCredentials,
        )

        connector = QuickBooksConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            redirect_uri="http://localhost:8080/callback/qbo",
            environment=QBOEnvironment.SANDBOX,
        )
        connector.set_credentials(
            QBOCredentials(
                access_token="test_token",
                refresh_token="test_refresh",
                realm_id="12345",
                expires_at=datetime.now(timezone.utc),
            )
        )
        return connector

    @pytest.mark.asyncio
    async def test_list_invoices_rejects_injection_in_customer_id(self, qbo_connector):
        """Test that SQL injection in customer_id is rejected."""
        with pytest.raises(ValueError, match="must be a numeric ID"):
            await qbo_connector.list_invoices(customer_id="1' OR '1'='1")

    @pytest.mark.asyncio
    async def test_list_invoices_rejects_semicolon_injection(self, qbo_connector):
        """Test that semicolon injection in customer_id is rejected."""
        with pytest.raises(ValueError, match="must be a numeric ID"):
            await qbo_connector.list_invoices(customer_id="1; DELETE FROM Invoice; --")

    @pytest.mark.asyncio
    async def test_list_invoices_accepts_valid_customer_id(self, qbo_connector):
        """Test that valid customer_id passes validation."""
        mock_response = {"QueryResponse": {"Invoice": []}}

        with patch.object(qbo_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            result = await qbo_connector.list_invoices(customer_id="12345")

            assert result == []
            # Verify the query was built correctly
            call_args = mock_request.call_args
            assert "CustomerRef = '12345'" in call_args[0][1]


class TestQBOListAccountsInjectionPrevention:
    """Tests for list_accounts SQL injection prevention."""

    @pytest.fixture
    def qbo_connector(self):
        """Create QBO connector with credentials."""
        from aragora.connectors.accounting import (
            QuickBooksConnector,
            QBOEnvironment,
            QBOCredentials,
        )

        connector = QuickBooksConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            redirect_uri="http://localhost:8080/callback/qbo",
            environment=QBOEnvironment.SANDBOX,
        )
        connector.set_credentials(
            QBOCredentials(
                access_token="test_token",
                refresh_token="test_refresh",
                realm_id="12345",
                expires_at=datetime.now(timezone.utc),
            )
        )
        return connector

    @pytest.mark.asyncio
    async def test_list_accounts_sanitizes_account_type(self, qbo_connector):
        """Test that account_type with quotes is sanitized."""
        mock_response = {"QueryResponse": {"Account": []}}

        with patch.object(qbo_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            # Account type with quote - should be escaped
            result = await qbo_connector.list_accounts(account_type="Income's Test")

            assert result == []
            call_args = mock_request.call_args
            # Verify quotes are escaped
            assert "Income''s Test" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_list_accounts_sanitizes_injection_attempt(self, qbo_connector):
        """Test that SQL injection in account_type is sanitized."""
        mock_response = {"QueryResponse": {"Account": []}}

        with patch.object(qbo_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            # Injection attempt - should be escaped
            await qbo_connector.list_accounts(account_type="'; DROP TABLE Account; --")

            call_args = mock_request.call_args
            # The injection should be escaped, not executed
            query_param = call_args[0][1]
            assert "'''; DROP TABLE Account; --" in query_param

    @pytest.mark.asyncio
    async def test_list_accounts_valid_account_type(self, qbo_connector):
        """Test that valid account_type works correctly."""
        mock_response = {"QueryResponse": {"Account": []}}

        with patch.object(qbo_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            result = await qbo_connector.list_accounts(account_type="Income")

            assert result == []
            call_args = mock_request.call_args
            assert "AccountType = 'Income'" in call_args[0][1]


class TestQBOGetVendorByNameSanitization:
    """Tests for get_vendor_by_name sanitization (existing but verify)."""

    @pytest.fixture
    def qbo_connector(self):
        """Create QBO connector with credentials."""
        from aragora.connectors.accounting import (
            QuickBooksConnector,
            QBOEnvironment,
            QBOCredentials,
        )

        connector = QuickBooksConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            redirect_uri="http://localhost:8080/callback/qbo",
            environment=QBOEnvironment.SANDBOX,
        )
        connector.set_credentials(
            QBOCredentials(
                access_token="test_token",
                refresh_token="test_refresh",
                realm_id="12345",
                expires_at=datetime.now(timezone.utc),
            )
        )
        return connector

    @pytest.mark.asyncio
    async def test_get_vendor_by_name_sanitizes_quotes(self, qbo_connector):
        """Test that vendor name with quotes is sanitized."""
        mock_response = {"QueryResponse": {"Vendor": []}}

        with patch.object(qbo_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            result = await qbo_connector.get_vendor_by_name("O'Brien & Sons")

            assert result is None
            call_args = mock_request.call_args
            # Verify quotes are escaped
            assert "O''Brien & Sons" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_get_vendor_by_name_sanitizes_injection(self, qbo_connector):
        """Test that SQL injection in vendor name is sanitized."""
        mock_response = {"QueryResponse": {"Vendor": []}}

        with patch.object(qbo_connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await qbo_connector.get_vendor_by_name("'; DROP TABLE Vendor; --")

            call_args = mock_request.call_args
            # The injection should be escaped
            assert "'''; DROP TABLE Vendor; --" in call_args[0][1]
