"""
Comprehensive tests for Plaid Bank Connector.

Tests cover:
- Dataclass serialization
- Enum values
- Account and transaction operations
- Categorization mapping
- Mock data generation
"""

import pytest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.connectors.accounting.plaid import (
    PlaidConnector,
    PlaidCredentials,
    PlaidEnvironment,
    PlaidError,
    BankAccount,
    BankTransaction,
    AccountType,
    TransactionCategory,
    CategoryMapping,
    get_mock_accounts,
    get_mock_transactions,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestPlaidEnvironment:
    """Tests for PlaidEnvironment enum."""

    def test_environment_values(self):
        """Test all environment values."""
        assert PlaidEnvironment.SANDBOX.value == "sandbox"
        assert PlaidEnvironment.DEVELOPMENT.value == "development"
        assert PlaidEnvironment.PRODUCTION.value == "production"


class TestTransactionCategory:
    """Tests for TransactionCategory enum."""

    def test_category_values(self):
        """Test all category values."""
        assert TransactionCategory.INCOME.value == "income"
        assert TransactionCategory.EXPENSE.value == "expense"
        assert TransactionCategory.TRANSFER.value == "transfer"
        assert TransactionCategory.PAYROLL.value == "payroll"
        assert TransactionCategory.LOAN.value == "loan"
        assert TransactionCategory.REFUND.value == "refund"
        assert TransactionCategory.INVESTMENT.value == "investment"
        assert TransactionCategory.UNKNOWN.value == "unknown"


class TestAccountType:
    """Tests for AccountType enum."""

    def test_account_type_values(self):
        """Test all account type values."""
        assert AccountType.CHECKING.value == "checking"
        assert AccountType.SAVINGS.value == "savings"
        assert AccountType.CREDIT.value == "credit"
        assert AccountType.INVESTMENT.value == "investment"
        assert AccountType.LOAN.value == "loan"
        assert AccountType.OTHER.value == "other"


# =============================================================================
# PlaidCredentials Tests
# =============================================================================


class TestPlaidCredentials:
    """Tests for PlaidCredentials dataclass."""

    def test_credentials_creation(self):
        """Test credentials initialization."""
        creds = PlaidCredentials(
            access_token="access-token-123",
            item_id="item-456",
            institution_id="ins_789",
            institution_name="Test Bank",
            user_id="user_001",
            tenant_id="tenant_001",
        )
        assert creds.access_token == "access-token-123"
        assert creds.item_id == "item-456"
        assert creds.institution_name == "Test Bank"
        assert creds.last_sync is None

    def test_credentials_to_dict_masks_token(self):
        """Test credentials serialization masks access token."""
        creds = PlaidCredentials(
            access_token="super-secret-access-token-12345",
            item_id="item-456",
            institution_id="ins_789",
            institution_name="Test Bank",
            user_id="user_001",
            tenant_id="tenant_001",
        )
        data = creds.to_dict()
        assert data["access_token"] == "super-secret-access-..."
        assert len(data["access_token"]) == 23  # 20 chars + "..."
        assert data["item_id"] == "item-456"
        assert data["institution_name"] == "Test Bank"

    def test_credentials_with_last_sync(self):
        """Test credentials with last sync time."""
        now = datetime.now(timezone.utc)
        creds = PlaidCredentials(
            access_token="token",
            item_id="item",
            institution_id="ins",
            institution_name="Bank",
            user_id="user",
            tenant_id="tenant",
            last_sync=now,
        )
        data = creds.to_dict()
        assert data["last_sync"] is not None
        assert now.isoformat()[:19] in data["last_sync"]


# =============================================================================
# BankAccount Tests
# =============================================================================


class TestBankAccount:
    """Tests for BankAccount dataclass."""

    def test_account_creation(self):
        """Test bank account initialization."""
        account = BankAccount(
            account_id="acc_001",
            name="My Checking",
            official_name="Personal Checking Account",
            account_type=AccountType.CHECKING,
            subtype="checking",
            mask="1234",
            current_balance=Decimal("5000.00"),
            available_balance=Decimal("4800.00"),
            limit=None,
            institution_name="Test Bank",
        )
        assert account.account_id == "acc_001"
        assert account.name == "My Checking"
        assert account.account_type == AccountType.CHECKING
        assert account.current_balance == Decimal("5000.00")

    def test_account_to_dict(self):
        """Test bank account serialization."""
        account = BankAccount(
            account_id="acc_002",
            name="Credit Card",
            official_name="Visa Signature",
            account_type=AccountType.CREDIT,
            subtype="credit card",
            mask="5678",
            current_balance=Decimal("1500.50"),
            available_balance=None,
            limit=Decimal("10000.00"),
        )
        data = account.to_dict()
        assert data["account_id"] == "acc_002"
        assert data["account_type"] == "credit"
        assert data["current_balance"] == 1500.50
        assert data["limit"] == 10000.0
        assert data["available_balance"] is None


# =============================================================================
# BankTransaction Tests
# =============================================================================


class TestBankTransaction:
    """Tests for BankTransaction dataclass."""

    def test_transaction_creation(self):
        """Test transaction initialization."""
        today = date.today()
        txn = BankTransaction(
            transaction_id="txn_001",
            account_id="acc_001",
            amount=Decimal("99.99"),
            date=today,
            name="Amazon.com",
            merchant_name="Amazon",
            pending=False,
            category=["Shopping", "Online"],
        )
        assert txn.transaction_id == "txn_001"
        assert txn.amount == Decimal("99.99")
        assert txn.name == "Amazon.com"
        assert txn.accounting_category == TransactionCategory.UNKNOWN

    def test_transaction_properties(self):
        """Test transaction computed properties."""
        # Outflow (expense)
        expense = BankTransaction(
            transaction_id="txn_001",
            account_id="acc_001",
            amount=Decimal("50.00"),
            date=date.today(),
            name="Coffee Shop",
            merchant_name="Starbucks",
            pending=False,
        )
        assert expense.is_outflow is True
        assert expense.is_inflow is False
        assert expense.absolute_amount == Decimal("50.00")

        # Inflow (income)
        income = BankTransaction(
            transaction_id="txn_002",
            account_id="acc_001",
            amount=Decimal("-1000.00"),
            date=date.today(),
            name="Payroll",
            merchant_name=None,
            pending=False,
        )
        assert income.is_inflow is True
        assert income.is_outflow is False
        assert income.absolute_amount == Decimal("1000.00")

    def test_transaction_to_dict(self):
        """Test transaction serialization."""
        today = date.today()
        txn = BankTransaction(
            transaction_id="txn_003",
            account_id="acc_002",
            amount=Decimal("-250.00"),
            date=today,
            name="Client Payment",
            merchant_name=None,
            pending=True,
            category=["Transfer", "Deposit"],
            accounting_category=TransactionCategory.INCOME,
            confidence=0.95,
            categorization_source="agent",
        )
        data = txn.to_dict()
        assert data["transaction_id"] == "txn_003"
        assert data["amount"] == -250.0
        assert data["pending"] is True
        assert data["accounting_category"] == "income"
        assert data["confidence"] == 0.95
        assert data["categorization_source"] == "agent"

    def test_transaction_with_anomaly(self):
        """Test transaction with anomaly flags."""
        txn = BankTransaction(
            transaction_id="txn_004",
            account_id="acc_001",
            amount=Decimal("9999.99"),
            date=date.today(),
            name="Unusual Large Purchase",
            merchant_name="Unknown Vendor",
            pending=False,
            is_anomaly=True,
            anomaly_reason="Amount significantly higher than average",
            anomaly_score=0.85,
        )
        assert txn.is_anomaly is True
        assert txn.anomaly_score == 0.85
        assert "higher than average" in txn.anomaly_reason


# =============================================================================
# CategoryMapping Tests
# =============================================================================


class TestCategoryMapping:
    """Tests for CategoryMapping dataclass."""

    def test_mapping_creation(self):
        """Test category mapping initialization."""
        mapping = CategoryMapping(
            plaid_category="FOOD_AND_DRINK_RESTAURANTS",
            qbo_account_id="6100",
            qbo_account_name="Meals & Entertainment",
            accounting_category=TransactionCategory.EXPENSE,
            confidence=0.9,
        )
        assert mapping.plaid_category == "FOOD_AND_DRINK_RESTAURANTS"
        assert mapping.qbo_account_id == "6100"
        assert mapping.accounting_category == TransactionCategory.EXPENSE


# =============================================================================
# PlaidConnector Tests
# =============================================================================


class TestPlaidConnectorInit:
    """Tests for PlaidConnector initialization."""

    def test_connector_with_explicit_config(self):
        """Test connector with explicit configuration."""
        connector = PlaidConnector(
            client_id="test_client",
            secret="test_secret",
            environment=PlaidEnvironment.SANDBOX,
        )
        assert connector.client_id == "test_client"
        assert connector.secret == "test_secret"
        assert connector.environment == PlaidEnvironment.SANDBOX
        assert connector.base_url == PlaidConnector.SANDBOX_URL

    def test_connector_production_environment(self):
        """Test connector with production environment."""
        connector = PlaidConnector(
            client_id="client",
            secret="secret",
            environment=PlaidEnvironment.PRODUCTION,
        )
        assert connector.environment == PlaidEnvironment.PRODUCTION
        assert connector.base_url == PlaidConnector.PRODUCTION_URL

    def test_connector_development_environment(self):
        """Test connector with development environment."""
        connector = PlaidConnector(
            client_id="client",
            secret="secret",
            environment=PlaidEnvironment.DEVELOPMENT,
        )
        assert connector.environment == PlaidEnvironment.DEVELOPMENT
        assert connector.base_url == PlaidConnector.DEVELOPMENT_URL

    def test_is_configured_true(self):
        """Test is_configured when properly configured."""
        connector = PlaidConnector(
            client_id="client",
            secret="secret",
        )
        assert connector.is_configured is True

    def test_is_configured_false(self):
        """Test is_configured when missing credentials."""
        with patch.dict("os.environ", {}, clear=True):
            connector = PlaidConnector()
            assert connector.is_configured is False

    def test_default_category_mappings_loaded(self):
        """Test that default category mappings are loaded."""
        connector = PlaidConnector(client_id="c", secret="s")
        assert len(connector._category_mappings) > 0
        assert "INCOME_WAGES" in connector._category_mappings
        assert "FOOD_AND_DRINK_RESTAURANTS" in connector._category_mappings


class TestPlaidConnectorOperations:
    """Tests for PlaidConnector operations."""

    @pytest.fixture
    def connector(self):
        """Create connector for testing."""
        return PlaidConnector(
            client_id="test_client",
            secret="test_secret",
            environment=PlaidEnvironment.SANDBOX,
        )

    @pytest.mark.asyncio
    async def test_create_link_token_success(self, connector):
        """Test successful link token creation."""
        mock_response = {
            "link_token": "link-token-123",
            "expiration": "2024-01-15T12:00:00Z",
            "request_id": "req-456",
        }

        with patch.object(connector, "_request", new=AsyncMock(return_value=mock_response)):
            result = await connector.create_link_token(
                user_id="user_001",
                tenant_id="tenant_001",
            )

            assert result["link_token"] == "link-token-123"
            assert result["expiration"] == "2024-01-15T12:00:00Z"

    @pytest.mark.asyncio
    async def test_exchange_public_token_success(self, connector):
        """Test successful public token exchange."""
        mock_response = {
            "access_token": "access-token-xyz",
            "item_id": "item-123",
        }

        with patch.object(connector, "_request", new=AsyncMock(return_value=mock_response)):
            creds = await connector.exchange_public_token(
                public_token="public-token",
                user_id="user_001",
                tenant_id="tenant_001",
                institution_id="ins_123",
                institution_name="Test Bank",
            )

            assert creds.access_token == "access-token-xyz"
            assert creds.item_id == "item-123"
            assert creds.institution_name == "Test Bank"

    @pytest.mark.asyncio
    async def test_get_accounts_success(self, connector):
        """Test successful account retrieval."""
        mock_response = {
            "accounts": [
                {
                    "account_id": "acc_001",
                    "name": "Checking",
                    "official_name": "Premium Checking",
                    "type": "depository",
                    "subtype": "checking",
                    "mask": "1234",
                    "balances": {
                        "current": 5000.00,
                        "available": 4800.00,
                        "iso_currency_code": "USD",
                    },
                },
                {
                    "account_id": "acc_002",
                    "name": "Credit Card",
                    "official_name": "Rewards Card",
                    "type": "credit",
                    "subtype": "credit card",
                    "mask": "5678",
                    "balances": {
                        "current": 1500.00,
                        "limit": 10000.00,
                        "iso_currency_code": "USD",
                    },
                },
            ]
        }

        credentials = PlaidCredentials(
            access_token="token",
            item_id="item",
            institution_id="ins",
            institution_name="Bank",
            user_id="user",
            tenant_id="tenant",
        )

        with patch.object(connector, "_request", new=AsyncMock(return_value=mock_response)):
            accounts = await connector.get_accounts(credentials)

            assert len(accounts) == 2
            assert accounts[0].name == "Checking"
            assert accounts[0].account_type == AccountType.CHECKING
            assert accounts[1].name == "Credit Card"
            assert accounts[1].account_type == AccountType.CREDIT

    def test_map_account_type(self, connector):
        """Test account type mapping."""
        assert connector._map_account_type("depository") == AccountType.CHECKING
        assert connector._map_account_type("credit") == AccountType.CREDIT
        assert connector._map_account_type("investment") == AccountType.INVESTMENT
        assert connector._map_account_type("loan") == AccountType.LOAN
        assert connector._map_account_type("unknown") == AccountType.OTHER


# =============================================================================
# PlaidError Tests
# =============================================================================


class TestPlaidError:
    """Tests for PlaidError exception."""

    def test_error_creation(self):
        """Test error creation."""
        error = PlaidError("INVALID_CREDENTIALS", "Invalid credentials provided")
        assert "[INVALID_CREDENTIALS]" in str(error)
        assert "Invalid credentials provided" in str(error)
        assert error.error_code == "INVALID_CREDENTIALS"
        assert error.error_message == "Invalid credentials provided"

    def test_error_attributes(self):
        """Test error attributes."""
        error = PlaidError("ITEM_LOGIN_REQUIRED", "User must re-authenticate")
        assert error.error_code == "ITEM_LOGIN_REQUIRED"
        assert "re-authenticate" in error.error_message


# =============================================================================
# Mock Data Tests
# =============================================================================


class TestMockData:
    """Tests for mock data generation."""

    def test_get_mock_accounts(self):
        """Test mock account generation."""
        accounts = get_mock_accounts()
        assert len(accounts) > 0
        assert all(isinstance(acc, BankAccount) for acc in accounts)

        # Check we have different account types
        account_types = {acc.account_type for acc in accounts}
        assert len(account_types) > 1

    def test_get_mock_transactions(self):
        """Test mock transaction generation."""
        transactions = get_mock_transactions()
        assert len(transactions) > 0
        assert all(isinstance(txn, BankTransaction) for txn in transactions)

        # Check for variety
        has_inflow = any(txn.is_inflow for txn in transactions)
        has_outflow = any(txn.is_outflow for txn in transactions)
        assert has_inflow or has_outflow

    def test_mock_accounts_have_valid_data(self):
        """Test mock accounts have valid data."""
        accounts = get_mock_accounts()
        for account in accounts:
            assert account.account_id is not None
            assert account.name is not None
            assert account.mask is not None
            assert account.account_type in AccountType

    def test_mock_transactions_have_valid_data(self):
        """Test mock transactions have valid data."""
        transactions = get_mock_transactions()
        for txn in transactions:
            assert txn.transaction_id is not None
            assert txn.account_id is not None
            assert txn.name is not None
            assert isinstance(txn.amount, Decimal)
            assert isinstance(txn.date, date)
