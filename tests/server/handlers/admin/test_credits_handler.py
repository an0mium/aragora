"""
Tests for the Credits Admin Handler.

Tests cover:
- CreditsAdminHandler credit operations
- Issue credit endpoint
- Get credit account endpoint
- List transactions endpoint
- Adjust balance endpoint
- Get expiring credits endpoint
- Input validation
- Route registration
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.admin.credits import (
    CreditsAdminHandler,
    register_credits_admin_routes,
)


# =============================================================================
# Mock Classes
# =============================================================================


@dataclass
class MockCreditTransaction:
    """Mock credit transaction for testing."""

    id: str = "txn-123"
    org_id: str = "org-456"
    amount_cents: int = 1000
    transaction_type: str = "promotional"
    description: str = "Test credit"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "admin-user"
    reference_id: Optional[str] = None
    expires_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "org_id": self.org_id,
            "amount_cents": self.amount_cents,
            "transaction_type": self.transaction_type,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "reference_id": self.reference_id,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class MockCreditAccount:
    """Mock credit account for testing."""

    org_id: str = "org-456"
    balance_cents: int = 5000
    total_issued_cents: int = 10000
    total_used_cents: int = 5000
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "org_id": self.org_id,
            "balance_cents": self.balance_cents,
            "total_issued_cents": self.total_issued_cents,
            "total_used_cents": self.total_used_cents,
            "created_at": self.created_at.isoformat(),
        }


class MockCreditManager:
    """Mock credit manager for testing."""

    def __init__(self):
        self.issue_credit = AsyncMock(return_value=MockCreditTransaction())
        self.get_account = AsyncMock(return_value=MockCreditAccount())
        self.get_transactions = AsyncMock(return_value=[MockCreditTransaction()])
        self.adjust_balance = AsyncMock(return_value=MockCreditTransaction(amount_cents=-500))
        self.get_expiring_credits = AsyncMock(
            return_value=[
                MockCreditTransaction(expires_at=datetime.now(timezone.utc) + timedelta(days=7))
            ]
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_credit_manager():
    """Create a mock credit manager."""
    return MockCreditManager()


@pytest.fixture
def handler():
    """Create a CreditsAdminHandler instance with mocked context."""
    mock_context = {
        "storage": None,
        "elo_system": None,
    }
    return CreditsAdminHandler(mock_context)


@pytest.fixture(autouse=True)
def mock_rbac_decorators():
    """Mock RBAC decorators to bypass permission checks."""
    with patch(
        "aragora.server.handlers.admin.credits.require_permission",
        lambda perm: lambda fn: fn,
    ):
        yield


# =============================================================================
# Test Issue Credit
# =============================================================================


class TestIssueCreditSuccess:
    """Tests for successful credit issuance."""

    @pytest.mark.asyncio
    async def test_issue_credit_basic(self, handler, mock_credit_manager):
        """Test basic credit issuance."""
        with patch(
            "aragora.server.handlers.admin.credits.get_credit_manager",
            return_value=mock_credit_manager,
        ):
            result = await handler.issue_credit(
                org_id="org-456",
                data={
                    "amount_cents": 1000,
                    "type": "promotional",
                    "description": "Welcome bonus",
                },
                user_id="admin-user",
            )

            assert result.status_code == 201
            data = json.loads(result.body)
            assert "transaction" in data
            mock_credit_manager.issue_credit.assert_called_once()

    @pytest.mark.asyncio
    async def test_issue_credit_with_expiration(self, handler, mock_credit_manager):
        """Test credit issuance with expiration."""
        with patch(
            "aragora.server.handlers.admin.credits.get_credit_manager",
            return_value=mock_credit_manager,
        ):
            result = await handler.issue_credit(
                org_id="org-456",
                data={
                    "amount_cents": 5000,
                    "type": "promotional",
                    "description": "Limited offer",
                    "expires_days": 30,
                },
                user_id="admin-user",
            )

            assert result.status_code == 201
            # Verify expires_at was passed
            call_kwargs = mock_credit_manager.issue_credit.call_args.kwargs
            assert call_kwargs["expires_at"] is not None


class TestIssueCreditValidation:
    """Tests for credit issuance validation."""

    @pytest.mark.asyncio
    async def test_issue_credit_missing_amount(self, handler):
        """Test credit issuance fails without amount."""
        result = await handler.issue_credit(
            org_id="org-456",
            data={"type": "promotional", "description": "Test"},
            user_id="admin-user",
        )

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "amount_cents" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_issue_credit_invalid_amount(self, handler):
        """Test credit issuance fails with invalid amount."""
        result = await handler.issue_credit(
            org_id="org-456",
            data={"amount_cents": -100, "type": "promotional", "description": "Test"},
            user_id="admin-user",
        )

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_issue_credit_invalid_type(self, handler):
        """Test credit issuance fails with invalid type."""
        result = await handler.issue_credit(
            org_id="org-456",
            data={"amount_cents": 1000, "type": "invalid_type", "description": "Test"},
            user_id="admin-user",
        )

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "Invalid credit type" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_issue_credit_missing_description(self, handler):
        """Test credit issuance fails without description."""
        result = await handler.issue_credit(
            org_id="org-456",
            data={"amount_cents": 1000, "type": "promotional"},
            user_id="admin-user",
        )

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "description" in data.get("error", "")


# =============================================================================
# Test Get Credit Account
# =============================================================================


class TestGetCreditAccount:
    """Tests for getting credit account details."""

    @pytest.mark.asyncio
    async def test_get_account_success(self, handler, mock_credit_manager):
        """Test successful account retrieval."""
        with patch(
            "aragora.server.handlers.admin.credits.get_credit_manager",
            return_value=mock_credit_manager,
        ):
            result = await handler.get_credit_account(org_id="org-456")

            assert result.status_code == 200
            data = json.loads(result.body)
            assert "account" in data
            assert data["account"]["org_id"] == "org-456"
            mock_credit_manager.get_account.assert_called_once_with("org-456")


# =============================================================================
# Test List Transactions
# =============================================================================


class TestListTransactions:
    """Tests for listing credit transactions."""

    @pytest.mark.asyncio
    async def test_list_transactions_success(self, handler, mock_credit_manager):
        """Test successful transaction listing."""
        with patch(
            "aragora.server.handlers.admin.credits.get_credit_manager",
            return_value=mock_credit_manager,
        ):
            result = await handler.list_transactions(org_id="org-456")

            assert result.status_code == 200
            data = json.loads(result.body)
            assert "transactions" in data
            assert "count" in data
            assert data["count"] == 1

    @pytest.mark.asyncio
    async def test_list_transactions_with_pagination(self, handler, mock_credit_manager):
        """Test transaction listing with pagination."""
        with patch(
            "aragora.server.handlers.admin.credits.get_credit_manager",
            return_value=mock_credit_manager,
        ):
            result = await handler.list_transactions(
                org_id="org-456",
                limit=50,
                offset=10,
            )

            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["limit"] == 50
            assert data["offset"] == 10

    @pytest.mark.asyncio
    async def test_list_transactions_with_type_filter(self, handler, mock_credit_manager):
        """Test transaction listing with type filter."""
        with patch(
            "aragora.server.handlers.admin.credits.get_credit_manager",
            return_value=mock_credit_manager,
        ):
            result = await handler.list_transactions(
                org_id="org-456",
                transaction_type="promotional",
            )

            assert result.status_code == 200
            call_kwargs = mock_credit_manager.get_transactions.call_args.kwargs
            assert call_kwargs["transaction_type"] is not None

    @pytest.mark.asyncio
    async def test_list_transactions_invalid_type(self, handler):
        """Test transaction listing fails with invalid type."""
        result = await handler.list_transactions(
            org_id="org-456",
            transaction_type="invalid_type",
        )

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "Invalid transaction type" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_list_transactions_caps_limit(self, handler, mock_credit_manager):
        """Test transaction listing caps limit at 500."""
        with patch(
            "aragora.server.handlers.admin.credits.get_credit_manager",
            return_value=mock_credit_manager,
        ):
            await handler.list_transactions(org_id="org-456", limit=1000)

            call_kwargs = mock_credit_manager.get_transactions.call_args.kwargs
            assert call_kwargs["limit"] == 500


# =============================================================================
# Test Adjust Balance
# =============================================================================


class TestAdjustBalance:
    """Tests for balance adjustments."""

    @pytest.mark.asyncio
    async def test_adjust_balance_positive(self, handler, mock_credit_manager):
        """Test positive balance adjustment."""
        with patch(
            "aragora.server.handlers.admin.credits.get_credit_manager",
            return_value=mock_credit_manager,
        ):
            result = await handler.adjust_balance(
                org_id="org-456",
                data={
                    "amount_cents": 500,
                    "description": "Goodwill adjustment",
                },
                user_id="admin-user",
            )

            assert result.status_code == 200
            data = json.loads(result.body)
            assert "transaction" in data

    @pytest.mark.asyncio
    async def test_adjust_balance_negative(self, handler, mock_credit_manager):
        """Test negative balance adjustment."""
        with patch(
            "aragora.server.handlers.admin.credits.get_credit_manager",
            return_value=mock_credit_manager,
        ):
            result = await handler.adjust_balance(
                org_id="org-456",
                data={
                    "amount_cents": -500,
                    "description": "Correction",
                },
                user_id="admin-user",
            )

            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_adjust_balance_missing_amount(self, handler):
        """Test adjustment fails without amount."""
        result = await handler.adjust_balance(
            org_id="org-456",
            data={"description": "Test"},
            user_id="admin-user",
        )

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_adjust_balance_zero_amount(self, handler):
        """Test adjustment fails with zero amount."""
        result = await handler.adjust_balance(
            org_id="org-456",
            data={"amount_cents": 0, "description": "Test"},
            user_id="admin-user",
        )

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "cannot be zero" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_adjust_balance_missing_description(self, handler):
        """Test adjustment fails without description."""
        result = await handler.adjust_balance(
            org_id="org-456",
            data={"amount_cents": 500},
            user_id="admin-user",
        )

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "description" in data.get("error", "")


# =============================================================================
# Test Get Expiring Credits
# =============================================================================


class TestGetExpiringCredits:
    """Tests for getting expiring credits."""

    @pytest.mark.asyncio
    async def test_get_expiring_success(self, handler, mock_credit_manager):
        """Test successful expiring credits retrieval."""
        with patch(
            "aragora.server.handlers.admin.credits.get_credit_manager",
            return_value=mock_credit_manager,
        ):
            result = await handler.get_expiring_credits(org_id="org-456")

            assert result.status_code == 200
            data = json.loads(result.body)
            assert "expiring_credits" in data
            assert "total_expiring_cents" in data
            assert "total_expiring_usd" in data
            assert data["within_days"] == 30

    @pytest.mark.asyncio
    async def test_get_expiring_custom_days(self, handler, mock_credit_manager):
        """Test expiring credits with custom days."""
        with patch(
            "aragora.server.handlers.admin.credits.get_credit_manager",
            return_value=mock_credit_manager,
        ):
            result = await handler.get_expiring_credits(
                org_id="org-456",
                within_days=90,
            )

            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["within_days"] == 90

    @pytest.mark.asyncio
    async def test_get_expiring_caps_days(self, handler, mock_credit_manager):
        """Test expiring credits caps days at 365."""
        with patch(
            "aragora.server.handlers.admin.credits.get_credit_manager",
            return_value=mock_credit_manager,
        ):
            await handler.get_expiring_credits(org_id="org-456", within_days=1000)

            call_kwargs = mock_credit_manager.get_expiring_credits.call_args.kwargs
            assert call_kwargs["within_days"] == 365

    @pytest.mark.asyncio
    async def test_get_expiring_empty(self, handler, mock_credit_manager):
        """Test expiring credits when none exist."""
        mock_credit_manager.get_expiring_credits = AsyncMock(return_value=[])

        with patch(
            "aragora.server.handlers.admin.credits.get_credit_manager",
            return_value=mock_credit_manager,
        ):
            result = await handler.get_expiring_credits(org_id="org-456")

            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["total_expiring_cents"] == 0


# =============================================================================
# Test Route Registration
# =============================================================================


class TestRouteRegistration:
    """Tests for route registration."""

    def test_register_routes(self):
        """Test that routes are registered correctly."""
        from aiohttp import web

        app = web.Application()
        mock_context = {"storage": None, "elo_system": None}
        handler = CreditsAdminHandler(mock_context)

        register_credits_admin_routes(app, handler)

        # Check versioned routes exist
        route_paths = [r.resource.canonical for r in app.router.routes()]
        assert "/api/v1/admin/credits/{org_id}/issue" in route_paths
        assert "/api/v1/admin/credits/{org_id}" in route_paths
        assert "/api/v1/admin/credits/{org_id}/transactions" in route_paths
        assert "/api/v1/admin/credits/{org_id}/adjust" in route_paths
        assert "/api/v1/admin/credits/{org_id}/expiring" in route_paths

        # Check non-versioned routes exist
        assert "/api/admin/credits/{org_id}/issue" in route_paths
        assert "/api/admin/credits/{org_id}" in route_paths
