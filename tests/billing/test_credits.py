"""Tests for the Credits System."""

import pytest
from datetime import datetime, timedelta, timezone

from aragora.billing.credits import (
    CreditTransactionType,
    CreditTransaction,
    CreditAccount,
    DeductionResult,
    CreditManager,
    reset_credit_manager,
)


@pytest.fixture
def credit_manager(tmp_path):
    """Create a credit manager with temp database."""
    reset_credit_manager()
    db_path = str(tmp_path / "test_credits.db")
    manager = CreditManager(db_path)
    yield manager
    reset_credit_manager()


class TestCreditTransaction:
    """Tests for CreditTransaction dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(days=30)
        tx = CreditTransaction(
            id="cred_test123",
            org_id="org_456",
            amount_cents=5000,
            transaction_type=CreditTransactionType.PROMOTIONAL,
            description="Welcome credit",
            expires_at=expires,
            created_at=now,
            created_by="admin",
            reference_id="promo_123",
        )
        d = tx.to_dict()
        assert d["id"] == "cred_test123"
        assert d["org_id"] == "org_456"
        assert d["amount_cents"] == 5000
        assert d["transaction_type"] == "promotional"
        assert d["description"] == "Welcome credit"
        assert d["expires_at"] == expires.isoformat()
        assert d["created_by"] == "admin"

    def test_to_dict_no_expiration(self):
        """Test serialization without expiration."""
        tx = CreditTransaction(
            id="cred_test",
            org_id="org_1",
            amount_cents=1000,
            transaction_type=CreditTransactionType.REFUND,
            description="Refund",
        )
        d = tx.to_dict()
        assert d["expires_at"] is None


class TestCreditAccount:
    """Tests for CreditAccount dataclass."""

    def test_to_dict_includes_usd(self):
        """Test that to_dict includes USD amounts."""
        account = CreditAccount(
            org_id="org_123",
            balance_cents=5000,
            lifetime_issued_cents=10000,
            lifetime_redeemed_cents=5000,
            lifetime_expired_cents=0,
        )
        d = account.to_dict()
        assert d["balance_cents"] == 5000
        assert d["balance_usd"] == 50.0
        assert d["lifetime_issued_usd"] == 100.0
        assert d["lifetime_redeemed_usd"] == 50.0


class TestCreditManager:
    """Tests for CreditManager."""

    @pytest.mark.asyncio
    async def test_issue_credit(self, credit_manager):
        """Test issuing credits."""
        tx = await credit_manager.issue_credit(
            org_id="org_test",
            amount_cents=5000,
            credit_type=CreditTransactionType.PROMOTIONAL,
            description="Welcome credit",
            created_by="admin",
        )

        assert tx.id.startswith("cred_")
        assert tx.org_id == "org_test"
        assert tx.amount_cents == 5000
        assert tx.transaction_type == CreditTransactionType.PROMOTIONAL

    @pytest.mark.asyncio
    async def test_issue_credit_with_expiration(self, credit_manager):
        """Test issuing credits with expiration."""
        expires = datetime.now(timezone.utc) + timedelta(days=30)
        tx = await credit_manager.issue_credit(
            org_id="org_test",
            amount_cents=2000,
            credit_type=CreditTransactionType.PROMOTIONAL,
            description="30-day promo",
            expires_at=expires,
        )

        assert tx.expires_at is not None
        assert tx.expires_at == expires

    @pytest.mark.asyncio
    async def test_issue_credit_negative_amount_fails(self, credit_manager):
        """Test that negative amounts are rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            await credit_manager.issue_credit(
                org_id="org_test",
                amount_cents=-100,
                credit_type=CreditTransactionType.PROMOTIONAL,
                description="Invalid",
            )

    @pytest.mark.asyncio
    async def test_get_balance(self, credit_manager):
        """Test balance calculation."""
        # Issue some credits
        await credit_manager.issue_credit(
            org_id="org_test",
            amount_cents=5000,
            credit_type=CreditTransactionType.PROMOTIONAL,
            description="Credit 1",
        )
        await credit_manager.issue_credit(
            org_id="org_test",
            amount_cents=3000,
            credit_type=CreditTransactionType.REFERRAL,
            description="Credit 2",
        )

        balance = await credit_manager.get_balance("org_test")
        assert balance == 8000

    @pytest.mark.asyncio
    async def test_get_balance_excludes_expired(self, credit_manager):
        """Test that expired credits are excluded from balance."""
        # Issue expired credit
        expired = datetime.now(timezone.utc) - timedelta(days=1)
        await credit_manager.issue_credit(
            org_id="org_test",
            amount_cents=5000,
            credit_type=CreditTransactionType.PROMOTIONAL,
            description="Expired",
            expires_at=expired,
        )
        # Issue valid credit
        await credit_manager.issue_credit(
            org_id="org_test",
            amount_cents=2000,
            credit_type=CreditTransactionType.PROMOTIONAL,
            description="Valid",
        )

        balance = await credit_manager.get_balance("org_test")
        assert balance == 2000  # Only non-expired credit

    @pytest.mark.asyncio
    async def test_deduct_credit_full(self, credit_manager):
        """Test deducting full amount from credits."""
        await credit_manager.issue_credit(
            org_id="org_test",
            amount_cents=5000,
            credit_type=CreditTransactionType.PROMOTIONAL,
            description="Initial",
        )

        result = await credit_manager.deduct_credit(
            org_id="org_test",
            amount_cents=3000,
            description="API usage",
        )

        assert result.success is True
        assert result.amount_deducted_cents == 3000
        assert result.remaining_amount_cents == 0

        # Check balance decreased
        balance = await credit_manager.get_balance("org_test")
        assert balance == 2000

    @pytest.mark.asyncio
    async def test_deduct_credit_partial(self, credit_manager):
        """Test partial deduction when insufficient credits."""
        await credit_manager.issue_credit(
            org_id="org_test",
            amount_cents=2000,
            credit_type=CreditTransactionType.PROMOTIONAL,
            description="Initial",
        )

        result = await credit_manager.deduct_credit(
            org_id="org_test",
            amount_cents=5000,
            description="Large usage",
        )

        assert result.success is True
        assert result.amount_deducted_cents == 2000  # Only available amount
        assert result.remaining_amount_cents == 3000  # Still owed

    @pytest.mark.asyncio
    async def test_deduct_credit_no_balance(self, credit_manager):
        """Test deduction with no credits available."""
        result = await credit_manager.deduct_credit(
            org_id="org_test",
            amount_cents=1000,
            description="Usage",
        )

        assert result.success is True
        assert result.amount_deducted_cents == 0
        assert result.remaining_amount_cents == 1000

    @pytest.mark.asyncio
    async def test_get_account(self, credit_manager):
        """Test getting full account details."""
        await credit_manager.issue_credit(
            org_id="org_test",
            amount_cents=10000,
            credit_type=CreditTransactionType.PROMOTIONAL,
            description="Initial",
        )
        await credit_manager.deduct_credit(
            org_id="org_test",
            amount_cents=3000,
            description="Usage",
        )

        account = await credit_manager.get_account("org_test")

        assert account.org_id == "org_test"
        assert account.balance_cents == 7000
        assert account.lifetime_issued_cents == 10000
        assert account.lifetime_redeemed_cents == 3000

    @pytest.mark.asyncio
    async def test_get_transactions(self, credit_manager):
        """Test getting transaction history."""
        await credit_manager.issue_credit(
            org_id="org_test",
            amount_cents=5000,
            credit_type=CreditTransactionType.PROMOTIONAL,
            description="Credit 1",
        )
        await credit_manager.issue_credit(
            org_id="org_test",
            amount_cents=3000,
            credit_type=CreditTransactionType.REFERRAL,
            description="Credit 2",
        )

        transactions = await credit_manager.get_transactions("org_test")

        assert len(transactions) == 2
        # Newest first
        assert transactions[0].description == "Credit 2"
        assert transactions[1].description == "Credit 1"

    @pytest.mark.asyncio
    async def test_get_transactions_with_type_filter(self, credit_manager):
        """Test filtering transactions by type."""
        await credit_manager.issue_credit(
            org_id="org_test",
            amount_cents=5000,
            credit_type=CreditTransactionType.PROMOTIONAL,
            description="Promo",
        )
        await credit_manager.issue_credit(
            org_id="org_test",
            amount_cents=3000,
            credit_type=CreditTransactionType.REFERRAL,
            description="Referral",
        )

        promo_txs = await credit_manager.get_transactions(
            "org_test", transaction_type=CreditTransactionType.PROMOTIONAL
        )

        assert len(promo_txs) == 1
        assert promo_txs[0].description == "Promo"

    @pytest.mark.asyncio
    async def test_get_expiring_credits(self, credit_manager):
        """Test getting credits expiring soon."""
        now = datetime.now(timezone.utc)

        # Expiring in 7 days
        await credit_manager.issue_credit(
            org_id="org_test",
            amount_cents=1000,
            credit_type=CreditTransactionType.PROMOTIONAL,
            description="Expiring soon",
            expires_at=now + timedelta(days=7),
        )
        # Expiring in 60 days (outside window)
        await credit_manager.issue_credit(
            org_id="org_test",
            amount_cents=2000,
            credit_type=CreditTransactionType.PROMOTIONAL,
            description="Later",
            expires_at=now + timedelta(days=60),
        )
        # No expiration
        await credit_manager.issue_credit(
            org_id="org_test",
            amount_cents=3000,
            credit_type=CreditTransactionType.REFERRAL,
            description="No expiry",
        )

        expiring = await credit_manager.get_expiring_credits("org_test", within_days=30)

        assert len(expiring) == 1
        assert expiring[0].description == "Expiring soon"

    @pytest.mark.asyncio
    async def test_adjust_balance_positive(self, credit_manager):
        """Test positive balance adjustment."""
        tx = await credit_manager.adjust_balance(
            org_id="org_test",
            amount_cents=5000,
            description="Support credit",
            created_by="support_agent",
        )

        assert tx.amount_cents == 5000
        assert tx.transaction_type == CreditTransactionType.ADJUSTMENT
        assert tx.created_by == "support_agent"

        balance = await credit_manager.get_balance("org_test")
        assert balance == 5000

    @pytest.mark.asyncio
    async def test_adjust_balance_negative(self, credit_manager):
        """Test negative balance adjustment (correction)."""
        await credit_manager.issue_credit(
            org_id="org_test",
            amount_cents=5000,
            credit_type=CreditTransactionType.PROMOTIONAL,
            description="Initial",
        )

        await credit_manager.adjust_balance(
            org_id="org_test",
            amount_cents=-2000,
            description="Correction for duplicate issue",
            created_by="admin",
        )

        balance = await credit_manager.get_balance("org_test")
        assert balance == 3000

    @pytest.mark.asyncio
    async def test_org_isolation(self, credit_manager):
        """Test that orgs are isolated from each other."""
        await credit_manager.issue_credit(
            org_id="org_1",
            amount_cents=5000,
            credit_type=CreditTransactionType.PROMOTIONAL,
            description="Org 1",
        )
        await credit_manager.issue_credit(
            org_id="org_2",
            amount_cents=3000,
            credit_type=CreditTransactionType.PROMOTIONAL,
            description="Org 2",
        )

        balance_1 = await credit_manager.get_balance("org_1")
        balance_2 = await credit_manager.get_balance("org_2")

        assert balance_1 == 5000
        assert balance_2 == 3000


class TestDeductionResult:
    """Tests for DeductionResult dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = DeductionResult(
            success=True,
            amount_deducted_cents=2000,
            remaining_amount_cents=1000,
            message="Partial deduction",
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["amount_deducted_cents"] == 2000
        assert d["remaining_amount_cents"] == 1000
        assert d["message"] == "Partial deduction"
