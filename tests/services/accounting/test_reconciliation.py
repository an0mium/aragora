"""
Tests for Bank Reconciliation Service.

Tests cover:
- Dataclass serialization
- Enum values
- ReconciliationService initialization
- Matching algorithms
- Discrepancy detection
- Mock data generation
"""

import pytest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.services.accounting.reconciliation import (
    ReconciliationService,
    ReconciliationResult,
    MatchedTransaction,
    Discrepancy,
    DiscrepancyType,
    DiscrepancySeverity,
    ResolutionStatus,
    get_mock_reconciliation_result,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestDiscrepancyType:
    """Tests for DiscrepancyType enum."""

    def test_discrepancy_type_values(self):
        """Test all discrepancy type values."""
        assert DiscrepancyType.UNMATCHED_BANK.value == "unmatched_bank"
        assert DiscrepancyType.UNMATCHED_BOOK.value == "unmatched_book"
        assert DiscrepancyType.AMOUNT_MISMATCH.value == "amount_mismatch"
        assert DiscrepancyType.DATE_MISMATCH.value == "date_mismatch"
        assert DiscrepancyType.POTENTIAL_DUPLICATE.value == "potential_duplicate"
        assert DiscrepancyType.CATEGORIZATION.value == "categorization"


class TestDiscrepancySeverity:
    """Tests for DiscrepancySeverity enum."""

    def test_severity_values(self):
        """Test all severity values."""
        assert DiscrepancySeverity.LOW.value == "low"
        assert DiscrepancySeverity.MEDIUM.value == "medium"
        assert DiscrepancySeverity.HIGH.value == "high"
        assert DiscrepancySeverity.CRITICAL.value == "critical"


class TestResolutionStatus:
    """Tests for ResolutionStatus enum."""

    def test_resolution_status_values(self):
        """Test all resolution status values."""
        assert ResolutionStatus.PENDING.value == "pending"
        assert ResolutionStatus.AGENT_SUGGESTED.value == "agent_suggested"
        assert ResolutionStatus.USER_RESOLVED.value == "user_resolved"
        assert ResolutionStatus.AUTO_RESOLVED.value == "auto_resolved"
        assert ResolutionStatus.IGNORED.value == "ignored"


# =============================================================================
# MatchedTransaction Tests
# =============================================================================


class TestMatchedTransaction:
    """Tests for MatchedTransaction dataclass."""

    def test_matched_transaction_creation(self):
        """Test matched transaction initialization."""
        today = date.today()
        match = MatchedTransaction(
            bank_txn_id="bank_001",
            book_txn_id="book_001",
            bank_amount=Decimal("100.00"),
            book_amount=Decimal("100.00"),
            bank_date=today,
            book_date=today,
            bank_description="Payment ABC",
            book_description="Invoice ABC",
        )
        assert match.bank_txn_id == "bank_001"
        assert match.book_txn_id == "book_001"
        assert match.match_confidence == 1.0
        assert match.match_method == "exact"

    def test_matched_transaction_to_dict(self):
        """Test matched transaction serialization."""
        today = date.today()
        match = MatchedTransaction(
            bank_txn_id="bank_001",
            book_txn_id="book_001",
            bank_amount=Decimal("250.50"),
            book_amount=Decimal("250.50"),
            bank_date=today,
            book_date=today,
            bank_description="AWS Services",
            book_description="AWS Invoice",
            match_confidence=0.95,
            match_method="fuzzy",
        )
        data = match.to_dict()
        assert data["bank_txn_id"] == "bank_001"
        assert data["book_txn_id"] == "book_001"
        assert data["bank_amount"] == 250.50
        assert data["book_amount"] == 250.50
        assert data["match_confidence"] == 0.95
        assert data["match_method"] == "fuzzy"
        assert data["bank_date"] == today.isoformat()


# =============================================================================
# Discrepancy Tests
# =============================================================================


class TestDiscrepancy:
    """Tests for Discrepancy dataclass."""

    def test_discrepancy_creation(self):
        """Test discrepancy initialization."""
        disc = Discrepancy(
            discrepancy_id="disc_001",
            discrepancy_type=DiscrepancyType.UNMATCHED_BANK,
            severity=DiscrepancySeverity.MEDIUM,
            description="Bank transaction not found in books",
        )
        assert disc.discrepancy_id == "disc_001"
        assert disc.discrepancy_type == DiscrepancyType.UNMATCHED_BANK
        assert disc.severity == DiscrepancySeverity.MEDIUM
        assert disc.resolution_status == ResolutionStatus.PENDING

    def test_discrepancy_with_all_fields(self):
        """Test discrepancy with all optional fields."""
        today = date.today()
        disc = Discrepancy(
            discrepancy_id="disc_002",
            discrepancy_type=DiscrepancyType.AMOUNT_MISMATCH,
            severity=DiscrepancySeverity.HIGH,
            description="Amount mismatch between bank and books",
            bank_txn_id="bank_001",
            book_txn_id="book_001",
            bank_amount=Decimal("100.00"),
            book_amount=Decimal("95.00"),
            bank_date=today,
            book_date=today,
            bank_description="Payment",
            book_description="Invoice",
            resolution_status=ResolutionStatus.AGENT_SUGGESTED,
            resolution_suggestion="Investigate fee deduction",
            resolution_confidence=0.75,
        )
        assert disc.bank_txn_id == "bank_001"
        assert disc.bank_amount == Decimal("100.00")
        assert disc.resolution_confidence == 0.75

    def test_discrepancy_to_dict(self):
        """Test discrepancy serialization."""
        today = date.today()
        disc = Discrepancy(
            discrepancy_id="disc_003",
            discrepancy_type=DiscrepancyType.UNMATCHED_BOOK,
            severity=DiscrepancySeverity.LOW,
            description="Test discrepancy",
            book_amount=Decimal("50.00"),
            book_date=today,
        )
        data = disc.to_dict()
        assert data["discrepancy_id"] == "disc_003"
        assert data["discrepancy_type"] == "unmatched_book"
        assert data["severity"] == "low"
        assert data["book_amount"] == 50.0
        assert data["bank_amount"] is None
        assert data["resolution_status"] == "pending"


# =============================================================================
# ReconciliationResult Tests
# =============================================================================


class TestReconciliationResult:
    """Tests for ReconciliationResult dataclass."""

    def test_result_creation(self):
        """Test reconciliation result initialization."""
        today = date.today()
        result = ReconciliationResult(
            reconciliation_id="recon_001",
            start_date=today - timedelta(days=30),
            end_date=today,
            account_id="acc_001",
            account_name="Checking Account",
        )
        assert result.reconciliation_id == "recon_001"
        assert result.account_name == "Checking Account"
        assert result.bank_transaction_count == 0
        assert result.is_reconciled is False

    def test_result_properties(self):
        """Test computed properties."""
        today = date.today()
        result = ReconciliationResult(
            reconciliation_id="recon_002",
            start_date=today,
            end_date=today,
            account_id="acc",
            account_name="Test",
            bank_transaction_count=10,
            book_transaction_count=8,
            matched_count=6,
            discrepancies=[
                Discrepancy(
                    discrepancy_id="d1",
                    discrepancy_type=DiscrepancyType.UNMATCHED_BANK,
                    severity=DiscrepancySeverity.LOW,
                    description="Test",
                ),
                Discrepancy(
                    discrepancy_id="d2",
                    discrepancy_type=DiscrepancyType.UNMATCHED_BOOK,
                    severity=DiscrepancySeverity.LOW,
                    description="Test",
                ),
            ],
        )
        assert result.discrepancy_count == 2
        # match_rate = (6 * 2) / (10 + 8) = 12/18 = 0.666...
        assert 0.66 < result.match_rate < 0.67

    def test_result_match_rate_empty(self):
        """Test match rate with no transactions."""
        result = ReconciliationResult(
            reconciliation_id="recon_003",
            start_date=date.today(),
            end_date=date.today(),
            account_id="acc",
            account_name="Test",
        )
        assert result.match_rate == 1.0

    def test_result_to_dict(self):
        """Test reconciliation result serialization."""
        today = date.today()
        result = ReconciliationResult(
            reconciliation_id="recon_004",
            start_date=today - timedelta(days=7),
            end_date=today,
            account_id="acc_123",
            account_name="Business Checking",
            bank_transaction_count=25,
            book_transaction_count=23,
            matched_count=20,
            bank_total=Decimal("-5000.00"),
            book_total=Decimal("-4850.00"),
            difference=Decimal("-150.00"),
        )
        data = result.to_dict()
        assert data["reconciliation_id"] == "recon_004"
        assert data["account_name"] == "Business Checking"
        assert data["bank_total"] == -5000.0
        assert data["difference"] == -150.0
        assert data["match_rate"] > 0.8


# =============================================================================
# ReconciliationService Tests
# =============================================================================


class TestReconciliationServiceInit:
    """Tests for ReconciliationService initialization."""

    def test_service_creation_without_connectors(self):
        """Test service creation without connectors."""
        service = ReconciliationService()
        assert service.plaid is None
        assert service.qbo is None
        assert service.match_tolerance_days == 3
        assert service.amount_tolerance == Decimal("0.01")

    def test_service_with_custom_tolerances(self):
        """Test service with custom tolerances."""
        service = ReconciliationService(
            match_tolerance_days=5,
            amount_tolerance=Decimal("0.10"),
        )
        assert service.match_tolerance_days == 5
        assert service.amount_tolerance == Decimal("0.10")

    def test_service_with_mock_connectors(self):
        """Test service with mock connectors."""
        mock_plaid = MagicMock()
        mock_qbo = MagicMock()
        service = ReconciliationService(
            plaid_connector=mock_plaid,
            qbo_connector=mock_qbo,
        )
        assert service.plaid is mock_plaid
        assert service.qbo is mock_qbo


class TestReconciliationServiceOperations:
    """Tests for ReconciliationService operations."""

    @pytest.fixture
    def service_with_mocks(self):
        """Create service with mock connectors."""
        mock_plaid = MagicMock()
        mock_qbo = MagicMock()
        return ReconciliationService(
            plaid_connector=mock_plaid,
            qbo_connector=mock_qbo,
        )

    @pytest.mark.asyncio
    async def test_reconcile_no_transactions(self, service_with_mocks):
        """Test reconciliation with no transactions."""
        service = service_with_mocks
        service.plaid.get_transactions = AsyncMock(return_value=([], 0))
        service.plaid.get_accounts = AsyncMock(return_value=[])
        service.qbo.list_invoices = AsyncMock(return_value=[])
        service.qbo.list_expenses = AsyncMock(return_value=[])

        mock_creds = MagicMock()
        result = await service.reconcile(
            plaid_credentials=mock_creds,
            start_date=date.today() - timedelta(days=30),
            end_date=date.today(),
            use_agents=False,
        )

        assert result is not None
        assert result.bank_transaction_count == 0
        assert result.book_transaction_count == 0
        assert result.is_reconciled is True

    def test_resolve_discrepancy_not_found(self, service_with_mocks):
        """Test resolving non-existent discrepancy."""
        service = service_with_mocks
        result = service.get_reconciliation("nonexistent")
        assert result is None

    def test_list_reconciliations_empty(self, service_with_mocks):
        """Test listing reconciliations when empty."""
        service = service_with_mocks
        results = service.list_reconciliations()
        assert results == []

    def test_list_reconciliations_with_limit(self, service_with_mocks):
        """Test listing reconciliations with limit."""
        service = service_with_mocks
        results = service.list_reconciliations(limit=5)
        assert len(results) == 0


class TestMatchingAlgorithm:
    """Tests for transaction matching algorithm."""

    @pytest.fixture
    def service(self):
        """Create service for testing."""
        return ReconciliationService()

    @pytest.mark.asyncio
    async def test_exact_match(self, service):
        """Test exact amount and date matching."""
        today = date.today()

        # Create mock bank transaction
        bank_txn = MagicMock()
        bank_txn.transaction_id = "bank_001"
        bank_txn.amount = Decimal("100.00")
        bank_txn.date = today
        bank_txn.name = "Payment"

        # Create mock book transaction
        book_txn = MagicMock()
        book_txn.id = "book_001"
        book_txn.total_amount = 100.00
        book_txn.txn_date = datetime.combine(today, datetime.min.time())
        book_txn.memo = "Invoice"
        book_txn.customer_name = None

        matched, unmatched_bank, unmatched_book = await service._match_transactions(
            [bank_txn], [book_txn]
        )

        assert len(matched) == 1
        assert matched[0].match_method == "exact"
        assert matched[0].match_confidence == 1.0
        assert len(unmatched_bank) == 0
        assert len(unmatched_book) == 0

    @pytest.mark.asyncio
    async def test_fuzzy_match_date_tolerance(self, service):
        """Test fuzzy matching within date tolerance."""
        today = date.today()

        bank_txn = MagicMock()
        bank_txn.transaction_id = "bank_001"
        bank_txn.amount = Decimal("200.00")
        bank_txn.date = today
        bank_txn.name = "Payment"

        book_txn = MagicMock()
        book_txn.id = "book_001"
        book_txn.total_amount = 200.00
        book_txn.txn_date = datetime.combine(today - timedelta(days=2), datetime.min.time())
        book_txn.memo = "Invoice"
        book_txn.customer_name = None

        matched, _, _ = await service._match_transactions([bank_txn], [book_txn])

        assert len(matched) == 1
        assert matched[0].match_method == "fuzzy"
        assert matched[0].match_confidence < 1.0

    @pytest.mark.asyncio
    async def test_no_match_amount_difference(self, service):
        """Test no match when amounts differ too much."""
        today = date.today()

        bank_txn = MagicMock()
        bank_txn.transaction_id = "bank_001"
        bank_txn.amount = Decimal("100.00")
        bank_txn.date = today
        bank_txn.name = "Payment"

        book_txn = MagicMock()
        book_txn.id = "book_001"
        book_txn.total_amount = 150.00
        book_txn.txn_date = datetime.combine(today, datetime.min.time())
        book_txn.memo = "Invoice"

        matched, unmatched_bank, unmatched_book = await service._match_transactions(
            [bank_txn], [book_txn]
        )

        assert len(matched) == 0
        assert len(unmatched_bank) == 1
        assert len(unmatched_book) == 1


class TestDiscrepancyGeneration:
    """Tests for discrepancy generation."""

    @pytest.fixture
    def service(self):
        """Create service for testing."""
        return ReconciliationService()

    def test_unmatched_bank_discrepancy(self, service):
        """Test discrepancy generation for unmatched bank transaction."""
        today = date.today()

        bank_txn = MagicMock()
        bank_txn.transaction_id = "bank_001"
        bank_txn.amount = Decimal("-500.00")
        bank_txn.date = today
        bank_txn.name = "Office Supplies"

        discrepancies = service._generate_discrepancies([bank_txn], [], [])

        assert len(discrepancies) == 1
        assert discrepancies[0].discrepancy_type == DiscrepancyType.UNMATCHED_BANK
        assert discrepancies[0].bank_txn_id == "bank_001"

    def test_unmatched_book_discrepancy(self, service):
        """Test discrepancy generation for unmatched book transaction."""
        today = date.today()

        book_txn = MagicMock()
        book_txn.id = "book_001"
        book_txn.total_amount = 750.00
        book_txn.txn_date = datetime.combine(today, datetime.min.time())
        book_txn.memo = "Consulting Invoice"
        book_txn.doc_number = "INV-1001"

        discrepancies = service._generate_discrepancies([], [book_txn], [])

        assert len(discrepancies) == 1
        assert discrepancies[0].discrepancy_type == DiscrepancyType.UNMATCHED_BOOK
        assert discrepancies[0].book_txn_id == "book_001"

    def test_severity_based_on_amount(self, service):
        """Test discrepancy severity based on amount."""
        today = date.today()

        small_txn = MagicMock()
        small_txn.transaction_id = "bank_001"
        small_txn.amount = Decimal("-50.00")
        small_txn.date = today
        small_txn.name = "Small purchase"

        large_txn = MagicMock()
        large_txn.transaction_id = "bank_002"
        large_txn.amount = Decimal("-2000.00")
        large_txn.date = today
        large_txn.name = "Large purchase"

        discrepancies = service._generate_discrepancies([small_txn, large_txn], [], [])

        small_disc = next(d for d in discrepancies if d.bank_txn_id == "bank_001")
        large_disc = next(d for d in discrepancies if d.bank_txn_id == "bank_002")

        assert small_disc.severity == DiscrepancySeverity.MEDIUM
        assert large_disc.severity == DiscrepancySeverity.HIGH

    def test_potential_duplicate_detection(self, service):
        """Test detection of potential duplicate transactions."""
        today = date.today()

        # Create multiple matches with same amount
        matches = [
            MatchedTransaction(
                bank_txn_id=f"bank_{i}",
                book_txn_id=f"book_{i}",
                bank_amount=Decimal("99.99"),
                book_amount=Decimal("99.99"),
                bank_date=today,
                book_date=today,
                bank_description="Monthly Fee",
                book_description="Fee",
            )
            for i in range(4)
        ]

        discrepancies = service._generate_discrepancies([], [], matches)

        duplicate_disc = [
            d for d in discrepancies if d.discrepancy_type == DiscrepancyType.POTENTIAL_DUPLICATE
        ]
        assert len(duplicate_disc) == 1


# =============================================================================
# Mock Data Tests
# =============================================================================


class TestMockData:
    """Tests for mock data generation."""

    def test_get_mock_reconciliation_result(self):
        """Test mock reconciliation result generation."""
        result = get_mock_reconciliation_result()

        assert result is not None
        assert isinstance(result, ReconciliationResult)
        assert result.reconciliation_id == "recon_demo_001"
        assert result.account_name == "Business Checking"
        assert result.bank_transaction_count == 45
        assert result.matched_count == 40
        assert len(result.discrepancies) == 2
        assert len(result.matched_transactions) == 1

    def test_mock_result_discrepancies(self):
        """Test mock result has valid discrepancies."""
        result = get_mock_reconciliation_result()

        unmatched_bank = [
            d for d in result.discrepancies if d.discrepancy_type == DiscrepancyType.UNMATCHED_BANK
        ]
        unmatched_book = [
            d for d in result.discrepancies if d.discrepancy_type == DiscrepancyType.UNMATCHED_BOOK
        ]

        assert len(unmatched_bank) == 1
        assert len(unmatched_book) == 1

        # Check that one has agent suggestion
        agent_suggested = [
            d
            for d in result.discrepancies
            if d.resolution_status == ResolutionStatus.AGENT_SUGGESTED
        ]
        assert len(agent_suggested) == 1
        assert agent_suggested[0].resolution_confidence == 0.85

    def test_mock_result_matched_transaction(self):
        """Test mock result has valid matched transaction."""
        result = get_mock_reconciliation_result()

        assert len(result.matched_transactions) == 1
        match = result.matched_transactions[0]
        assert match.bank_description == "AWS Cloud Services"
        assert match.book_description == "AWS Monthly Invoice"
        assert match.match_confidence == 1.0
        assert match.match_method == "exact"
