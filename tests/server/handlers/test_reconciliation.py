"""
Tests for Bank Reconciliation Handler.
"""

import pytest
from datetime import date, datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.features.reconciliation import (
    ReconciliationHandler,
    get_reconciliation_handler,
    handle_reconciliation,
    get_reconciliation_service,
)


class TestReconciliationHandler:
    """Tests for ReconciliationHandler."""

    def test_handler_routes(self):
        """Test handler has expected routes."""
        handler = ReconciliationHandler()

        expected_routes = [
            "/api/v1/reconciliation/run",
            "/api/v1/reconciliation/list",
            "/api/v1/reconciliation/{reconciliation_id}",
            "/api/v1/reconciliation/{reconciliation_id}/report",
            "/api/v1/reconciliation/{reconciliation_id}/resolve",
            "/api/v1/reconciliation/{reconciliation_id}/approve",
            "/api/v1/reconciliation/discrepancies",
            "/api/v1/reconciliation/discrepancies/bulk-resolve",
        ]

        for route in expected_routes:
            assert any(route in r for r in handler.ROUTES), f"Missing route: {route}"

    def test_get_handler_instance(self):
        """Test getting handler instance."""
        handler1 = get_reconciliation_handler()
        handler2 = get_reconciliation_handler()

        assert handler1 is handler2


class TestRunReconciliation:
    """Tests for running reconciliation."""

    @pytest.mark.asyncio
    async def test_run_requires_dates(self):
        """Test run requires start_date and end_date."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={"account_id": "acc_123"})

        result = await handler.handle(request, "/api/v1/reconciliation/run", "POST")

        assert result is not None
        assert result.status_code == 400
        assert b"required" in result.body

    @pytest.mark.asyncio
    async def test_run_validates_date_format(self):
        """Test run validates date format."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(
            return_value={
                "start_date": "invalid-date",
                "end_date": "2024-01-31",
            }
        )

        result = await handler.handle(request, "/api/v1/reconciliation/run", "POST")

        assert result is not None
        assert result.status_code == 400
        assert b"Invalid date format" in result.body

    @pytest.mark.asyncio
    async def test_run_validates_date_order(self):
        """Test run validates end_date is after start_date."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(
            return_value={
                "start_date": "2024-01-31",
                "end_date": "2024-01-01",
            }
        )

        result = await handler.handle(request, "/api/v1/reconciliation/run", "POST")

        assert result is not None
        assert result.status_code == 400
        assert b"end_date must be after start_date" in result.body

    @pytest.mark.asyncio
    async def test_run_returns_demo_data_without_plaid(self):
        """Test run returns demo data when no plaid token provided."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(
            return_value={
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
            }
        )

        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service"
        ) as mock_get_service:
            mock_service = MagicMock()
            mock_get_service.return_value = mock_service

            # Mock at the source location (dynamic import)
            with patch(
                "aragora.services.accounting.reconciliation.get_mock_reconciliation_result"
            ) as mock_demo:
                mock_result = MagicMock()
                mock_result.start_date = date(2024, 1, 1)
                mock_result.end_date = date(2024, 1, 31)
                mock_result.to_dict.return_value = {"id": "demo_123"}
                mock_result.discrepancies = []
                mock_result.matched_transactions = []
                mock_demo.return_value = mock_result

                result = await handler.handle(request, "/api/v1/reconciliation/run", "POST")

                assert result is not None
                assert result.status_code == 200
                assert b"is_demo" in result.body


class TestListReconciliations:
    """Tests for listing reconciliations."""

    @pytest.mark.asyncio
    async def test_list_returns_empty_when_no_service(self):
        """Test list returns empty when service unavailable."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {}

        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service"
        ) as mock_get:
            mock_get.return_value = None

            result = await handler.handle(request, "/api/v1/reconciliation/list", "GET")

            assert result is not None
            assert result.status_code == 200
            assert b"reconciliations" in result.body

    @pytest.mark.asyncio
    async def test_list_with_account_filter(self):
        """Test list with account_id filter."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {"account_id": "acc_123", "limit": "10"}

        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service"
        ) as mock_get:
            mock_service = MagicMock()
            mock_service.list_reconciliations.return_value = []
            mock_get.return_value = mock_service

            result = await handler.handle(request, "/api/v1/reconciliation/list", "GET")

            assert result is not None
            mock_service.list_reconciliations.assert_called_once_with(
                account_id="acc_123", limit=10
            )


class TestGetReconciliation:
    """Tests for getting a specific reconciliation."""

    @pytest.mark.asyncio
    async def test_get_not_found(self):
        """Test get returns 404 when not found."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"

        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service"
        ) as mock_get:
            mock_service = MagicMock()
            mock_service.get_reconciliation.return_value = None
            mock_get.return_value = mock_service

            result = await handler.handle(request, "/api/v1/reconciliation/rec_123", "GET")

            assert result is not None
            assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_success(self):
        """Test get returns reconciliation details."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"

        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service"
        ) as mock_get:
            mock_result = MagicMock()
            mock_result.to_dict.return_value = {"id": "rec_123"}
            mock_result.discrepancies = []
            mock_result.matched_transactions = []

            mock_service = MagicMock()
            mock_service.get_reconciliation.return_value = mock_result
            mock_get.return_value = mock_service

            result = await handler.handle(request, "/api/v1/reconciliation/rec_123", "GET")

            assert result is not None
            assert result.status_code == 200


class TestResolveDiscrepancy:
    """Tests for resolving discrepancies."""

    @pytest.mark.asyncio
    async def test_resolve_requires_discrepancy_id(self):
        """Test resolve requires discrepancy_id."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={"resolution": "Fixed"})

        result = await handler.handle(request, "/api/v1/reconciliation/rec_123/resolve", "POST")

        assert result is not None
        assert result.status_code == 400
        assert b"discrepancy_id is required" in result.body

    @pytest.mark.asyncio
    async def test_resolve_success(self):
        """Test successful discrepancy resolution."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.user_id = "user_123"
        request.json = AsyncMock(
            return_value={
                "discrepancy_id": "disc_001",
                "resolution": "Created expense entry",
                "action": "create_entry",
            }
        )

        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service"
        ) as mock_get:
            mock_result = MagicMock()
            mock_result.is_reconciled = False
            mock_result.discrepancies = []

            mock_service = MagicMock()
            mock_service.resolve_discrepancy = AsyncMock(return_value=True)
            mock_service.get_reconciliation.return_value = mock_result
            mock_get.return_value = mock_service

            result = await handler.handle(request, "/api/v1/reconciliation/rec_123/resolve", "POST")

            assert result is not None
            assert result.status_code == 200
            assert b"resolved" in result.body


class TestBulkResolve:
    """Tests for bulk resolving discrepancies."""

    @pytest.mark.asyncio
    async def test_bulk_resolve_requires_reconciliation_id(self):
        """Test bulk resolve requires reconciliation_id."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={"resolutions": []})

        result = await handler.handle(
            request, "/api/v1/reconciliation/discrepancies/bulk-resolve", "POST"
        )

        assert result is not None
        assert result.status_code == 400
        assert b"reconciliation_id is required" in result.body

    @pytest.mark.asyncio
    async def test_bulk_resolve_success(self):
        """Test successful bulk resolution."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.user_id = "user_123"
        request.json = AsyncMock(
            return_value={
                "reconciliation_id": "rec_123",
                "resolutions": [
                    {"discrepancy_id": "disc_001", "resolution": "Fixed", "action": "ignore"},
                    {
                        "discrepancy_id": "disc_002",
                        "resolution": "Matched",
                        "action": "match_manual",
                    },
                ],
            }
        )

        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service"
        ) as mock_get:
            mock_result = MagicMock()
            mock_result.is_reconciled = False

            mock_service = MagicMock()
            mock_service.resolve_discrepancy = AsyncMock(return_value=True)
            mock_service.get_reconciliation.return_value = mock_result
            mock_get.return_value = mock_service

            result = await handler.handle(
                request, "/api/v1/reconciliation/discrepancies/bulk-resolve", "POST"
            )

            assert result is not None
            assert result.status_code == 200
            assert b"resolved_count" in result.body


class TestApproveReconciliation:
    """Tests for approving reconciliation."""

    @pytest.mark.asyncio
    async def test_approve_not_found(self):
        """Test approve returns 404 when not found."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={"notes": "Approved"})

        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service"
        ) as mock_get:
            mock_service = MagicMock()
            mock_service.get_reconciliation.return_value = None
            mock_get.return_value = mock_service

            result = await handler.handle(request, "/api/v1/reconciliation/rec_123/approve", "POST")

            assert result is not None
            assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_approve_fails_with_pending_discrepancies(self):
        """Test approve fails when there are pending discrepancies."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={"notes": "Approved"})

        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service"
        ) as mock_get:
            # Create mock discrepancy with pending status
            mock_disc = MagicMock()
            mock_disc.resolution_status.value = "pending"

            mock_result = MagicMock()
            mock_result.discrepancies = [mock_disc]

            mock_service = MagicMock()
            mock_service.get_reconciliation.return_value = mock_result
            mock_get.return_value = mock_service

            result = await handler.handle(request, "/api/v1/reconciliation/rec_123/approve", "POST")

            assert result is not None
            assert result.status_code == 400
            assert b"unresolved discrepancies" in result.body

    @pytest.mark.asyncio
    async def test_approve_success(self):
        """Test successful reconciliation approval."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.user_id = "user_123"
        request.json = AsyncMock(return_value={"notes": "Reviewed and approved"})

        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service"
        ) as mock_get:
            mock_result = MagicMock()
            mock_result.discrepancies = []
            mock_result.is_reconciled = False

            mock_service = MagicMock()
            mock_service.get_reconciliation.return_value = mock_result
            mock_get.return_value = mock_service

            result = await handler.handle(request, "/api/v1/reconciliation/rec_123/approve", "POST")

            assert result is not None
            assert result.status_code == 200
            assert b"approved" in result.body
            assert mock_result.is_reconciled is True


class TestGetDiscrepancies:
    """Tests for getting discrepancies."""

    @pytest.mark.asyncio
    async def test_get_discrepancies_empty(self):
        """Test get discrepancies returns empty when no service."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {}

        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service"
        ) as mock_get:
            mock_get.return_value = None

            result = await handler.handle(request, "/api/v1/reconciliation/discrepancies", "GET")

            assert result is not None
            assert result.status_code == 200
            assert b"discrepancies" in result.body

    @pytest.mark.asyncio
    async def test_get_discrepancies_with_filters(self):
        """Test get discrepancies with status and severity filters."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {"status": "pending", "severity": "high", "limit": "10"}

        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service"
        ) as mock_get:
            mock_service = MagicMock()
            mock_service.list_reconciliations.return_value = []
            mock_get.return_value = mock_service

            result = await handler.handle(request, "/api/v1/reconciliation/discrepancies", "GET")

            assert result is not None
            assert result.status_code == 200


class TestGenerateReport:
    """Tests for report generation."""

    @pytest.mark.asyncio
    async def test_report_not_found(self):
        """Test report returns 404 when reconciliation not found."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {}

        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service"
        ) as mock_get:
            mock_service = MagicMock()
            mock_service.get_reconciliation.return_value = None
            mock_get.return_value = mock_service

            result = await handler.handle(request, "/api/v1/reconciliation/rec_123/report", "GET")

            assert result is not None
            assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_report_json_format(self):
        """Test report generation in JSON format."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {"format": "json"}

        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service"
        ) as mock_get:
            mock_result = MagicMock()
            mock_result.account_name = "Checking Account"
            mock_result.start_date = date(2024, 1, 1)
            mock_result.end_date = date(2024, 1, 31)
            mock_result.bank_total = Decimal("10000.00")
            mock_result.book_total = Decimal("9500.00")
            mock_result.difference = Decimal("500.00")
            mock_result.matched_count = 45
            mock_result.discrepancy_count = 5
            mock_result.match_rate = 0.90
            mock_result.is_reconciled = False
            mock_result.discrepancies = []

            mock_service = MagicMock()
            mock_service.get_reconciliation.return_value = mock_result
            mock_get.return_value = mock_service

            result = await handler.handle(request, "/api/v1/reconciliation/rec_123/report", "GET")

            assert result is not None
            assert result.status_code == 200
            assert b"report" in result.body
            assert b"summary" in result.body

    @pytest.mark.asyncio
    async def test_report_csv_format(self):
        """Test report generation in CSV format."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {"format": "csv"}

        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service"
        ) as mock_get:
            mock_result = MagicMock()
            mock_result.discrepancies = []

            mock_service = MagicMock()
            mock_service.get_reconciliation.return_value = mock_result
            mock_get.return_value = mock_service

            result = await handler.handle(request, "/api/v1/reconciliation/rec_123/report", "GET")

            assert result is not None
            assert result.status_code == 200
            assert result.content_type == "text/csv"

    @pytest.mark.asyncio
    async def test_report_unsupported_format(self):
        """Test report returns error for unsupported format."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {"format": "pdf"}

        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service"
        ) as mock_get:
            mock_result = MagicMock()
            mock_service = MagicMock()
            mock_service.get_reconciliation.return_value = mock_result
            mock_get.return_value = mock_service

            result = await handler.handle(request, "/api/v1/reconciliation/rec_123/report", "GET")

            assert result is not None
            assert result.status_code == 400
            assert b"Unsupported format" in result.body


class TestDemoEndpoint:
    """Tests for demo endpoint."""

    @pytest.mark.asyncio
    async def test_demo_endpoint(self):
        """Test demo endpoint returns mock data."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"

        # Mock at the source location (dynamic import)
        with patch(
            "aragora.services.accounting.reconciliation.get_mock_reconciliation_result"
        ) as mock_demo:
            mock_result = MagicMock()
            mock_result.to_dict.return_value = {"id": "demo_123"}
            mock_result.discrepancies = []
            mock_result.matched_transactions = []
            mock_demo.return_value = mock_result

            result = await handler.handle(request, "/api/v1/reconciliation/demo", "GET")

            assert result is not None
            assert result.status_code == 200
            assert b"is_demo" in result.body


class TestHandleReconciliation:
    """Tests for handle_reconciliation entry point."""

    @pytest.mark.asyncio
    async def test_entry_point(self):
        """Test entry point function."""
        request = MagicMock()
        request.tenant_id = "test"
        request.query = {}

        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service"
        ) as mock_get:
            mock_get.return_value = None

            result = await handle_reconciliation(request, "/api/v1/reconciliation/list", "GET")

            assert result is not None


class TestNotFoundRoute:
    """Tests for not found route."""

    @pytest.mark.asyncio
    async def test_unknown_route(self):
        """Test handling unknown route."""
        handler = ReconciliationHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"

        result = await handler.handle(request, "/api/v1/reconciliation/unknown/path", "GET")

        assert result is not None
        assert result.status_code == 404


class TestImports:
    """Test that imports work correctly."""

    def test_import_from_package(self):
        """Test imports from features package."""
        from aragora.server.handlers.features import (
            ReconciliationHandler,
            handle_reconciliation,
            get_reconciliation_handler,
        )

        assert ReconciliationHandler is not None
        assert handle_reconciliation is not None
        assert get_reconciliation_handler is not None
