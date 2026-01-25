"""
Tests for Audit Export API Handler.

Tests for audit log query and compliance exports:
- Query audit events
- Get audit statistics
- Export in JSON/CSV/SOC2 formats
- Verify audit log integrity
"""

import pytest
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
from aiohttp import web
from aiohttp.test_utils import make_mocked_request


class TestAuditEventsHandler:
    """Tests for audit events query endpoint."""

    @pytest.fixture
    def mock_audit_log(self):
        """Create mock audit log."""
        audit = MagicMock()
        audit.query.return_value = []
        return audit

    @pytest.mark.asyncio
    async def test_handle_audit_events_default_params(self, mock_audit_log):
        """Test querying events with default parameters."""
        from aragora.server.handlers.audit_export import handle_audit_events

        request = make_mocked_request("GET", "/api/v1/audit/events")

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            response = await handle_audit_events(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert "events" in data
        assert "count" in data
        assert "query" in data

    @pytest.mark.asyncio
    async def test_handle_audit_events_with_date_range(self, mock_audit_log):
        """Test querying events with date range."""
        from aragora.server.handlers.audit_export import handle_audit_events

        start = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        end = datetime.now(timezone.utc).isoformat()

        request = make_mocked_request(
            "GET",
            f"/api/v1/audit/events?start_date={start}&end_date={end}",
        )

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            with patch("aragora.audit.AuditQuery") as MockQuery:
                MockQuery.return_value = MagicMock()
                response = await handle_audit_events(request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_handle_audit_events_invalid_start_date(self, mock_audit_log):
        """Test querying with invalid start date."""
        from aragora.server.handlers.audit_export import handle_audit_events

        request = make_mocked_request(
            "GET",
            "/api/v1/audit/events?start_date=invalid",
        )

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            response = await handle_audit_events(request)

        assert response.status == 400
        data = json.loads(response.body)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_handle_audit_events_invalid_end_date(self, mock_audit_log):
        """Test querying with invalid end date."""
        from aragora.server.handlers.audit_export import handle_audit_events

        request = make_mocked_request(
            "GET",
            "/api/v1/audit/events?end_date=not-a-date",
        )

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            response = await handle_audit_events(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_handle_audit_events_with_filters(self, mock_audit_log):
        """Test querying with various filters."""
        from aragora.server.handlers.audit_export import handle_audit_events

        request = make_mocked_request(
            "GET",
            "/api/v1/audit/events?action=login&actor_id=user123&limit=50",
        )

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            with patch("aragora.audit.AuditQuery") as MockQuery:
                MockQuery.return_value = MagicMock()
                response = await handle_audit_events(request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_handle_audit_events_invalid_category(self, mock_audit_log):
        """Test querying with invalid category."""
        from aragora.server.handlers.audit_export import handle_audit_events

        request = make_mocked_request(
            "GET",
            "/api/v1/audit/events?category=invalid_category",
        )

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            with patch("aragora.audit.AuditQuery") as MockQuery:
                MockQuery.return_value = MagicMock()
                with patch(
                    "aragora.audit.AuditCategory",
                    side_effect=ValueError("Invalid"),
                ):
                    response = await handle_audit_events(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_handle_audit_events_with_results(self, mock_audit_log):
        """Test querying with actual results."""
        from aragora.server.handlers.audit_export import handle_audit_events

        mock_event = MagicMock()
        mock_event.to_dict.return_value = {
            "id": "evt_123",
            "action": "login",
            "actor_id": "user_456",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        mock_audit_log.query.return_value = [mock_event]

        request = make_mocked_request("GET", "/api/v1/audit/events")

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            response = await handle_audit_events(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["count"] == 1
        assert len(data["events"]) == 1


class TestAuditStatsHandler:
    """Tests for audit statistics endpoint."""

    @pytest.fixture
    def mock_audit_log(self):
        """Create mock audit log."""
        audit = MagicMock()
        audit.get_stats.return_value = {
            "total_events": 1000,
            "events_by_category": {"auth": 500, "data": 300, "admin": 200},
            "events_by_outcome": {"success": 900, "failure": 100},
        }
        return audit

    @pytest.mark.asyncio
    async def test_handle_audit_stats(self, mock_audit_log):
        """Test getting audit statistics."""
        from aragora.server.handlers.audit_export import handle_audit_stats

        request = make_mocked_request("GET", "/api/v1/audit/stats")

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            response = await handle_audit_stats(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["total_events"] == 1000
        assert "events_by_category" in data


class TestAuditExportHandler:
    """Tests for audit export endpoint."""

    @pytest.fixture
    def mock_audit_log(self):
        """Create mock audit log."""
        audit = MagicMock()
        audit.export_json.return_value = 10
        audit.export_csv.return_value = 10
        audit.export_soc2.return_value = {"events_exported": 10}
        return audit

    @pytest.mark.asyncio
    async def test_handle_audit_export_missing_start_date(self, mock_audit_log):
        """Test export without start_date."""
        from aragora.server.handlers.audit_export import handle_audit_export

        request = make_mocked_request("POST", "/api/v1/audit/export")
        request.json = AsyncMock(return_value={"end_date": "2024-01-01"})

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            response = await handle_audit_export(request)

        assert response.status == 400
        data = json.loads(response.body)
        assert "start_date" in data["error"]

    @pytest.mark.asyncio
    async def test_handle_audit_export_missing_end_date(self, mock_audit_log):
        """Test export without end_date."""
        from aragora.server.handlers.audit_export import handle_audit_export

        request = make_mocked_request("POST", "/api/v1/audit/export")
        request.json = AsyncMock(return_value={"start_date": "2024-01-01"})

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            response = await handle_audit_export(request)

        assert response.status == 400
        data = json.loads(response.body)
        assert "end_date" in data["error"]

    @pytest.mark.asyncio
    async def test_handle_audit_export_invalid_json(self, mock_audit_log):
        """Test export with invalid JSON body."""
        from aragora.server.handlers.audit_export import handle_audit_export

        request = make_mocked_request("POST", "/api/v1/audit/export")
        request.json = AsyncMock(side_effect=json.JSONDecodeError("msg", "doc", 0))

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            response = await handle_audit_export(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_handle_audit_export_invalid_date_format(self, mock_audit_log):
        """Test export with invalid date format."""
        from aragora.server.handlers.audit_export import handle_audit_export

        request = make_mocked_request("POST", "/api/v1/audit/export")
        request.json = AsyncMock(return_value={"start_date": "invalid", "end_date": "also-invalid"})

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            response = await handle_audit_export(request)

        assert response.status == 400
        data = json.loads(response.body)
        assert "date format" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_handle_audit_export_invalid_format(self, mock_audit_log):
        """Test export with invalid export format."""
        from aragora.server.handlers.audit_export import handle_audit_export

        request = make_mocked_request("POST", "/api/v1/audit/export")
        request.json = AsyncMock(
            return_value={"start_date": "2024-01-01", "end_date": "2024-01-31", "format": "xml"}
        )

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            response = await handle_audit_export(request)

        assert response.status == 400
        data = json.loads(response.body)
        assert "format" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_handle_audit_export_json_success(self, mock_audit_log):
        """Test successful JSON export."""
        from aragora.server.handlers.audit_export import handle_audit_export
        import tempfile

        request = make_mocked_request("POST", "/api/v1/audit/export")
        request.json = AsyncMock(
            return_value={"start_date": "2024-01-01", "end_date": "2024-01-31", "format": "json"}
        )

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            with patch.object(
                tempfile,
                "NamedTemporaryFile",
                return_value=MagicMock(
                    __enter__=MagicMock(return_value=MagicMock(name="/tmp/test.json")),
                    __exit__=MagicMock(),
                ),
            ):
                with patch(
                    "builtins.open",
                    MagicMock(
                        return_value=MagicMock(
                            __enter__=MagicMock(
                                return_value=MagicMock(
                                    read=MagicMock(return_value='{"events": []}')
                                )
                            ),
                            __exit__=MagicMock(),
                        )
                    ),
                ):
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch("pathlib.Path.unlink"):
                            response = await handle_audit_export(request)

        assert response.status == 200
        assert response.content_type == "application/json"
        assert "Content-Disposition" in response.headers

    @pytest.mark.asyncio
    async def test_handle_audit_export_csv_success(self, mock_audit_log):
        """Test successful CSV export."""
        from aragora.server.handlers.audit_export import handle_audit_export
        import tempfile

        request = make_mocked_request("POST", "/api/v1/audit/export")
        request.json = AsyncMock(
            return_value={"start_date": "2024-01-01", "end_date": "2024-01-31", "format": "csv"}
        )

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            with patch.object(
                tempfile,
                "NamedTemporaryFile",
                return_value=MagicMock(
                    __enter__=MagicMock(return_value=MagicMock(name="/tmp/test.csv")),
                    __exit__=MagicMock(),
                ),
            ):
                with patch(
                    "builtins.open",
                    MagicMock(
                        return_value=MagicMock(
                            __enter__=MagicMock(
                                return_value=MagicMock(
                                    read=MagicMock(return_value="id,action,timestamp\n")
                                )
                            ),
                            __exit__=MagicMock(),
                        )
                    ),
                ):
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch("pathlib.Path.unlink"):
                            response = await handle_audit_export(request)

        assert response.status == 200
        assert response.content_type == "text/csv"

    @pytest.mark.asyncio
    async def test_handle_audit_export_soc2_success(self, mock_audit_log):
        """Test successful SOC2 export."""
        from aragora.server.handlers.audit_export import handle_audit_export
        import tempfile

        request = make_mocked_request("POST", "/api/v1/audit/export")
        request.json = AsyncMock(
            return_value={"start_date": "2024-01-01", "end_date": "2024-01-31", "format": "soc2"}
        )

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            with patch.object(
                tempfile,
                "NamedTemporaryFile",
                return_value=MagicMock(
                    __enter__=MagicMock(return_value=MagicMock(name="/tmp/test.json")),
                    __exit__=MagicMock(),
                ),
            ):
                with patch(
                    "builtins.open",
                    MagicMock(
                        return_value=MagicMock(
                            __enter__=MagicMock(
                                return_value=MagicMock(
                                    read=MagicMock(return_value='{"soc2_report": {}}')
                                )
                            ),
                            __exit__=MagicMock(),
                        )
                    ),
                ):
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch("pathlib.Path.unlink"):
                            response = await handle_audit_export(request)

        assert response.status == 200
        assert response.content_type == "application/json"


class TestAuditVerifyHandler:
    """Tests for audit integrity verification endpoint."""

    @pytest.fixture
    def mock_audit_log(self):
        """Create mock audit log."""
        audit = MagicMock()
        return audit

    @pytest.mark.asyncio
    async def test_handle_audit_verify_success(self, mock_audit_log):
        """Test successful integrity verification."""
        from aragora.server.handlers.audit_export import handle_audit_verify

        mock_audit_log.verify_integrity.return_value = (True, [])

        request = make_mocked_request("POST", "/api/v1/audit/verify")
        request.json = AsyncMock(return_value={})

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            response = await handle_audit_verify(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["verified"] is True
        assert data["total_errors"] == 0

    @pytest.mark.asyncio
    async def test_handle_audit_verify_with_errors(self, mock_audit_log):
        """Test verification with integrity errors."""
        from aragora.server.handlers.audit_export import handle_audit_verify

        errors = ["Hash mismatch at event_123", "Missing link at event_456"]
        mock_audit_log.verify_integrity.return_value = (False, errors)

        request = make_mocked_request("POST", "/api/v1/audit/verify")
        request.json = AsyncMock(return_value={})

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            response = await handle_audit_verify(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert data["verified"] is False
        assert data["total_errors"] == 2

    @pytest.mark.asyncio
    async def test_handle_audit_verify_with_date_range(self, mock_audit_log):
        """Test verification with date range."""
        from aragora.server.handlers.audit_export import handle_audit_verify

        mock_audit_log.verify_integrity.return_value = (True, [])

        request = make_mocked_request("POST", "/api/v1/audit/verify")
        request.json = AsyncMock(
            return_value={"start_date": "2024-01-01", "end_date": "2024-01-31"}
        )

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            response = await handle_audit_verify(request)

        assert response.status == 200
        data = json.loads(response.body)
        assert "2024-01-01" in data["verified_range"]["start_date"]

    @pytest.mark.asyncio
    async def test_handle_audit_verify_invalid_start_date(self, mock_audit_log):
        """Test verification with invalid start date."""
        from aragora.server.handlers.audit_export import handle_audit_verify

        request = make_mocked_request("POST", "/api/v1/audit/verify")
        request.json = AsyncMock(return_value={"start_date": "invalid"})

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            response = await handle_audit_verify(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_handle_audit_verify_empty_body(self, mock_audit_log):
        """Test verification with empty body."""
        from aragora.server.handlers.audit_export import handle_audit_verify

        mock_audit_log.verify_integrity.return_value = (True, [])

        request = make_mocked_request("POST", "/api/v1/audit/verify")
        request.json = AsyncMock(side_effect=json.JSONDecodeError("msg", "doc", 0))

        with patch(
            "aragora.server.handlers.audit_export.get_audit_log",
            return_value=mock_audit_log,
        ):
            response = await handle_audit_verify(request)

        # Should handle empty body gracefully
        assert response.status == 200


class TestRegisterHandlers:
    """Tests for handler registration."""

    def test_register_handlers(self):
        """Test handlers are registered correctly."""
        from aragora.server.handlers.audit_export import register_handlers

        app = MagicMock()
        app.router = MagicMock()

        register_handlers(app)

        assert app.router.add_get.call_count >= 2
        assert app.router.add_post.call_count >= 2


class TestGetAuditLog:
    """Tests for audit log singleton."""

    def test_get_audit_log_creates_instance(self):
        """Test get_audit_log creates instance on first call."""
        import aragora.server.handlers.audit_export as module

        # Reset singleton
        module._audit_log = None

        with patch("aragora.audit.AuditLog") as MockAuditLog:
            MockAuditLog.return_value = MagicMock()
            result = module.get_audit_log()

            MockAuditLog.assert_called_once()
            assert result is not None

    def test_get_audit_log_returns_cached(self):
        """Test get_audit_log returns cached instance."""
        import aragora.server.handlers.audit_export as module

        mock_audit = MagicMock()
        module._audit_log = mock_audit

        result = module.get_audit_log()

        assert result is mock_audit

        # Reset for other tests
        module._audit_log = None
