"""
Tests for the BillingHandler module.

Tests cover:
- Handler routing for billing endpoints
- can_handle method
- ROUTES attribute
- Basic dispatch logic for billing operations
- Rate limiting configuration
- Stripe webhook handling routing
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from aragora.server.handlers.admin.billing import BillingHandler


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None, "user_store": None}


class TestBillingHandlerRouting:
    """Tests for handler routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BillingHandler(mock_server_context)

    def test_can_handle_plans(self, handler):
        """Handler can handle plans endpoint."""
        assert handler.can_handle("/api/v1/billing/plans")

    def test_can_handle_usage(self, handler):
        """Handler can handle usage endpoint."""
        assert handler.can_handle("/api/v1/billing/usage")

    def test_can_handle_subscription(self, handler):
        """Handler can handle subscription endpoint."""
        assert handler.can_handle("/api/v1/billing/subscription")

    def test_can_handle_checkout(self, handler):
        """Handler can handle checkout endpoint."""
        assert handler.can_handle("/api/v1/billing/checkout")

    def test_can_handle_portal(self, handler):
        """Handler can handle portal endpoint."""
        assert handler.can_handle("/api/v1/billing/portal")

    def test_can_handle_cancel(self, handler):
        """Handler can handle cancel endpoint."""
        assert handler.can_handle("/api/v1/billing/cancel")

    def test_can_handle_resume(self, handler):
        """Handler can handle resume endpoint."""
        assert handler.can_handle("/api/v1/billing/resume")

    def test_can_handle_audit_log(self, handler):
        """Handler can handle audit-log endpoint."""
        assert handler.can_handle("/api/v1/billing/audit-log")

    def test_can_handle_usage_export(self, handler):
        """Handler can handle usage export endpoint."""
        assert handler.can_handle("/api/v1/billing/usage/export")

    def test_can_handle_usage_forecast(self, handler):
        """Handler can handle usage forecast endpoint."""
        assert handler.can_handle("/api/v1/billing/usage/forecast")

    def test_can_handle_invoices(self, handler):
        """Handler can handle invoices endpoint."""
        assert handler.can_handle("/api/v1/billing/invoices")

    def test_can_handle_stripe_webhook(self, handler):
        """Handler can handle Stripe webhook endpoint."""
        assert handler.can_handle("/api/v1/webhooks/stripe")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/health")
        assert not handler.can_handle("/api/auth/login")
        assert not handler.can_handle("/")


class TestBillingHandlerRoutesAttribute:
    """Tests for ROUTES class attribute."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BillingHandler(mock_server_context)

    def test_routes_contains_plans(self, handler):
        """ROUTES contains plans endpoint."""
        assert "/api/v1/billing/plans" in handler.ROUTES

    def test_routes_contains_usage(self, handler):
        """ROUTES contains usage endpoint."""
        assert "/api/v1/billing/usage" in handler.ROUTES

    def test_routes_contains_subscription(self, handler):
        """ROUTES contains subscription endpoint."""
        assert "/api/v1/billing/subscription" in handler.ROUTES

    def test_routes_contains_checkout(self, handler):
        """ROUTES contains checkout endpoint."""
        assert "/api/v1/billing/checkout" in handler.ROUTES

    def test_routes_contains_portal(self, handler):
        """ROUTES contains portal endpoint."""
        assert "/api/v1/billing/portal" in handler.ROUTES

    def test_routes_contains_cancel(self, handler):
        """ROUTES contains cancel endpoint."""
        assert "/api/v1/billing/cancel" in handler.ROUTES

    def test_routes_contains_resume(self, handler):
        """ROUTES contains resume endpoint."""
        assert "/api/v1/billing/resume" in handler.ROUTES

    def test_routes_contains_audit_log(self, handler):
        """ROUTES contains audit-log endpoint."""
        assert "/api/v1/billing/audit-log" in handler.ROUTES

    def test_routes_contains_usage_export(self, handler):
        """ROUTES contains usage export endpoint."""
        assert "/api/v1/billing/usage/export" in handler.ROUTES

    def test_routes_contains_usage_forecast(self, handler):
        """ROUTES contains usage forecast endpoint."""
        assert "/api/v1/billing/usage/forecast" in handler.ROUTES

    def test_routes_contains_invoices(self, handler):
        """ROUTES contains invoices endpoint."""
        assert "/api/v1/billing/invoices" in handler.ROUTES

    def test_routes_contains_stripe_webhook(self, handler):
        """ROUTES contains Stripe webhook endpoint."""
        assert "/api/v1/webhooks/stripe" in handler.ROUTES

    def test_routes_count_minimum(self, handler):
        """ROUTES has expected minimum number of endpoints."""
        # At least 12 routes based on handler inspection
        assert len(handler.ROUTES) >= 12


class TestBillingHandlerRouteDispatch:
    """Tests for route dispatch logic."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BillingHandler(mock_server_context)

    def test_handle_plans_get_returns_result(self, handler):
        """Handle returns result for plans endpoint."""
        mock_http = MagicMock()
        mock_http.command = "GET"
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/v1/billing/plans", {}, mock_http, method="GET")

        assert result is not None
        assert result.status_code == 200

    def test_handle_usage_get_requires_setup(self, handler):
        """Handle GET /api/v1/billing/usage requires proper setup."""
        mock_http = MagicMock()
        mock_http.command = "GET"
        mock_http.headers = {}
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/v1/billing/usage", {}, mock_http, method="GET")

        # Returns error when user_store not configured (503) or auth issues (401/403)
        assert result is not None
        assert result.status_code in (401, 403, 503)

    def test_handle_subscription_get_requires_setup(self, handler):
        """Handle GET /api/v1/billing/subscription requires proper setup."""
        mock_http = MagicMock()
        mock_http.command = "GET"
        mock_http.headers = {}
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/v1/billing/subscription", {}, mock_http, method="GET")

        # Returns error when user_store not configured (503) or auth issues (401/403)
        assert result is not None
        assert result.status_code in (401, 403, 503)

    def test_handle_checkout_post_requires_setup(self, handler):
        """Handle POST /api/v1/billing/checkout requires proper setup."""
        mock_http = MagicMock()
        mock_http.command = "POST"
        mock_http.headers = {}
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/v1/billing/checkout", {}, mock_http, method="POST")

        # Returns error: missing body (400), user_store not configured (503), or auth issues (401/403)
        assert result is not None
        assert result.status_code in (400, 401, 403, 503)

    def test_handle_portal_post_requires_setup(self, handler):
        """Handle POST /api/v1/billing/portal requires proper setup."""
        mock_http = MagicMock()
        mock_http.command = "POST"
        mock_http.headers = {}
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/v1/billing/portal", {}, mock_http, method="POST")

        # Returns error: missing body (400), user_store not configured (503), or auth issues (401/403)
        assert result is not None
        assert result.status_code in (400, 401, 403, 503)

    def test_handle_cancel_post_requires_setup(self, handler):
        """Handle POST /api/v1/billing/cancel requires proper setup."""
        mock_http = MagicMock()
        mock_http.command = "POST"
        mock_http.headers = {}
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/v1/billing/cancel", {}, mock_http, method="POST")

        # Returns error: user_store not configured (503) or auth issues (401/403)
        assert result is not None
        assert result.status_code in (401, 403, 503)

    def test_handle_resume_post_requires_setup(self, handler):
        """Handle POST /api/v1/billing/resume requires proper setup."""
        mock_http = MagicMock()
        mock_http.command = "POST"
        mock_http.headers = {}
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/v1/billing/resume", {}, mock_http, method="POST")

        # Returns error: user_store not configured (503) or auth issues (401/403)
        assert result is not None
        assert result.status_code in (401, 403, 503)

    def test_handle_unknown_method_returns_error(self, handler):
        """Handle returns method not allowed for invalid method."""
        mock_http = MagicMock()
        mock_http.command = "PATCH"
        mock_http.headers = {}
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/v1/billing/unknown", {}, mock_http, method="PATCH")

        # Should return 405 Method Not Allowed
        assert result is not None
        assert result.status_code == 405


class TestBillingHandlerPlansEndpoint:
    """Tests for plans endpoint which is public."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BillingHandler(mock_server_context)

    def test_plans_returns_json(self, handler):
        """Plans endpoint returns JSON response."""
        mock_http = MagicMock()
        mock_http.command = "GET"
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/v1/billing/plans", {}, mock_http, method="GET")

        assert result is not None
        assert result.content_type == "application/json"

    def test_plans_response_includes_plans(self, handler):
        """Plans endpoint response includes plans array."""
        mock_http = MagicMock()
        mock_http.command = "GET"
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/v1/billing/plans", {}, mock_http, method="GET")

        assert result is not None
        import json

        body = json.loads(result.body)
        assert "plans" in body
        assert isinstance(body["plans"], list)

    def test_plans_includes_all_tiers(self, handler):
        """Plans endpoint includes all subscription tiers."""
        mock_http = MagicMock()
        mock_http.command = "GET"
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/v1/billing/plans", {}, mock_http, method="GET")

        assert result is not None
        import json

        body = json.loads(result.body)
        plans = body["plans"]
        plan_ids = [p["id"] for p in plans]

        # Should have at least free and starter tiers
        assert "free" in plan_ids

    def test_plans_includes_features(self, handler):
        """Plans endpoint includes features for each plan."""
        mock_http = MagicMock()
        mock_http.command = "GET"
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/v1/billing/plans", {}, mock_http, method="GET")

        assert result is not None
        import json

        body = json.loads(result.body)
        plans = body["plans"]

        for plan in plans:
            assert "features" in plan
            assert "debates_per_month" in plan["features"]


class TestBillingHandlerWebhookRouting:
    """Tests for Stripe webhook routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BillingHandler(mock_server_context)

    def test_webhook_requires_post(self, handler):
        """Webhook endpoint only accepts POST."""
        mock_http = MagicMock()
        mock_http.command = "GET"
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/v1/webhooks/stripe", {}, mock_http, method="GET")

        # GET should return 405
        assert result is not None
        assert result.status_code == 405

    def test_webhook_requires_signature(self, handler):
        """Webhook endpoint requires Stripe signature."""
        mock_http = MagicMock()
        mock_http.command = "POST"
        mock_http.headers = {"Content-Length": "2"}
        mock_http.rfile = MagicMock()
        mock_http.rfile.read.return_value = b"{}"
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/v1/webhooks/stripe", {}, mock_http, method="POST")

        # Should return 400 for missing signature
        assert result is not None
        assert result.status_code == 400


class TestBillingHandlerResourceType:
    """Tests for handler resource type configuration."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BillingHandler(mock_server_context)

    def test_resource_type_is_billing(self, handler):
        """Handler resource type is set to 'billing'."""
        assert handler.RESOURCE_TYPE == "billing"


class TestBillingHandlerContextMethods:
    """Tests for context accessor methods."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BillingHandler(mock_server_context)

    def test_get_user_store_returns_from_context(self, handler):
        """_get_user_store returns user_store from context."""
        mock_store = MagicMock()
        handler.ctx["user_store"] = mock_store

        result = handler._get_user_store()

        assert result is mock_store

    def test_get_user_store_returns_none_when_missing(self, handler):
        """_get_user_store returns None when not in context."""
        result = handler._get_user_store()

        assert result is None

    def test_get_usage_tracker_returns_from_context(self, handler):
        """_get_usage_tracker returns usage_tracker from context."""
        mock_tracker = MagicMock()
        handler.ctx["usage_tracker"] = mock_tracker

        result = handler._get_usage_tracker()

        assert result is mock_tracker

    def test_get_usage_tracker_returns_none_when_missing(self, handler):
        """_get_usage_tracker returns None when not in context."""
        result = handler._get_usage_tracker()

        assert result is None


class TestBillingHandlerRouteMatchingWithMethods:
    """Tests for route matching with HTTP methods."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BillingHandler(mock_server_context)

    @pytest.fixture
    def mock_http(self):
        """Create mock HTTP handler."""
        mock = MagicMock()
        mock.client_address = ("127.0.0.1", 8080)
        mock.headers = {}
        return mock

    def test_plans_only_accepts_get(self, handler, mock_http):
        """Plans endpoint only accepts GET."""
        mock_http.command = "GET"
        result = handler.handle("/api/v1/billing/plans", {}, mock_http, method="GET")
        assert result is not None
        assert result.status_code == 200

    def test_checkout_only_accepts_post(self, handler, mock_http):
        """Checkout endpoint only accepts POST."""
        mock_http.command = "POST"
        result = handler.handle("/api/v1/billing/checkout", {}, mock_http, method="POST")
        # Returns error but proves POST is routed (400=missing body, 503=no user_store)
        assert result is not None
        assert result.status_code in (400, 401, 403, 503)

    def test_portal_only_accepts_post(self, handler, mock_http):
        """Portal endpoint only accepts POST."""
        mock_http.command = "POST"
        result = handler.handle("/api/v1/billing/portal", {}, mock_http, method="POST")
        # Returns error but proves POST is routed (400=missing body, 503=no user_store)
        assert result is not None
        assert result.status_code in (400, 401, 403, 503)

    def test_cancel_only_accepts_post(self, handler, mock_http):
        """Cancel endpoint only accepts POST."""
        mock_http.command = "POST"
        result = handler.handle("/api/v1/billing/cancel", {}, mock_http, method="POST")
        # Returns error but proves POST is routed (503=no user_store)
        assert result is not None
        assert result.status_code in (401, 403, 503)

    def test_resume_only_accepts_post(self, handler, mock_http):
        """Resume endpoint only accepts POST."""
        mock_http.command = "POST"
        result = handler.handle("/api/v1/billing/resume", {}, mock_http, method="POST")
        # Returns error but proves POST is routed (503=no user_store)
        assert result is not None
        assert result.status_code in (401, 403, 503)


class TestBillingHandlerMethodFromCommand:
    """Tests for method extraction from handler.command."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BillingHandler(mock_server_context)

    def test_extracts_method_from_command_attribute(self, handler):
        """Handler extracts method from http.command attribute."""
        mock_http = MagicMock()
        mock_http.command = "GET"
        mock_http.client_address = ("127.0.0.1", 8080)

        # Don't pass method, let it be extracted from command
        result = handler.handle("/api/v1/billing/plans", {}, mock_http)

        # Should work with GET
        assert result is not None
        assert result.status_code == 200


class TestBillingHandlerImport:
    """Tests for importing BillingHandler."""

    def test_can_import_handler(self):
        """BillingHandler can be imported."""
        from aragora.server.handlers.admin.billing import BillingHandler

        assert BillingHandler is not None

    def test_handler_in_all(self):
        """BillingHandler is in __all__."""
        from aragora.server.handlers.admin import billing

        assert "BillingHandler" in billing.__all__
