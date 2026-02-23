"""Tests for FeaturesHandler REST endpoints.

Covers all routes and behavior of the FeaturesHandler class:
- can_handle() routing for all ROUTES and parameterized paths
- GET /api/features - Feature availability summary
- GET /api/features/available - List of available features
- GET /api/features/all - Full feature matrix grouped by category
- GET /api/features/handlers - Handler stability classifications
- GET /api/features/config - User feature preferences (GET)
- POST /api/features/config - Update user feature preferences (POST)
- GET /api/features/discover - Full API discovery catalog
- GET /api/features/endpoints - Flat list of all endpoints
- GET /api/features/{feature_id} - Individual feature status
- Rate limiting
- Feature detection helpers
- Feature unavailable response helper
- Error handling and edge cases
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.features.features import (
    FEATURE_REGISTRY,
    FeatureInfo,
    FeaturesHandler,
    _check_feature_available,
    feature_unavailable_response,
    get_all_features,
    get_available_features,
    get_unavailable_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to FeaturesHandler methods."""

    def __init__(
        self,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        path: str = "",
        headers: dict[str, str] | None = None,
    ):
        self.command = method
        self.headers: dict[str, str] = headers or {"User-Agent": "test-agent"}
        self.rfile = MagicMock()
        self.client_address = ("127.0.0.1", 12345)
        self.path = path

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers["Content-Length"] = str(len(raw))
            self.headers["Content-Type"] = "application/json"
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a FeaturesHandler with minimal server context."""
    ctx: dict[str, Any] = {}
    return FeaturesHandler(ctx)


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the rate limiter before each test to prevent cross-test leaking."""
    from aragora.server.handlers.features.features import _features_limiter

    _features_limiter._buckets.clear()
    yield
    _features_limiter._buckets.clear()


@pytest.fixture(autouse=True)
def _disable_rate_limit(monkeypatch):
    """Disable rate limiting globally for tests."""
    import importlib

    rl_mod = importlib.import_module("aragora.server.handlers.utils.rate_limit")
    monkeypatch.setattr(rl_mod, "RATE_LIMITING_DISABLED", True)


# ---------------------------------------------------------------------------
# can_handle() routing
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for route matching via can_handle()."""

    def test_features_root(self, handler):
        assert handler.can_handle("/api/features")

    def test_features_root_versioned(self, handler):
        assert handler.can_handle("/api/v1/features")

    def test_features_available(self, handler):
        assert handler.can_handle("/api/features/available")

    def test_features_available_versioned(self, handler):
        assert handler.can_handle("/api/v1/features/available")

    def test_features_all(self, handler):
        assert handler.can_handle("/api/features/all")

    def test_features_all_versioned(self, handler):
        assert handler.can_handle("/api/v1/features/all")

    def test_features_handlers(self, handler):
        assert handler.can_handle("/api/features/handlers")

    def test_features_handlers_versioned(self, handler):
        assert handler.can_handle("/api/v1/features/handlers")

    def test_features_config(self, handler):
        assert handler.can_handle("/api/features/config")

    def test_features_config_versioned(self, handler):
        assert handler.can_handle("/api/v1/features/config")

    def test_features_discover(self, handler):
        assert handler.can_handle("/api/features/discover")

    def test_features_discover_versioned(self, handler):
        assert handler.can_handle("/api/v1/features/discover")

    def test_features_endpoints(self, handler):
        assert handler.can_handle("/api/features/endpoints")

    def test_features_endpoints_versioned(self, handler):
        assert handler.can_handle("/api/v1/features/endpoints")

    def test_parameterized_feature_id(self, handler):
        """Parameterized route /api/features/{feature_id} should match."""
        assert handler.can_handle("/api/features/pulse")

    def test_parameterized_feature_id_versioned(self, handler):
        assert handler.can_handle("/api/v1/features/pulse")

    def test_parameterized_genesis(self, handler):
        assert handler.can_handle("/api/features/genesis")

    def test_parameterized_elo(self, handler):
        assert handler.can_handle("/api/features/elo")

    def test_rejects_unrelated_path(self, handler):
        assert not handler.can_handle("/api/debates")

    def test_rejects_deeply_nested_path(self, handler):
        """Paths with 5+ segments under /api/features/ should not match."""
        assert not handler.can_handle("/api/features/pulse/details/extra")

    def test_rejects_root_api(self, handler):
        assert not handler.can_handle("/api")

    def test_rejects_health(self, handler):
        assert not handler.can_handle("/api/health")

    def test_rejects_empty_path(self, handler):
        assert not handler.can_handle("")

    def test_rejects_partial_prefix(self, handler):
        assert not handler.can_handle("/api/feature")


# ---------------------------------------------------------------------------
# GET /api/features (summary)
# ---------------------------------------------------------------------------


class TestGetFeaturesSummary:
    """Tests for the features summary endpoint."""

    def test_returns_200(self, handler):
        result = handler.handle("/api/features", {})
        assert _status(result) == 200

    def test_returns_expected_keys(self, handler):
        result = handler.handle("/api/features", {})
        body = _body(result)
        assert "available_count" in body
        assert "unavailable_count" in body
        assert "available" in body
        assert "unavailable" in body
        assert "categories" in body

    def test_available_is_list(self, handler):
        result = handler.handle("/api/features", {})
        body = _body(result)
        assert isinstance(body["available"], list)

    def test_unavailable_is_list(self, handler):
        result = handler.handle("/api/features", {})
        body = _body(result)
        assert isinstance(body["unavailable"], list)

    def test_categories_is_dict(self, handler):
        result = handler.handle("/api/features", {})
        body = _body(result)
        assert isinstance(body["categories"], dict)

    def test_counts_add_up(self, handler):
        result = handler.handle("/api/features", {})
        body = _body(result)
        total_registered = len(FEATURE_REGISTRY)
        assert body["available_count"] + body["unavailable_count"] == total_registered

    def test_versioned_path(self, handler):
        result = handler.handle("/api/v1/features", {})
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# GET /api/features/available
# ---------------------------------------------------------------------------


class TestGetAvailable:
    """Tests for the available features endpoint."""

    def test_returns_200(self, handler):
        result = handler.handle("/api/features/available", {})
        assert _status(result) == 200

    def test_returns_features_and_count(self, handler):
        result = handler.handle("/api/features/available", {})
        body = _body(result)
        assert "features" in body
        assert "count" in body
        assert body["count"] == len(body["features"])

    def test_features_are_strings(self, handler):
        result = handler.handle("/api/features/available", {})
        body = _body(result)
        for f in body["features"]:
            assert isinstance(f, str)

    def test_versioned_path(self, handler):
        result = handler.handle("/api/v1/features/available", {})
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# GET /api/features/all
# ---------------------------------------------------------------------------


class TestGetAllFeatures:
    """Tests for the full feature matrix endpoint."""

    def test_returns_200(self, handler):
        result = handler.handle("/api/features/all", {})
        assert _status(result) == 200

    def test_returns_features_dict(self, handler):
        result = handler.handle("/api/features/all", {})
        body = _body(result)
        assert "features" in body
        assert isinstance(body["features"], dict)

    def test_returns_by_category(self, handler):
        result = handler.handle("/api/features/all", {})
        body = _body(result)
        assert "by_category" in body
        assert isinstance(body["by_category"], dict)

    def test_returns_total_count(self, handler):
        result = handler.handle("/api/features/all", {})
        body = _body(result)
        assert body["total"] == len(FEATURE_REGISTRY)

    def test_feature_entry_structure(self, handler):
        result = handler.handle("/api/features/all", {})
        body = _body(result)
        # Pick the first feature entry and check its structure
        for feature_id, info in body["features"].items():
            assert "id" in info
            assert "name" in info
            assert "description" in info
            assert "category" in info
            assert "status" in info
            assert "available" in info
            assert "endpoints" in info
            break  # Just check the first one

    def test_by_category_groups_correct(self, handler):
        result = handler.handle("/api/features/all", {})
        body = _body(result)
        # Every feature should appear in some category group
        all_ids_in_categories = set()
        for cat_features in body["by_category"].values():
            for f in cat_features:
                all_ids_in_categories.add(f["id"])
        assert all_ids_in_categories == set(body["features"].keys())

    def test_versioned_path(self, handler):
        result = handler.handle("/api/v1/features/all", {})
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# GET /api/features/{feature_id}
# ---------------------------------------------------------------------------


class TestGetFeatureStatus:
    """Tests for individual feature status endpoint."""

    def test_known_feature_returns_200(self, handler):
        result = handler.handle("/api/features/pulse", {})
        assert _status(result) == 200

    def test_known_feature_structure(self, handler):
        result = handler.handle("/api/features/pulse", {})
        body = _body(result)
        assert body["id"] == "pulse"
        assert body["name"] == "Trending Topics"
        assert "description" in body
        assert "category" in body
        assert "status" in body
        assert "available" in body
        assert "endpoints" in body
        assert "requires" in body

    def test_genesis_feature(self, handler):
        result = handler.handle("/api/features/genesis", {})
        body = _body(result)
        assert body["id"] == "genesis"
        assert body["name"] == "Agent Evolution"

    def test_elo_feature(self, handler):
        result = handler.handle("/api/features/elo", {})
        body = _body(result)
        assert body["id"] == "elo"
        assert body["category"] == "competition"

    def test_supermemory_feature(self, handler):
        result = handler.handle("/api/features/supermemory", {})
        body = _body(result)
        assert body["id"] == "supermemory"
        assert body["category"] == "memory"

    def test_unknown_feature_returns_404(self, handler):
        result = handler.handle("/api/features/nonexistent_xyz", {})
        assert _status(result) == 404

    def test_unknown_feature_error_body(self, handler):
        result = handler.handle("/api/features/nonexistent_xyz", {})
        body = _body(result)
        assert "error" in body or "message" in body

    def test_available_feature_has_no_install_hint(self, handler):
        """When a feature is available, install_hint should be None."""
        with patch(
            "aragora.server.handlers.features.features._check_feature_available",
            return_value=(True, None),
        ):
            result = handler.handle("/api/features/elo", {})
            body = _body(result)
            assert body["install_hint"] is None

    def test_unavailable_feature_has_install_hint(self, handler):
        """When a feature is unavailable, install_hint should be set."""
        with patch(
            "aragora.server.handlers.features.features._check_feature_available",
            return_value=(False, "Not installed"),
        ):
            result = handler.handle("/api/features/pulse", {})
            body = _body(result)
            assert body["install_hint"] is not None
            assert len(body["install_hint"]) > 0

    def test_versioned_path_feature(self, handler):
        result = handler.handle("/api/v1/features/pulse", {})
        assert _status(result) == 200

    def test_all_registered_features_accessible(self, handler):
        """Every feature in the registry should be accessible via the endpoint."""
        for feature_id in FEATURE_REGISTRY:
            result = handler.handle(f"/api/features/{feature_id}", {})
            assert _status(result) == 200, f"Feature {feature_id} returned non-200"


# ---------------------------------------------------------------------------
# GET /api/features/handlers
# ---------------------------------------------------------------------------


class TestGetHandlerStability:
    """Tests for handler stability endpoint."""

    def test_returns_200(self, handler):
        mock_all_handlers = [type("FakeHandler", (), {"__name__": "FakeHandler"})]
        mock_stability = {"FakeHandler": "stable"}
        with (
            patch(
                "aragora.server.handlers.ALL_HANDLERS",
                mock_all_handlers,
            ),
            patch(
                "aragora.server.handlers.get_all_handler_stability",
                return_value=mock_stability,
            ),
        ):
            result = handler.handle("/api/features/handlers", {})
            assert _status(result) == 200

    def test_response_structure(self, handler):
        mock_all_handlers = [
            type("StableHandler", (), {"__name__": "StableHandler"}),
            type("ExperimentalHandler", (), {"__name__": "ExperimentalHandler"}),
        ]
        mock_stability = {
            "StableHandler": "stable",
            "ExperimentalHandler": "experimental",
        }
        with (
            patch(
                "aragora.server.handlers.ALL_HANDLERS",
                mock_all_handlers,
            ),
            patch(
                "aragora.server.handlers.get_all_handler_stability",
                return_value=mock_stability,
            ),
        ):
            result = handler.handle("/api/features/handlers", {})
            body = _body(result)
            assert "handlers" in body
            assert "by_stability" in body
            assert "counts" in body
            assert "total" in body

    def test_groups_by_stability_level(self, handler):
        mock_all_handlers = [
            type("HandlerA", (), {"__name__": "HandlerA"}),
            type("HandlerB", (), {"__name__": "HandlerB"}),
        ]
        mock_stability = {"HandlerA": "stable", "HandlerB": "deprecated"}
        with (
            patch(
                "aragora.server.handlers.ALL_HANDLERS",
                mock_all_handlers,
            ),
            patch(
                "aragora.server.handlers.get_all_handler_stability",
                return_value=mock_stability,
            ),
        ):
            result = handler.handle("/api/features/handlers", {})
            body = _body(result)
            assert "HandlerA" in body["by_stability"]["stable"]
            assert "HandlerB" in body["by_stability"]["deprecated"]

    def test_unknown_handler_defaults_to_experimental(self, handler):
        mock_all_handlers = [
            type("UnknownHandler", (), {"__name__": "UnknownHandler"}),
        ]
        # Stability map does NOT include UnknownHandler
        mock_stability = {}
        with (
            patch(
                "aragora.server.handlers.ALL_HANDLERS",
                mock_all_handlers,
            ),
            patch(
                "aragora.server.handlers.get_all_handler_stability",
                return_value=mock_stability,
            ),
        ):
            result = handler.handle("/api/features/handlers", {})
            body = _body(result)
            assert "UnknownHandler" in body["by_stability"]["experimental"]

    def test_versioned_path(self, handler):
        mock_all_handlers = []
        mock_stability = {}
        with (
            patch(
                "aragora.server.handlers.ALL_HANDLERS",
                mock_all_handlers,
            ),
            patch(
                "aragora.server.handlers.get_all_handler_stability",
                return_value=mock_stability,
            ),
        ):
            result = handler.handle("/api/v1/features/handlers", {})
            assert _status(result) == 200


# ---------------------------------------------------------------------------
# GET /api/features/config
# ---------------------------------------------------------------------------


class TestGetConfig:
    """Tests for GET /api/features/config (user preferences)."""

    def test_returns_200_unauthenticated(self, handler):
        mock_handler = MockHTTPHandler(method="GET")
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=None,
        ):
            result = handler.handle("/api/features/config", {}, mock_handler)
            assert _status(result) == 200

    def test_returns_default_preferences(self, handler):
        mock_handler = MockHTTPHandler(method="GET")
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=None,
        ):
            result = handler.handle("/api/features/config", {}, mock_handler)
            body = _body(result)
            assert "preferences" in body
            assert "defaults" in body
            assert body["defaults"] == FeaturesHandler.DEFAULT_PREFERENCES

    def test_unauthenticated_flag(self, handler):
        mock_handler = MockHTTPHandler(method="GET")
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=None,
        ):
            result = handler.handle("/api/features/config", {}, mock_handler)
            body = _body(result)
            assert body["is_authenticated"] is False

    def test_authenticated_user_loads_preferences(self, handler):
        """Authenticated user gets stored preferences merged with defaults."""
        user_ctx = MagicMock()
        user_ctx.is_authenticated = True
        user_ctx.user_id = "user-123"

        user_store = MagicMock()
        user_store.get_user_preferences.return_value = {
            "trickster": True,
            "compact_mode": True,
        }

        handler.ctx["user_store"] = user_store
        mock_handler = MockHTTPHandler(method="GET")

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=user_ctx,
        ):
            result = handler.handle("/api/features/config", {}, mock_handler)
            body = _body(result)
            assert body["is_authenticated"] is True
            # User prefs should override defaults
            assert body["preferences"]["trickster"] is True
            assert body["preferences"]["compact_mode"] is True
            # Default values should still be present for unset keys
            assert "calibration" in body["preferences"]

    def test_authenticated_user_store_error_falls_back_to_defaults(self, handler):
        """If user store raises, handler falls back to defaults gracefully."""
        user_ctx = MagicMock()
        user_ctx.is_authenticated = True
        user_ctx.user_id = "user-123"

        user_store = MagicMock()
        user_store.get_user_preferences.side_effect = ValueError("DB error")

        handler.ctx["user_store"] = user_store
        mock_handler = MockHTTPHandler(method="GET")

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=user_ctx,
        ):
            result = handler.handle("/api/features/config", {}, mock_handler)
            assert _status(result) == 200
            body = _body(result)
            # Should still return defaults
            assert body["preferences"]["trickster"] is False  # default

    def test_feature_toggles_in_response(self, handler):
        mock_handler = MockHTTPHandler(method="GET")
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=None,
        ):
            result = handler.handle("/api/features/config", {}, mock_handler)
            body = _body(result)
            assert "feature_toggles" in body
            assert isinstance(body["feature_toggles"], list)

    def test_feature_toggles_have_expected_fields(self, handler):
        mock_handler = MockHTTPHandler(method="GET")
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=None,
        ):
            result = handler.handle("/api/features/config", {}, mock_handler)
            body = _body(result)
            if body["feature_toggles"]:
                toggle = body["feature_toggles"][0]
                assert "id" in toggle
                assert "name" in toggle
                assert "description" in toggle
                assert "category" in toggle
                assert "available" in toggle
                assert "enabled" in toggle

    def test_versioned_path(self, handler):
        mock_handler = MockHTTPHandler(method="GET")
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=None,
        ):
            result = handler.handle("/api/v1/features/config", {}, mock_handler)
            assert _status(result) == 200


# ---------------------------------------------------------------------------
# POST /api/features/config
# ---------------------------------------------------------------------------


class TestUpdateConfig:
    """Tests for POST /api/features/config (update user preferences)."""

    def test_guest_update_acknowledged(self, handler):
        """Guest updates are acknowledged but not persisted server-side."""
        body_data = {"trickster": True}
        mock_handler = MockHTTPHandler(method="POST", body=body_data)

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=None,
        ):
            result = handler.handle("/api/features/config", {}, mock_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["success"] is True
            assert body["is_authenticated"] is False
            assert "trickster" in body["updated"]

    def test_authenticated_update_persisted(self, handler):
        """Authenticated user updates are saved to the user store."""
        user_ctx = MagicMock()
        user_ctx.is_authenticated = True
        user_ctx.user_id = "user-456"

        user_store = MagicMock()
        user_store.get_user_preferences.return_value = {}

        handler.ctx["user_store"] = user_store
        body_data = {"compact_mode": True}
        mock_handler = MockHTTPHandler(method="POST", body=body_data)

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=user_ctx,
        ):
            result = handler.handle("/api/features/config", {}, mock_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["success"] is True
            assert "compact_mode" in body["updated"]
            user_store.set_user_preferences.assert_called_once()

    def test_invalid_keys_rejected(self, handler):
        """Updating with unknown preference keys returns 400."""
        body_data = {"invalid_key_xyz": True}
        mock_handler = MockHTTPHandler(method="POST", body=body_data)

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=None,
        ):
            result = handler.handle("/api/features/config", {}, mock_handler)
            assert _status(result) == 400

    def test_invalid_type_rejected(self, handler):
        """Updating with wrong value type returns 400."""
        # "trickster" expects bool, not string
        body_data = {"trickster": "yes"}
        mock_handler = MockHTTPHandler(method="POST", body=body_data)

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=None,
        ):
            result = handler.handle("/api/features/config", {}, mock_handler)
            assert _status(result) == 400

    def test_invalid_json_body(self, handler):
        """Malformed JSON body returns 400."""
        mock_handler = MockHTTPHandler(method="POST")
        # Override rfile with invalid JSON
        mock_handler.rfile.read.return_value = b"not json"
        mock_handler.headers["Content-Length"] = "8"

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=None,
        ):
            result = handler.handle("/api/features/config", {}, mock_handler)
            assert _status(result) == 400

    def test_empty_body_is_valid(self, handler):
        """Empty update (no keys) should succeed."""
        body_data = {}
        mock_handler = MockHTTPHandler(method="POST", body=body_data)

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=None,
        ):
            result = handler.handle("/api/features/config", {}, mock_handler)
            assert _status(result) == 200

    def test_save_failure_returns_500(self, handler):
        """If user store save fails, returns 500."""
        user_ctx = MagicMock()
        user_ctx.is_authenticated = True
        user_ctx.user_id = "user-789"

        user_store = MagicMock()
        user_store.get_user_preferences.return_value = {}
        user_store.set_user_preferences.side_effect = OSError("Disk error")

        handler.ctx["user_store"] = user_store
        body_data = {"trickster": True}
        mock_handler = MockHTTPHandler(method="POST", body=body_data)

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=user_ctx,
        ):
            result = handler.handle("/api/features/config", {}, mock_handler)
            assert _status(result) == 500

    def test_method_not_allowed(self, handler):
        """Methods other than GET/POST return 405."""
        mock_handler = MockHTTPHandler(method="DELETE")

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=None,
        ):
            result = handler.handle("/api/features/config", {}, mock_handler)
            assert _status(result) == 405

    def test_multiple_valid_keys_updated(self, handler):
        """Multiple valid preference keys can be updated at once."""
        body_data = {
            "trickster": True,
            "compact_mode": True,
            "theme": "dark",
        }
        mock_handler = MockHTTPHandler(method="POST", body=body_data)

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=None,
        ):
            result = handler.handle("/api/features/config", {}, mock_handler)
            assert _status(result) == 200
            body = _body(result)
            assert set(body["updated"]) == {"trickster", "compact_mode", "theme"}

    def test_int_type_validation(self, handler):
        """Integer preferences must be integers, not strings."""
        body_data = {"default_rounds": "three"}
        mock_handler = MockHTTPHandler(method="POST", body=body_data)

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=None,
        ):
            result = handler.handle("/api/features/config", {}, mock_handler)
            assert _status(result) == 400

    def test_float_type_validation(self, handler):
        """Float preferences must be numeric."""
        body_data = {"consensus_alert_threshold": "high"}
        mock_handler = MockHTTPHandler(method="POST", body=body_data)

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=None,
        ):
            result = handler.handle("/api/features/config", {}, mock_handler)
            assert _status(result) == 400


# ---------------------------------------------------------------------------
# GET /api/features/discover
# ---------------------------------------------------------------------------


class TestGetApiDiscovery:
    """Tests for the API discovery catalog endpoint."""

    def _mock_handlers(self):
        """Create mock handler classes for discovery tests."""
        handler_a = type(
            "DebatesHandler",
            (),
            {"__name__": "DebatesHandler", "ROUTES": ["/api/v1/debates", "/api/v1/debates/recent"]},
        )
        handler_b = type(
            "HealthHandler",
            (),
            {"__name__": "HealthHandler", "ROUTES": ["/api/v1/health", "/healthz"]},
        )
        return [handler_a, handler_b]

    def test_returns_200(self, handler):
        mock_handlers = self._mock_handlers()
        with (
            patch("aragora.server.handlers.ALL_HANDLERS", mock_handlers),
            patch(
                "aragora.server.handlers.get_all_handler_stability",
                return_value={"DebatesHandler": "stable", "HealthHandler": "stable"},
            ),
        ):
            result = handler.handle("/api/features/discover", {})
            assert _status(result) == 200

    def test_response_structure(self, handler):
        mock_handlers = self._mock_handlers()
        with (
            patch("aragora.server.handlers.ALL_HANDLERS", mock_handlers),
            patch(
                "aragora.server.handlers.get_all_handler_stability",
                return_value={"DebatesHandler": "stable", "HealthHandler": "stable"},
            ),
        ):
            result = handler.handle("/api/features/discover", {})
            body = _body(result)
            assert "total_endpoints" in body
            assert "frontend_integrated" in body
            assert "hidden_features" in body
            assert "integration_percentage" in body
            assert "endpoints" in body
            assert "by_category" in body
            assert "by_stability" in body
            assert "categories" in body
            assert "message" in body

    def test_endpoint_info_structure(self, handler):
        mock_handlers = self._mock_handlers()
        with (
            patch("aragora.server.handlers.ALL_HANDLERS", mock_handlers),
            patch(
                "aragora.server.handlers.get_all_handler_stability",
                return_value={"DebatesHandler": "stable"},
            ),
        ):
            result = handler.handle("/api/features/discover", {})
            body = _body(result)
            if body["endpoints"]:
                ep = body["endpoints"][0]
                assert "path" in ep
                assert "handler" in ep
                assert "category" in ep
                assert "stability" in ep
                assert "frontend_integrated" in ep
                assert "methods" in ep

    def test_frontend_integrated_flag(self, handler):
        """Endpoints known to be frontend-integrated should be flagged."""
        handler_cls = type(
            "DebatesHandler",
            (),
            {"__name__": "DebatesHandler", "ROUTES": ["/api/v1/debates"]},
        )
        with (
            patch("aragora.server.handlers.ALL_HANDLERS", [handler_cls]),
            patch(
                "aragora.server.handlers.get_all_handler_stability",
                return_value={"DebatesHandler": "stable"},
            ),
        ):
            result = handler.handle("/api/features/discover", {})
            body = _body(result)
            debates_ep = [e for e in body["endpoints"] if e["path"] == "/api/v1/debates"]
            assert len(debates_ep) == 1
            assert debates_ep[0]["frontend_integrated"] is True

    def test_non_integrated_endpoint_not_flagged(self, handler):
        handler_cls = type(
            "CustomHandler",
            (),
            {"__name__": "CustomHandler", "ROUTES": ["/api/v1/custom/obscure"]},
        )
        with (
            patch("aragora.server.handlers.ALL_HANDLERS", [handler_cls]),
            patch(
                "aragora.server.handlers.get_all_handler_stability",
                return_value={},
            ),
        ):
            result = handler.handle("/api/features/discover", {})
            body = _body(result)
            custom_ep = [e for e in body["endpoints"] if e["path"] == "/api/v1/custom/obscure"]
            assert len(custom_ep) == 1
            assert custom_ep[0]["frontend_integrated"] is False

    def test_parameterized_routes_excluded(self, handler):
        """Routes with {param} should be skipped in the catalog."""
        handler_cls = type(
            "TestHandler",
            (),
            {
                "__name__": "TestHandler",
                "ROUTES": ["/api/v1/items", "/api/v1/items/{id}"],
            },
        )
        with (
            patch("aragora.server.handlers.ALL_HANDLERS", [handler_cls]),
            patch(
                "aragora.server.handlers.get_all_handler_stability",
                return_value={},
            ),
        ):
            result = handler.handle("/api/features/discover", {})
            body = _body(result)
            paths = [e["path"] for e in body["endpoints"]]
            assert "/api/v1/items" in paths
            assert "/api/v1/items/{id}" not in paths

    def test_empty_handlers_list(self, handler):
        with (
            patch("aragora.server.handlers.ALL_HANDLERS", []),
            patch(
                "aragora.server.handlers.get_all_handler_stability",
                return_value={},
            ),
        ):
            result = handler.handle("/api/features/discover", {})
            body = _body(result)
            assert body["total_endpoints"] == 0
            assert body["integration_percentage"] == 0

    def test_dict_routes_handled(self, handler):
        """Handlers with dict-style ROUTES should also work."""
        handler_cls = type(
            "DictHandler",
            (),
            {
                "__name__": "DictHandler",
                "ROUTES": {"/api/v1/dict/route": "handle_route"},
            },
        )
        with (
            patch("aragora.server.handlers.ALL_HANDLERS", [handler_cls]),
            patch(
                "aragora.server.handlers.get_all_handler_stability",
                return_value={},
            ),
        ):
            result = handler.handle("/api/features/discover", {})
            body = _body(result)
            paths = [e["path"] for e in body["endpoints"]]
            assert "/api/v1/dict/route" in paths

    def test_versioned_path(self, handler):
        with (
            patch("aragora.server.handlers.ALL_HANDLERS", []),
            patch(
                "aragora.server.handlers.get_all_handler_stability",
                return_value={},
            ),
        ):
            result = handler.handle("/api/v1/features/discover", {})
            assert _status(result) == 200


# ---------------------------------------------------------------------------
# GET /api/features/endpoints
# ---------------------------------------------------------------------------


class TestGetAllEndpoints:
    """Tests for the flat endpoint list."""

    def test_returns_200(self, handler):
        with patch("aragora.server.handlers._registry.ALL_HANDLERS", []):
            result = handler.handle("/api/features/endpoints", {})
            assert _status(result) == 200

    def test_response_structure(self, handler):
        handler_cls = type(
            "TestHandler",
            (),
            {"__name__": "TestHandler", "ROUTES": ["/api/v1/test"]},
        )
        with patch("aragora.server.handlers._registry.ALL_HANDLERS", [handler_cls]):
            result = handler.handle("/api/features/endpoints", {})
            body = _body(result)
            assert "total" in body
            assert "handlers" in body
            assert "endpoints" in body
            assert "by_handler" in body

    def test_endpoints_sorted(self, handler):
        handler_cls = type(
            "TestHandler",
            (),
            {"__name__": "TestHandler", "ROUTES": ["/api/v1/z", "/api/v1/a"]},
        )
        with patch("aragora.server.handlers._registry.ALL_HANDLERS", [handler_cls]):
            result = handler.handle("/api/features/endpoints", {})
            body = _body(result)
            assert body["endpoints"] == sorted(body["endpoints"])

    def test_parameterized_routes_excluded(self, handler):
        handler_cls = type(
            "TestHandler",
            (),
            {
                "__name__": "TestHandler",
                "ROUTES": ["/api/v1/items", "/api/v1/items/{id}"],
            },
        )
        with patch("aragora.server.handlers._registry.ALL_HANDLERS", [handler_cls]):
            result = handler.handle("/api/features/endpoints", {})
            body = _body(result)
            assert "/api/v1/items" in body["endpoints"]
            assert "/api/v1/items/{id}" not in body["endpoints"]

    def test_by_handler_groups(self, handler):
        handler_cls = type(
            "FooHandler",
            (),
            {"__name__": "FooHandler", "ROUTES": ["/api/v1/foo"]},
        )
        with patch("aragora.server.handlers._registry.ALL_HANDLERS", [handler_cls]):
            result = handler.handle("/api/features/endpoints", {})
            body = _body(result)
            assert "FooHandler" in body["by_handler"]
            assert "/api/v1/foo" in body["by_handler"]["FooHandler"]

    def test_handler_without_routes_attr(self, handler):
        """Handlers without ROUTES attribute should not break the endpoint."""
        handler_cls = type(
            "NoRoutesHandler",
            (),
            {"__name__": "NoRoutesHandler"},
        )
        with patch("aragora.server.handlers._registry.ALL_HANDLERS", [handler_cls]):
            result = handler.handle("/api/features/endpoints", {})
            assert _status(result) == 200
            body = _body(result)
            assert body["by_handler"]["NoRoutesHandler"] == []

    def test_versioned_path(self, handler):
        with patch("aragora.server.handlers._registry.ALL_HANDLERS", []):
            result = handler.handle("/api/v1/features/endpoints", {})
            assert _status(result) == 200


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for rate limiting on features endpoints."""

    def test_rate_limit_exceeded_returns_429(self, handler, monkeypatch):
        """When rate limit is exceeded, handler returns 429."""
        # Re-enable rate limiting for this test
        import importlib

        rl_mod = importlib.import_module("aragora.server.handlers.utils.rate_limit")
        monkeypatch.setattr(rl_mod, "RATE_LIMITING_DISABLED", False)
        from aragora.server.handlers.features.features import _features_limiter

        # Patch is_allowed to return False
        _features_limiter.is_allowed = lambda key: False
        try:
            mock_handler = MockHTTPHandler()
            result = handler.handle("/api/features", {}, mock_handler)
            assert _status(result) == 429
        finally:
            # Restore
            from aragora.server.handlers.utils.rate_limit import RateLimiter

            _features_limiter.is_allowed = RateLimiter.is_allowed.__get__(
                _features_limiter, RateLimiter
            )

    def test_handler_none_returns_unknown_ip(self, handler, monkeypatch):
        """When handler is None, get_client_ip returns 'unknown' but still works."""
        result = handler.handle("/api/features", {}, None)
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# Feature Detection Helpers (unit tests for module-level functions)
# ---------------------------------------------------------------------------


class TestCheckFeatureAvailable:
    """Tests for _check_feature_available()."""

    def test_unknown_feature_returns_false(self):
        available, reason = _check_feature_available("totally_fake_feature")
        assert available is False
        assert "Unknown feature" in reason

    def test_coming_soon_feature(self):
        """Features with status 'coming_soon' should not be available."""
        # Temporarily add a coming_soon feature
        FEATURE_REGISTRY["_test_coming_soon"] = FeatureInfo(
            name="Test",
            description="test",
            requires=[],
            endpoints=[],
            status="coming_soon",
        )
        try:
            available, reason = _check_feature_available("_test_coming_soon")
            assert available is False
            assert "coming soon" in reason
        finally:
            del FEATURE_REGISTRY["_test_coming_soon"]

    def test_deprecated_feature(self):
        """Features with status 'deprecated' should not be available."""
        FEATURE_REGISTRY["_test_deprecated"] = FeatureInfo(
            name="Test",
            description="test",
            requires=[],
            endpoints=[],
            status="deprecated",
        )
        try:
            available, reason = _check_feature_available("_test_deprecated")
            assert available is False
            assert "deprecated" in reason
        finally:
            del FEATURE_REGISTRY["_test_deprecated"]

    def test_feature_with_no_requirements(self):
        """Feature with empty requires list should be available."""
        FEATURE_REGISTRY["_test_no_reqs"] = FeatureInfo(
            name="Test",
            description="test",
            requires=[],
            endpoints=[],
        )
        try:
            available, reason = _check_feature_available("_test_no_reqs")
            assert available is True
            assert reason is None
        finally:
            del FEATURE_REGISTRY["_test_no_reqs"]

    def test_feature_with_unknown_requirement_assumed_available(self):
        """Unknown requirements are assumed available."""
        FEATURE_REGISTRY["_test_unknown_req"] = FeatureInfo(
            name="Test",
            description="test",
            requires=["completely_unknown_requirement_xyz"],
            endpoints=[],
        )
        try:
            available, reason = _check_feature_available("_test_unknown_req")
            assert available is True
        finally:
            del FEATURE_REGISTRY["_test_unknown_req"]

    def test_requirement_check_exception_handled(self):
        """If a requirement check raises, it should be caught gracefully."""
        with patch(
            "aragora.server.handlers.features.features._check_pulse",
            side_effect=ImportError("No module"),
        ):
            available, reason = _check_feature_available("pulse")
            assert available is False
            assert reason == "Check failed"


class TestGetAllFeaturesUnit:
    """Tests for get_all_features()."""

    def test_returns_dict(self):
        result = get_all_features()
        assert isinstance(result, dict)

    def test_all_registry_features_present(self):
        result = get_all_features()
        for feature_id in FEATURE_REGISTRY:
            assert feature_id in result

    def test_feature_entry_has_expected_keys(self):
        result = get_all_features()
        for feature_id, info in result.items():
            assert "id" in info
            assert "name" in info
            assert "description" in info
            assert "category" in info
            assert "status" in info
            assert "available" in info
            assert "endpoints" in info

    def test_unavailable_feature_has_install_hint(self):
        """Unavailable features should have install_hint populated."""
        with patch(
            "aragora.server.handlers.features.features._check_feature_available",
            return_value=(False, "Not ready"),
        ):
            result = get_all_features()
            for feature_id, info in result.items():
                assert info["install_hint"] is not None


class TestGetAvailableFeatures:
    """Tests for get_available_features()."""

    def test_returns_list(self):
        result = get_available_features()
        assert isinstance(result, list)

    def test_only_available_features_returned(self):
        """All returned features should actually be available."""
        result = get_available_features()
        for feature_id in result:
            available, _ = _check_feature_available(feature_id)
            assert available is True


class TestGetUnavailableFeatures:
    """Tests for get_unavailable_features()."""

    def test_returns_dict(self):
        result = get_unavailable_features()
        assert isinstance(result, dict)

    def test_all_entries_have_reason(self):
        result = get_unavailable_features()
        for feature_id, reason in result.items():
            assert isinstance(reason, str)
            assert len(reason) > 0


# ---------------------------------------------------------------------------
# feature_unavailable_response() helper
# ---------------------------------------------------------------------------


class TestFeatureUnavailableResponse:
    """Tests for the feature_unavailable_response helper."""

    def test_known_feature_returns_503(self):
        result = feature_unavailable_response("pulse")
        assert _status(result) == 503

    def test_known_feature_has_install_hint(self):
        result = feature_unavailable_response("pulse")
        body = _body(result)
        error_obj = body.get("error", body)
        assert "suggestion" in error_obj or "install_hint" in error_obj or "details" in error_obj

    def test_unknown_feature_returns_503(self):
        result = feature_unavailable_response("totally_unknown_feature")
        assert _status(result) == 503

    def test_custom_message(self):
        result = feature_unavailable_response("pulse", message="Custom error msg")
        body = _body(result)
        # The custom message should appear somewhere in the response
        body_str = json.dumps(body)
        assert "Custom error msg" in body_str

    def test_error_code_set(self):
        result = feature_unavailable_response("elo")
        body = _body(result)
        error_obj = body.get("error", body)
        assert error_obj.get("code") == "FEATURE_UNAVAILABLE"


# ---------------------------------------------------------------------------
# FeatureInfo dataclass
# ---------------------------------------------------------------------------


class TestFeatureInfo:
    """Tests for the FeatureInfo dataclass."""

    def test_defaults(self):
        info = FeatureInfo(
            name="Test",
            description="A test feature",
            requires=["test_module"],
            endpoints=["/api/test"],
        )
        assert info.status == "optional"
        assert info.category == "general"
        assert info.install_hint == ""

    def test_custom_values(self):
        info = FeatureInfo(
            name="Custom",
            description="A custom feature",
            requires=["a", "b"],
            endpoints=["/api/custom"],
            install_hint="pip install custom",
            status="coming_soon",
            category="advanced",
        )
        assert info.name == "Custom"
        assert info.requires == ["a", "b"]
        assert info.status == "coming_soon"
        assert info.category == "advanced"


# ---------------------------------------------------------------------------
# FEATURE_REGISTRY integrity
# ---------------------------------------------------------------------------


class TestFeatureRegistry:
    """Tests for the feature registry data integrity."""

    def test_all_features_have_names(self):
        for fid, info in FEATURE_REGISTRY.items():
            assert info.name, f"Feature {fid} has no name"

    def test_all_features_have_descriptions(self):
        for fid, info in FEATURE_REGISTRY.items():
            assert info.description, f"Feature {fid} has no description"

    def test_all_features_have_requires(self):
        for fid, info in FEATURE_REGISTRY.items():
            assert isinstance(info.requires, list), f"Feature {fid} requires is not a list"

    def test_all_features_have_endpoints(self):
        for fid, info in FEATURE_REGISTRY.items():
            assert isinstance(info.endpoints, list), f"Feature {fid} endpoints is not a list"

    def test_all_categories_are_strings(self):
        for fid, info in FEATURE_REGISTRY.items():
            assert isinstance(info.category, str), f"Feature {fid} category is not a string"

    def test_all_statuses_valid(self):
        valid_statuses = {"optional", "coming_soon", "deprecated"}
        for fid, info in FEATURE_REGISTRY.items():
            assert info.status in valid_statuses, f"Feature {fid} has invalid status: {info.status}"

    def test_known_feature_ids(self):
        """Verify expected feature IDs are present."""
        expected = {
            "pulse",
            "genesis",
            "verification",
            "laboratory",
            "calibration",
            "evolution",
            "red_team",
            "capability_probes",
            "continuum_memory",
            "consensus_memory",
            "insights",
            "moments",
            "tournaments",
            "elo",
            "crux",
            "rhetorical",
            "trickster",
            "plugins",
            "memory",
            "supermemory",
        }
        assert expected == set(FEATURE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# DEFAULT_PREFERENCES integrity
# ---------------------------------------------------------------------------


class TestDefaultPreferences:
    """Tests for the default preferences dictionary."""

    def test_has_feature_toggles(self):
        prefs = FeaturesHandler.DEFAULT_PREFERENCES
        assert "calibration" in prefs
        assert "trickster" in prefs
        assert "rhetorical" in prefs

    def test_has_display_preferences(self):
        prefs = FeaturesHandler.DEFAULT_PREFERENCES
        assert "show_advanced_metrics" in prefs
        assert "compact_mode" in prefs
        assert "theme" in prefs

    def test_has_debate_preferences(self):
        prefs = FeaturesHandler.DEFAULT_PREFERENCES
        assert "default_mode" in prefs
        assert "default_rounds" in prefs
        assert "default_agents" in prefs

    def test_has_notification_preferences(self):
        prefs = FeaturesHandler.DEFAULT_PREFERENCES
        assert "telegram_enabled" in prefs
        assert "email_digest" in prefs
        assert "consensus_alert_threshold" in prefs


# ---------------------------------------------------------------------------
# Path parameter extraction
# ---------------------------------------------------------------------------


class TestPathParameterExtraction:
    """Tests for extracting feature_id from path segments."""

    def test_extracts_pulse(self, handler):
        result = handler.handle("/api/features/pulse", {})
        body = _body(result)
        assert body["id"] == "pulse"

    def test_extracts_genesis(self, handler):
        result = handler.handle("/api/features/genesis", {})
        body = _body(result)
        assert body["id"] == "genesis"

    def test_extracts_from_versioned_path(self, handler):
        result = handler.handle("/api/v1/features/elo", {})
        body = _body(result)
        assert body["id"] == "elo"

    def test_handles_unknown_id_gracefully(self, handler):
        result = handler.handle("/api/features/unknown_id_123", {})
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# Handler routing (handle() dispatch)
# ---------------------------------------------------------------------------


class TestHandleRouting:
    """Tests for the handle() dispatch logic."""

    def test_returns_none_for_unhandled_path(self, handler):
        result = handler.handle("/api/unrelated", {})
        assert result is None

    def test_config_receives_handler_arg(self, handler):
        """The config endpoint should receive the HTTP handler for auth."""
        mock_handler = MockHTTPHandler(method="GET")
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=None,
        ):
            result = handler.handle("/api/features/config", {}, mock_handler)
            assert _status(result) == 200

    def test_non_config_routes_ignore_handler_arg(self, handler):
        """Non-config routes work without HTTP handler."""
        result = handler.handle("/api/features/available", {})
        assert _status(result) == 200

    def test_deeply_nested_path_returns_none(self, handler):
        """Paths with too many segments are not handled."""
        result = handler.handle("/api/features/pulse/sub/path", {})
        assert result is None


# ---------------------------------------------------------------------------
# ROUTES class attribute
# ---------------------------------------------------------------------------


class TestRoutesList:
    """Tests for the ROUTES class attribute used by SDK audit."""

    def test_routes_is_list(self):
        assert isinstance(FeaturesHandler.ROUTES, list)

    def test_routes_include_versioned_paths(self):
        versioned = [r for r in FeaturesHandler.ROUTES if "/api/v1/" in r]
        assert len(versioned) > 0

    def test_routes_include_unversioned_paths(self):
        unversioned = [r for r in FeaturesHandler.ROUTES if r.startswith("/api/features")]
        assert len(unversioned) > 0

    def test_routes_include_wildcard(self):
        wildcards = [r for r in FeaturesHandler.ROUTES if "*" in r]
        assert len(wildcards) > 0, "Should have wildcard routes for parameterized paths"

    def test_routes_count(self):
        """There should be versioned + unversioned route pairs."""
        assert len(FeaturesHandler.ROUTES) >= 14  # 7 versioned + 7 unversioned at minimum
