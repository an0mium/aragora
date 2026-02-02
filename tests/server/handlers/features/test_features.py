"""Tests for Features Handler."""

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import pytest
from unittest.mock import MagicMock, patch

from aragora.server.handlers.features.features import (
    FeaturesHandler,
    FEATURE_REGISTRY,
    FeatureInfo,
    _features_limiter,
    _check_feature_available,
    get_all_features,
    get_available_features,
    get_unavailable_features,
    feature_unavailable_response,
)


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter between tests."""
    _features_limiter._buckets.clear()
    yield


@pytest.fixture
def handler():
    """Create handler instance."""
    return FeaturesHandler(server_context={})


class TestFeatureInfo:
    """Tests for FeatureInfo dataclass."""

    def test_feature_info_creation(self):
        """Test creating FeatureInfo instance."""
        info = FeatureInfo(
            name="Test Feature",
            description="A test feature",
            requires=["test_module"],
            endpoints=["/api/test"],
        )
        assert info.name == "Test Feature"
        assert info.description == "A test feature"
        assert info.requires == ["test_module"]
        assert info.endpoints == ["/api/test"]

    def test_feature_info_defaults(self):
        """Test FeatureInfo default values."""
        info = FeatureInfo(
            name="Test",
            description="Test",
            requires=[],
            endpoints=[],
        )
        assert info.install_hint == ""
        assert info.status == "optional"
        assert info.category == "general"


class TestFeatureRegistry:
    """Tests for feature registry."""

    def test_registry_exists(self):
        """Test feature registry is defined."""
        assert FEATURE_REGISTRY is not None
        assert len(FEATURE_REGISTRY) > 0

    def test_registry_contains_known_features(self):
        """Test registry contains expected features."""
        assert "pulse" in FEATURE_REGISTRY
        assert "genesis" in FEATURE_REGISTRY
        assert "memory" in FEATURE_REGISTRY
        assert "supermemory" in FEATURE_REGISTRY

    def test_registry_feature_structure(self):
        """Test registry features have required structure."""
        for feature_id, info in FEATURE_REGISTRY.items():
            assert isinstance(info, FeatureInfo)
            assert info.name
            assert info.description
            assert isinstance(info.requires, list)
            assert isinstance(info.endpoints, list)


class TestCheckFeatureAvailable:
    """Tests for _check_feature_available function."""

    def test_unknown_feature(self):
        """Test checking unknown feature."""
        available, reason = _check_feature_available("unknown_feature")
        assert available is False
        assert "Unknown feature" in reason

    def test_coming_soon_feature(self):
        """Test checking coming soon feature."""
        with patch.dict(
            FEATURE_REGISTRY,
            {
                "test_feature": FeatureInfo(
                    name="Test",
                    description="Test",
                    requires=[],
                    endpoints=[],
                    status="coming_soon",
                )
            },
        ):
            available, reason = _check_feature_available("test_feature")
            assert available is False
            assert "coming soon" in reason.lower()

    def test_deprecated_feature(self):
        """Test checking deprecated feature."""
        with patch.dict(
            FEATURE_REGISTRY,
            {
                "test_feature": FeatureInfo(
                    name="Test",
                    description="Test",
                    requires=[],
                    endpoints=[],
                    status="deprecated",
                )
            },
        ):
            available, reason = _check_feature_available("test_feature")
            assert available is False
            assert "deprecated" in reason.lower()


class TestGetAllFeatures:
    """Tests for get_all_features function."""

    def test_get_all_features(self):
        """Test getting all features."""
        with patch(
            "aragora.server.handlers.features.features._check_feature_available",
            return_value=(True, None),
        ):
            features = get_all_features()
            assert len(features) > 0

            for feature_id, info in features.items():
                assert "id" in info
                assert "name" in info
                assert "available" in info


class TestGetAvailableFeatures:
    """Tests for get_available_features function."""

    def test_get_available_features(self):
        """Test getting available features."""
        features = get_available_features()
        assert isinstance(features, list)


class TestGetUnavailableFeatures:
    """Tests for get_unavailable_features function."""

    def test_get_unavailable_features(self):
        """Test getting unavailable features."""
        features = get_unavailable_features()
        assert isinstance(features, dict)


class TestFeatureUnavailableResponse:
    """Tests for feature_unavailable_response function."""

    def test_response_known_feature(self):
        """Test response for known feature."""
        response = feature_unavailable_response("pulse", "Pulse not available")
        assert response.status_code == 503

    def test_response_unknown_feature(self):
        """Test response for unknown feature."""
        response = feature_unavailable_response("unknown")
        assert response.status_code == 503


class TestFeaturesHandler:
    """Tests for FeaturesHandler class."""

    def test_handler_creation(self, handler):
        """Test creating handler instance."""
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(FeaturesHandler, "ROUTES")
        routes = FeaturesHandler.ROUTES
        assert "/api/features" in routes
        assert "/api/features/available" in routes
        assert "/api/features/all" in routes

    def test_can_handle_features_routes(self, handler):
        """Test can_handle for features routes."""
        assert handler.can_handle("/api/features") is True
        assert handler.can_handle("/api/features/available") is True
        assert handler.can_handle("/api/features/all") is True

    def test_can_handle_specific_feature(self, handler):
        """Test can_handle for specific feature."""
        assert handler.can_handle("/api/features/pulse") is True
        assert handler.can_handle("/api/features/genesis") is True

    def test_can_handle_invalid_routes(self, handler):
        """Test can_handle rejects invalid routes."""
        assert handler.can_handle("/api/capabilities/") is False


class TestFeaturesEndpoints:
    """Tests for features handler endpoints."""

    def test_get_features_summary(self, handler):
        """Test getting features summary."""
        result = handler._get_features_summary()
        assert result.status_code == 200

        import json

        body = json.loads(result.body)
        assert "available_count" in body
        assert "unavailable_count" in body
        assert "available" in body

    def test_get_available(self, handler):
        """Test getting available features list."""
        result = handler._get_available()
        assert result.status_code == 200

        import json

        body = json.loads(result.body)
        assert "features" in body
        assert "count" in body

    def test_get_all_features(self, handler):
        """Test getting all features."""
        result = handler._get_all_features()
        assert result.status_code == 200

        import json

        body = json.loads(result.body)
        assert "features" in body
        assert "by_category" in body
        assert "total" in body

    def test_get_feature_status_known(self, handler):
        """Test getting status of known feature."""
        result = handler._get_feature_status("pulse")
        assert result.status_code == 200

        import json

        body = json.loads(result.body)
        assert body["id"] == "pulse"
        assert "available" in body

    def test_get_feature_status_unknown(self, handler):
        """Test getting status of unknown feature."""
        result = handler._get_feature_status("unknown_feature")
        assert result.status_code == 404


class TestFeaturesRateLimiting:
    """Tests for features rate limiting."""

    def test_rate_limiter_exists(self):
        """Test that rate limiter is configured."""
        assert _features_limiter is not None
        assert _features_limiter.rpm == 100

    def test_rate_limit_exceeded(self, handler):
        """Test rate limit enforcement."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        # Exhaust rate limit
        for _ in range(101):
            _features_limiter.is_allowed("127.0.0.1")

        with patch(
            "aragora.server.handlers.features.features.get_client_ip",
            return_value="127.0.0.1",
        ):
            result = handler.handle("/api/features", {}, mock_handler)
            assert result.status_code == 429
