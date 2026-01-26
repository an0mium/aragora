"""
Tests for FeaturesHandler - feature availability and discovery.

Tests cover:
- GET /api/features - Feature summary
- GET /api/features/available - List of available features
- GET /api/features/all - Full feature matrix
- GET /api/features/{feature_id} - Individual feature status
- Feature detection logic for all registered features
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from aragora.server.handlers.features import (
    FEATURE_REGISTRY,
    FeatureInfo,
    FeaturesHandler,
    feature_unavailable_response,
    get_all_features,
    get_available_features,
    get_unavailable_features,
    _check_feature_available,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def features_handler():
    """Create FeaturesHandler instance."""
    return FeaturesHandler(server_context={})


# ============================================================================
# FeatureInfo Tests
# ============================================================================


class TestFeatureInfo:
    """Tests for FeatureInfo dataclass."""

    def test_feature_info_creation(self):
        """Test creating a FeatureInfo instance."""
        info = FeatureInfo(
            name="Test Feature",
            description="A test feature",
            requires=["test_module"],
            endpoints=["/api/test"],
            install_hint="pip install test",
            status="optional",
            category="testing",
        )

        assert info.name == "Test Feature"
        assert info.description == "A test feature"
        assert info.requires == ["test_module"]
        assert info.endpoints == ["/api/test"]
        assert info.install_hint == "pip install test"
        assert info.status == "optional"
        assert info.category == "testing"

    def test_feature_info_defaults(self):
        """Test FeatureInfo default values."""
        info = FeatureInfo(
            name="Minimal Feature",
            description="Minimal",
            requires=[],
            endpoints=[],
        )

        assert info.install_hint == ""
        assert info.status == "optional"
        assert info.category == "general"


# ============================================================================
# Feature Registry Tests
# ============================================================================


class TestFeatureRegistry:
    """Tests for FEATURE_REGISTRY configuration."""

    def test_registry_has_required_features(self):
        """Test that essential features are registered."""
        required = ["pulse", "genesis", "elo", "continuum_memory"]
        for feature_id in required:
            assert feature_id in FEATURE_REGISTRY, f"Missing feature: {feature_id}"

    def test_all_features_have_required_fields(self):
        """Test that all registered features have required fields."""
        for feature_id, feature in FEATURE_REGISTRY.items():
            assert feature.name, f"{feature_id} missing name"
            assert feature.description, f"{feature_id} missing description"
            assert isinstance(feature.requires, list), f"{feature_id} requires should be list"
            assert isinstance(feature.endpoints, list), f"{feature_id} endpoints should be list"
            assert feature.category, f"{feature_id} missing category"

    def test_features_have_valid_status(self):
        """Test that all features have valid status."""
        valid_statuses = {"optional", "coming_soon", "deprecated"}
        for feature_id, feature in FEATURE_REGISTRY.items():
            assert feature.status in valid_statuses, (
                f"{feature_id} has invalid status: {feature.status}"
            )

    def test_categories_are_consistent(self):
        """Test that category names are consistent."""
        expected_categories = {
            "discovery",
            "evolution",
            "analysis",
            "memory",
            "competition",
            "security",
            "system",
        }
        for feature_id, feature in FEATURE_REGISTRY.items():
            assert feature.category in expected_categories, (
                f"{feature_id} has unexpected category: {feature.category}"
            )


# ============================================================================
# Feature Detection Tests
# ============================================================================


class TestFeatureDetection:
    """Tests for feature availability detection."""

    def test_unknown_feature_returns_false(self):
        """Test that unknown features are reported as unavailable."""
        available, reason = _check_feature_available("nonexistent_feature")
        assert not available
        assert "Unknown feature" in reason

    def test_coming_soon_features_unavailable(self):
        """Test that coming_soon features are unavailable."""
        # Find a coming_soon feature
        coming_soon = [fid for fid, f in FEATURE_REGISTRY.items() if f.status == "coming_soon"]
        if coming_soon:
            available, reason = _check_feature_available(coming_soon[0])
            assert not available
            assert "coming soon" in reason.lower()

    @patch("aragora.server.handlers.features.features._check_pulse")
    def test_pulse_availability_check(self, mock_check):
        """Test pulse feature detection."""
        mock_check.return_value = (True, None)
        available, reason = _check_feature_available("pulse")
        assert available
        assert reason is None

    @patch("aragora.server.handlers.features.features._check_pulse")
    def test_pulse_unavailable_with_reason(self, mock_check):
        """Test pulse feature reports correct reason when unavailable."""
        mock_check.return_value = (False, "API keys not configured")
        available, reason = _check_feature_available("pulse")
        assert not available
        assert "API keys" in reason

    @patch("aragora.server.handlers.features.features._check_genesis")
    def test_genesis_availability_check(self, mock_check):
        """Test genesis feature detection."""
        mock_check.return_value = (True, None)
        available, reason = _check_feature_available("genesis")
        assert available

    @patch("aragora.server.handlers.features.features._check_elo")
    def test_elo_availability_check(self, mock_check):
        """Test ELO feature detection."""
        mock_check.return_value = (True, None)
        available, reason = _check_feature_available("elo")
        assert available


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestHelperFunctions:
    """Tests for feature helper functions."""

    @patch("aragora.server.handlers.features.features._check_feature_available")
    def test_get_all_features_structure(self, mock_check):
        """Test get_all_features returns correct structure."""
        mock_check.return_value = (True, None)
        features = get_all_features()

        assert isinstance(features, dict)
        for feature_id, info in features.items():
            assert "id" in info
            assert "name" in info
            assert "description" in info
            assert "category" in info
            assert "status" in info
            assert "available" in info
            assert "endpoints" in info

    @patch("aragora.server.handlers.features.features._check_feature_available")
    def test_get_available_features_returns_list(self, mock_check):
        """Test get_available_features returns list of IDs."""
        mock_check.return_value = (True, None)
        available = get_available_features()

        assert isinstance(available, list)
        assert all(isinstance(fid, str) for fid in available)

    @patch("aragora.server.handlers.features.features._check_feature_available")
    def test_get_unavailable_features_returns_dict(self, mock_check):
        """Test get_unavailable_features returns dict with reasons."""
        mock_check.return_value = (False, "Test reason")
        unavailable = get_unavailable_features()

        assert isinstance(unavailable, dict)
        for feature_id, reason in unavailable.items():
            assert isinstance(feature_id, str)
            assert isinstance(reason, str)


# ============================================================================
# FeaturesHandler Tests
# ============================================================================


class TestFeaturesHandler:
    """Tests for FeaturesHandler endpoints."""

    def test_get_features_summary(self, features_handler):
        """Test /api/features returns summary."""
        result = features_handler._get_features_summary()

        assert result.status_code == 200
        data = json.loads(result.body)

        assert "available_count" in data
        assert "unavailable_count" in data
        assert "available" in data
        assert "unavailable" in data
        assert "categories" in data

    def test_get_available_features(self, features_handler):
        """Test /api/features/available returns list."""
        result = features_handler._get_available()

        assert result.status_code == 200
        data = json.loads(result.body)

        assert "features" in data
        assert "count" in data
        assert isinstance(data["features"], list)
        assert data["count"] == len(data["features"])

    def test_get_all_features(self, features_handler):
        """Test /api/features/all returns full matrix."""
        result = features_handler._get_all_features()

        assert result.status_code == 200
        data = json.loads(result.body)

        assert "features" in data
        assert "by_category" in data
        assert "total" in data
        assert isinstance(data["features"], dict)
        assert isinstance(data["by_category"], dict)

    def test_get_known_feature_status(self, features_handler):
        """Test /api/features/{feature_id} for known feature."""
        result = features_handler._get_feature_status("elo")

        assert result.status_code == 200
        data = json.loads(result.body)

        assert data["id"] == "elo"
        assert data["name"] == "ELO Rankings"
        assert "available" in data
        assert "description" in data
        assert "endpoints" in data
        assert "requires" in data

    def test_get_unknown_feature_status(self, features_handler):
        """Test /api/features/{feature_id} for unknown feature."""
        result = features_handler._get_feature_status("nonexistent_feature")

        assert result.status_code == 404
        data = json.loads(result.body)
        assert "error" in data

    def test_get_categories(self, features_handler):
        """Test _get_categories helper."""
        categories = features_handler._get_categories()

        assert isinstance(categories, dict)
        # Should have at least some categories
        assert len(categories) > 0
        # Values should be counts
        assert all(isinstance(v, int) for v in categories.values())


# ============================================================================
# Error Response Tests
# ============================================================================


class TestFeatureUnavailableResponse:
    """Tests for feature_unavailable_response helper."""

    def test_known_feature_response(self):
        """Test response for known unavailable feature."""
        result = feature_unavailable_response("pulse")

        assert result.status_code == 503
        data = json.loads(result.body)

        assert "error" in data
        # error is now a structured dict with message and details
        assert "Trending Topics" in data["error"]["message"]

    def test_known_feature_with_custom_message(self):
        """Test response with custom message."""
        result = feature_unavailable_response("pulse", "Custom message")

        assert result.status_code == 503
        data = json.loads(result.body)
        # error is now a structured dict with message and details
        assert data["error"]["message"] == "Custom message"

    def test_unknown_feature_response(self):
        """Test response for unknown feature."""
        result = feature_unavailable_response("nonexistent")

        assert result.status_code == 503
        data = json.loads(result.body)
        # error is now a structured dict with message and details
        assert "nonexistent" in data["error"]["message"]


# ============================================================================
# Integration Tests
# ============================================================================


class TestFeaturesIntegration:
    """Integration tests for feature system."""

    def test_feature_counts_consistent(self, features_handler):
        """Test that feature counts are consistent across endpoints."""
        summary = json.loads(features_handler._get_features_summary().body)
        all_features = json.loads(features_handler._get_all_features().body)

        total_from_summary = summary["available_count"] + summary["unavailable_count"]
        total_from_all = all_features["total"]

        assert total_from_summary == total_from_all

    def test_available_subset_of_all(self, features_handler):
        """Test that available features are subset of all features."""
        available = json.loads(features_handler._get_available().body)
        all_features = json.loads(features_handler._get_all_features().body)

        available_ids = set(available["features"])
        all_ids = set(all_features["features"].keys())

        assert available_ids.issubset(all_ids)

    def test_categories_sum_to_total(self, features_handler):
        """Test that category counts sum to total features."""
        summary = json.loads(features_handler._get_features_summary().body)

        total_in_categories = sum(summary["categories"].values())
        total_features = len(FEATURE_REGISTRY)

        assert total_in_categories == total_features
