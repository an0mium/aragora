"""
Tests for aragora.server.handlers.feature_flags - Public feature flags handler.

Tests cover:
- Instantiation and ROUTES
- can_handle() route matching with version prefix stripping
- GET /api/v1/feature-flags - list all flags
- GET /api/v1/feature-flags/:name - get specific flag
- Method not allowed (non-GET)
- Flags system unavailable (503)
- Category filter validation
- Flag not found (404)
- Empty flag name returns 400
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.feature_flags import FeatureFlagsHandler


# ===========================================================================
# Mock Flag System
# ===========================================================================


class MockFlagCategory(Enum):
    CORE = "core"
    KNOWLEDGE = "knowledge"
    EXPERIMENTAL = "experimental"


class MockFlagStatus(Enum):
    ACTIVE = "active"
    BETA = "beta"
    DEPRECATED = "deprecated"


class MockFlagDefinition:
    """Mock a feature flag definition."""

    def __init__(
        self,
        name: str,
        default: Any = False,
        description: str = "",
        category: MockFlagCategory = MockFlagCategory.CORE,
        status: MockFlagStatus = MockFlagStatus.ACTIVE,
    ):
        self.name = name
        self.default = default
        self.description = description
        self.category = category
        self.status = status
        self.flag_type = type(default)
        self.env_var = f"ARAGORA_{name.upper()}"
        self.deprecated_since = None
        self.removed_in = None
        self.replacement = None


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def handler():
    """Create a FeatureFlagsHandler with mocked context."""
    ctx: dict[str, Any] = {}
    return FeatureFlagsHandler(ctx)


@pytest.fixture
def mock_get():
    """Create a mock HTTP GET handler."""
    mock = MagicMock()
    mock.command = "GET"
    mock.headers = {}
    return mock


@pytest.fixture
def sample_flags():
    """Create sample flag definitions."""
    return [
        MockFlagDefinition(
            name="enable_knowledge_mound",
            default=True,
            description="Enable knowledge mound integration",
            category=MockFlagCategory.KNOWLEDGE,
            status=MockFlagStatus.ACTIVE,
        ),
        MockFlagDefinition(
            name="enable_trickster",
            default=False,
            description="Enable trickster agent hollow consensus detection",
            category=MockFlagCategory.CORE,
            status=MockFlagStatus.BETA,
        ),
    ]


@pytest.fixture
def mock_registry(sample_flags):
    """Create a mock flag registry."""
    registry = MagicMock()
    registry.get_all_flags.return_value = sample_flags
    registry.get_value.side_effect = lambda name, default: default
    registry.get_definition.side_effect = lambda name: next(
        (f for f in sample_flags if f.name == name), None
    )
    return registry


# ===========================================================================
# Instantiation and Routes
# ===========================================================================


class TestSetup:
    """Tests for handler instantiation and route registration."""

    def test_instantiation(self, handler):
        """Should create handler with context."""
        assert handler is not None

    def test_routes_defined(self):
        """Should define expected ROUTES."""
        assert "/api/v1/feature-flags" in FeatureFlagsHandler.ROUTES
        assert "/api/v1/feature-flags/*" in FeatureFlagsHandler.ROUTES

    def test_routes_count(self):
        """Should have exactly 2 routes."""
        assert len(FeatureFlagsHandler.ROUTES) == 2


# ===========================================================================
# can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle route matching."""

    def test_can_handle_flags_list(self, handler):
        """Should handle /api/v1/feature-flags."""
        assert handler.can_handle("/api/v1/feature-flags") is True

    def test_can_handle_flag_by_name(self, handler):
        """Should handle /api/v1/feature-flags/:name."""
        assert handler.can_handle("/api/v1/feature-flags/enable_trickster") is True
        assert handler.can_handle("/api/v1/feature-flags/some_flag") is True

    def test_cannot_handle_unknown(self, handler):
        """Should not handle unknown paths."""
        assert handler.can_handle("/api/v1/flags") is False
        assert handler.can_handle("/api/v1/admin/feature-flags") is False
        assert handler.can_handle("/api/v1/debates") is False


# ===========================================================================
# Method Not Allowed
# ===========================================================================


class TestMethodNotAllowed:
    """Tests for non-GET method rejection."""

    def test_post_returns_405(self, handler):
        """Should reject POST with 405."""
        mock = MagicMock()
        mock.command = "POST"
        with patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True):
            result = handler.handle("/api/v1/feature-flags", {}, mock)
        assert result.status_code == 405

    def test_put_returns_405(self, handler):
        """Should reject PUT with 405."""
        mock = MagicMock()
        mock.command = "PUT"
        with patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True):
            result = handler.handle("/api/v1/feature-flags", {}, mock)
        assert result.status_code == 405


# ===========================================================================
# Flags System Unavailable
# ===========================================================================


class TestFlagsUnavailable:
    """Tests for when flag system is not imported."""

    def test_flags_unavailable_returns_503(self, handler, mock_get):
        """Should return 503 when feature flag system not available."""
        with patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", False):
            result = handler.handle("/api/v1/feature-flags", {}, mock_get)
        assert result.status_code == 503

    def test_flags_unavailable_on_specific_flag(self, handler, mock_get):
        """Should return 503 for specific flag when system unavailable."""
        with patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", False):
            result = handler.handle(
                "/api/v1/feature-flags/enable_trickster", {}, mock_get
            )
        assert result.status_code == 503


# ===========================================================================
# GET /api/v1/feature-flags - List All Flags
# ===========================================================================


class TestListFlags:
    """Tests for GET /api/v1/feature-flags."""

    def test_list_flags_success(self, handler, mock_get, mock_registry):
        """Should return list of feature flags."""
        with patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.feature_flags.get_flag_registry",
                return_value=mock_registry,
            ):
                with patch(
                    "aragora.server.handlers.feature_flags.FlagCategory",
                    MockFlagCategory,
                ):
                    result = handler.handle("/api/v1/feature-flags", {}, mock_get)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total"] == 2
        assert len(data["flags"]) == 2

        flag_names = [f["name"] for f in data["flags"]]
        assert "enable_knowledge_mound" in flag_names
        assert "enable_trickster" in flag_names

    def test_list_flags_with_category_filter(self, handler, mock_get, mock_registry):
        """Should filter flags by category."""
        with patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.feature_flags.get_flag_registry",
                return_value=mock_registry,
            ):
                with patch(
                    "aragora.server.handlers.feature_flags.FlagCategory",
                    MockFlagCategory,
                ):
                    result = handler.handle(
                        "/api/v1/feature-flags",
                        {"category": "knowledge"},
                        mock_get,
                    )

        assert result.status_code == 200
        mock_registry.get_all_flags.assert_called_once_with(
            category=MockFlagCategory.KNOWLEDGE
        )

    def test_list_flags_invalid_category_returns_400(
        self, handler, mock_get, mock_registry
    ):
        """Should return 400 for invalid category filter."""
        with patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.feature_flags.get_flag_registry",
                return_value=mock_registry,
            ):
                with patch(
                    "aragora.server.handlers.feature_flags.FlagCategory",
                    MockFlagCategory,
                ):
                    result = handler.handle(
                        "/api/v1/feature-flags",
                        {"category": "nonexistent"},
                        mock_get,
                    )

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "Invalid category" in data.get("error", "")

    def test_list_includes_flag_details(self, handler, mock_get, mock_registry):
        """Should include description, category, status in response."""
        with patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.feature_flags.get_flag_registry",
                return_value=mock_registry,
            ):
                with patch(
                    "aragora.server.handlers.feature_flags.FlagCategory",
                    MockFlagCategory,
                ):
                    result = handler.handle("/api/v1/feature-flags", {}, mock_get)

        data = json.loads(result.body)
        flag = data["flags"][0]
        assert "name" in flag
        assert "value" in flag
        assert "description" in flag
        assert "category" in flag
        assert "status" in flag


# ===========================================================================
# GET /api/v1/feature-flags/:name - Get Specific Flag
# ===========================================================================


class TestGetFlag:
    """Tests for GET /api/v1/feature-flags/:name."""

    def test_get_flag_success(self, handler, mock_get, mock_registry):
        """Should return specific flag details."""
        with patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.feature_flags.get_flag_registry",
                return_value=mock_registry,
            ):
                result = handler.handle(
                    "/api/v1/feature-flags/enable_trickster", {}, mock_get
                )

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["name"] == "enable_trickster"
        assert data["value"] is False
        assert data["status"] == "beta"

    def test_get_flag_not_found(self, handler, mock_get, mock_registry):
        """Should return 404 for nonexistent flag."""
        with patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.feature_flags.get_flag_registry",
                return_value=mock_registry,
            ):
                result = handler.handle(
                    "/api/v1/feature-flags/nonexistent", {}, mock_get
                )

        assert result.status_code == 404
        data = json.loads(result.body)
        assert "not found" in data.get("error", "").lower()

    def test_empty_flag_name_returns_400(self, handler, mock_get):
        """Should return 400 when flag name is empty."""
        with patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.feature_flags.get_flag_registry",
                return_value=MagicMock(),
            ):
                result = handler.handle(
                    "/api/v1/feature-flags/", {}, mock_get
                )

        # The path /api/v1/feature-flags/ with empty name should return 400
        assert result.status_code == 400
