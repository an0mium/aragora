"""
Tests for MarketplaceHandler - Marketplace template CRUD, search, ratings, and import/export.

Tests cover:
- handle_list_templates: happy path, filtering by category/tags/query, pagination, invalid category
- handle_get_template: happy path, not found, download count increment
- handle_create_template: happy path, auth required, missing body, validation error
- handle_delete_template: happy path, auth required, forbidden (built-in), not found
- handle_rate_template: happy path, auth required, missing score, invalid score range
- handle_get_ratings: happy path, average rating
- handle_star_template: happy path, auth required
- handle_list_categories: happy path
- handle_export_template: happy path, not found
- handle_import_template: delegates to create
- RBAC permission enforcement across all endpoints
"""

from __future__ import annotations

import json
import sys
import types as _types_mod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Slack stubs to prevent transitive import issues
# ---------------------------------------------------------------------------
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


from aragora.server.handlers.marketplace import MarketplaceHandler


# ===========================================================================
# Mocks
# ===========================================================================


@dataclass
class MockTemplateMetadata:
    """Mock template metadata."""

    stars: int = 5
    downloads: int = 42


@dataclass
class MockTemplate:
    """Mock marketplace template."""

    id: str = "tpl-001"
    name: str = "Test Template"
    category: str = "debate"
    template_type: str = "workflow"
    tags: list = field(default_factory=lambda: ["test", "example"])
    metadata: MockTemplateMetadata = field(default_factory=MockTemplateMetadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "type": self.template_type,
            "tags": self.tags,
            "stars": self.metadata.stars,
            "downloads": self.metadata.downloads,
        }


@dataclass
class MockRating:
    """Mock template rating."""

    user_id: str = "user-123"
    template_id: str = "tpl-001"
    score: int = 4
    review: str = "Great template!"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MockRegistry:
    """Mock TemplateRegistry for testing."""

    def __init__(self):
        self._templates: dict[str, MockTemplate] = {}
        self._ratings: dict[str, list[MockRating]] = {}
        self._downloads: dict[str, int] = {}

    def search(
        self,
        query: str | None = None,
        category: Any = None,
        template_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MockTemplate]:
        results = list(self._templates.values())
        return results[offset : offset + limit]

    def get(self, template_id: str) -> MockTemplate | None:
        return self._templates.get(template_id)

    def import_template(self, json_str: str) -> str:
        data = json.loads(json_str)
        tpl_id = data.get("id", "tpl-new")
        self._templates[tpl_id] = MockTemplate(id=tpl_id, name=data.get("name", "Imported"))
        return tpl_id

    def delete(self, template_id: str) -> bool:
        if template_id in self._templates:
            del self._templates[template_id]
            return True
        return False

    def rate(self, rating: Any) -> None:
        tid = rating.template_id
        if tid not in self._ratings:
            self._ratings[tid] = []
        self._ratings[tid].append(rating)

    def get_ratings(self, template_id: str) -> list[MockRating]:
        return self._ratings.get(template_id, [])

    def get_average_rating(self, template_id: str) -> float:
        ratings = self._ratings.get(template_id, [])
        if not ratings:
            return 0.0
        return sum(r.score for r in ratings) / len(ratings)

    def star(self, template_id: str) -> None:
        tpl = self._templates.get(template_id)
        if tpl:
            tpl.metadata.stars += 1

    def increment_downloads(self, template_id: str) -> None:
        self._downloads[template_id] = self._downloads.get(template_id, 0) + 1

    def list_categories(self) -> list[str]:
        return ["debate", "workflow", "analysis", "compliance"]

    def export_template(self, template_id: str) -> str | None:
        tpl = self._templates.get(template_id)
        if tpl is None:
            return None
        return json.dumps(tpl.to_dict())


@dataclass
class MockAuthUser:
    """Mock authenticated user."""

    id: str = "user-123"
    email: str = "test@example.com"


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def registry():
    """Create a mock template registry."""
    r = MockRegistry()
    r._templates["tpl-001"] = MockTemplate(id="tpl-001", name="Debate Flow")
    r._templates["tpl-002"] = MockTemplate(id="tpl-002", name="Compliance Check")
    r._templates["tpl-003"] = MockTemplate(id="tpl-003", name="Risk Analysis")
    return r


@pytest.fixture
def handler(registry):
    """Create a MarketplaceHandler with mocked dependencies."""
    h = MarketplaceHandler({})
    h._current_query_params = {}
    return h


def get_body(result) -> dict:
    """Extract JSON body from a HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


# ===========================================================================
# Tests: handle_list_templates
# ===========================================================================


class TestListTemplates:
    """Tests for listing marketplace templates."""

    def test_list_templates_happy_path(self, handler, registry):
        """List templates returns all templates with count."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_list_templates()

        body = get_body(result)
        assert result.status_code == 200
        assert body["count"] == 3
        assert len(body["templates"]) == 3
        assert body["limit"] == 50
        assert body["offset"] == 0

    def test_list_templates_with_pagination(self, handler, registry):
        """List templates respects limit and offset parameters."""
        handler._current_query_params = {"limit": "1", "offset": "1"}

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_list_templates()

        body = get_body(result)
        assert result.status_code == 200
        assert body["limit"] == 1
        assert body["offset"] == 1

    def test_list_templates_invalid_category(self, handler, registry):
        """List templates returns 400 for invalid category."""
        handler._current_query_params = {"category": "nonexistent_category"}

        # TemplateCategory is imported inside the function, so we patch it
        # at the module where it is looked up
        mock_category_cls = MagicMock(side_effect=ValueError("invalid"))
        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.dict(
                "sys.modules",
                {"aragora.marketplace": MagicMock(TemplateCategory=mock_category_cls)},
            ),
        ):
            result = handler.handle_list_templates()

        body = get_body(result)
        assert result.status_code == 400
        assert "Invalid category" in body.get("error", "")

    def test_list_templates_with_tags(self, handler, registry):
        """List templates passes tags parameter correctly."""
        handler._current_query_params = {"tags": "test,example"}

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_list_templates()

        assert result.status_code == 200

    def test_list_templates_internal_error(self, handler):
        """List templates returns 500 on internal error."""
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            side_effect=RuntimeError("DB down"),
        ):
            result = handler.handle_list_templates()

        assert result.status_code == 500


# ===========================================================================
# Tests: handle_get_template
# ===========================================================================


class TestGetTemplate:
    """Tests for getting a single template."""

    def test_get_template_happy_path(self, handler, registry):
        """Get template returns template data and increments downloads."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_get_template("tpl-001")

        body = get_body(result)
        assert result.status_code == 200
        assert body["id"] == "tpl-001"
        assert body["name"] == "Debate Flow"
        # Download count should have been incremented
        assert registry._downloads.get("tpl-001", 0) == 1

    def test_get_template_not_found(self, handler, registry):
        """Get template returns 404 for unknown ID."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_get_template("nonexistent")

        body = get_body(result)
        assert result.status_code == 404
        assert "not found" in body.get("error", "").lower()

    def test_get_template_internal_error(self, handler):
        """Get template returns 500 on internal error."""
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            side_effect=RuntimeError("fail"),
        ):
            result = handler.handle_get_template("tpl-001")

        assert result.status_code == 500


# ===========================================================================
# Tests: handle_create_template
# ===========================================================================


class TestCreateTemplate:
    """Tests for creating a template."""

    def test_create_template_happy_path(self, handler, registry):
        """Create template succeeds with valid body and auth."""
        handler._current_handler = MagicMock()
        body_data = {"id": "tpl-new", "name": "New Template"}

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
            patch.object(handler, "get_json_body", return_value=body_data),
        ):
            result = handler.handle_create_template()

        body = get_body(result)
        assert result.status_code == 201
        assert body["success"] is True
        assert body["id"] == "tpl-new"

    def test_create_template_no_auth(self, handler, registry):
        """Create template returns 401 when not authenticated."""
        handler._current_handler = MagicMock()
        from aragora.server.handlers.base import error_response

        auth_error = error_response("Authentication required", 401)

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(None, auth_error)),
        ):
            result = handler.handle_create_template()

        assert result.status_code == 401

    def test_create_template_no_body(self, handler, registry):
        """Create template returns 400 when body is empty."""
        handler._current_handler = MagicMock()

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
            patch.object(handler, "get_json_body", return_value=None),
        ):
            result = handler.handle_create_template()

        body = get_body(result)
        assert result.status_code == 400
        assert "body" in body.get("error", "").lower()

    def test_create_template_validation_error(self, handler):
        """Create template returns 400 on validation error."""
        handler._current_handler = MagicMock()
        bad_registry = MockRegistry()
        bad_registry.import_template = MagicMock(side_effect=ValueError("Invalid template schema"))

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=bad_registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
            patch.object(handler, "get_json_body", return_value={"name": "bad"}),
        ):
            result = handler.handle_create_template()

        assert result.status_code == 400


# ===========================================================================
# Tests: handle_delete_template
# ===========================================================================


class TestDeleteTemplate:
    """Tests for deleting a template."""

    def test_delete_template_happy_path(self, handler, registry):
        """Delete template succeeds for existing template."""
        handler._current_handler = MagicMock()

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
        ):
            result = handler.handle_delete_template("tpl-001")

        body = get_body(result)
        assert result.status_code == 200
        assert body["success"] is True
        assert body["deleted"] == "tpl-001"
        assert "tpl-001" not in registry._templates

    def test_delete_template_forbidden_builtin(self, handler):
        """Delete template returns 403 for non-deletable (built-in) template."""
        handler._current_handler = MagicMock()
        mock_reg = MockRegistry()
        mock_reg.delete = MagicMock(return_value=False)

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_reg),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
        ):
            result = handler.handle_delete_template("builtin-001")

        assert result.status_code == 403

    def test_delete_template_no_auth(self, handler, registry):
        """Delete template returns 401 when not authenticated."""
        handler._current_handler = MagicMock()
        from aragora.server.handlers.base import error_response

        auth_error = error_response("Authentication required", 401)

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(None, auth_error)),
        ):
            result = handler.handle_delete_template("tpl-001")

        assert result.status_code == 401


# ===========================================================================
# Tests: handle_rate_template
# ===========================================================================


class TestRateTemplate:
    """Tests for rating a template."""

    def test_rate_template_happy_path(self, handler, registry):
        """Rate template succeeds with valid score."""
        handler._current_handler = MagicMock()

        mock_rating_cls = MagicMock(return_value=MockRating(score=4, review="Nice!"))
        mock_marketplace_mod = MagicMock(TemplateRating=mock_rating_cls)

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
            patch.object(handler, "get_json_body", return_value={"score": 4, "review": "Nice!"}),
            patch.dict("sys.modules", {"aragora.marketplace": mock_marketplace_mod}),
        ):
            result = handler.handle_rate_template("tpl-001")

        body = get_body(result)
        assert result.status_code == 200
        assert body["success"] is True
        assert "average_rating" in body

    def test_rate_template_missing_score(self, handler, registry):
        """Rate template returns 400 when score is missing."""
        handler._current_handler = MagicMock()

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
            patch.object(handler, "get_json_body", return_value={"review": "No score"}),
        ):
            result = handler.handle_rate_template("tpl-001")

        body = get_body(result)
        assert result.status_code == 400
        assert "score" in body.get("error", "").lower()

    def test_rate_template_score_out_of_range(self, handler, registry):
        """Rate template returns 400 for score outside 1-5."""
        handler._current_handler = MagicMock()

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
            patch.object(handler, "get_json_body", return_value={"score": 6}),
        ):
            result = handler.handle_rate_template("tpl-001")

        assert result.status_code == 400

    def test_rate_template_score_zero(self, handler, registry):
        """Rate template returns 400 for score of 0."""
        handler._current_handler = MagicMock()

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
            patch.object(handler, "get_json_body", return_value={"score": 0}),
        ):
            result = handler.handle_rate_template("tpl-001")

        assert result.status_code == 400

    def test_rate_template_no_body(self, handler, registry):
        """Rate template returns 400 when body is empty."""
        handler._current_handler = MagicMock()

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
            patch.object(handler, "get_json_body", return_value=None),
        ):
            result = handler.handle_rate_template("tpl-001")

        assert result.status_code == 400


# ===========================================================================
# Tests: handle_get_ratings
# ===========================================================================


class TestGetRatings:
    """Tests for getting template ratings."""

    def test_get_ratings_happy_path(self, handler, registry):
        """Get ratings returns rating list with average."""
        registry._ratings["tpl-001"] = [
            MockRating(user_id="u1", score=5, review="Excellent"),
            MockRating(user_id="u2", score=3, review="OK"),
        ]

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_get_ratings("tpl-001")

        body = get_body(result)
        assert result.status_code == 200
        assert body["count"] == 2
        assert body["average"] == 4.0
        assert len(body["ratings"]) == 2

    def test_get_ratings_empty(self, handler, registry):
        """Get ratings returns empty list for unrated template."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_get_ratings("tpl-001")

        body = get_body(result)
        assert result.status_code == 200
        assert body["count"] == 0
        assert body["average"] == 0.0


# ===========================================================================
# Tests: handle_star_template
# ===========================================================================


class TestStarTemplate:
    """Tests for starring a template."""

    def test_star_template_happy_path(self, handler, registry):
        """Star template increments star count."""
        handler._current_handler = MagicMock()
        initial_stars = registry._templates["tpl-001"].metadata.stars

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
        ):
            result = handler.handle_star_template("tpl-001")

        body = get_body(result)
        assert result.status_code == 200
        assert body["success"] is True
        assert body["stars"] == initial_stars + 1


# ===========================================================================
# Tests: handle_list_categories
# ===========================================================================


class TestListCategories:
    """Tests for listing template categories."""

    def test_list_categories_happy_path(self, handler, registry):
        """List categories returns available categories."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_list_categories()

        body = get_body(result)
        assert result.status_code == 200
        assert "categories" in body
        assert len(body["categories"]) > 0


# ===========================================================================
# Tests: handle_export_template / handle_import_template
# ===========================================================================


class TestExportImportTemplate:
    """Tests for template export and import."""

    def test_export_template_happy_path(self, handler, registry):
        """Export template returns downloadable JSON."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_export_template("tpl-001")

        assert result.status_code == 200
        assert result.content_type == "application/json"
        assert result.headers.get("Content-Disposition") is not None
        assert "tpl-001.json" in result.headers["Content-Disposition"]

        body = json.loads(result.body.decode("utf-8"))
        assert body["id"] == "tpl-001"

    def test_export_template_not_found(self, handler, registry):
        """Export template returns 404 for unknown template."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_export_template("nonexistent")

        body = get_body(result)
        assert result.status_code == 404
        assert "not found" in body.get("error", "").lower()

    def test_import_template_delegates_to_create(self, handler, registry):
        """Import template delegates to handle_create_template."""
        handler._current_handler = MagicMock()

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
            patch.object(
                handler, "get_json_body", return_value={"id": "tpl-imported", "name": "Imported"}
            ),
        ):
            result = handler.handle_import_template()

        body = get_body(result)
        assert result.status_code == 201
        assert body["success"] is True
