"""Tests for the Template Discovery API handler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.deliberation.templates.base import DeliberationTemplate, TemplateCategory
from aragora.deliberation.templates.registry import TemplateRegistry
from aragora.server.handlers.template_discovery import TemplateDiscoveryHandler


@pytest.fixture
def handler():
    """Create a TemplateDiscoveryHandler with mock server context."""
    ctx = {"storage": MagicMock()}
    return TemplateDiscoveryHandler(ctx)


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler."""
    h = MagicMock()
    h.client_address = ("127.0.0.1", 12345)
    h.headers = {}
    return h


class TestTemplateDiscoveryList:
    """Tests for GET /api/v1/templates."""

    def test_list_returns_all_templates(self, handler, mock_handler):
        """Listing templates returns all built-in templates."""
        result = handler.handle("/api/v1/templates", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        data = result[0]  # body data (auto-decoded JSON)
        assert "templates" in data
        assert "count" in data
        assert data["count"] >= 24

    def test_list_filter_by_category(self, handler, mock_handler):
        """Filtering by category returns only matching templates."""
        result = handler.handle("/api/v1/templates", {"category": "code"}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        data = result[0]
        for t in data["templates"]:
            assert t["category"] == "code"

    def test_list_filter_by_search(self, handler, mock_handler):
        """Search filter matches template name/description."""
        result = handler.handle("/api/v1/templates", {"search": "hiring"}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        data = result[0]
        assert data["count"] >= 1
        names = [t["name"] for t in data["templates"]]
        assert "hiring_decision" in names

    def test_list_unknown_category_returns_all(self, handler, mock_handler):
        """Unknown category is ignored, returns all templates."""
        result = handler.handle("/api/v1/templates", {"category": "nonexistent"}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        data = result[0]
        assert data["count"] >= 24


class TestTemplateDiscoveryCategories:
    """Tests for GET /api/v1/templates/categories."""

    def test_categories_returns_counts(self, handler, mock_handler):
        """Categories endpoint returns category-to-count mapping."""
        result = handler.handle("/api/v1/templates/categories", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        data = result[0]
        assert "categories" in data
        categories = data["categories"]
        assert "code" in categories
        assert "business" in categories
        assert "general" in categories
        for count in categories.values():
            assert count > 0


class TestTemplateDiscoveryRecommend:
    """Tests for GET /api/v1/templates/recommend."""

    def test_recommend_returns_hiring_for_hire_question(self, handler, mock_handler):
        """Recommend returns hiring_decision for 'hire VP' question."""
        result = handler.handle(
            "/api/v1/templates/recommend",
            {"question": "Should we hire this VP candidate?"},
            mock_handler,
        )
        assert result is not None
        assert result.status_code == 200
        data = result[0]
        assert "recommended" in data
        assert len(data["recommended"]) > 0
        names = [r["name"] for r in data["recommended"]]
        assert "hiring_decision" in names

    def test_recommend_missing_question_returns_400(self, handler, mock_handler):
        """Missing question parameter returns 400."""
        result = handler.handle("/api/v1/templates/recommend", {}, mock_handler)
        assert result is not None
        assert result.status_code == 400

    def test_recommend_with_domain_boost(self, handler, mock_handler):
        """Domain parameter boosts matching category templates."""
        result = handler.handle(
            "/api/v1/templates/recommend",
            {"question": "review this code change", "domain": "code"},
            mock_handler,
        )
        assert result is not None
        assert result.status_code == 200
        data = result[0]
        assert len(data["recommended"]) > 0
        assert data["recommended"][0]["category"] == "code"

    def test_recommend_returns_max_3(self, handler, mock_handler):
        """Recommend returns at most 3 results."""
        result = handler.handle(
            "/api/v1/templates/recommend",
            {"question": "review audit compliance security code"},
            mock_handler,
        )
        assert result is not None
        assert result.status_code == 200
        data = result[0]
        assert len(data["recommended"]) <= 3


class TestTemplateDiscoveryDetail:
    """Tests for GET /api/v1/templates/{name}."""

    def test_detail_returns_template(self, handler, mock_handler):
        """Detail endpoint returns a specific template."""
        result = handler.handle("/api/v1/templates/code_review", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        data = result[0]
        assert data["name"] == "code_review"
        assert data["category"] == "code"
        assert "example_topics" in data
        assert len(data["example_topics"]) == 3

    def test_detail_unknown_returns_404(self, handler, mock_handler):
        """Unknown template name returns 404."""
        result = handler.handle("/api/v1/templates/nonexistent_template", {}, mock_handler)
        assert result is not None
        assert result.status_code == 404


class TestExampleTopics:
    """Tests for example_topics on templates."""

    def test_all_builtins_have_example_topics(self):
        """Every built-in template should have exactly 3 example_topics."""
        from aragora.deliberation.templates.builtins import BUILTIN_TEMPLATES

        for name, template in BUILTIN_TEMPLATES.items():
            assert len(template.example_topics) == 3, (
                f"Template '{name}' has {len(template.example_topics)} example_topics, expected 3"
            )

    def test_example_topics_in_to_dict(self):
        """example_topics should appear in to_dict() output."""
        from aragora.deliberation.templates.builtins import CODE_REVIEW

        d = CODE_REVIEW.to_dict()
        assert "example_topics" in d
        assert len(d["example_topics"]) == 3

    def test_example_topics_from_dict(self):
        """example_topics should round-trip through from_dict()."""
        from aragora.deliberation.templates.builtins import CODE_REVIEW

        d = CODE_REVIEW.to_dict()
        reconstructed = DeliberationTemplate.from_dict(d)
        assert reconstructed.example_topics == CODE_REVIEW.example_topics


class TestTemplateRegistryRecommend:
    """Tests for the TemplateRegistry.recommend() method."""

    def test_recommend_empty_question(self):
        """Empty question returns empty list."""
        registry = TemplateRegistry()
        result = registry.recommend("")
        assert result == []

    def test_recommend_scores_by_keyword_overlap(self):
        """Keywords from question should match template names/descriptions."""
        registry = TemplateRegistry()
        result = registry.recommend("code review security")
        assert len(result) > 0
        assert result[0].name == "code_review"

    def test_recommend_respects_limit(self):
        """Limit parameter constrains results."""
        registry = TemplateRegistry()
        result = registry.recommend("business hiring vendor budget", limit=2)
        assert len(result) <= 2
