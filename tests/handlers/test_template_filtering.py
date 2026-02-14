"""Tests for template browser with filtering in orchestration handler."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.orchestration import OrchestrationHandler


@dataclass
class FakeTemplate:
    """Minimal template for testing."""

    name: str
    description: str = ""
    category: str = "general"
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
        }


FAKE_TEMPLATES = [
    FakeTemplate("hiring_decision", "Evaluate candidates", "business", ["sme", "hr"]),
    FakeTemplate("code_review", "Review pull requests", "code", ["dev", "security"]),
    FakeTemplate("budget_approval", "Approve budgets", "business", ["sme", "finance"]),
    FakeTemplate("legal_review", "Review contracts", "legal", ["compliance"]),
    FakeTemplate("security_audit", "Security assessment", "code", ["security"]),
]


def _parse_body(result) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


@pytest.fixture
def handler():
    return OrchestrationHandler({})


class TestTemplateFilteringNoParams:
    """Without query params, all templates are returned."""

    def test_returns_all_templates(self, handler):
        with patch(
            "aragora.server.handlers.orchestration.templates._list_templates",
            return_value=FAKE_TEMPLATES,
        ):
            result = handler._get_templates({})

        body = _parse_body(result)
        assert body["count"] == 5
        assert len(body["templates"]) == 5

    def test_returns_template_dicts(self, handler):
        with patch(
            "aragora.server.handlers.orchestration.templates._list_templates",
            return_value=FAKE_TEMPLATES[:1],
        ):
            result = handler._get_templates({})

        body = _parse_body(result)
        assert body["templates"][0]["name"] == "hiring_decision"


class TestTemplateFilterByCategory:
    """Filter templates by category query param."""

    def test_category_passed_to_list_templates(self, handler):
        mock_list = MagicMock(return_value=[])
        with patch(
            "aragora.server.handlers.orchestration.templates._list_templates",
            mock_list,
        ):
            handler._get_templates({"category": "business"})

        mock_list.assert_called_once()
        assert mock_list.call_args.kwargs["category"] == "business"

    def test_no_category_passes_none(self, handler):
        mock_list = MagicMock(return_value=[])
        with patch(
            "aragora.server.handlers.orchestration.templates._list_templates",
            mock_list,
        ):
            handler._get_templates({})

        assert mock_list.call_args.kwargs["category"] is None


class TestTemplateFilterBySearch:
    """Filter templates by search query param."""

    def test_search_passed_to_list_templates(self, handler):
        mock_list = MagicMock(return_value=[])
        with patch(
            "aragora.server.handlers.orchestration.templates._list_templates",
            mock_list,
        ):
            handler._get_templates({"search": "hiring"})

        assert mock_list.call_args.kwargs["search"] == "hiring"

    def test_no_search_passes_none(self, handler):
        mock_list = MagicMock(return_value=[])
        with patch(
            "aragora.server.handlers.orchestration.templates._list_templates",
            mock_list,
        ):
            handler._get_templates({})

        assert mock_list.call_args.kwargs["search"] is None


class TestTemplateFilterByTags:
    """Filter templates by tags query param."""

    def test_tags_split_by_comma(self, handler):
        mock_list = MagicMock(return_value=[])
        with patch(
            "aragora.server.handlers.orchestration.templates._list_templates",
            mock_list,
        ):
            handler._get_templates({"tags": "sme,finance"})

        assert mock_list.call_args.kwargs["tags"] == ["sme", "finance"]

    def test_tags_strips_whitespace(self, handler):
        mock_list = MagicMock(return_value=[])
        with patch(
            "aragora.server.handlers.orchestration.templates._list_templates",
            mock_list,
        ):
            handler._get_templates({"tags": " sme , finance "})

        assert mock_list.call_args.kwargs["tags"] == ["sme", "finance"]

    def test_no_tags_passes_none(self, handler):
        mock_list = MagicMock(return_value=[])
        with patch(
            "aragora.server.handlers.orchestration.templates._list_templates",
            mock_list,
        ):
            handler._get_templates({})

        assert mock_list.call_args.kwargs["tags"] is None

    def test_empty_tags_passes_none(self, handler):
        mock_list = MagicMock(return_value=[])
        with patch(
            "aragora.server.handlers.orchestration.templates._list_templates",
            mock_list,
        ):
            handler._get_templates({"tags": ""})

        assert mock_list.call_args.kwargs["tags"] is None


class TestTemplatePagination:
    """Pagination via limit and offset query params."""

    def test_default_limit_50(self, handler):
        mock_list = MagicMock(return_value=[])
        with patch(
            "aragora.server.handlers.orchestration.templates._list_templates",
            mock_list,
        ):
            handler._get_templates({})

        assert mock_list.call_args.kwargs["limit"] == 50

    def test_custom_limit(self, handler):
        mock_list = MagicMock(return_value=[])
        with patch(
            "aragora.server.handlers.orchestration.templates._list_templates",
            mock_list,
        ):
            handler._get_templates({"limit": "10"})

        assert mock_list.call_args.kwargs["limit"] == 10

    def test_default_offset_0(self, handler):
        mock_list = MagicMock(return_value=[])
        with patch(
            "aragora.server.handlers.orchestration.templates._list_templates",
            mock_list,
        ):
            handler._get_templates({})

        assert mock_list.call_args.kwargs["offset"] == 0

    def test_custom_offset(self, handler):
        mock_list = MagicMock(return_value=[])
        with patch(
            "aragora.server.handlers.orchestration.templates._list_templates",
            mock_list,
        ):
            handler._get_templates({"offset": "5"})

        assert mock_list.call_args.kwargs["offset"] == 5

    def test_invalid_limit_defaults_to_50(self, handler):
        mock_list = MagicMock(return_value=[])
        with patch(
            "aragora.server.handlers.orchestration.templates._list_templates",
            mock_list,
        ):
            handler._get_templates({"limit": "not_a_number"})

        assert mock_list.call_args.kwargs["limit"] == 50

    def test_invalid_offset_defaults_to_0(self, handler):
        mock_list = MagicMock(return_value=[])
        with patch(
            "aragora.server.handlers.orchestration.templates._list_templates",
            mock_list,
        ):
            handler._get_templates({"offset": "abc"})

        assert mock_list.call_args.kwargs["offset"] == 0


class TestTemplateCombinedFilters:
    """Multiple filters work together."""

    def test_all_filters_combined(self, handler):
        mock_list = MagicMock(return_value=[])
        with patch(
            "aragora.server.handlers.orchestration.templates._list_templates",
            mock_list,
        ):
            handler._get_templates({
                "category": "business",
                "search": "hiring",
                "tags": "sme,hr",
                "limit": "20",
                "offset": "2",
            })

        kwargs = mock_list.call_args.kwargs
        assert kwargs["category"] == "business"
        assert kwargs["search"] == "hiring"
        assert kwargs["tags"] == ["sme", "hr"]
        assert kwargs["limit"] == 20
        assert kwargs["offset"] == 2


class TestTemplateFallback:
    """When _list_templates is None, falls back to TEMPLATES dict."""

    def test_fallback_returns_templates_dict(self, handler):
        fake_template = FakeTemplate("fallback_template", "A test", "general")
        with (
            patch(
                "aragora.server.handlers.orchestration.templates._list_templates",
                None,
            ),
            patch(
                "aragora.server.handlers.orchestration.handler.TEMPLATES",
                {"fallback_template": fake_template},
            ),
        ):
            result = handler._get_templates({})

        body = _parse_body(result)
        assert body["count"] == 1
        assert body["templates"][0]["name"] == "fallback_template"


class TestTemplateResponseFormat:
    """Response format is correct."""

    def test_response_has_templates_and_count_keys(self, handler):
        with patch(
            "aragora.server.handlers.orchestration.templates._list_templates",
            return_value=[],
        ):
            result = handler._get_templates({})

        body = _parse_body(result)
        assert "templates" in body
        assert "count" in body

    def test_count_matches_templates_length(self, handler):
        with patch(
            "aragora.server.handlers.orchestration.templates._list_templates",
            return_value=FAKE_TEMPLATES[:3],
        ):
            result = handler._get_templates({})

        body = _parse_body(result)
        assert body["count"] == len(body["templates"]) == 3

    def test_status_code_200(self, handler):
        with patch(
            "aragora.server.handlers.orchestration.templates._list_templates",
            return_value=[],
        ):
            result = handler._get_templates({})

        assert result.status_code == 200
