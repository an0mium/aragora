"""Tests for the orchestration templates module.

Tests the template loading, fallback behaviour, and public API exposed by
``aragora.server.handlers.orchestration.templates``.

Covers:
- Successful import path (uses aragora.deliberation.templates)
- Fallback path (ImportError triggers in-module _FallbackDeliberationTemplate)
- _list_templates() and _get_template() functions
- _FallbackDeliberationTemplate dataclass and to_dict()
- TEMPLATES dictionary contents
- OutputFormat enum values in to_dict()
- Edge cases (missing templates, empty dicts, custom defaults)
"""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reload_templates_module():
    """Force-reload the templates module to re-trigger import logic."""
    mod_name = "aragora.server.handlers.orchestration.templates"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


def _reload_with_import_error():
    """Reload the templates module with deliberation.templates import blocked.

    Returns the reloaded module so tests can inspect fallback objects.
    """
    mod_name = "aragora.server.handlers.orchestration.templates"
    deliberation_mod = "aragora.deliberation.templates"

    # Remove cached modules
    for key in list(sys.modules):
        if key == mod_name or key.startswith(deliberation_mod):
            del sys.modules[key]

    # Block the deliberation import
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _blocked_import(name, *args, **kwargs):
        if name == deliberation_mod:
            raise ImportError("Simulated: deliberation.templates not available")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_blocked_import):
        mod = importlib.import_module(mod_name)

    return mod


# ============================================================================
# A. Normal (successful import) path
# ============================================================================


class TestSuccessfulImportPath:
    """Tests when aragora.deliberation.templates is available."""

    def test_templates_dict_is_populated(self):
        mod = _reload_templates_module()
        assert isinstance(mod.TEMPLATES, dict)
        assert len(mod.TEMPLATES) > 0

    def test_templates_contains_code_review(self):
        mod = _reload_templates_module()
        assert "code_review" in mod.TEMPLATES

    def test_templates_contains_quick_decision(self):
        mod = _reload_templates_module()
        assert "quick_decision" in mod.TEMPLATES

    def test_deliberation_template_class_imported(self):
        mod = _reload_templates_module()
        assert mod.DeliberationTemplate is not None

    def test_list_templates_callable(self):
        mod = _reload_templates_module()
        assert callable(mod._list_templates)

    def test_list_templates_returns_list(self):
        mod = _reload_templates_module()
        result = mod._list_templates()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_template_returns_existing(self):
        mod = _reload_templates_module()
        template = mod._get_template("code_review")
        assert template is not None
        assert template.name == "code_review"

    def test_get_template_returns_none_for_missing(self):
        mod = _reload_templates_module()
        template = mod._get_template("nonexistent_template_xyz")
        assert template is None

    def test_templates_use_builtin_templates(self):
        """Verify TEMPLATES is BUILTIN_TEMPLATES from the deliberation module."""
        mod = _reload_templates_module()
        from aragora.deliberation.templates import BUILTIN_TEMPLATES

        assert mod.TEMPLATES is BUILTIN_TEMPLATES

    def test_template_has_to_dict(self):
        mod = _reload_templates_module()
        template = mod._get_template("code_review")
        assert hasattr(template, "to_dict")
        d = template.to_dict()
        assert isinstance(d, dict)
        assert "name" in d
        assert d["name"] == "code_review"

    def test_all_templates_have_name(self):
        mod = _reload_templates_module()
        for name, template in mod.TEMPLATES.items():
            assert template.name == name

    def test_builtin_template_count(self):
        """There should be many builtin templates (currently 23+)."""
        mod = _reload_templates_module()
        assert len(mod.TEMPLATES) >= 10


# ============================================================================
# B. Fallback (ImportError) path
# ============================================================================


class TestFallbackPath:
    """Tests when aragora.deliberation.templates is NOT available."""

    def test_fallback_templates_dict_populated(self):
        mod = _reload_with_import_error()
        assert isinstance(mod.TEMPLATES, dict)
        assert len(mod.TEMPLATES) > 0

    def test_fallback_has_code_review(self):
        mod = _reload_with_import_error()
        assert "code_review" in mod.TEMPLATES

    def test_fallback_has_quick_decision(self):
        mod = _reload_with_import_error()
        assert "quick_decision" in mod.TEMPLATES

    def test_fallback_only_two_templates(self):
        mod = _reload_with_import_error()
        assert len(mod.TEMPLATES) == 2

    def test_fallback_list_templates_returns_all(self):
        mod = _reload_with_import_error()
        result = mod._list_templates()
        assert isinstance(result, list)
        assert len(result) == 2

    def test_fallback_get_template_existing(self):
        mod = _reload_with_import_error()
        template = mod._get_template("code_review")
        assert template is not None
        assert template.name == "code_review"

    def test_fallback_get_template_missing(self):
        mod = _reload_with_import_error()
        result = mod._get_template("nonexistent")
        assert result is None

    def test_fallback_get_template_quick_decision(self):
        mod = _reload_with_import_error()
        template = mod._get_template("quick_decision")
        assert template is not None
        assert template.name == "quick_decision"


# ============================================================================
# C. _FallbackDeliberationTemplate dataclass
# ============================================================================


class TestFallbackDeliberationTemplate:
    """Tests for the _FallbackDeliberationTemplate dataclass."""

    def _get_fallback_class(self):
        mod = _reload_with_import_error()
        return mod.DeliberationTemplate

    def test_is_dataclass(self):
        from dataclasses import fields

        cls = self._get_fallback_class()
        assert len(fields(cls)) > 0

    def test_default_values(self):
        cls = self._get_fallback_class()
        instance = cls(name="test", description="desc")
        assert instance.name == "test"
        assert instance.description == "desc"
        assert instance.default_agents == []
        assert instance.default_knowledge_sources == []
        assert instance.consensus_threshold == 0.7
        assert instance.personas == []

    def test_max_rounds_default(self):
        from aragora.config import MAX_ROUNDS

        cls = self._get_fallback_class()
        instance = cls(name="t", description="d")
        assert instance.max_rounds == MAX_ROUNDS

    def test_output_format_default(self):
        from aragora.server.handlers.orchestration.models import OutputFormat

        cls = self._get_fallback_class()
        instance = cls(name="t", description="d")
        assert instance.output_format == OutputFormat.STANDARD

    def test_custom_agents(self):
        cls = self._get_fallback_class()
        agents = ["a1", "a2", "a3"]
        instance = cls(name="t", description="d", default_agents=agents)
        assert instance.default_agents == agents

    def test_custom_consensus_threshold(self):
        cls = self._get_fallback_class()
        instance = cls(name="t", description="d", consensus_threshold=0.9)
        assert instance.consensus_threshold == 0.9

    def test_custom_max_rounds(self):
        cls = self._get_fallback_class()
        instance = cls(name="t", description="d", max_rounds=7)
        assert instance.max_rounds == 7

    def test_custom_personas(self):
        cls = self._get_fallback_class()
        instance = cls(name="t", description="d", personas=["p1", "p2"])
        assert instance.personas == ["p1", "p2"]


# ============================================================================
# D. _FallbackDeliberationTemplate.to_dict()
# ============================================================================


class TestFallbackToDict:
    """Tests for the to_dict method of the fallback template."""

    def _make_template(self, **kwargs):
        mod = _reload_with_import_error()
        defaults = {"name": "test_template", "description": "A test template"}
        defaults.update(kwargs)
        return mod.DeliberationTemplate(**defaults)

    def test_to_dict_returns_dict(self):
        t = self._make_template()
        d = t.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_all_keys(self):
        t = self._make_template()
        d = t.to_dict()
        expected_keys = {
            "name",
            "description",
            "default_agents",
            "default_knowledge_sources",
            "output_format",
            "consensus_threshold",
            "max_rounds",
            "personas",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_name(self):
        t = self._make_template(name="my_template")
        assert t.to_dict()["name"] == "my_template"

    def test_to_dict_description(self):
        t = self._make_template(description="Hello world")
        assert t.to_dict()["description"] == "Hello world"

    def test_to_dict_default_agents(self):
        t = self._make_template(default_agents=["x", "y"])
        assert t.to_dict()["default_agents"] == ["x", "y"]

    def test_to_dict_default_knowledge_sources(self):
        t = self._make_template(default_knowledge_sources=["gh:pr"])
        assert t.to_dict()["default_knowledge_sources"] == ["gh:pr"]

    def test_to_dict_consensus_threshold(self):
        t = self._make_template(consensus_threshold=0.85)
        assert t.to_dict()["consensus_threshold"] == 0.85

    def test_to_dict_max_rounds(self):
        t = self._make_template(max_rounds=10)
        assert t.to_dict()["max_rounds"] == 10

    def test_to_dict_personas(self):
        t = self._make_template(personas=["security", "performance"])
        assert t.to_dict()["personas"] == ["security", "performance"]

    def test_to_dict_output_format_enum_value(self):
        """OutputFormat enum should be serialized to its .value string."""
        from aragora.server.handlers.orchestration.models import OutputFormat

        t = self._make_template(output_format=OutputFormat.GITHUB_REVIEW)
        assert t.to_dict()["output_format"] == "github_review"

    def test_to_dict_output_format_standard(self):
        from aragora.server.handlers.orchestration.models import OutputFormat

        t = self._make_template(output_format=OutputFormat.STANDARD)
        assert t.to_dict()["output_format"] == "standard"

    def test_to_dict_output_format_summary(self):
        from aragora.server.handlers.orchestration.models import OutputFormat

        t = self._make_template(output_format=OutputFormat.SUMMARY)
        assert t.to_dict()["output_format"] == "summary"

    def test_to_dict_output_format_string_fallback(self):
        """When output_format is a plain string (no .value), str() is used."""
        t = self._make_template(output_format="custom_format")
        assert t.to_dict()["output_format"] == "custom_format"


# ============================================================================
# E. Fallback template data correctness
# ============================================================================


class TestFallbackTemplateData:
    """Verify the specific fallback template definitions."""

    def _get_templates(self):
        mod = _reload_with_import_error()
        return mod.TEMPLATES

    def test_code_review_name(self):
        templates = self._get_templates()
        assert templates["code_review"].name == "code_review"

    def test_code_review_description(self):
        templates = self._get_templates()
        assert "code review" in templates["code_review"].description.lower()

    def test_code_review_agents(self):
        templates = self._get_templates()
        agents = templates["code_review"].default_agents
        assert "anthropic-api" in agents
        assert "openai-api" in agents
        assert "codestral" in agents

    def test_code_review_knowledge_sources(self):
        templates = self._get_templates()
        assert templates["code_review"].default_knowledge_sources == ["github:pr"]

    def test_code_review_output_format(self):
        from aragora.server.handlers.orchestration.models import OutputFormat

        templates = self._get_templates()
        assert templates["code_review"].output_format == OutputFormat.GITHUB_REVIEW

    def test_code_review_consensus_threshold(self):
        templates = self._get_templates()
        assert templates["code_review"].consensus_threshold == 0.7

    def test_code_review_max_rounds(self):
        templates = self._get_templates()
        assert templates["code_review"].max_rounds == 3

    def test_code_review_personas(self):
        templates = self._get_templates()
        personas = templates["code_review"].personas
        assert "security" in personas
        assert "performance" in personas
        assert "maintainability" in personas

    def test_quick_decision_name(self):
        templates = self._get_templates()
        assert templates["quick_decision"].name == "quick_decision"

    def test_quick_decision_description(self):
        templates = self._get_templates()
        assert "fast" in templates["quick_decision"].description.lower()

    def test_quick_decision_agents(self):
        templates = self._get_templates()
        agents = templates["quick_decision"].default_agents
        assert "anthropic-api" in agents
        assert "openai-api" in agents
        assert len(agents) == 2

    def test_quick_decision_output_format(self):
        from aragora.server.handlers.orchestration.models import OutputFormat

        templates = self._get_templates()
        assert templates["quick_decision"].output_format == OutputFormat.SUMMARY

    def test_quick_decision_consensus_threshold(self):
        templates = self._get_templates()
        assert templates["quick_decision"].consensus_threshold == 0.5

    def test_quick_decision_max_rounds(self):
        templates = self._get_templates()
        assert templates["quick_decision"].max_rounds == 2

    def test_quick_decision_no_knowledge_sources(self):
        templates = self._get_templates()
        assert templates["quick_decision"].default_knowledge_sources == []

    def test_quick_decision_no_personas(self):
        templates = self._get_templates()
        assert templates["quick_decision"].personas == []


# ============================================================================
# F. Module-level exports
# ============================================================================


class TestModuleExports:
    """Verify that the module exposes the expected names."""

    def test_exports_deliberation_template(self):
        mod = _reload_templates_module()
        assert hasattr(mod, "DeliberationTemplate")

    def test_exports_templates(self):
        mod = _reload_templates_module()
        assert hasattr(mod, "TEMPLATES")

    def test_exports_list_templates(self):
        mod = _reload_templates_module()
        assert hasattr(mod, "_list_templates")

    def test_exports_get_template(self):
        mod = _reload_templates_module()
        assert hasattr(mod, "_get_template")


# ============================================================================
# G. list_templates / get_template behaviour (normal path)
# ============================================================================


class TestListGetTemplatesNormal:
    """Test _list_templates and _get_template with the real deliberation module."""

    def test_list_templates_includes_code_review(self):
        mod = _reload_templates_module()
        names = [t.name for t in mod._list_templates()]
        assert "code_review" in names

    def test_list_templates_includes_quick_decision(self):
        mod = _reload_templates_module()
        names = [t.name for t in mod._list_templates()]
        assert "quick_decision" in names

    def test_get_template_code_review_has_to_dict(self):
        mod = _reload_templates_module()
        template = mod._get_template("code_review")
        d = template.to_dict()
        assert d["name"] == "code_review"
        assert "output_format" in d

    def test_get_template_returns_none_for_empty_string(self):
        mod = _reload_templates_module()
        assert mod._get_template("") is None
