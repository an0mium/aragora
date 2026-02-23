"""Tests for auto-recommend templates on question submission."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.debate_controller import DebateController


class TestAutoRecommendTemplates:
    """Tests that _quick_classify includes suggested_templates."""

    def test_quick_classify_includes_suggested_templates(self):
        """_quick_classify emits suggested_templates in stream event."""
        factory = MagicMock()
        emitter = MagicMock()

        controller = DebateController(
            factory=factory,
            emitter=emitter,
        )

        # Mock run_async to return a classification
        with patch("aragora.server.debate_controller.run_async") as mock_run_async:
            mock_run_async.return_value = {
                "type": "technical",
                "domain": "technology",
                "complexity": "moderate",
                "aspects": ["security", "performance"],
                "approach": "Multi-agent analysis",
            }

            controller._quick_classify("Should we migrate to microservices?", "test-debate-1")

        # Verify emit was called
        assert emitter.emit.called
        event = emitter.emit.call_args[0][0]
        assert event.data.get("suggested_templates") is not None
        templates = event.data["suggested_templates"]
        assert isinstance(templates, list)
        # Each template should have name and description
        for t in templates:
            assert "name" in t
            assert "description" in t

    def test_quick_classify_suggested_templates_on_api_failure(self):
        """Templates are still suggested even if classification API fails."""
        factory = MagicMock()
        emitter = MagicMock()

        controller = DebateController(
            factory=factory,
            emitter=emitter,
        )

        # Mock run_async to return default classification (API failure)
        with patch("aragora.server.debate_controller.run_async") as mock_run_async:
            mock_run_async.return_value = {
                "type": "general",
                "domain": "other",
                "complexity": "moderate",
                "aspects": [],
                "approach": "",
            }

            controller._quick_classify("hire VP of engineering", "test-debate-2")

        event = emitter.emit.call_args[0][0]
        templates = event.data.get("suggested_templates", [])
        # Should still get template suggestions from keyword matching
        assert isinstance(templates, list)

    def test_quick_classify_hiring_suggests_hiring_template(self):
        """Hiring-related question should suggest hiring_decision template."""
        factory = MagicMock()
        emitter = MagicMock()

        controller = DebateController(
            factory=factory,
            emitter=emitter,
        )

        with patch("aragora.server.debate_controller.run_async") as mock_run_async:
            mock_run_async.return_value = {
                "type": "general",
                "domain": "other",
                "complexity": "moderate",
                "aspects": [],
                "approach": "",
            }

            controller._quick_classify("Should we hire this candidate for VP?", "test-debate-3")

        event = emitter.emit.call_args[0][0]
        templates = event.data.get("suggested_templates", [])
        template_names = [t["name"] for t in templates]
        assert "hiring_decision" in template_names

    def test_quick_classify_code_review_suggests_code_templates(self):
        """Code-related question with domain boost suggests code templates."""
        factory = MagicMock()
        emitter = MagicMock()

        controller = DebateController(
            factory=factory,
            emitter=emitter,
        )

        with patch("aragora.server.debate_controller.run_async") as mock_run_async:
            mock_run_async.return_value = {
                "type": "technical",
                "domain": "code",
                "complexity": "moderate",
                "aspects": ["quality"],
                "approach": "Review",
            }

            controller._quick_classify("Review this pull request for bugs", "test-debate-4")

        event = emitter.emit.call_args[0][0]
        templates = event.data.get("suggested_templates", [])
        template_names = [t["name"] for t in templates]
        assert "code_review" in template_names

    def test_quick_classify_compliance_suggests_compliance_templates(self):
        """Compliance question suggests compliance templates."""
        factory = MagicMock()
        emitter = MagicMock()

        controller = DebateController(
            factory=factory,
            emitter=emitter,
        )

        with patch("aragora.server.debate_controller.run_async") as mock_run_async:
            mock_run_async.return_value = {
                "type": "policy",
                "domain": "compliance",
                "complexity": "complex",
                "aspects": ["GDPR", "privacy"],
                "approach": "Compliance assessment",
            }

            controller._quick_classify("Is our data processing GDPR compliant?", "test-debate-5")

        event = emitter.emit.call_args[0][0]
        templates = event.data.get("suggested_templates", [])
        assert len(templates) > 0
        # Should include compliance-related templates
        template_names = [t["name"] for t in templates]
        assert any("gdpr" in name or "compliance" in name for name in template_names)

    def test_quick_classify_graceful_on_template_import_error(self):
        """Classification works even if template registry import fails."""
        factory = MagicMock()
        emitter = MagicMock()

        controller = DebateController(
            factory=factory,
            emitter=emitter,
        )

        with patch("aragora.server.debate_controller.run_async") as mock_run_async:
            mock_run_async.return_value = {
                "type": "general",
                "domain": "other",
                "complexity": "simple",
                "aspects": [],
                "approach": "",
            }
            # Patch the registry import to fail
            with patch.dict(
                "sys.modules",
                {"aragora.deliberation.templates.registry": None},
            ):
                # Should not raise, just emit without templates
                controller._quick_classify("test question", "test-debate-6")

        # Event should still be emitted
        assert emitter.emit.called
