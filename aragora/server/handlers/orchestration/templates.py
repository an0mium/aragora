"""
Deliberation template loading for orchestration.

Attempts to import from the dedicated aragora.deliberation.templates module,
falling back to a minimal built-in set if unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aragora.config import MAX_ROUNDS
from aragora.server.handlers.orchestration.models import OutputFormat

# =============================================================================
# Deliberation Templates - Import from templates module
# =============================================================================

# Pre-declare types for fallback case
DeliberationTemplate: Any = None
TEMPLATES: dict[str, Any] = {}
_list_templates: Any = None
_get_template: Any = None

try:
    from aragora.deliberation.templates import (
        DeliberationTemplate,
        BUILTIN_TEMPLATES,
        list_templates as _list_templates,
        get_template as _get_template,
    )

    # Use the expanded template set from the templates module
    TEMPLATES = BUILTIN_TEMPLATES
except ImportError:
    # Fallback if templates module not available
    @dataclass
    class _FallbackDeliberationTemplate:
        """Pre-built vetted decisionmaking pattern (fallback)."""

        name: str
        description: str
        default_agents: list[str] = field(default_factory=list)
        default_knowledge_sources: list[str] = field(default_factory=list)
        output_format: Any = OutputFormat.STANDARD
        consensus_threshold: float = 0.7
        max_rounds: int = MAX_ROUNDS
        personas: list[str] = field(default_factory=list)

        def to_dict(self) -> dict[str, Any]:
            """Convert to dictionary."""
            output_fmt = self.output_format
            output_value = output_fmt.value if hasattr(output_fmt, "value") else str(output_fmt)
            return {
                "name": self.name,
                "description": self.description,
                "default_agents": self.default_agents,
                "default_knowledge_sources": self.default_knowledge_sources,
                "output_format": output_value,
                "consensus_threshold": self.consensus_threshold,
                "max_rounds": self.max_rounds,
                "personas": self.personas,
            }

    DeliberationTemplate = _FallbackDeliberationTemplate

    # Fallback templates
    TEMPLATES = {
        "code_review": _FallbackDeliberationTemplate(
            name="code_review",
            description="Multi-agent code review with security focus",
            default_agents=["anthropic-api", "openai-api", "codestral"],
            default_knowledge_sources=["github:pr"],
            output_format=OutputFormat.GITHUB_REVIEW,
            consensus_threshold=0.7,
            max_rounds=3,
            personas=["security", "performance", "maintainability"],
        ),
        "quick_decision": _FallbackDeliberationTemplate(
            name="quick_decision",
            description="Fast decision with minimal agents",
            default_agents=["anthropic-api", "openai-api"],
            output_format=OutputFormat.SUMMARY,
            consensus_threshold=0.5,
            max_rounds=2,
        ),
    }

    def _list_templates(**kwargs: Any) -> list[Any]:
        return list(TEMPLATES.values())

    def _get_template(name: str) -> Any:
        return TEMPLATES.get(name)
