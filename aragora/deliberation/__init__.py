"""
Aragora Deliberation Module.

Provides deliberation templates, patterns, and utilities for
multi-agent decision making.
"""

from aragora.deliberation.templates import (
    DeliberationTemplate,
    TemplateRegistry,
    get_template,
    list_templates,
    register_template,
    BUILTIN_TEMPLATES,
)

__all__ = [
    "DeliberationTemplate",
    "TemplateRegistry",
    "get_template",
    "list_templates",
    "register_template",
    "BUILTIN_TEMPLATES",
]
