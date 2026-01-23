"""
Deliberation Templates Module.

Provides pre-built deliberation patterns for common use cases:
- Code review
- Contract review
- Architecture decisions
- Compliance checks
- And more vertical-specific templates

Templates can be:
- Used directly via the orchestration API
- Customized with overrides
- Extended with new templates
- Loaded from YAML files
"""

from aragora.deliberation.templates.base import (
    DeliberationTemplate,
    OutputFormat,
    TeamStrategy,
    TemplateCategory,
)
from aragora.deliberation.templates.registry import (
    TemplateRegistry,
    get_template,
    list_templates,
    register_template,
    load_templates_from_yaml,
)
from aragora.deliberation.templates.builtins import BUILTIN_TEMPLATES

__all__ = [
    # Base classes
    "DeliberationTemplate",
    "OutputFormat",
    "TeamStrategy",
    "TemplateCategory",
    # Registry functions
    "TemplateRegistry",
    "get_template",
    "list_templates",
    "register_template",
    "load_templates_from_yaml",
    # Built-in templates
    "BUILTIN_TEMPLATES",
]
