"""
Template Registry for managing deliberation templates.

Provides a centralized registry for template discovery, lookup,
and management. Templates can be:
- Built-in (defined in code)
- Loaded from YAML files
- Registered at runtime
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from aragora.deliberation.templates.base import (
    DeliberationTemplate,
    TemplateCategory,
)

logger = logging.getLogger(__name__)


class TemplateRegistry:
    """
    Central registry for deliberation templates.

    Supports:
    - Template registration and lookup
    - Filtering by category and tags
    - Loading from YAML files
    - Listing with pagination
    """

    def __init__(self) -> None:
        self._templates: Dict[str, DeliberationTemplate] = {}
        self._initialized = False

    def register(self, template: DeliberationTemplate) -> None:
        """Register a template."""
        if template.name in self._templates:
            logger.warning(f"Overwriting existing template: {template.name}")
        self._templates[template.name] = template
        logger.debug(f"Registered template: {template.name}")

    def unregister(self, name: str) -> bool:
        """Unregister a template by name."""
        if name in self._templates:
            del self._templates[name]
            return True
        return False

    def get(self, name: str) -> Optional[DeliberationTemplate]:
        """Get a template by name."""
        self._ensure_initialized()
        return self._templates.get(name)

    def list(
        self,
        category: Optional[TemplateCategory] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[DeliberationTemplate]:
        """
        List templates with optional filtering.

        Args:
            category: Filter by category
            tags: Filter by tags (OR matching)
            search: Search in name and description
            limit: Maximum results to return
            offset: Offset for pagination
        """
        self._ensure_initialized()
        templates = list(self._templates.values())

        # Filter by category
        if category:
            templates = [t for t in templates if t.category == category]

        # Filter by tags (OR matching)
        if tags:
            templates = [t for t in templates if any(tag in t.tags for tag in tags)]

        # Search filter
        if search:
            search_lower = search.lower()
            templates = [
                t
                for t in templates
                if search_lower in t.name.lower() or search_lower in t.description.lower()
            ]

        # Sort by name
        templates.sort(key=lambda t: t.name)

        # Apply pagination
        return templates[offset : offset + limit]

    def count(
        self,
        category: Optional[TemplateCategory] = None,
        tags: Optional[List[str]] = None,
    ) -> int:
        """Count templates with optional filtering."""
        return len(self.list(category=category, tags=tags, limit=10000))

    def categories(self) -> Dict[str, int]:
        """Get template counts by category."""
        self._ensure_initialized()
        counts: Dict[str, int] = {}
        for template in self._templates.values():
            cat = template.category.value
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def load_from_yaml(self, yaml_path: Path) -> int:
        """
        Load templates from a YAML file.

        Returns the number of templates loaded.
        """
        try:
            import yaml

            with open(yaml_path) as f:
                data = yaml.safe_load(f)

            if not data:
                return 0

            templates = data.get("templates", [])
            if isinstance(data, list):
                templates = data

            count = 0
            for template_data in templates:
                try:
                    template = DeliberationTemplate.from_dict(template_data)
                    self.register(template)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to load template from YAML: {e}")

            return count
        except ImportError:
            logger.warning("PyYAML not installed, cannot load YAML templates")
            return 0
        except Exception as e:
            logger.error(f"Failed to load templates from {yaml_path}: {e}")
            return 0

    def load_from_directory(self, directory: Path) -> int:
        """
        Load all YAML templates from a directory.

        Returns the number of templates loaded.
        """
        count = 0
        for yaml_file in directory.glob("*.yaml"):
            count += self.load_from_yaml(yaml_file)
        for yaml_file in directory.glob("*.yml"):
            count += self.load_from_yaml(yaml_file)
        return count

    def _ensure_initialized(self) -> None:
        """Ensure built-in templates are loaded."""
        if not self._initialized:
            from aragora.deliberation.templates.builtins import BUILTIN_TEMPLATES

            for template in BUILTIN_TEMPLATES.values():
                if template.name not in self._templates:
                    self._templates[template.name] = template
            self._initialized = True


# Global registry instance
_global_registry = TemplateRegistry()


def get_template(name: str) -> Optional[DeliberationTemplate]:
    """Get a template from the global registry."""
    return _global_registry.get(name)


def list_templates(
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[DeliberationTemplate]:
    """List templates from the global registry."""
    cat = None
    if category:
        try:
            cat = TemplateCategory(category)
        except ValueError:
            pass
    return _global_registry.list(category=cat, tags=tags, search=search, limit=limit, offset=offset)


def register_template(template: DeliberationTemplate) -> None:
    """Register a template in the global registry."""
    _global_registry.register(template)


def load_templates_from_yaml(path: Path) -> int:
    """Load templates from YAML into the global registry."""
    if path.is_dir():
        return _global_registry.load_from_directory(path)
    return _global_registry.load_from_yaml(path)


def get_template_dict(name: str) -> Optional[Dict[str, Any]]:
    """Get a template as a dictionary."""
    template = get_template(name)
    return template.to_dict() if template else None
