"""
Workflow Template Loader.

Loads workflow templates from YAML files and registers them with the workflow system.
Supports enterprise verticals with industry-specific templates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from aragora.workflow.types import WorkflowDefinition, WorkflowCategory

logger = logging.getLogger(__name__)


# Template directory (relative to this file)
TEMPLATES_DIR = Path(__file__).parent / "templates"


# Category to directory mapping
# Note: Multiple directories can map to same category
CATEGORY_DIRS: Dict[WorkflowCategory, str] = {
    WorkflowCategory.GENERAL: "general",
    WorkflowCategory.LEGAL: "legal",
    WorkflowCategory.HEALTHCARE: "healthcare",
    WorkflowCategory.FINANCE: "accounting",  # Finance covers accounting
    WorkflowCategory.CODE: "software",
    WorkflowCategory.ACADEMIC: "academic",
    WorkflowCategory.COMPLIANCE: "regulatory",
}

# Additional directories that also contain templates (mapped to categories)
ADDITIONAL_TEMPLATE_DIRS: Dict[str, WorkflowCategory] = {
    "finance": WorkflowCategory.FINANCE,  # Investment/trading templates
}


class TemplateLoader:
    """
    Loads workflow templates from YAML files.

    Templates are organized by vertical:
        templates/
            legal/
                contract_review.yaml
                due_diligence.yaml
            healthcare/
                clinical_review.yaml
                hipaa_compliance.yaml
            ...
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Initialize template loader.

        Args:
            templates_dir: Directory containing template YAML files
        """
        self._templates_dir = templates_dir or TEMPLATES_DIR
        self._templates: Dict[str, WorkflowDefinition] = {}
        self._loaded = False

    def load_all(self) -> Dict[str, WorkflowDefinition]:
        """
        Load all templates from the templates directory.

        Returns:
            Dictionary of template_id -> WorkflowDefinition
        """
        if self._loaded:
            return self._templates

        if not self._templates_dir.exists():
            logger.warning(f"Templates directory not found: {self._templates_dir}")
            return {}

        # Load templates from each category subdirectory
        for category, dir_name in CATEGORY_DIRS.items():
            category_dir = self._templates_dir / dir_name
            if category_dir.exists():
                self._load_category(category_dir, category)

        # Load from additional template directories
        for dir_name, category in ADDITIONAL_TEMPLATE_DIRS.items():
            category_dir = self._templates_dir / dir_name
            if category_dir.exists():
                self._load_category(category_dir, category)

        # Also load any templates in the root templates directory
        for yaml_file in self._templates_dir.glob("*.yaml"):
            self._load_template_file(yaml_file)

        self._loaded = True
        logger.info(f"Loaded {len(self._templates)} workflow templates")
        return self._templates

    def _load_category(self, category_dir: Path, category: WorkflowCategory) -> None:
        """Load all templates from a category directory."""
        for yaml_file in category_dir.glob("*.yaml"):
            template = self._load_template_file(yaml_file)
            if template:
                # Ensure category is set correctly
                template.category = category
                template.is_template = True
                self._templates[template.id] = template

    def _load_template_file(self, yaml_file: Path) -> Optional[WorkflowDefinition]:
        """Load a single template from a YAML file."""
        try:
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)

            if not data:
                logger.warning(f"Empty template file: {yaml_file}")
                return None

            # Generate ID from filename if not present
            if "id" not in data:
                data["id"] = f"template_{yaml_file.stem}"

            # Mark as template
            data["is_template"] = True

            template = WorkflowDefinition.from_dict(data)
            self._templates[template.id] = template

            logger.debug(f"Loaded template: {template.id} from {yaml_file}")
            return template

        except yaml.YAMLError as e:
            logger.error(f"YAML error in {yaml_file}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load template {yaml_file}: {e}")
            return None

    def get_template(self, template_id: str) -> Optional[WorkflowDefinition]:
        """Get a template by ID."""
        if not self._loaded:
            self.load_all()
        return self._templates.get(template_id)

    def list_templates(
        self,
        category: Optional[WorkflowCategory] = None,
        tags: Optional[List[str]] = None,
    ) -> List[WorkflowDefinition]:
        """
        List templates with optional filtering.

        Args:
            category: Filter by category
            tags: Filter by tags (any match)

        Returns:
            List of matching templates
        """
        if not self._loaded:
            self.load_all()

        templates = list(self._templates.values())

        if category:
            templates = [t for t in templates if t.category == category]

        if tags:
            templates = [t for t in templates if any(tag in t.tags for tag in tags)]

        return templates

    def reload(self) -> Dict[str, WorkflowDefinition]:
        """Force reload all templates."""
        self._loaded = False
        self._templates.clear()
        return self.load_all()


# Global loader instance
_loader: Optional[TemplateLoader] = None


def get_template_loader() -> TemplateLoader:
    """Get or create the global template loader."""
    global _loader
    if _loader is None:
        _loader = TemplateLoader()
    return _loader


def load_templates() -> Dict[str, WorkflowDefinition]:
    """Load all workflow templates."""
    return get_template_loader().load_all()


def get_template(template_id: str) -> Optional[WorkflowDefinition]:
    """Get a template by ID."""
    return get_template_loader().get_template(template_id)


def list_templates(
    category: Optional[WorkflowCategory] = None,
    tags: Optional[List[str]] = None,
) -> List[WorkflowDefinition]:
    """List templates with optional filtering."""
    return get_template_loader().list_templates(category, tags)


__all__ = [
    "TemplateLoader",
    "get_template_loader",
    "load_templates",
    "get_template",
    "list_templates",
    "TEMPLATES_DIR",
]
