"""
Workflow Template Packaging System.

Provides a structured format for packaging, versioning, and distributing
workflow templates with metadata, documentation, and dependencies.

Usage:
    from aragora.workflow.templates.package import TemplatePackage, create_package

    # Create a package from an existing template
    package = create_package(
        template=CONTRACT_REVIEW_TEMPLATE,
        version="1.0.0",
        author="Aragora Team",
    )

    # Export to file
    package.save("contract_review.pkg.json")

    # Load from file
    loaded = TemplatePackage.load("contract_review.pkg.json")
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


class TemplateStatus(str, Enum):
    """Status of a template package."""

    DRAFT = "draft"  # Work in progress
    STABLE = "stable"  # Ready for production use
    DEPRECATED = "deprecated"  # Should migrate to successor
    ARCHIVED = "archived"  # No longer maintained


class TemplateCategory(str, Enum):
    """Template category for organization."""

    LEGAL = "legal"
    HEALTHCARE = "healthcare"
    CODE = "code"
    ACCOUNTING = "accounting"
    AI_ML = "ai_ml"
    DEVOPS = "devops"
    PRODUCT = "product"
    GENERAL = "general"
    COMPLIANCE = "compliance"
    FINANCE = "finance"
    CUSTOM = "custom"
    # SME-specific categories
    SME = "sme"
    QUICKSTART = "quickstart"
    RETAIL = "retail"


@dataclass
class TemplateAuthor:
    """Template author information."""

    name: str
    email: Optional[str] = None
    organization: Optional[str] = None
    url: Optional[str] = None


@dataclass
class TemplateDependency:
    """Dependency on a step type or other template."""

    name: str
    type: str  # "step_type", "template", "agent"
    required: bool = True
    version: Optional[str] = None


@dataclass
class TemplateMetadata:
    """Extended metadata for a template."""

    # Core identifiers
    id: str
    name: str
    version: str

    # Description
    description: str = ""
    long_description: str = ""  # Markdown content

    # Classification
    category: TemplateCategory = TemplateCategory.GENERAL
    tags: list[str] = field(default_factory=list)
    status: TemplateStatus = TemplateStatus.STABLE

    # Authorship
    author: Optional[TemplateAuthor] = None
    contributors: list[TemplateAuthor] = field(default_factory=list)
    license: str = "MIT"

    # Version management
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    successor: Optional[str] = None  # Template ID of replacement

    # Dependencies
    dependencies: list[TemplateDependency] = field(default_factory=list)
    min_aragora_version: Optional[str] = None

    # Usage hints
    estimated_duration: Optional[str] = None  # e.g., "5-10 minutes"
    complexity: str = "medium"  # low, medium, high
    recommended_agents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        if result.get("category"):
            result["category"] = result["category"].value
        if result.get("status"):
            result["status"] = result["status"].value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemplateMetadata":
        """Create from dictionary."""
        # Convert enums
        if "category" in data and isinstance(data["category"], str):
            data["category"] = TemplateCategory(data["category"])
        if "status" in data and isinstance(data["status"], str):
            data["status"] = TemplateStatus(data["status"])

        # Convert nested objects
        if "author" in data and isinstance(data["author"], dict):
            data["author"] = TemplateAuthor(**data["author"])
        if "contributors" in data:
            data["contributors"] = [
                TemplateAuthor(**c) if isinstance(c, dict) else c for c in data["contributors"]
            ]
        if "dependencies" in data:
            data["dependencies"] = [
                TemplateDependency(**d) if isinstance(d, dict) else d for d in data["dependencies"]
            ]

        return cls(**data)


@dataclass
class TemplatePackage:
    """
    A packaged workflow template with metadata and documentation.

    Combines a workflow definition with versioning, authorship,
    dependencies, and documentation for distribution.
    """

    # Package metadata
    metadata: TemplateMetadata

    # The actual workflow template
    template: dict[str, Any]

    # Documentation
    readme: str = ""  # Markdown documentation
    changelog: str = ""  # Version history
    examples: list[dict[str, Any]] = field(default_factory=list)

    # Package integrity
    checksum: Optional[str] = None

    def __post_init__(self) -> None:
        """Compute checksum if not provided."""
        if self.checksum is None:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute SHA-256 checksum of template content."""
        content = json.dumps(self.template, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def verify_checksum(self) -> bool:
        """Verify template integrity."""
        return self.checksum == self._compute_checksum()

    @property
    def id(self) -> str:
        """Get template ID."""
        return self.metadata.id

    @property
    def version(self) -> str:
        """Get template version."""
        return self.metadata.version

    @property
    def is_deprecated(self) -> bool:
        """Check if template is deprecated."""
        return self.metadata.status == TemplateStatus.DEPRECATED

    @property
    def is_stable(self) -> bool:
        """Check if template is stable."""
        return self.metadata.status == TemplateStatus.STABLE

    def to_dict(self) -> dict[str, Any]:
        """Convert package to dictionary."""
        return {
            "package_version": "1.0",
            "metadata": self.metadata.to_dict(),
            "template": self.template,
            "readme": self.readme,
            "changelog": self.changelog,
            "examples": self.examples,
            "checksum": self.checksum,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert package to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemplatePackage":
        """Create package from dictionary."""
        metadata = TemplateMetadata.from_dict(data["metadata"])
        return cls(
            metadata=metadata,
            template=data["template"],
            readme=data.get("readme", ""),
            changelog=data.get("changelog", ""),
            examples=data.get("examples", []),
            checksum=data.get("checksum"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "TemplatePackage":
        """Create package from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, path: Union[str, Path]) -> Path:
        """Save package to file."""
        path = Path(path)
        path.write_text(self.to_json())
        logger.info(f"Saved template package to {path}")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TemplatePackage":
        """Load package from file."""
        path = Path(path)
        data = json.loads(path.read_text())
        return cls.from_dict(data)


def create_package(
    template: dict[str, Any],
    version: str = "1.0.0",
    author: Optional[Union[str, TemplateAuthor]] = None,
    description: Optional[str] = None,
    category: Optional[Union[str, TemplateCategory]] = None,
    tags: Optional[list[str]] = None,
    readme: str = "",
    **kwargs: Any,
) -> TemplatePackage:
    """
    Create a TemplatePackage from an existing template definition.

    Args:
        template: The workflow template dictionary
        version: Semantic version string
        author: Author name or TemplateAuthor object
        description: Short description
        category: Template category
        tags: List of tags
        readme: Markdown documentation
        **kwargs: Additional metadata fields

    Returns:
        TemplatePackage instance
    """
    # Extract info from template
    template_id = template.get("id") or template.get("name", "unknown").lower().replace(" ", "-")
    template_name = template.get("name", template_id)
    template_desc = description or template.get("description", "")

    # Normalize author
    if isinstance(author, str):
        author = TemplateAuthor(name=author)

    # Normalize category
    if isinstance(category, str):
        try:
            category = TemplateCategory(category)
        except ValueError:
            category = TemplateCategory.CUSTOM
    elif category is None:
        # Try to infer from template
        cat_str = template.get("category", "general")
        try:
            category = TemplateCategory(cat_str)
        except ValueError:
            category = TemplateCategory.CUSTOM

    # Build metadata
    metadata = TemplateMetadata(
        id=template_id,
        name=template_name,
        version=version,
        description=template_desc,
        category=category,
        tags=tags or template.get("tags", []),
        author=author,
        **kwargs,
    )

    return TemplatePackage(
        metadata=metadata,
        template=template,
        readme=readme,
    )


def package_all_templates() -> dict[str, TemplatePackage]:
    """
    Package all registered workflow templates.

    Returns:
        Dictionary mapping template IDs to packages
    """
    from aragora.workflow.templates import WORKFLOW_TEMPLATES

    packages = {}
    for template_id, template in WORKFLOW_TEMPLATES.items():
        category, name = template_id.split("/") if "/" in template_id else ("general", template_id)
        package = create_package(
            template=template,
            version="1.0.0",
            author=TemplateAuthor(name="Aragora Team", organization="Aragora"),
            category=category,
        )
        packages[template_id] = package

    return packages


# Template registry with versioning support
_template_registry: dict[str, list[TemplatePackage]] = {}


def register_package(package: TemplatePackage) -> None:
    """Register a template package."""
    template_id = package.metadata.id
    if template_id not in _template_registry:
        _template_registry[template_id] = []
    _template_registry[template_id].append(package)
    # Sort by version (newest first)
    _template_registry[template_id].sort(
        key=lambda p: p.metadata.version,
        reverse=True,
    )


def get_package(
    template_id: str,
    version: Optional[str] = None,
) -> Optional[TemplatePackage]:
    """
    Get a template package by ID and optionally version.

    Args:
        template_id: Template identifier
        version: Specific version (defaults to latest)

    Returns:
        TemplatePackage or None
    """
    packages = _template_registry.get(template_id, [])
    if not packages:
        return None

    if version is None:
        # Return latest stable version
        for pkg in packages:
            if pkg.is_stable:
                return pkg
        return packages[0]  # Fall back to latest

    # Find specific version
    for pkg in packages:
        if pkg.version == version:
            return pkg

    return None


def list_packages(
    category: Optional[Union[str, TemplateCategory]] = None,
    tags: Optional[list[str]] = None,
    include_deprecated: bool = False,
) -> list[TemplatePackage]:
    """
    List available template packages.

    Args:
        category: Filter by category
        tags: Filter by tags (any match)
        include_deprecated: Include deprecated packages

    Returns:
        List of matching packages
    """
    results = []

    for packages in _template_registry.values():
        if not packages:
            continue

        # Get latest version
        pkg = packages[0]

        # Filter by status
        if not include_deprecated and pkg.is_deprecated:
            continue

        # Filter by category
        if category:
            if isinstance(category, str):
                try:
                    category = TemplateCategory(category)
                except ValueError:
                    pass
            if isinstance(category, TemplateCategory) and pkg.metadata.category != category:
                continue
            elif isinstance(category, str) and pkg.metadata.category.value != category:
                continue

        # Filter by tags
        if tags:
            pkg_tags = set(pkg.metadata.tags)
            if not any(t in pkg_tags for t in tags):
                continue

        results.append(pkg)

    return results


__all__ = [
    "TemplatePackage",
    "TemplateMetadata",
    "TemplateAuthor",
    "TemplateDependency",
    "TemplateStatus",
    "TemplateCategory",
    "create_package",
    "package_all_templates",
    "register_package",
    "get_package",
    "list_packages",
]
