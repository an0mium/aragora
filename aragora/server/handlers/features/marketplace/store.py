"""Marketplace in-memory storage and template discovery/loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from .circuit_breaker import (
    _get_marketplace_circuit_breaker,
    _reset_circuit_breaker,
)
from .models import (
    TemplateCategory,
    TemplateDeployment,
    TemplateMetadata,
    TemplateRating,
)

logger = logging.getLogger(__name__)

# =============================================================================
# In-Memory Storage
# =============================================================================

# Template cache: template_id -> TemplateMetadata
_templates_cache: dict[str, TemplateMetadata] = {}

# Deployments: tenant_id -> deployment_id -> TemplateDeployment
_deployments: dict[str, dict[str, TemplateDeployment]] = {}

# Ratings: template_id -> list[TemplateRating]
_ratings: dict[str, list[TemplateRating]] = {}

# Download counts: template_id -> count
_download_counts: dict[str, int] = {}


def _clear_marketplace_state() -> None:
    """Clear all marketplace state (for testing)."""
    global _templates_cache
    _templates_cache.clear()
    _deployments.clear()
    _ratings.clear()
    _download_counts.clear()
    _reset_circuit_breaker()


def _clear_marketplace_components() -> None:
    """Compatibility wrapper to clear marketplace caches and circuit breaker."""
    _clear_marketplace_state()


# =============================================================================
# Template Discovery
# =============================================================================


def _get_templates_dir() -> Path:
    """Get the workflow templates directory."""
    return Path(__file__).parent.parent.parent.parent.parent / "workflow" / "templates"


def _load_templates() -> dict[str, TemplateMetadata]:
    """Load all templates from the templates directory.

    Uses circuit breaker to handle persistent template loading failures gracefully.
    Returns cached templates when circuit is open.
    """
    global _templates_cache

    if _templates_cache:
        return _templates_cache

    cb = _get_marketplace_circuit_breaker()

    if not cb.is_allowed():
        logger.warning("Marketplace circuit breaker is open, returning cached templates")
        return _templates_cache

    try:
        templates_dir = _get_templates_dir()
        if not templates_dir.exists():
            logger.warning(f"Templates directory not found: {templates_dir}")
            cb.record_success()
            return _templates_cache

        # Find all YAML templates
        for yaml_file in templates_dir.rglob("*.yaml"):
            try:
                template = _parse_template_file(yaml_file)
                if template:
                    _templates_cache[template.id] = template
            except Exception as e:
                logger.warning(f"Failed to parse template {yaml_file}: {e}")

        logger.info(f"Loaded {len(_templates_cache)} templates from {templates_dir}")
        cb.record_success()
        return _templates_cache

    except Exception as e:
        logger.exception(f"Error loading templates: {e}")
        cb.record_failure()
        return _templates_cache


def _parse_template_file(file_path: Path) -> TemplateMetadata | None:
    """Parse a YAML template file into metadata."""
    try:
        with open(file_path) as f:
            data = yaml.safe_load(f)

        if not data or not data.get("is_template", False):
            return None

        # Parse category
        category_str = data.get("category", "general").lower()
        try:
            category = TemplateCategory(category_str)
        except ValueError:
            category = TemplateCategory.GENERAL

        # Count steps and detect features
        steps = data.get("steps", [])
        steps_count = len(steps)
        has_debate = any(s.get("step_type") == "debate" for s in steps)
        has_human_checkpoint = any(s.get("step_type") == "human_checkpoint" for s in steps)

        # Estimate duration based on features
        if has_human_checkpoint:
            estimated_duration = "hours to days"
        elif has_debate:
            estimated_duration = "minutes to hours"
        elif steps_count > 5:
            estimated_duration = "1-5 minutes"
        else:
            estimated_duration = "< 1 minute"

        return TemplateMetadata(
            id=data.get("id", file_path.stem),
            name=data.get("name", file_path.stem.replace("_", " ").title()),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            category=category,
            tags=data.get("tags", []),
            icon=data.get("icon", "document"),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            steps_count=steps_count,
            has_debate=has_debate,
            has_human_checkpoint=has_human_checkpoint,
            estimated_duration=estimated_duration,
            file_path=str(file_path),
        )

    except Exception as e:
        logger.warning(f"Error parsing template {file_path}: {e}")
        return None


def _get_full_template(template_id: str) -> Optional[dict[str, Any]]:
    """Load the full template content."""
    templates = _load_templates()
    meta = templates.get(template_id)

    if not meta or not meta.file_path:
        return None

    try:
        with open(meta.file_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Error loading template {template_id}: {e}")
        return None


def _get_tenant_deployments(tenant_id: str) -> dict[str, TemplateDeployment]:
    """Get deployments for a tenant."""
    if tenant_id not in _deployments:
        _deployments[tenant_id] = {}
    return _deployments[tenant_id]


def get_ratings() -> dict[str, list[TemplateRating]]:
    """Get the ratings storage dict."""
    return _ratings


def get_download_counts() -> dict[str, int]:
    """Get the download counts storage dict."""
    return _download_counts


def get_deployments() -> dict[str, dict[str, TemplateDeployment]]:
    """Get the deployments storage dict."""
    return _deployments
