"""
SME Quick-Start Workflow Bundles.

Pre-packaged workflow bundles for common SME business processes.
Each bundle is a complete, ready-to-use workflow with multi-agent debates.

Available Bundles:
- month_end_close: Monthly financial close process
- hiring_sprint: Rapid hiring decision workflow
- product_planning: Product/sprint planning cycle
- vendor_refresh: Vendor evaluation and transition
- q_planning: Quarterly business planning

Usage:
    from aragora.workflow.templates.sme.bundles import get_bundle, list_bundles

    # List all available bundles
    bundles = list_bundles()

    # Get a specific bundle
    month_end = get_bundle("month_end_close")

    # Load all bundles
    all_bundles = load_all_bundles()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Bundle directory
BUNDLE_DIR = Path(__file__).parent

# Bundle metadata
BUNDLE_INFO = {
    "month_end_close": {
        "name": "Month-End Close Bundle",
        "description": "Complete month-end financial close workflow",
        "icon": "calendar-check",
        "estimated_duration": "15-30 minutes",
        "complexity": "medium",
        "tags": ["finance", "monthly", "close", "budget", "reporting"],
    },
    "hiring_sprint": {
        "name": "Hiring Sprint Bundle",
        "description": "Accelerated hiring decision workflow",
        "icon": "users",
        "estimated_duration": "20-40 minutes",
        "complexity": "medium",
        "tags": ["hiring", "hr", "recruitment", "team"],
    },
    "product_planning": {
        "name": "Product Planning Bundle",
        "description": "End-to-end product planning cycle",
        "icon": "layers",
        "estimated_duration": "25-45 minutes",
        "complexity": "medium",
        "tags": ["product", "planning", "sprint", "agile", "prioritization"],
    },
    "vendor_refresh": {
        "name": "Vendor Refresh Bundle",
        "description": "Vendor evaluation and transition workflow",
        "icon": "refresh-cw",
        "estimated_duration": "30-50 minutes",
        "complexity": "high",
        "tags": ["vendor", "procurement", "evaluation", "contract"],
    },
    "q_planning": {
        "name": "Quarterly Planning Bundle",
        "description": "Comprehensive quarterly business planning",
        "icon": "target",
        "estimated_duration": "40-60 minutes",
        "complexity": "high",
        "tags": ["planning", "quarterly", "okr", "budget", "strategy"],
    },
}


def list_bundles() -> list[dict[str, Any]]:
    """
    List all available SME bundles with metadata.

    Returns:
        List of bundle metadata dictionaries.

    Example:
        >>> bundles = list_bundles()
        >>> for b in bundles:
        ...     print(f"{b['id']}: {b['name']}")
    """
    bundles = []
    for bundle_id, info in BUNDLE_INFO.items():
        bundles.append(
            {
                "id": bundle_id,
                **info,
            }
        )
    return bundles


def get_bundle(bundle_id: str) -> dict[str, Any] | None:
    """
    Load a specific bundle by ID.

    Args:
        bundle_id: Bundle identifier (e.g., "month_end_close")

    Returns:
        Workflow definition dict, or None if not found.

    Example:
        >>> bundle = get_bundle("month_end_close")
        >>> print(bundle["name"])
        Month-End Close Bundle
    """
    if bundle_id not in BUNDLE_INFO:
        logger.warning("Unknown bundle: %s", bundle_id)
        return None

    yaml_path = BUNDLE_DIR / f"{bundle_id}.yaml"
    if not yaml_path.exists():
        logger.error("Bundle file not found: %s", yaml_path)
        return None

    try:
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    except (yaml.YAMLError, OSError, ValueError, TypeError) as e:
        logger.error("Failed to load bundle %s: %s", bundle_id, e)
        return None


def load_all_bundles() -> dict[str, dict[str, Any]]:
    """
    Load all available bundles.

    Returns:
        Dictionary mapping bundle_id to workflow definition.

    Example:
        >>> bundles = load_all_bundles()
        >>> print(list(bundles.keys()))
        ['month_end_close', 'hiring_sprint', ...]
    """
    bundles = {}
    for bundle_id in BUNDLE_INFO:
        bundle = get_bundle(bundle_id)
        if bundle:
            bundles[bundle_id] = bundle
    return bundles


def get_bundle_for_use_case(use_case: str) -> dict[str, Any] | None:
    """
    Get recommended bundle for a use case.

    Args:
        use_case: Description of what the user wants to do.

    Returns:
        Best matching bundle, or None if no good match.

    Example:
        >>> bundle = get_bundle_for_use_case("close the books for january")
        >>> print(bundle["id"])
        sme_bundle_month_end_close
    """
    use_case_lower = use_case.lower()

    # Simple keyword matching (could be enhanced with semantic search)
    keyword_map = {
        "month_end_close": ["month end", "close", "financial close", "accounting", "variance"],
        "hiring_sprint": ["hire", "hiring", "recruit", "candidate", "interview", "offer"],
        "product_planning": ["product", "sprint", "feature", "backlog", "prioritize", "roadmap"],
        "vendor_refresh": ["vendor", "tool", "software", "evaluation", "procurement", "contract"],
        "q_planning": ["quarterly", "okr", "goal", "budget", "planning", "kpi"],
    }

    best_match = None
    best_score = 0

    for bundle_id, keywords in keyword_map.items():
        score = sum(1 for kw in keywords if kw in use_case_lower)
        if score > best_score:
            best_score = score
            best_match = bundle_id

    if best_match and best_score > 0:
        return get_bundle(best_match)

    return None


__all__ = [
    "BUNDLE_INFO",
    "list_bundles",
    "get_bundle",
    "load_all_bundles",
    "get_bundle_for_use_case",
]
