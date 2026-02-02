"""Marketplace Handler - Template discovery and deployment.

Provides API endpoints for discovering, browsing, and deploying workflow templates
across different industry verticals.

Stability: STABLE
- Circuit breaker pattern for template loading resilience
- Rate limiting on write operations (deploy, rate)
- Comprehensive input validation (IDs, pagination, ratings, config)
- RBAC permission enforcement

Endpoints:
- GET  /api/v1/marketplace/templates         - List all available templates
- GET  /api/v1/marketplace/templates/{id}    - Get template details
- GET  /api/v1/marketplace/categories        - List template categories
- GET  /api/v1/marketplace/search            - Search templates
- POST /api/v1/marketplace/templates/{id}/deploy - Deploy a template
- GET  /api/v1/marketplace/deployments       - List deployed templates
- GET  /api/v1/marketplace/popular           - Get popular templates
- POST /api/v1/marketplace/templates/{id}/rate - Rate a template
- GET  /api/v1/marketplace/demo              - Get demo marketplace data
- GET  /api/v1/marketplace/status            - Circuit breaker and health status
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import yaml

from ..base import (
    HandlerResult,
    error_response,
    json_response,
)
from ..utils import parse_json_body
from ..utils.rate_limit import rate_limit
from aragora.rbac.decorators import require_permission
from aragora.server.validation.core import sanitize_string

logger = logging.getLogger(__name__)

# =============================================================================
# Constants for Input Validation
# =============================================================================

# Safe pattern for template IDs and deployment IDs: alphanumeric, hyphens, underscores
SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$")

MAX_TEMPLATE_NAME_LENGTH = 200
MAX_DEPLOYMENT_NAME_LENGTH = 200
MAX_REVIEW_LENGTH = 2000
MAX_SEARCH_QUERY_LENGTH = 500
MAX_CONFIG_KEYS = 50
MAX_CONFIG_SIZE = MAX_CONFIG_KEYS
MIN_RATING = 1
MAX_RATING = 5
DEFAULT_LIMIT = 50
MIN_LIMIT = 1
MAX_LIMIT = 200
MAX_OFFSET = 10000

# =============================================================================
# Input Validation Functions
# =============================================================================


def _validate_id(value: str, label: str = "ID") -> tuple[bool, str]:
    """Validate an ID string (template_id or deployment_id).

    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.
    """
    if not value or not isinstance(value, str):
        return False, f"{label} is required"
    if len(value) > 128:
        return False, f"{label} must be at most 128 characters"
    if not SAFE_ID_PATTERN.match(value):
        return False, f"{label} contains invalid characters"
    return True, ""


def _validate_pagination(query: dict[str, Any]) -> tuple[int, int, str]:
    """Validate and clamp pagination parameters.

    Returns:
        Tuple of (limit, offset, error_message). error_message is empty if valid.
    """
    try:
        limit = int(query.get("limit", DEFAULT_LIMIT))
    except (ValueError, TypeError):
        return DEFAULT_LIMIT, 0, "limit must be an integer"

    try:
        offset = int(query.get("offset", 0))
    except (ValueError, TypeError):
        return DEFAULT_LIMIT, 0, "offset must be an integer"

    limit, offset = _clamp_pagination(limit, offset)

    return limit, offset, ""


def _validate_rating_value(value: Any) -> tuple[bool, int, str]:
    """Validate a rating value.

    Returns:
        Tuple of (is_valid, sanitized_value, error_message).
    """
    if value is None:
        return False, 0, "Rating is required"
    if not isinstance(value, int):
        return False, 0, "Rating must be an integer"
    if value < MIN_RATING or value > MAX_RATING:
        return False, 0, f"Rating must be between {MIN_RATING} and {MAX_RATING}"
    return True, value, ""


def _validate_review_internal(value: Any) -> tuple[bool, str | None, str]:
    """Validate a review string.

    Returns:
        Tuple of (is_valid, sanitized_value, error_message).
    """
    if value is None:
        return True, None, ""
    if not isinstance(value, str):
        return False, None, "Review must be a string"
    if len(value) > MAX_REVIEW_LENGTH:
        return False, None, f"Review must be at most {MAX_REVIEW_LENGTH} characters"
    return True, sanitize_string(value), ""


def _validate_deployment_name_internal(value: Any, fallback: str) -> tuple[bool, str, str]:
    """Validate a deployment name.

    Returns:
        Tuple of (is_valid, sanitized_value, error_message).
    """
    if value is None:
        return True, fallback, ""
    if not isinstance(value, str):
        return False, "", "Deployment name must be a string"
    if len(value) > MAX_DEPLOYMENT_NAME_LENGTH:
        return False, "", f"Deployment name must be at most {MAX_DEPLOYMENT_NAME_LENGTH} characters"
    sanitized = sanitize_string(value, MAX_DEPLOYMENT_NAME_LENGTH)
    if not sanitized:
        return True, fallback, ""
    return True, sanitized, ""


def _validate_config(value: Any) -> tuple[bool, dict[str, Any], str]:
    """Validate deployment config.

    Returns:
        Tuple of (is_valid, sanitized_value, error_message).
    """
    if value is None:
        return True, {}, ""
    if not isinstance(value, dict):
        return False, {}, "Config must be a dictionary"
    if len(value) > MAX_CONFIG_SIZE:
        return False, {}, f"Config must have at most {MAX_CONFIG_SIZE} keys"
    return True, value, ""


def _validate_search_query(value: Any) -> tuple[bool, str, str]:
    """Validate a search query string.

    Returns:
        Tuple of (is_valid, sanitized_value, error_message).
    """
    if value is None or value == "":
        return True, "", ""
    if not isinstance(value, str):
        return False, "", "Search query must be a string"
    if len(value) > MAX_SEARCH_QUERY_LENGTH:
        return False, "", f"Search query must be at most {MAX_SEARCH_QUERY_LENGTH} characters"
    return True, sanitize_string(value, MAX_SEARCH_QUERY_LENGTH).lower(), ""


def _validate_category_filter(value: Any) -> tuple[bool, str | None, str]:
    """Validate a category filter value.

    Returns:
        Tuple of (is_valid, sanitized_value, error_message).
    """
    valid, category, err = _validate_category(value)
    if not valid:
        return False, None, err
    return True, category.value if category else None, ""


# =============================================================================
# Enums and Data Classes
# =============================================================================


class TemplateCategory(Enum):
    """Template categories for industry verticals."""

    ACCOUNTING = "accounting"
    LEGAL = "legal"
    HEALTHCARE = "healthcare"
    SOFTWARE = "software"
    REGULATORY = "regulatory"
    ACADEMIC = "academic"
    FINANCE = "finance"
    GENERAL = "general"
    DEVOPS = "devops"
    MARKETING = "marketing"


class DeploymentStatus(Enum):
    """Deployment status."""

    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class TemplateMetadata:
    """Template metadata for marketplace listing."""

    id: str
    name: str
    description: str
    version: str
    category: TemplateCategory
    tags: list[str] = field(default_factory=list)
    icon: str = "document"
    author: str = "Aragora"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    downloads: int = 0
    rating: float = 0.0
    rating_count: int = 0
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    steps_count: int = 0
    has_debate: bool = False
    has_human_checkpoint: bool = False
    estimated_duration: str = "varies"
    file_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category.value,
            "tags": self.tags,
            "icon": self.icon,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "downloads": self.downloads,
            "rating": self.rating,
            "rating_count": self.rating_count,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "steps_count": self.steps_count,
            "has_debate": self.has_debate,
            "has_human_checkpoint": self.has_human_checkpoint,
            "estimated_duration": self.estimated_duration,
        }


@dataclass
class TemplateDeployment:
    """Record of a deployed template."""

    id: str
    template_id: str
    tenant_id: str
    name: str
    status: DeploymentStatus
    config: dict[str, Any] = field(default_factory=dict)
    deployed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_run: datetime | None = None
    run_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "template_id": self.template_id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "status": self.status.value,
            "config": self.config,
            "deployed_at": self.deployed_at.isoformat(),
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "run_count": self.run_count,
        }


@dataclass
class TemplateRating:
    """Template rating from a user."""

    id: str
    template_id: str
    tenant_id: str
    user_id: str
    rating: int  # 1-5
    review: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "template_id": self.template_id,
            "rating": self.rating,
            "review": self.review,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# Circuit Breaker
# =============================================================================


class MarketplaceCircuitBreaker:
    """Circuit breaker for marketplace template loading operations.

    Prevents cascading failures when template directory or YAML parsing is unavailable.
    Uses a simple state machine: CLOSED -> OPEN -> HALF_OPEN -> CLOSED.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls

        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Get current circuit state."""
        with self._lock:
            return self._check_state()

    def _check_state(self) -> str:
        """Check and potentially transition state (must hold lock)."""
        if self._state == self.OPEN:
            if (
                self._last_failure_time is not None
                and time.time() - self._last_failure_time >= self.cooldown_seconds
            ):
                self._state = self.HALF_OPEN
                self._half_open_calls = 0
                logger.info("Marketplace circuit breaker transitioning to HALF_OPEN")
        return self._state

    def can_proceed(self) -> bool:
        """Check if a call can proceed."""
        with self._lock:
            state = self._check_state()
            if state == self.CLOSED:
                return True
            elif state == self.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            else:
                return False

    def is_allowed(self) -> bool:
        """Alias for can_proceed (public API)."""
        return self.can_proceed()

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    self._state = self.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Marketplace circuit breaker closed after successful recovery")
            elif self._state == self.CLOSED:
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == self.HALF_OPEN:
                self._state = self.OPEN
                self._success_count = 0
                logger.warning("Marketplace circuit breaker reopened after failure in HALF_OPEN")
            elif self._state == self.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = self.OPEN
                    logger.warning(
                        f"Marketplace circuit breaker opened after {self._failure_count} failures"
                    )

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        with self._lock:
            return {
                "state": self._check_state(),
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "cooldown_seconds": self.cooldown_seconds,
                "last_failure_time": self._last_failure_time,
            }

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = self.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0


# Global circuit breaker instance
_circuit_breaker: MarketplaceCircuitBreaker | None = None
_circuit_breaker_lock = threading.Lock()


def _get_circuit_breaker() -> MarketplaceCircuitBreaker:
    """Get or create the marketplace circuit breaker."""
    global _circuit_breaker
    with _circuit_breaker_lock:
        if _circuit_breaker is None:
            _circuit_breaker = MarketplaceCircuitBreaker()
        return _circuit_breaker


def _get_marketplace_circuit_breaker() -> MarketplaceCircuitBreaker:
    """Public accessor for the marketplace circuit breaker (for testing)."""
    return _get_circuit_breaker()


def get_marketplace_circuit_breaker_status() -> dict[str, Any]:
    """Get the marketplace circuit breaker status."""
    return _get_marketplace_circuit_breaker().get_status()


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
    global _templates_cache, _circuit_breaker
    _templates_cache.clear()
    _deployments.clear()
    _ratings.clear()
    _download_counts.clear()
    with _circuit_breaker_lock:
        _circuit_breaker = None


def _clear_marketplace_components() -> None:
    """Compatibility wrapper to clear marketplace caches and circuit breaker."""
    _clear_marketplace_state()


# =============================================================================
# Template Discovery
# =============================================================================


def _get_templates_dir() -> Path:
    """Get the workflow templates directory."""
    return Path(__file__).parent.parent.parent.parent / "workflow" / "templates"


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


# =============================================================================
# Category Information
# =============================================================================

CATEGORY_INFO = {
    TemplateCategory.ACCOUNTING: {
        "name": "Accounting & Finance",
        "description": "Invoice processing, expense management, financial audits, and QBO integration",
        "icon": "calculator",
        "color": "#4299e1",
    },
    TemplateCategory.LEGAL: {
        "name": "Legal",
        "description": "Contract review, due diligence, compliance checking, and document analysis",
        "icon": "scale",
        "color": "#9f7aea",
    },
    TemplateCategory.HEALTHCARE: {
        "name": "Healthcare",
        "description": "HIPAA compliance, clinical reviews, patient data processing",
        "icon": "heart",
        "color": "#f56565",
    },
    TemplateCategory.SOFTWARE: {
        "name": "Software Development",
        "description": "Code review, security audits, bug detection, and CI/CD automation",
        "icon": "code",
        "color": "#48bb78",
    },
    TemplateCategory.REGULATORY: {
        "name": "Regulatory Compliance",
        "description": "SOC 2, GDPR, SOX compliance assessments and audit preparation",
        "icon": "shield",
        "color": "#ed8936",
    },
    TemplateCategory.ACADEMIC: {
        "name": "Academic & Research",
        "description": "Citation verification, research validation, peer review workflows",
        "icon": "book",
        "color": "#38b2ac",
    },
    TemplateCategory.FINANCE: {
        "name": "Investment & Finance",
        "description": "Investment analysis, risk assessment, portfolio management",
        "icon": "trending-up",
        "color": "#667eea",
    },
    TemplateCategory.GENERAL: {
        "name": "General",
        "description": "Multi-purpose research and analysis workflows",
        "icon": "folder",
        "color": "#718096",
    },
    TemplateCategory.DEVOPS: {
        "name": "DevOps & IT",
        "description": "Infrastructure automation, incident response, monitoring",
        "icon": "server",
        "color": "#2d3748",
    },
    TemplateCategory.MARKETING: {
        "name": "Marketing",
        "description": "Content strategy, campaign analysis, market research",
        "icon": "megaphone",
        "color": "#d53f8c",
    },
}

# =============================================================================
# Handler
# =============================================================================


class MarketplaceHandler:
    """Handler for marketplace API endpoints.

    Production-ready with:
    - Circuit breaker for template loading resilience
    - Rate limiting on write operations (deploy: 20/min, rate: 10/min)
    - Input validation for IDs, pagination, ratings, config, search queries
    - RBAC permission enforcement
    """

    ROUTES = [
        "/api/v1/marketplace/templates",
        "/api/v1/marketplace/templates/{template_id}",
        "/api/v1/marketplace/templates/{template_id}/deploy",
        "/api/v1/marketplace/templates/{template_id}/rate",
        "/api/v1/marketplace/categories",
        "/api/v1/marketplace/search",
        "/api/v1/marketplace/deployments",
        "/api/v1/marketplace/deployments/{deployment_id}",
        "/api/v1/marketplace/popular",
        "/api/v1/marketplace/demo",
        "/api/v1/marketplace/status",
    ]

    ctx: dict[str, Any]

    def __init__(self, server_context: dict[str, Any] | None = None):
        """Initialize handler with optional server context."""
        self.ctx = server_context if server_context is not None else {}
        # Pre-load templates
        _load_templates()

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the given path."""
        # Check exact routes
        if path in self.ROUTES:
            return True
        # Check template-specific paths with IDs
        if path.startswith("/api/v1/marketplace/templates/"):
            return True
        if path.startswith("/api/v1/marketplace/deployments/"):
            return True
        return False

    @require_permission("marketplace:read")
    async def handle(self, request: Any, path: str, method: str) -> HandlerResult:
        """Route requests to appropriate handler methods."""
        try:
            tenant_id = self._get_tenant_id(request)

            # List templates
            if path == "/api/v1/marketplace/templates" and method == "GET":
                return await self._handle_list_templates(request, tenant_id)

            # List categories
            elif path == "/api/v1/marketplace/categories" and method == "GET":
                return await self._handle_list_categories(request, tenant_id)

            # Search templates
            elif path == "/api/v1/marketplace/search" and method == "GET":
                return await self._handle_search(request, tenant_id)

            # Popular templates
            elif path == "/api/v1/marketplace/popular" and method == "GET":
                return await self._handle_popular(request, tenant_id)

            # List deployments
            elif path == "/api/v1/marketplace/deployments" and method == "GET":
                return await self._handle_list_deployments(request, tenant_id)

            # Demo data
            elif path == "/api/v1/marketplace/demo" and method == "GET":
                return await self._handle_demo(request, tenant_id)

            # Health/status
            elif path == "/api/v1/marketplace/status" and method == "GET":
                return await self._handle_status(request, tenant_id)

            # Template-specific paths
            elif path.startswith("/api/v1/marketplace/templates/"):
                parts = path.split("/")
                if len(parts) >= 6:
                    template_id = parts[5]

                    # Validate template ID
                    valid, err = _validate_template_id(template_id)
                    if not valid:
                        return error_response(err, 400)

                    if len(parts) == 6:
                        if method == "GET":
                            return await self._handle_get_template(request, tenant_id, template_id)

                    elif len(parts) == 7:
                        action = parts[6]
                        if action == "deploy" and method == "POST":
                            return await self._handle_deploy(request, tenant_id, template_id)
                        elif action == "rate" and method == "POST":
                            return await self._handle_rate(request, tenant_id, template_id)

            # Deployment-specific paths
            elif path.startswith("/api/v1/marketplace/deployments/"):
                parts = path.split("/")
                if len(parts) >= 6:
                    deployment_id = parts[5]

                    # Validate deployment ID
                    valid, err = _validate_deployment_id(deployment_id)
                    if not valid:
                        return error_response(err, 400)

                    if len(parts) == 6:
                        if method == "GET":
                            return await self._handle_get_deployment(
                                request, tenant_id, deployment_id
                            )
                        elif method == "DELETE":
                            return await self._handle_delete_deployment(
                                request, tenant_id, deployment_id
                            )

            return error_response("Not found", 404)

        except Exception as e:
            logger.exception(f"Error in marketplace handler: {e}")
            return error_response(f"Internal error: {e}", 500)

    def _get_tenant_id(self, request: Any) -> str:
        """Extract tenant ID from request context."""
        tenant_id = getattr(request, "tenant_id", None)
        if not tenant_id or not isinstance(tenant_id, str):
            return "default"
        # Sanitize tenant ID
        if len(tenant_id) > 128:
            return "default"
        return tenant_id

    async def _get_json_body(self, request: Any) -> dict[str, Any]:
        """Parse JSON body from request."""
        if hasattr(request, "json"):
            body, _err = await parse_json_body(request, context="marketplace._get_json_body")
            return body if body is not None else {}
        return {}

    # =========================================================================
    # List Templates
    # =========================================================================

    async def _handle_list_templates(self, request: Any, tenant_id: str) -> HandlerResult:
        """List all available templates."""
        templates = _load_templates()
        query = getattr(request, "query", {})

        # Validate category filter
        category_filter = query.get("category")
        if category_filter:
            valid, _, err = _validate_category_filter(category_filter)
            if not valid:
                return error_response(err, 400)
            templates = {k: v for k, v in templates.items() if v.category.value == category_filter}

        # Convert to list and sort by downloads
        template_list = sorted(
            templates.values(),
            key=lambda t: t.downloads,
            reverse=True,
        )

        # Validate pagination
        limit, offset, err = _validate_pagination(query)
        if err:
            return error_response(err, 400)

        total = len(template_list)
        template_list = template_list[offset : offset + limit]

        return json_response(
            {
                "templates": [t.to_dict() for t in template_list],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )

    # =========================================================================
    # Get Template Details
    # =========================================================================

    async def _handle_get_template(
        self, request: Any, tenant_id: str, template_id: str
    ) -> HandlerResult:
        """Get detailed template information."""
        templates = _load_templates()
        meta = templates.get(template_id)

        if not meta:
            return error_response("Template not found", 404)

        # Load full template content
        full_template = _get_full_template(template_id)

        # Get ratings
        ratings = _ratings.get(template_id, [])
        recent_ratings = sorted(ratings, key=lambda r: r.created_at, reverse=True)[:5]

        return json_response(
            {
                "template": meta.to_dict(),
                "full_definition": full_template,
                "ratings": {
                    "average": meta.rating,
                    "count": meta.rating_count,
                    "recent": [r.to_dict() for r in recent_ratings],
                },
                "related": self._get_related_templates(meta, templates),
            }
        )

    def _get_related_templates(
        self, template: TemplateMetadata, all_templates: dict[str, TemplateMetadata]
    ) -> list[dict[str, Any]]:
        """Find related templates based on category and tags."""
        related: list[tuple[int, TemplateMetadata]] = []

        for other in all_templates.values():
            if other.id == template.id:
                continue

            # Same category
            if other.category == template.category:
                score = 2
            else:
                score = 0

            # Shared tags
            shared_tags = set(template.tags) & set(other.tags)
            score += len(shared_tags)

            if score > 0:
                related.append((score, other))

        # Sort by score and take top 5
        related.sort(key=lambda x: x[0], reverse=True)
        related_templates = [t.to_dict() for _, t in related]

        if len(related_templates) < 5:
            # Fill remaining slots with popular templates (by downloads/rating)
            related_ids = {t["id"] for t in related_templates}
            fallback = [
                t for t in all_templates.values() if t.id != template.id and t.id not in related_ids
            ]
            fallback.sort(key=lambda t: (t.downloads, t.rating), reverse=True)
            for extra in fallback:
                related_templates.append(extra.to_dict())
                if len(related_templates) >= 5:
                    break

        return related_templates[:5]

    # =========================================================================
    # List Categories
    # =========================================================================

    async def _handle_list_categories(self, request: Any, tenant_id: str) -> HandlerResult:
        """List all template categories with counts."""
        templates = _load_templates()

        # Count templates per category
        category_counts: dict[str, int] = {}
        for template in templates.values():
            cat = template.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        categories = []
        for category in TemplateCategory:
            info = CATEGORY_INFO.get(category, {})
            categories.append(
                {
                    "id": category.value,
                    "name": info.get("name", category.value.title()),
                    "description": info.get("description", ""),
                    "icon": info.get("icon", "folder"),
                    "color": info.get("color", "#718096"),
                    "template_count": category_counts.get(category.value, 0),
                }
            )

        return json_response({"categories": categories})

    # =========================================================================
    # Search Templates
    # =========================================================================

    async def _handle_search(self, request: Any, tenant_id: str) -> HandlerResult:
        """Search templates by query."""
        templates = _load_templates()
        query = getattr(request, "query", {})

        # Validate search query
        raw_query = query.get("q", "")
        valid, search_query, err = _validate_search_query(raw_query)
        if not valid:
            return error_response(err, 400)

        # Validate category filter
        category_filter = query.get("category")
        if category_filter:
            valid, category_filter, err = _validate_category_filter(category_filter)
            if not valid:
                return error_response(err, 400)

        tags_filter = query.get("tags", "").split(",") if query.get("tags") else []
        # Sanitize tags
        tags_filter = [sanitize_string(t.strip(), 100) for t in tags_filter if t.strip()]

        has_debate = query.get("has_debate")
        has_checkpoint = query.get("has_checkpoint")

        results = []
        for template in templates.values():
            # Text search
            if search_query:
                searchable = (
                    f"{template.name} {template.description} {' '.join(template.tags)}".lower()
                )
                if search_query not in searchable:
                    continue

            # Category filter
            if category_filter and template.category.value != category_filter:
                continue

            # Tags filter
            if tags_filter:
                if not any(tag in template.tags for tag in tags_filter):
                    continue

            # Feature filters
            if has_debate == "true" and not template.has_debate:
                continue
            if has_checkpoint == "true" and not template.has_human_checkpoint:
                continue

            results.append(template)

        # Sort by relevance (downloads as proxy)
        results.sort(key=lambda t: t.downloads, reverse=True)

        # Validate pagination
        limit, offset, _ = _validate_pagination(query)
        paged_results = results[offset : offset + limit]

        return json_response(
            {
                "results": [t.to_dict() for t in paged_results],
                "total": len(results),
                "query": search_query,
            }
        )

    # =========================================================================
    # Popular Templates
    # =========================================================================

    async def _handle_popular(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get most popular templates."""
        templates = _load_templates()
        query = getattr(request, "query", {})

        limit, _, _ = _validate_pagination(query)
        if limit > 50:
            limit = 50

        # Sort by downloads and rating
        sorted_templates = sorted(
            templates.values(),
            key=lambda t: (t.downloads, t.rating),
            reverse=True,
        )[:limit]

        return json_response(
            {
                "popular": [t.to_dict() for t in sorted_templates],
            }
        )

    # =========================================================================
    # Deploy Template
    # =========================================================================

    @rate_limit(requests_per_minute=20, limiter_name="marketplace.deploy")
    async def _handle_deploy(self, request: Any, tenant_id: str, template_id: str) -> HandlerResult:
        """Deploy a template for the tenant."""
        cb = _get_marketplace_circuit_breaker()
        if not cb.is_allowed():
            return error_response("Marketplace temporarily unavailable", 503)

        try:
            templates = _load_templates()
            meta = templates.get(template_id)

            if not meta:
                return error_response("Template not found", 404)

            body = await self._get_json_body(request)

            # Validate deployment name
            valid, name, err = _validate_deployment_name_internal(body.get("name"), meta.name)
            if not valid:
                return error_response(err, 400)

            # Validate config
            valid, config, err = _validate_config(body.get("config"))
            if not valid:
                return error_response(err, 400)

            # Create deployment
            deployment_id = f"deploy_{uuid4().hex[:12]}"
            deployment = TemplateDeployment(
                id=deployment_id,
                template_id=template_id,
                tenant_id=tenant_id,
                name=name,
                status=DeploymentStatus.ACTIVE,
                config=config,
            )

            # Store deployment
            tenant_deployments = _get_tenant_deployments(tenant_id)
            tenant_deployments[deployment_id] = deployment

            # Increment download count
            _download_counts[template_id] = _download_counts.get(template_id, 0) + 1
            meta.downloads = _download_counts[template_id]

            cb.record_success()

            return json_response(
                {
                    "deployment": deployment.to_dict(),
                    "template": meta.to_dict(),
                    "message": f"Successfully deployed {meta.name}",
                }
            )

        except Exception as e:
            cb.record_failure()
            logger.exception(f"Error deploying template: {e}")
            return error_response("Deployment failed", 500)

    # =========================================================================
    # List Deployments
    # =========================================================================

    async def _handle_list_deployments(self, request: Any, tenant_id: str) -> HandlerResult:
        """List all deployments for the tenant."""
        deployments = _get_tenant_deployments(tenant_id)

        deployment_list = sorted(
            deployments.values(),
            key=lambda d: d.deployed_at,
            reverse=True,
        )

        return json_response(
            {
                "deployments": [d.to_dict() for d in deployment_list],
                "total": len(deployment_list),
            }
        )

    async def _handle_get_deployment(
        self, request: Any, tenant_id: str, deployment_id: str
    ) -> HandlerResult:
        """Get deployment details."""
        deployments = _get_tenant_deployments(tenant_id)
        deployment = deployments.get(deployment_id)

        if not deployment:
            return error_response("Deployment not found", 404)

        # Get template info
        templates = _load_templates()
        template = templates.get(deployment.template_id)

        return json_response(
            {
                "deployment": deployment.to_dict(),
                "template": template.to_dict() if template else None,
            }
        )

    async def _handle_delete_deployment(
        self, request: Any, tenant_id: str, deployment_id: str
    ) -> HandlerResult:
        """Archive a deployment."""
        deployments = _get_tenant_deployments(tenant_id)
        deployment = deployments.get(deployment_id)

        if not deployment:
            return error_response("Deployment not found", 404)

        deployment.status = DeploymentStatus.ARCHIVED

        return json_response(
            {
                "message": "Deployment archived",
                "deployment": deployment.to_dict(),
            }
        )

    # =========================================================================
    # Rate Template
    # =========================================================================

    @rate_limit(requests_per_minute=10, limiter_name="marketplace.rate")
    async def _handle_rate(self, request: Any, tenant_id: str, template_id: str) -> HandlerResult:
        """Rate a template."""
        cb = _get_marketplace_circuit_breaker()
        if not cb.is_allowed():
            return error_response("Marketplace temporarily unavailable", 503)

        try:
            templates = _load_templates()
            meta = templates.get(template_id)

            if not meta:
                return error_response("Template not found", 404)

            body = await self._get_json_body(request)

            # Validate rating
            valid, rating_value, err = _validate_rating(body.get("rating"))
            if not valid:
                return error_response(err, 400)

            # Validate review
            valid, review, err = _validate_review_internal(body.get("review"))
            if not valid:
                return error_response(err, 400)

            # Create rating
            rating = TemplateRating(
                id=f"rating_{uuid4().hex[:12]}",
                template_id=template_id,
                tenant_id=tenant_id,
                user_id=getattr(request, "user_id", "anonymous"),
                rating=rating_value,
                review=review,
            )

            # Store rating
            if template_id not in _ratings:
                _ratings[template_id] = []
            _ratings[template_id].append(rating)

            # Update average rating
            all_ratings = _ratings[template_id]
            meta.rating = sum(r.rating for r in all_ratings) / len(all_ratings)
            meta.rating_count = len(all_ratings)

            cb.record_success()

            return json_response(
                {
                    "rating": rating.to_dict(),
                    "template_rating": {
                        "average": meta.rating,
                        "count": meta.rating_count,
                    },
                }
            )

        except Exception as e:
            cb.record_failure()
            logger.exception(f"Error rating template: {e}")
            return error_response("Rating failed", 500)

    # =========================================================================
    # Health / Status
    # =========================================================================

    async def _handle_status(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get marketplace health status including circuit breaker state."""
        templates = _load_templates()
        cb_status = get_marketplace_circuit_breaker_status()

        return json_response(
            {
                "status": "healthy" if cb_status["state"] == "closed" else "degraded",
                "templates_loaded": len(templates),
                "circuit_breaker": cb_status,
                "deployments_count": sum(len(d) for d in _deployments.values()),
                "ratings_count": sum(len(r) for r in _ratings.values()),
            }
        )

    # =========================================================================
    # Demo Data
    # =========================================================================

    async def _handle_demo(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get demo marketplace data for development."""
        templates = _load_templates()

        # Group by category
        by_category: dict[str, list[dict[str, Any]]] = {}
        for template in templates.values():
            cat = template.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(template.to_dict())

        # Get featured templates (highest rated or most downloads)
        featured = sorted(
            templates.values(),
            key=lambda t: (t.rating, t.downloads),
            reverse=True,
        )[:6]

        return json_response(
            {
                "featured": [t.to_dict() for t in featured],
                "by_category": by_category,
                "categories": [
                    {
                        "id": cat.value,
                        **CATEGORY_INFO.get(cat, {}),
                        "count": len(by_category.get(cat.value, [])),
                    }
                    for cat in TemplateCategory
                ],
                "total_templates": len(templates),
            }
        )


# =============================================================================
# Module-level helpers
# =============================================================================


def get_marketplace_handler() -> MarketplaceHandler:
    """Get a MarketplaceHandler instance."""
    return MarketplaceHandler()


async def handle_marketplace(request: Any, path: str, method: str) -> HandlerResult:
    """Handle a marketplace request."""
    handler = get_marketplace_handler()
    return await handler.handle(request, path, method)


# =============================================================================
# Aliases for backward compatibility with existing tests
# =============================================================================

# Clear function alias
_clear_marketplace_components = _clear_marketplace_state


def _validate_template_id(value: str) -> tuple[bool, str | None]:  # type: ignore[no-redef]
    """Validate a template ID (backward-compatible signature).

    Returns (is_valid, error_or_None).
    """
    valid, err = _validate_id(value, "Template ID")
    return valid, err if not valid else None


def _validate_deployment_id(value: str) -> tuple[bool, str | None]:  # type: ignore[no-redef]
    """Validate a deployment ID (backward-compatible signature).

    Returns (is_valid, error_or_None).
    """
    valid, err = _validate_id(value, "Deployment ID")
    return valid, err if not valid else None


def _validate_deployment_name(value: Any, fallback: str = "") -> tuple[bool, str | None]:  # type: ignore[no-redef]
    """Validate a deployment name (backward-compatible 2-tuple signature).

    Returns (is_valid, error_or_None).
    """
    valid, _, err = _validate_deployment_name_internal(value, fallback)
    return valid, err if not valid else None


def _validate_review(value: Any) -> tuple[bool, str | None]:  # type: ignore[no-redef]
    """Validate a review (backward-compatible 2-tuple signature).

    Returns (is_valid, error_or_None).
    """
    valid, _, err = _validate_review_internal(value)
    return valid, err if not valid else None


def _validate_rating(value: Any) -> tuple[bool, int, str]:  # type: ignore[no-redef]
    """Validate a rating value (alias for _validate_rating_value)."""
    return _validate_rating_value(value)


def _validate_category(value: Any) -> tuple[bool, TemplateCategory | None, str]:  # type: ignore[no-redef]
    """Validate a category filter (backward-compatible, returns enum).

    Returns (is_valid, TemplateCategory_or_None, error_message).
    """
    if value is None or value == "":
        return True, None, ""
    if not isinstance(value, str):
        return False, None, "Category must be a string"
    # Case-insensitive lookup
    lower = value.lower()
    valid_categories = {cat.value: cat for cat in TemplateCategory}
    if lower not in valid_categories:
        return (
            False,
            None,
            f"Invalid category. Must be one of: {', '.join(sorted(valid_categories))}",
        )
    return True, valid_categories[lower], ""


def _clamp_pagination(limit: Any, offset: Any) -> tuple[int, int]:  # type: ignore[no-redef]
    """Clamp pagination values (backward-compatible direct args)."""
    try:
        limit_int = int(limit) if limit is not None else DEFAULT_LIMIT
    except (ValueError, TypeError):
        limit_int = DEFAULT_LIMIT
    try:
        offset_int = int(offset) if offset is not None else 0
    except (ValueError, TypeError):
        offset_int = 0
    return (
        max(MIN_LIMIT, min(limit_int, MAX_LIMIT)),
        max(0, min(offset_int, MAX_OFFSET)),
    )


# Constant aliases
MAX_CONFIG_SIZE = MAX_CONFIG_KEYS
SAFE_TEMPLATE_ID_PATTERN = SAFE_ID_PATTERN
