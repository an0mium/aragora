"""Marketplace Handler - Template discovery and deployment.

Provides API endpoints for discovering, browsing, and deploying workflow templates
across different industry verticals.

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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)


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
    tags: List[str] = field(default_factory=list)
    icon: str = "document"
    author: str = "Aragora"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    downloads: int = 0
    rating: float = 0.0
    rating_count: int = 0
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    steps_count: int = 0
    has_debate: bool = False
    has_human_checkpoint: bool = False
    estimated_duration: str = "varies"
    file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
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
    config: Dict[str, Any] = field(default_factory=dict)
    deployed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_run: Optional[datetime] = None
    run_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
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
    review: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "template_id": self.template_id,
            "rating": self.rating,
            "review": self.review,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# In-Memory Storage
# =============================================================================

# Template cache: template_id -> TemplateMetadata
_templates_cache: Dict[str, TemplateMetadata] = {}

# Deployments: tenant_id -> deployment_id -> TemplateDeployment
_deployments: Dict[str, Dict[str, TemplateDeployment]] = {}

# Ratings: template_id -> List[TemplateRating]
_ratings: Dict[str, List[TemplateRating]] = {}

# Download counts: template_id -> count
_download_counts: Dict[str, int] = {}


# =============================================================================
# Template Discovery
# =============================================================================


def _get_templates_dir() -> Path:
    """Get the workflow templates directory."""
    return Path(__file__).parent.parent.parent.parent / "workflow" / "templates"


def _load_templates() -> Dict[str, TemplateMetadata]:
    """Load all templates from the templates directory."""
    global _templates_cache

    if _templates_cache:
        return _templates_cache

    templates_dir = _get_templates_dir()
    if not templates_dir.exists():
        logger.warning(f"Templates directory not found: {templates_dir}")
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
    return _templates_cache


def _parse_template_file(file_path: Path) -> Optional[TemplateMetadata]:
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


def _get_full_template(template_id: str) -> Optional[Dict[str, Any]]:
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


def _get_tenant_deployments(tenant_id: str) -> Dict[str, TemplateDeployment]:
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


class MarketplaceHandler(BaseHandler):
    """Handler for marketplace API endpoints."""

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
    ]

    def __init__(self, server_context: Optional[Dict[str, Any]] = None):
        """Initialize handler with optional server context."""
        super().__init__(server_context or {})  # type: ignore[arg-type]
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

    async def handle(self, request: Any, path: str, method: str) -> HandlerResult:  # type: ignore[override]
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

            # Template-specific paths
            # Path format: /api/v1/marketplace/templates/{template_id}[/{action}]
            # Split: ['', 'api', 'v1', 'marketplace', 'templates', 'template_id', ...]
            # Index:  0     1      2         3             4          5
            elif path.startswith("/api/v1/marketplace/templates/"):
                parts = path.split("/")
                if len(parts) >= 6:
                    template_id = parts[5]

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
            # Path format: /api/v1/marketplace/deployments/{deployment_id}
            # Split: ['', 'api', 'v1', 'marketplace', 'deployments', 'deployment_id']
            # Index:  0     1      2         3             4              5
            elif path.startswith("/api/v1/marketplace/deployments/"):
                parts = path.split("/")
                if len(parts) == 6:
                    deployment_id = parts[5]
                    if method == "GET":
                        return await self._handle_get_deployment(request, tenant_id, deployment_id)
                    elif method == "DELETE":
                        return await self._handle_delete_deployment(
                            request, tenant_id, deployment_id
                        )

            return error_response("Not found", 404)

        except Exception as e:
            logger.exception(f"Error in marketplace handler: {e}")
            return error_response(f"Internal error: {str(e)}", 500)

    def _get_tenant_id(self, request: Any) -> str:
        """Extract tenant ID from request context."""
        return getattr(request, "tenant_id", "default")

    async def _get_json_body(self, request: Any) -> Dict[str, Any]:
        """Parse JSON body from request."""
        if hasattr(request, "json"):
            body = await request.json()
            return body if isinstance(body, dict) else {}
        return {}

    # =========================================================================
    # List Templates
    # =========================================================================

    async def _handle_list_templates(self, request: Any, tenant_id: str) -> HandlerResult:
        """List all available templates.

        Query params:
        - category: Filter by category
        - limit: Max results (default 50)
        - offset: Pagination offset
        """
        templates = _load_templates()
        query = getattr(request, "query", {})

        # Filter by category
        category_filter = query.get("category")
        if category_filter:
            templates = {k: v for k, v in templates.items() if v.category.value == category_filter}

        # Convert to list and sort by downloads
        template_list = sorted(
            templates.values(),
            key=lambda t: t.downloads,
            reverse=True,
        )

        # Pagination
        limit = int(query.get("limit", 50))
        offset = int(query.get("offset", 0))
        total = len(template_list)
        template_list = template_list[offset : offset + limit]

        return success_response(
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

        return success_response(
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
        self, template: TemplateMetadata, all_templates: Dict[str, TemplateMetadata]
    ) -> List[Dict[str, Any]]:
        """Find related templates based on category and tags."""
        related = []

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
        return [t.to_dict() for _, t in related[:5]]

    # =========================================================================
    # List Categories
    # =========================================================================

    async def _handle_list_categories(self, request: Any, tenant_id: str) -> HandlerResult:
        """List all template categories with counts."""
        templates = _load_templates()

        # Count templates per category
        category_counts: Dict[str, int] = {}
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

        return success_response({"categories": categories})

    # =========================================================================
    # Search Templates
    # =========================================================================

    async def _handle_search(self, request: Any, tenant_id: str) -> HandlerResult:
        """Search templates by query.

        Query params:
        - q: Search query
        - category: Filter by category
        - tags: Filter by tags (comma-separated)
        - has_debate: Filter by debate feature
        - has_checkpoint: Filter by human checkpoint feature
        """
        templates = _load_templates()
        query = getattr(request, "query", {})

        search_query = query.get("q", "").lower()
        category_filter = query.get("category")
        tags_filter = query.get("tags", "").split(",") if query.get("tags") else []
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

        return success_response(
            {
                "results": [t.to_dict() for t in results[:50]],
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

        limit = int(query.get("limit", 10))

        # Sort by downloads and rating
        sorted_templates = sorted(
            templates.values(),
            key=lambda t: (t.downloads, t.rating),
            reverse=True,
        )[:limit]

        return success_response(
            {
                "popular": [t.to_dict() for t in sorted_templates],
            }
        )

    # =========================================================================
    # Deploy Template
    # =========================================================================

    async def _handle_deploy(self, request: Any, tenant_id: str, template_id: str) -> HandlerResult:
        """Deploy a template for the tenant.

        Request body:
        {
            "name": "My Invoice Processor",
            "config": {
                "auto_approve_threshold": 1000,
                ...
            }
        }
        """
        try:
            templates = _load_templates()
            meta = templates.get(template_id)

            if not meta:
                return error_response("Template not found", 404)

            body = await self._get_json_body(request)
            name = body.get("name", meta.name)
            config = body.get("config", {})

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

            return success_response(
                {
                    "deployment": deployment.to_dict(),
                    "template": meta.to_dict(),
                    "message": f"Successfully deployed {meta.name}",
                }
            )

        except Exception as e:
            logger.exception(f"Error deploying template: {e}")
            return error_response(f"Deployment failed: {str(e)}", 500)

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

        return success_response(
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

        return success_response(
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

        return success_response(
            {
                "message": "Deployment archived",
                "deployment": deployment.to_dict(),
            }
        )

    # =========================================================================
    # Rate Template
    # =========================================================================

    async def _handle_rate(self, request: Any, tenant_id: str, template_id: str) -> HandlerResult:
        """Rate a template.

        Request body:
        {
            "rating": 5,
            "review": "Great template!"
        }
        """
        try:
            templates = _load_templates()
            meta = templates.get(template_id)

            if not meta:
                return error_response("Template not found", 404)

            body = await self._get_json_body(request)
            rating_value = body.get("rating")
            review = body.get("review")

            if not rating_value or not isinstance(rating_value, int):
                return error_response("Rating must be an integer", 400)

            if rating_value < 1 or rating_value > 5:
                return error_response("Rating must be between 1 and 5", 400)

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

            return success_response(
                {
                    "rating": rating.to_dict(),
                    "template_rating": {
                        "average": meta.rating,
                        "count": meta.rating_count,
                    },
                }
            )

        except Exception as e:
            logger.exception(f"Error rating template: {e}")
            return error_response(f"Rating failed: {str(e)}", 500)

    # =========================================================================
    # Demo Data
    # =========================================================================

    async def _handle_demo(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get demo marketplace data for development."""
        templates = _load_templates()

        # Group by category
        by_category: Dict[str, List[Dict[str, Any]]] = {}
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

        return success_response(
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
