"""
Template Marketplace API Handler.

Provides community marketplace functionality for workflow templates:
- Browse and search community templates
- Publish templates to marketplace
- Rate and review templates
- Import templates from marketplace
- Featured and trending templates

Endpoints:
- GET  /api/marketplace/templates          - Browse marketplace templates
- GET  /api/marketplace/templates/:id      - Get marketplace template details
- POST /api/marketplace/templates          - Publish template to marketplace
- POST /api/marketplace/templates/:id/rate - Rate a template
- GET  /api/marketplace/templates/:id/reviews - Get template reviews
- POST /api/marketplace/templates/:id/reviews - Submit a review
- POST /api/marketplace/templates/:id/import  - Import to workspace
- GET  /api/marketplace/featured           - Get featured templates
- GET  /api/marketplace/trending           - Get trending templates
- GET  /api/marketplace/categories         - Get marketplace categories
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_bounded_string_param,
    get_clamped_int_param,
    get_int_param,
    handle_errors,
    json_response,
)
from .utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiters
_marketplace_limiter = RateLimiter(requests_per_minute=120)
_publish_limiter = RateLimiter(requests_per_minute=10)
_rate_limiter = RateLimiter(requests_per_minute=30)


@dataclass
class MarketplaceTemplate:
    """A template published to the marketplace."""

    id: str
    name: str
    description: str
    category: str
    pattern: str
    author_id: str
    author_name: str
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    workflow_definition: Dict[str, Any] = field(default_factory=dict)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    documentation: str = ""
    examples: List[Dict[str, Any]] = field(default_factory=list)
    rating: float = 0.0
    rating_count: int = 0
    download_count: int = 0
    is_featured: bool = False
    is_verified: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "pattern": self.pattern,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "version": self.version,
            "tags": self.tags,
            "workflow_definition": self.workflow_definition,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "documentation": self.documentation,
            "examples": self.examples,
            "rating": self.rating,
            "rating_count": self.rating_count,
            "download_count": self.download_count,
            "is_featured": self.is_featured,
            "is_verified": self.is_verified,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_summary(self) -> Dict[str, Any]:
        """Convert to summary (for listings)."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description[:200] + "..."
            if len(self.description) > 200
            else self.description,
            "category": self.category,
            "pattern": self.pattern,
            "author_name": self.author_name,
            "version": self.version,
            "tags": self.tags[:5],
            "rating": self.rating,
            "rating_count": self.rating_count,
            "download_count": self.download_count,
            "is_featured": self.is_featured,
            "is_verified": self.is_verified,
            "created_at": self.created_at,
        }


@dataclass
class TemplateReview:
    """A review for a marketplace template."""

    id: str
    template_id: str
    user_id: str
    user_name: str
    rating: int  # 1-5
    title: str
    content: str
    helpful_count: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "template_id": self.template_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "rating": self.rating,
            "title": self.title,
            "content": self.content,
            "helpful_count": self.helpful_count,
            "created_at": self.created_at,
        }


# In-memory storage (fallback when database unavailable)
_marketplace_templates: Dict[str, MarketplaceTemplate] = {}
_template_reviews: Dict[str, List[TemplateReview]] = {}
_user_ratings: Dict[str, Dict[str, int]] = {}  # user_id -> {template_id: rating}

# Production persistence (optional)
_use_persistent_store: bool = False
_persistent_store: Optional[Any] = None


def _init_persistent_store() -> bool:
    """Initialize persistent store if available."""
    global _use_persistent_store, _persistent_store
    try:
        from aragora.storage.marketplace_store import get_marketplace_store

        _persistent_store = get_marketplace_store()
        _use_persistent_store = True
        logger.info("Template marketplace using persistent SQLite storage")
        return True
    except Exception as e:
        logger.warning(
            f"TEMPLATE MARKETPLACE: Persistent storage unavailable ({e}). "
            "Using in-memory fallback - TEMPLATES AND REVIEWS WILL BE LOST ON RESTART!"
        )
        _use_persistent_store = False
        return False


def _get_persistent_store():
    """Get the persistent store, initializing if needed."""
    global _persistent_store
    if _persistent_store is None:
        _init_persistent_store()
    return _persistent_store if _use_persistent_store else None


def _seed_marketplace_templates() -> None:
    """Seed marketplace with sample templates."""
    if _marketplace_templates:
        return  # Already seeded

    sample_templates = [
        MarketplaceTemplate(
            id="security/code-audit",
            name="Security Code Audit",
            description="Comprehensive multi-agent security code review with vulnerability scanning, dependency analysis, and OWASP compliance checking.",
            category="security",
            pattern="review_cycle",
            author_id="aragora-team",
            author_name="Aragora Team",
            tags=["security", "code-review", "owasp", "vulnerabilities"],
            rating=4.8,
            rating_count=156,
            download_count=2340,
            is_featured=True,
            is_verified=True,
        ),
        MarketplaceTemplate(
            id="research/deep-dive",
            name="Research Deep Dive",
            description="Multi-agent research workflow combining web search, document analysis, and synthesis for comprehensive topic exploration.",
            category="research",
            pattern="hive_mind",
            author_id="aragora-team",
            author_name="Aragora Team",
            tags=["research", "analysis", "synthesis", "web-search"],
            rating=4.6,
            rating_count=89,
            download_count=1567,
            is_featured=True,
            is_verified=True,
        ),
        MarketplaceTemplate(
            id="legal/contract-review",
            name="Contract Review Assistant",
            description="Automated contract analysis with risk identification, clause extraction, and compliance verification.",
            category="legal",
            pattern="pipeline",
            author_id="legal-team",
            author_name="Legal Templates",
            tags=["legal", "contracts", "compliance", "risk-analysis"],
            rating=4.5,
            rating_count=67,
            download_count=890,
            is_featured=False,
            is_verified=True,
        ),
        MarketplaceTemplate(
            id="content/blog-writer",
            name="AI Blog Writer",
            description="Content creation workflow with research, outline generation, drafting, and SEO optimization.",
            category="content",
            pattern="pipeline",
            author_id="content-creators",
            author_name="Content Creators Hub",
            tags=["content", "writing", "seo", "blog"],
            rating=4.3,
            rating_count=234,
            download_count=3456,
            is_featured=True,
            is_verified=False,
        ),
        MarketplaceTemplate(
            id="data/analysis-pipeline",
            name="Data Analysis Pipeline",
            description="End-to-end data analysis with cleaning, exploration, statistical analysis, and visualization generation.",
            category="data",
            pattern="map_reduce",
            author_id="data-science-team",
            author_name="Data Science Templates",
            tags=["data", "analysis", "statistics", "visualization"],
            rating=4.7,
            rating_count=112,
            download_count=1890,
            is_featured=False,
            is_verified=True,
        ),
    ]

    for template in sample_templates:
        _marketplace_templates[template.id] = template


class TemplateMarketplaceHandler(BaseHandler):
    """Handler for template marketplace API endpoints."""

    ROUTES: list[str] = [
        "/api/marketplace/templates",
        "/api/marketplace/templates/*",
        "/api/marketplace/featured",
        "/api/marketplace/trending",
        "/api/marketplace/categories",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path.startswith("/api/marketplace/")

    def handle(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Route marketplace requests."""
        # Ensure marketplace is seeded
        _seed_marketplace_templates()

        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _marketplace_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded", 429)

        method = handler.command if hasattr(handler, "command") else "GET"

        # Featured templates
        if path == "/api/marketplace/featured":
            return self._get_featured()

        # Trending templates
        if path == "/api/marketplace/trending":
            return self._get_trending(query_params)

        # Categories
        if path == "/api/marketplace/categories":
            return self._get_categories()

        # Template list/search
        if path == "/api/marketplace/templates":
            if method == "GET":
                return self._list_templates(query_params)
            elif method == "POST":
                return self._publish_template(handler, client_ip)
            else:
                return error_response(f"Method {method} not allowed", 405)

        # Specific template operations
        if path.startswith("/api/marketplace/templates/"):
            parts = path.split("/")
            if len(parts) >= 4:
                template_id = "/".join(parts[4:])

                # Check for special routes
                if template_id.endswith("/rate"):
                    template_id = template_id[:-5]
                    return self._rate_template(template_id, handler, client_ip)
                elif template_id.endswith("/reviews"):
                    template_id = template_id[:-8]
                    if method == "GET":
                        return self._get_reviews(template_id, query_params)
                    elif method == "POST":
                        return self._submit_review(template_id, handler, client_ip)
                elif template_id.endswith("/import"):
                    template_id = template_id[:-7]
                    return self._import_template(template_id, handler)
                else:
                    return self._get_template(template_id)

        return error_response("Invalid path", 400)

    @handle_errors("list marketplace templates")
    def _list_templates(self, query_params: dict) -> HandlerResult:
        """List marketplace templates with search and filters."""
        category = get_bounded_string_param(query_params, "category", None, max_length=50)
        pattern = get_bounded_string_param(query_params, "pattern", None, max_length=50)
        search = get_bounded_string_param(query_params, "search", None, max_length=100)
        tags = get_bounded_string_param(query_params, "tags", None, max_length=200)
        sort_by = get_bounded_string_param(query_params, "sort_by", "downloads", max_length=20)
        verified_only = query_params.get("verified_only", "false").lower() == "true"
        limit = get_clamped_int_param(query_params, "limit", 20, min_val=1, max_val=50)
        offset = get_int_param(query_params, "offset", 0)

        # Get all templates
        templates = list(_marketplace_templates.values())

        # Apply filters
        if category:
            templates = [t for t in templates if t.category == category]

        if pattern:
            templates = [t for t in templates if t.pattern == pattern]

        if verified_only:
            templates = [t for t in templates if t.is_verified]

        if tags:
            tag_list = [t.strip().lower() for t in tags.split(",")]
            templates = [
                t
                for t in templates
                if any(tag.lower() in [x.lower() for x in t.tags] for tag in tag_list)
            ]

        if search:
            search_lower = search.lower()
            templates = [
                t
                for t in templates
                if search_lower in t.name.lower()
                or search_lower in t.description.lower()
                or any(search_lower in tag.lower() for tag in t.tags)
            ]

        # Sort
        if sort_by == "rating":
            templates.sort(key=lambda t: (t.rating, t.rating_count), reverse=True)
        elif sort_by == "downloads":
            templates.sort(key=lambda t: t.download_count, reverse=True)
        elif sort_by == "newest":
            templates.sort(key=lambda t: t.created_at, reverse=True)
        elif sort_by == "name":
            templates.sort(key=lambda t: t.name.lower())

        # Count total before pagination
        total = len(templates)

        # Apply pagination
        templates = templates[offset : offset + limit]

        return json_response(
            {
                "templates": [t.to_summary() for t in templates],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )

    @handle_errors("get marketplace template")
    def _get_template(self, template_id: str) -> HandlerResult:
        """Get full details of a marketplace template."""
        template = _marketplace_templates.get(template_id)
        if not template:
            return error_response(f"Template not found: {template_id}", 404)

        return json_response(template.to_dict())

    @handle_errors("publish template")
    def _publish_template(self, handler: Any, client_ip: str) -> HandlerResult:
        """Publish a new template to the marketplace."""
        # Rate limit for publishing
        if not _publish_limiter.is_allowed(client_ip):
            return error_response("Publishing rate limit exceeded", 429)

        # Parse request body
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        # Validate required fields
        required = ["name", "description", "category", "pattern", "workflow_definition"]
        for required_field in required:
            if required_field not in data:
                return error_response(f"Missing required field: {required_field}", 400)

        # Generate ID
        template_id = f"{data['category']}/{data['name'].lower().replace(' ', '-')}"

        # Check for duplicates
        if template_id in _marketplace_templates:
            return error_response("Template with this name already exists in category", 409)

        # Create template
        template = MarketplaceTemplate(
            id=template_id,
            name=data["name"],
            description=data["description"],
            category=data["category"],
            pattern=data["pattern"],
            author_id=data.get("author_id", "anonymous"),
            author_name=data.get("author_name", "Anonymous"),
            version=data.get("version", "1.0.0"),
            tags=data.get("tags", []),
            workflow_definition=data["workflow_definition"],
            input_schema=data.get("input_schema", {}),
            output_schema=data.get("output_schema", {}),
            documentation=data.get("documentation", ""),
            examples=data.get("examples", []),
        )

        _marketplace_templates[template_id] = template

        return json_response(
            {
                "status": "published",
                "template_id": template_id,
                "template": template.to_summary(),
            },
            status=201,
        )

    @handle_errors("rate template")
    def _rate_template(self, template_id: str, handler: Any, client_ip: str) -> HandlerResult:
        """Rate a marketplace template."""
        if not _rate_limiter.is_allowed(client_ip):
            return error_response("Rating rate limit exceeded", 429)

        template = _marketplace_templates.get(template_id)
        if not template:
            return error_response(f"Template not found: {template_id}", 404)

        # Parse request body
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        rating = data.get("rating")
        if not isinstance(rating, (int, float)) or rating < 1 or rating > 5:
            return error_response("Rating must be between 1 and 5", 400)

        user_id = data.get("user_id", client_ip)

        # Check if user already rated
        if user_id in _user_ratings and template_id in _user_ratings[user_id]:
            old_rating = _user_ratings[user_id][template_id]
            # Update average rating
            total = template.rating * template.rating_count
            total = total - old_rating + rating
            template.rating = total / template.rating_count
        else:
            # New rating
            total = template.rating * template.rating_count
            template.rating_count += 1
            template.rating = (total + rating) / template.rating_count

        # Store user rating
        if user_id not in _user_ratings:
            _user_ratings[user_id] = {}
        _user_ratings[user_id][template_id] = rating

        return json_response(
            {
                "status": "rated",
                "template_id": template_id,
                "your_rating": rating,
                "average_rating": round(template.rating, 2),
                "rating_count": template.rating_count,
            }
        )

    @handle_errors("get reviews")
    def _get_reviews(self, template_id: str, query_params: dict) -> HandlerResult:
        """Get reviews for a template."""
        if template_id not in _marketplace_templates:
            return error_response(f"Template not found: {template_id}", 404)

        reviews = _template_reviews.get(template_id, [])
        limit = get_clamped_int_param(query_params, "limit", 10, min_val=1, max_val=50)
        offset = get_int_param(query_params, "offset", 0)

        # Sort by most helpful
        reviews_sorted = sorted(reviews, key=lambda r: r.helpful_count, reverse=True)
        total = len(reviews_sorted)
        reviews_page = reviews_sorted[offset : offset + limit]

        return json_response(
            {
                "reviews": [r.to_dict() for r in reviews_page],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )

    @handle_errors("submit review")
    def _submit_review(self, template_id: str, handler: Any, client_ip: str) -> HandlerResult:
        """Submit a review for a template."""
        if template_id not in _marketplace_templates:
            return error_response(f"Template not found: {template_id}", 404)

        # Parse request body
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        # Validate required fields
        if "rating" not in data:
            return error_response("Rating is required", 400)
        if "content" not in data:
            return error_response("Review content is required", 400)

        rating = data["rating"]
        if not isinstance(rating, (int, float)) or rating < 1 or rating > 5:
            return error_response("Rating must be between 1 and 5", 400)

        review = TemplateReview(
            id=str(uuid.uuid4()),
            template_id=template_id,
            user_id=data.get("user_id", client_ip),
            user_name=data.get("user_name", "Anonymous"),
            rating=int(rating),
            title=data.get("title", ""),
            content=data["content"],
        )

        if template_id not in _template_reviews:
            _template_reviews[template_id] = []
        _template_reviews[template_id].append(review)

        # Also update template rating
        template = _marketplace_templates[template_id]
        total = template.rating * template.rating_count
        template.rating_count += 1
        template.rating = (total + rating) / template.rating_count

        return json_response(
            {
                "status": "submitted",
                "review": review.to_dict(),
            },
            status=201,
        )

    @handle_errors("import template")
    def _import_template(self, template_id: str, handler: Any) -> HandlerResult:
        """Import a template to user's workspace."""
        template = _marketplace_templates.get(template_id)
        if not template:
            return error_response(f"Template not found: {template_id}", 404)

        # Parse request body
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError):
            data = {}

        workspace_id = data.get("workspace_id")

        # Increment download count
        template.download_count += 1

        # Return the full template definition for import
        return json_response(
            {
                "status": "imported",
                "template_id": template_id,
                "workspace_id": workspace_id,
                "workflow_definition": template.workflow_definition,
                "input_schema": template.input_schema,
                "output_schema": template.output_schema,
                "documentation": template.documentation,
                "download_count": template.download_count,
            }
        )

    @handle_errors("get featured")
    def _get_featured(self) -> HandlerResult:
        """Get featured marketplace templates."""
        featured = [t.to_summary() for t in _marketplace_templates.values() if t.is_featured]
        # Sort by rating
        featured.sort(key=lambda t: t["rating"], reverse=True)

        return json_response(
            {
                "featured": featured[:10],
                "total": len(featured),
            }
        )

    @handle_errors("get trending")
    def _get_trending(self, query_params: dict) -> HandlerResult:
        """Get trending marketplace templates."""
        period = get_bounded_string_param(query_params, "period", "week", max_length=20)
        limit = get_clamped_int_param(query_params, "limit", 10, min_val=1, max_val=20)

        # In production, this would factor in recent downloads, ratings, etc.
        # For now, sort by download count
        templates = sorted(
            _marketplace_templates.values(), key=lambda t: t.download_count, reverse=True
        )[:limit]

        return json_response(
            {
                "trending": [t.to_summary() for t in templates],
                "period": period,
                "total": len(templates),
            }
        )

    @handle_errors("get categories")
    def _get_categories(self) -> HandlerResult:
        """Get marketplace categories with counts."""
        # Count templates per category
        category_counts: Dict[str, Dict[str, Any]] = {}
        for template in _marketplace_templates.values():
            cat = template.category
            if cat not in category_counts:
                category_counts[cat] = {
                    "id": cat,
                    "name": cat.replace("_", " ").title(),
                    "template_count": 0,
                    "total_downloads": 0,
                }
            category_counts[cat]["template_count"] += 1
            category_counts[cat]["total_downloads"] += template.download_count

        categories = list(category_counts.values())
        categories.sort(key=lambda c: c["template_count"], reverse=True)

        return json_response(
            {
                "categories": categories,
                "total": len(categories),
            }
        )
