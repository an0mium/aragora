"""
Reviews Handler - Serve shareable code reviews.

Endpoints:
- GET /api/reviews/{id} - Get a specific review by ID
- GET /api/reviews - List recent reviews
"""

import json
import logging
from pathlib import Path
from typing import Any

from .base import BaseHandler, error_response
from .utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for reviews endpoints (30 requests per minute)
_reviews_limiter = RateLimiter(requests_per_minute=30)

# Reviews are stored at ~/.aragora/reviews/
REVIEWS_DIR = Path.home() / ".aragora" / "reviews"


class ReviewsHandler(BaseHandler):
    """Handler for serving shareable code reviews."""

    prefix = "/api/reviews"

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the request."""
        return path.startswith(self.prefix)

    def handle(self, handler: Any, path: str, method: str) -> dict | None:
        """Handle the request."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _reviews_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for reviews endpoint: {client_ip}")
            return {"error": "Rate limit exceeded. Please try again later.", "status": 429}

        if method != "GET":
            return {"error": "Method not allowed", "status": 405}

        # Strip prefix
        subpath = path[len(self.prefix):]

        if not subpath or subpath == "/":
            # List recent reviews
            return self._list_reviews()
        elif subpath.startswith("/"):
            # Get specific review
            review_id = subpath[1:].split("/")[0]
            return self._get_review(review_id)

        return None

    def _list_reviews(self, limit: int = 20) -> dict:
        """List recent reviews."""
        if not REVIEWS_DIR.exists():
            return {"reviews": [], "total": 0}

        reviews = []
        for review_file in sorted(
            REVIEWS_DIR.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )[:limit]:
            try:
                data = json.loads(review_file.read_text())
                # Return summary only
                reviews.append({
                    "id": data.get("id"),
                    "created_at": data.get("created_at"),
                    "agents": data.get("agents", []),
                    "pr_url": data.get("pr_url"),
                    "unanimous_count": len(data.get("findings", {}).get("unanimous_critiques", [])),
                    "agreement_score": data.get("findings", {}).get("agreement_score", 0),
                })
            except (json.JSONDecodeError, KeyError):
                continue

        return {"reviews": reviews, "total": len(reviews)}

    def _get_review(self, review_id: str) -> dict:
        """Get a specific review by ID."""
        # Validate ID: must be alphanumeric and reasonable length
        if not review_id or not review_id.isalnum() or len(review_id) > 64:
            return {"error": "Invalid review ID", "status": 400}

        review_path = REVIEWS_DIR / f"{review_id}.json"
        if not review_path.exists():
            return {"error": "Review not found", "status": 404}

        try:
            data = json.loads(review_path.read_text())
            return {"review": data}
        except json.JSONDecodeError:
            return {"error": "Invalid review data", "status": 500}
