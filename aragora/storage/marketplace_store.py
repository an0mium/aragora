"""
SQLite-backed marketplace template store with production persistence.

Provides durable storage for community workflow templates, ratings, and reviews.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from aragora.storage.base_store import SQLiteStore
from aragora.storage.schema import SchemaManager

logger = logging.getLogger(__name__)


@dataclass
class StoredTemplate:
    """Template stored in the marketplace."""

    id: str
    name: str
    description: str
    author_id: str
    author_name: str
    category: str
    pattern: str
    tags: list[str] = field(default_factory=list)
    workflow_definition: dict[str, Any] = field(default_factory=dict)
    download_count: int = 0
    rating_sum: float = 0.0
    rating_count: int = 0
    is_featured: bool = False
    is_trending: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def rating(self) -> float:
        """Calculate average rating."""
        if self.rating_count == 0:
            return 0.0
        return self.rating_sum / self.rating_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to API response format."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "category": self.category,
            "pattern": self.pattern,
            "tags": self.tags,
            "download_count": self.download_count,
            "rating": self.rating,
            "rating_count": self.rating_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_full_dict(self) -> dict[str, Any]:
        """Convert to full format including workflow definition."""
        result = self.to_dict()
        result["workflow_definition"] = self.workflow_definition
        return result


@dataclass
class StoredReview:
    """Review stored in the marketplace."""

    id: str
    template_id: str
    user_id: str
    user_name: str
    rating: int
    title: str
    content: str
    helpful_count: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to API response format."""
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


class MarketplaceStore(SQLiteStore):
    """
    SQLite-backed store for marketplace templates, ratings, and reviews.

    Provides:
    - Template CRUD operations
    - Rating aggregation
    - Review management
    - Featured/trending template tracking
    - Category and tag-based filtering
    """

    SCHEMA_NAME = "marketplace_store"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        -- Templates table
        CREATE TABLE IF NOT EXISTS templates (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL,
            author_id TEXT NOT NULL,
            author_name TEXT NOT NULL,
            category TEXT NOT NULL,
            pattern TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '[]',
            workflow_definition TEXT NOT NULL DEFAULT '{}',
            download_count INTEGER DEFAULT 0,
            rating_sum REAL DEFAULT 0.0,
            rating_count INTEGER DEFAULT 0,
            is_featured INTEGER DEFAULT 0,
            is_trending INTEGER DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_templates_category ON templates(category);
        CREATE INDEX IF NOT EXISTS idx_templates_author ON templates(author_id);
        CREATE INDEX IF NOT EXISTS idx_templates_rating ON templates(rating_sum / NULLIF(rating_count, 0) DESC);
        CREATE INDEX IF NOT EXISTS idx_templates_downloads ON templates(download_count DESC);
        CREATE INDEX IF NOT EXISTS idx_templates_featured ON templates(is_featured);
        CREATE INDEX IF NOT EXISTS idx_templates_trending ON templates(is_trending);

        -- Reviews table
        CREATE TABLE IF NOT EXISTS reviews (
            id TEXT PRIMARY KEY,
            template_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            user_name TEXT NOT NULL,
            rating INTEGER NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            helpful_count INTEGER DEFAULT 0,
            created_at REAL NOT NULL,
            FOREIGN KEY (template_id) REFERENCES templates(id) ON DELETE CASCADE,
            UNIQUE(template_id, user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_reviews_template ON reviews(template_id);
        CREATE INDEX IF NOT EXISTS idx_reviews_user ON reviews(user_id);

        -- Ratings table (individual ratings, not reviews)
        CREATE TABLE IF NOT EXISTS ratings (
            template_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            rating INTEGER NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (template_id, user_id),
            FOREIGN KEY (template_id) REFERENCES templates(id) ON DELETE CASCADE
        );

        -- Categories table (for metadata)
        CREATE TABLE IF NOT EXISTS categories (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            template_count INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_categories_count ON categories(template_count DESC);
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        auto_init: bool = True,
    ):
        """Initialize the marketplace store.

        Args:
            db_path: Path to SQLite database. Defaults to standard data path.
            auto_init: If True, initialize schema on construction.
        """
        if db_path is None:
            db_path = Path.home() / ".aragora" / "marketplace.db"

        super().__init__(db_path, auto_init=auto_init)

    def _post_init(self) -> None:
        """Initialize default categories after schema."""
        default_categories = [
            ("security", "Security", "Security analysis and vulnerability scanning"),
            ("code-review", "Code Review", "Code review and quality analysis"),
            ("compliance", "Compliance", "Regulatory compliance and auditing"),
            ("legal", "Legal", "Legal document review and analysis"),
            ("healthcare", "Healthcare", "Healthcare and medical workflows"),
            ("testing", "Testing", "Test generation and validation"),
            ("research", "Research", "Research and analysis workflows"),
        ]

        with self.connection() as conn:
            for cat_id, name, desc in default_categories:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO categories (id, name, description, template_count)
                    VALUES (?, ?, ?, 0)
                    """,
                    (cat_id, name, desc),
                )

    # =========================================================================
    # Template Operations
    # =========================================================================

    def create_template(
        self,
        name: str,
        description: str,
        author_id: str,
        author_name: str,
        category: str,
        pattern: str,
        workflow_definition: dict[str, Any],
        tags: Optional[list[str]] = None,
    ) -> StoredTemplate:
        """Create a new template.

        Args:
            name: Template name (must be unique)
            description: Template description
            author_id: Author's user ID
            author_name: Author's display name
            category: Category ID
            pattern: Workflow pattern name
            workflow_definition: Full workflow definition
            tags: Optional list of tags

        Returns:
            Created template

        Raises:
            ValueError: If name already exists
        """
        template_id = f"tpl-{uuid.uuid4().hex[:12]}"
        now = time.time()
        tags = tags or []

        template = StoredTemplate(
            id=template_id,
            name=name,
            description=description,
            author_id=author_id,
            author_name=author_name,
            category=category,
            pattern=pattern,
            tags=tags,
            workflow_definition=workflow_definition,
            created_at=now,
            updated_at=now,
        )

        with self.connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO templates (
                        id, name, description, author_id, author_name,
                        category, pattern, tags, workflow_definition,
                        download_count, rating_sum, rating_count,
                        is_featured, is_trending, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        template.id,
                        template.name,
                        template.description,
                        template.author_id,
                        template.author_name,
                        template.category,
                        template.pattern,
                        json.dumps(template.tags),
                        json.dumps(template.workflow_definition),
                        template.download_count,
                        template.rating_sum,
                        template.rating_count,
                        int(template.is_featured),
                        int(template.is_trending),
                        template.created_at,
                        template.updated_at,
                    ),
                )

                # Update category count
                conn.execute(
                    "UPDATE categories SET template_count = template_count + 1 WHERE id = ?",
                    (category,),
                )

            except Exception as e:
                if "UNIQUE constraint" in str(e):
                    raise ValueError(f"Template with name '{name}' already exists") from e
                raise

        logger.info(f"Created template: {template.id} ({template.name})")
        return template

    def get_template(self, template_id: str) -> Optional[StoredTemplate]:
        """Get a template by ID.

        Args:
            template_id: Template ID

        Returns:
            Template if found, None otherwise
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM templates WHERE id = ?", (template_id,)
            ).fetchone()

            if row is None:
                return None

            return self._row_to_template(row)

    def list_templates(
        self,
        category: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: str = "rating",
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[StoredTemplate], int]:
        """List templates with filtering and pagination.

        Args:
            category: Filter by category
            search: Search in name, description, and tags
            sort_by: Sort field (rating, downloads, newest, updated)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Tuple of (templates, total_count)
        """
        conditions = []
        params: list[Any] = []

        if category:
            conditions.append("category = ?")
            params.append(category)

        if search:
            conditions.append(
                "(name LIKE ? OR description LIKE ? OR tags LIKE ?)"
            )
            search_pattern = f"%{search}%"
            params.extend([search_pattern, search_pattern, search_pattern])

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        # Sort order
        order_clause = {
            "rating": "rating_sum / NULLIF(rating_count, 0) DESC",
            "downloads": "download_count DESC",
            "newest": "created_at DESC",
            "updated": "updated_at DESC",
        }.get(sort_by, "rating_sum / NULLIF(rating_count, 0) DESC")

        with self.connection() as conn:
            # Get total count
            count_row = conn.execute(
                f"SELECT COUNT(*) FROM templates{where_clause}", params
            ).fetchone()
            total = count_row[0] if count_row else 0

            # Get templates
            rows = conn.execute(
                f"""
                SELECT * FROM templates
                {where_clause}
                ORDER BY {order_clause}
                LIMIT ? OFFSET ?
                """,
                params + [limit, offset],
            ).fetchall()

            templates = [self._row_to_template(row) for row in rows]

        return templates, total

    def get_featured(self, limit: int = 10) -> list[StoredTemplate]:
        """Get featured templates.

        Args:
            limit: Maximum results

        Returns:
            List of featured templates
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM templates
                WHERE is_featured = 1
                ORDER BY rating_sum / NULLIF(rating_count, 0) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            return [self._row_to_template(row) for row in rows]

    def get_trending(self, limit: int = 10) -> list[StoredTemplate]:
        """Get trending templates (most downloads in recent period).

        Args:
            limit: Maximum results

        Returns:
            List of trending templates
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM templates
                WHERE is_trending = 1
                ORDER BY download_count DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            return [self._row_to_template(row) for row in rows]

    def increment_download(self, template_id: str) -> None:
        """Increment template download count.

        Args:
            template_id: Template ID
        """
        with self.connection() as conn:
            conn.execute(
                "UPDATE templates SET download_count = download_count + 1 WHERE id = ?",
                (template_id,),
            )

    def set_featured(self, template_id: str, featured: bool) -> None:
        """Set template featured status.

        Args:
            template_id: Template ID
            featured: Whether template is featured
        """
        with self.connection() as conn:
            conn.execute(
                "UPDATE templates SET is_featured = ? WHERE id = ?",
                (int(featured), template_id),
            )

    def set_trending(self, template_id: str, trending: bool) -> None:
        """Set template trending status.

        Args:
            template_id: Template ID
            trending: Whether template is trending
        """
        with self.connection() as conn:
            conn.execute(
                "UPDATE templates SET is_trending = ? WHERE id = ?",
                (int(trending), template_id),
            )

    # =========================================================================
    # Rating Operations
    # =========================================================================

    def rate_template(
        self, template_id: str, user_id: str, rating: int
    ) -> tuple[float, int]:
        """Rate a template.

        Args:
            template_id: Template ID
            user_id: User ID
            rating: Rating value (1-5)

        Returns:
            Tuple of (new_average_rating, total_rating_count)

        Raises:
            ValueError: If rating is out of range
        """
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")

        now = time.time()

        with self.connection() as conn:
            # Check for existing rating
            existing = conn.execute(
                "SELECT rating FROM ratings WHERE template_id = ? AND user_id = ?",
                (template_id, user_id),
            ).fetchone()

            if existing:
                old_rating = existing[0]
                # Update existing rating
                conn.execute(
                    """
                    UPDATE ratings SET rating = ?, created_at = ?
                    WHERE template_id = ? AND user_id = ?
                    """,
                    (rating, now, template_id, user_id),
                )
                # Update template rating sum (replace old with new)
                conn.execute(
                    """
                    UPDATE templates
                    SET rating_sum = rating_sum - ? + ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (old_rating, rating, now, template_id),
                )
            else:
                # Insert new rating
                conn.execute(
                    "INSERT INTO ratings (template_id, user_id, rating, created_at) VALUES (?, ?, ?, ?)",
                    (template_id, user_id, rating, now),
                )
                # Update template rating sum and count
                conn.execute(
                    """
                    UPDATE templates
                    SET rating_sum = rating_sum + ?, rating_count = rating_count + 1, updated_at = ?
                    WHERE id = ?
                    """,
                    (rating, now, template_id),
                )

            # Get updated stats
            row = conn.execute(
                "SELECT rating_sum, rating_count FROM templates WHERE id = ?",
                (template_id,),
            ).fetchone()

            if row:
                avg = row[0] / row[1] if row[1] > 0 else 0.0
                return (avg, row[1])

        return (0.0, 0)

    # =========================================================================
    # Review Operations
    # =========================================================================

    def create_review(
        self,
        template_id: str,
        user_id: str,
        user_name: str,
        rating: int,
        title: str,
        content: str,
    ) -> StoredReview:
        """Create a review for a template.

        Args:
            template_id: Template ID
            user_id: User ID
            user_name: User display name
            rating: Rating value (1-5)
            title: Review title
            content: Review content

        Returns:
            Created review
        """
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")

        review_id = f"rev-{uuid.uuid4().hex[:12]}"
        now = time.time()

        review = StoredReview(
            id=review_id,
            template_id=template_id,
            user_id=user_id,
            user_name=user_name,
            rating=rating,
            title=title,
            content=content,
            created_at=now,
        )

        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO reviews (
                    id, template_id, user_id, user_name, rating,
                    title, content, helpful_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review.id,
                    review.template_id,
                    review.user_id,
                    review.user_name,
                    review.rating,
                    review.title,
                    review.content,
                    review.helpful_count,
                    review.created_at,
                ),
            )

        # Also record as a rating
        self.rate_template(template_id, user_id, rating)

        return review

    def list_reviews(
        self, template_id: str, limit: int = 20, offset: int = 0
    ) -> list[StoredReview]:
        """List reviews for a template.

        Args:
            template_id: Template ID
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of reviews
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM reviews
                WHERE template_id = ?
                ORDER BY helpful_count DESC, created_at DESC
                LIMIT ? OFFSET ?
                """,
                (template_id, limit, offset),
            ).fetchall()

            return [self._row_to_review(row) for row in rows]

    # =========================================================================
    # Category Operations
    # =========================================================================

    def list_categories(self) -> list[dict[str, Any]]:
        """List all categories with template counts.

        Returns:
            List of category dicts
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, name, description, template_count
                FROM categories
                ORDER BY template_count DESC
                """
            ).fetchall()

            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "template_count": row[3],
                }
                for row in rows
            ]

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _row_to_template(self, row: tuple) -> StoredTemplate:
        """Convert database row to StoredTemplate."""
        return StoredTemplate(
            id=row[0],
            name=row[1],
            description=row[2],
            author_id=row[3],
            author_name=row[4],
            category=row[5],
            pattern=row[6],
            tags=json.loads(row[7]) if row[7] else [],
            workflow_definition=json.loads(row[8]) if row[8] else {},
            download_count=row[9],
            rating_sum=row[10],
            rating_count=row[11],
            is_featured=bool(row[12]),
            is_trending=bool(row[13]),
            created_at=row[14],
            updated_at=row[15],
        )

    def _row_to_review(self, row: tuple) -> StoredReview:
        """Convert database row to StoredReview."""
        return StoredReview(
            id=row[0],
            template_id=row[1],
            user_id=row[2],
            user_name=row[3],
            rating=row[4],
            title=row[5],
            content=row[6],
            helpful_count=row[7],
            created_at=row[8],
        )


# Default instance
_marketplace_store: Optional[MarketplaceStore] = None


def get_marketplace_store() -> MarketplaceStore:
    """Get or create the default marketplace store instance."""
    global _marketplace_store
    if _marketplace_store is None:
        _marketplace_store = MarketplaceStore()
    return _marketplace_store
