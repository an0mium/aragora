"""
SQLite-backed marketplace template store with production persistence.

Provides durable storage for community workflow templates, ratings, and reviews.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from aragora.storage.base_store import SQLiteStore

if TYPE_CHECKING:
    from asyncpg import Pool

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

    # Explicit columns for SELECT queries - prevents SELECT * data exposure
    _TEMPLATE_COLUMNS = (
        "id, name, description, author_id, author_name, category, pattern, "
        "tags, workflow_definition, download_count, rating_sum, rating_count, "
        "is_featured, is_trending, created_at, updated_at"
    )
    _REVIEW_COLUMNS = (
        "id, template_id, user_id, user_name, rating, title, content, " "helpful_count, created_at"
    )

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
                f"SELECT {self._TEMPLATE_COLUMNS} FROM templates WHERE id = ?",
                (template_id,),
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
            conditions.append("(name LIKE ? OR description LIKE ? OR tags LIKE ?)")
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
                SELECT {self._TEMPLATE_COLUMNS} FROM templates
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
                f"""
                SELECT {self._TEMPLATE_COLUMNS} FROM templates
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
                f"""
                SELECT {self._TEMPLATE_COLUMNS} FROM templates
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

    def list_templates_with_rank(
        self,
        category: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: str = "rating",
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List templates with ranking using window functions.

        Returns templates with their global rank and category rank in a single query,
        avoiding the N+1 pattern of separate COUNT queries.

        Args:
            category: Filter by category
            search: Search in name, description, and tags
            sort_by: Sort field (rating, downloads, newest, updated)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Tuple of (templates with rank info, total_count)
        """
        conditions = []
        params: list[Any] = []

        if category:
            conditions.append("category = ?")
            params.append(category)

        if search:
            conditions.append("(name LIKE ? OR description LIKE ? OR tags LIKE ?)")
            search_pattern = f"%{search}%"
            params.extend([search_pattern, search_pattern, search_pattern])

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        # Build ORDER BY expression for window function
        order_expr = {
            "rating": "rating_sum * 1.0 / NULLIF(rating_count, 0) DESC",
            "downloads": "download_count DESC",
            "newest": "created_at DESC",
            "updated": "updated_at DESC",
        }.get(sort_by, "rating_sum * 1.0 / NULLIF(rating_count, 0) DESC")

        with self.connection() as conn:
            # Single query with window functions for count and ranking
            rows = conn.execute(
                f"""
                SELECT
                    *,
                    ROW_NUMBER() OVER (ORDER BY {order_expr}) as global_rank,
                    ROW_NUMBER() OVER (PARTITION BY category ORDER BY {order_expr}) as category_rank,
                    COUNT(*) OVER () as total_count
                FROM templates
                {where_clause}
                ORDER BY {order_expr}
                LIMIT ? OFFSET ?
                """,
                params + [limit, offset],
            ).fetchall()

            if not rows:
                return [], 0

            # Total count is the same for all rows (from window function)
            total = rows[0][-1] if rows else 0

            # Build results with rank info
            results = []
            for row in rows:
                template = self._row_to_template(row)
                result = template.to_dict()
                result["global_rank"] = row[-3]  # global_rank
                result["category_rank"] = row[-2]  # category_rank
                results.append(result)

            return results, total

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

    def rate_template(self, template_id: str, user_id: str, rating: int) -> tuple[float, int]:
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
                f"""
                SELECT {self._REVIEW_COLUMNS} FROM reviews
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


class PostgresMarketplaceStore:
    """
    PostgreSQL-backed marketplace store.

    Async implementation for production multi-instance deployments
    with horizontal scaling and concurrent writes.
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
            tags JSONB NOT NULL DEFAULT '[]',
            workflow_definition JSONB NOT NULL DEFAULT '{}',
            download_count INTEGER DEFAULT 0,
            rating_sum REAL DEFAULT 0.0,
            rating_count INTEGER DEFAULT 0,
            is_featured BOOLEAN DEFAULT FALSE,
            is_trending BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_templates_category ON templates(category);
        CREATE INDEX IF NOT EXISTS idx_templates_author ON templates(author_id);
        CREATE INDEX IF NOT EXISTS idx_templates_rating ON templates((rating_sum / NULLIF(rating_count, 0)));
        CREATE INDEX IF NOT EXISTS idx_templates_downloads ON templates(download_count);
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
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT fk_reviews_template FOREIGN KEY (template_id) REFERENCES templates(id) ON DELETE CASCADE,
            UNIQUE(template_id, user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_reviews_template ON reviews(template_id);
        CREATE INDEX IF NOT EXISTS idx_reviews_user ON reviews(user_id);

        -- Ratings table (individual ratings, not reviews)
        CREATE TABLE IF NOT EXISTS ratings (
            template_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            rating INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (template_id, user_id),
            CONSTRAINT fk_ratings_template FOREIGN KEY (template_id) REFERENCES templates(id) ON DELETE CASCADE
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

    def __init__(self, pool: "Pool"):
        """Initialize the PostgreSQL marketplace store.

        Args:
            pool: asyncpg connection pool
        """
        self._pool = pool
        self._initialized = False
        logger.info("PostgresMarketplaceStore initialized")

    async def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        async with self._pool.acquire() as conn:
            await conn.execute(self.INITIAL_SCHEMA)

        self._initialized = True
        logger.debug(f"[{self.SCHEMA_NAME}] Schema initialized")

        # Initialize default categories
        await self._init_categories_async()

    async def _init_categories_async(self) -> None:
        """Initialize default categories."""
        default_categories = [
            ("security", "Security", "Security analysis and vulnerability scanning"),
            ("code-review", "Code Review", "Code review and quality analysis"),
            ("compliance", "Compliance", "Regulatory compliance and auditing"),
            ("legal", "Legal", "Legal document review and analysis"),
            ("healthcare", "Healthcare", "Healthcare and medical workflows"),
            ("testing", "Testing", "Test generation and validation"),
            ("research", "Research", "Research and analysis workflows"),
        ]

        async with self._pool.acquire() as conn:
            for cat_id, name, desc in default_categories:
                await conn.execute(
                    """
                    INSERT INTO categories (id, name, description, template_count)
                    VALUES ($1, $2, $3, 0)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    cat_id,
                    name,
                    desc,
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
        """Create a new template (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.create_template_async(
                name,
                description,
                author_id,
                author_name,
                category,
                pattern,
                workflow_definition,
                tags,
            )
        )

    async def create_template_async(
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
        """Create a new template asynchronously."""
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

        async with self._pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO templates (
                        id, name, description, author_id, author_name,
                        category, pattern, tags, workflow_definition,
                        download_count, rating_sum, rating_count,
                        is_featured, is_trending, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, to_timestamp($15), to_timestamp($16))
                    """,
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
                    template.is_featured,
                    template.is_trending,
                    template.created_at,
                    template.updated_at,
                )

                # Update category count
                await conn.execute(
                    "UPDATE categories SET template_count = template_count + 1 WHERE id = $1",
                    category,
                )

            except Exception as e:
                if "duplicate key" in str(e).lower() or "unique" in str(e).lower():
                    raise ValueError(f"Template with name '{name}' already exists") from e
                raise

        logger.info(f"Created template: {template.id} ({template.name})")
        return template

    def get_template(self, template_id: str) -> Optional[StoredTemplate]:
        """Get a template by ID (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(self.get_template_async(template_id))

    async def get_template_async(self, template_id: str) -> Optional[StoredTemplate]:
        """Get a template by ID asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, name, description, author_id, author_name, category, pattern,
                       tags, workflow_definition, download_count, rating_sum, rating_count,
                       is_featured, is_trending,
                       EXTRACT(EPOCH FROM created_at) as created_at,
                       EXTRACT(EPOCH FROM updated_at) as updated_at
                FROM templates WHERE id = $1
                """,
                template_id,
            )

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
        """List templates (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.list_templates_async(category, search, sort_by, limit, offset)
        )

    async def list_templates_async(
        self,
        category: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: str = "rating",
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[StoredTemplate], int]:
        """List templates asynchronously."""
        conditions = []
        params: list[Any] = []
        param_idx = 1

        if category:
            conditions.append(f"category = ${param_idx}")
            params.append(category)
            param_idx += 1

        if search:
            conditions.append(
                f"(name ILIKE ${param_idx} OR description ILIKE ${param_idx + 1} OR tags::text ILIKE ${param_idx + 2})"
            )
            search_pattern = f"%{search}%"
            params.extend([search_pattern, search_pattern, search_pattern])
            param_idx += 3

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        # Sort order
        order_clause = {
            "rating": "rating_sum / NULLIF(rating_count, 0) DESC NULLS LAST",
            "downloads": "download_count DESC",
            "newest": "created_at DESC",
            "updated": "updated_at DESC",
        }.get(sort_by, "rating_sum / NULLIF(rating_count, 0) DESC NULLS LAST")

        async with self._pool.acquire() as conn:
            # Get total count
            count_row = await conn.fetchrow(
                f"SELECT COUNT(*) FROM templates{where_clause}", *params
            )
            total = count_row[0] if count_row else 0

            # Get templates
            query_params = params + [limit, offset]
            rows = await conn.fetch(
                f"""
                SELECT id, name, description, author_id, author_name, category, pattern,
                       tags, workflow_definition, download_count, rating_sum, rating_count,
                       is_featured, is_trending,
                       EXTRACT(EPOCH FROM created_at) as created_at,
                       EXTRACT(EPOCH FROM updated_at) as updated_at
                FROM templates
                {where_clause}
                ORDER BY {order_clause}
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
                """,
                *query_params,
            )

            templates = [self._row_to_template(row) for row in rows]

        return templates, total

    def get_featured(self, limit: int = 10) -> list[StoredTemplate]:
        """Get featured templates (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(self.get_featured_async(limit))

    async def get_featured_async(self, limit: int = 10) -> list[StoredTemplate]:
        """Get featured templates asynchronously."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, name, description, author_id, author_name, category, pattern,
                       tags, workflow_definition, download_count, rating_sum, rating_count,
                       is_featured, is_trending,
                       EXTRACT(EPOCH FROM created_at) as created_at,
                       EXTRACT(EPOCH FROM updated_at) as updated_at
                FROM templates
                WHERE is_featured = TRUE
                ORDER BY rating_sum / NULLIF(rating_count, 0) DESC NULLS LAST
                LIMIT $1
                """,
                limit,
            )

            return [self._row_to_template(row) for row in rows]

    def get_trending(self, limit: int = 10) -> list[StoredTemplate]:
        """Get trending templates (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(self.get_trending_async(limit))

    async def get_trending_async(self, limit: int = 10) -> list[StoredTemplate]:
        """Get trending templates asynchronously."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, name, description, author_id, author_name, category, pattern,
                       tags, workflow_definition, download_count, rating_sum, rating_count,
                       is_featured, is_trending,
                       EXTRACT(EPOCH FROM created_at) as created_at,
                       EXTRACT(EPOCH FROM updated_at) as updated_at
                FROM templates
                WHERE is_trending = TRUE
                ORDER BY download_count DESC
                LIMIT $1
                """,
                limit,
            )

            return [self._row_to_template(row) for row in rows]

    def increment_download(self, template_id: str) -> None:
        """Increment download count (sync wrapper for async)."""
        asyncio.get_event_loop().run_until_complete(self.increment_download_async(template_id))

    async def increment_download_async(self, template_id: str) -> None:
        """Increment download count asynchronously."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE templates SET download_count = download_count + 1 WHERE id = $1",
                template_id,
            )

    def set_featured(self, template_id: str, featured: bool) -> None:
        """Set featured status (sync wrapper for async)."""
        asyncio.get_event_loop().run_until_complete(self.set_featured_async(template_id, featured))

    async def set_featured_async(self, template_id: str, featured: bool) -> None:
        """Set featured status asynchronously."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE templates SET is_featured = $1 WHERE id = $2",
                featured,
                template_id,
            )

    def set_trending(self, template_id: str, trending: bool) -> None:
        """Set trending status (sync wrapper for async)."""
        asyncio.get_event_loop().run_until_complete(self.set_trending_async(template_id, trending))

    async def set_trending_async(self, template_id: str, trending: bool) -> None:
        """Set trending status asynchronously."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE templates SET is_trending = $1 WHERE id = $2",
                trending,
                template_id,
            )

    # =========================================================================
    # Rating Operations
    # =========================================================================

    def rate_template(self, template_id: str, user_id: str, rating: int) -> tuple[float, int]:
        """Rate a template (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.rate_template_async(template_id, user_id, rating)
        )

    async def rate_template_async(
        self, template_id: str, user_id: str, rating: int
    ) -> tuple[float, int]:
        """Rate a template asynchronously."""
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")

        now = time.time()

        async with self._pool.acquire() as conn:
            # Check for existing rating
            existing = await conn.fetchrow(
                "SELECT rating FROM ratings WHERE template_id = $1 AND user_id = $2",
                template_id,
                user_id,
            )

            if existing:
                old_rating = existing[0]
                # Update existing rating
                await conn.execute(
                    """
                    UPDATE ratings SET rating = $1, created_at = to_timestamp($2)
                    WHERE template_id = $3 AND user_id = $4
                    """,
                    rating,
                    now,
                    template_id,
                    user_id,
                )
                # Update template rating sum (replace old with new)
                await conn.execute(
                    """
                    UPDATE templates
                    SET rating_sum = rating_sum - $1 + $2, updated_at = to_timestamp($3)
                    WHERE id = $4
                    """,
                    old_rating,
                    rating,
                    now,
                    template_id,
                )
            else:
                # Insert new rating
                await conn.execute(
                    "INSERT INTO ratings (template_id, user_id, rating, created_at) VALUES ($1, $2, $3, to_timestamp($4))",
                    template_id,
                    user_id,
                    rating,
                    now,
                )
                # Update template rating sum and count
                await conn.execute(
                    """
                    UPDATE templates
                    SET rating_sum = rating_sum + $1, rating_count = rating_count + 1, updated_at = to_timestamp($2)
                    WHERE id = $3
                    """,
                    rating,
                    now,
                    template_id,
                )

            # Get updated stats
            row = await conn.fetchrow(
                "SELECT rating_sum, rating_count FROM templates WHERE id = $1",
                template_id,
            )

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
        """Create a review (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.create_review_async(template_id, user_id, user_name, rating, title, content)
        )

    async def create_review_async(
        self,
        template_id: str,
        user_id: str,
        user_name: str,
        rating: int,
        title: str,
        content: str,
    ) -> StoredReview:
        """Create a review asynchronously."""
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

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO reviews (
                    id, template_id, user_id, user_name, rating,
                    title, content, helpful_count, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, to_timestamp($9))
                ON CONFLICT (template_id, user_id) DO UPDATE SET
                    id = EXCLUDED.id, rating = EXCLUDED.rating, title = EXCLUDED.title,
                    content = EXCLUDED.content, created_at = EXCLUDED.created_at
                """,
                review.id,
                review.template_id,
                review.user_id,
                review.user_name,
                review.rating,
                review.title,
                review.content,
                review.helpful_count,
                review.created_at,
            )

        # Also record as a rating
        await self.rate_template_async(template_id, user_id, rating)

        return review

    def list_reviews(
        self, template_id: str, limit: int = 20, offset: int = 0
    ) -> list[StoredReview]:
        """List reviews (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.list_reviews_async(template_id, limit, offset)
        )

    async def list_reviews_async(
        self, template_id: str, limit: int = 20, offset: int = 0
    ) -> list[StoredReview]:
        """List reviews asynchronously."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, template_id, user_id, user_name, rating, title, content,
                       helpful_count, EXTRACT(EPOCH FROM created_at) as created_at
                FROM reviews
                WHERE template_id = $1
                ORDER BY helpful_count DESC, created_at DESC
                LIMIT $2 OFFSET $3
                """,
                template_id,
                limit,
                offset,
            )

            return [self._row_to_review(row) for row in rows]

    # =========================================================================
    # Category Operations
    # =========================================================================

    def list_categories(self) -> list[dict[str, Any]]:
        """List categories (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(self.list_categories_async())

    async def list_categories_async(self) -> list[dict[str, Any]]:
        """List categories asynchronously."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, name, description, template_count
                FROM categories
                ORDER BY template_count DESC
                """
            )

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

    def _row_to_template(self, row: Any) -> StoredTemplate:
        """Convert database row to StoredTemplate."""
        tags_data = row["tags"]
        if isinstance(tags_data, str):
            tags = json.loads(tags_data)
        else:
            tags = tags_data if tags_data else []

        workflow_data = row["workflow_definition"]
        if isinstance(workflow_data, str):
            workflow = json.loads(workflow_data)
        else:
            workflow = workflow_data if workflow_data else {}

        return StoredTemplate(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            author_id=row["author_id"],
            author_name=row["author_name"],
            category=row["category"],
            pattern=row["pattern"],
            tags=tags,
            workflow_definition=workflow,
            download_count=row["download_count"],
            rating_sum=row["rating_sum"],
            rating_count=row["rating_count"],
            is_featured=bool(row["is_featured"]),
            is_trending=bool(row["is_trending"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_review(self, row: Any) -> StoredReview:
        """Convert database row to StoredReview."""
        return StoredReview(
            id=row["id"],
            template_id=row["template_id"],
            user_id=row["user_id"],
            user_name=row["user_name"],
            rating=row["rating"],
            title=row["title"],
            content=row["content"],
            helpful_count=row["helpful_count"],
            created_at=row["created_at"],
        )

    def close(self) -> None:
        """Close is a no-op for pool-based stores (pool managed externally)."""
        pass


# Default instance
_marketplace_store: Optional[Union[MarketplaceStore, PostgresMarketplaceStore]] = None


def get_marketplace_store() -> Union[MarketplaceStore, PostgresMarketplaceStore]:
    """
    Get or create the default marketplace store instance.

    Backend selection (in preference order):
    1. Supabase PostgreSQL (if SUPABASE_URL + SUPABASE_DB_PASSWORD configured)
    2. Self-hosted PostgreSQL (if DATABASE_URL or ARAGORA_POSTGRES_DSN configured)
    3. SQLite (fallback, with production warning)

    Override via:
    - ARAGORA_MARKETPLACE_STORE_BACKEND: "sqlite", "postgres", or "supabase"
    - ARAGORA_DB_BACKEND: Global override

    Returns:
        Configured marketplace store instance
    """
    global _marketplace_store
    if _marketplace_store is not None:
        return _marketplace_store

    from aragora.storage.connection_factory import create_persistent_store

    _marketplace_store = create_persistent_store(
        store_name="marketplace",
        sqlite_class=MarketplaceStore,
        postgres_class=PostgresMarketplaceStore,
        db_filename="marketplace.db",
    )

    return _marketplace_store


def set_marketplace_store(store: Union[MarketplaceStore, PostgresMarketplaceStore]) -> None:
    """Set a custom marketplace store instance."""
    global _marketplace_store
    _marketplace_store = store


def reset_marketplace_store() -> None:
    """Reset the global marketplace store (for testing)."""
    global _marketplace_store
    _marketplace_store = None


__all__ = [
    "StoredTemplate",
    "StoredReview",
    "MarketplaceStore",
    "PostgresMarketplaceStore",
    "get_marketplace_store",
    "set_marketplace_store",
    "reset_marketplace_store",
]
