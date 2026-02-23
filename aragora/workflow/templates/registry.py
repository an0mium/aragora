"""Public Template Registry.

SQLite-backed registry for community-submitted workflow templates.
Supports submission, approval, search, installation, and ratings.

Follows the SkillMarketplace pattern for consistency.

Usage:
    from aragora.workflow.templates.registry import TemplateRegistry, get_template_registry

    registry = get_template_registry()

    # Search templates
    results = registry.search("contract review")

    # Submit a template
    listing_id = registry.submit(template_data, author="user-1")

    # Install a template
    definition = registry.install(listing_id)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TemplateStatus(str, Enum):
    """Status of a template submission."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class TemplateCategory(str, Enum):
    """Categories for template organization."""

    LEGAL = "legal"
    HEALTHCARE = "healthcare"
    CODE = "code"
    ACCOUNTING = "accounting"
    AI_ML = "ai_ml"
    DEVOPS = "devops"
    PRODUCT = "product"
    MARKETING = "marketing"
    SME = "sme"
    GENERAL = "general"
    CUSTOM = "custom"


@dataclass
class RegistryListing:
    """A template listing in the registry."""

    id: str
    name: str
    description: str
    category: str
    author_id: str
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    status: TemplateStatus = TemplateStatus.PENDING
    is_verified: bool = False
    is_builtin: bool = False
    install_count: int = 0
    rating_average: float = 0.0
    rating_count: int = 0
    template_data: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    approved_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "author_id": self.author_id,
            "version": self.version,
            "tags": self.tags,
            "status": self.status.value if isinstance(self.status, TemplateStatus) else self.status,
            "is_verified": self.is_verified,
            "is_builtin": self.is_builtin,
            "install_count": self.install_count,
            "rating_average": self.rating_average,
            "rating_count": self.rating_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "approved_by": self.approved_by,
        }


class TemplateRegistry:
    """SQLite-backed template registry."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or os.path.join(
            os.environ.get("ARAGORA_DATA_DIR", ".aragora_data"),
            "template_registry.db",
        )
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._initialized = False

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._init_db()
        return self._conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        conn = self._conn
        if conn is None:
            return

        with self._lock:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS template_registry (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    category TEXT DEFAULT 'general',
                    author_id TEXT NOT NULL,
                    version TEXT DEFAULT '1.0.0',
                    tags TEXT DEFAULT '[]',
                    status TEXT DEFAULT 'pending',
                    is_verified INTEGER DEFAULT 0,
                    is_builtin INTEGER DEFAULT 0,
                    install_count INTEGER DEFAULT 0,
                    template_data TEXT DEFAULT '{}',
                    created_at TEXT,
                    updated_at TEXT,
                    approved_by TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS template_ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    template_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    review TEXT,
                    created_at TEXT,
                    UNIQUE(template_id, user_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS template_installs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    template_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    installed_at TEXT
                )
            """)

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_registry_category ON template_registry(category)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_registry_status ON template_registry(status)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_registry_author ON template_registry(author_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_ratings_template ON template_ratings(template_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_installs_template ON template_installs(template_id)"
            )

            conn.commit()
            self._initialized = True
            logger.info("Template registry database initialized")

    # =========================================================================
    # Submission
    # =========================================================================

    def submit(
        self,
        template_data: dict[str, Any],
        name: str,
        description: str,
        category: str = "general",
        author_id: str = "",
        tags: list[str] | None = None,
        version: str = "1.0.0",
    ) -> str:
        """Submit a new template to the registry.

        Args:
            template_data: The workflow template definition.
            name: Display name for the template.
            description: Human-readable description.
            category: Template category.
            author_id: ID of the submitting user.
            tags: Optional list of tags for discovery.
            version: Semantic version string.

        Returns:
            The new listing ID.
        """
        listing_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        with self._lock:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO template_registry (
                    id, name, description, category, author_id,
                    version, tags, status, template_data,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    listing_id,
                    name,
                    description,
                    category,
                    author_id,
                    version,
                    json.dumps(tags or []),
                    TemplateStatus.PENDING.value,
                    json.dumps(template_data),
                    now,
                    now,
                ),
            )
            conn.commit()

        logger.info("Template submitted: %s by %s", listing_id, author_id)
        return listing_id

    # =========================================================================
    # Admin Actions
    # =========================================================================

    def approve(self, listing_id: str, approved_by: str) -> bool:
        """Approve a pending template.

        Args:
            listing_id: ID of the listing to approve.
            approved_by: ID of the admin approving.

        Returns:
            True if found and updated.
        """
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        with self._lock:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE template_registry
                SET status = ?, approved_by = ?, updated_at = ?
                WHERE id = ?
                """,
                (TemplateStatus.APPROVED.value, approved_by, now, listing_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def reject(self, listing_id: str, reason: str | None = None) -> bool:
        """Reject a pending template.

        Args:
            listing_id: ID of the listing to reject.
            reason: Optional rejection reason (logged, not stored).

        Returns:
            True if found and updated.
        """
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        with self._lock:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE template_registry
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (TemplateStatus.REJECTED.value, now, listing_id),
            )
            conn.commit()
            if cursor.rowcount > 0:
                if reason:
                    logger.info("Template %s rejected: %s", listing_id, reason)
                return True
            return False

    # =========================================================================
    # Discovery
    # =========================================================================

    def search(
        self,
        query: str = "",
        category: str | None = None,
        tags: list[str] | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[RegistryListing]:
        """Search the template registry.

        Args:
            query: Free-text search over name and description.
            category: Filter by category.
            tags: Filter by tags (all must match).
            status: Filter by status (default: approved only).
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            List of matching RegistryListings.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        sql = "SELECT * FROM template_registry WHERE 1=1"
        params: list[Any] = []

        if status:
            sql += " AND status = ?"
            params.append(status)
        else:
            sql += " AND status = ?"
            params.append(TemplateStatus.APPROVED.value)

        if query:
            sql += " AND (name LIKE ? OR description LIKE ?)"
            search_term = f"%{query}%"
            params.extend([search_term, search_term])

        if category:
            sql += " AND category = ?"
            params.append(category)

        if tags:
            for tag in tags:
                sql += " AND tags LIKE ?"
                params.append(f'%"{tag}"%')

        sql += " ORDER BY install_count DESC"
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        return [self._row_to_listing(row) for row in rows]

    def get(self, listing_id: str) -> RegistryListing | None:
        """Get a single listing by ID.

        Args:
            listing_id: The listing identifier.

        Returns:
            RegistryListing if found, None otherwise.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM template_registry WHERE id = ?", (listing_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_listing(row)

    def list_builtin(self) -> list[RegistryListing]:
        """Return all built-in template listings.

        Returns:
            List of RegistryListings where is_builtin is True.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM template_registry WHERE is_builtin = 1")
        rows = cursor.fetchall()
        return [self._row_to_listing(row) for row in rows]

    # =========================================================================
    # Installation
    # =========================================================================

    def install(self, listing_id: str, user_id: str = "") -> dict[str, Any]:
        """Install a template and return its data.

        Increments the install count and records the installation.

        Args:
            listing_id: ID of the template to install.
            user_id: ID of the installing user.

        Returns:
            The template_data dict.

        Raises:
            ValueError: If the listing is not found or not approved.
        """
        listing = self.get(listing_id)
        if listing is None:
            raise ValueError(f"Template not found: {listing_id}")

        status_val = (
            listing.status.value if isinstance(listing.status, TemplateStatus) else listing.status
        )
        if status_val != TemplateStatus.APPROVED.value and not listing.is_builtin:
            raise ValueError(f"Template is not approved: {listing_id}")

        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        with self._lock:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE template_registry SET install_count = install_count + 1, updated_at = ? WHERE id = ?",
                (now, listing_id),
            )
            cursor.execute(
                "INSERT INTO template_installs (template_id, user_id, installed_at) VALUES (?, ?, ?)",
                (listing_id, user_id, now),
            )
            conn.commit()

        logger.info("Template %s installed by %s", listing_id, user_id)
        return listing.template_data

    # =========================================================================
    # Ratings
    # =========================================================================

    def rate(
        self,
        listing_id: str,
        user_id: str,
        rating: int,
        review: str | None = None,
    ) -> bool:
        """Rate a template.

        Inserts or updates a rating, then recalculates the listing average.

        Args:
            listing_id: ID of the template to rate.
            user_id: ID of the rating user.
            rating: Integer rating from 1 to 5.
            review: Optional review text.

        Returns:
            True on success.

        Raises:
            ValueError: If rating is not between 1 and 5.
        """
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")

        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        with self._lock:
            cursor = conn.cursor()

            # Upsert rating
            cursor.execute(
                "SELECT id FROM template_ratings WHERE template_id = ? AND user_id = ?",
                (listing_id, user_id),
            )
            existing = cursor.fetchone()

            if existing:
                cursor.execute(
                    "UPDATE template_ratings SET rating = ?, review = ?, created_at = ? WHERE template_id = ? AND user_id = ?",
                    (rating, review, now, listing_id, user_id),
                )
            else:
                cursor.execute(
                    "INSERT INTO template_ratings (template_id, user_id, rating, review, created_at) VALUES (?, ?, ?, ?, ?)",
                    (listing_id, user_id, rating, review, now),
                )

            # Recalculate average
            cursor.execute(
                "SELECT AVG(rating) as avg_rating, COUNT(*) as cnt FROM template_ratings WHERE template_id = ?",
                (listing_id,),
            )
            stats = cursor.fetchone()
            avg = stats["avg_rating"] or 0.0
            cnt = stats["cnt"] or 0

            cursor.execute(
                "UPDATE template_registry SET rating_average = ?, rating_count = ?, updated_at = ? WHERE id = ?",
                (avg, cnt, now, listing_id),
            )
            conn.commit()

        logger.info("User %s rated template %s: %d stars", user_id, listing_id, rating)
        return True

    # =========================================================================
    # Analytics
    # =========================================================================

    def get_analytics(self, listing_id: str) -> dict[str, Any]:
        """Get analytics for a template listing.

        Args:
            listing_id: The listing identifier.

        Returns:
            Dict with install_count, rating stats, and recent installs.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT install_count FROM template_registry WHERE id = ?",
            (listing_id,),
        )
        row = cursor.fetchone()
        install_count = row["install_count"] if row else 0

        cursor.execute(
            "SELECT AVG(rating) as avg_rating, COUNT(*) as cnt FROM template_ratings WHERE template_id = ?",
            (listing_id,),
        )
        stats = cursor.fetchone()

        cursor.execute(
            "SELECT user_id, installed_at FROM template_installs WHERE template_id = ? ORDER BY installed_at DESC LIMIT 10",
            (listing_id,),
        )
        recent = [
            {"user_id": r["user_id"], "installed_at": r["installed_at"]} for r in cursor.fetchall()
        ]

        return {
            "listing_id": listing_id,
            "install_count": install_count,
            "rating_average": round(stats["avg_rating"] or 0.0, 2),
            "rating_count": stats["cnt"] or 0,
            "recent_installs": recent,
        }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _row_to_listing(self, row: sqlite3.Row) -> RegistryListing:
        """Convert a database row to a RegistryListing."""
        rating_count = row["rating_count"] if "rating_count" in row.keys() else 0

        # rating_average is stored directly (recalculated on rate())
        rating_average = row["rating_average"] if "rating_average" in row.keys() else 0.0

        # If no rating_average column yet, compute from rating_count
        if rating_average is None:
            rating_average = 0.0

        try:
            status = TemplateStatus(row["status"])
        except (ValueError, KeyError):
            status = TemplateStatus.PENDING

        return RegistryListing(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            category=row["category"] or "general",
            author_id=row["author_id"],
            version=row["version"] or "1.0.0",
            tags=json.loads(row["tags"] or "[]"),
            status=status,
            is_verified=bool(row["is_verified"]),
            is_builtin=bool(row["is_builtin"]),
            install_count=row["install_count"] or 0,
            rating_average=rating_average,
            rating_count=rating_count or 0,
            template_data=json.loads(row["template_data"] or "{}"),
            created_at=row["created_at"] or "",
            updated_at=row["updated_at"] or "",
            approved_by=row["approved_by"],
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._initialized = False


# =========================================================================
# Singleton Accessor
# =========================================================================

_registry_instance: TemplateRegistry | None = None
_registry_lock = threading.Lock()


def get_template_registry(db_path: str | None = None) -> TemplateRegistry:
    """Get the global template registry instance."""
    global _registry_instance
    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = TemplateRegistry(db_path)
    return _registry_instance
