"""
Skill Marketplace.

A registry for published skills that supports:
- Skill publishing and versioning
- Search and discovery
- Ratings and reviews
- RBAC-gated installation

Usage:
    from aragora.skills.marketplace import SkillMarketplace, get_marketplace

    marketplace = get_marketplace()

    # Search for skills
    results = await marketplace.search("web search")

    # Get skill details
    listing = await marketplace.get_skill("web-search-skill")

    # Install a skill
    result = await marketplace.install("web-search-skill", tenant_id="tenant-1")

    # Rate a skill
    await marketplace.rate("web-search-skill", user_id="user-1", rating=5)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import Skill, SkillCapability

logger = logging.getLogger(__name__)


class SkillTier(str, Enum):
    """Skill tiers for access control and pricing."""

    FREE = "free"  # Available to all users
    STANDARD = "standard"  # Requires standard subscription
    PREMIUM = "premium"  # Requires premium subscription
    ENTERPRISE = "enterprise"  # Enterprise-only


class SkillCategory(str, Enum):
    """Skill categories for organization."""

    DATA_ANALYSIS = "data_analysis"
    WEB_TOOLS = "web_tools"
    CODE_EXECUTION = "code_execution"
    COMMUNICATION = "communication"
    KNOWLEDGE = "knowledge"
    AUTOMATION = "automation"
    INTEGRATIONS = "integrations"
    DEBATE = "debate"
    CUSTOM = "custom"


@dataclass
class SkillRating:
    """A user rating for a skill."""

    skill_id: str
    user_id: str
    rating: int  # 1-5 stars
    review: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    helpful_votes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "skill_id": self.skill_id,
            "user_id": self.user_id,
            "rating": self.rating,
            "review": self.review,
            "created_at": self.created_at.isoformat(),
            "helpful_votes": self.helpful_votes,
        }


@dataclass
class SkillVersion:
    """A specific version of a skill."""

    version: str
    changelog: str = ""
    published_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    downloads: int = 0
    is_stable: bool = True
    min_aragora_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "version": self.version,
            "changelog": self.changelog,
            "published_at": self.published_at.isoformat(),
            "downloads": self.downloads,
            "is_stable": self.is_stable,
            "min_aragora_version": self.min_aragora_version,
        }


@dataclass
class SkillDependency:
    """A dependency on another skill."""

    skill_id: str
    version_constraint: str = "*"  # Semver constraint
    optional: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "skill_id": self.skill_id,
            "version_constraint": self.version_constraint,
            "optional": self.optional,
        }


@dataclass
class SkillListing:
    """
    A skill listing in the marketplace.

    Contains all metadata for discovery, installation, and rating.
    """

    skill_id: str
    name: str
    description: str
    author_id: str
    author_name: str

    # Classification
    category: SkillCategory = SkillCategory.CUSTOM
    tier: SkillTier = SkillTier.FREE
    tags: List[str] = field(default_factory=list)

    # Versioning
    current_version: str = "1.0.0"
    versions: List[SkillVersion] = field(default_factory=list)

    # Dependencies
    dependencies: List[SkillDependency] = field(default_factory=list)

    # Capabilities (from manifest)
    capabilities: List[SkillCapability] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)

    # Metrics
    install_count: int = 0
    active_installs: int = 0
    rating_average: float = 0.0
    rating_count: int = 0

    # Status
    is_published: bool = False
    is_verified: bool = False
    is_deprecated: bool = False

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    published_at: Optional[datetime] = None

    # Additional metadata
    homepage_url: Optional[str] = None
    repository_url: Optional[str] = None
    documentation_url: Optional[str] = None
    icon_url: Optional[str] = None
    screenshots: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "category": self.category.value,
            "tier": self.tier.value,
            "tags": self.tags,
            "current_version": self.current_version,
            "versions": [v.to_dict() for v in self.versions],
            "dependencies": [d.to_dict() for d in self.dependencies],
            "capabilities": [c.value for c in self.capabilities],
            "required_permissions": self.required_permissions,
            "install_count": self.install_count,
            "active_installs": self.active_installs,
            "rating_average": round(self.rating_average, 1),
            "rating_count": self.rating_count,
            "is_published": self.is_published,
            "is_verified": self.is_verified,
            "is_deprecated": self.is_deprecated,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "homepage_url": self.homepage_url,
            "repository_url": self.repository_url,
            "documentation_url": self.documentation_url,
            "icon_url": self.icon_url,
            "screenshots": self.screenshots,
        }


@dataclass
class InstallResult:
    """Result of a skill installation."""

    success: bool
    skill_id: str
    version: str
    error: Optional[str] = None
    installed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    dependencies_installed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "skill_id": self.skill_id,
            "version": self.version,
            "error": self.error,
            "installed_at": self.installed_at.isoformat(),
            "dependencies_installed": self.dependencies_installed,
        }


class SkillMarketplace:
    """
    Marketplace for skill discovery, installation, and rating.

    Uses SQLite for persistent storage of skill listings, ratings,
    and installation records.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the marketplace.

        Args:
            db_path: Path to SQLite database file. If None, uses in-memory DB.
        """
        self._db_path = db_path or os.environ.get(
            "ARAGORA_MARKETPLACE_DB",
            ":memory:",
        )
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._initialized = False

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        conn = self._conn
        if conn is None:
            return

        with self._lock:
            cursor = conn.cursor()

            # Skills table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    skill_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    author_id TEXT NOT NULL,
                    author_name TEXT NOT NULL,
                    category TEXT DEFAULT 'custom',
                    tier TEXT DEFAULT 'free',
                    tags TEXT DEFAULT '[]',
                    current_version TEXT DEFAULT '1.0.0',
                    versions TEXT DEFAULT '[]',
                    dependencies TEXT DEFAULT '[]',
                    capabilities TEXT DEFAULT '[]',
                    required_permissions TEXT DEFAULT '[]',
                    install_count INTEGER DEFAULT 0,
                    active_installs INTEGER DEFAULT 0,
                    rating_sum REAL DEFAULT 0,
                    rating_count INTEGER DEFAULT 0,
                    is_published INTEGER DEFAULT 0,
                    is_verified INTEGER DEFAULT 0,
                    is_deprecated INTEGER DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT,
                    published_at TEXT,
                    homepage_url TEXT,
                    repository_url TEXT,
                    documentation_url TEXT,
                    icon_url TEXT,
                    screenshots TEXT DEFAULT '[]'
                )
            """)

            # Ratings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    review TEXT,
                    created_at TEXT,
                    helpful_votes INTEGER DEFAULT 0,
                    UNIQUE(skill_id, user_id)
                )
            """)

            # Installations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS installations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    installed_at TEXT,
                    uninstalled_at TEXT,
                    is_active INTEGER DEFAULT 1,
                    UNIQUE(skill_id, tenant_id)
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_skills_category ON skills(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_skills_author ON skills(author_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ratings_skill ON ratings(skill_id)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_installations_tenant ON installations(tenant_id)"
            )

            conn.commit()
            self._initialized = True
            logger.info("Skill marketplace database initialized")

    # ==========================================================================
    # Publishing
    # ==========================================================================

    async def publish(
        self,
        skill: Skill,
        author_id: str,
        author_name: str,
        category: SkillCategory = SkillCategory.CUSTOM,
        tier: SkillTier = SkillTier.FREE,
        changelog: str = "",
        **kwargs: Any,
    ) -> SkillListing:
        """
        Publish a skill to the marketplace.

        Args:
            skill: Skill instance to publish
            author_id: Publisher's user ID
            author_name: Publisher's display name
            category: Skill category
            tier: Skill tier (affects access)
            changelog: Version changelog
            **kwargs: Additional listing metadata

        Returns:
            SkillListing for the published skill
        """
        manifest = skill.manifest
        skill_id = f"{author_id}:{manifest.name}"

        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()

        # Check if skill already exists
        cursor = conn.cursor()
        cursor.execute("SELECT skill_id FROM skills WHERE skill_id = ?", (skill_id,))
        existing = cursor.fetchone()

        version = SkillVersion(
            version=manifest.version,
            changelog=changelog,
            published_at=datetime.now(timezone.utc),
        )

        if existing:
            # Update existing skill
            cursor.execute(
                """
                UPDATE skills SET
                    description = ?,
                    current_version = ?,
                    versions = ?,
                    capabilities = ?,
                    required_permissions = ?,
                    tags = ?,
                    updated_at = ?,
                    is_published = 1
                WHERE skill_id = ?
            """,
                (
                    manifest.description,
                    manifest.version,
                    json.dumps([version.to_dict()]),
                    json.dumps([c.value for c in manifest.capabilities]),
                    json.dumps(manifest.required_permissions),
                    json.dumps(manifest.tags),
                    now,
                    skill_id,
                ),
            )
        else:
            # Insert new skill
            cursor.execute(
                """
                INSERT INTO skills (
                    skill_id, name, description, author_id, author_name,
                    category, tier, tags, current_version, versions,
                    capabilities, required_permissions,
                    is_published, created_at, updated_at, published_at,
                    homepage_url, repository_url, documentation_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)
            """,
                (
                    skill_id,
                    manifest.name,
                    manifest.description,
                    author_id,
                    author_name,
                    category.value,
                    tier.value,
                    json.dumps(manifest.tags),
                    manifest.version,
                    json.dumps([version.to_dict()]),
                    json.dumps([c.value for c in manifest.capabilities]),
                    json.dumps(manifest.required_permissions),
                    now,
                    now,
                    now,
                    kwargs.get("homepage_url"),
                    kwargs.get("repository_url"),
                    kwargs.get("documentation_url"),
                ),
            )

        conn.commit()
        logger.info(f"Published skill: {skill_id} v{manifest.version}")

        return await self.get_skill(skill_id)  # type: ignore

    async def unpublish(self, skill_id: str, author_id: str) -> bool:
        """
        Unpublish a skill from the marketplace.

        Args:
            skill_id: Skill to unpublish
            author_id: Author ID (for authorization)

        Returns:
            True if unpublished successfully
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Verify ownership
        cursor.execute(
            "SELECT author_id FROM skills WHERE skill_id = ?",
            (skill_id,),
        )
        row = cursor.fetchone()
        if not row:
            return False
        if row["author_id"] != author_id:
            logger.warning(f"Unauthorized unpublish attempt for {skill_id}")
            return False

        # Mark as unpublished
        cursor.execute(
            "UPDATE skills SET is_published = 0, updated_at = ? WHERE skill_id = ?",
            (datetime.now(timezone.utc).isoformat(), skill_id),
        )
        conn.commit()
        logger.info(f"Unpublished skill: {skill_id}")
        return True

    # ==========================================================================
    # Discovery
    # ==========================================================================

    async def search(
        self,
        query: str = "",
        category: Optional[SkillCategory] = None,
        tier: Optional[SkillTier] = None,
        tags: Optional[List[str]] = None,
        author_id: Optional[str] = None,
        sort_by: str = "rating",  # rating, downloads, recent
        limit: int = 20,
        offset: int = 0,
    ) -> List[SkillListing]:
        """
        Search for skills in the marketplace.

        Args:
            query: Search query (matches name, description, tags)
            category: Filter by category
            tier: Filter by tier
            tags: Filter by tags
            author_id: Filter by author
            sort_by: Sort order (rating, downloads, recent)
            limit: Maximum results
            offset: Result offset for pagination

        Returns:
            List of matching SkillListings
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Build query
        sql = "SELECT * FROM skills WHERE is_published = 1"
        params: List[Any] = []

        if query:
            sql += " AND (name LIKE ? OR description LIKE ? OR tags LIKE ?)"
            search_term = f"%{query}%"
            params.extend([search_term, search_term, search_term])

        if category:
            sql += " AND category = ?"
            params.append(category.value)

        if tier:
            sql += " AND tier = ?"
            params.append(tier.value)

        if tags:
            for tag in tags:
                sql += " AND tags LIKE ?"
                params.append(f'%"{tag}"%')

        if author_id:
            sql += " AND author_id = ?"
            params.append(author_id)

        # Sort
        if sort_by == "rating":
            sql += " ORDER BY (CASE WHEN rating_count > 0 THEN rating_sum / rating_count ELSE 0 END) DESC"
        elif sort_by == "downloads":
            sql += " ORDER BY install_count DESC"
        else:  # recent
            sql += " ORDER BY published_at DESC"

        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        return [self._row_to_listing(row) for row in rows]

    async def get_skill(self, skill_id: str) -> Optional[SkillListing]:
        """
        Get a skill listing by ID.

        Args:
            skill_id: Skill identifier

        Returns:
            SkillListing if found, None otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM skills WHERE skill_id = ?", (skill_id,))
        row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_listing(row)

    async def get_versions(self, skill_id: str) -> List[SkillVersion]:
        """
        Get all versions of a skill.

        Args:
            skill_id: Skill identifier

        Returns:
            List of SkillVersions
        """
        listing = await self.get_skill(skill_id)
        if not listing:
            return []
        return listing.versions

    def _row_to_listing(self, row: sqlite3.Row) -> SkillListing:
        """Convert database row to SkillListing."""
        versions_data = json.loads(row["versions"] or "[]")
        versions = [
            SkillVersion(
                version=v["version"],
                changelog=v.get("changelog", ""),
                published_at=datetime.fromisoformat(v["published_at"])
                if v.get("published_at")
                else datetime.now(timezone.utc),
                downloads=v.get("downloads", 0),
                is_stable=v.get("is_stable", True),
            )
            for v in versions_data
        ]

        deps_data = json.loads(row["dependencies"] or "[]")
        dependencies = [
            SkillDependency(
                skill_id=d["skill_id"],
                version_constraint=d.get("version_constraint", "*"),
                optional=d.get("optional", False),
            )
            for d in deps_data
        ]

        capabilities = [SkillCapability(c) for c in json.loads(row["capabilities"] or "[]")]

        rating_count = row["rating_count"] or 0
        rating_sum = row["rating_sum"] or 0
        rating_average = rating_sum / rating_count if rating_count > 0 else 0.0

        return SkillListing(
            skill_id=row["skill_id"],
            name=row["name"],
            description=row["description"] or "",
            author_id=row["author_id"],
            author_name=row["author_name"],
            category=SkillCategory(row["category"]) if row["category"] else SkillCategory.CUSTOM,
            tier=SkillTier(row["tier"]) if row["tier"] else SkillTier.FREE,
            tags=json.loads(row["tags"] or "[]"),
            current_version=row["current_version"] or "1.0.0",
            versions=versions,
            dependencies=dependencies,
            capabilities=capabilities,
            required_permissions=json.loads(row["required_permissions"] or "[]"),
            install_count=row["install_count"] or 0,
            active_installs=row["active_installs"] or 0,
            rating_average=rating_average,
            rating_count=rating_count,
            is_published=bool(row["is_published"]),
            is_verified=bool(row["is_verified"]),
            is_deprecated=bool(row["is_deprecated"]),
            created_at=datetime.fromisoformat(row["created_at"])
            if row["created_at"]
            else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(row["updated_at"])
            if row["updated_at"]
            else datetime.now(timezone.utc),
            published_at=datetime.fromisoformat(row["published_at"])
            if row["published_at"]
            else None,
            homepage_url=row["homepage_url"],
            repository_url=row["repository_url"],
            documentation_url=row["documentation_url"],
            icon_url=row["icon_url"],
            screenshots=json.loads(row["screenshots"] or "[]"),
        )

    # ==========================================================================
    # Ratings
    # ==========================================================================

    async def rate(
        self,
        skill_id: str,
        user_id: str,
        rating: int,
        review: Optional[str] = None,
    ) -> SkillRating:
        """
        Rate a skill.

        Args:
            skill_id: Skill to rate
            user_id: User submitting rating
            rating: 1-5 star rating
            review: Optional review text

        Returns:
            Created or updated SkillRating
        """
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")

        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()

        # Check for existing rating
        cursor.execute(
            "SELECT rating FROM ratings WHERE skill_id = ? AND user_id = ?",
            (skill_id, user_id),
        )
        existing = cursor.fetchone()

        if existing:
            old_rating = existing["rating"]
            # Update existing rating
            cursor.execute(
                """
                UPDATE ratings SET rating = ?, review = ?, created_at = ?
                WHERE skill_id = ? AND user_id = ?
            """,
                (rating, review, now, skill_id, user_id),
            )
            # Update skill rating sum
            cursor.execute(
                """
                UPDATE skills SET rating_sum = rating_sum - ? + ?, updated_at = ?
                WHERE skill_id = ?
            """,
                (old_rating, rating, now, skill_id),
            )
        else:
            # Insert new rating
            cursor.execute(
                """
                INSERT INTO ratings (skill_id, user_id, rating, review, created_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (skill_id, user_id, rating, review, now),
            )
            # Update skill rating sum and count
            cursor.execute(
                """
                UPDATE skills SET
                    rating_sum = rating_sum + ?,
                    rating_count = rating_count + 1,
                    updated_at = ?
                WHERE skill_id = ?
            """,
                (rating, now, skill_id),
            )

        conn.commit()
        logger.info(f"User {user_id} rated skill {skill_id}: {rating} stars")

        return SkillRating(
            skill_id=skill_id,
            user_id=user_id,
            rating=rating,
            review=review,
            created_at=datetime.fromisoformat(now),
        )

    async def get_ratings(
        self,
        skill_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[SkillRating]:
        """
        Get ratings for a skill.

        Args:
            skill_id: Skill to get ratings for
            limit: Maximum results
            offset: Result offset for pagination

        Returns:
            List of SkillRatings
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM ratings WHERE skill_id = ?
            ORDER BY created_at DESC LIMIT ? OFFSET ?
        """,
            (skill_id, limit, offset),
        )
        rows = cursor.fetchall()

        return [
            SkillRating(
                skill_id=row["skill_id"],
                user_id=row["user_id"],
                rating=row["rating"],
                review=row["review"],
                created_at=datetime.fromisoformat(row["created_at"])
                if row["created_at"]
                else datetime.now(timezone.utc),
                helpful_votes=row["helpful_votes"] or 0,
            )
            for row in rows
        ]

    # ==========================================================================
    # Installation
    # ==========================================================================

    async def install(
        self,
        skill_id: str,
        tenant_id: str,
        user_id: str,
        version: Optional[str] = None,
    ) -> InstallResult:
        """
        Install a skill for a tenant.

        Args:
            skill_id: Skill to install
            tenant_id: Tenant installing the skill
            user_id: User performing installation
            version: Specific version (default: latest)

        Returns:
            InstallResult with outcome
        """
        # Get skill listing
        listing = await self.get_skill(skill_id)
        if not listing:
            return InstallResult(
                success=False,
                skill_id=skill_id,
                version="",
                error="Skill not found",
            )

        if not listing.is_published:
            return InstallResult(
                success=False,
                skill_id=skill_id,
                version="",
                error="Skill is not published",
            )

        if listing.is_deprecated:
            logger.warning(f"Installing deprecated skill: {skill_id}")

        # Check tier access (simplified - real implementation would check subscription)
        # This would integrate with RBAC and tenant subscription status

        install_version = version or listing.current_version

        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()

        # Check for existing installation
        cursor.execute(
            """
            SELECT is_active FROM installations
            WHERE skill_id = ? AND tenant_id = ?
        """,
            (skill_id, tenant_id),
        )
        existing = cursor.fetchone()

        if existing:
            if existing["is_active"]:
                # Already installed - update version
                cursor.execute(
                    """
                    UPDATE installations SET version = ?, installed_at = ?
                    WHERE skill_id = ? AND tenant_id = ?
                """,
                    (install_version, now, skill_id, tenant_id),
                )
            else:
                # Reactivate
                cursor.execute(
                    """
                    UPDATE installations SET
                        is_active = 1,
                        version = ?,
                        installed_at = ?,
                        uninstalled_at = NULL
                    WHERE skill_id = ? AND tenant_id = ?
                """,
                    (install_version, now, skill_id, tenant_id),
                )
                # Increment active installs
                cursor.execute(
                    """
                    UPDATE skills SET active_installs = active_installs + 1
                    WHERE skill_id = ?
                """,
                    (skill_id,),
                )
        else:
            # New installation
            cursor.execute(
                """
                INSERT INTO installations (
                    skill_id, tenant_id, user_id, version, installed_at, is_active
                ) VALUES (?, ?, ?, ?, ?, 1)
            """,
                (skill_id, tenant_id, user_id, install_version, now),
            )
            # Increment install counts
            cursor.execute(
                """
                UPDATE skills SET
                    install_count = install_count + 1,
                    active_installs = active_installs + 1
                WHERE skill_id = ?
            """,
                (skill_id,),
            )

        conn.commit()
        logger.info(f"Installed skill {skill_id} v{install_version} for tenant {tenant_id}")

        return InstallResult(
            success=True,
            skill_id=skill_id,
            version=install_version,
            installed_at=datetime.fromisoformat(now),
        )

    async def uninstall(
        self,
        skill_id: str,
        tenant_id: str,
    ) -> bool:
        """
        Uninstall a skill from a tenant.

        Args:
            skill_id: Skill to uninstall
            tenant_id: Tenant to uninstall from

        Returns:
            True if uninstalled successfully
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()

        cursor.execute(
            """
            UPDATE installations SET is_active = 0, uninstalled_at = ?
            WHERE skill_id = ? AND tenant_id = ? AND is_active = 1
        """,
            (now, skill_id, tenant_id),
        )

        if cursor.rowcount > 0:
            # Decrement active installs
            cursor.execute(
                """
                UPDATE skills SET active_installs = MAX(0, active_installs - 1)
                WHERE skill_id = ?
            """,
                (skill_id,),
            )
            conn.commit()
            logger.info(f"Uninstalled skill {skill_id} from tenant {tenant_id}")
            return True

        return False

    async def get_installed_skills(
        self,
        tenant_id: str,
    ) -> List[SkillListing]:
        """
        Get all skills installed for a tenant.

        Args:
            tenant_id: Tenant to get installations for

        Returns:
            List of installed SkillListings
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT s.* FROM skills s
            JOIN installations i ON s.skill_id = i.skill_id
            WHERE i.tenant_id = ? AND i.is_active = 1
        """,
            (tenant_id,),
        )
        rows = cursor.fetchall()

        return [self._row_to_listing(row) for row in rows]

    async def is_installed(self, skill_id: str, tenant_id: str) -> bool:
        """Check if a skill is installed for a tenant."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT 1 FROM installations
            WHERE skill_id = ? AND tenant_id = ? AND is_active = 1
        """,
            (skill_id, tenant_id),
        )
        return cursor.fetchone() is not None

    # ==========================================================================
    # Statistics
    # ==========================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as count FROM skills WHERE is_published = 1")
        published_count = cursor.fetchone()["count"]

        cursor.execute("SELECT SUM(install_count) as total FROM skills")
        total_installs = cursor.fetchone()["total"] or 0

        cursor.execute("SELECT COUNT(*) as count FROM ratings")
        total_ratings = cursor.fetchone()["count"]

        cursor.execute("""
            SELECT category, COUNT(*) as count
            FROM skills WHERE is_published = 1
            GROUP BY category
        """)
        categories = {row["category"]: row["count"] for row in cursor.fetchall()}

        return {
            "published_skills": published_count,
            "total_installs": total_installs,
            "total_ratings": total_ratings,
            "categories": categories,
        }


# ==========================================================================
# Global Instance
# ==========================================================================

_marketplace: Optional[SkillMarketplace] = None
_marketplace_lock = threading.Lock()


def get_marketplace() -> SkillMarketplace:
    """Get the global skill marketplace instance."""
    global _marketplace
    if _marketplace is None:
        with _marketplace_lock:
            if _marketplace is None:
                _marketplace = SkillMarketplace()
    return _marketplace
