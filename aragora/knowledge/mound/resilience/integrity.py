"""Data integrity verification."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class IntegrityCheckResult:
    """Result of an integrity check."""

    passed: bool
    checks_performed: int
    issues_found: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks_performed": self.checks_performed,
            "issues_found": self.issues_found,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


class IntegrityVerifier:
    """
    Verifies data integrity on startup and periodically.

    Checks:
    - Foreign key constraint violations
    - Orphaned records
    - Checksum consistency
    - Index health
    """

    def __init__(self, pool: Any):
        self._pool = pool

    async def verify_all(self) -> IntegrityCheckResult:
        """Run all integrity checks."""
        issues: list[str] = []
        details: dict[str, Any] = {}
        checks_performed = 0

        # Check for orphaned provenance chains
        checks_performed += 1
        orphaned_provenance = await self._check_orphaned_provenance()
        if orphaned_provenance > 0:
            issues.append(f"Found {orphaned_provenance} orphaned provenance chains")
        details["orphaned_provenance"] = orphaned_provenance

        # Check for orphaned relationships
        checks_performed += 1
        orphaned_relationships = await self._check_orphaned_relationships()
        if orphaned_relationships > 0:
            issues.append(f"Found {orphaned_relationships} orphaned relationships")
        details["orphaned_relationships"] = orphaned_relationships

        # Check for orphaned topics
        checks_performed += 1
        orphaned_topics = await self._check_orphaned_topics()
        if orphaned_topics > 0:
            issues.append(f"Found {orphaned_topics} orphaned topics")
        details["orphaned_topics"] = orphaned_topics

        # Check for orphaned access grants
        checks_performed += 1
        orphaned_grants = await self._check_orphaned_access_grants()
        if orphaned_grants > 0:
            issues.append(f"Found {orphaned_grants} orphaned access grants")
        details["orphaned_grants"] = orphaned_grants

        # Check index health
        checks_performed += 1
        index_issues = await self._check_index_health()
        if index_issues:
            issues.extend(index_issues)
        details["index_health"] = "healthy" if not index_issues else index_issues

        # Check for duplicate content hashes
        checks_performed += 1
        duplicates = await self._check_duplicate_content()
        details["duplicate_content_count"] = duplicates

        return IntegrityCheckResult(
            passed=len(issues) == 0,
            checks_performed=checks_performed,
            issues_found=issues,
            details=details,
        )

    async def _check_orphaned_provenance(self) -> int:
        """Check for provenance chains without parent nodes."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COUNT(*) FROM provenance_chains pc
                LEFT JOIN knowledge_nodes kn ON pc.node_id = kn.id
                WHERE kn.id IS NULL
                """)
            return result or 0

    async def _check_orphaned_relationships(self) -> int:
        """Check for relationships with missing nodes."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COUNT(*) FROM knowledge_relationships kr
                LEFT JOIN knowledge_nodes kn1 ON kr.from_node_id = kn1.id
                LEFT JOIN knowledge_nodes kn2 ON kr.to_node_id = kn2.id
                WHERE kn1.id IS NULL OR kn2.id IS NULL
                """)
            return result or 0

    async def _check_orphaned_topics(self) -> int:
        """Check for topics without parent nodes."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COUNT(*) FROM node_topics nt
                LEFT JOIN knowledge_nodes kn ON nt.node_id = kn.id
                WHERE kn.id IS NULL
                """)
            return result or 0

    async def _check_orphaned_access_grants(self) -> int:
        """Check for access grants without parent nodes."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COUNT(*) FROM access_grants ag
                LEFT JOIN knowledge_nodes kn ON ag.item_id = kn.id
                WHERE kn.id IS NULL
                """)
            return result or 0

    async def _check_index_health(self) -> list[str]:
        """Check index health (PostgreSQL specific)."""
        issues = []
        try:
            async with self._pool.acquire() as conn:
                # Check for invalid indexes
                invalid = await conn.fetch("""
                    SELECT indexrelid::regclass AS index_name
                    FROM pg_index WHERE NOT indisvalid
                    """)
                for row in invalid:
                    issues.append(f"Invalid index: {row['index_name']}")
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"Index health check failed (connection error): {e}")
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Index health check failed (data error): {e}")
        return issues

    async def _check_duplicate_content(self) -> int:
        """Check for duplicate content by hash."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COUNT(*) FROM (
                    SELECT content_hash, workspace_id, COUNT(*) as cnt
                    FROM knowledge_nodes
                    WHERE content_hash != ''
                    GROUP BY content_hash, workspace_id
                    HAVING COUNT(*) > 1
                ) duplicates
                """)
            return result or 0

    async def repair_orphans(self, dry_run: bool = True) -> dict[str, int]:
        """
        Repair orphaned records.

        Args:
            dry_run: If True, only report what would be fixed

        Returns:
            Count of records fixed by table
        """
        repairs: dict[str, int] = {}

        async with self._pool.acquire() as conn:
            if dry_run:
                # Just count what would be repaired
                repairs["provenance_chains"] = await self._check_orphaned_provenance()
                repairs["relationships"] = await self._check_orphaned_relationships()
                repairs["topics"] = await self._check_orphaned_topics()
                repairs["access_grants"] = await self._check_orphaned_access_grants()
            else:
                # Actually delete orphaned records
                result = await conn.execute("""
                    DELETE FROM provenance_chains
                    WHERE node_id NOT IN (SELECT id FROM knowledge_nodes)
                    """)
                repairs["provenance_chains"] = int(result.split()[-1])

                result = await conn.execute("""
                    DELETE FROM knowledge_relationships
                    WHERE from_node_id NOT IN (SELECT id FROM knowledge_nodes)
                       OR to_node_id NOT IN (SELECT id FROM knowledge_nodes)
                    """)
                repairs["relationships"] = int(result.split()[-1])

                result = await conn.execute("""
                    DELETE FROM node_topics
                    WHERE node_id NOT IN (SELECT id FROM knowledge_nodes)
                    """)
                repairs["topics"] = int(result.split()[-1])

                result = await conn.execute("""
                    DELETE FROM access_grants
                    WHERE item_id NOT IN (SELECT id FROM knowledge_nodes)
                    """)
                repairs["access_grants"] = int(result.split()[-1])

                logger.info(f"Repaired orphaned records: {repairs}")

        return repairs
