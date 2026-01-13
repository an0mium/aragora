"""
OrganizationRepository - Organization and team management operations.

Extracted from UserStore for better modularity. Manages organization CRUD,
member management, and Stripe integration lookups.
"""

from __future__ import annotations

import json
import logging
import secrets
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Optional

if TYPE_CHECKING:
    from aragora.billing.models import Organization, SubscriptionTier, User

logger = logging.getLogger(__name__)


class OrganizationRepository:
    """
    Repository for organization and team management operations.

    This class manages:
    - Organization CRUD operations
    - Member management (add/remove users)
    - Stripe integration lookups
    - Batch operations for efficient multi-org queries
    """

    _COLUMN_MAP = {
        "name": "name",
        "slug": "slug",
        "tier": "tier",
        "owner_id": "owner_id",
        "stripe_customer_id": "stripe_customer_id",
        "stripe_subscription_id": "stripe_subscription_id",
        "debates_used_this_month": "debates_used_this_month",
        "billing_cycle_start": "billing_cycle_start",
        "settings": "settings",
    }

    def __init__(
        self,
        transaction_fn: Callable[[], ContextManager[sqlite3.Cursor]],
        row_to_user_fn: Optional[Callable[[sqlite3.Row], "User"]] = None,
    ) -> None:
        """
        Initialize the organization repository.

        Args:
            transaction_fn: Function that returns a transaction context manager.
            row_to_user_fn: Optional function to convert rows to User objects.
        """
        self._transaction = transaction_fn
        self._row_to_user = row_to_user_fn

    def create(
        self,
        name: str,
        owner_id: str,
        slug: Optional[str] = None,
        tier: Optional["SubscriptionTier"] = None,
    ) -> "Organization":
        """
        Create a new organization.

        Args:
            name: Organization name
            owner_id: User ID of owner
            slug: URL-friendly slug (auto-generated if not provided)
            tier: Subscription tier

        Returns:
            Created Organization object
        """
        from aragora.billing.models import Organization, SubscriptionTier

        if tier is None:
            tier = SubscriptionTier.FREE

        if slug is None:
            slug = name.lower().replace(" ", "-").replace("_", "-")
            base_slug = slug
            for _ in range(10):
                with self._transaction() as cursor:
                    cursor.execute("SELECT 1 FROM organizations WHERE slug = ?", (slug,))
                    if not cursor.fetchone():
                        break
                    slug = f"{base_slug}-{secrets.token_hex(4)}"

        org = Organization(
            name=name,
            slug=slug,
            tier=tier,
            owner_id=owner_id,
        )

        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO organizations (
                    id, name, slug, tier, owner_id, stripe_customer_id,
                    stripe_subscription_id, debates_used_this_month,
                    billing_cycle_start, settings, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    org.id,
                    org.name,
                    org.slug,
                    org.tier.value,
                    org.owner_id,
                    org.stripe_customer_id,
                    org.stripe_subscription_id,
                    org.debates_used_this_month,
                    org.billing_cycle_start.isoformat(),
                    json.dumps(org.settings),
                    org.created_at.isoformat(),
                    org.updated_at.isoformat(),
                ),
            )

            # Update owner's org_id and role
            cursor.execute(
                "UPDATE users SET org_id = ?, role = ?, updated_at = ? WHERE id = ?",
                (org.id, "owner", datetime.utcnow().isoformat(), owner_id),
            )

        logger.info(f"organization_created id={org.id} name={name} owner={owner_id}")
        return org

    def get_by_id(self, org_id: str) -> Optional["Organization"]:
        """Get organization by ID."""
        with self._transaction() as cursor:
            cursor.execute("SELECT * FROM organizations WHERE id = ?", (org_id,))
            row = cursor.fetchone()
            return self._row_to_org(row) if row else None

    def get_by_slug(self, slug: str) -> Optional["Organization"]:
        """Get organization by slug."""
        with self._transaction() as cursor:
            cursor.execute("SELECT * FROM organizations WHERE slug = ?", (slug,))
            row = cursor.fetchone()
            return self._row_to_org(row) if row else None

    def get_by_stripe_customer(self, stripe_customer_id: str) -> Optional["Organization"]:
        """Get organization by Stripe customer ID."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM organizations WHERE stripe_customer_id = ?",
                (stripe_customer_id,),
            )
            row = cursor.fetchone()
            return self._row_to_org(row) if row else None

    def get_by_subscription(self, subscription_id: str) -> Optional["Organization"]:
        """Get organization by Stripe subscription ID."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM organizations WHERE stripe_subscription_id = ?",
                (subscription_id,),
            )
            row = cursor.fetchone()
            return self._row_to_org(row) if row else None

    def update(self, org_id: str, **fields: Any) -> bool:
        """
        Update organization fields.

        Args:
            org_id: Organization ID
            **fields: Fields to update

        Returns:
            True if organization was updated
        """
        from aragora.billing.models import SubscriptionTier

        if not fields:
            return False

        updates: list[str] = []
        values: list[Any] = []

        for field, value in fields.items():
            if field in self._COLUMN_MAP:
                updates.append(f"{self._COLUMN_MAP[field]} = ?")
                if field == "tier" and isinstance(value, SubscriptionTier):
                    values.append(value.value)
                elif field == "settings" and isinstance(value, dict):
                    values.append(json.dumps(value))
                elif isinstance(value, datetime):
                    values.append(value.isoformat())
                else:
                    values.append(value)

        if not updates:
            return False

        updates.append("updated_at = ?")
        values.append(datetime.utcnow().isoformat())
        values.append(org_id)

        with self._transaction() as cursor:
            cursor.execute(
                f"UPDATE organizations SET {', '.join(updates)} WHERE id = ?",
                values,
            )
            return cursor.rowcount > 0

    def reset_usage(self, org_id: str) -> bool:
        """Reset monthly usage for an organization."""
        with self._transaction() as cursor:
            cursor.execute(
                """
                UPDATE organizations
                SET debates_used_this_month = 0,
                    billing_cycle_start = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (datetime.utcnow().isoformat(), datetime.utcnow().isoformat(), org_id),
            )
            return cursor.rowcount > 0

    def add_member(self, user_id: str, org_id: str, role: str = "member") -> bool:
        """Add user to organization."""
        with self._transaction() as cursor:
            cursor.execute(
                "UPDATE users SET org_id = ?, role = ?, updated_at = ? WHERE id = ?",
                (org_id, role, datetime.utcnow().isoformat(), user_id),
            )
            return cursor.rowcount > 0

    def remove_member(self, user_id: str) -> bool:
        """Remove user from organization."""
        with self._transaction() as cursor:
            cursor.execute(
                "UPDATE users SET org_id = NULL, role = 'member', updated_at = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), user_id),
            )
            return cursor.rowcount > 0

    def get_members(self, org_id: str) -> list["User"]:
        """Get all members of an organization."""
        if self._row_to_user is None:
            raise RuntimeError("row_to_user_fn not provided to OrganizationRepository")

        with self._transaction() as cursor:
            cursor.execute("SELECT * FROM users WHERE org_id = ?", (org_id,))
            return [self._row_to_user(row) for row in cursor.fetchall()]

    def get_with_members(self, org_id: str) -> tuple[Optional["Organization"], list["User"]]:
        """
        Get organization and all its members in a single query operation.

        Args:
            org_id: Organization ID

        Returns:
            Tuple of (Organization or None, list of User members)
        """
        if self._row_to_user is None:
            raise RuntimeError("row_to_user_fn not provided to OrganizationRepository")

        with self._transaction() as cursor:
            cursor.execute("SELECT * FROM organizations WHERE id = ?", (org_id,))
            org_row = cursor.fetchone()
            if not org_row:
                return None, []

            org = self._row_to_org(org_row)

            cursor.execute("SELECT * FROM users WHERE org_id = ?", (org_id,))
            members = [self._row_to_user(row) for row in cursor.fetchall()]

            return org, members

    def get_batch_with_members(
        self,
        org_ids: list[str],
    ) -> dict[str, tuple["Organization", list["User"]]]:
        """
        Get multiple organizations with their members in optimized queries.

        Args:
            org_ids: List of organization IDs

        Returns:
            Dict mapping org_id to (Organization, members list) tuple
        """
        if not org_ids:
            return {}

        if self._row_to_user is None:
            raise RuntimeError("row_to_user_fn not provided to OrganizationRepository")

        unique_ids = list(dict.fromkeys(org_ids))
        result: dict[str, tuple["Organization", list["User"]]] = {}

        with self._transaction() as cursor:
            placeholders = ",".join("?" * len(unique_ids))
            cursor.execute(
                f"SELECT * FROM organizations WHERE id IN ({placeholders})",
                unique_ids,
            )
            orgs = {row["id"]: self._row_to_org(row) for row in cursor.fetchall()}

            cursor.execute(
                f"SELECT * FROM users WHERE org_id IN ({placeholders})",
                unique_ids,
            )

            members_by_org: dict[str, list["User"]] = {oid: [] for oid in orgs}
            for row in cursor.fetchall():
                user = self._row_to_user(row)
                if user.org_id in members_by_org:
                    members_by_org[user.org_id].append(user)

            for org_id, org in orgs.items():
                result[org_id] = (org, members_by_org.get(org_id, []))

        return result

    @staticmethod
    def _row_to_org(row: sqlite3.Row) -> "Organization":
        """Convert database row to Organization object."""
        from aragora.billing.models import Organization, SubscriptionTier

        return Organization(
            id=row["id"],
            name=row["name"],
            slug=row["slug"],
            tier=SubscriptionTier(row["tier"]),
            owner_id=row["owner_id"],
            stripe_customer_id=row["stripe_customer_id"],
            stripe_subscription_id=row["stripe_subscription_id"],
            debates_used_this_month=row["debates_used_this_month"],
            billing_cycle_start=datetime.fromisoformat(row["billing_cycle_start"]),
            settings=json.loads(row["settings"]) if row["settings"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
