"""
OrganizationStore - SQLite backend for organization and invitation persistence.

Extracted from UserStore to improve modularity.
Provides CRUD operations for:
- Organizations (team management, settings)
- Invitations (member invites, token validation)
"""

from __future__ import annotations

import json
import logging
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Optional

from aragora.billing.models import Organization, OrganizationInvitation, SubscriptionTier, User
from aragora.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class OrganizationStore:
    """
    SQLite-backed storage for organizations and invitations.

    Can be used standalone or composed with UserStore.
    Thread-safe with connection pooling via thread-local storage.
    """

    def __init__(
        self,
        db_path: Path | str,
        get_connection: Optional[Callable[[], sqlite3.Connection]] = None,
        update_user: Optional[Callable[..., bool]] = None,
        row_to_user: Optional[Callable[[sqlite3.Row], User]] = None,
    ):
        """
        Initialize OrganizationStore.

        Args:
            db_path: Path to SQLite database file
            get_connection: Optional connection factory (for sharing with UserStore)
            update_user: Optional user update function (for org membership)
            row_to_user: Optional row-to-user converter (for get_org_members)
        """
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._external_get_connection = get_connection
        self._external_update_user = update_user
        self._external_row_to_user = row_to_user

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if self._external_get_connection:
            return self._external_get_connection()

        if not hasattr(self._local, "connection"):
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.connection = conn
        return self._local.connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for database transactions."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.debug("Transaction rolled back due to: %s", e)
            raise

    # =========================================================================
    # Organization Methods
    # =========================================================================

    def create_organization(
        self,
        name: str,
        owner_id: str,
        slug: Optional[str] = None,
        tier: SubscriptionTier = SubscriptionTier.FREE,
    ) -> Organization:
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
        if slug is None:
            slug = name.lower().replace(" ", "-").replace("_", "-")
            # Make unique by appending random suffix if needed
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

        # Update owner's org_id if we have a user updater
        if self._external_update_user:
            self._external_update_user(owner_id, org_id=org.id, role="owner")

        logger.info(f"organization_created id={org.id} name={name} owner={owner_id}")
        return org

    def get_organization_by_id(self, org_id: str) -> Optional[Organization]:
        """Get organization by ID."""
        with self._transaction() as cursor:
            cursor.execute("SELECT * FROM organizations WHERE id = ?", (org_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_org(row)
        return None

    def get_organization_by_slug(self, slug: str) -> Optional[Organization]:
        """Get organization by slug."""
        with self._transaction() as cursor:
            cursor.execute("SELECT * FROM organizations WHERE slug = ?", (slug,))
            row = cursor.fetchone()
            if row:
                return self._row_to_org(row)
        return None

    def get_organization_by_stripe_customer(
        self, stripe_customer_id: str
    ) -> Optional[Organization]:
        """Get organization by Stripe customer ID."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM organizations WHERE stripe_customer_id = ?",
                (stripe_customer_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_org(row)
        return None

    def get_organization_by_subscription(self, subscription_id: str) -> Optional[Organization]:
        """Get organization by Stripe subscription ID."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM organizations WHERE stripe_subscription_id = ?",
                (subscription_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_org(row)
        return None

    def reset_org_usage(self, org_id: str) -> bool:
        """Reset monthly usage for a single organization."""
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

    def update_organization(self, org_id: str, **fields) -> bool:
        """Update organization fields."""
        if not fields:
            return False

        column_map = {
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

        updates = []
        values = []
        for field, value in fields.items():
            if field in column_map:
                updates.append(f"{column_map[field]} = ?")
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
            query = f"UPDATE organizations SET {', '.join(updates)} WHERE id = ?"  # nosec B608
            cursor.execute(query, values)
            return cursor.rowcount > 0

    def add_user_to_org(self, user_id: str, org_id: str, role: str = "member") -> bool:
        """Add user to organization."""
        if self._external_update_user:
            return self._external_update_user(user_id, org_id=org_id, role=role)
        raise ConfigurationError(
            component="OrganizationStore",
            reason="update_user callback required for add_user_to_org",
        )

    def remove_user_from_org(self, user_id: str) -> bool:
        """Remove user from organization."""
        if self._external_update_user:
            return self._external_update_user(user_id, org_id=None, role="member")
        raise ConfigurationError(
            component="OrganizationStore",
            reason="update_user callback required for remove_user_from_org",
        )

    def get_org_members(self, org_id: str) -> list[User]:
        """Get all members of an organization."""
        if not self._external_row_to_user:
            raise ConfigurationError(
                component="OrganizationStore",
                reason="row_to_user callback required for get_org_members",
            )

        with self._transaction() as cursor:
            cursor.execute("SELECT * FROM users WHERE org_id = ?", (org_id,))
            return [self._external_row_to_user(row) for row in cursor.fetchall()]

    def _row_to_org(self, row: sqlite3.Row) -> Organization:
        """Convert database row to Organization object."""
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

    # =========================================================================
    # Invitation Methods
    # =========================================================================

    def create_invitation(self, invitation: OrganizationInvitation) -> bool:
        """Create a new organization invitation."""
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO org_invitations (
                    id, org_id, email, role, token, invited_by,
                    status, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    invitation.id,
                    invitation.org_id,
                    invitation.email,
                    invitation.role,
                    invitation.token,
                    invitation.invited_by,
                    invitation.status,
                    invitation.created_at.isoformat(),
                    invitation.expires_at.isoformat() if invitation.expires_at else None,
                ),
            )
            return cursor.rowcount > 0

    def get_invitation_by_id(self, invitation_id: str) -> Optional[OrganizationInvitation]:
        """Get invitation by ID."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM org_invitations WHERE id = ?",
                (invitation_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_invitation(row)
        return None

    def get_invitation_by_token(self, token: str) -> Optional[OrganizationInvitation]:
        """Get invitation by token."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM org_invitations WHERE token = ?",
                (token,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_invitation(row)
        return None

    def get_invitation_by_email(
        self, email: str, org_id: str, status: str = "pending"
    ) -> Optional[OrganizationInvitation]:
        """Get invitation by email and org."""
        with self._transaction() as cursor:
            cursor.execute(
                """
                SELECT * FROM org_invitations
                WHERE email = ? AND org_id = ? AND status = ?
                """,
                (email, org_id, status),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_invitation(row)
        return None

    def get_invitations_for_org(self, org_id: str) -> list[OrganizationInvitation]:
        """Get all invitations for an organization."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM org_invitations WHERE org_id = ? ORDER BY created_at DESC",
                (org_id,),
            )
            return [self._row_to_invitation(row) for row in cursor.fetchall()]

    def get_pending_invitations_by_email(self, email: str) -> list[OrganizationInvitation]:
        """Get all pending invitations for an email address."""
        with self._transaction() as cursor:
            cursor.execute(
                """
                SELECT * FROM org_invitations
                WHERE email = ? AND status = 'pending'
                ORDER BY created_at DESC
                """,
                (email,),
            )
            return [self._row_to_invitation(row) for row in cursor.fetchall()]

    def update_invitation_status(
        self,
        invitation_id: str,
        status: str,
        accepted_by: Optional[str] = None,
        accepted_at: Optional[datetime] = None,
    ) -> bool:
        """Update invitation status."""
        with self._transaction() as cursor:
            if status == "accepted" and accepted_by:
                cursor.execute(
                    """
                    UPDATE org_invitations
                    SET status = ?, accepted_by = ?, accepted_at = ?
                    WHERE id = ?
                    """,
                    (
                        status,
                        accepted_by,
                        accepted_at.isoformat() if accepted_at else datetime.utcnow().isoformat(),
                        invitation_id,
                    ),
                )
            else:
                cursor.execute(
                    "UPDATE org_invitations SET status = ? WHERE id = ?",
                    (status, invitation_id),
                )
            return cursor.rowcount > 0

    def delete_invitation(self, invitation_id: str) -> bool:
        """Delete an invitation."""
        with self._transaction() as cursor:
            cursor.execute(
                "DELETE FROM org_invitations WHERE id = ?",
                (invitation_id,),
            )
            return cursor.rowcount > 0

    def cleanup_expired_invitations(self) -> int:
        """Delete expired invitations."""
        with self._transaction() as cursor:
            cursor.execute(
                """
                DELETE FROM org_invitations
                WHERE status = 'pending'
                AND expires_at IS NOT NULL
                AND expires_at < ?
                """,
                (datetime.utcnow().isoformat(),),
            )
            count = cursor.rowcount
            if count > 0:
                logger.info(f"Cleaned up {count} expired invitations")
            return count

    def _row_to_invitation(self, row: sqlite3.Row) -> OrganizationInvitation:
        """Convert database row to OrganizationInvitation object."""
        return OrganizationInvitation(
            id=row["id"],
            org_id=row["org_id"],
            email=row["email"],
            role=row["role"],
            token=row["token"],
            invited_by=row["invited_by"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            accepted_by=row["accepted_by"] if "accepted_by" in row.keys() else None,
            accepted_at=(
                datetime.fromisoformat(row["accepted_at"])
                if ("accepted_at" in row.keys() and row["accepted_at"])
                else None
            ),
        )

    def close(self) -> None:
        """Close database connection if we own it."""
        if self._external_get_connection is None and hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection
