"""
OAuthRepository - OAuth provider linking and lookup operations.

Extracted from UserStore for better modularity. Manages the lifecycle
of OAuth provider connections for social login functionality.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Callable, ContextManager, Optional

logger = logging.getLogger(__name__)


class OAuthRepository:
    """
    Repository for OAuth provider operations.

    This class manages:
    - Linking OAuth providers to user accounts
    - Unlinking OAuth providers
    - Looking up users by OAuth provider ID
    - Listing linked providers for a user
    """

    def __init__(
        self,
        transaction_fn: Callable[[], ContextManager[sqlite3.Cursor]],
    ) -> None:
        """
        Initialize the OAuth repository.

        Args:
            transaction_fn: Function that returns a transaction context manager
                           with a cursor.
        """
        self._transaction = transaction_fn

    def link_provider(
        self,
        user_id: str,
        provider: str,
        provider_user_id: str,
        email: Optional[str] = None,
    ) -> bool:
        """
        Link an OAuth provider to a user account.

        Args:
            user_id: User ID to link to
            provider: OAuth provider name (e.g., 'google', 'github')
            provider_user_id: User ID from the OAuth provider
            email: Email from OAuth provider (optional)

        Returns:
            True if linked successfully
        """
        try:
            with self._transaction() as cursor:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO oauth_providers
                    (user_id, provider, provider_user_id, email, linked_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        provider.lower(),
                        provider_user_id,
                        email,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            logger.info(f"OAuth linked: user={user_id} provider={provider}")
            return True
        except Exception as e:
            logger.error(f"Failed to link OAuth: {e}")
            return False

    def unlink_provider(self, user_id: str, provider: str) -> bool:
        """
        Unlink an OAuth provider from a user account.

        Args:
            user_id: User ID to unlink from
            provider: OAuth provider name

        Returns:
            True if unlinked successfully
        """
        with self._transaction() as cursor:
            cursor.execute(
                "DELETE FROM oauth_providers WHERE user_id = ? AND provider = ?",
                (user_id, provider.lower()),
            )
            if cursor.rowcount > 0:
                logger.info(f"OAuth unlinked: user={user_id} provider={provider}")
                return True
        return False

    def get_user_id_by_provider(
        self,
        provider: str,
        provider_user_id: str,
    ) -> Optional[str]:
        """
        Get user ID by OAuth provider credentials.

        Args:
            provider: OAuth provider name
            provider_user_id: User ID from the OAuth provider

        Returns:
            User ID if found, None otherwise
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                SELECT user_id FROM oauth_providers
                WHERE provider = ? AND provider_user_id = ?
                """,
                (provider.lower(), provider_user_id),
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def get_providers_for_user(self, user_id: str) -> list[dict]:
        """
        Get all OAuth providers linked to a user.

        Args:
            user_id: User ID

        Returns:
            List of linked providers with their details
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                SELECT provider, provider_user_id, email, linked_at
                FROM oauth_providers
                WHERE user_id = ?
                """,
                (user_id,),
            )
            return [
                {
                    "provider": row[0],
                    "provider_user_id": row[1],
                    "email": row[2],
                    "linked_at": row[3],
                }
                for row in cursor.fetchall()
            ]

    def has_provider(self, user_id: str, provider: str) -> bool:
        """
        Check if a user has a specific OAuth provider linked.

        Args:
            user_id: User ID
            provider: OAuth provider name

        Returns:
            True if the provider is linked
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                SELECT 1 FROM oauth_providers
                WHERE user_id = ? AND provider = ?
                LIMIT 1
                """,
                (user_id, provider.lower()),
            )
            return cursor.fetchone() is not None
