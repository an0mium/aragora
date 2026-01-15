"""In-memory user store for development/testing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aragora.billing.models import Organization, User


class InMemoryUserStore:
    """
    Simple in-memory user store for development/testing.

    Production should use a proper database backend.
    """

    def __init__(self) -> None:
        self.users: dict[str, User] = {}  # id -> User
        self.users_by_email: dict[str, str] = {}  # email -> id
        self.organizations: dict[str, Organization] = {}  # id -> Organization
        self.api_keys: dict[str, str] = {}  # api_key -> user_id

    def save_user(self, user: User) -> None:
        """Save a user."""
        self.users[user.id] = user
        self.users_by_email[user.email.lower()] = user.id
        # Legacy: store plaintext API key if present
        if user.api_key:
            self.api_keys[user.api_key] = user.id

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        user_id = self.users_by_email.get(email.lower())
        if user_id:
            return self.users.get(user_id)
        return None

    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key.

        Supports both:
        - Legacy: plaintext API key lookup in cache
        - Secure: hash-based verification against all users with API keys
        """
        # First try legacy plaintext lookup
        user_id = self.api_keys.get(api_key)
        if user_id:
            return self.users.get(user_id)

        # Fall back to hash-based verification for secure storage
        for user in self.users.values():
            if hasattr(user, "verify_api_key") and user.verify_api_key(api_key):
                return user

        return None

    def save_organization(self, org: Organization) -> None:
        """Save an organization."""
        self.organizations[org.id] = org

    def get_organization_by_id(self, org_id: str) -> Optional[Organization]:
        """Get organization by ID."""
        return self.organizations.get(org_id)


__all__ = ["InMemoryUserStore"]
