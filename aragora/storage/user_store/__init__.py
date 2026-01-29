"""
UserStore - SQLite and PostgreSQL backends for user and organization persistence.

Provides CRUD operations for:
- Users (registration, authentication, API keys)
- Organizations (team management, billing)
- Usage tracking (debate counts, monthly resets)

This class serves as a facade over specialized repositories:
- UserRepository: User CRUD and authentication
- OrganizationRepository: Team management and billing
- OAuthRepository: Social login provider linking
- UsageRepository: Rate limiting and usage tracking
- AuditRepository: Audit logging for compliance
- InvitationRepository: Organization invitation workflow
- SecurityRepository: Account lockout and login security

The repositories are created internally and methods delegate to them,
maintaining backward compatibility while enabling modular testing.
"""

from .sqlite_store import UserStore
from .postgres_store import PostgresUserStore
from .singleton import get_user_store, set_user_store, reset_user_store

__all__ = [
    "UserStore",
    "PostgresUserStore",
    "get_user_store",
    "set_user_store",
    "reset_user_store",
]
