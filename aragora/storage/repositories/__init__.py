"""
repositories - Modular storage repositories.

This package provides repository classes for different aspects of user
and organization management, extracted from UserStore for better modularity:

- UserRepository: User identity, authentication, preferences
- OrganizationRepository: Team management, member operations
- OAuthRepository: Social login provider linking
- UsageRepository: Rate limiting and usage tracking
- InvitationRepository: Organization invitation workflow
- AuditRepository: Audit logging for compliance
- SecurityRepository: Account lockout and login security

These repositories are designed to be composed with the main UserStore,
accepting its transaction function for database access.

Example:
    from aragora.storage.repositories import UserRepository, OrganizationRepository

    # Repositories accept a transaction function for database access
    users = UserRepository(user_store._transaction)
    orgs = OrganizationRepository(user_store._transaction, users._row_to_user)
"""

from .audit import AuditRepository
from .external_identity import ExternalIdentityRepository, get_external_identity_repository
from .invitations import InvitationRepository
from .oauth import OAuthRepository
from .organizations import OrganizationRepository
from .security import SecurityRepository
from .usage import UsageRepository
from .users import UserRepository

__all__ = [
    "AuditRepository",
    "ExternalIdentityRepository",
    "get_external_identity_repository",
    "InvitationRepository",
    "OAuthRepository",
    "OrganizationRepository",
    "SecurityRepository",
    "UsageRepository",
    "UserRepository",
]
