"""
repositories - Modular storage repositories.

This package provides repository classes for different aspects of user
and organization management, extracted from UserStore for better modularity:

- InvitationRepository: Organization invitation workflow
- AuditRepository: Audit logging for compliance
- SecurityRepository: Account lockout and login security

These repositories are designed to be composed with the main UserStore,
accepting its transaction function for database access.

Example:
    from aragora.storage.repositories import InvitationRepository, AuditRepository

    # Repositories accept a transaction function for database access
    invitations = InvitationRepository(user_store._transaction)
    audit = AuditRepository(user_store._transaction)
"""

from .audit import AuditRepository
from .invitations import InvitationRepository
from .security import SecurityRepository

__all__ = [
    "AuditRepository",
    "InvitationRepository",
    "SecurityRepository",
]
