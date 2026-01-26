"""
OAuth providers package.

Contains provider-specific OAuth implementations:
- google.py: Google OAuth 2.0
- github.py: GitHub OAuth
- microsoft.py: Microsoft OAuth (Azure AD)
- apple.py: Apple Sign-In
- oidc.py: Generic OIDC

Each provider module exports functions for initiating auth and handling callbacks.
"""

# Provider implementations will be migrated here incrementally from _oauth_impl.py
