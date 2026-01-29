"""
OAuth handler package.

Provides a modular OAuth authentication handler split into provider-specific
mixins for maintainability:

- base.py: OAuthHandler class (combines all mixins)
- google.py: GoogleOAuthMixin
- github.py: GitHubOAuthMixin
- microsoft.py: MicrosoftOAuthMixin
- apple.py: AppleOAuthMixin
- oidc.py: OIDCOAuthMixin
- account.py: AccountManagementMixin
- utils.py: Shared utility functions
"""

from .base import OAuthHandler

__all__ = ["OAuthHandler"]
