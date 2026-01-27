"""Versioning compatibility module for handlers.

Re-exports versioning utilities from the main server versioning module.
"""

from aragora.server.versioning.compat import strip_version_prefix

__all__ = ["strip_version_prefix"]
