"""
Git Repository Connectors.

Provides enterprise-grade Git repository integration:
- GitHub (with webhook support)
- GitLab (planned)
- Bitbucket (planned)

Features:
- Incremental sync using commit SHA
- AST parsing for code intelligence
- Dependency graph building
- PR/Issue integration
"""

from aragora.connectors.enterprise.git.github import GitHubEnterpriseConnector

__all__ = [
    "GitHubEnterpriseConnector",
]
