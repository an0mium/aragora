"""
Evidence Connectors - Ground debates in real data.

Connectors fetch evidence from external sources and integrate
with the provenance system for traceability:

- LocalDocsConnector: Search local documentation, markdown, code
- GitHubConnector: Fetch issues, PRs, discussions
- WebConnector: Search and fetch live web content
- ArXivConnector: Academic papers and preprints
- HackerNewsConnector: Tech community discussions

All connectors record evidence through ProvenanceManager
with proper source typing and confidence scoring.
"""

from aragora.connectors.local_docs import LocalDocsConnector
from aragora.connectors.github import GitHubConnector
from aragora.connectors.web import WebConnector
from aragora.connectors.arxiv import ArXivConnector, ARXIV_CATEGORIES
from aragora.connectors.hackernews import HackerNewsConnector
from aragora.connectors.base import BaseConnector, Evidence
from aragora.connectors.exceptions import (
    ConnectorError,
    ConnectorAuthError,
    ConnectorRateLimitError,
    ConnectorTimeoutError,
    ConnectorNetworkError,
    ConnectorAPIError,
    ConnectorValidationError,
    ConnectorNotFoundError,
    ConnectorQuotaError,
    ConnectorParseError,
    is_retryable_error,
    get_retry_delay,
)

__all__ = [
    # Base classes
    "BaseConnector",
    "Evidence",
    # Connectors
    "LocalDocsConnector",
    "GitHubConnector",
    "WebConnector",
    "ArXivConnector",
    "ARXIV_CATEGORIES",
    "HackerNewsConnector",
    # Exceptions
    "ConnectorError",
    "ConnectorAuthError",
    "ConnectorRateLimitError",
    "ConnectorTimeoutError",
    "ConnectorNetworkError",
    "ConnectorAPIError",
    "ConnectorValidationError",
    "ConnectorNotFoundError",
    "ConnectorQuotaError",
    "ConnectorParseError",
    # Utilities
    "is_retryable_error",
    "get_retry_delay",
]
