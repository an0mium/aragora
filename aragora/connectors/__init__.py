"""
Evidence Connectors - Ground debates in real data.

Connectors fetch evidence from external sources and integrate
with the provenance system for traceability:

- LocalDocsConnector: Search local documentation, markdown, code
- GitHubConnector: Fetch issues, PRs, discussions
- WebConnector: Search and fetch live web content
- ArXivConnector: Academic papers and preprints
- HackerNewsConnector: Tech community discussions
- WikipediaConnector: Encyclopedia articles and reference knowledge
- RedditConnector: Community discussions and sentiment
- TwitterConnector: Public discourse and real-time updates
- SQLConnector: Query SQL databases (PostgreSQL, MySQL, SQLite)
- NewsAPIConnector: News articles from multiple sources
- SECConnector: SEC EDGAR financial filings

All connectors record evidence through ProvenanceManager
with proper source typing and confidence scoring.
"""

from aragora.connectors.arxiv import ARXIV_CATEGORIES, ArXivConnector
from aragora.connectors.base import BaseConnector, Evidence
from aragora.connectors.exceptions import (
    ConnectorAPIError,
    ConnectorAuthError,
    ConnectorError,
    ConnectorNetworkError,
    ConnectorNotFoundError,
    ConnectorParseError,
    ConnectorQuotaError,
    ConnectorRateLimitError,
    ConnectorTimeoutError,
    ConnectorValidationError,
    get_retry_delay,
    is_retryable_error,
)
from aragora.connectors.github import GitHubConnector
from aragora.connectors.hackernews import HackerNewsConnector
from aragora.connectors.local_docs import LocalDocsConnector
from aragora.connectors.newsapi import (
    HIGH_CREDIBILITY_SOURCES,
    MEDIUM_CREDIBILITY_SOURCES,
    NewsAPIConnector,
)
from aragora.connectors.reddit import RedditConnector
from aragora.connectors.sec import FORM_TYPES, SECConnector
from aragora.connectors.sql import SQLConnector, SQLQueryResult
from aragora.connectors.twitter import TwitterConnector
from aragora.connectors.web import WebConnector
from aragora.connectors.wikipedia import WikipediaConnector

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
    "WikipediaConnector",
    "RedditConnector",
    "TwitterConnector",
    "SQLConnector",
    "SQLQueryResult",
    "NewsAPIConnector",
    "HIGH_CREDIBILITY_SOURCES",
    "MEDIUM_CREDIBILITY_SOURCES",
    "SECConnector",
    "FORM_TYPES",
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
