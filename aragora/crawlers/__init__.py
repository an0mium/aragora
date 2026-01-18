"""
Crawlers for Enterprise Multi-Agent Control Plane.

.. deprecated:: 2.0
    The crawlers module is deprecated. Use the connectors module instead:

    - For repository crawling: use ``aragora.connectors.repository_crawler``
    - For local docs: use ``aragora.connectors.local_docs``
    - For web content: use ``aragora.connectors.web``

    The connectors module provides:
    - Better AST-based symbol extraction
    - Concurrent processing
    - Provenance tracking
    - Reliability scoring

Legacy usage (deprecated):
    from aragora.crawlers import RepositoryCrawler

New usage (recommended):
    from aragora.connectors.repository_crawler import RepositoryCrawler
"""

import warnings

from aragora.crawlers.base import (
    BaseCrawler,
    ContentType,
    CrawlerConfig,
    CrawlResult,
    CrawlStats,
    CrawlStatus,
    IndexResult,
)
from aragora.crawlers.repository import (
    RepositoryCrawler as _DeprecatedRepositoryCrawler,
    RepositoryCrawlerConfig,
    RepositoryInfo,
)

# Emit deprecation warning on module import
warnings.warn(
    "The aragora.crawlers module is deprecated. "
    "Use aragora.connectors.repository_crawler for repository crawling instead. "
    "See module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)


class RepositoryCrawler(_DeprecatedRepositoryCrawler):
    """
    Repository crawler for git repositories.

    .. deprecated:: 2.0
        Use ``aragora.connectors.repository_crawler.RepositoryCrawler`` instead.
        The connectors version provides better AST parsing, concurrent processing,
        and provenance tracking.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "aragora.crawlers.RepositoryCrawler is deprecated. "
            "Use aragora.connectors.repository_crawler.RepositoryCrawler instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

__all__ = [
    # Base
    "BaseCrawler",
    "ContentType",
    "CrawlerConfig",
    "CrawlResult",
    "CrawlStats",
    "CrawlStatus",
    "IndexResult",
    # Repository
    "RepositoryCrawler",
    "RepositoryCrawlerConfig",
    "RepositoryInfo",
]
