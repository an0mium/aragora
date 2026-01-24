"""
Performance Optimization Layer.

Provides centralized performance optimization components:
- DataLoader: Batch query resolution with deduplication
- LazyLoader: N+1 query prevention with lazy loading decorators
- AdaptiveCache: Dynamic TTL management based on access patterns
- Compression: Response compression middleware
- QueryAnalyzer: Query plan analysis and optimization hints
"""

from aragora.performance.data_loader import (
    DataLoader,
    BatchResolver,
    create_data_loader,
)
from aragora.performance.lazy_loading import (
    lazy_property,
    LazyLoader,
    prefetch,
)
from aragora.performance.adaptive_cache import (
    AdaptiveTTLCache,
    AccessPattern,
    CacheOptimizer,
)
from aragora.performance.compression import (
    CompressionMiddleware,
    compress_response,
    should_compress,
)

__all__ = [
    # DataLoader
    "DataLoader",
    "BatchResolver",
    "create_data_loader",
    # Lazy Loading
    "lazy_property",
    "LazyLoader",
    "prefetch",
    # Adaptive Cache
    "AdaptiveTTLCache",
    "AccessPattern",
    "CacheOptimizer",
    # Compression
    "CompressionMiddleware",
    "compress_response",
    "should_compress",
]
