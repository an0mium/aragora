"""
Enterprise Security Proxy for External Framework Integration.

Provides a secure, resilient proxy layer for all external framework calls with:
- Connection pooling and timeout management
- Circuit breaker per external framework
- Retry with exponential backoff
- Bulkhead isolation (max concurrent requests)
- Pre/post request hooks for auth and audit
- Request/response sanitization
- Tenant context header injection

Usage:
    from aragora.gateway.enterprise.proxy import (
        EnterpriseProxy,
        ExternalFrameworkConfig,
        ProxyConfig,
    )

    # Configure external frameworks
    proxy = EnterpriseProxy(
        config=ProxyConfig(
            default_timeout=30.0,
            max_connections=100,
        ),
        frameworks={
            "openai": ExternalFrameworkConfig(
                base_url="https://api.openai.com",
                timeout=60.0,
                max_retries=3,
            ),
            "anthropic": ExternalFrameworkConfig(
                base_url="https://api.anthropic.com",
                timeout=120.0,
                circuit_breaker_threshold=3,
            ),
        },
    )

    # Make proxied requests
    async with proxy:
        response = await proxy.request(
            framework="openai",
            method="POST",
            path="/v1/chat/completions",
            json={"model": "gpt-4", "messages": [...]},
            auth_context=auth_ctx,
            tenant_id="tenant-123",
        )

    # Register hooks for custom auth/audit
    proxy.add_pre_request_hook(verify_auth_hook)
    proxy.add_post_request_hook(audit_log_hook)
"""

from .config import (
    BulkheadSettings,
    CircuitBreakerSettings,
    ExternalFrameworkConfig,
    HealthStatus,
    ProxyConfig,
    RetrySettings,
    RetryStrategy,
    SanitizationSettings,
)
from .core import EnterpriseProxy
from .exceptions import (
    BulkheadFullError,
    CircuitOpenError,
    FrameworkNotConfiguredError,
    ProxyError,
    RequestTimeoutError,
    SanitizationError,
)
from .models import (
    ErrorHook,
    HealthCheckResult,
    PostRequestHook,
    PreRequestHook,
    ProxyRequest,
    ProxyResponse,
)
from .resilience import FrameworkBulkhead, FrameworkCircuitBreaker
from .sanitizer import RequestSanitizer

__all__ = [
    # Exceptions
    "ProxyError",
    "CircuitOpenError",
    "BulkheadFullError",
    "RequestTimeoutError",
    "FrameworkNotConfiguredError",
    "SanitizationError",
    # Enums
    "HealthStatus",
    "RetryStrategy",
    # Configuration
    "CircuitBreakerSettings",
    "RetrySettings",
    "BulkheadSettings",
    "SanitizationSettings",
    "ExternalFrameworkConfig",
    "ProxyConfig",
    # Request/Response
    "ProxyRequest",
    "ProxyResponse",
    # Hook types
    "PreRequestHook",
    "PostRequestHook",
    "ErrorHook",
    # Health
    "HealthCheckResult",
    # Components
    "FrameworkCircuitBreaker",
    "FrameworkBulkhead",
    "RequestSanitizer",
    # Main class
    "EnterpriseProxy",
]
