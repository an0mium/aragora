"""
Enterprise Gateway Components.

Provides enterprise-grade gateway features for compliance and security:
- AuditInterceptor: Request/response audit logging with cryptographic trails
- AuthBridge: Authentication pass-through between Aragora and external frameworks
- EnterpriseProxy: Secure, resilient proxy for external framework calls
- TenantRouter: Multi-tenant routing with isolation and quota enforcement
- PII redaction and GDPR compliance
- SOC 2 audit evidence generation
- SIEM integration via webhooks

Usage:
    from aragora.gateway.enterprise import AuditInterceptor, AuditConfig

    interceptor = AuditInterceptor(config=AuditConfig(
        retention_days=365,
        emit_events=True,
    ))

    # Intercept a request/response
    record = await interceptor.intercept(
        request=request_data,
        response=response_data,
        correlation_id="req-123",
    )

AuthBridge Usage:
    from aragora.gateway.enterprise import (
        AuthBridge,
        AuthContext,
        PermissionMapping,
    )

    # Initialize bridge with permission mappings
    bridge = AuthBridge(
        permission_mappings=[
            PermissionMapping(
                aragora_permission="debates.create",
                external_action="create_conversation",
            ),
        ]
    )

    # Verify incoming request
    context = await bridge.verify_request(token="...")

    # Check if action is allowed
    if bridge.is_action_allowed(context, "create_conversation"):
        # Proceed with external framework action
        ...

    # Exchange token for external framework
    exchange_result = await bridge.exchange_token(context, target_audience="external-api")

TenantRouter Usage:
    from aragora.gateway.enterprise import (
        TenantRouter,
        TenantRoutingConfig,
        EndpointConfig,
        TenantQuotas,
    )

    # Configure tenant routing
    router = TenantRouter(
        configs=[
            TenantRoutingConfig(
                tenant_id="acme-corp",
                endpoints=[
                    EndpointConfig(url="https://acme.api.example.com"),
                ],
                quotas=TenantQuotas(
                    requests_per_minute=100,
                    requests_per_day=10000,
                ),
            ),
        ]
    )

    # Route a request
    decision = await router.route(tenant_id="acme-corp", request=request_data)
    print(decision.target_endpoint)  # "https://acme.api.example.com"

    # Check quota status
    status = await router.get_quota_status("acme-corp")

EnterpriseProxy Usage:
    from aragora.gateway.enterprise import (
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
            ),
            "anthropic": ExternalFrameworkConfig(
                base_url="https://api.anthropic.com",
                timeout=120.0,
            ),
        },
    )

    # Make proxied requests with resilience (circuit breakers, retries, bulkhead)
    async with proxy:
        response = await proxy.request(
            framework="openai",
            method="POST",
            path="/v1/chat/completions",
            json={"model": "gpt-4", "messages": [...]},
            tenant_id="tenant-123",
        )

    # Register hooks for custom auth/audit
    proxy.add_pre_request_hook(verify_auth_hook)
    proxy.add_post_request_hook(audit_log_hook)
"""

from aragora.gateway.enterprise.audit_interceptor import (
    AuditInterceptor,
    AuditRecord,
    AuditConfig,
    PIIRedactionRule,
    RedactionType,
    AuditEventType,
    AuditStorage,
    InMemoryAuditStorage,
    PostgresAuditStorage,
)

from aragora.gateway.enterprise.auth_bridge import (
    AuthBridge,
    AuthBridgeError,
    AuthenticationError,
    PermissionDeniedError,
    SessionExpiredError,
    TokenType,
    AuthContext,
    PermissionMapping,
    TokenExchangeResult,
    BridgedSession,
    AuditEntry,
    SessionLifecycleHook,
)

from aragora.gateway.enterprise.tenant_router import (
    TenantRouter,
    TenantRoutingConfig,
    TenantRoutingContext,
    EndpointConfig,
    EndpointHealth,
    TenantQuotas,
    QuotaStatus,
    RoutingDecision,
    RoutingAuditEntry,
    QuotaTracker,
    EndpointHealthTracker,
    LoadBalancingStrategy,
    EndpointStatus,
    RoutingEventType,
    TenantRoutingError,
    TenantNotFoundError,
    NoAvailableEndpointError,
    QuotaExceededError,
    CrossTenantAccessError,
)

from aragora.gateway.enterprise.proxy import (
    # Exceptions
    ProxyError,
    CircuitOpenError,
    BulkheadFullError,
    RequestTimeoutError,
    FrameworkNotConfiguredError,
    SanitizationError,
    # Enums
    HealthStatus,
    RetryStrategy,
    # Configuration
    CircuitBreakerSettings,
    RetrySettings,
    BulkheadSettings,
    SanitizationSettings,
    ExternalFrameworkConfig,
    ProxyConfig,
    # Request/Response
    ProxyRequest,
    ProxyResponse,
    # Health
    HealthCheckResult,
    # Components
    FrameworkCircuitBreaker,
    FrameworkBulkhead,
    RequestSanitizer,
    # Main class
    EnterpriseProxy,
)

__all__ = [
    # Audit Interceptor
    "AuditInterceptor",
    "AuditRecord",
    "AuditConfig",
    "PIIRedactionRule",
    "RedactionType",
    "AuditEventType",
    "AuditStorage",
    "InMemoryAuditStorage",
    "PostgresAuditStorage",
    # Auth Bridge
    "AuthBridge",
    "AuthBridgeError",
    "AuthenticationError",
    "PermissionDeniedError",
    "SessionExpiredError",
    "TokenType",
    "AuthContext",
    "PermissionMapping",
    "TokenExchangeResult",
    "BridgedSession",
    "AuditEntry",
    "SessionLifecycleHook",
    # Tenant Router
    "TenantRouter",
    "TenantRoutingConfig",
    "TenantRoutingContext",
    "EndpointConfig",
    "EndpointHealth",
    "TenantQuotas",
    "QuotaStatus",
    "RoutingDecision",
    "RoutingAuditEntry",
    "QuotaTracker",
    "EndpointHealthTracker",
    "LoadBalancingStrategy",
    "EndpointStatus",
    "RoutingEventType",
    "TenantRoutingError",
    "TenantNotFoundError",
    "NoAvailableEndpointError",
    "QuotaExceededError",
    "CrossTenantAccessError",
    # Enterprise Proxy - Exceptions
    "ProxyError",
    "CircuitOpenError",
    "BulkheadFullError",
    "RequestTimeoutError",
    "FrameworkNotConfiguredError",
    "SanitizationError",
    # Enterprise Proxy - Enums
    "HealthStatus",
    "RetryStrategy",
    # Enterprise Proxy - Configuration
    "CircuitBreakerSettings",
    "RetrySettings",
    "BulkheadSettings",
    "SanitizationSettings",
    "ExternalFrameworkConfig",
    "ProxyConfig",
    # Enterprise Proxy - Request/Response
    "ProxyRequest",
    "ProxyResponse",
    # Enterprise Proxy - Health
    "HealthCheckResult",
    # Enterprise Proxy - Components
    "FrameworkCircuitBreaker",
    "FrameworkBulkhead",
    "RequestSanitizer",
    # Enterprise Proxy - Main class
    "EnterpriseProxy",
]
