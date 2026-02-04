# Gateway

Enterprise security gateway for external agent frameworks. Routes messages, enforces tenant isolation, bridges authentication, and provides auditable integration with OpenClaw, AutoGPT, CrewAI, LangGraph, and other AI frameworks.

## Overview

The gateway sits between Aragora's debate engine and external agent frameworks, providing unified message routing, enterprise-grade security controls, and intelligent decision routing. It aggregates channels into a unified inbox, manages device registrations, and enforces RBAC, PII redaction, and compliance policies on every request.

## Module Reference

| Sub-package / Module | Purpose |
|----------------------|---------|
| `server.py` | `LocalGateway` HTTP/WebSocket server |
| `inbox.py` | Unified inbox aggregation across channels |
| `router.py`, `capability_router.py` | Rule-based and device-aware agent routing |
| `device_registry.py`, `device_security.py` | Device registration, heartbeats, secure pairing |
| `protocol.py` | WebSocket session and presence management |
| `persistence.py` | Storage backends (memory, file, Redis) |
| `credential_proxy.py` | Tenant-scoped credential proxy with rate limiting |
| `decision_router.py` | Intelligent debate vs. execution routing |
| `enterprise/` | Auth bridge, tenant router, audit interceptor, resilient proxy |
| `openclaw/` | OpenClaw adapter, action filter, credential vault, sandbox |
| `federation/` | Multi-framework registry and capability discovery |
| `security/` | Credential vault, output filter, audit bridge |
| `orchestration/` | Task routing strategies and fallback chains |
| `external_agents/` | Base adapter, policy, and sandbox for external agents |

## Quick Start

```python
from aragora.gateway import LocalGateway, GatewayConfig
from aragora.gateway.decision_router import DecisionRouter, RoutingCriteria

gw = LocalGateway(config=GatewayConfig(host="0.0.0.0", port=8090, enable_auth=True))
await gw.start()

router = DecisionRouter(criteria=RoutingCriteria(
    financial_threshold=10000.0,
    risk_levels={"high", "critical"},
    compliance_flags={"pii", "hipaa", "gdpr"},
))
decision = await router.route(
    {"action": "transfer", "amount": 50000},
    context={"tenant_id": "acme-corp"},
)  # decision.destination == RouteDestination.DEBATE
```

## Enterprise Security

The `enterprise/` sub-package provides compliance-ready security controls.

| Component | Module | Purpose |
|-----------|--------|---------|
| `AuthBridge` | `auth_bridge.py` | Token verification, permission mapping, token exchange |
| `TenantRouter` | `tenant_router.py` | Multi-tenant routing with quota enforcement |
| `AuditInterceptor` | `audit_interceptor.py` | Request/response audit with PII redaction |
| `EnterpriseProxy` | `proxy.py` | Resilient proxy with circuit breakers and bulkheads |

```python
from aragora.gateway.enterprise import (
    AuthBridge, PermissionMapping,
    TenantRouter, TenantRoutingConfig, TenantQuotas, EndpointConfig,
    AuditInterceptor, AuditConfig,
)

# Map Aragora permissions to external framework actions
bridge = AuthBridge(permission_mappings=[
    PermissionMapping(aragora_permission="debates.create", external_action="create_conversation"),
])
context = await bridge.verify_request(token="...")

# Tenant routing with quota enforcement
tenant_router = TenantRouter(configs=[TenantRoutingConfig(
    tenant_id="acme-corp",
    endpoints=[EndpointConfig(url="https://acme.api.example.com")],
    quotas=TenantQuotas(requests_per_minute=100, requests_per_day=10000),
)])

# Audit with PII redaction
interceptor = AuditInterceptor(config=AuditConfig(retention_days=365, emit_events=True))
record = await interceptor.intercept(request=req, response=resp, correlation_id="req-123")
```

## OpenClaw Integration

The `openclaw/` sub-package wraps OpenClaw with RBAC, sandboxing, and action filtering.

| Component | Module | Purpose |
|-----------|--------|---------|
| `OpenClawAdapter` | `adapter.py` | Session management and action execution |
| `ActionFilter` | `action_filter.py` | Allowlist/denylist with risk assessment |
| `CredentialVault` | `credential_vault.py` | AES-256-GCM encrypted credential storage |
| `CapabilityFilter` | `capabilities.py` | Allow/block/require-approval per capability |
| `OpenClawSandbox` | `sandbox.py` | Resource-limited sandbox isolation |

```python
from aragora.gateway.openclaw import (
    OpenClawAdapter, OpenClawAction, CredentialType, get_credential_vault,
)

adapter = OpenClawAdapter(openclaw_endpoint="http://localhost:8081", rbac_checker=checker)
session = await adapter.create_session(user_id="user-123", channel="telegram", tenant_id="t-456")
result = await adapter.execute_action(
    session_id=session.session_id,
    action=OpenClawAction(action_type="browser_navigate", parameters={"url": "https://example.com"}),
)

# Encrypted credential storage with rotation policies
vault = get_credential_vault()
await vault.store_credential(
    tenant_id="acme", framework="openai",
    credential_type=CredentialType.API_KEY, value="sk-...", auth_context=ctx,
)
```

## Federation

The `federation/` sub-package provides a registry for external AI frameworks.

```python
from aragora.gateway.federation import FederationRegistry, FrameworkCapability

registry = FederationRegistry()
await registry.connect()
await registry.register(
    name="autogpt", version="0.5.0",
    capabilities=[FrameworkCapability(
        name="autonomous_task", description="Execute autonomous multi-step tasks",
        parameters={"task": "str", "max_steps": "int"}, returns="TaskResult",
    )],
    endpoints={"base": "http://localhost:8090", "health": "/health"},
)
frameworks = await registry.find_by_capability("autonomous_task")
```

## Decision Routing

The `DecisionRouter` determines whether a request needs multi-agent debate or direct execution. Evaluation order: explicit user intent, custom rules, financial thresholds, risk levels, compliance flags, stakeholder count, then category defaults.

| Destination | When Used |
|-------------|-----------|
| `DEBATE` | High-risk, financial, compliance, or multi-stakeholder decisions |
| `EXECUTE` | Low-risk tasks, explicit execution requests |
| `HYBRID_DEBATE_THEN_EXECUTE` | Debate first, then execute the outcome |
| `HYBRID_EXECUTE_WITH_VALIDATION` | Execute first, validate via debate |
| `REJECT` | Request denied by policy |

```python
from aragora.gateway.decision_router import (
    DecisionRouter, RoutingCriteria, RoutingRule,
    RouteDestination, TenantRoutingConfig, SimpleAnomalyDetector,
)

router = DecisionRouter(
    criteria=RoutingCriteria(financial_threshold=10000.0),
    anomaly_detector=SimpleAnomalyDetector(min_samples=100),
)
router.add_rule(RoutingRule(
    rule_id="large-transactions",
    condition=lambda req: req.get("amount", 0) > 50000,
    destination=RouteDestination.DEBATE, priority=100,
    reason="Large financial transactions require debate consensus",
))
await router.add_tenant_config(TenantRoutingConfig(
    tenant_id="acme-corp", criteria=RoutingCriteria(financial_threshold=5000.0),
))
decision = await router.route({"action": "transfer", "amount": 75000}, context={"tenant_id": "acme-corp"})
```

## Configuration

| Setting | Default |
|---------|---------|
| `ARAGORA_GATEWAY_STORE` (env) | `auto` (Redis if available, else file) |
| `ARAGORA_GATEWAY_STORE_PATH` (env) | `~/.aragora/gateway.json` |
| `ARAGORA_GATEWAY_STORE_REDIS_URL` (env) | `redis://localhost:6379` |
| `ARAGORA_GATEWAY_SESSION_STORE` (env) | `auto` (Redis if available, else file) |
| `ARAGORA_GATEWAY_SESSION_PATH` (env) | `~/.aragora/gateway.json` |
| `ARAGORA_GATEWAY_SESSION_REDIS_URL` (env) | `redis://localhost:6379` |
| `GatewayConfig.port` | `8090` |
| `GatewayConfig.enable_auth` | `False` |
| `RoutingCriteria.financial_threshold` | `10000.0` |
| `RoutingCriteria.confidence_threshold` | `0.85` |
| `AuditConfig.retention_days` | `365` |
| `TenantQuotas.requests_per_minute` | Per-tenant |

### Routing Criteria Env Overrides

You can centralize Gateway routing criteria for the unified router by setting:

- `ARAGORA_ROUTING_FINANCIAL_THRESHOLD`
- `ARAGORA_ROUTING_RISK_LEVELS` (comma-separated, e.g. `high,critical`)
- `ARAGORA_ROUTING_COMPLIANCE_FLAGS` (comma-separated, e.g. `pii,hipaa,gdpr`)
- `ARAGORA_ROUTING_STAKEHOLDER_THRESHOLD`
- `ARAGORA_ROUTING_REQUIRE_DEBATE_KEYWORDS` (comma-separated)
- `ARAGORA_ROUTING_REQUIRE_EXECUTE_KEYWORDS` (comma-separated)
- `ARAGORA_ROUTING_TIME_SENSITIVE_SECONDS`
- `ARAGORA_ROUTING_CONFIDENCE_THRESHOLD`
