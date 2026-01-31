# OpenClaw Secure Gateway Adapter - Implementation Plan

## Overview

This plan implements a **secure enterprise gateway** that wraps OpenClaw's open-source AI assistant capabilities with Aragora's enterprise security layer. This hybrid approach leverages OpenClaw's 100k+ star community momentum while providing enterprise-grade security, RBAC, audit logging, and policy enforcement.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Aragora Enterprise Layer                      │
│  ┌─────────┐ ┌──────┐ ┌───────┐ ┌────────┐ ┌──────────────┐    │
│  │  Auth   │ │ RBAC │ │ Audit │ │ Policy │ │ KM Receipts  │    │
│  └────┬────┘ └───┬──┘ └───┬───┘ └───┬────┘ └──────┬───────┘    │
├───────┼──────────┼────────┼─────────┼─────────────┼─────────────┤
│       └──────────┴────────┴─────────┴─────────────┘             │
│                    Secure Gateway Adapter                        │
│  ┌─────────────┐ ┌──────────────┐ ┌───────────────────────┐     │
│  │  Protocol   │ │   Sandbox    │ │  Capability Filter    │     │
│  │ Translator  │ │   Runner     │ │  (allow/block/wrap)   │     │
│  └──────┬──────┘ └──────┬───────┘ └───────────┬───────────┘     │
├─────────┼───────────────┼─────────────────────┼─────────────────┤
│         └───────────────┴─────────────────────┘                  │
│                      OpenClaw Runtime                            │
│  ┌────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────────┐     │
│  │ Tasks  │ │ Devices  │ │ Channels │ │ Community Plugins │     │
│  └────────┘ └──────────┘ └──────────┘ └───────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Steps

### Phase 1: Core Gateway Module Structure

**Location:** `aragora/gateway/openclaw/`

#### Step 1.1: Module Scaffold
Create the module structure:
```
aragora/gateway/openclaw/
├── __init__.py           # Public API exports
├── adapter.py            # OpenClawGatewayAdapter main class
├── protocol.py           # Protocol translation (Aragora ↔ OpenClaw)
├── sandbox.py            # Isolated execution environment
├── capabilities.py       # Capability filtering/wrapping
├── policy.py             # Gateway-specific policy rules
└── audit.py              # Gateway audit event types
```

#### Step 1.2: Gateway Adapter Core (`adapter.py`)
```python
class OpenClawGatewayAdapter:
    """Secure enterprise gateway to OpenClaw runtime."""

    def __init__(
        self,
        openclaw_endpoint: str,
        rbac_checker: PermissionChecker,
        audit_logger: AuditLogger,
        policy_engine: PolicyEngine,
        sandbox_config: SandboxConfig,
    ): ...

    async def execute_task(
        self,
        task: OpenClawTask,
        auth_context: AuthorizationContext,
    ) -> GatewayResult: ...

    async def register_device(
        self,
        device: DeviceRegistration,
        auth_context: AuthorizationContext,
    ) -> DeviceHandle: ...
```

### Phase 2: Security Layer Integration

#### Step 2.1: RBAC Integration
Define OpenClaw-specific permissions in `aragora/rbac/defaults.py`:
```python
# New permissions for OpenClaw gateway
"openclaw:task:execute"      # Execute tasks via OpenClaw
"openclaw:task:execute:*"    # Execute any task type
"openclaw:device:register"   # Register devices
"openclaw:device:admin"      # Manage all devices
"openclaw:plugin:install"    # Install community plugins
"openclaw:plugin:admin"      # Manage plugin allowlist
```

#### Step 2.2: Audit Event Types
Add to `aragora/gateway/openclaw/audit.py`:
```python
class OpenClawAuditEvents:
    TASK_SUBMITTED = "openclaw.task.submitted"
    TASK_COMPLETED = "openclaw.task.completed"
    TASK_BLOCKED = "openclaw.task.blocked"
    DEVICE_REGISTERED = "openclaw.device.registered"
    PLUGIN_INSTALLED = "openclaw.plugin.installed"
    CAPABILITY_DENIED = "openclaw.capability.denied"
```

#### Step 2.3: Policy Rules
Create gateway-specific policies in `capabilities.py`:
```python
class CapabilityFilter:
    """Filter OpenClaw capabilities based on enterprise policy."""

    # Capabilities that require approval gates
    APPROVAL_REQUIRED = {
        "file_system_write",
        "network_external",
        "code_execution",
        "credential_access",
    }

    # Capabilities blocked by default (can be enabled per-tenant)
    BLOCKED_BY_DEFAULT = {
        "shell_execute",
        "admin_escalate",
        "data_export_bulk",
    }

    # Capabilities always allowed
    ALWAYS_ALLOWED = {
        "text_generation",
        "search_internal",
        "calendar_read",
    }
```

### Phase 3: Sandbox Runner

#### Step 3.1: Sandbox Configuration
```python
@dataclass
class SandboxConfig:
    """Configuration for isolated OpenClaw execution."""

    # Resource limits
    max_memory_mb: int = 512
    max_cpu_percent: int = 50
    max_execution_seconds: int = 300

    # Network isolation
    allow_external_network: bool = False
    allowed_domains: list[str] = field(default_factory=list)

    # File system isolation
    allowed_paths: list[str] = field(default_factory=list)
    read_only_paths: list[str] = field(default_factory=list)

    # Plugin restrictions
    allowed_plugins: list[str] = field(default_factory=list)
    plugin_allowlist_mode: bool = True  # Only allowed plugins can run
```

#### Step 3.2: Sandbox Runner (`sandbox.py`)
```python
class OpenClawSandbox:
    """Isolated execution environment for OpenClaw tasks."""

    async def execute(
        self,
        task: OpenClawTask,
        config: SandboxConfig,
    ) -> SandboxResult:
        """Execute task in isolated environment with resource limits."""
        ...

    async def _apply_resource_limits(self): ...
    async def _setup_network_isolation(self): ...
    async def _mount_filesystem(self): ...
```

### Phase 4: Protocol Translation

#### Step 4.1: Protocol Translator (`protocol.py`)
```python
class OpenClawProtocolTranslator:
    """Translate between Aragora and OpenClaw message formats."""

    def aragora_to_openclaw(
        self,
        request: AragoraRequest,
    ) -> OpenClawTask:
        """Convert Aragora decision request to OpenClaw task."""
        ...

    def openclaw_to_aragora(
        self,
        result: OpenClawResult,
    ) -> DecisionResult:
        """Convert OpenClaw result to Aragora decision result."""
        ...

    def wrap_with_context(
        self,
        task: OpenClawTask,
        auth_context: AuthorizationContext,
        tenant_context: TenantContext,
    ) -> OpenClawTask:
        """Inject enterprise context into OpenClaw task."""
        ...
```

### Phase 5: Integration Points

#### Step 5.1: HTTP Handler (`aragora/server/handlers/gateway/openclaw.py`)
```python
@require_permission("openclaw:task:execute")
@auth_rate_limit(requests_per_minute=30)
async def handle_openclaw_execute(
    data: dict[str, Any],
    user_id: str,
) -> HandlerResult:
    """Execute task via OpenClaw gateway."""
    ...

@require_permission("openclaw:device:register")
async def handle_openclaw_register_device(
    data: dict[str, Any],
    user_id: str,
) -> HandlerResult:
    """Register device via OpenClaw gateway."""
    ...
```

#### Step 5.2: WebSocket Stream (`aragora/server/stream/openclaw_stream.py`)
For real-time task updates from OpenClaw:
```python
class OpenClawStreamHandler:
    """WebSocket handler for OpenClaw task streaming."""

    async def handle_task_update(self, update: TaskUpdate): ...
    async def handle_device_event(self, event: DeviceEvent): ...
```

#### Step 5.3: Knowledge Mound Adapter
Create `aragora/knowledge/mound/adapters/openclaw_adapter.py`:
- Store task receipts
- Index device capabilities
- Track plugin usage patterns

### Phase 6: Deployment Configuration

#### Step 6.1: Environment Variables
```bash
# OpenClaw Gateway Configuration
OPENCLAW_ENDPOINT=http://localhost:8081
OPENCLAW_API_KEY=...
OPENCLAW_SANDBOX_ENABLED=true
OPENCLAW_MAX_CONCURRENT_TASKS=10
OPENCLAW_PLUGIN_ALLOWLIST=plugin1,plugin2,plugin3
```

#### Step 6.2: Docker Compose Profile
Add to `docker-compose.yml`:
```yaml
services:
  aragora-openclaw-gateway:
    build:
      context: .
      dockerfile: Dockerfile.openclaw-gateway
    environment:
      - OPENCLAW_ENDPOINT=${OPENCLAW_ENDPOINT}
    depends_on:
      - aragora-api
      - openclaw-runtime
    networks:
      - aragora-internal
      - openclaw-isolated
```

### Phase 7: Testing

#### Step 7.1: Unit Tests
- `tests/gateway/openclaw/test_adapter.py`
- `tests/gateway/openclaw/test_protocol.py`
- `tests/gateway/openclaw/test_sandbox.py`
- `tests/gateway/openclaw/test_capabilities.py`

#### Step 7.2: Integration Tests
- `tests/gateway/openclaw/test_integration.py` - End-to-end flow
- `tests/gateway/openclaw/test_security.py` - RBAC, audit, policy

### Implementation Order

1. **Module scaffold** - Create directory structure and `__init__.py`
2. **Protocol translator** - Core message translation
3. **Capability filter** - Security policy enforcement
4. **Gateway adapter** - Main orchestration class
5. **Sandbox runner** - Isolated execution
6. **RBAC permissions** - Add to defaults
7. **Audit events** - Define event types
8. **HTTP handlers** - REST API endpoints
9. **WebSocket stream** - Real-time updates
10. **KM adapter** - Receipt storage
11. **Tests** - Unit and integration
12. **Documentation** - API docs and deployment guide

## Security Considerations

1. **All OpenClaw tasks pass through RBAC** - No direct access
2. **Audit logging for every operation** - Full traceability
3. **Sandbox isolation** - Resource limits, network isolation
4. **Plugin allowlist** - Only approved plugins can execute
5. **Capability filtering** - Enterprise policy enforcement
6. **Tenant isolation** - Multi-tenant safe

## Success Criteria

- [ ] OpenClaw tasks execute with full audit trail
- [ ] RBAC enforced on all gateway operations
- [ ] Sandbox limits prevent resource abuse
- [ ] Plugin allowlist blocks unauthorized extensions
- [ ] Knowledge Mound stores task receipts
- [ ] WebSocket streaming works for real-time updates
- [ ] 90%+ test coverage on gateway module
