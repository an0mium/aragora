# Sandbox Module

Secure code execution environment for Aragora, providing Docker-based isolation, container pooling, tool policy enforcement, and session lifecycle management for agent evaluation.

## Overview

The sandbox module enables:

- **Docker Isolation**: Full container isolation with resource limits and network restrictions
- **Container Pooling**: Pre-warmed container pool for low-latency execution (<100ms acquisition)
- **Tool Policies**: Allowlist/denylist-based tool and file access control
- **Session Management**: Per-session containers with tenant isolation
- **Resource Limits**: CPU, memory, time, and process limits per execution

## Architecture

```
aragora/sandbox/
├── __init__.py          # Module exports
├── executor.py          # SandboxExecutor for code execution
├── pool.py              # ContainerPool for pre-warmed containers
├── policies.py          # Tool policies and access control
└── lifecycle.py         # Session container lifecycle management
```

## Key Classes

### Executor

- **`SandboxExecutor`**: Executes code in isolated environments
  - Supports Docker, subprocess, and mock execution modes
  - Applies resource limits (CPU, memory, time)
  - Enforces tool policies before execution
  - Captures stdout, stderr, and created files

- **`ExecutionResult`**: Result of sandboxed execution
  - Status, exit code, stdout/stderr
  - Duration, memory usage
  - Files created, policy violations

- **`SandboxConfig`**: Execution configuration
  - Mode (Docker/subprocess/mock)
  - Docker image, workspace path
  - Network and cleanup settings

### Container Pool

- **`ContainerPool`**: Pre-warmed container pool
  - Dynamic scaling based on demand
  - Health monitoring and automatic recovery
  - Session binding for per-session isolation

- **`ContainerPoolConfig`**: Pool configuration
  - Min/max pool size, warmup count
  - Idle timeout, container age limits
  - Resource limits per container

- **`PooledContainer`**: Managed container instance
  - Container ID and state
  - Session binding, execution count
  - Health check tracking

### Policies

- **`ToolPolicy`**: Complete policy definition
  - Tool rules (allowlist/denylist)
  - Path rules (read/write access)
  - Network rules (host, port, protocol)
  - Resource limits

- **`ToolPolicyChecker`**: Policy enforcement
  - Check tool, path, and network access
  - Audit logging for policy decisions

### Session Lifecycle

- **`SessionContainerManager`**: Per-session container management
  - Session-bound container allocation
  - Policy-based execution control
  - Tenant isolation enforcement
  - Automatic cleanup of expired sessions

- **`ContainerSession`**: Session state and configuration
  - Session ID, tenant ID, user ID
  - Execution count, total time
  - Files created, environment variables

## Usage Example

### Basic Code Execution

```python
from aragora.sandbox import SandboxExecutor, SandboxConfig, ExecutionMode

# Create executor with default policy
executor = SandboxExecutor()

# Execute Python code
result = await executor.execute(
    code="""
import math
print(f"Pi is approximately {math.pi:.10f}")
for i in range(5):
    print(f"Square of {i} is {i**2}")
""",
    language="python",
    timeout=30.0,
)

print(f"Status: {result.status}")
print(f"Exit code: {result.exit_code}")
print(f"Output:\n{result.stdout}")
print(f"Duration: {result.duration_seconds:.2f}s")

# Execute with Docker isolation
docker_executor = SandboxExecutor(
    config=SandboxConfig(
        mode=ExecutionMode.DOCKER,
        docker_image="python:3.11-slim",
        network_enabled=False,
    )
)

result = await docker_executor.execute(
    code="print('Hello from Docker!')",
    language="python",
)
```

### Using Container Pool

```python
from aragora.sandbox import (
    ContainerPool,
    ContainerPoolConfig,
    get_container_pool,
)

# Create pool with custom configuration
config = ContainerPoolConfig(
    min_pool_size=5,
    max_pool_size=50,
    warmup_count=10,
    idle_timeout_seconds=300,
    memory_limit_mb=512,
    cpu_limit=1.0,
    network_mode="none",
)
pool = ContainerPool(config)

# Start the pool
await pool.start()

# Acquire container for a session
container = await pool.acquire("session-123")
print(f"Acquired container: {container.container_id}")

# Execute code in container
# (use SessionContainerManager for this)

# Release container back to pool
await pool.release("session-123")

# Get pool statistics
stats = pool.stats
print(f"Total containers: {stats.total_containers}")
print(f"Ready: {stats.ready_containers}")
print(f"Acquired: {stats.acquired_containers}")
print(f"Avg acquire time: {stats.avg_acquire_time_ms:.1f}ms")

# Scale pool dynamically
await pool.scale_up(10)
await pool.scale_down(5)

# Stop pool gracefully
await pool.stop(graceful=True)
```

### Custom Tool Policies

```python
from aragora.sandbox import (
    ToolPolicy,
    ToolPolicyChecker,
    ToolRule,
    PathRule,
    NetworkRule,
    ResourceLimit,
    PolicyAction,
    create_default_policy,
    create_strict_policy,
)

# Use pre-built policies
default_policy = create_default_policy()  # Balanced security
strict_policy = create_strict_policy()    # Minimal permissions

# Create custom policy
custom_policy = ToolPolicy(
    name="data-science",
    description="Policy for data science workloads",
    default_tool_action=PolicyAction.DENY,
    default_path_action=PolicyAction.DENY,
    default_network_action=PolicyAction.DENY,
)

# Allow Python and common data science tools
custom_policy.add_tool_allowlist([
    r"^python3?$",
    r"^pip3?$",
    r"^jupyter$",
    r"^nbconvert$",
], reason="Data science tools")

# Deny dangerous operations
custom_policy.add_tool_denylist([
    r"^rm$", r"^sudo$", r"^curl$", r"^wget$",
], reason="Security restrictions")

# Allow workspace and read-only library access
custom_policy.add_path_allowlist(
    patterns=[r"^/workspace/.*$"],
    read=True, write=True,
    reason="Workspace directory",
)
custom_policy.add_path_allowlist(
    patterns=[r"^/usr/lib/.*$", r"^/home/.*\.local/.*$"],
    read=True, write=False,
    reason="Libraries (read-only)",
)

# Allow specific network access for package installation
custom_policy.add_network_allowlist(
    host_patterns=[r"^pypi\.org$", r"^files\.pythonhosted\.org$"],
    ports=(443, 443),
    reason="PyPI access",
)

# Set resource limits
custom_policy.resource_limits = ResourceLimit(
    max_memory_mb=2048,
    max_cpu_percent=200,
    max_execution_seconds=600,
    max_processes=20,
)

# Use policy with executor
executor = SandboxExecutor(
    config=SandboxConfig(policy=custom_policy)
)

# Check policy manually
checker = ToolPolicyChecker(custom_policy)
allowed, reason = checker.check_tool("python3")
print(f"python3 allowed: {allowed} ({reason})")

allowed, reason = checker.check_path("/workspace/data.csv", "read")
print(f"Read /workspace/data.csv: {allowed} ({reason})")

allowed, reason = checker.check_network("pypi.org", 443, "https")
print(f"HTTPS to pypi.org: {allowed} ({reason})")
```

### Session Management

```python
from aragora.sandbox import (
    SessionContainerManager,
    SessionConfig,
    get_session_manager,
)

# Get or create the global session manager
manager = get_session_manager()
await manager.start()

# Create session configuration
session_config = SessionConfig(
    max_execution_time_seconds=300,
    max_session_duration_seconds=3600,
    max_executions=100,
    max_memory_mb=512,
    network_enabled=False,
    file_persistence=True,
)

# Create a session for a tenant
session = await manager.create_session(
    tenant_id="tenant-456",
    user_id="user-123",
    config=session_config,
    metadata={"purpose": "data analysis"},
)
print(f"Session created: {session.session_id}")

# Execute code in the session
result = await manager.execute(
    session_id=session.session_id,
    code="x = 10\nprint(f'x = {x}')",
    language="python",
)
print(f"Output: {result.stdout}")

# Variables persist between executions
result = await manager.execute(
    session_id=session.session_id,
    code="print(f'x is still {x}')",
    language="python",
)
print(f"Output: {result.stdout}")

# Get session statistics
print(f"Executions: {session.execution_count}")
print(f"Total time: {session.total_execution_time_seconds:.2f}s")

# List sessions for a tenant
tenant_sessions = await manager.list_sessions(tenant_id="tenant-456")

# Suspend and resume session
await manager.suspend_session(session.session_id)
await manager.resume_session(session.session_id)

# Terminate session
await manager.terminate_session(session.session_id)

# Terminate all sessions for a tenant
count = await manager.terminate_tenant_sessions("tenant-456")
print(f"Terminated {count} sessions")

# Stop manager
await manager.stop()
```

## Integration Points

### With Agent System
- Execute agent-generated code safely
- Policy enforcement for agent capabilities
- Resource limits prevent runaway processes

### With Debate Engine
- Code execution for code review debates
- Sandboxed evaluation of proposed solutions
- Output capture for consensus building

### With Multi-Tenancy
- Tenant-isolated sessions
- Per-tenant resource quotas
- Automatic cleanup on tenant termination

### With Observability
- Execution metrics (duration, memory, success)
- Policy violation tracking
- Container pool statistics

## Resource Limits

### Default Limits (per execution)

| Resource | Limit |
|----------|-------|
| Memory | 512 MB |
| CPU | 100% of 1 core |
| Execution time | 60 seconds |
| Processes | 10 |
| File size | 10 MB |
| Files created | 100 |
| Network requests | 50 |

### Container Pool Defaults

| Setting | Value |
|---------|-------|
| Min pool size | 5 |
| Max pool size | 50 |
| Warmup count | 10 |
| Idle timeout | 300 seconds |
| Max container age | 3600 seconds |
| Health check interval | 30 seconds |

## Security Considerations

1. **Docker Security**: Containers run with `--security-opt=no-new-privileges`, `--read-only`, and `--network=none` by default
2. **Tool Allowlisting**: Only explicitly allowed tools can be executed
3. **Path Restrictions**: File access limited to designated directories
4. **Resource Limits**: Prevent DoS through CPU, memory, and time limits
5. **Audit Logging**: All policy decisions logged for security review

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_CONTAINER_POOL_MIN` | Minimum pool size | `5` |
| `ARAGORA_CONTAINER_POOL_MAX` | Maximum pool size | `50` |
| `ARAGORA_CONTAINER_POOL_WARMUP` | Initial warmup count | `10` |
| `ARAGORA_SANDBOX_IMAGE` | Docker image for sandbox | `python:3.11-slim` |
| `ARAGORA_SANDBOX_NETWORK` | Docker network mode | `none` |

## See Also

- `aragora/agents/airlock.py` - Agent resilience proxy
- `aragora/gauntlet/` - Agent testing framework
- `docs/SECURITY.md` - Security architecture
- `docs/SANDBOX.md` - Full sandbox guide
