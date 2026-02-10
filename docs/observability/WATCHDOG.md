# Three-Tier Watchdog Guide

The Three-Tier Watchdog implements the Gastown pattern for multi-level agent monitoring and escalation. Each tier handles different aspects of system health, from low-level heartbeats to business-level SLA compliance.

## Overview

The Watchdog system provides:
- **Three-tier monitoring**: Mechanical, Boot Agent, and Deacon tiers
- **Automatic escalation**: Issues escalate to higher tiers when unresolved
- **Agent health tracking**: Per-agent metrics for requests, latency, memory
- **Issue management**: Detect, track, and resolve issues systematically
- **Configurable thresholds**: Per-tier configuration for warnings and criticals
- **Handler registration**: Custom callbacks for each tier's issues

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Three-Tier Watchdog                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Tier 3: DEACON (Business Level)                                    │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  SLA Compliance | Cross-Agent Coordination | Global Policy   │   │
│  └────────────────────────────┬─────────────────────────────────┘   │
│                               │ escalate                            │
│  Tier 2: BOOT_AGENT (Quality Level)                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Response Quality | Latency Monitoring | Error Rate Tracking │   │
│  └────────────────────────────┬─────────────────────────────────┘   │
│                               │ escalate                            │
│  Tier 1: MECHANICAL (Infrastructure Level)                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Heartbeat | Memory Usage | Circuit Breaker State            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
from aragora.control_plane.watchdog import (
    ThreeTierWatchdog,
    WatchdogConfig,
    WatchdogTier,
    WatchdogIssue,
    IssueSeverity,
    IssueCategory,
    get_watchdog,
)

# Create watchdog
watchdog = ThreeTierWatchdog()

# Configure mechanical tier
watchdog.configure_tier(WatchdogConfig(
    tier=WatchdogTier.MECHANICAL,
    check_interval_seconds=5.0,
    heartbeat_timeout_seconds=30.0,
    memory_warning_mb=1024.0,
    memory_critical_mb=2048.0,
))

# Register agents for monitoring
watchdog.register_agent("claude-opus")
watchdog.register_agent("gpt-4o")

# Start monitoring
await watchdog.start()

# Record agent activity
watchdog.record_heartbeat("claude-opus")
watchdog.record_request("claude-opus", success=True, latency_ms=1500)
watchdog.update_memory_usage("claude-opus", memory_mb=512)

# Get health status
health = watchdog.get_agent_health("claude-opus")
print(f"Error rate: {health.error_rate:.1%}")
print(f"Avg latency: {health.average_latency_ms:.0f}ms")

# Stop monitoring
await watchdog.stop()
```

## Tier Responsibilities

### Tier 1: MECHANICAL

Low-level infrastructure checks:

| Check | Category | Description |
|-------|----------|-------------|
| Heartbeat | `HEARTBEAT_MISSING` | Agent not responding within timeout |
| Memory | `MEMORY_EXCEEDED` | Memory usage above threshold |
| Circuit Breaker | `CIRCUIT_OPEN` | Circuit breaker has opened |
| Resources | `RESOURCE_EXHAUSTED` | System resources depleted |

### Tier 2: BOOT_AGENT

Quality and performance monitoring:

| Check | Category | Description |
|-------|----------|-------------|
| Latency | `LATENCY_EXCEEDED` | Response time above threshold |
| Quality | `RESPONSE_QUALITY_LOW` | Response quality score too low |
| Errors | `ERROR_RATE_HIGH` | Error rate above threshold |
| Semantic | `SEMANTIC_DRIFT` | Agent responses drifting from expected |

### Tier 3: DEACON

Business-level oversight:

| Check | Category | Description |
|-------|----------|-------------|
| SLA | `SLA_VIOLATION` | Availability below SLA target |
| Coordination | `COORDINATION_FAILURE` | Majority of agents unhealthy |
| Policy | `POLICY_VIOLATION` | Global policy rules violated |
| Consensus | `CONSENSUS_BLOCKED` | Debate consensus cannot be reached |

## Core Classes

### WatchdogTier

The three monitoring tiers:

```python
from aragora.control_plane.watchdog import WatchdogTier

class WatchdogTier(str, Enum):
    MECHANICAL = "mechanical"   # Tier 1: Infrastructure
    BOOT_AGENT = "boot_agent"   # Tier 2: Quality
    DEACON = "deacon"           # Tier 3: Business
```

### IssueSeverity

Issue severity levels:

```python
from aragora.control_plane.watchdog import IssueSeverity

class IssueSeverity(IntEnum):
    INFO = 0       # Informational, no action needed
    WARNING = 1    # Potential issue, monitor closely
    ERROR = 2      # Active issue, needs attention
    CRITICAL = 3   # Severe issue, immediate action required
```

### WatchdogIssue

Represents a detected issue:

```python
from aragora.control_plane.watchdog import WatchdogIssue, IssueCategory

issue = WatchdogIssue(
    severity=IssueSeverity.ERROR,
    category=IssueCategory.HEARTBEAT_MISSING,
    agent="claude-opus",
    message="No heartbeat for 45.2s",
    details={"elapsed_seconds": 45.2},
)

# Issue tracking
print(f"ID: {issue.id}")
print(f"Detected at: {issue.detected_at}")
print(f"Detected by: {issue.detected_by}")

# Resolution
issue.resolved = True
issue.resolved_at = datetime.now(timezone.utc)
issue.resolution_notes = "Agent restarted"

# Serialize
data = issue.to_dict()
```

### WatchdogConfig

Per-tier configuration:

```python
from aragora.control_plane.watchdog import WatchdogConfig, WatchdogTier

config = WatchdogConfig(
    tier=WatchdogTier.MECHANICAL,

    # Timing
    check_interval_seconds=5.0,      # How often to run checks
    heartbeat_timeout_seconds=30.0,  # Max time without heartbeat

    # Memory thresholds
    memory_warning_mb=1024.0,        # Warning at 1GB
    memory_critical_mb=2048.0,       # Critical at 2GB

    # Latency thresholds
    latency_warning_ms=5000.0,       # Warning at 5s
    latency_critical_ms=15000.0,     # Critical at 15s

    # Error rate thresholds
    error_rate_warning=0.1,          # Warning at 10%
    error_rate_critical=0.3,         # Critical at 30%

    # Escalation
    auto_escalate=True,              # Auto-escalate errors
    escalation_threshold=3,          # Issues before escalating

    # SLA (Deacon tier)
    sla_response_time_ms=10000.0,    # Target response time
    sla_availability_pct=99.0,       # Target availability
)
```

### AgentHealth

Per-agent health metrics:

```python
from aragora.control_plane.watchdog import AgentHealth

health = watchdog.get_agent_health("claude-opus")

# Request metrics
print(f"Total requests: {health.total_requests}")
print(f"Failed requests: {health.failed_requests}")
print(f"Error rate: {health.error_rate:.1%}")
print(f"Avg latency: {health.average_latency_ms:.0f}ms")

# Health state
print(f"Last heartbeat: {health.last_heartbeat}")
print(f"Consecutive failures: {health.consecutive_failures}")
print(f"Memory usage: {health.memory_usage_mb:.1f}MB")
print(f"Circuit breaker: {health.circuit_breaker_state}")

# Active issues
for issue in health.active_issues:
    print(f"  - {issue.severity.name}: {issue.message}")
```

## ThreeTierWatchdog

The main watchdog class.

### Initialization

```python
from aragora.control_plane.watchdog import ThreeTierWatchdog

# Create with default configs
watchdog = ThreeTierWatchdog()

# Each tier starts with default configuration
# Customize as needed
watchdog.configure_tier(WatchdogConfig(
    tier=WatchdogTier.MECHANICAL,
    heartbeat_timeout_seconds=15.0,  # Stricter timeout
))
```

### Agent Registration

```python
# Register agents for monitoring
watchdog.register_agent("claude-opus")
watchdog.register_agent("gpt-4o")
watchdog.register_agent("gemini-pro")

# Unregister when agent removed
watchdog.unregister_agent("gemini-pro")
```

### Starting and Stopping

```python
# Start monitoring loops for all tiers
await watchdog.start()

# Watchdog runs check loops in background:
# - MECHANICAL: checks heartbeats, memory, circuits
# - BOOT_AGENT: checks latency, error rates
# - DEACON: checks SLAs, coordination

# Stop all monitoring
await watchdog.stop()
```

### Recording Metrics

```python
# Record heartbeat (call periodically from agent)
watchdog.record_heartbeat("claude-opus")

# Record request outcome
watchdog.record_request(
    "claude-opus",
    success=True,
    latency_ms=1234.5,
)

# Update memory usage
watchdog.update_memory_usage("claude-opus", memory_mb=512.0)

# Update circuit breaker state
watchdog.update_circuit_breaker("claude-opus", state="open")
# States: "closed", "open", "half-open"
```

### Handling Issues

```python
# Register handlers for specific tiers
def handle_mechanical_issue(issue: WatchdogIssue):
    if issue.category == IssueCategory.HEARTBEAT_MISSING:
        # Attempt to restart agent
        restart_agent(issue.agent)
    elif issue.category == IssueCategory.MEMORY_EXCEEDED:
        # Clear caches
        clear_agent_cache(issue.agent)

unregister = watchdog.register_handler(
    WatchdogTier.MECHANICAL,
    handle_mechanical_issue,
)

# Async handlers supported
async def handle_deacon_issue(issue: WatchdogIssue):
    if issue.category == IssueCategory.SLA_VIOLATION:
        await send_alert("SLA violation detected!")
        await scale_up_agents()

watchdog.register_handler(WatchdogTier.DEACON, handle_deacon_issue)

# Unregister when done
unregister()
```

### Manual Escalation

```python
# Manually escalate an issue
result = await watchdog.escalate(
    source_tier=WatchdogTier.MECHANICAL,
    issue=WatchdogIssue(
        severity=IssueSeverity.CRITICAL,
        category=IssueCategory.HEARTBEAT_MISSING,
        agent="claude-opus",
        message="Agent completely unresponsive",
    ),
)

print(f"Escalated to: {result.escalated_to}")
print(f"Accepted: {result.accepted}")
print(f"Action: {result.action_taken}")
```

### Resolving Issues

```python
# Get active issues
issues = watchdog.get_active_issues()

# Filter by severity
critical_issues = watchdog.get_active_issues(severity=IssueSeverity.CRITICAL)

# Filter by agent
agent_issues = watchdog.get_active_issues(agent="claude-opus")

# Resolve an issue
watchdog.resolve_issue(
    issue_id="issue-123456",
    notes="Agent restarted and responding normally",
)
```

### Statistics

```python
stats = watchdog.get_stats()

print(f"Issues detected: {stats['issues_detected']}")
print(f"Issues resolved: {stats['issues_resolved']}")
print(f"Escalations: {stats['escalations']}")
print(f"Active issues: {stats['active_issues']}")
print(f"Monitored agents: {stats['monitored_agents']}")
print(f"Running: {stats['is_running']}")

# Per-tier check counts
for tier, count in stats['tier_checks'].items():
    print(f"  {tier} checks: {count}")
```

## Global Watchdog

Access the singleton instance:

```python
from aragora.control_plane.watchdog import (
    get_watchdog,
    reset_watchdog,
)

# Get singleton
watchdog = get_watchdog()

# Reset for testing
reset_watchdog()
```

## Integration with Control Plane

The watchdog integrates with the Aragora control plane:

```python
from aragora.control_plane.coordinator import ControlPlaneCoordinator
from aragora.control_plane.watchdog import get_watchdog

# Control plane sets up watchdog automatically
coordinator = ControlPlaneCoordinator()

# Agents register with control plane
await coordinator.register_agent(agent)

# Watchdog monitors via control plane hooks
watchdog = get_watchdog()

# Control plane forwards heartbeats
@coordinator.on_heartbeat
def forward_heartbeat(agent_name: str):
    watchdog.record_heartbeat(agent_name)

# Control plane forwards request metrics
@coordinator.on_request_complete
def forward_request(agent_name: str, success: bool, latency_ms: float):
    watchdog.record_request(agent_name, success, latency_ms)
```

## Configuration Examples

### Strict Monitoring (Production)

```python
watchdog.configure_tier(WatchdogConfig(
    tier=WatchdogTier.MECHANICAL,
    check_interval_seconds=2.0,       # Frequent checks
    heartbeat_timeout_seconds=10.0,   # Quick detection
    auto_escalate=True,
    escalation_threshold=2,           # Escalate quickly
))

watchdog.configure_tier(WatchdogConfig(
    tier=WatchdogTier.BOOT_AGENT,
    check_interval_seconds=5.0,
    latency_warning_ms=2000.0,
    latency_critical_ms=5000.0,
    error_rate_warning=0.05,          # 5% warning
    error_rate_critical=0.15,         # 15% critical
))

watchdog.configure_tier(WatchdogConfig(
    tier=WatchdogTier.DEACON,
    check_interval_seconds=10.0,
    sla_availability_pct=99.9,        # High availability
    sla_response_time_ms=5000.0,
))
```

### Relaxed Monitoring (Development)

```python
watchdog.configure_tier(WatchdogConfig(
    tier=WatchdogTier.MECHANICAL,
    check_interval_seconds=30.0,      # Less frequent
    heartbeat_timeout_seconds=120.0,  # Allow longer gaps
    auto_escalate=False,              # Manual escalation
))

watchdog.configure_tier(WatchdogConfig(
    tier=WatchdogTier.BOOT_AGENT,
    check_interval_seconds=60.0,
    latency_warning_ms=30000.0,       # Allow slow responses
    error_rate_warning=0.3,           # Higher tolerance
))
```

## Environment Variables

Configure via environment:

```bash
# Enable watchdog
CP_ENABLE_WATCHDOG=true

# Heartbeat settings
CP_WATCHDOG_HEARTBEAT_TIMEOUT=30
CP_WATCHDOG_CHECK_INTERVAL=5

# Escalation
CP_WATCHDOG_AUTO_ESCALATE=true
CP_WATCHDOG_ESCALATION_THRESHOLD=3

# Memory thresholds (MB)
CP_WATCHDOG_MEMORY_WARNING=1024
CP_WATCHDOG_MEMORY_CRITICAL=2048

# Latency thresholds (ms)
CP_WATCHDOG_LATENCY_WARNING=5000
CP_WATCHDOG_LATENCY_CRITICAL=15000

# SLA targets
CP_WATCHDOG_SLA_AVAILABILITY=99.0
CP_WATCHDOG_SLA_RESPONSE_TIME=10000
```

## Best Practices

1. **Configure appropriate thresholds** - Adjust for your workload:
   ```python
   # Low-latency workload
   config = WatchdogConfig(
       tier=WatchdogTier.BOOT_AGENT,
       latency_warning_ms=500.0,
       latency_critical_ms=2000.0,
   )

   # Batch processing workload
   config = WatchdogConfig(
       tier=WatchdogTier.BOOT_AGENT,
       latency_warning_ms=30000.0,
       latency_critical_ms=120000.0,
   )
   ```

2. **Register handlers for all tiers**:
   ```python
   # Mechanical: restart/recover
   watchdog.register_handler(WatchdogTier.MECHANICAL, auto_recover)

   # Boot Agent: adjust routing
   watchdog.register_handler(WatchdogTier.BOOT_AGENT, adjust_routing)

   # Deacon: alert operations
   watchdog.register_handler(WatchdogTier.DEACON, alert_operations)
   ```

3. **Track heartbeats regularly**:
   ```python
   # In agent's main loop
   async def agent_loop(agent_name: str):
       watchdog = get_watchdog()
       while running:
           watchdog.record_heartbeat(agent_name)
           await process_work()
           await asyncio.sleep(1)
   ```

4. **Record all requests**:
   ```python
   async def execute_request(agent_name: str, request):
       start = time.time()
       try:
           result = await agent.execute(request)
           watchdog.record_request(agent_name, success=True,
               latency_ms=(time.time() - start) * 1000)
           return result
       except Exception as e:
           watchdog.record_request(agent_name, success=False,
               latency_ms=(time.time() - start) * 1000)
           raise
   ```

5. **Resolve issues properly**:
   ```python
   # Don't just clear issues - document resolution
   watchdog.resolve_issue(
       issue_id=issue.id,
       notes=f"Restarted agent, latency now {new_latency}ms",
   )
   ```

## Testing

```python
from aragora.control_plane.watchdog import (
    ThreeTierWatchdog,
    WatchdogConfig,
    WatchdogTier,
    IssueSeverity,
    IssueCategory,
    reset_watchdog,
)
import asyncio

def test_heartbeat_detection():
    """Missing heartbeat triggers issue."""
    watchdog = ThreeTierWatchdog()
    watchdog.configure_tier(WatchdogConfig(
        tier=WatchdogTier.MECHANICAL,
        heartbeat_timeout_seconds=1.0,  # Short timeout
        check_interval_seconds=0.5,
    ))

    watchdog.register_agent("test-agent")

    # Run check without heartbeat
    issues = asyncio.run(watchdog._check_mechanical(
        watchdog._configs[WatchdogTier.MECHANICAL]
    ))

    # Should not detect issue (no heartbeat recorded yet)
    assert len(issues) == 0

    # Record heartbeat then wait
    watchdog.record_heartbeat("test-agent")
    import time
    time.sleep(1.5)  # Exceed timeout

    issues = asyncio.run(watchdog._check_mechanical(
        watchdog._configs[WatchdogTier.MECHANICAL]
    ))

    # Should detect missing heartbeat
    assert len(issues) == 1
    assert issues[0].category == IssueCategory.HEARTBEAT_MISSING

def test_escalation():
    """Issues escalate to higher tiers."""
    watchdog = ThreeTierWatchdog()
    escalated_to = []

    def boot_handler(issue):
        escalated_to.append(WatchdogTier.BOOT_AGENT)

    watchdog.register_handler(WatchdogTier.BOOT_AGENT, boot_handler)

    issue = WatchdogIssue(
        severity=IssueSeverity.CRITICAL,
        category=IssueCategory.HEARTBEAT_MISSING,
        agent="test",
        message="Test issue",
    )

    result = asyncio.run(watchdog.escalate(
        WatchdogTier.MECHANICAL, issue
    ))

    assert result.accepted
    assert result.escalated_to == WatchdogTier.BOOT_AGENT
    assert WatchdogTier.BOOT_AGENT in escalated_to

def test_error_rate_tracking():
    """Error rate calculated correctly."""
    watchdog = ThreeTierWatchdog()
    watchdog.register_agent("test-agent")

    # Record mixed results
    for _ in range(7):
        watchdog.record_request("test-agent", success=True, latency_ms=100)
    for _ in range(3):
        watchdog.record_request("test-agent", success=False, latency_ms=100)

    health = watchdog.get_agent_health("test-agent")
    assert health.error_rate == 0.3  # 30%
```

## See Also

- [Control Plane](CONTROL_PLANE.md) - Agent registry and coordination
- [Resilience](RESILIENCE.md) - Circuit breaker patterns
- [Propulsion Engine](PROPULSION.md) - Push-based work distribution
