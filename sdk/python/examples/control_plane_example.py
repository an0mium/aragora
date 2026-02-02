"""
Control Plane Example

Demonstrates enterprise control plane features including agent registry,
task scheduling, health monitoring, and policy management.

Usage:
    python examples/control_plane_example.py

Environment:
    ARAGORA_API_KEY - Your API key
    ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

from aragora_sdk import AragoraAsyncClient

# =============================================================================
# Agent Registry
# =============================================================================


async def agent_registry(client: AragoraAsyncClient) -> None:
    """Demonstrate agent registry operations."""
    print("=== Agent Registry ===\n")

    # List registered agents
    print("Registered agents:")
    agents = await client.control_plane.list_agents()

    for agent in agents.get("agents", [])[:5]:
        name = agent.get("name", "Unknown")
        status = agent.get("status", "unknown")
        last_heartbeat = agent.get("last_heartbeat", "N/A")
        capabilities = agent.get("capabilities", [])

        status_icon = {
            "healthy": "[OK]",
            "degraded": "[!]",
            "unhealthy": "[X]",
            "unknown": "[?]",
        }.get(status, "[?]")

        print(f"  {status_icon} {name:<15} Last heartbeat: {last_heartbeat}")
        if capabilities:
            print(f"      Capabilities: {', '.join(capabilities[:3])}")

    # Register a custom agent (example)
    print("\nRegistering custom agent...")
    print("""
    To register a custom agent:

    registration = await client.control_plane.register_agent(
        name="my-custom-agent",
        endpoint="https://my-agent.example.com/api",
        capabilities=["text-generation", "code-review"],
        metadata={
            "version": "1.0.0",
            "provider": "custom",
        },
    )
    """)


# =============================================================================
# Health Monitoring
# =============================================================================


async def health_monitoring(client: AragoraAsyncClient) -> None:
    """Demonstrate health monitoring."""
    print("\n=== Health Monitoring ===\n")

    # Get system health
    health = await client.control_plane.get_health()

    print(f"System Status: {health.get('status', 'unknown').upper()}")
    print(f"Timestamp: {health.get('timestamp', 'N/A')}")

    # Component health
    components = health.get("components", {})
    print("\nComponent Health:")
    for component, status in components.items():
        status_icon = {"healthy": "[OK]", "degraded": "[!]", "unhealthy": "[X]"}.get(
            status.get("status", "unknown"), "[?]"
        )
        print(f"  {status_icon} {component}: {status.get('status', 'unknown')}")
        if status.get("latency_ms"):
            print(f"      Latency: {status['latency_ms']}ms")

    # Agent health
    print("\nAgent Health:")
    agent_health = await client.control_plane.get_agent_health()

    for agent in agent_health.get("agents", [])[:5]:
        name = agent.get("name", "Unknown")
        status = agent.get("health_status", "unknown")
        success_rate = agent.get("success_rate", 0)
        avg_latency = agent.get("avg_latency_ms", 0)

        print(f"  {name}:")
        print(f"    Status: {status}")
        print(f"    Success rate: {success_rate:.1%}")
        print(f"    Avg latency: {avg_latency}ms")


# =============================================================================
# Task Scheduling
# =============================================================================


async def task_scheduling(client: AragoraAsyncClient) -> None:
    """Demonstrate task scheduling."""
    print("\n=== Task Scheduling ===\n")

    # List scheduled tasks
    print("Scheduled tasks:")
    tasks = await client.control_plane.list_scheduled_tasks()

    for task in tasks.get("tasks", [])[:5]:
        task_id = task.get("id", "Unknown")
        task_type = task.get("type", "unknown")
        status = task.get("status", "unknown")
        next_run = task.get("next_run", "N/A")

        print(f"  [{status}] {task_id}")
        print(f"      Type: {task_type}")
        print(f"      Next run: {next_run}")

    # Schedule a new task
    print("\nScheduling a new task...")
    print("""
    To schedule a recurring debate:

    task = await client.control_plane.schedule_task(
        task_type="recurring_debate",
        schedule="0 9 * * MON",  # Every Monday at 9 AM (cron)
        config={
            "task": "Weekly team retrospective - what went well?",
            "agents": ["claude", "gpt-4"],
            "rounds": 3,
        },
        metadata={
            "owner": "team-leads",
            "priority": "normal",
        },
    )
    """)

    # Priority queue
    print("\nTask Priority Queue:")
    queue = await client.control_plane.get_task_queue()

    for i, item in enumerate(queue.get("queue", [])[:5], 1):
        task_id = item.get("task_id", "Unknown")
        priority = item.get("priority", 0)
        wait_time = item.get("wait_time_seconds", 0)

        print(f"  {i}. [{priority}] {task_id} (waiting {wait_time}s)")


# =============================================================================
# Policy Management
# =============================================================================


async def policy_management(client: AragoraAsyncClient) -> None:
    """Demonstrate policy management."""
    print("\n=== Policy Management ===\n")

    # List policies
    print("Active policies:")
    policies = await client.control_plane.list_policies()

    for policy in policies.get("policies", [])[:5]:
        name = policy.get("name", "Unknown")
        policy_type = policy.get("type", "unknown")
        enabled = policy.get("enabled", False)
        scope = policy.get("scope", "global")

        status = "[ON]" if enabled else "[OFF]"
        print(f"  {status} {name}")
        print(f"      Type: {policy_type}")
        print(f"      Scope: {scope}")

    # Example policy creation
    print("\nPolicy types available:")
    print("  - rate_limit: Control request rates")
    print("  - agent_selection: Constrain which agents can be used")
    print("  - content_filter: Filter debate content")
    print("  - cost_limit: Budget constraints")
    print("  - retention: Data retention rules")

    print("""
    To create a rate limit policy:

    policy = await client.control_plane.create_policy(
        name="api-rate-limit",
        type="rate_limit",
        config={
            "requests_per_minute": 60,
            "burst_limit": 10,
            "scope": "per_user",
        },
        enabled=True,
    )
    """)


# =============================================================================
# Resource Quotas
# =============================================================================


async def resource_quotas(client: AragoraAsyncClient) -> None:
    """Demonstrate resource quota management."""
    print("\n=== Resource Quotas ===\n")

    # Get current usage
    usage = await client.control_plane.get_usage()

    print("Current Usage:")
    print(f"  Debates today: {usage.get('debates_today', 0)}")
    print(f"  API calls today: {usage.get('api_calls_today', 0)}")
    print(f"  Tokens used today: {usage.get('tokens_today', 0):,}")

    # Get quotas
    quotas = await client.control_plane.get_quotas()

    print("\nQuota Limits:")
    for quota_name, quota_info in quotas.get("quotas", {}).items():
        limit = quota_info.get("limit", "unlimited")
        used = quota_info.get("used", 0)

        if limit == "unlimited":
            print(f"  {quota_name}: {used} used (unlimited)")
        else:
            pct = (used / limit) * 100 if limit > 0 else 0
            bar = "#" * int(pct / 10) + "-" * (10 - int(pct / 10))
            print(f"  {quota_name}: [{bar}] {used}/{limit} ({pct:.0f}%)")

    # Set quota alert
    print("""
    To set a quota alert:

    alert = await client.control_plane.create_quota_alert(
        quota="debates_daily",
        threshold=0.8,  # Alert at 80% usage
        channels=["email", "slack"],
    )
    """)


# =============================================================================
# Audit Trail
# =============================================================================


async def audit_trail(client: AragoraAsyncClient) -> None:
    """Demonstrate audit trail access."""
    print("\n=== Audit Trail ===\n")

    # Get recent audit events
    audit_events = await client.control_plane.get_audit_trail(
        limit=10,
        since=datetime.now(timezone.utc) - timedelta(hours=24),
    )

    print("Recent audit events (last 24h):")
    for event in audit_events.get("events", [])[:5]:
        timestamp = event.get("timestamp", "N/A")
        action = event.get("action", "unknown")
        actor = event.get("actor", "unknown")
        resource = event.get("resource", "N/A")
        outcome = event.get("outcome", "unknown")

        outcome_icon = {"success": "[OK]", "failure": "[X]", "denied": "[!]"}.get(outcome, "[?]")

        print(f"  {timestamp}")
        print(f"    {outcome_icon} {action} by {actor}")
        print(f"       Resource: {resource}")

    # Export audit log
    print("""
    To export audit log:

    export = await client.control_plane.export_audit_trail(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 31),
        format="csv",  # or "json", "parquet"
    )
    download_url = export["download_url"]
    """)


# =============================================================================
# Notifications
# =============================================================================


async def notifications(client: AragoraAsyncClient) -> None:
    """Demonstrate notification configuration."""
    print("\n=== Notifications ===\n")

    # List notification channels
    channels = await client.control_plane.list_notification_channels()

    print("Configured notification channels:")
    for channel in channels.get("channels", []):
        name = channel.get("name", "Unknown")
        channel_type = channel.get("type", "unknown")
        enabled = channel.get("enabled", False)

        status = "[ON]" if enabled else "[OFF]"
        print(f"  {status} {name} ({channel_type})")

    print("""
    To configure Slack notifications:

    channel = await client.control_plane.create_notification_channel(
        name="team-alerts",
        type="slack",
        config={
            "webhook_url": "https://hooks.slack.com/...",
            "channel": "#aragora-alerts",
        },
        events=["debate_completed", "quota_warning", "agent_unhealthy"],
    )
    """)


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run control plane demonstrations."""
    print("Aragora SDK Control Plane Example")
    print("=" * 60)

    # Check if we should run actual examples
    run_examples = os.environ.get("RUN_EXAMPLES", "false").lower() == "true"

    if not run_examples:
        print("\nControl plane features:")
        print("  1. Agent Registry: Register and manage agents")
        print("  2. Health Monitoring: System and agent health")
        print("  3. Task Scheduling: Schedule recurring debates")
        print("  4. Policy Management: Rate limits, content filters")
        print("  5. Resource Quotas: Usage limits and alerts")
        print("  6. Audit Trail: Activity logging and export")
        print("  7. Notifications: Slack, email, webhook alerts")
        print("\nSet RUN_EXAMPLES=true to run actual API examples.")
        return

    async with AragoraAsyncClient(
        base_url=os.environ.get("ARAGORA_API_URL", "https://api.aragora.ai"),
        api_key=os.environ.get("ARAGORA_API_KEY"),
    ) as client:
        # Agent registry
        await agent_registry(client)

        # Health monitoring
        await health_monitoring(client)

        # Task scheduling
        await task_scheduling(client)

        # Policy management
        await policy_management(client)

        # Resource quotas
        await resource_quotas(client)

        # Audit trail
        await audit_trail(client)

        # Notifications
        await notifications(client)

    print("\n" + "=" * 60)
    print("Control plane example complete!")
    print("\nEnterprise Features:")
    print("  - Centralized agent management")
    print("  - Proactive health monitoring")
    print("  - Flexible task scheduling")
    print("  - Fine-grained policy control")
    print("  - Comprehensive audit logging")


if __name__ == "__main__":
    asyncio.run(main())
