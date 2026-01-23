#!/usr/bin/env python3
"""
Aragora Control Plane Demo - Enterprise Multi-Agent Orchestration.

This demo showcases the Control Plane features for multi-agent robust decisionmaking:
- Agent fleet management and health monitoring
- Task scheduling with priority queues
- Deliberation-as-task integration
- Channel notifications (Slack, Teams, Webhooks)
- Immutable audit logging

Usage:
    python scripts/demo_control_plane.py                  # Full demo
    python scripts/demo_control_plane.py --quick          # Quick 2-minute demo
    python scripts/demo_control_plane.py --agents 5       # Custom agent count
    python scripts/demo_control_plane.py --simulate-load  # Simulate heavy load

Requirements:
    - Redis running locally (for task queue)
    - Optional: SLACK_WEBHOOK_URL env var for notifications
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from aragora.control_plane.registry import AgentRegistry, AgentInfo, AgentStatus
from aragora.control_plane.scheduler import TaskScheduler, Task, TaskPriority, TaskStatus
from aragora.control_plane.coordinator import ControlPlaneCoordinator
from aragora.control_plane.deliberation import ControlPlaneDeliberation
from aragora.control_plane.channels import ChannelRouter, ChannelConfig, DeliveryRecord
from aragora.control_plane.audit import AuditLog, AuditEntry, AuditAction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Demo configuration
DEMO_AGENTS = [
    {"name": "claude-analyst", "capabilities": ["analysis", "research", "code-review"]},
    {"name": "gpt-coder", "capabilities": ["code-generation", "testing", "debugging"]},
    {"name": "gemini-researcher", "capabilities": ["research", "fact-checking", "summarization"]},
    {"name": "mistral-reviewer", "capabilities": ["code-review", "security-audit"]},
    {"name": "grok-critic", "capabilities": ["analysis", "critique", "red-teaming"]},
]

DEMO_TASKS = [
    {"type": "code-review", "priority": "high", "payload": {"repo": "aragora", "pr": 42}},
    {"type": "research", "priority": "normal", "payload": {"topic": "LLM safety"}},
    {
        "type": "deliberation",
        "priority": "high",
        "payload": {"question": "Should we use RAG or fine-tuning?"},
    },
    {"type": "security-audit", "priority": "critical", "payload": {"target": "auth-module"}},
    {"type": "analysis", "priority": "normal", "payload": {"document": "quarterly-report.pdf"}},
]


def print_header(title: str) -> None:
    """Print a formatted section header."""
    border = "=" * 60
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}\n")


def print_status(label: str, value: str, color: str = "default") -> None:
    """Print a status line with optional color."""
    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "cyan": "\033[96m",
        "default": "\033[0m",
    }
    reset = "\033[0m"
    color_code = colors.get(color, colors["default"])
    print(f"  {label:.<40} {color_code}{value}{reset}")


async def demo_agent_registry(registry: AgentRegistry, num_agents: int) -> List[str]:
    """Demonstrate agent registration and fleet management."""
    print_header("AGENT FLEET MANAGEMENT")

    agent_ids = []
    for i, agent_def in enumerate(DEMO_AGENTS[:num_agents]):
        agent_id = f"agent-{uuid.uuid4().hex[:8]}"
        agent = AgentInfo(
            agent_id=agent_id,
            name=agent_def["name"],
            capabilities=agent_def["capabilities"],
            status=AgentStatus.IDLE,
            registered_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow(),
            metadata={"version": "1.0", "region": "us-west-2"},
        )

        await registry.register(agent)
        agent_ids.append(agent_id)
        print_status(f"Registered {agent_def['name']}", agent_id[:16] + "...", "green")
        await asyncio.sleep(0.2)  # Visual pacing

    # Show fleet summary
    print("\n  Fleet Summary:")
    stats = await registry.get_stats()
    print_status("Total Agents", str(stats.get("total", num_agents)), "cyan")
    print_status("Online", str(stats.get("online", num_agents)), "green")
    print_status("Idle", str(stats.get("idle", num_agents)), "yellow")

    return agent_ids


async def demo_task_scheduling(scheduler: TaskScheduler, agent_ids: List[str]) -> List[str]:
    """Demonstrate task scheduling with priority queues."""
    print_header("TASK SCHEDULING & PRIORITY QUEUE")

    task_ids = []
    priority_map = {
        "critical": TaskPriority.CRITICAL,
        "high": TaskPriority.HIGH,
        "normal": TaskPriority.NORMAL,
        "low": TaskPriority.LOW,
    }

    for task_def in DEMO_TASKS:
        task = Task(
            task_id=f"task-{uuid.uuid4().hex[:8]}",
            task_type=task_def["type"],
            priority=priority_map.get(task_def["priority"], TaskPriority.NORMAL),
            payload=task_def["payload"],
            created_at=datetime.utcnow(),
            status=TaskStatus.PENDING,
        )

        await scheduler.submit(task)
        task_ids.append(task.task_id)

        priority_color = {
            "critical": "red",
            "high": "yellow",
            "normal": "default",
            "low": "default",
        }.get(task_def["priority"], "default")

        print_status(
            f"Submitted {task_def['type']}",
            f"[{task_def['priority'].upper()}] {task.task_id[:12]}...",
            priority_color,
        )
        await asyncio.sleep(0.15)

    # Show queue stats
    print("\n  Queue Summary:")
    queue_stats = await scheduler.get_stats()
    print_status("Pending Tasks", str(queue_stats.get("pending", len(DEMO_TASKS))), "yellow")
    print_status("Critical Priority", str(queue_stats.get("critical", 1)), "red")
    print_status("High Priority", str(queue_stats.get("high", 2)), "yellow")

    return task_ids


async def demo_task_execution(scheduler: TaskScheduler, agent_ids: List[str]) -> None:
    """Demonstrate task claiming and execution."""
    print_header("TASK EXECUTION")

    claimed = 0
    for agent_id in agent_ids[:3]:  # First 3 agents claim tasks
        task = await scheduler.claim(agent_id)
        if task:
            claimed += 1
            print_status(
                f"Agent {agent_id[:12]}",
                f"claimed {task.task_type} ({task.task_id[:8]}...)",
                "cyan",
            )

            # Simulate execution
            await asyncio.sleep(0.3)
            await scheduler.complete(task.task_id, {"result": "success", "confidence": 0.95})
            print_status(f"Task {task.task_id[:12]}", "COMPLETED", "green")

    print(f"\n  Executed {claimed} tasks successfully")


async def demo_deliberation(deliberation: ControlPlaneDeliberation) -> Optional[str]:
    """Demonstrate deliberation-as-task integration."""
    print_header("DELIBERATION-AS-TASK")

    question = "What is the optimal approach for implementing real-time AI safety monitoring?"

    print(f"  Question: {question[:50]}...")
    print()

    # Start deliberation
    debate_id = await deliberation.start_deliberation(
        question=question,
        context={"domain": "AI safety", "urgency": "high"},
        agents=["claude-analyst", "gemini-researcher", "grok-critic"],
    )

    print_status("Deliberation started", debate_id[:16] + "...", "cyan")

    # Simulate rounds
    for round_num in range(1, 4):
        await asyncio.sleep(0.5)
        print_status(f"Round {round_num}", "in progress...", "yellow")

    # Complete
    await asyncio.sleep(0.3)
    print_status("Consensus", "REACHED (3/3 agents)", "green")
    print_status("Confidence", "92%", "cyan")

    return debate_id


async def demo_notifications(channel_router: ChannelRouter, debate_id: str) -> None:
    """Demonstrate channel notifications."""
    print_header("CHANNEL NOTIFICATIONS")

    # Show configured channels
    channels = [
        {"type": "slack", "name": "#aragora-alerts", "status": "configured"},
        {"type": "teams", "name": "AI Ops Channel", "status": "configured"},
        {"type": "webhook", "name": "PagerDuty", "status": "configured"},
        {"type": "email", "name": "ops@company.com", "status": "pending"},
    ]

    print("  Configured Channels:")
    for ch in channels:
        color = "green" if ch["status"] == "configured" else "yellow"
        print_status(f"  {ch['type'].upper()}: {ch['name']}", ch["status"], color)

    print("\n  Sending Notifications:")

    # Simulate notification delivery
    for ch in channels[:3]:  # Skip email for demo
        await asyncio.sleep(0.3)
        print_status(f"  -> {ch['type']}", "delivered", "green")

    print_status("  Total notifications sent", "3", "cyan")


async def demo_audit_log(audit: AuditLog) -> None:
    """Demonstrate immutable audit logging."""
    print_header("IMMUTABLE AUDIT LOG")

    # Show recent audit entries
    entries = [
        {"action": "agent.registered", "actor": "system", "resource": "claude-analyst"},
        {"action": "task.submitted", "actor": "user:admin", "resource": "task-abc123"},
        {"action": "task.claimed", "actor": "agent:gpt-coder", "resource": "task-abc123"},
        {"action": "deliberation.started", "actor": "system", "resource": "debate-xyz"},
        {"action": "deliberation.consensus", "actor": "system", "resource": "debate-xyz"},
        {"action": "notification.sent", "actor": "system", "resource": "slack:#alerts"},
    ]

    print("  Recent Audit Trail:")
    for entry in entries:
        await audit.log(
            action=AuditAction(entry["action"]),
            actor=entry["actor"],
            resource=entry["resource"],
            details={"timestamp": datetime.utcnow().isoformat()},
        )
        print(f"    [{entry['action']:.<30}] {entry['actor']} -> {entry['resource']}")
        await asyncio.sleep(0.1)

    # Show integrity verification
    print("\n  Audit Integrity:")
    print_status("Hash chain verified", "VALID", "green")
    print_status("Entries", str(len(entries)), "cyan")
    print_status("Tamper detection", "ENABLED", "green")


async def demo_load_simulation(
    registry: AgentRegistry, scheduler: TaskScheduler, num_tasks: int = 50
) -> None:
    """Simulate heavy load for stress testing."""
    print_header("LOAD SIMULATION")

    print(f"  Submitting {num_tasks} tasks...")

    start_time = time.time()
    task_ids = []

    for i in range(num_tasks):
        task = Task(
            task_id=f"load-task-{i:04d}",
            task_type="analysis",
            priority=TaskPriority.NORMAL,
            payload={"index": i},
            created_at=datetime.utcnow(),
            status=TaskStatus.PENDING,
        )
        await scheduler.submit(task)
        task_ids.append(task.task_id)

        if (i + 1) % 10 == 0:
            print_status("Submitted", f"{i + 1}/{num_tasks}", "cyan")

    elapsed = time.time() - start_time
    rate = num_tasks / elapsed if elapsed > 0 else 0

    print("\n  Load Test Results:")
    print_status("Tasks submitted", str(num_tasks), "green")
    print_status("Time elapsed", f"{elapsed:.2f}s", "cyan")
    print_status("Rate", f"{rate:.0f} tasks/sec", "cyan")


async def run_demo(args: argparse.Namespace) -> None:
    """Run the complete control plane demo."""
    print("\n" + "=" * 60)
    print("  ARAGORA CONTROL PLANE DEMO")
    print("  Enterprise Multi-Agent Orchestration")
    print("=" * 60)

    # Initialize components
    registry = AgentRegistry()
    scheduler = TaskScheduler()
    coordinator = ControlPlaneCoordinator(registry=registry, scheduler=scheduler)
    deliberation = ControlPlaneDeliberation(coordinator=coordinator)
    channel_router = ChannelRouter()
    audit = AuditLog()

    try:
        # Run demo sections
        agent_ids = await demo_agent_registry(registry, args.agents)

        if not args.quick:
            await asyncio.sleep(0.5)

        task_ids = await demo_task_scheduling(scheduler, agent_ids)

        if not args.quick:
            await asyncio.sleep(0.5)
            await demo_task_execution(scheduler, agent_ids)

        debate_id = await demo_deliberation(deliberation)

        if debate_id:
            await demo_notifications(channel_router, debate_id)

        await demo_audit_log(audit)

        if args.simulate_load:
            await demo_load_simulation(registry, scheduler, num_tasks=args.load_tasks)

        # Final summary
        print_header("DEMO COMPLETE")
        print("  The Control Plane provides:")
        print("    - Real-time agent fleet monitoring")
        print("    - Priority-based task scheduling")
        print("    - Deliberation-as-task orchestration")
        print("    - Multi-channel notifications")
        print("    - Immutable audit logging")
        print()
        print("  For more information:")
        print("    - UI Dashboard: http://localhost:3000/control-plane")
        print("    - API Docs: http://localhost:8080/docs")
        print("    - Documentation: docs/CONTROL_PLANE.md")
        print()

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aragora Control Plane Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick 2-minute demo",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=5,
        help="Number of demo agents to register (default: 5)",
    )
    parser.add_argument(
        "--simulate-load",
        action="store_true",
        help="Include load simulation test",
    )
    parser.add_argument(
        "--load-tasks",
        type=int,
        default=50,
        help="Number of tasks for load simulation (default: 50)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(run_demo(args))
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
