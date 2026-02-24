#!/usr/bin/env python3
"""OpenClaw Gateway End-to-End Demo.

Demonstrates the full OpenClaw enterprise gateway flow:
1. Create a governance session with configuration
2. Add policy rules (require approval for shell commands)
3. Execute a safe action (file read - auto-approved)
4. Execute a risky action (shell command - triggers approval)
5. List pending approvals and approve the action
6. Query audit trail for compliance
7. Export decision receipt

Usage:
    python examples/openclaw_gateway.py --demo    # Mock mode (no server needed)
    python examples/openclaw_gateway.py           # Live mode (requires server)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


# =============================================================================
# Mock Client (for --demo mode)
# =============================================================================


@dataclass
class MockSession:
    session_id: str = "sess_demo_001"
    user_id: str = "user_demo"
    tenant_id: str = "tenant_acme"
    workspace_id: str = "ws_default"
    roles: list[str] = field(default_factory=lambda: ["operator", "analyst"])
    status: str = "active"
    action_count: int = 0


@dataclass
class MockActionResult:
    success: bool = True
    action_id: str = ""
    decision: str = "allow"
    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    requires_approval: bool = False
    approval_id: str | None = None


@dataclass
class MockApproval:
    approval_id: str = ""
    action_type: str = ""
    description: str = ""
    status: str = "pending"
    requested_by: str = "user_demo"
    requested_at: str = ""


class MockOpenClawClient:
    """In-memory mock of the OpenClaw gateway API."""

    def __init__(self) -> None:
        self._sessions: dict[str, MockSession] = {}
        self._policies: list[dict[str, Any]] = []
        self._approvals: list[MockApproval] = []
        self._audit_log: list[dict[str, Any]] = []
        self._action_counter = 0

    def create_session(self, user_id: str, roles: list[str] | None = None) -> MockSession:
        session = MockSession(
            session_id=f"sess_{int(time.time())}",
            user_id=user_id,
            roles=roles or ["operator"],
        )
        self._sessions[session.session_id] = session
        self._audit_log.append(
            {
                "event": "session_created",
                "session_id": session.session_id,
                "user_id": user_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        return session

    def add_policy_rule(self, rule: dict[str, Any]) -> dict[str, Any]:
        self._policies.append(rule)
        self._audit_log.append(
            {
                "event": "policy_added",
                "rule": rule,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        return {"status": "added", "rule": rule}

    def execute_action(
        self, session_id: str, action_type: str, params: dict[str, Any]
    ) -> MockActionResult:
        self._action_counter += 1
        action_id = f"act_{self._action_counter:04d}"

        # Check policies
        requires_approval = False
        for policy in self._policies:
            if (
                policy.get("action_type") == action_type
                and policy.get("decision") == "require_approval"
            ):
                requires_approval = True
                break

        if requires_approval:
            approval = MockApproval(
                approval_id=f"apr_{self._action_counter:04d}",
                action_type=action_type,
                description=f"{action_type}: {json.dumps(params)}",
                requested_at=datetime.now(timezone.utc).isoformat(),
            )
            self._approvals.append(approval)
            self._audit_log.append(
                {
                    "event": "action_requires_approval",
                    "action_id": action_id,
                    "action_type": action_type,
                    "approval_id": approval.approval_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            return MockActionResult(
                success=False,
                action_id=action_id,
                decision="require_approval",
                requires_approval=True,
                approval_id=approval.approval_id,
            )

        # Auto-approved action
        result = {"output": f"Executed {action_type} successfully", "params": params}
        self._audit_log.append(
            {
                "event": "action_executed",
                "action_id": action_id,
                "action_type": action_type,
                "decision": "allow",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        session = self._sessions.get(session_id)
        if session:
            session.action_count += 1

        return MockActionResult(
            success=True,
            action_id=action_id,
            decision="allow",
            result=result,
            execution_time_ms=12.5,
        )

    def list_approvals(self, status: str = "pending") -> list[MockApproval]:
        return [a for a in self._approvals if a.status == status]

    def approve_action(self, approval_id: str, approver: str = "admin") -> MockActionResult:
        for approval in self._approvals:
            if approval.approval_id == approval_id:
                approval.status = "approved"
                self._audit_log.append(
                    {
                        "event": "action_approved",
                        "approval_id": approval_id,
                        "approver": approver,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                return MockActionResult(
                    success=True,
                    action_id=f"act_approved_{approval_id}",
                    decision="allow",
                    result={"output": "Shell command executed after approval"},
                    execution_time_ms=45.2,
                )
        return MockActionResult(success=False, action_id="", decision="deny", error="Not found")

    def get_audit_trail(self, session_id: str | None = None) -> list[dict[str, Any]]:
        if session_id:
            return [e for e in self._audit_log if e.get("session_id") == session_id]
        return list(self._audit_log)

    def end_session(self, session_id: str) -> dict[str, Any]:
        session = self._sessions.get(session_id)
        if session:
            session.status = "ended"
            self._audit_log.append(
                {
                    "event": "session_ended",
                    "session_id": session_id,
                    "action_count": session.action_count,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        return {"status": "ended", "session_id": session_id}


# =============================================================================
# Demo Runner
# =============================================================================


def _print_step(num: int, title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  Step {num}: {title}")
    print(f"{'─' * 60}")


def _print_json(data: Any) -> None:
    if hasattr(data, "__dataclass_fields__"):
        data = asdict(data)
    print(json.dumps(data, indent=2, default=str))


def run_demo() -> None:
    """Run the full OpenClaw gateway demo with mock client."""
    client = MockOpenClawClient()

    print("=" * 60)
    print("  OpenClaw Enterprise Gateway Demo")
    print("  Policy-gated autonomous execution with audit trails")
    print("=" * 60)

    # Step 1: Create session
    _print_step(1, "Create Governance Session")
    session = client.create_session(
        user_id="alice@acme.com",
        roles=["operator", "analyst"],
    )
    print(f"  Session ID: {session.session_id}")
    print(f"  User: {session.user_id}")
    print(f"  Roles: {', '.join(session.roles)}")
    print(f"  Status: {session.status}")

    # Step 2: Add policy rules
    _print_step(2, "Configure Policy Rules")

    # Safe actions auto-approved
    rule1 = client.add_policy_rule(
        {
            "action_type": "file_read",
            "decision": "allow",
            "description": "File reads are auto-approved for analysts",
        }
    )
    print(f"  Rule 1 (file_read -> allow): {rule1['status']}")

    # Shell commands require approval
    rule2 = client.add_policy_rule(
        {
            "action_type": "shell_command",
            "decision": "require_approval",
            "description": "Shell commands require human approval",
        }
    )
    print(f"  Rule 2 (shell_command -> require_approval): {rule2['status']}")

    # Step 3: Execute safe action
    _print_step(3, "Execute Safe Action (file read)")
    safe_result = client.execute_action(
        session_id=session.session_id,
        action_type="file_read",
        params={"path": "/data/quarterly_report.csv"},
    )
    print(f"  Decision: {safe_result.decision}")
    print(f"  Success: {safe_result.success}")
    print(f"  Time: {safe_result.execution_time_ms}ms")
    if safe_result.result:
        print(f"  Result: {safe_result.result}")

    # Step 4: Execute risky action
    _print_step(4, "Execute Risky Action (shell command)")
    risky_result = client.execute_action(
        session_id=session.session_id,
        action_type="shell_command",
        params={"command": "kubectl scale deployment api --replicas=5"},
    )
    print(f"  Decision: {risky_result.decision}")
    print(f"  Requires approval: {risky_result.requires_approval}")
    print(f"  Approval ID: {risky_result.approval_id}")

    # Step 5: Approve the action
    _print_step(5, "Review and Approve Pending Action")
    pending = client.list_approvals(status="pending")
    print(f"  Pending approvals: {len(pending)}")
    for approval in pending:
        print(f"    - {approval.approval_id}: {approval.description}")

    if pending:
        approved = client.approve_action(
            approval_id=pending[0].approval_id,
            approver="admin@acme.com",
        )
        print(f"\n  Approved: {approved.success}")
        print(f"  Result: {approved.result}")

    # Step 6: Query audit trail
    _print_step(6, "Query Audit Trail")
    audit = client.get_audit_trail()
    print(f"  Total audit events: {len(audit)}")
    for event in audit:
        print(f"    [{event['timestamp'][:19]}] {event['event']}")

    # Step 7: Export receipt
    _print_step(7, "Export Decision Receipt")
    receipt = {
        "type": "openclaw_gateway_receipt",
        "session_id": session.session_id,
        "user": session.user_id,
        "actions_executed": session.action_count,
        "policies_applied": len(client._policies),
        "approvals_processed": len([a for a in client._approvals if a.status == "approved"]),
        "audit_events": len(audit),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _print_json(receipt)

    # End session
    client.end_session(session.session_id)

    print(f"\n{'=' * 60}")
    print("  Demo complete! Session ended.")
    print(f"{'=' * 60}")
    print("\nTo run with a live server:")
    print("  1. Start: aragora serve")
    print("  2. Create session: aragora openclaw session create")
    print("  3. Execute actions: aragora openclaw action execute")


def run_live() -> None:
    """Run with live OpenClaw API (requires running server)."""
    try:
        from aragora.client import AragoraClient  # noqa: F401 — availability check
    except ImportError:
        print("Error: aragora package required for live mode.")
        print("Use --demo for mock mode, or install aragora.")
        sys.exit(1)

    print("Live mode requires a running Aragora server.")
    print("Start one with: aragora serve")
    print("\nFalling back to demo mode...\n")
    run_demo()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenClaw Gateway End-to-End Demo",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=True,
        help="Use mock client (default: True)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live API (requires running server)",
    )
    args = parser.parse_args()

    if args.live:
        run_live()
    else:
        run_demo()


if __name__ == "__main__":
    main()
