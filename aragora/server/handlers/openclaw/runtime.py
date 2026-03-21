"""
Real OpenClaw runtime dispatch for the public gateway handler path.

This replaces the store-and-mark-running stub with:
- policy evaluation before execution
- pending approval records when execution must wait
- local sandbox-backed execution for supported actions
- bounded failure reasons for unsupported or rejected actions
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import shlex
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aragora.gateway.openclaw_policy import (
    ActionRequest,
    ActionType,
    OpenClawPolicy,
    PolicyDecision,
    PolicyRule,
    create_enterprise_policy,
)
from aragora.gateway.openclaw_sandbox import (
    OpenClawActionSandbox,
    SandboxActionResult,
    SandboxConfig,
)
from aragora.server.handlers.openclaw.models import Action, ActionStatus, Session

logger = logging.getLogger(__name__)

_FAILURE_REASON_LIMIT = 240


def _run_coro(coro: Any) -> Any:
    """Run an async coroutine from the sync HTTP handler context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result(timeout=120)
    return asyncio.run(coro)


def _normalize_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _bounded_failure_reason(code: str, detail: str | None = None) -> str:
    message = code if not detail else f"{code}: {detail}"
    compact = " ".join(str(message).split())
    if len(compact) <= _FAILURE_REASON_LIMIT:
        return compact
    return compact[: _FAILURE_REASON_LIMIT - 3].rstrip() + "..."


@dataclass
class NormalizedAction:
    """Executable action normalized for policy + runtime dispatch."""

    action_type: ActionType
    path: str | None = None
    command: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeApproval:
    """Pending approval for a not-yet-executed OpenClaw action."""

    approval_id: str
    action_id: str
    session_id: str
    user_id: str
    tenant_id: str | None
    action_type: str
    normalized_action_type: str
    action_data: dict[str, Any]
    metadata: dict[str, Any]
    status: str = "pending"
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    decided_at: datetime | None = None
    decided_by: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for handler responses."""
        return {
            "id": self.approval_id,
            "action_id": self.action_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "status": self.status,
            "action_type": self.action_type,
            "action_data": self.action_data,
            "requested_at": self.requested_at.isoformat(),
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "decided_by": self.decided_by,
            "reason": self.reason,
        }


@dataclass
class RuntimeDispatchResult:
    """Outcome of dispatching one stored OpenClaw action."""

    action_id: str
    status: ActionStatus
    executed: bool = False
    output_data: dict[str, Any] | None = None
    error: str | None = None
    approval_id: str | None = None
    execution_time_ms: int = 0
    audit_result: str = "success"
    audit_details: dict[str, Any] = field(default_factory=dict)


class OpenClawExecutionRuntime:
    """Policy-gated runtime backed by the local OpenClaw action sandbox."""

    def __init__(self) -> None:
        self._policy = self._build_policy()
        self._sandbox = OpenClawActionSandbox()
        self._approvals: dict[str, tuple[RuntimeApproval, NormalizedAction, Session]] = {}
        self._lock = threading.Lock()

    def dispatch_action(self, session: Session, action: Action) -> RuntimeDispatchResult:
        """Evaluate policy and execute immediately when allowed."""
        normalized, error = self._normalize_action(action)
        if error is not None:
            return RuntimeDispatchResult(
                action_id=action.id,
                status=ActionStatus.FAILED,
                error=error,
                audit_result="failed",
                audit_details={"failure_reason": error},
            )

        policy_result = self._policy.evaluate(
            ActionRequest(
                action_type=normalized.action_type,
                user_id=session.user_id,
                session_id=session.id,
                workspace_id=session.id,
                path=normalized.path,
                command=normalized.command,
                url=normalized.url,
                tenant_id=session.tenant_id,
                metadata=normalized.metadata,
            )
        )

        if policy_result.decision == PolicyDecision.DENY:
            error = _bounded_failure_reason("policy_denied", policy_result.reason)
            return RuntimeDispatchResult(
                action_id=action.id,
                status=ActionStatus.FAILED,
                error=error,
                audit_result="failed",
                audit_details={"failure_reason": error, "policy_reason": policy_result.reason},
            )

        if policy_result.decision == PolicyDecision.REQUIRE_APPROVAL:
            approval = RuntimeApproval(
                approval_id=str(uuid.uuid4()),
                action_id=action.id,
                session_id=session.id,
                user_id=session.user_id,
                tenant_id=session.tenant_id,
                action_type=action.action_type,
                normalized_action_type=normalized.action_type.value,
                action_data=dict(action.input_data),
                metadata=dict(normalized.metadata),
                reason=policy_result.reason,
            )
            with self._lock:
                self._approvals[approval.approval_id] = (approval, normalized, session)
            return RuntimeDispatchResult(
                action_id=action.id,
                status=ActionStatus.PENDING,
                approval_id=approval.approval_id,
                audit_result="approval_required",
                audit_details={
                    "approval_id": approval.approval_id,
                    "policy_reason": policy_result.reason,
                },
            )

        return self._execute(session, action.id, normalized)

    def list_approvals(
        self,
        tenant_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[RuntimeApproval], int]:
        """List pending approvals."""
        with self._lock:
            approvals = [
                approval
                for approval, _normalized, _session in self._approvals.values()
                if approval.status == "pending"
                and (tenant_id is None or approval.tenant_id == tenant_id)
            ]
        approvals.sort(key=lambda approval: approval.requested_at, reverse=True)
        total = len(approvals)
        return approvals[offset : offset + limit], total

    def get_approval(self, approval_id: str) -> RuntimeApproval | None:
        """Fetch one approval record by ID."""
        with self._lock:
            record = self._approvals.get(approval_id)
            if record is None:
                return None
            approval, _normalized, _session = record
            return approval

    def approve_action(
        self,
        approval_id: str,
        approver_id: str,
        reason: str = "",
    ) -> RuntimeDispatchResult:
        """Approve a pending action and dispatch it."""
        with self._lock:
            record = self._approvals.get(approval_id)
            if record is None:
                error = _bounded_failure_reason("approval_not_found", approval_id)
                return RuntimeDispatchResult(
                    action_id="",
                    status=ActionStatus.FAILED,
                    error=error,
                    audit_result="failed",
                    audit_details={"failure_reason": error},
                )
            approval, normalized, session = record
            approval.status = "approved"
            approval.decided_by = approver_id
            approval.reason = reason
            approval.decided_at = datetime.now(timezone.utc)

        result = self._execute(session, approval.action_id, normalized)
        result.audit_details.setdefault("approval_id", approval_id)
        return result

    def deny_action(self, approval_id: str, denier_id: str, reason: str = "") -> bool:
        """Record a denied approval."""
        with self._lock:
            record = self._approvals.get(approval_id)
            if record is None:
                return False
            approval, _normalized, _session = record
            approval.status = "denied"
            approval.decided_by = denier_id
            approval.reason = reason
            approval.decided_at = datetime.now(timezone.utc)
            return True

    def cancel_pending_approval(self, approval_id: str, actor_id: str) -> bool:
        """Cancel a pending approval because the action was cancelled."""
        with self._lock:
            record = self._approvals.get(approval_id)
            if record is None:
                return False
            approval, _normalized, _session = record
            approval.status = "cancelled"
            approval.decided_by = actor_id
            approval.reason = "Cancelled"
            approval.decided_at = datetime.now(timezone.utc)
            return True

    def close_session(self, session_id: str) -> None:
        """Clean up sandbox state and pending approvals for a session."""
        sandbox = self._sandbox.get_sandbox_for_session(session_id)
        if sandbox is not None:
            _run_coro(self._sandbox.destroy_sandbox(sandbox.sandbox_id))

        with self._lock:
            for approval, _normalized, _session in self._approvals.values():
                if approval.session_id != session_id or approval.status != "pending":
                    continue
                approval.status = "cancelled"
                approval.reason = "Session closed"
                approval.decided_at = datetime.now(timezone.utc)

    def _build_policy(self) -> OpenClawPolicy:
        policy = create_enterprise_policy()
        policy.add_rule(
            PolicyRule(
                name="allow_keyboard_actions",
                action_types=[ActionType.KEYBOARD],
                decision=PolicyDecision.ALLOW,
                priority=10,
                description="Allow keyboard actions into the configured runtime",
            )
        )
        policy.add_rule(
            PolicyRule(
                name="allow_mouse_actions",
                action_types=[ActionType.MOUSE],
                decision=PolicyDecision.ALLOW,
                priority=10,
                description="Allow mouse actions into the configured runtime",
            )
        )
        return policy

    def _normalize_action(self, action: Action) -> tuple[NormalizedAction | None, str | None]:
        input_data = _normalize_dict(action.input_data)
        metadata = _normalize_dict(action.metadata)
        action_type = str(action.action_type).strip().lower()

        if action_type in {"shell", "shell.execute", "shell_exec", "command.run", "system.shell"}:
            command = input_data.get("command") or input_data.get("cmd")
            if not isinstance(command, str) or not command.strip():
                return None, _bounded_failure_reason("invalid_input", "command is required")
            return NormalizedAction(
                action_type=ActionType.SHELL,
                command=command,
                metadata=metadata,
            ), None

        if action_type in {"code.execute", "code.run", "code.eval"}:
            code = input_data.get("code")
            if not isinstance(code, str) or not code.strip():
                return None, _bounded_failure_reason("invalid_input", "code payload is required")
            language = str(
                input_data.get("lang")
                or metadata.get("lang")
                or metadata.get("language")
                or "python"
            ).lower()
            if language in {"python", "python3"}:
                command = f"python -c {shlex.quote(code)}"
            elif language in {"node", "javascript", "js"}:
                command = f"node -e {shlex.quote(code)}"
            else:
                return None, _bounded_failure_reason(
                    "unsupported_language",
                    language,
                )
            updated_metadata = {
                **metadata,
                "lang": language,
                "source_action_type": action.action_type,
            }
            return NormalizedAction(
                action_type=ActionType.SHELL,
                command=command,
                metadata=updated_metadata,
            ), None

        if action_type in {"file.read", "file_read"}:
            path = input_data.get("path")
            if not isinstance(path, str) or not path.strip():
                return None, _bounded_failure_reason("invalid_input", "path is required")
            return NormalizedAction(ActionType.FILE_READ, path=path, metadata=metadata), None

        if action_type in {"file.write", "file_write"}:
            path = input_data.get("path")
            if not isinstance(path, str) or not path.strip():
                return None, _bounded_failure_reason("invalid_input", "path is required")
            updated_metadata = {**metadata, "content": input_data.get("content", "")}
            return NormalizedAction(
                ActionType.FILE_WRITE, path=path, metadata=updated_metadata
            ), None

        if action_type in {"file.delete", "file_delete"}:
            path = input_data.get("path")
            if not isinstance(path, str) or not path.strip():
                return None, _bounded_failure_reason("invalid_input", "path is required")
            return NormalizedAction(ActionType.FILE_DELETE, path=path, metadata=metadata), None

        if action_type in {"browser", "browser.navigate", "browser.open"}:
            url = input_data.get("url")
            if not isinstance(url, str) or not url.strip():
                return None, _bounded_failure_reason("invalid_input", "url is required")
            return NormalizedAction(ActionType.BROWSER, url=url, metadata=metadata), None

        if action_type in {"screenshot", "browser.screenshot"}:
            url = input_data.get("url")
            return NormalizedAction(ActionType.SCREENSHOT, url=url, metadata=metadata), None

        if action_type in {"api", "api.call", "http.request"}:
            url = input_data.get("url")
            if not isinstance(url, str) or not url.strip():
                return None, _bounded_failure_reason("invalid_input", "url is required")
            return NormalizedAction(ActionType.API, url=url, metadata=metadata), None

        if action_type in {"keyboard", "send-keys", "type"}:
            text = input_data.get("text") or metadata.get("text")
            if not isinstance(text, str) or not text:
                return None, _bounded_failure_reason("invalid_input", "text is required")
            return NormalizedAction(
                ActionType.KEYBOARD,
                metadata={**metadata, "text": text},
            ), None

        if action_type in {"mouse", "click", "computer.click"}:
            x = input_data.get("x")
            y = input_data.get("y")
            if x is None or y is None:
                return None, _bounded_failure_reason("invalid_input", "x and y are required")
            return NormalizedAction(
                ActionType.MOUSE,
                metadata={**metadata, "x": x, "y": y},
            ), None

        return None, _bounded_failure_reason("unsupported_action", action.action_type)

    def _execute(
        self, session: Session, action_id: str, normalized: NormalizedAction
    ) -> RuntimeDispatchResult:
        if normalized.action_type == ActionType.BROWSER:
            return self._runtime_unavailable(
                action_id, normalized, "browser automation is not configured"
            )
        if normalized.action_type == ActionType.SCREENSHOT:
            return self._runtime_unavailable(
                action_id, normalized, "screenshot runtime is not configured"
            )
        if normalized.action_type == ActionType.KEYBOARD:
            return self._runtime_unavailable(
                action_id, normalized, "keyboard runtime is not configured"
            )
        if normalized.action_type == ActionType.MOUSE:
            return self._runtime_unavailable(
                action_id, normalized, "mouse runtime is not configured"
            )
        if normalized.action_type == ActionType.API:
            return self._runtime_unavailable(action_id, normalized, "API runtime is not configured")

        sandbox = _run_coro(self._ensure_sandbox(session))
        if normalized.action_type == ActionType.SHELL:
            sandbox_result = _run_coro(
                self._sandbox.execute_shell(
                    sandbox.sandbox_id,
                    normalized.command or "",
                    timeout=self._timeout_from_session(session),
                )
            )
        elif normalized.action_type == ActionType.FILE_READ:
            sandbox_result = _run_coro(
                self._sandbox.read_file(sandbox.sandbox_id, normalized.path or "")
            )
        elif normalized.action_type == ActionType.FILE_WRITE:
            sandbox_result = _run_coro(
                self._sandbox.write_file(
                    sandbox.sandbox_id,
                    normalized.path or "",
                    str(normalized.metadata.get("content", "")),
                )
            )
        elif normalized.action_type == ActionType.FILE_DELETE:
            sandbox_result = _run_coro(
                self._sandbox.delete_file(sandbox.sandbox_id, normalized.path or "")
            )
        else:
            return self._runtime_unavailable(
                action_id, normalized, "runtime mapping is not configured"
            )

        return self._map_sandbox_result(action_id, normalized, sandbox_result)

    async def _ensure_sandbox(self, session: Session) -> Any:
        existing = self._sandbox.get_sandbox_for_session(session.id)
        if existing is not None:
            return existing

        config = SandboxConfig(
            max_execution_time_seconds=self._timeout_from_session(session),
        )
        return await self._sandbox.create_sandbox(
            session_id=session.id,
            user_id=session.user_id,
            tenant_id=session.tenant_id or "default",
            config=config,
        )

    def _timeout_from_session(self, session: Session) -> int:
        config = _normalize_dict(getattr(session, "config", {}))
        timeout = config.get("timeout", 300)
        try:
            return max(1, min(int(timeout), 3600))
        except (TypeError, ValueError):
            return 300

    def _runtime_unavailable(
        self,
        action_id: str,
        normalized: NormalizedAction,
        detail: str,
    ) -> RuntimeDispatchResult:
        error = _bounded_failure_reason("runtime_unavailable", detail)
        output = {
            "runtime": "openclaw_action_runtime",
            "normalized_action_type": normalized.action_type.value,
            "failure_reason": error,
        }
        return RuntimeDispatchResult(
            action_id=action_id,
            status=ActionStatus.FAILED,
            output_data=output,
            error=error,
            audit_result="failed",
            audit_details={"failure_reason": error},
        )

    def _map_sandbox_result(
        self,
        action_id: str,
        normalized: NormalizedAction,
        sandbox_result: SandboxActionResult,
    ) -> RuntimeDispatchResult:
        output = {
            "runtime": "openclaw_action_sandbox",
            "normalized_action_type": normalized.action_type.value,
            "result": sandbox_result.output,
            "execution_time_ms": int(sandbox_result.execution_time_ms),
        }

        if sandbox_result.success:
            return RuntimeDispatchResult(
                action_id=action_id,
                status=ActionStatus.COMPLETED,
                executed=True,
                output_data=output,
                execution_time_ms=int(sandbox_result.execution_time_ms),
                audit_result="success",
                audit_details={"execution_time_ms": int(sandbox_result.execution_time_ms)},
            )

        raw_error = sandbox_result.error or "execution failed"
        lowered = raw_error.lower()
        if "timed out" in lowered:
            status = ActionStatus.TIMEOUT
            audit_result = "timeout"
            failure_code = "timeout"
        else:
            status = ActionStatus.FAILED
            audit_result = "failed"
            failure_code = "execution_failed"

        error = _bounded_failure_reason(failure_code, raw_error)
        output["failure_reason"] = error
        return RuntimeDispatchResult(
            action_id=action_id,
            status=status,
            executed=True,
            output_data=output,
            error=error,
            execution_time_ms=int(sandbox_result.execution_time_ms),
            audit_result=audit_result,
            audit_details={
                "execution_time_ms": int(sandbox_result.execution_time_ms),
                "failure_reason": error,
            },
        )


_runtime: OpenClawExecutionRuntime | None = None


def get_openclaw_execution_runtime() -> OpenClawExecutionRuntime:
    """Get the process-global runtime manager."""
    global _runtime
    if _runtime is None:
        _runtime = OpenClawExecutionRuntime()
    return _runtime


__all__ = [
    "RuntimeApproval",
    "RuntimeDispatchResult",
    "get_openclaw_execution_runtime",
]
