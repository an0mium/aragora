"""
Nomic loop streaming via WebSocket.

Provides hook functions for emitting nomic loop events to connected clients
in real-time. Works with the existing DebateStreamServer infrastructure.

Refactored to use category-based organization for improved maintainability.
"""

from typing import Callable, Optional

from .stream import StreamEvent, StreamEventType, SyncEventEmitter


def _emit(emitter: SyncEventEmitter, event_type: StreamEventType, data: dict, **kwargs) -> None:
    """Helper to emit a stream event with optional round/agent fields."""
    emitter.emit(
        StreamEvent(
            type=event_type,
            data=data,
            round=kwargs.get("round", 0),
            agent=kwargs.get("agent", ""),
        )
    )


def _create_cycle_hooks(emitter: SyncEventEmitter) -> dict[str, Callable]:
    """Create hooks for cycle lifecycle events."""

    def on_cycle_start(cycle_num: int, max_cycles: int, started_at: str) -> None:
        _emit(
            emitter,
            StreamEventType.CYCLE_START,
            {
                "cycle": cycle_num,
                "max_cycles": max_cycles,
                "started_at": started_at,
            },
        )

    def on_cycle_end(cycle_num: int, success: bool, duration_seconds: float, outcome: str) -> None:
        _emit(
            emitter,
            StreamEventType.CYCLE_END,
            {
                "cycle": cycle_num,
                "success": success,
                "duration_seconds": duration_seconds,
                "outcome": outcome,
            },
        )

    return {"on_cycle_start": on_cycle_start, "on_cycle_end": on_cycle_end}


def _create_phase_hooks(emitter: SyncEventEmitter) -> dict[str, Callable]:
    """Create hooks for phase lifecycle events."""

    def on_phase_start(phase: str, cycle: int, details: Optional[dict] = None) -> None:
        _emit(
            emitter,
            StreamEventType.PHASE_START,
            {
                "phase": phase,
                "cycle": cycle,
                **(details or {}),
            },
        )

    def on_phase_end(
        phase: str,
        cycle: int,
        success: bool,
        duration_seconds: float,
        result: Optional[dict] = None,
    ) -> None:
        _emit(
            emitter,
            StreamEventType.PHASE_END,
            {
                "phase": phase,
                "cycle": cycle,
                "success": success,
                "duration_seconds": duration_seconds,
                **(result or {}),
            },
        )

    return {"on_phase_start": on_phase_start, "on_phase_end": on_phase_end}


def _create_task_hooks(emitter: SyncEventEmitter) -> dict[str, Callable]:
    """Create hooks for implementation task events."""

    def on_task_start(
        task_id: str,
        description: str,
        complexity: str,
        model: str,
        total_tasks: int,
        completed_tasks: int,
    ) -> None:
        _emit(
            emitter,
            StreamEventType.TASK_START,
            {
                "task_id": task_id,
                "description": description,
                "complexity": complexity,
                "model": model,
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
            },
        )

    def on_task_complete(
        task_id: str,
        success: bool,
        duration_seconds: float,
        diff_preview: str = "",
        error: Optional[str] = None,
    ) -> None:
        _emit(
            emitter,
            StreamEventType.TASK_COMPLETE,
            {
                "task_id": task_id,
                "success": success,
                "duration_seconds": duration_seconds,
                "diff_preview": diff_preview,
                "error": error,
            },
        )

    def on_task_retry(task_id: str, attempt: int, reason: str, timeout: int) -> None:
        _emit(
            emitter,
            StreamEventType.TASK_RETRY,
            {
                "task_id": task_id,
                "attempt": attempt,
                "reason": reason,
                "timeout": timeout,
            },
        )

    return {
        "on_task_start": on_task_start,
        "on_task_complete": on_task_complete,
        "on_task_retry": on_task_retry,
    }


def _create_verification_hooks(emitter: SyncEventEmitter) -> dict[str, Callable]:
    """Create hooks for verification phase events."""

    def on_verification_start(checks: list[str]) -> None:
        _emit(emitter, StreamEventType.VERIFICATION_START, {"checks": checks})

    def on_verification_result(check_name: str, passed: bool, message: str = "") -> None:
        _emit(
            emitter,
            StreamEventType.VERIFICATION_RESULT,
            {
                "check": check_name,
                "passed": passed,
                "message": message,
            },
        )

    return {
        "on_verification_start": on_verification_start,
        "on_verification_result": on_verification_result,
    }


def _create_backup_hooks(emitter: SyncEventEmitter) -> dict[str, Callable]:
    """Create hooks for backup/commit events."""

    def on_commit(commit_hash: str, message: str, files_changed: int) -> None:
        _emit(
            emitter,
            StreamEventType.COMMIT,
            {
                "commit_hash": commit_hash,
                "message": message,
                "files_changed": files_changed,
            },
        )

    def on_backup_created(backup_name: str, files_count: int, reason: str) -> None:
        _emit(
            emitter,
            StreamEventType.BACKUP_CREATED,
            {
                "backup_name": backup_name,
                "files_count": files_count,
                "reason": reason,
            },
        )

    def on_backup_restored(backup_name: str, files_count: int, reason: str) -> None:
        _emit(
            emitter,
            StreamEventType.BACKUP_RESTORED,
            {
                "backup_name": backup_name,
                "files_count": files_count,
                "reason": reason,
            },
        )

    return {
        "on_commit": on_commit,
        "on_backup_created": on_backup_created,
        "on_backup_restored": on_backup_restored,
    }


def _create_log_hooks(emitter: SyncEventEmitter) -> dict[str, Callable]:
    """Create hooks for logging and error events."""

    def on_error(phase: str, message: str, recoverable: bool = True) -> None:
        _emit(
            emitter,
            StreamEventType.ERROR,
            {
                "phase": phase,
                "message": message,
                "recoverable": recoverable,
            },
        )

    def on_log_message(
        message: str, level: str = "info", phase: Optional[str] = None, agent: Optional[str] = None
    ) -> None:
        _emit(
            emitter,
            StreamEventType.LOG_MESSAGE,
            {
                "message": message,
                "level": level,
                "phase": phase,
            },
            agent=agent or "",
        )

    def on_match_recorded(
        debate_id: str,
        participants: list[str],
        elo_changes: dict[str, float],
        domain: Optional[str] = None,
        winner: Optional[str] = None,
        loop_id: Optional[str] = None,
    ) -> None:
        _emit(
            emitter,
            StreamEventType.MATCH_RECORDED,
            {
                "debate_id": debate_id,
                "participants": participants,
                "elo_changes": elo_changes,
                "domain": domain,
                "winner": winner,
                "loop_id": loop_id,
            },
        )

    return {
        "on_error": on_error,
        "on_log_message": on_log_message,
        "on_match_recorded": on_match_recorded,
    }


def _create_probe_hooks(emitter: SyncEventEmitter) -> dict[str, Callable]:
    """Create hooks for capability probing events."""

    def on_probe_start(
        probe_id: str, target_agent: str, probe_types: list[str], probes_per_type: int = 3
    ) -> None:
        _emit(
            emitter,
            StreamEventType.PROBE_START,
            {
                "probe_id": probe_id,
                "target_agent": target_agent,
                "probe_types": probe_types,
                "probes_per_type": probes_per_type,
                "total_probes": len(probe_types) * probes_per_type,
            },
        )

    def on_probe_result(
        probe_id: str,
        probe_type: str,
        passed: bool,
        severity: Optional[str] = None,
        description: str = "",
        response_time_ms: float = 0,
    ) -> None:
        _emit(
            emitter,
            StreamEventType.PROBE_RESULT,
            {
                "probe_id": probe_id,
                "probe_type": probe_type,
                "passed": passed,
                "severity": severity,
                "description": description,
                "response_time_ms": response_time_ms,
            },
        )

    def on_probe_complete(
        report_id: str,
        target_agent: str,
        probes_run: int,
        vulnerabilities_found: int,
        vulnerability_rate: float,
        elo_penalty: float,
        by_severity: Optional[dict] = None,
    ) -> None:
        _emit(
            emitter,
            StreamEventType.PROBE_COMPLETE,
            {
                "report_id": report_id,
                "target_agent": target_agent,
                "probes_run": probes_run,
                "vulnerabilities_found": vulnerabilities_found,
                "vulnerability_rate": vulnerability_rate,
                "elo_penalty": elo_penalty,
                "by_severity": by_severity or {},
            },
        )

    return {
        "on_probe_start": on_probe_start,
        "on_probe_result": on_probe_result,
        "on_probe_complete": on_probe_complete,
    }


def _create_audit_hooks(emitter: SyncEventEmitter) -> dict[str, Callable]:
    """Create hooks for deep audit events."""

    def on_audit_start(
        audit_id: str, task: str, agents: list[str], config: Optional[dict] = None
    ) -> None:
        _emit(
            emitter,
            StreamEventType.AUDIT_START,
            {
                "audit_id": audit_id,
                "task": task,
                "agents": agents,
                "config": config or {},
                "rounds": config.get("rounds", 6) if config else 6,
            },
        )

    def on_audit_round(
        audit_id: str,
        round_num: int,
        round_name: str,
        cognitive_role: str,
        messages: list[dict],
        duration_ms: float = 0,
    ) -> None:
        _emit(
            emitter,
            StreamEventType.AUDIT_ROUND,
            {
                "audit_id": audit_id,
                "round": round_num,
                "name": round_name,
                "cognitive_role": cognitive_role,
                "messages": messages,
                "duration_ms": duration_ms,
            },
            round=round_num,
        )

    def on_audit_finding(
        audit_id: str,
        category: str,
        summary: str,
        details: str,
        agents_agree: list[str],
        agents_disagree: list[str],
        confidence: float,
        severity: float = 0.0,
    ) -> None:
        _emit(
            emitter,
            StreamEventType.AUDIT_FINDING,
            {
                "audit_id": audit_id,
                "category": category,
                "summary": summary,
                "details": details,
                "agents_agree": agents_agree,
                "agents_disagree": agents_disagree,
                "confidence": confidence,
                "severity": severity,
            },
        )

    def on_audit_cross_exam(
        audit_id: str, synthesizer: str, questions: list[str], notes: str
    ) -> None:
        _emit(
            emitter,
            StreamEventType.AUDIT_CROSS_EXAM,
            {
                "audit_id": audit_id,
                "synthesizer": synthesizer,
                "questions": questions,
                "notes": notes,
            },
        )

    def on_audit_verdict(
        audit_id: str,
        task: str,
        recommendation: str,
        confidence: float,
        unanimous_issues: list[str],
        split_opinions: list[str],
        risk_areas: list[str],
        rounds_completed: int,
        total_duration_ms: float,
        agents: list[str],
        elo_adjustments: Optional[dict] = None,
    ) -> None:
        _emit(
            emitter,
            StreamEventType.AUDIT_VERDICT,
            {
                "audit_id": audit_id,
                "task": task,
                "recommendation": recommendation,
                "confidence": confidence,
                "unanimous_issues": unanimous_issues,
                "split_opinions": split_opinions,
                "risk_areas": risk_areas,
                "rounds_completed": rounds_completed,
                "total_duration_ms": total_duration_ms,
                "agents": agents,
                "elo_adjustments": elo_adjustments or {},
            },
        )

    return {
        "on_audit_start": on_audit_start,
        "on_audit_round": on_audit_round,
        "on_audit_finding": on_audit_finding,
        "on_audit_cross_exam": on_audit_cross_exam,
        "on_audit_verdict": on_audit_verdict,
    }


def create_nomic_hooks(emitter: SyncEventEmitter) -> dict[str, Callable]:
    """
    Create hook functions for nomic loop event emission.

    These hooks should be called by NomicLoop at key points during execution.
    Events are emitted to the emitter queue for async WebSocket broadcast.

    Hook Categories:
        - Cycle: on_cycle_start, on_cycle_end
        - Phase: on_phase_start, on_phase_end
        - Task: on_task_start, on_task_complete, on_task_retry
        - Verification: on_verification_start, on_verification_result
        - Backup: on_commit, on_backup_created, on_backup_restored
        - Log: on_error, on_log_message, on_match_recorded
        - Probe: on_probe_start, on_probe_result, on_probe_complete
        - Audit: on_audit_start, on_audit_round, on_audit_finding,
                 on_audit_cross_exam, on_audit_verdict

    Returns:
        dict of hook name -> callback function
    """
    hooks = {}
    hooks.update(_create_cycle_hooks(emitter))
    hooks.update(_create_phase_hooks(emitter))
    hooks.update(_create_task_hooks(emitter))
    hooks.update(_create_verification_hooks(emitter))
    hooks.update(_create_backup_hooks(emitter))
    hooks.update(_create_log_hooks(emitter))
    hooks.update(_create_probe_hooks(emitter))
    hooks.update(_create_audit_hooks(emitter))
    return hooks
