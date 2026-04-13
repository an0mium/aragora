"""Autonomous conductor substrate for swarm worker follow-ups.

This module decides what to do after a worker finishes:

- classify the terminal state
- determine whether to retry, switch agents, decompose, escalate, or stop
- generate the next prompt without manual copy-paste
- persist issue/session history so the next attempt has context

It is intentionally standalone for now and is not wired into ``boss_loop.py``.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from aragora.swarm.ping_pong import build_handoff_prompt
from aragora.swarm.terminal_truth import (
    TerminalClass,
    classify_from_metrics,
    extract_run_worker_outcome,
)

NextAction = Literal["retry_same", "retry_different_agent", "decompose", "escalate", "done"]
_VALID_NEXT_ACTIONS = frozenset(
    {"retry_same", "retry_different_agent", "decompose", "escalate", "done"}
)

_ALREADY_DONE_MARKERS = (
    "already implemented",
    "already exists",
    "no changes needed",
    "nothing to commit",
)
_AUTH_MARKERS = (
    "auth",
    "authentication",
    "unauthorized",
    "permission denied",
    "credentials",
    "token",
    "login",
    "forbidden",
    "403",
    "401",
)
_RUNNER_MARKERS = (
    "no runner",
    "runner unavailable",
    "agent unavailable",
    "worker type blocked",
    "command not found",
)
_SCOPE_MARKERS = (
    "too broad",
    "split",
    "decompose",
    "decomposition",
    "multiple concerns",
    "multiple files",
    "too many files",
    "out of scope",
    "scope too large",
)
_VALIDATION_MARKERS = (
    "pytest",
    "test failed",
    "tests failed",
    "assertionerror",
    "typecheck",
    "mypy",
    "ruff",
    "lint",
    "verification failed",
)
_TIMEOUT_MARKERS = ("timed out", "timeout", "no progress")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        value = _text(raw)
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _truncate(text: str, *, limit: int = 1600) -> str:
    normalized = _text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _coerce_next_action(value: Any) -> NextAction:
    action = _text(value)
    if action in _VALID_NEXT_ACTIONS:
        return cast(NextAction, action)
    return "escalate"


def _git_common_dir(repo_root: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    path_text = _text(result.stdout)
    if result.returncode != 0 or not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path if path.exists() else None


@dataclass(slots=True)
class ConductorStep:
    issue_number: int
    session_id: str
    worker_output: str
    terminal_class: TerminalClass
    changed_files: list[str]
    next_action: NextAction
    next_prompt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_number": self.issue_number,
            "session_id": self.session_id,
            "worker_output": self.worker_output,
            "terminal_class": self.terminal_class.value,
            "changed_files": list(self.changed_files),
            "next_action": self.next_action,
            "next_prompt": self.next_prompt,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConductorStep":
        return cls(
            issue_number=int(data.get("issue_number", 0) or 0),
            session_id=_text(data.get("session_id")) or "unknown-session",
            worker_output=_text(data.get("worker_output")),
            terminal_class=TerminalClass(_text(data.get("terminal_class"))),
            changed_files=[
                _text(item) for item in list(data.get("changed_files", []) or []) if _text(item)
            ],
            next_action=_coerce_next_action(data.get("next_action")),
            next_prompt=_text(data.get("next_prompt")),
        )


class SessionStateStore:
    """Small JSON-backed issue/session history for conductor retries."""

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root).resolve()
        git_common = _git_common_dir(self.repo_root)
        self.storage_dir = (
            git_common / "aragora_conductor"
            if git_common is not None
            else self.repo_root / ".aragora_conductor"
        )
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _issue_path(self, issue_number: int) -> Path:
        return self.storage_dir / f"issue-{int(issue_number)}.json"

    def load_steps(self, issue_number: int) -> list[ConductorStep]:
        path = self._issue_path(issue_number)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        steps: list[ConductorStep] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                steps.append(ConductorStep.from_dict(item))
            except (TypeError, ValueError):
                continue
        return steps

    def append_step(self, step: ConductorStep) -> None:
        steps = self.load_steps(step.issue_number)
        steps.append(step)
        path = self._issue_path(step.issue_number)
        path.write_text(
            json.dumps([item.to_dict() for item in steps], indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def latest_step(self, issue_number: int) -> ConductorStep | None:
        steps = self.load_steps(issue_number)
        return steps[-1] if steps else None

    def latest_session_id(self, issue_number: int) -> str | None:
        latest = self.latest_step(issue_number)
        if latest is None:
            return None
        return latest.session_id


class Conductor:
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root).resolve()
        self._session_store = SessionStateStore(self.repo_root)

    def evaluate_worker_output(
        self, issue_number: int, worker_result: dict[str, Any]
    ) -> ConductorStep:
        """Read worker output and decide what to do next."""
        history = self._session_store.load_steps(issue_number)
        worker_output = self._extract_worker_output(worker_result)
        changed_files = self._extract_changed_files(worker_result)
        terminal_class = self._classify_terminal(worker_result, worker_output, changed_files)
        session_id = self._extract_session_id(issue_number, worker_result, history)
        failure_reason = self._failure_reason(worker_result, worker_output, terminal_class)

        if self._is_done(worker_result, worker_output, terminal_class):
            next_action: NextAction = "done"
            next_prompt = ""
        else:
            next_action = self._choose_next_action(
                terminal_class=terminal_class,
                failure_reason=failure_reason,
                worker_output=worker_output,
                changed_files=changed_files,
                history=history,
            )
            if next_action == "retry_same":
                next_prompt = self.generate_retry_prompt(
                    issue_number, worker_output, failure_reason
                )
            elif next_action == "retry_different_agent":
                next_prompt = self._build_different_agent_prompt(
                    issue_number=issue_number,
                    worker_result=worker_result,
                    worker_output=worker_output,
                    failure_reason=failure_reason,
                    changed_files=changed_files,
                    history=history,
                )
            elif next_action == "decompose":
                next_prompt = self._build_decomposition_prompt(
                    issue_number=issue_number,
                    worker_output=worker_output,
                    failure_reason=failure_reason,
                    changed_files=changed_files,
                    terminal_class=terminal_class,
                )
            else:
                next_prompt = self._build_escalation_prompt(
                    issue_number=issue_number,
                    worker_output=worker_output,
                    failure_reason=failure_reason,
                    terminal_class=terminal_class,
                )

        step = ConductorStep(
            issue_number=issue_number,
            session_id=session_id,
            worker_output=worker_output,
            terminal_class=terminal_class,
            changed_files=changed_files,
            next_action=next_action,
            next_prompt=next_prompt,
        )
        self._session_store.append_step(step)
        return step

    def dispatch_step_to_tmux(
        self,
        step: ConductorStep,
        *,
        session_name: str,
        wait_seconds: int = 0,
        harvest_lines: int = 200,
    ) -> dict[str, Any]:
        """Send a generated follow-up prompt to a tmux-managed agent session.

        This stays intentionally thin: it reuses the already-landed shell
        transport scripts instead of reimplementing pane control here.
        """
        prompt_text = _text(step.next_prompt)
        if not prompt_text:
            return {
                "sent": False,
                "reason": "no_prompt",
                "session_name": session_name,
                "next_action": step.next_action,
            }

        send_script = self.repo_root / "scripts" / "tmux_send_prompt.sh"
        harvest_script = self.repo_root / "scripts" / "tmux_harvest.sh"
        if not send_script.exists():
            raise FileNotFoundError(f"tmux send script not found: {send_script}")

        prompt_path = self._write_dispatch_prompt_file(step, prompt_text)
        try:
            send_cmd = [
                str(send_script),
                "--name",
                session_name,
                "--prompt-file",
                str(prompt_path),
            ]
            if wait_seconds > 0:
                send_cmd.extend(["--wait", str(wait_seconds)])
            send_result = subprocess.run(
                send_cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            prompt_path.unlink(missing_ok=True)

        if send_result.returncode != 0:
            detail = _text(send_result.stderr) or _text(send_result.stdout) or "tmux send failed"
            raise RuntimeError(detail)

        harvest_output = ""
        harvest_error = ""
        if harvest_lines > 0 and harvest_script.exists():
            harvest_result = subprocess.run(
                [
                    str(harvest_script),
                    "--name",
                    session_name,
                    "--lines",
                    str(harvest_lines),
                ],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if harvest_result.returncode == 0:
                harvest_output = _text(harvest_result.stdout)
            else:
                harvest_error = _text(harvest_result.stderr) or _text(harvest_result.stdout)

        return {
            "sent": True,
            "session_name": session_name,
            "next_action": step.next_action,
            "prompt_chars": len(prompt_text),
            "wait_seconds": max(wait_seconds, 0),
            "harvest_lines": max(harvest_lines, 0),
            "harvest_output": harvest_output,
            "harvest_error": harvest_error,
        }

    def generate_retry_prompt(
        self, issue_number: int, prior_output: str, failure_reason: str
    ) -> str:
        """Generate a targeted retry prompt based on what failed."""
        steps = self._session_store.load_steps(issue_number)
        lines = [f"# Retry issue #{issue_number}", "", "## What was tried"]

        if steps:
            start = max(0, len(steps) - 3)
            for index, step in enumerate(steps[start:], start=start + 1):
                lines.append(
                    f"- Attempt {index}: terminal={step.terminal_class.value}, "
                    f"conductor_action={step.next_action}"
                )
        else:
            lines.append("- No prior conductor attempts recorded.")

        excerpt = _truncate(prior_output, limit=1800) or "(no worker output captured)"
        lines.extend(
            [
                "",
                "Most recent worker output:",
                excerpt,
                "",
                "## Why it failed",
                failure_reason or "unknown",
                "",
                "## What to try differently",
            ]
        )

        for item in self._repair_guidance(failure_reason=failure_reason, prior_output=prior_output):
            lines.append(f"- {item}")

        lines.extend(
            [
                "",
                "## Constraints",
                "- Reuse any correct partial progress instead of starting over blindly.",
                "- Keep the attempt bounded and end with a concrete deliverable or an exact blocker.",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _extract_session_id(
        issue_number: int,
        worker_result: dict[str, Any],
        history: list[ConductorStep],
    ) -> str:
        for key in ("session_id", "run_id", "receipt_id"):
            value = _text(worker_result.get(key))
            if value:
                return value
        if history:
            return history[-1].session_id
        return f"issue-{issue_number}"

    @staticmethod
    def _extract_worker_output(worker_result: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in (
            "worker_output",
            "error",
            "stdout",
            "stderr",
            "stdout_tail",
            "stderr_tail",
            "transcript",
            "log_tail",
        ):
            value = _text(worker_result.get(key))
            if value:
                parts.append(value)

        for item in list(worker_result.get("reasons", []) or []):
            value = _text(item)
            if value:
                parts.append(value)

        run_dict = worker_result.get("run")
        if isinstance(run_dict, dict):
            for work_order in run_dict.get("work_orders", []):
                if not isinstance(work_order, dict):
                    continue
                for key in (
                    "stdout_tail",
                    "stderr_tail",
                    "transcript",
                    "log_tail",
                    "failure_reason",
                    "blocking_question",
                ):
                    value = _text(work_order.get(key))
                    if value:
                        parts.append(value)

        return "\n\n".join(_ordered_unique(parts))

    @staticmethod
    def _extract_changed_files(worker_result: dict[str, Any]) -> list[str]:
        files: list[str] = []
        for key in ("changed_files", "changed_paths"):
            value = worker_result.get(key)
            if isinstance(value, list):
                files.extend(_text(item) for item in value if _text(item))

        deliverable = worker_result.get("deliverable")
        if isinstance(deliverable, dict):
            changed_paths = deliverable.get("changed_paths")
            if isinstance(changed_paths, list):
                files.extend(_text(item) for item in changed_paths if _text(item))

        run_dict = worker_result.get("run")
        if isinstance(run_dict, dict):
            for work_order in run_dict.get("work_orders", []):
                if not isinstance(work_order, dict):
                    continue
                changed_paths = work_order.get("changed_paths")
                if isinstance(changed_paths, list):
                    files.extend(_text(item) for item in changed_paths if _text(item))

        return _ordered_unique(files)

    def _classify_terminal(
        self,
        worker_result: dict[str, Any],
        worker_output: str,
        changed_files: list[str],
    ) -> TerminalClass:
        raw_terminal = worker_result.get("terminal_class")
        if isinstance(raw_terminal, TerminalClass):
            return raw_terminal
        raw_terminal_text = _text(raw_terminal)
        if raw_terminal_text:
            try:
                return TerminalClass(raw_terminal_text)
            except ValueError:
                pass

        outcome = _text(worker_result.get("outcome")).lower()
        deliverable = worker_result.get("deliverable")
        deliverable_type = _text(deliverable.get("type")) if isinstance(deliverable, dict) else ""

        if outcome == "pr_adopted" or deliverable_type in {"pr", "adopted_pr"}:
            return TerminalClass.DELIVERABLE_PR_CREATED
        if outcome == "deliverable_created":
            return TerminalClass.DELIVERABLE_BRANCH_PUSHED
        if outcome == "issue_already_resolved" or (
            not changed_files and _contains_any(worker_output, _ALREADY_DONE_MARKERS)
        ):
            return TerminalClass.ISSUE_ALREADY_RESOLVED

        run_dict = worker_result.get("run")
        worker_outcome = outcome
        if not worker_outcome and isinstance(run_dict, dict):
            worker_outcome = _text(extract_run_worker_outcome(run_dict)).lower()

        if not worker_outcome and _contains_any(worker_output, _TIMEOUT_MARKERS):
            worker_outcome = "timeout"
        elif not worker_outcome and _contains_any(worker_output, _AUTH_MARKERS):
            worker_outcome = "auth_failure"
        elif not worker_outcome and _contains_any(worker_output, _RUNNER_MARKERS):
            worker_outcome = "no_fresh_runner"
        elif not worker_outcome and _contains_any(worker_output, _SCOPE_MARKERS):
            worker_outcome = "blocked"

        metrics_row = {
            "worker_status": _text(worker_result.get("status")).lower(),
            "worker_outcome": worker_outcome,
            "publish_action": _text(
                worker_result.get("publish_action")
                or ((worker_result.get("publish_result") or {}).get("action"))
            ).lower(),
            "files_changed": len(changed_files),
            "elapsed_seconds": float(worker_result.get("elapsed_seconds", 0.0) or 0.0),
            "has_deliverable": bool(outcome == "deliverable_created"),
        }
        return classify_from_metrics(metrics_row)

    def _is_done(
        self,
        worker_result: dict[str, Any],
        worker_output: str,
        terminal_class: TerminalClass,
    ) -> bool:
        if terminal_class.is_success:
            return True
        outcome = _text(worker_result.get("outcome")).lower()
        if outcome in {"deliverable_created", "pr_adopted", "issue_already_resolved"}:
            return True
        return not self._extract_changed_files(worker_result) and _contains_any(
            worker_output, _ALREADY_DONE_MARKERS
        )

    def _failure_reason(
        self,
        worker_result: dict[str, Any],
        worker_output: str,
        terminal_class: TerminalClass,
    ) -> str:
        candidates: list[str] = []
        for key in ("failure_reason", "error", "blocked_reason"):
            value = _text(worker_result.get(key))
            if value:
                candidates.append(value)

        for item in list(worker_result.get("reasons", []) or []):
            value = _text(item)
            if value:
                candidates.append(value)

        run_dict = worker_result.get("run")
        if isinstance(run_dict, dict):
            for work_order in run_dict.get("work_orders", []):
                if not isinstance(work_order, dict):
                    continue
                for key in ("failure_reason", "blocking_question", "dispatch_error"):
                    value = _text(work_order.get(key))
                    if value:
                        candidates.append(value)

        for value in candidates:
            if value:
                return value

        if _contains_any(worker_output, _AUTH_MARKERS):
            return "auth_failure"
        if _contains_any(worker_output, _RUNNER_MARKERS):
            return "no_runner"
        if _contains_any(worker_output, _VALIDATION_MARKERS):
            return "verification_failed"
        if _contains_any(worker_output, _SCOPE_MARKERS):
            return "task_too_broad"
        if _contains_any(worker_output, _TIMEOUT_MARKERS):
            return "worker_timeout"
        return terminal_class.value

    def _choose_next_action(
        self,
        *,
        terminal_class: TerminalClass,
        failure_reason: str,
        worker_output: str,
        changed_files: list[str],
        history: list[ConductorStep],
    ) -> NextAction:
        if terminal_class in {
            TerminalClass.BLOCKED_AUTH_FAILURE,
            TerminalClass.BLOCKED_NO_RUNNER,
            TerminalClass.BLOCKED_VALIDATION_TARGET_MISSING,
            TerminalClass.BLOCKED_DECOMPOSITION_LIMIT,
            TerminalClass.RESCUE_PUBLISH_DEFERRED,
        }:
            return "escalate"

        if terminal_class in {
            TerminalClass.BLOCKED_NOT_DISPATCH_BOUNDED,
            TerminalClass.BLOCKED_SANITATION_FAILED,
        }:
            if self._decomposition_would_help(failure_reason, worker_output, changed_files):
                return "decompose"
            return "escalate"

        if terminal_class == TerminalClass.RESCUE_TIMEOUT:
            return "retry_same"

        if terminal_class == TerminalClass.RESCUE_VERIFICATION_FAILED:
            return "retry_different_agent"

        if terminal_class == TerminalClass.RESCUE_WORKER_CRASH:
            if self._is_infrastructure_failure(failure_reason, worker_output):
                return "escalate"
            if history and not changed_files:
                return "retry_different_agent"
            return "retry_same"

        if terminal_class == TerminalClass.RESCUE_NO_DELIVERABLE:
            if self._decomposition_would_help(failure_reason, worker_output, changed_files):
                return "decompose"
            if history or changed_files:
                return "retry_different_agent"
            return "retry_same"

        return "retry_same"

    @staticmethod
    def _decomposition_would_help(
        failure_reason: str,
        worker_output: str,
        changed_files: list[str],
    ) -> bool:
        combined = "\n".join([failure_reason, worker_output])
        if _contains_any(combined, _SCOPE_MARKERS):
            return True
        mentioned_files = re.findall(r"\b[\w./-]+\.(?:py|ts|tsx|js|jsx|md)\b", combined)
        return len(_ordered_unique(changed_files + mentioned_files)) >= 4

    @staticmethod
    def _is_infrastructure_failure(failure_reason: str, worker_output: str) -> bool:
        combined = "\n".join([failure_reason, worker_output])
        return _contains_any(combined, _AUTH_MARKERS + _RUNNER_MARKERS)

    def _build_different_agent_prompt(
        self,
        *,
        issue_number: int,
        worker_result: dict[str, Any],
        worker_output: str,
        failure_reason: str,
        changed_files: list[str],
        history: list[ConductorStep],
    ) -> str:
        previous_agent = (
            _text(worker_result.get("agent"))
            or _text(worker_result.get("target_agent"))
            or _text((worker_result.get("receipt_metadata") or {}).get("requested_target_agent"))
            or "previous_worker"
        )
        handoff = build_handoff_prompt(
            goal=f"Repair issue #{issue_number} without repeating the failed approach.",
            previous_transcript=worker_output or failure_reason or "(no transcript captured)",
            previous_agent=previous_agent,
            next_agent="different_agent",
            round_number=len(history) + 1,
            files_changed=changed_files,
            remaining_issues=[failure_reason] if failure_reason else [],
        )
        return (
            handoff + "\n\n## Additional guidance\n"
            "- Start by verifying the failure independently.\n"
            "- Keep any correct partial edits, but do not trust the prior approach blindly."
        )

    @staticmethod
    def _build_decomposition_prompt(
        *,
        issue_number: int,
        worker_output: str,
        failure_reason: str,
        changed_files: list[str],
        terminal_class: TerminalClass,
    ) -> str:
        lines = [
            f"# Decompose issue #{issue_number}",
            "",
            "The last autonomous attempt should not be retried as a single bounded worker task.",
            "",
            "## Why decomposition is needed",
            f"- terminal_class: {terminal_class.value}",
            f"- failure_reason: {failure_reason or 'unknown'}",
        ]
        if changed_files:
            lines.append(f"- files involved: {', '.join(changed_files[:10])}")
        excerpt = _truncate(worker_output, limit=1400)
        if excerpt:
            lines.extend(["", "## Worker evidence", excerpt])
        lines.extend(
            [
                "",
                "## Your task",
                "Split the issue into 2-5 bounded subtasks.",
                "Each subtask must include file scope, validation, and a single concrete deliverable.",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _build_escalation_prompt(
        *,
        issue_number: int,
        worker_output: str,
        failure_reason: str,
        terminal_class: TerminalClass,
    ) -> str:
        lines = [
            f"# Escalate issue #{issue_number}",
            "",
            "The conductor is stopping autonomous retries until an external blocker is cleared.",
            "",
            "## Escalation summary",
            f"- terminal_class: {terminal_class.value}",
            f"- failure_reason: {failure_reason or 'unknown'}",
        ]
        excerpt = _truncate(worker_output, limit=1200)
        if excerpt:
            lines.extend(["", "## Worker evidence", excerpt])
        lines.extend(
            [
                "",
                "## Next owner",
                "Route this to a human or infrastructure owner, clear the blocker, then re-queue.",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _repair_guidance(*, failure_reason: str, prior_output: str) -> list[str]:
        combined = "\n".join([failure_reason, prior_output]).lower()
        guidance: list[str] = []

        if _contains_any(combined, _VALIDATION_MARKERS):
            guidance.append(
                "Reproduce the failing verification first and fix the smallest failing assertion, lint, or type error."
            )
        if _contains_any(combined, _TIMEOUT_MARKERS):
            guidance.append(
                "Avoid full-suite validation or broad searches; stay within the files already implicated."
            )
        if "crash" in combined or "traceback" in combined:
            guidance.append(
                "Start by reproducing the crashing command or traceback before changing more code."
            )
        if _contains_any(combined, _SCOPE_MARKERS):
            guidance.append(
                "Choose one bounded slice and explicitly defer any remaining scope instead of widening the task."
            )
        if not guidance:
            guidance.append(
                "Change the approach, not just the wording: verify the failure, make the smallest bounded edit, then validate."
            )
        guidance.append("Do not repeat the same command sequence unless you learned something new.")
        return _ordered_unique(guidance)

    def _write_dispatch_prompt_file(self, step: ConductorStep, prompt_text: str) -> Path:
        temp = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f"conductor-issue-{step.issue_number}-",
            suffix=".md",
            dir=self._session_store.storage_dir,
            delete=False,
        )
        with temp:
            temp.write(prompt_text)
        return Path(temp.name)


__all__ = ["Conductor", "ConductorStep", "SessionStateStore"]
