from __future__ import annotations

import uuid
from pathlib import Path
from typing import Callable

from .exceptions import TransportError
from .footer import build_footer_instruction
from .footer import build_repair_prompt
from .harnesses import create_transport
from .harnesses.base import Transport
from .harnesses.base import TransportResult
from .store import BridgeStore
from .types import BridgeRun
from .types import BridgeSession
from .types import EventType
from .types import ParseStatus
from .types import Participant
from .types import RunStatus
from .types import SessionRegistry
from .types import TurnRecord
from .types import utc_now_iso

TransportFactory = Callable[..., Transport]


class AgentBridgeBroker:
    def __init__(
        self,
        repo_root: Path,
        *,
        store: BridgeStore | None = None,
        transport_factory: TransportFactory = create_transport,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.store = store or BridgeStore(self.repo_root)
        self.transport_factory = transport_factory

    def start_run(
        self,
        *,
        task: str,
        sessions: dict[str, BridgeSession],
        next_actor: str | None = None,
        run_id: str | None = None,
        footer_mode: str = "prompt_injected",
        worktree_path: str | None = None,
        worktree_agent_slug: str = "codex",
        repair_budget_per_turn: int = 1,
        worktree_cleanup_mode: str = "operator_triggered",
    ) -> BridgeRun:
        created_at = utc_now_iso()
        normalized_run_id = run_id or f"bridge_{uuid.uuid4().hex[:12]}"
        ordered_roles = list(sessions)
        participants = [
            Participant(role=role, harness=session.harness, model=session.model)
            for role, session in sessions.items()
        ]
        registry_sessions: dict[str, BridgeSession] = {}
        for role, session in sessions.items():
            registry_sessions[role] = BridgeSession(
                role=role,
                harness=session.harness,
                model=session.model,
                session_id=session.session_id,
                worktree_agent_slug=session.worktree_agent_slug,
                worktree_path=session.worktree_path,
                branch=session.branch,
                session_status=session.session_status,
                started_at=session.started_at,
                last_turn_index=session.last_turn_index,
                last_completed_at=session.last_completed_at,
                harness_options=dict(session.harness_options),
            )
        run = BridgeRun(
            run_id=normalized_run_id,
            task=task.strip(),
            created_at=created_at,
            updated_at=created_at,
            status="running",
            completed_at=None,
            last_turn_index=0,
            next_actor=next_actor or (ordered_roles[0] if ordered_roles else None),
            repair_budget_per_turn=repair_budget_per_turn,
            footer_mode=footer_mode,
            worktree_cleanup_mode=worktree_cleanup_mode,
            participants=participants,
            worktree_path=worktree_path or str(self.repo_root),
            worktree_agent_slug=worktree_agent_slug,
        )
        registry = SessionRegistry(
            run_id=normalized_run_id,
            updated_at=created_at,
            sessions=registry_sessions,
        )
        self.store.save_run(run)
        self.store.save_sessions(normalized_run_id, registry)
        start_event = self._event(
            run_id=normalized_run_id,
            turn_index=0,
            event_type="run_started",
            seq=0,
            role=run.next_actor or "system",
            harness="broker",
            session_id=None,
            payload={
                "task": run.task,
                "next_actor": run.next_actor,
                "participants": [participant.to_dict() for participant in participants],
            },
        )
        self._append_event(run, start_event)
        self.store.save_run(run)
        return run

    def load_run(self, run_id: str) -> BridgeRun:
        return self.store.load_run(run_id)

    def load_sessions(self, run_id: str) -> SessionRegistry:
        return self.store.load_sessions(run_id)

    def load_events(self, run_id: str) -> list[TurnRecord]:
        return self.store.load_events(run_id)

    def list_runs(self, *, status: RunStatus | None = None) -> list[BridgeRun]:
        runs: list[BridgeRun] = []
        for run_path in self.store.runs_root().glob("*/run.json"):
            run = self.store.load_run(run_path.parent.name)
            if status is None or run.status == status:
                runs.append(run)
        runs.sort(key=lambda item: item.updated_at, reverse=True)
        return runs

    def dispatch_turn(self, *, run_id: str, role: str, prompt: str) -> TurnRecord:
        run = self.store.load_run(run_id)
        registry = self.store.load_sessions(run_id)
        if role not in registry.sessions:
            raise KeyError(f"Unknown role '{role}' for run {run_id}")

        session = registry.sessions[role]
        allowed_roles = set(registry.sessions)
        transport = self._transport_for_session(run, session)
        prompt_text = (
            f"{prompt.rstrip()}\n\n{build_footer_instruction(roles=sorted(allowed_roles))}"
        )
        turn_index = run.last_turn_index + 1
        seq = 0
        started_at = utc_now_iso()

        started_event = self._event(
            run_id=run_id,
            turn_index=turn_index,
            event_type="turn.started",
            seq=seq,
            role=role,
            harness=session.harness,
            session_id=session.session_id,
            payload={"prompt": prompt, "prompt_injected": prompt_text},
        )
        seq += 1
        self._append_event(run, started_event)

        try:
            if session.session_id is not None:
                result = transport.resume(
                    session.session_id,
                    prompt_text,
                    allowed_roles=allowed_roles,
                )
            else:
                result = transport.launch(
                    prompt_text,
                    allowed_roles=allowed_roles,
                )
        except TransportError as exc:
            registry.updated_at = utc_now_iso()
            session.session_status = "failed"
            failed_event = self._event(
                run_id=run_id,
                turn_index=turn_index,
                event_type="run_failed",
                seq=seq,
                role=role,
                harness=session.harness,
                session_id=session.session_id,
                payload={"error": str(exc)},
            )
            self._append_event(run, failed_event)
            run.status = "failed"
            run.updated_at = registry.updated_at
            self.store.save_sessions(run_id, registry)
            self.store.save_run(run)
            raise

        completed_at = utc_now_iso()
        self._update_session_after_result(
            session,
            session_id=result.session_id,
            started_at=started_at,
            completed_at=completed_at,
            turn_index=turn_index,
            status="active",
        )
        registry.updated_at = completed_at

        transcript_path = self.store.write_turn_transcript(
            run_id,
            turn_index=turn_index,
            role=role,
            harness=session.harness,
            model=session.model,
            session_id=result.session_id,
            started_at=started_at,
            completed_at=completed_at,
            exit_code=result.exit_code,
            prompt=prompt_text,
            raw_stdout=result.raw_stdout,
            raw_stderr=result.raw_stderr,
            parsed_turn=result.parsed_turn,
        )

        result_event = self._turn_result_event(
            run_id=run_id,
            turn_index=turn_index,
            seq=seq,
            role=role,
            harness=session.harness,
            session_id=result.session_id,
            result=result,
            transcript_path=self.store.relative_path(transcript_path),
            event_type="turn.result",
        )
        seq += 1
        self._append_event(run, result_event)

        final_result = result
        final_completed_at = completed_at
        repair_attempts: list[dict[str, object]] = []

        if result.parsed_turn.parse_status != "ok":
            parse_event = self._event(
                run_id=run_id,
                turn_index=turn_index,
                event_type=self._footer_event_type(result.parsed_turn.parse_status),
                seq=seq,
                role=role,
                harness=session.harness,
                session_id=result.session_id,
                parse_status=result.parsed_turn.parse_status,
                payload={
                    "errors": list(result.parsed_turn.parse_errors),
                    "transcript_path": self.store.relative_path(transcript_path),
                },
            )
            seq += 1
            self._append_event(run, parse_event)

            repair_prompt = build_repair_prompt(
                parse_errors=result.parsed_turn.parse_errors,
                original_message=result.message_text,
                allowed_roles=allowed_roles,
            )
            repair_requested = self._event(
                run_id=run_id,
                turn_index=turn_index,
                event_type="turn.repair_requested",
                seq=seq,
                role=role,
                harness=session.harness,
                session_id=result.session_id,
                parse_status=result.parsed_turn.parse_status,
                payload={"errors": list(result.parsed_turn.parse_errors)},
            )
            seq += 1
            self._append_event(run, repair_requested)

            repair_started_at = utc_now_iso()
            repair_result = transport.resume(
                result.session_id,
                repair_prompt,
                allowed_roles=allowed_roles,
            )
            repair_completed_at = utc_now_iso()
            self._update_session_after_result(
                session,
                session_id=repair_result.session_id,
                started_at=session.started_at or repair_started_at,
                completed_at=repair_completed_at,
                turn_index=turn_index,
                status="active",
            )
            registry.updated_at = repair_completed_at
            repair_attempts.append(
                {
                    "prompt": repair_prompt,
                    "raw_stdout": repair_result.raw_stdout,
                    "raw_stderr": repair_result.raw_stderr,
                    "parsed_turn": repair_result.parsed_turn,
                }
            )
            transcript_path = self.store.write_turn_transcript(
                run_id,
                turn_index=turn_index,
                role=role,
                harness=session.harness,
                model=session.model,
                session_id=repair_result.session_id,
                started_at=started_at,
                completed_at=repair_completed_at,
                exit_code=repair_result.exit_code,
                prompt=prompt_text,
                raw_stdout=result.raw_stdout,
                raw_stderr=result.raw_stderr,
                parsed_turn=result.parsed_turn,
                repair_attempts=repair_attempts,
            )
            repair_completed = self._turn_result_event(
                run_id=run_id,
                turn_index=turn_index,
                seq=seq,
                role=role,
                harness=session.harness,
                session_id=repair_result.session_id,
                result=repair_result,
                transcript_path=self.store.relative_path(transcript_path),
                event_type="turn.completed",
            )
            seq += 1
            self._append_event(run, repair_completed)
            final_result = repair_result
            final_completed_at = repair_completed_at

            if repair_result.parsed_turn.parse_status != "ok":
                exhausted_event = self._event(
                    run_id=run_id,
                    turn_index=turn_index,
                    event_type=self._footer_event_type(repair_result.parsed_turn.parse_status),
                    seq=seq,
                    role=role,
                    harness=session.harness,
                    session_id=repair_result.session_id,
                    parse_status=repair_result.parsed_turn.parse_status,
                    payload={
                        "errors": list(repair_result.parsed_turn.parse_errors),
                        "repair_exhausted": True,
                        "transcript_path": self.store.relative_path(transcript_path),
                    },
                )
                self._append_event(run, exhausted_event)
                run.status = "awaiting_human"
                run.next_actor = role
                run.last_turn_index = turn_index
                run.updated_at = repair_completed_at
                self.store.save_sessions(run_id, registry)
                self.store.save_run(run)
                return exhausted_event

        footer = final_result.parsed_turn.footer
        if footer is None:
            raise RuntimeError("footer unexpectedly missing after repair handling")

        footer_ok = self._event(
            run_id=run_id,
            turn_index=turn_index,
            event_type="footer_ok",
            seq=seq,
            role=role,
            harness=session.harness,
            session_id=final_result.session_id,
            parse_status="ok",
            payload={
                "footer": footer.to_dict(),
                "transcript_path": self.store.relative_path(transcript_path),
            },
        )
        self._append_event(run, footer_ok)

        run.next_actor = footer.next_actor
        run.last_turn_index = turn_index
        run.updated_at = final_completed_at
        run.completed_at = final_completed_at if footer.done else None
        if footer.done:
            run.status = "completed"
            session.session_status = "completed"
        elif footer.needs_human:
            run.status = "awaiting_human"
        else:
            run.status = "running"
        self.store.save_sessions(run_id, registry)
        if run.status == "completed":
            completed_event = self._event(
                run_id=run_id,
                turn_index=turn_index,
                event_type="run_completed",
                seq=seq + 1,
                role=role,
                harness=session.harness,
                session_id=final_result.session_id,
                payload={"next_actor": run.next_actor},
            )
            self._append_event(run, completed_event)
            self.store.save_run(run)
            return completed_event
        self.store.save_run(run)
        return footer_ok

    def _transport_for_session(self, run: BridgeRun, session: BridgeSession) -> Transport:
        cwd = Path(session.worktree_path) if session.worktree_path else Path(run.worktree_path)
        return self.transport_factory(
            session.harness,
            cwd=cwd,
            model=session.model or None,
            harness_options=session.harness_options,
        )

    def _update_session_after_result(
        self,
        session: BridgeSession,
        *,
        session_id: str,
        started_at: str,
        completed_at: str,
        turn_index: int,
        status: str,
    ) -> None:
        session.session_id = session_id
        if session.started_at is None:
            session.started_at = started_at
        session.last_turn_index = turn_index
        session.last_completed_at = completed_at
        session.session_status = status  # type: ignore[assignment]

    def _append_event(self, run: BridgeRun, record: TurnRecord) -> None:
        appended = self.store.append_event(run.run_id, record)
        if appended:
            run.last_event_id = record.event_id

    def _event(
        self,
        *,
        run_id: str,
        turn_index: int,
        event_type: EventType,
        seq: int,
        role: str,
        harness: str,
        session_id: str | None,
        payload: dict[str, object],
        parse_status: ParseStatus | None = None,
    ) -> TurnRecord:
        return TurnRecord(
            event_id=f"{run_id}:turn:{turn_index:03d}:{event_type}:{seq}",
            run_id=run_id,
            turn_index=turn_index,
            event_type=event_type,
            role=role,
            harness=harness,
            session_id=session_id,
            parse_status=parse_status,
            ts=utc_now_iso(),
            payload=dict(payload),
        )

    def _footer_event_type(self, parse_status: ParseStatus) -> EventType:
        if parse_status == "missing":
            return "footer_missing"
        if parse_status == "malformed":
            return "footer_malformed"
        return "footer_ok"

    def _turn_result_event(
        self,
        *,
        run_id: str,
        turn_index: int,
        seq: int,
        role: str,
        harness: str,
        session_id: str,
        result: TransportResult,
        transcript_path: str,
        event_type: EventType,
    ) -> TurnRecord:
        payload: dict[str, object] = {
            "exit_code": result.exit_code,
            "command": list(result.command),
            "transcript_path": transcript_path,
            "message_text": result.message_text,
            "usage": dict(result.usage),
        }
        if result.parsed_turn.footer is not None:
            payload["footer"] = result.parsed_turn.footer.to_dict()
        return self._event(
            run_id=run_id,
            turn_index=turn_index,
            event_type=event_type,
            seq=seq,
            role=role,
            harness=harness,
            session_id=session_id,
            parse_status=result.parsed_turn.parse_status,
            payload=payload,
        )
