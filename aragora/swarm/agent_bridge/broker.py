from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any
from typing import Callable

from .exceptions import TransportError
from .footer import build_footer_instruction
from .footer import build_repair_prompt
from .harnesses import create_transport
from .harnesses.base import TransportResult
from .store import BridgeStore
from .types import BridgeRun
from .types import BridgeSession
from .types import EventType
from .types import ParseStatus
from .types import SessionRegistry
from .types import TurnRecord
from .types import utc_now_iso

TransportFactory = Callable[..., Any]


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
        roles: dict[str, BridgeSession],
        active_role: str | None = None,
        run_id: str | None = None,
        footer_mode: str = "prompt_injected",
        worktree_path: str | None = None,
        worktree_agent_slug: str = "codex",
    ) -> BridgeRun:
        created_at = utc_now_iso()
        normalized_run_id = run_id or f"bridge_{uuid.uuid4().hex[:12]}"
        participants = list(roles)
        run = BridgeRun(
            run_id=normalized_run_id,
            created_at=created_at,
            updated_at=created_at,
            status="running",
            active_role=active_role or (participants[0] if participants else None),
            footer_mode=footer_mode,
            participants=participants,
            worktree_path=worktree_path or str(self.repo_root),
            worktree_agent_slug=worktree_agent_slug,
        )
        registry = SessionRegistry(roles=roles)
        self.store.save_run(run)
        self.store.save_sessions(normalized_run_id, registry)
        start_event = self._event(
            run_id=normalized_run_id,
            turn_index=0,
            event_type="run_started",
            seq=0,
            role=run.active_role or "system",
            payload={"participants": participants, "active_role": run.active_role},
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

    def list_runs(self) -> list[BridgeRun]:
        runs: list[BridgeRun] = []
        for run_path in self.store.runs_root().glob("*/run.json"):
            runs.append(self.store.load_run(run_path.parent.name))
        runs.sort(key=lambda item: item.updated_at, reverse=True)
        return runs

    def dispatch_turn(self, *, run_id: str, role: str, prompt: str) -> TurnRecord:
        run = self.store.load_run(run_id)
        registry = self.store.load_sessions(run_id)
        if role not in registry.roles:
            raise KeyError(f"Unknown role '{role}' for run {run_id}")
        session = registry.roles[role]
        allowed_roles = set(registry.roles)
        transport = self._transport_for_session(run, role, session)
        prompt_text = (
            f"{prompt.rstrip()}\n\n{build_footer_instruction(roles=sorted(allowed_roles))}"
        )
        turn_index = run.turn_count + 1
        seq = 0

        turn_started = self._event(
            run_id=run_id,
            turn_index=turn_index,
            event_type="turn_started",
            seq=seq,
            role=role,
            payload={"session_id": session.session_id},
        )
        seq += 1
        self._append_event(run, turn_started)

        try:
            result = (
                transport.resume(session.session_id, prompt_text, allowed_roles=allowed_roles)
                if session.session_id
                else transport.launch(prompt_text, allowed_roles=allowed_roles)
            )
        except TransportError as exc:
            run.status = "failed"
            run.updated_at = utc_now_iso()
            failed_event = self._event(
                run_id=run_id,
                turn_index=turn_index,
                event_type="run_failed",
                seq=seq,
                role=role,
                payload={"error": str(exc)},
            )
            self._append_event(run, failed_event)
            self.store.save_run(run)
            self.store.save_sessions(run_id, registry)
            raise

        session.session_id = result.session_id
        if session.last_turn_at is None:
            session.created_at = utc_now_iso()
        session.last_turn_at = utc_now_iso()

        turn_completed = self._turn_completed_event(
            run_id=run_id,
            turn_index=turn_index,
            seq=seq,
            role=role,
            result=result,
        )
        seq += 1
        self._append_event(run, turn_completed)

        final_result = result
        final_parse_status = result.parsed_turn.parse_status
        if final_parse_status != "ok":
            footer_event_type = self._footer_event_type(final_parse_status)
            footer_event = self._event(
                run_id=run_id,
                turn_index=turn_index,
                event_type=footer_event_type,
                seq=seq,
                role=role,
                parse_status=final_parse_status,
                payload={"errors": list(result.parsed_turn.parse_errors)},
            )
            seq += 1
            self._append_event(run, footer_event)

            repair_prompt = build_repair_prompt(
                parse_errors=result.parsed_turn.parse_errors,
                original_message=result.message_text,
                allowed_roles=allowed_roles,
            )
            repair_requested = self._event(
                run_id=run_id,
                turn_index=turn_index,
                event_type="footer_repair_requested",
                seq=seq,
                role=role,
                parse_status=final_parse_status,
                payload={"session_id": session.session_id},
            )
            seq += 1
            self._append_event(run, repair_requested)
            final_result = transport.resume(
                result.session_id,
                repair_prompt,
                allowed_roles=allowed_roles,
            )
            repair_completed = self._turn_completed_event(
                run_id=run_id,
                turn_index=turn_index,
                seq=seq,
                role=role,
                result=final_result,
            )
            seq += 1
            self._append_event(run, repair_completed)
            session.session_id = final_result.session_id
            session.last_turn_at = utc_now_iso()

            if final_result.parsed_turn.parse_status == "ok":
                footer_ok = self._event(
                    run_id=run_id,
                    turn_index=turn_index,
                    event_type="footer_ok",
                    seq=seq,
                    role=role,
                    parse_status="ok",
                    payload={"footer": final_result.parsed_turn.footer.to_dict()},
                )
                self._append_event(run, footer_ok)
            else:
                exhausted_event = self._event(
                    run_id=run_id,
                    turn_index=turn_index,
                    event_type=self._footer_event_type(final_result.parsed_turn.parse_status),
                    seq=seq,
                    role=role,
                    parse_status=final_result.parsed_turn.parse_status,
                    payload={
                        "errors": list(final_result.parsed_turn.parse_errors),
                        "repair_exhausted": True,
                    },
                )
                self._append_event(run, exhausted_event)
                run.status = "awaiting_human"
                run.active_role = role
                run.turn_count = turn_index
                run.updated_at = utc_now_iso()
                self.store.save_sessions(run_id, registry)
                self.store.save_run(run)
                return exhausted_event
        else:
            footer_ok = self._event(
                run_id=run_id,
                turn_index=turn_index,
                event_type="footer_ok",
                seq=seq,
                role=role,
                parse_status="ok",
                payload={"footer": final_result.parsed_turn.footer.to_dict()},
            )
            self._append_event(run, footer_ok)

        footer = final_result.parsed_turn.footer
        if footer is None:
            raise RuntimeError("footer unexpectedly missing after repair handling")
        run.active_role = footer.next_actor
        run.turn_count = turn_index
        run.updated_at = utc_now_iso()
        if footer.done:
            run.status = "completed"
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
                payload={"active_role": run.active_role},
            )
            self._append_event(run, completed_event)
            self.store.save_run(run)
            return completed_event
        self.store.save_run(run)
        return self._event(
            run_id=run_id,
            turn_index=turn_index,
            event_type="footer_ok",
            seq=seq,
            role=role,
            parse_status="ok",
            payload={"footer": footer.to_dict()},
        )

    def _transport_for_session(
        self,
        run: BridgeRun,
        role: str,
        session: BridgeSession,
    ) -> Any:
        model = session.harness_options.get("model")
        model_value = str(model) if isinstance(model, str) else None
        worktree_path = session.harness_options.get("worktree_path")
        cwd = (
            Path(str(worktree_path)) if isinstance(worktree_path, str) else Path(run.worktree_path)
        )
        return self.transport_factory(
            session.harness,
            cwd=cwd,
            model=model_value,
            harness_options=session.harness_options,
        )

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
        payload: dict[str, Any],
        parse_status: ParseStatus | None = None,
    ) -> TurnRecord:
        return TurnRecord(
            event_id=f"{run_id}:turn:{turn_index:03d}:{event_type}:{seq}",
            turn_index=turn_index,
            type=event_type,
            role=role,
            parse_status=parse_status,
            at=utc_now_iso(),
            payload=payload,
        )

    def _footer_event_type(self, parse_status: ParseStatus) -> EventType:
        if parse_status == "missing":
            return "footer_missing"
        if parse_status == "malformed":
            return "footer_malformed"
        return "footer_ok"

    def _turn_completed_event(
        self,
        *,
        run_id: str,
        turn_index: int,
        seq: int,
        role: str,
        result: TransportResult,
    ) -> TurnRecord:
        return self._event(
            run_id=run_id,
            turn_index=turn_index,
            event_type="turn_completed",
            seq=seq,
            role=role,
            parse_status=result.parsed_turn.parse_status,
            payload={
                "session_id": result.session_id,
                "command": list(result.command),
                "exit_code": result.exit_code,
                "message_text": result.message_text,
                "stderr": result.raw_stderr,
                "usage": dict(result.usage),
            },
        )
