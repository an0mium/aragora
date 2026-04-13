from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any
import uuid

_REGISTRY_SCHEMA_VERSION = 1
_PROMPT_MARKER_PREFIX = "=== ARAGORA_SESSION_MUX_PROMPT "


def resolve_repo_root(path_hint: Path | None = None) -> Path:
    candidate = (path_hint or Path.cwd()).resolve()
    proc = subprocess.run(
        ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return Path(proc.stdout.strip()).resolve()
    return candidate


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sanitize_session_name(name: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-" for char in name.strip()
    )
    normalized = cleaned.strip("-_")
    if not normalized:
        raise ValueError("Session name must contain at least one alphanumeric character")
    return normalized


def _registry_dir(repo_root: Path) -> Path:
    return repo_root / ".aragora" / "session_mux"


def _registry_path(repo_root: Path) -> Path:
    return _registry_dir(repo_root) / "registry.json"


def _logs_dir(repo_root: Path) -> Path:
    return _registry_dir(repo_root) / "logs"


def _session_log_path(repo_root: Path, session_name: str) -> Path:
    safe_name = _sanitize_session_name(session_name)
    return _logs_dir(repo_root) / f"{safe_name}.log"


def _tail_text(text: str, line_count: int) -> str:
    if line_count <= 0:
        return text
    lines = text.splitlines()
    return "\n".join(lines[-line_count:])


def prompt_marker(prompt_id: str, *, at: datetime | None = None) -> str:
    return f"{_PROMPT_MARKER_PREFIX}{prompt_id} {_isoformat_utc(at or _utc_now())}"


def append_prompt_marker(log_path: Path, *, prompt_id: str, at: datetime | None = None) -> str:
    marker = prompt_marker(prompt_id, at=at)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{marker}\n")
    return marker


def extract_output_after_marker(text: str, *, prompt_id: str | None) -> str:
    if not prompt_id:
        return text
    prefix = f"{_PROMPT_MARKER_PREFIX}{prompt_id} "
    marker_index = -1
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            marker_index = index
    if marker_index < 0:
        return text
    return "\n".join(lines[marker_index + 1 :])


@dataclass(slots=True)
class SessionRecord:
    name: str
    tmux_session: str
    tmux_window: str
    tmux_pane: str
    launcher_command: str
    started_at: str
    log_path: str
    last_prompt_at: str | None = None
    last_prompt_id: str | None = None
    worktree_path: str | None = None
    branch: str | None = None
    launcher_log_path: str | None = None
    meta_path: str | None = None

    @property
    def tmux_target(self) -> str:
        return f"{self.tmux_session}:{self.tmux_window}.{self.tmux_pane}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tmux_session": self.tmux_session,
            "tmux_window": self.tmux_window,
            "tmux_pane": self.tmux_pane,
            "launcher_command": self.launcher_command,
            "started_at": self.started_at,
            "last_prompt_at": self.last_prompt_at,
            "last_prompt_id": self.last_prompt_id,
            "worktree_path": self.worktree_path,
            "branch": self.branch,
            "log_path": self.log_path,
            "launcher_log_path": self.launcher_log_path,
            "meta_path": self.meta_path,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionRecord":
        return cls(
            name=str(payload.get("name", "") or ""),
            tmux_session=str(payload.get("tmux_session", "") or ""),
            tmux_window=str(payload.get("tmux_window", "") or "0"),
            tmux_pane=str(payload.get("tmux_pane", "") or "0"),
            launcher_command=str(payload.get("launcher_command", "") or ""),
            started_at=str(payload.get("started_at", "") or ""),
            log_path=str(payload.get("log_path", "") or ""),
            last_prompt_at=str(payload.get("last_prompt_at"))
            if payload.get("last_prompt_at")
            else None,
            last_prompt_id=str(payload.get("last_prompt_id"))
            if payload.get("last_prompt_id")
            else None,
            worktree_path=str(payload.get("worktree_path"))
            if payload.get("worktree_path")
            else None,
            branch=str(payload.get("branch")) if payload.get("branch") else None,
            launcher_log_path=(
                str(payload.get("launcher_log_path")) if payload.get("launcher_log_path") else None
            ),
            meta_path=str(payload.get("meta_path")) if payload.get("meta_path") else None,
        )


class SessionMuxRegistry:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.path = _registry_path(self.repo_root)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": _REGISTRY_SCHEMA_VERSION, "sessions": {}}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid session mux registry payload: {self.path}")
        payload.setdefault("schema_version", _REGISTRY_SCHEMA_VERSION)
        payload.setdefault("sessions", {})
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def list(self) -> list[SessionRecord]:
        payload = self._load()
        sessions = payload.get("sessions", {})
        if not isinstance(sessions, dict):
            return []
        return [
            SessionRecord.from_dict(dict(item))
            for _, item in sorted(sessions.items(), key=lambda pair: str(pair[0]))
            if isinstance(item, dict)
        ]

    def get(self, name: str) -> SessionRecord | None:
        payload = self._load()
        sessions = payload.get("sessions", {})
        if not isinstance(sessions, dict):
            return None
        raw = sessions.get(name)
        if not isinstance(raw, dict):
            return None
        return SessionRecord.from_dict(raw)

    def upsert(self, record: SessionRecord) -> None:
        payload = self._load()
        sessions = payload.setdefault("sessions", {})
        if not isinstance(sessions, dict):
            sessions = {}
            payload["sessions"] = sessions
        sessions[record.name] = record.to_dict()
        self._save(payload)


def build_tmux_new_session_cmd(*, tmux_session: str, cwd: Path, command: str) -> list[str]:
    return ["tmux", "new-session", "-d", "-s", tmux_session, "-c", str(cwd), command]


def build_tmux_list_panes_cmd(tmux_session: str) -> list[str]:
    return [
        "tmux",
        "list-panes",
        "-t",
        tmux_session,
        "-F",
        "#{window_index}\t#{pane_index}\t#{pane_current_path}",
    ]


def build_tmux_pipe_pane_cmd(*, target: str, log_path: Path) -> list[str]:
    sink = f"cat >> {shlex.quote(str(log_path))}"
    return ["tmux", "pipe-pane", "-o", "-t", target, sink]


def build_tmux_load_buffer_cmd() -> list[str]:
    return ["tmux", "load-buffer", "-"]


def build_tmux_paste_buffer_cmd(*, target: str) -> list[str]:
    return ["tmux", "paste-buffer", "-d", "-t", target]


def build_tmux_send_enter_cmd(*, target: str) -> list[str]:
    return ["tmux", "send-keys", "-t", target, "Enter"]


def build_tmux_capture_pane_cmd(*, target: str, line_count: int) -> list[str]:
    return ["tmux", "capture-pane", "-p", "-t", target, "-S", f"-{max(line_count, 1)}"]


def _run_tmux(cmd: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )


def _tmux_session_exists(tmux_session: str) -> bool:
    result = _run_tmux(["tmux", "has-session", "-t", tmux_session])
    return result.returncode == 0


def _primary_pane(tmux_session: str) -> dict[str, str]:
    result = _run_tmux(build_tmux_list_panes_cmd(tmux_session))
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            f"Unable to inspect tmux panes for session {tmux_session}: {result.stderr.strip()}"
        )
    line = result.stdout.strip().splitlines()[0]
    parts = line.split("\t")
    if len(parts) != 3:
        raise RuntimeError(f"Unexpected tmux pane format for session {tmux_session}: {line}")
    window, pane, current_path = parts
    return {
        "window": window,
        "pane": pane,
        "current_path": current_path,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _codex_meta_candidates(repo_root: Path) -> list[Path]:
    worktrees_root = repo_root / ".worktrees"
    if not worktrees_root.exists():
        return []
    return sorted(worktrees_root.rglob(".codex_session_meta.json"))


def _best_codex_meta(repo_root: Path, *, agent_name: str) -> dict[str, Any]:
    matched: list[dict[str, Any]] = []
    for path in _codex_meta_candidates(repo_root):
        payload = _read_json(path)
        if str(payload.get("agent", "") or "") != agent_name:
            continue
        payload["meta_path"] = str(path)
        matched.append(payload)
    if not matched:
        return {}
    matched.sort(key=lambda item: str(item.get("started_at", "")))
    return matched[-1]


def _git_branch(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def refresh_session_record(repo_root: Path, record: SessionRecord) -> SessionRecord:
    refreshed = SessionRecord.from_dict(record.to_dict())
    if _tmux_session_exists(refreshed.tmux_session):
        pane = _primary_pane(refreshed.tmux_session)
        refreshed.tmux_window = pane["window"]
        refreshed.tmux_pane = pane["pane"]
        current_path = pane["current_path"].strip()
        if current_path:
            refreshed.worktree_path = current_path
            if refreshed.branch is None:
                refreshed.branch = _git_branch(Path(current_path))
    meta = _best_codex_meta(repo_root, agent_name=refreshed.name)
    if meta:
        worktree_path = str(meta.get("worktree_path", "") or "").strip()
        refreshed.worktree_path = worktree_path or refreshed.worktree_path
        refreshed.branch = str(meta.get("branch", "") or "").strip() or refreshed.branch
        refreshed.launcher_log_path = (
            str(meta.get("log_path", "") or "").strip() or refreshed.launcher_log_path
        )
        refreshed.meta_path = str(meta.get("meta_path", "") or "").strip() or refreshed.meta_path
    return refreshed


def session_status(repo_root: Path, *, name: str) -> dict[str, Any]:
    registry = SessionMuxRegistry(repo_root)
    record = registry.get(name)
    if record is None:
        raise KeyError(f"Unknown session: {name}")
    record = refresh_session_record(repo_root, record)
    registry.upsert(record)
    running = _tmux_session_exists(record.tmux_session)
    payload = record.to_dict()
    payload["running"] = running
    payload["tmux_target"] = record.tmux_target
    return payload


def list_sessions(repo_root: Path) -> list[dict[str, Any]]:
    registry = SessionMuxRegistry(repo_root)
    statuses: list[dict[str, Any]] = []
    for record in registry.list():
        refreshed = refresh_session_record(repo_root, record)
        registry.upsert(refreshed)
        status = refreshed.to_dict()
        status["running"] = _tmux_session_exists(refreshed.tmux_session)
        status["tmux_target"] = refreshed.tmux_target
        statuses.append(status)
    return statuses


def launch_session(repo_root: Path, *, name: str, command: str) -> dict[str, Any]:
    if not command.strip():
        raise ValueError("Launch command must not be empty")
    registry = SessionMuxRegistry(repo_root)
    existing = registry.get(name)
    if existing is not None and _tmux_session_exists(existing.tmux_session):
        raise ValueError(f"Session already running: {name}")

    tmux_session = _sanitize_session_name(name)
    if _tmux_session_exists(tmux_session):
        raise ValueError(f"tmux session already exists: {tmux_session}")

    started_at = _isoformat_utc(_utc_now())
    log_path = _session_log_path(repo_root, name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    create_result = _run_tmux(
        build_tmux_new_session_cmd(tmux_session=tmux_session, cwd=repo_root, command=command)
    )
    if create_result.returncode != 0:
        raise RuntimeError(create_result.stderr.strip() or f"tmux launch failed for {name}")

    pane = _primary_pane(tmux_session)
    base_record = SessionRecord(
        name=name,
        tmux_session=tmux_session,
        tmux_window=pane["window"],
        tmux_pane=pane["pane"],
        launcher_command=command,
        started_at=started_at,
        worktree_path=pane["current_path"].strip() or None,
        branch=_git_branch(Path(pane["current_path"])) if pane["current_path"].strip() else None,
        log_path=str(log_path),
    )
    pipe_result = _run_tmux(
        build_tmux_pipe_pane_cmd(target=base_record.tmux_target, log_path=log_path)
    )
    if pipe_result.returncode != 0:
        raise RuntimeError(pipe_result.stderr.strip() or f"tmux pipe-pane failed for {name}")

    record = refresh_session_record(repo_root, base_record)
    registry.upsert(record)
    payload = record.to_dict()
    payload["running"] = True
    payload["tmux_target"] = record.tmux_target
    return payload


def send_prompt(
    repo_root: Path,
    *,
    name: str,
    text: str | None = None,
    file_path: Path | None = None,
) -> dict[str, Any]:
    if not text and file_path is None:
        raise ValueError("Provide either text or file_path")
    if text and file_path is not None:
        raise ValueError("Provide either text or file_path, not both")

    registry = SessionMuxRegistry(repo_root)
    record = registry.get(name)
    if record is None:
        raise KeyError(f"Unknown session: {name}")
    record = refresh_session_record(repo_root, record)
    if not _tmux_session_exists(record.tmux_session):
        raise RuntimeError(f"tmux session is not running: {name}")

    prompt_text = text if text is not None else file_path.read_text(encoding="utf-8")
    prompt_id = uuid.uuid4().hex[:8]
    prompt_at = _utc_now()
    append_prompt_marker(Path(record.log_path), prompt_id=prompt_id, at=prompt_at)

    load_result = _run_tmux(build_tmux_load_buffer_cmd(), input_text=prompt_text)
    if load_result.returncode != 0:
        raise RuntimeError(load_result.stderr.strip() or f"tmux load-buffer failed for {name}")
    paste_result = _run_tmux(build_tmux_paste_buffer_cmd(target=record.tmux_target))
    if paste_result.returncode != 0:
        raise RuntimeError(paste_result.stderr.strip() or f"tmux paste-buffer failed for {name}")
    enter_result = _run_tmux(build_tmux_send_enter_cmd(target=record.tmux_target))
    if enter_result.returncode != 0:
        raise RuntimeError(enter_result.stderr.strip() or f"tmux send-keys failed for {name}")

    record.last_prompt_id = prompt_id
    record.last_prompt_at = _isoformat_utc(prompt_at)
    registry.upsert(record)
    payload = record.to_dict()
    payload["running"] = True
    payload["tmux_target"] = record.tmux_target
    return payload


def capture_output(repo_root: Path, *, name: str, tail_lines: int = 200) -> str:
    registry = SessionMuxRegistry(repo_root)
    record = registry.get(name)
    if record is None:
        raise KeyError(f"Unknown session: {name}")
    log_path = Path(record.log_path)
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    captured = extract_output_after_marker(text, prompt_id=record.last_prompt_id)
    return _tail_text(captured, tail_lines).rstrip()


def tail_output(repo_root: Path, *, name: str, line_count: int = 200) -> str:
    registry = SessionMuxRegistry(repo_root)
    record = registry.get(name)
    if record is None:
        raise KeyError(f"Unknown session: {name}")
    record = refresh_session_record(repo_root, record)
    registry.upsert(record)

    if _tmux_session_exists(record.tmux_session):
        result = _run_tmux(
            build_tmux_capture_pane_cmd(target=record.tmux_target, line_count=line_count)
        )
        if result.returncode == 0:
            return result.stdout.rstrip()
    log_path = Path(record.log_path)
    if not log_path.exists():
        return ""
    return _tail_text(log_path.read_text(encoding="utf-8"), line_count).rstrip()
