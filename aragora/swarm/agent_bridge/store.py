from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import BridgeRun
from .types import SessionRegistry
from .types import TurnRecord


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


class BridgeStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def runs_root(self) -> Path:
        path = self.root / ".aragora" / "agent_bridge" / "runs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_dir(self, run_id: str) -> Path:
        path = self.runs_root() / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "run.json"

    def sessions_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "sessions.json"

    def events_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "events.jsonl"

    def save_run(self, run: BridgeRun) -> None:
        self._write_atomic(self.run_path(run.run_id), _json_text(run.to_dict()))

    def load_run(self, run_id: str) -> BridgeRun:
        payload = json.loads(self.run_path(run_id).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("run payload must be a mapping")
        return BridgeRun.from_dict(payload)

    def save_sessions(self, run_id: str, registry: SessionRegistry) -> None:
        self._write_atomic(self.sessions_path(run_id), _json_text(registry.to_dict()))

    def load_sessions(self, run_id: str) -> SessionRegistry:
        payload = json.loads(self.sessions_path(run_id).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("sessions payload must be a mapping")
        return SessionRegistry.from_dict(payload)

    def append_event(self, run_id: str, record: TurnRecord) -> bool:
        path = self.events_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict) and payload.get("event_id") == record.event_id:
                    return False
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        return True

    def load_events(self, run_id: str) -> list[TurnRecord]:
        path = self.events_path(run_id)
        if not path.exists():
            return []
        records: list[TurnRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(TurnRecord.from_dict(payload))
        return records

    def _write_atomic(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(content, encoding="utf-8")
        self._replace_file(temp_path, path)

    def _replace_file(self, source: Path, target: Path) -> None:
        source.replace(target)
