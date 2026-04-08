from __future__ import annotations

import json
from pathlib import Path

from aragora.swarm.initiative_models import InitiativeRecord


class InitiativeStore:
    """Local-first JSON store for initiative planning artifacts."""

    def __init__(self, *, repo_root: Path | None = None, state_dir: Path | None = None) -> None:
        resolved_root = Path(repo_root or Path.cwd()).resolve()
        self._state_dir = (
            Path(state_dir).resolve()
            if state_dir is not None
            else resolved_root / ".aragora" / "initiatives"
        )
        self._state_dir.mkdir(parents=True, exist_ok=True)

    @property
    def state_dir(self) -> Path:
        return self._state_dir

    def path_for(self, initiative_id: str) -> Path:
        return self._state_dir / f"{initiative_id}.json"

    def save(self, initiative: InitiativeRecord) -> Path:
        initiative.touch()
        payload = json.dumps(initiative.to_dict(), indent=2, sort_keys=False) + "\n"
        destination = self.path_for(initiative.initiative_id)
        tmp_path = destination.with_suffix(".json.tmp")
        tmp_path.write_text(payload)
        tmp_path.replace(destination)
        return destination

    def get(self, initiative_id: str) -> InitiativeRecord | None:
        path = self.path_for(initiative_id)
        if not path.exists():
            return None
        return InitiativeRecord.from_dict(json.loads(path.read_text()))

    def list(self) -> list[InitiativeRecord]:
        items: list[InitiativeRecord] = []
        for path in sorted(self._state_dir.glob("*.json")):
            try:
                items.append(InitiativeRecord.from_dict(json.loads(path.read_text())))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        items.sort(
            key=lambda item: (item.updated_at, item.created_at, item.initiative_id), reverse=True
        )
        return items
