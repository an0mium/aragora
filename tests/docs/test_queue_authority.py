"""Conformance checks for the #8851 ARCH-015 queue disposition."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CHARTER = ROOT / "docs/architecture/charters.yaml"
DECISION = ROOT / "docs/architecture/QUEUE_ADOPTION_DISPOSITION.md"

EXPECTED_STORAGE_IMPORTERS = {
    "aragora/nomic/testfixer/queue_worker.py",
    "aragora/queue/workers/transcription_worker.py",
    "aragora/server/handlers/admin/health/workers.py",
    "aragora/server/handlers/transcription.py",
    "aragora/server/workers/gauntlet_worker.py",
    "aragora/server/workers/routing_worker.py",
}


def _imports_job_queue_store(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "aragora.storage.job_queue_store":
                return True
        elif isinstance(node, ast.Import):
            if any(alias.name == "aragora.storage.job_queue_store" for alias in node.names):
                return True
    return False


def test_arch_015_is_the_binding_queue_authority() -> None:
    payload = yaml.safe_load(CHARTER.read_text(encoding="utf-8"))
    entry = next(row for row in payload["authorities"] if row["id"] == "ARCH-015")

    assert payload["meta"]["status"] == "DRAFT"
    assert payload["meta"]["version"] == "0.6"
    assert entry["authority"] == "aragora/queue"
    assert entry["disposition"] == "adopt"
    assert entry["binding_in_draft"] is True
    assert entry["decision_record"] == "docs/architecture/QUEUE_ADOPTION_DISPOSITION.md"
    assert DECISION.is_file()


def test_queue_entrypoints_and_backend_split_remain_explicit() -> None:
    for relative_path in (
        "scripts/queue_worker.py",
        "aragora/server/handlers/queue.py",
        "aragora/server/startup/workers.py",
    ):
        assert (ROOT / relative_path).is_file(), relative_path

    importers = {
        path.relative_to(ROOT).as_posix()
        for source_root in (ROOT / "aragora", ROOT / "scripts")
        for path in source_root.rglob("*.py")
        if path != ROOT / "aragora/storage/job_queue_store.py"
        if _imports_job_queue_store(path)
    }
    assert importers == EXPECTED_STORAGE_IMPORTERS
