"""Conformance checks for the #8851 ARCH-015 queue disposition."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
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
EXPECTED_DYNAMIC_REFERENCERS = {
    "aragora/server/handlers/transcription.py",
    "aragora/server/initialization.py",
    "scripts/init_postgres_db.py",
}
JOB_QUEUE_STORE_MODULE = "aragora.storage.job_queue_store"


def _resolve_import_from(path: Path, node: ast.ImportFrom, *, root: Path) -> str | None:
    if node.level == 0:
        return node.module

    package_parts = path.relative_to(root).with_suffix("").parts[:-1]
    parent_levels = node.level - 1
    if parent_levels > len(package_parts):
        return None
    base_parts = package_parts[: len(package_parts) - parent_levels]
    if node.module:
        base_parts += tuple(node.module.split("."))
    return ".".join(base_parts)


def _imports_job_queue_store(path: Path, *, root: Path = ROOT) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = _resolve_import_from(path, node, root=root)
            if module == JOB_QUEUE_STORE_MODULE:
                return True
            if module and any(
                f"{module}.{alias.name}" == JOB_QUEUE_STORE_MODULE for alias in node.names
            ):
                return True
        elif isinstance(node, ast.Import):
            if any(alias.name == JOB_QUEUE_STORE_MODULE for alias in node.names):
                return True
    return False


def _references_job_queue_store_literal(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, ast.Constant) and node.value == JOB_QUEUE_STORE_MODULE
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize(
    ("relative_path", "source"),
    [
        ("consumer.py", "import aragora.storage.job_queue_store\n"),
        ("consumer.py", "from aragora.storage.job_queue_store import QueuedJob\n"),
        ("consumer.py", "from aragora.storage import job_queue_store\n"),
        ("consumer.py", "from aragora.storage import job_queue_store as store\n"),
        ("aragora/storage/consumer.py", "from . import job_queue_store\n"),
        ("aragora/server/consumer.py", "from ..storage import job_queue_store\n"),
        (
            "aragora/server/consumer.py",
            "from ..storage.job_queue_store import QueuedJob\n",
        ),
    ],
)
def test_job_queue_store_import_forms_are_detected(
    tmp_path: Path,
    relative_path: str,
    source: str,
) -> None:
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")

    assert _imports_job_queue_store(path, root=tmp_path)


def test_unrelated_queue_import_is_not_a_storage_consumer(tmp_path: Path) -> None:
    path = tmp_path / "consumer.py"
    path.write_text("from aragora.queue import QueueJob\n", encoding="utf-8")

    assert not _imports_job_queue_store(path, root=tmp_path)


def test_arch_015_is_the_binding_queue_authority() -> None:
    payload = yaml.safe_load(CHARTER.read_text(encoding="utf-8"))
    entry = next(row for row in payload["authorities"] if row["id"] == "ARCH-015")

    assert payload["meta"]["status"] == "DRAFT"
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

    source_paths = {
        path
        for source_root in (ROOT / "aragora", ROOT / "scripts")
        for path in source_root.rglob("*.py")
        if path != ROOT / "aragora/storage/job_queue_store.py"
    }
    importers = {
        path.relative_to(ROOT).as_posix() for path in source_paths if _imports_job_queue_store(path)
    }
    dynamic_referencers = {
        path.relative_to(ROOT).as_posix()
        for path in source_paths
        if _references_job_queue_store_literal(path)
    }

    assert not importers - EXPECTED_STORAGE_IMPORTERS
    assert not dynamic_referencers - EXPECTED_DYNAMIC_REFERENCERS
