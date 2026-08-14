"""Acceptance tests for commit-addressed generic Nomic context packs."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from aragora.nomic.context_builder import NomicContextBuilder
from aragora.nomic.repository_profile import (
    EvaluationCriterion,
    RepositoryStateError,
    load_nomic_repository_profile,
)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def clean_repository(tmp_path: Path) -> Path:
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "pack@example.test")
    git(tmp_path, "config", "user.name", "Pack Test")
    git(tmp_path, "remote", "add", "origin", "git@github.com:example/context-pack.git")
    (tmp_path / ".gitignore").write_text(".nomic/\n", encoding="utf-8")
    (tmp_path / ".aragora.yaml").write_text(
        """nomic:
  repository:
    name: Context Pack Example
    id: example/context-pack
    remote_url: https://github.com/example/context-pack
  roadmap_paths:
    - docs/ROADMAP.md
  context_entry_files:
    - README.md
  evaluation_criteria:
    - id: impact
      description: Creates measurable improvement
""",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Example\nEntry context.\n", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ROADMAP.md").write_text("# Roadmap\nShip planning.\n", encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "initial")
    return tmp_path


@pytest.mark.asyncio
async def test_pack_is_exact_portable_and_tracked_only(clean_repository: Path) -> None:
    (clean_repository / "UNTRACKED.md").write_text("not evidence\n", encoding="utf-8")
    git(clean_repository, "status", "--porcelain")
    (clean_repository / "UNTRACKED.md").unlink()
    profile = load_nomic_repository_profile(clean_repository)
    builder = NomicContextBuilder(clean_repository, full_corpus=True)

    pack = await builder.build_context_pack("Improve the roadmap", profile=profile)

    assert pack.revision.commit_sha == git(clean_repository, "rev-parse", "HEAD")
    assert len(pack.revision.commit_sha) == 40
    assert pack.pack_path == (
        clean_repository / ".nomic" / "context" / "packs" / pack.revision.commit_sha / pack.pack_id
    )
    paths = {item.path for item in pack.evidence}
    assert "README.md" in paths
    assert "docs/ROADMAP.md" in paths
    assert "src/app.py" in paths
    assert "UNTRACKED.md" not in paths
    metadata = (pack.pack_path / "context-pack.json").read_text(encoding="utf-8")
    assert str(clean_repository) not in metadata
    assert (pack.pack_path / "corpus.txt").is_file()
    assert "Roadmap" in (pack.pack_path / "context.md").read_text(encoding="utf-8")
    builder.verify_context_pack(pack)


@pytest.mark.asyncio
async def test_pack_reuses_deterministic_artifacts(clean_repository: Path) -> None:
    profile = load_nomic_repository_profile(clean_repository)
    first_builder = NomicContextBuilder(clean_repository, full_corpus=False)
    first_builder._build_pack_rlm_summary = AsyncMock(return_value="Stable RLM summary")
    first = await first_builder.build_context_pack("Plan", profile=profile)
    second_builder = NomicContextBuilder(clean_repository, full_corpus=False)
    second_builder._build_pack_rlm_summary = AsyncMock(return_value="Stable RLM summary")
    second = await second_builder.build_context_pack("Plan", profile=profile)

    assert first.pack_id == second.pack_id
    assert first.pack_path == second.pack_path
    assert "Stable RLM summary" in (first.pack_path / "context.md").read_text()
    assert not (first.pack_path / "corpus.txt").exists()


@pytest.mark.asyncio
async def test_pack_invalidates_for_profile_and_commit(clean_repository: Path) -> None:
    profile = load_nomic_repository_profile(clean_repository)
    builder = NomicContextBuilder(clean_repository, full_corpus=False)
    first = await builder.build_context_pack(profile=profile)
    changed_profile = replace(
        profile,
        evaluation_criteria=(
            EvaluationCriterion(id="risk", description="Reduces repository risk"),
        ),
    )
    second = await NomicContextBuilder(clean_repository, full_corpus=False).build_context_pack(
        profile=changed_profile
    )
    assert first.pack_id != second.pack_id
    assert first.revision.commit_sha == second.revision.commit_sha

    (clean_repository / "src" / "app.py").write_text("def main():\n    return 2\n")
    git(clean_repository, "add", "src/app.py")
    git(clean_repository, "commit", "-m", "change app")
    third = await NomicContextBuilder(clean_repository, full_corpus=False).build_context_pack(
        profile=profile
    )
    assert third.revision.commit_sha != first.revision.commit_sha
    assert third.pack_path.parent.name == third.revision.commit_sha


@pytest.mark.asyncio
async def test_dirty_repository_fails_before_artifact_creation(clean_repository: Path) -> None:
    (clean_repository / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    builder = NomicContextBuilder(clean_repository, full_corpus=False)

    with pytest.raises(RepositoryStateError, match="clean"):
        await builder.build_context_pack(profile=load_nomic_repository_profile(clean_repository))

    assert not (clean_repository / ".nomic").exists()


@pytest.mark.asyncio
async def test_unignored_runtime_directory_fails_before_artifacts(clean_repository: Path) -> None:
    git(clean_repository, "rm", ".gitignore")
    git(clean_repository, "commit", "-m", "remove runtime ignore")
    builder = NomicContextBuilder(clean_repository, full_corpus=False)

    with pytest.raises(RepositoryStateError, match="ignore .nomic"):
        await builder.build_context_pack(profile=load_nomic_repository_profile(clean_repository))

    assert not (clean_repository / ".nomic").exists()


@pytest.mark.asyncio
async def test_mid_build_revision_drift_is_not_published(clean_repository: Path) -> None:
    builder = NomicContextBuilder(clean_repository, full_corpus=False)
    builder._before_pack_publish = lambda: (clean_repository / "README.md").write_text("drift\n")

    with pytest.raises(RepositoryStateError, match="clean"):
        await builder.build_context_pack(profile=load_nomic_repository_profile(clean_repository))

    assert not list((clean_repository / ".nomic").rglob("context-pack.json"))


@pytest.mark.asyncio
async def test_artifact_tampering_is_detected(clean_repository: Path) -> None:
    builder = NomicContextBuilder(clean_repository, full_corpus=False)
    pack = await builder.build_context_pack(profile=load_nomic_repository_profile(clean_repository))
    (pack.pack_path / "context.md").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(RepositoryStateError, match="verification failed"):
        builder.verify_context_pack(pack)


@pytest.mark.asyncio
async def test_build_index_uses_git_tracked_files(clean_repository: Path) -> None:
    (clean_repository / "untracked.py").write_text("untracked = True\n", encoding="utf-8")
    index = await NomicContextBuilder(clean_repository, full_corpus=False).build_index()

    assert index.get_file("src/app.py") is not None
    assert index.get_file("untracked.py") is None


def test_manifest_has_complete_evidence_fields(clean_repository: Path) -> None:
    """The async artifact assertions cover values; pin the portable native schema."""
    from aragora.nomic.repository_profile import ContextEvidenceReference

    fields = set(ContextEvidenceReference.__dataclass_fields__)
    assert fields == {
        "evidence_id",
        "path",
        "blob_id",
        "sha256",
        "size_bytes",
        "line_count",
        "role",
        "uri",
        "http_permalink",
    }
    assert json.loads(json.dumps({"fields": sorted(fields)}))["fields"]
