"""Tests for typed generic Nomic repository profiles."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from aragora.nomic.repository_profile import (
    ContextPack,
    EvaluationCriterion,
    NomicProfileError,
    NomicRepositoryProfile,
    RepositoryRevision,
    assert_clean_revision,
    infer_repository_id,
    load_nomic_repository_profile,
    normalize_remote_url,
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
def repository(tmp_path: Path) -> Path:
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "nomic@example.test")
    git(tmp_path, "config", "user.name", "Nomic Test")
    (tmp_path / "README.md").write_text("# Example\n", encoding="utf-8")
    plans = tmp_path / "docs" / "plans"
    plans.mkdir(parents=True)
    (plans / "ROADMAP.md").write_text("# Roadmap\n", encoding="utf-8")
    git(tmp_path, "add", "README.md", "docs/plans/ROADMAP.md")
    git(tmp_path, "commit", "-m", "initial")
    git(tmp_path, "remote", "add", "origin", "git@github.com:example/project.git")
    return tmp_path


def test_defaults_infer_repository_identity(repository: Path) -> None:
    profile = load_nomic_repository_profile(repository)

    assert profile.repository_name == repository.name
    assert profile.repository_id == "example/project"
    assert profile.remote_url == "https://github.com/example/project"
    assert [item.id for item in profile.evaluation_criteria] == ["usefulness"]


def test_yaml_round_trip_and_external_config_hash(repository: Path, tmp_path: Path) -> None:
    config = tmp_path / "external.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "nomic": {
                    "repository": {
                        "name": "Example",
                        "id": "example/project",
                        "remote_url": "https://github.com/example/project.git",
                    },
                    "roadmap_paths": ["docs/plans/ROADMAP.md"],
                    "context_entry_files": ["README.md"],
                    "evaluation_criteria": [
                        {"id": "impact", "description": "Creates measurable impact"}
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    profile = load_nomic_repository_profile(repository, config)
    round_tripped = NomicRepositoryProfile.from_mapping(
        profile.to_dict(),
        repo_root=repository,
        source_config_sha256=profile.source_config_sha256,
    )

    assert round_tripped == profile
    assert profile.source_config_sha256
    assert profile.profile_hash == round_tripped.profile_hash
    assert str(tmp_path) not in str(profile.to_dict())


def test_duplicate_criterion_ids_rejected(repository: Path) -> None:
    with pytest.raises(NomicProfileError, match="unique"):
        NomicRepositoryProfile.from_mapping(
            {
                "evaluation_criteria": [
                    {"id": "impact", "description": "First"},
                    {"id": "impact", "description": "Second"},
                ]
            },
            repo_root=repository,
        )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"repository": {"name": 7}}, "repository.name"),
        ({"roadmap_paths": "README.md"}, "list of strings"),
        ({"context_entry_files": [1]}, "list of strings"),
        (
            {"evaluation_criteria": [{"id": "impact"}]},
            "requires string id and description",
        ),
        (
            {"evaluation_criteria": [{"id": "impact", "description": "Impact", "weight": 1}]},
            "unknown evaluation criterion",
        ),
    ],
)
def test_profile_fields_are_strictly_typed(
    repository: Path,
    value: dict,
    message: str,
) -> None:
    with pytest.raises(NomicProfileError, match=message):
        NomicRepositoryProfile.from_mapping(value, repo_root=repository)


@pytest.mark.parametrize("path", ["/tmp/ROADMAP.md", "../ROADMAP.md", "docs/../README.md"])
def test_invalid_repository_paths_rejected(repository: Path, path: str) -> None:
    with pytest.raises(NomicProfileError, match="path"):
        NomicRepositoryProfile.from_mapping(
            {"roadmap_paths": [path]},
            repo_root=repository,
        )


def test_missing_and_untracked_files_rejected(repository: Path) -> None:
    revision = RepositoryRevision.resolve(repository)
    missing = NomicRepositoryProfile.from_mapping(
        {"context_entry_files": ["MISSING.md"]}, repo_root=repository
    )
    with pytest.raises(NomicProfileError, match="missing"):
        missing.validate_files(repository, revision)

    (repository / "NOTES.md").write_text("untracked\n", encoding="utf-8")
    untracked = NomicRepositoryProfile.from_mapping(
        {"context_entry_files": ["NOTES.md"]}, repo_root=repository
    )
    with pytest.raises(NomicProfileError, match="not tracked"):
        untracked.validate_files(repository, revision)


def test_tracked_directory_is_not_a_configured_file(repository: Path) -> None:
    profile = NomicRepositoryProfile.from_mapping(
        {"context_entry_files": ["docs"]}, repo_root=repository
    )

    with pytest.raises(NomicProfileError, match="not a file"):
        profile.validate_files(repository, RepositoryRevision.resolve(repository))


def test_symlink_escaping_repository_rejected(repository: Path) -> None:
    external = repository.parent / f"{repository.name}-external.md"
    external.write_text("outside\n", encoding="utf-8")
    link = repository / "outside.md"
    link.symlink_to(external)
    git(repository, "add", "outside.md")
    git(repository, "commit", "-m", "track escaping link")
    profile = NomicRepositoryProfile.from_mapping(
        {"context_entry_files": ["outside.md"]}, repo_root=repository
    )

    with pytest.raises(NomicProfileError, match="escapes"):
        profile.validate_files(repository, RepositoryRevision.resolve(repository))


def test_revision_and_cleanliness(repository: Path) -> None:
    revision = assert_clean_revision(repository)
    assert len(revision.commit_sha) == 40
    assert len(revision.tree_sha) == 40
    assert revision.remote_url == "https://github.com/example/project"

    (repository / "README.md").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean"):
        assert_clean_revision(repository)

    git(repository, "add", "README.md")
    with pytest.raises(RuntimeError, match="clean"):
        assert_clean_revision(repository)


@pytest.mark.parametrize(
    ("raw", "normalized", "repository_id"),
    [
        (
            "git@github.com:synaptent/aragora.git",
            "https://github.com/synaptent/aragora",
            "synaptent/aragora",
        ),
        (
            "ssh://git@gitlab.com/group/project.git",
            "https://gitlab.com/group/project",
            "gitlab.com/group/project",
        ),
        (
            "git://github.com/example/project.git",
            "https://github.com/example/project",
            "example/project",
        ),
        ("file:///opt/example/project.git", None, "fallback"),
        ("/opt/example/project.git", None, "fallback"),
        ("../project.git", None, "fallback"),
        ("C:/repos/project.git", None, "fallback"),
        (r"C:\repos\project.git", None, "fallback"),
        (r"\\server\share\project.git", None, "fallback"),
    ],
)
def test_remote_normalization(raw: str, normalized: str | None, repository_id: str) -> None:
    assert normalize_remote_url(raw) == normalized
    assert infer_repository_id(raw, "fallback") == repository_id


def test_criterion_validation() -> None:
    with pytest.raises(NomicProfileError):
        EvaluationCriterion(id="Not Valid", description="description")
    with pytest.raises(NomicProfileError):
        EvaluationCriterion(id="valid", description=" ")


def test_context_pack_serializes_complete_metadata_contract(tmp_path: Path) -> None:
    profile = NomicRepositoryProfile(
        repository_name="Example",
        repository_id="example/project",
        remote_url="https://github.com/example/project",
    )
    revision = RepositoryRevision(
        commit_sha="a" * 40,
        tree_sha="b" * 40,
        branch="main",
        remote_url=profile.remote_url,
    )
    pack = ContextPack(
        pack_id="pack-id",
        objective="Improve the roadmap",
        repository=profile,
        revision=revision,
        profile_hash=profile.profile_hash,
        evidence=(),
        artifact_digests={"context.md": "digest"},
        pack_path=tmp_path,
        corpus_included=True,
        corpus_truncated=True,
        context_byte_budget=4096,
        include_tests=False,
        rlm_summary="Repository summary",
    )

    payload = pack.to_dict()

    assert payload["objective"] == "Improve the roadmap"
    assert payload["corpus_included"] is True
    assert payload["corpus_truncated"] is True
    assert payload["context_byte_budget"] == 4096
    assert payload["include_tests"] is False
    assert payload["rlm_summary"] == "Repository summary"
