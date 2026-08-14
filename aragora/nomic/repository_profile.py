"""Repository identity, revision, and evidence models for generic Nomic planning."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import quote, urlparse


class NomicProfileError(ValueError):
    """Raised when a repository planning profile is invalid."""


class RepositoryStateError(RuntimeError):
    """Raised when a repository cannot provide a stable clean revision."""


def _git(repo_root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RepositoryStateError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def normalize_remote_url(remote_url: str | None) -> str | None:
    """Normalize common Git remote forms without embedding credentials or ``.git``."""
    if not remote_url:
        return None
    value = remote_url.strip()
    scp_match = re.match(r"^(?:[^@]+@)?([^:]+):(.+)$", value)
    if scp_match and "://" not in value:
        host, path = scp_match.groups()
        value = f"https://{host}/{path}"
    elif value.startswith("ssh://"):
        parsed = urlparse(value)
        value = f"https://{parsed.hostname or ''}{parsed.path}"
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        host = (parsed.hostname or "").lower()
        path = parsed.path.rstrip("/")
        if path.endswith(".git"):
            path = path[:-4]
        return f"https://{host}{path}"
    return value.removesuffix(".git").rstrip("/")


def infer_repository_id(remote_url: str | None, fallback_name: str) -> str:
    """Infer a stable repository ID from a normalized remote URL."""
    normalized = normalize_remote_url(remote_url)
    if normalized:
        parsed = urlparse(normalized)
        path = parsed.path.strip("/")
        if path:
            return path if parsed.hostname == "github.com" else f"{parsed.hostname}/{path}"
    return fallback_name


def validate_repository_path(path: str) -> str:
    """Return a normalized explicit repository-relative POSIX path."""
    if not isinstance(path, str) or not path.strip():
        raise NomicProfileError("repository paths must be non-empty strings")
    if "\\" in path:
        raise NomicProfileError(f"repository path must use '/' separators: {path!r}")
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or path.startswith("/"):
        raise NomicProfileError(f"absolute repository path is not allowed: {path!r}")
    if (
        candidate == PurePosixPath(".")
        or candidate.as_posix() != path
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise NomicProfileError(f"repository path must be explicit and traversal-free: {path!r}")
    return candidate.as_posix()


@dataclass(frozen=True)
class EvaluationCriterion:
    """One ordered planning criterion returned as a normalized 0-1 score."""

    id: str
    description: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", self.id):
            raise NomicProfileError(f"invalid evaluation criterion id: {self.id!r}")
        if not self.description.strip():
            raise NomicProfileError(f"criterion {self.id!r} requires a description")


DEFAULT_EVALUATION_CRITERIA = (
    EvaluationCriterion(
        id="usefulness",
        description="Produces practical, measurable repository improvement",
    ),
)


@dataclass(frozen=True)
class RepositoryRevision:
    """Exact Git revision used to build a context pack."""

    commit_sha: str
    tree_sha: str
    branch: str | None
    remote_url: str | None

    @classmethod
    def resolve(cls, repo_root: Path) -> RepositoryRevision:
        root = repo_root.resolve()
        top_level = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
        if top_level != root:
            raise RepositoryStateError(f"repository root must be the Git top-level: {top_level}")
        commit_sha = _git(root, "rev-parse", "HEAD^{commit}")
        tree_sha = _git(root, "rev-parse", "HEAD^{tree}")
        branch = _git(root, "symbolic-ref", "--short", "-q", "HEAD", check=False) or None
        remote = _git(root, "remote", "get-url", "origin", check=False) or None
        return cls(commit_sha, tree_sha, branch, normalize_remote_url(remote))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assert_clean_revision(
    repo_root: Path, expected: RepositoryRevision | None = None
) -> RepositoryRevision:
    """Fail unless tracked, staged, and untracked state is clean at one exact HEAD."""
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RepositoryStateError("repository must be clean before planning")
    revision = RepositoryRevision.resolve(repo_root)
    if expected and (
        revision.commit_sha != expected.commit_sha or revision.tree_sha != expected.tree_sha
    ):
        raise RepositoryStateError(
            f"repository revision drifted from {expected.commit_sha} to {revision.commit_sha}"
        )
    return revision


@dataclass(frozen=True)
class NomicRepositoryProfile:
    """Typed repository-owned configuration for the generic planning path."""

    repository_name: str
    repository_id: str
    remote_url: str | None = None
    roadmap_paths: tuple[str, ...] = ()
    context_entry_files: tuple[str, ...] = ()
    evaluation_criteria: tuple[EvaluationCriterion, ...] = DEFAULT_EVALUATION_CRITERIA
    source_config_sha256: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.repository_name.strip() or not self.repository_id.strip():
            raise NomicProfileError("repository name and id are required")
        object.__setattr__(self, "remote_url", normalize_remote_url(self.remote_url))
        object.__setattr__(
            self, "roadmap_paths", tuple(validate_repository_path(p) for p in self.roadmap_paths)
        )
        object.__setattr__(
            self,
            "context_entry_files",
            tuple(validate_repository_path(p) for p in self.context_entry_files),
        )
        ids = [criterion.id for criterion in self.evaluation_criteria]
        if len(ids) != len(set(ids)):
            raise NomicProfileError("evaluation criterion IDs must be unique")
        if not ids:
            raise NomicProfileError("at least one evaluation criterion is required")

    @property
    def profile_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository": {
                "name": self.repository_name,
                "id": self.repository_id,
                "remote_url": self.remote_url,
            },
            "roadmap_paths": list(self.roadmap_paths),
            "context_entry_files": list(self.context_entry_files),
            "evaluation_criteria": [asdict(item) for item in self.evaluation_criteria],
            "source_config_sha256": self.source_config_sha256,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        repo_root: Path,
        source_config_sha256: str | None = None,
    ) -> NomicRepositoryProfile:
        repository = value.get("repository") or {}
        if not isinstance(repository, Mapping):
            raise NomicProfileError("nomic.repository must be a mapping")
        remote = repository.get("remote_url")
        if remote is None:
            remote = RepositoryRevision.resolve(repo_root).remote_url
        name = str(repository.get("name") or repo_root.resolve().name)
        repository_id = str(repository.get("id") or infer_repository_id(remote, name))
        raw_criteria = value.get("evaluation_criteria")
        criteria: tuple[EvaluationCriterion, ...]
        if raw_criteria is None:
            criteria = DEFAULT_EVALUATION_CRITERIA
        elif isinstance(raw_criteria, list) and all(
            isinstance(item, Mapping) for item in raw_criteria
        ):
            criteria = tuple(EvaluationCriterion(**item) for item in raw_criteria)
        else:
            raise NomicProfileError("nomic.evaluation_criteria must be a list of mappings")
        return cls(
            repository_name=name,
            repository_id=repository_id,
            remote_url=str(remote) if remote else None,
            roadmap_paths=tuple(value.get("roadmap_paths") or ()),
            context_entry_files=tuple(value.get("context_entry_files") or ()),
            evaluation_criteria=criteria,
            source_config_sha256=source_config_sha256,
        )

    def validate_files(self, repo_root: Path, revision: RepositoryRevision) -> None:
        root = repo_root.resolve()
        for relative in (*self.roadmap_paths, *self.context_entry_files):
            candidate = root / relative
            try:
                resolved = candidate.resolve(strict=True)
            except OSError as exc:
                raise NomicProfileError(
                    f"configured repository file is missing: {relative}"
                ) from exc
            if not resolved.is_relative_to(root):
                raise NomicProfileError(f"configured symlink escapes repository: {relative}")
            tracked = subprocess.run(
                ["git", "-C", str(root), "cat-file", "-e", f"{revision.commit_sha}:{relative}"],
                capture_output=True,
                check=False,
            )
            if tracked.returncode != 0:
                raise NomicProfileError(
                    f"configured file is not tracked at {revision.commit_sha}: {relative}"
                )


def load_nomic_repository_profile(
    repo_root: Path,
    config_path: Path | None = None,
) -> NomicRepositoryProfile:
    """Load the typed ``nomic`` section, supporting config files outside the repository."""
    import yaml

    root = repo_root.resolve()
    path = config_path.resolve() if config_path else root / ".aragora.yaml"
    if not path.exists():
        if config_path is not None:
            raise NomicProfileError(f"configuration file does not exist: {path.name}")
        return NomicRepositoryProfile.from_mapping({}, repo_root=root)
    raw = path.read_bytes()
    try:
        loaded = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise NomicProfileError(f"invalid YAML in {path.name}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise NomicProfileError(".aragora.yaml must contain a mapping")
    nomic = loaded.get("nomic") or {}
    if not isinstance(nomic, Mapping):
        raise NomicProfileError("nomic must be a mapping")
    return NomicRepositoryProfile.from_mapping(
        nomic,
        repo_root=root,
        source_config_sha256=hashlib.sha256(raw).hexdigest(),
    )


@dataclass(frozen=True)
class ContextEvidenceReference:
    """Portable file-level evidence included in a commit-addressed context pack."""

    evidence_id: str
    path: str
    blob_id: str
    sha256: str
    size_bytes: int
    line_count: int
    role: str
    uri: str
    http_permalink: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextPack:
    """Published context-pack metadata plus its local artifact directory."""

    pack_id: str
    repository: NomicRepositoryProfile
    revision: RepositoryRevision
    profile_hash: str
    evidence: tuple[ContextEvidenceReference, ...]
    artifact_digests: Mapping[str, str]
    pack_path: Path = field(compare=False, repr=False)
    corpus_included: bool = False

    @property
    def reference(self) -> str:
        return f".nomic/context/packs/{self.revision.commit_sha}/{self.pack_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "nomic-context-pack/1.0",
            "pack_id": self.pack_id,
            "reference": self.reference,
            "repository": self.repository.to_dict(),
            "revision": self.revision.to_dict(),
            "profile_hash": self.profile_hash,
            "evidence": [item.to_dict() for item in self.evidence],
            "artifact_digests": dict(sorted(self.artifact_digests.items())),
            "corpus_included": self.corpus_included,
        }


def portable_evidence_uri(repository_id: str, commit_sha: str, path: str, lines: int) -> str:
    end = max(lines, 1)
    return f"repo://{quote(repository_id, safe='/')}@{commit_sha}/{quote(path)}#L1-L{end}"


def http_permalink(remote_url: str | None, commit_sha: str, path: str, lines: int) -> str | None:
    normalized = normalize_remote_url(remote_url)
    if not normalized:
        return None
    host = urlparse(normalized).hostname
    encoded = quote(path)
    end = max(lines, 1)
    if host in {"github.com", "gitlab.com"}:
        return f"{normalized}/blob/{commit_sha}/{encoded}#L1-L{end}"
    if host == "bitbucket.org":
        return f"{normalized}/src/{commit_sha}/{encoded}#lines-1:{end}"
    return None
